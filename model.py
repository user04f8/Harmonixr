import math
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from einops import rearrange
from data import MIDIDataset

NUM_WORKERS_PER_DATALOADER = 0


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, eta_min=0, last_epoch=-1):
    """
    Create a learning rate scheduler with a warmup phase followed by a cosine decay.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        num_warmup_steps (int): Number of warmup steps.
        num_training_steps (int): Total number of training steps.
        eta_min (float): Minimum learning rate after decay.
        last_epoch (int): The index of the last epoch.

    Returns:
        LambdaLR: Learning rate scheduler.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(eta_min, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


class WraparoundConv3D(nn.Module):
    """
    A 3D convolutional layer that supports wraparound padding on the pitch dimension.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, use_wraparound=False):
        super(WraparoundConv3D, self).__init__()
        self.use_wraparound = use_wraparound
        self.conv = nn.Conv3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding
        )
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        if self.use_wraparound:
            # Apply wraparound padding on pitch dimension (dim=2)
            pad_front = x[:, :, -1:, :, :].clone()
            pad_back = x[:, :, :1, :, :].clone()
            x = torch.cat([pad_front, x, pad_back], dim=2)
        return self.conv(x)


class ResidualConv3D(nn.Module):
    """
    A conv layer with residual "skip" connections
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, use_wraparound=False):
        super(ResidualConv3D, self).__init__()
        self.use_wraparound = use_wraparound

        # Main Path
        self.conv1 = WraparoundConv3D(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding,
            use_wraparound=use_wraparound
        )
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size, stride=1, padding=padding)
        self.bn2 = nn.BatchNorm3d(out_channels)

        # Residual Path
        if in_channels != out_channels or stride != 1 or self.use_wraparound:
            self.downsample = nn.Sequential(
                WraparoundConv3D(
                    in_channels, out_channels, kernel_size=1,
                    stride=stride, padding=0, use_wraparound=self.use_wraparound
                ),
                nn.BatchNorm3d(out_channels)
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ContrastiveLoss(nn.Module):
    """
    Computes loss based on the distance between embeddings and the label indicating similarity.
    """
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, distance, label):
        loss = (label) * 0.5 * distance.pow(2) + \
               (1 - label) * 0.5 * torch.clamp(self.margin - distance, min=0.0).pow(2)
        return loss.mean()


class SiaViT(pl.LightningModule):
    """
    Siamese Network with Vision Transformer for MIDI data.

    Args:
        embedding_dim (int): Dimension of the output embeddings.
        data_dir (str): Directory containing the data.
        t (int): Number of time steps.
        o (int): Number of octaves.
        batch_size (int): Batch size for training.
        lr (float): Learning rate.
        num_conv_layers (int): Number of convolutional layers.
        conv_channels (list of int): Number of channels for each conv layer.
        conv_kernel_sizes (list of tuple): Kernel sizes for each conv layer.
        conv_strides (list of tuple): Strides for each conv layer.
        conv_paddings (list of tuple): Paddings for each conv layer.
        dropout_rates (list of float): Dropout rates for each conv layer.
        maxpool_kernel_sizes (list of tuple): Max pooling kernel sizes for each conv layer.
        transformer_d_model (int): Dimension of the transformer model.
        transformer_nhead (int): Number of heads in the transformer.
        transformer_num_layers (int): Number of transformer layers.
        fc_hidden_dims (list of int): Hidden dimensions for the fully connected layers.
        weight_decay (float): Weight decay for the optimizer.
        use_AdamW (bool): Whether to use AdamW optimizer.
        cl_margin (float): Margin for the contrastive loss.
        warmup_proportion (float): Proportion of warmup steps for the scheduler.
        wraparound_layers (list of bool): Whether to use wraparound padding for each conv layer.
    """
    def __init__(
        self,
        embedding_dim=500,
        data_dir='./data',
        t=20 * 60,
        o=6,
        batch_size=32,
        lr=1e-3,
        threshold='deprecated',  # now dynamic
        num_conv_layers=3,
        conv_channels=None,
        conv_kernel_sizes=None,
        conv_strides=None,
        conv_paddings=None,
        dropout_rates=None,
        maxpool_kernel_sizes=None,
        transformer_d_model=512,
        transformer_nhead=8,
        transformer_num_layers=6,
        fc_hidden_dims=None,
        weight_decay=1e-5,
        use_AdamW=True,
        cl_margin=1.0,
        warmup_proportion=0.1,
        wraparound_layers=None
    ):
        super(SiaViT, self).__init__()
        self.save_hyperparameters()
        
        # Ensure that list hyperparameters are provided
        assert conv_channels is not None, "conv_channels must be provided"
        assert conv_kernel_sizes is not None, "conv_kernel_sizes must be provided"
        assert conv_strides is not None, "conv_strides must be provided"
        assert conv_paddings is not None, "conv_paddings must be provided"
        assert dropout_rates is not None, "dropout_rates must be provided"
        assert maxpool_kernel_sizes is not None, "maxpool_kernel_sizes must be provided"
        assert fc_hidden_dims is not None, "fc_hidden_dims must be provided"
        assert wraparound_layers is not None, "wraparound_layers must be provided"

        # Build convolutional layers
        conv_layers = []
        in_channels = 1
        for i in range(num_conv_layers):
            out_channels = conv_channels[i]
            kernel_size = conv_kernel_sizes[i]
            stride = conv_strides[i]
            padding = conv_paddings[i]
            use_wraparound = wraparound_layers[i]
            
            conv_layer = ResidualConv3D(
                in_channels, out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                use_wraparound=use_wraparound
            )
            conv_layers.append(conv_layer)
            conv_layers.append(nn.MaxPool3d(kernel_size=maxpool_kernel_sizes[i]))
            conv_layers.append(nn.Dropout(dropout_rates[i]))
            in_channels = out_channels

        self.conv = nn.Sequential(*conv_layers)
        
        # Compute feature_dim based on the convolutional layers
        self.feature_dim = self._compute_feature_dim()

        # Initialize feature_projection
        self.feature_projection = nn.Linear(self.feature_dim, self.hparams.transformer_d_model)
        nn.init.xavier_uniform_(self.feature_projection.weight)
    
        # Transformer encoder with Layer Normalization
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_d_model,
            nhead=transformer_nhead,
            dim_feedforward=2048,
            dropout=0.1,
            activation='relu',
            layer_norm_eps=1e-5,
            batch_first=False,
            norm_first=False
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_num_layers,
            norm=nn.LayerNorm(transformer_d_model)
        )

        # Fully connected layers to get embeddings
        fc_layers = []
        fc_input_dim = transformer_d_model
        for dim in fc_hidden_dims:
            fc_layers.append(nn.Linear(fc_input_dim, dim))
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.Dropout(0.3))
            fc_input_dim = dim
        fc_layers.append(nn.Linear(fc_input_dim, embedding_dim))
        self.fc = nn.Sequential(*fc_layers)
        # Initialize fully connected layers
        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

        # Contrastive Loss
        self.criterion = ContrastiveLoss(margin=cl_margin)
        self.dynamic_threshold = 0.5

    def _compute_feature_dim(self):
        """
        Computes the feature dimension after the convolutional layers.
        """
        def compute_output_size(input_size, kernel_size, stride, padding):
            return (input_size + 2 * padding - kernel_size) // stride + 1

        # Initial sizes
        p_size = 12  # pitch dimension
        o_size = self.hparams.o  # octave dimension
        t_size = self.hparams.t  # time dimension

        for i in range(self.hparams.num_conv_layers):
            # Wraparound padding adds 2 to pitch size only if used
            if self.hparams.wraparound_layers[i]:
                p_size += 2

            # Conv layer parameters
            kernel_size = self.hparams.conv_kernel_sizes[i]
            stride = self.hparams.conv_strides[i]
            padding = self.hparams.conv_paddings[i]

            kernel_p, kernel_o, kernel_t = kernel_size
            stride_p, stride_o, stride_t = stride
            padding_p, padding_o, padding_t = padding

            # Compute output sizes after conv
            p_size = compute_output_size(p_size, kernel_p, stride_p, padding_p)
            o_size = compute_output_size(o_size, kernel_o, stride_o, padding_o)
            t_size = compute_output_size(t_size, kernel_t, stride_t, padding_t)

            # MaxPool parameters
            pool_size = self.hparams.maxpool_kernel_sizes[i]
            pool_p, pool_o, pool_t = pool_size

            # Compute output sizes after pooling
            p_size = p_size // pool_p
            o_size = o_size // pool_o
            t_size = t_size // pool_t

        # After convolutional layers, compute feature_dim
        c = self.hparams.conv_channels[-1]
        feature_dim = c * p_size * o_size
        self.seq_len = t_size  # Save sequence length for later use
        return feature_dim

    def forward_one(self, x):
        """
        Forward pass for one input in the siamese network.

        Args:
            x (Tensor): Input tensor of shape (batch_size, 12, o, t).

        Returns:
            Tensor: Normalized embedding of the input.
        """
        x = x.unsqueeze(1)  # (batch_size, 1, 12, o, t)
        x = self.conv(x)
        # x shape after conv: (batch_size, c, p, o, t)
        b, c, p, o, t = x.shape
        # Verify feature dimensions
        expected_feature_dim = c * p * o
        assert self.feature_dim == expected_feature_dim, \
            f"Feature dimensions do not match! Expected {self.feature_dim}, got {expected_feature_dim}"

        # Flatten for transformer: (batch_size, features, seq_len)
        x = x.view(b, c * p * o, t)
        x = x.permute(2, 0, 1)  # (seq_len, batch_size, features)
        x = self.feature_projection(x)  # Project features to transformer_d_model
        x = self.transformer(x)
        x = x.mean(dim=0)  # (batch_size, transformer_d_model)
        x = self.fc(x)
        x = nn.functional.normalize(x, p=2, dim=1)  # Euclidean normalization
        return x

    def compute_distance(self, emb1, emb2):
        """
        Computes the Euclidean distance between two embeddings.

        Args:
            emb1 (Tensor): First embedding.
            emb2 (Tensor): Second embedding.

        Returns:
            Tensor: Distance between embeddings.
        """
        return torch.norm(emb1 - emb2, p=2, dim=1)

    def forward(self, x1, x2):
        """
        Forward pass for the siamese network.

        Args:
            x1 (Tensor): First input tensor.
            x2 (Tensor): Second input tensor.

        Returns:
            Tensor: Distance between the embeddings of the inputs.
        """
        emb1 = self.forward_one(x1)
        emb2 = self.forward_one(x2)
        return self.compute_distance(emb1, emb2)

    def train_dataloader(self):
        """
        Returns the training dataloader.
        """
        train_dataset = MIDIDataset(self.hparams.data_dir, self.hparams.t, split='train')
        return DataLoader(train_dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=NUM_WORKERS_PER_DATALOADER)

    def training_step(self, batch, batch_idx):
        """
        Training step.

        Args:
            batch: Batch data.
            batch_idx: Batch index.

        Returns:
            Tensor: Loss value.
        """
        (x1, x2), y = batch
        distance = self.forward(x1, x2)
        y = y.float()
        loss = self.criterion(distance, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def val_dataloader(self):
        """
        Returns a list of validation dataloaders for different validation sets.
        """
        val_dataset_similar = MIDIDataset(
            self.hparams.data_dir, self.hparams.t, split='val', pair_type='similar'
        )
        val_dataset_dissimilar = MIDIDataset(
            self.hparams.data_dir, self.hparams.t, split='val', pair_type='dissimilar'
        )
        threshold_optim_dataset_mixed = MIDIDataset(
            self.hparams.data_dir, self.hparams.t, split='val', pair_type='mixed'
        )
        val_dataset_mixed = MIDIDataset(
            self.hparams.data_dir, self.hparams.t, split='val', pair_type='mixed'
        )
        val_loader_similar = DataLoader(
            val_dataset_similar, batch_size=self.hparams.batch_size, num_workers=NUM_WORKERS_PER_DATALOADER
        )
        val_loader_dissimilar = DataLoader(
            val_dataset_dissimilar, batch_size=self.hparams.batch_size, num_workers=NUM_WORKERS_PER_DATALOADER
        )
        val_loader_mixed = DataLoader(
            val_dataset_mixed, batch_size=self.hparams.batch_size, num_workers=NUM_WORKERS_PER_DATALOADER
        )
        threshold_optim_dataloader_mixed = DataLoader(
            threshold_optim_dataset_mixed, batch_size=self.hparams.batch_size, num_workers=NUM_WORKERS_PER_DATALOADER
        )
        return [
            threshold_optim_dataloader_mixed, 
            val_loader_mixed, 
            val_loader_similar, 
            val_loader_dissimilar
        ]

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """
        Validation step.

        Args:
            batch: Batch data.
            batch_idx: Batch index.
            dataloader_idx: Index of the dataloader.

        Returns:
            dict: Dictionary containing loss and accuracy.
        """
        (x1, x2), y = batch
        distance = self.forward(x1, x2)
        y = y.float()
        loss = self.criterion(distance, y)
        
        if dataloader_idx == 0:  # threshold_optim_dataloader_mixed
            thresholds = torch.linspace(0, 1, steps=1001).to(self.device)
            accuracies = []
            for threshold in thresholds:
                preds = (distance < threshold).long()
                acc = (preds == y.long()).float().mean()
                accuracies.append(acc.item())
            accuracies = torch.tensor(accuracies).to(self.device)
            max_idx = torch.argmax(accuracies)
            optimal_threshold = thresholds[max_idx].item()
            acc = accuracies[max_idx].item()
        else:
            preds = (distance < self.dynamic_threshold).long()
            acc = (preds == y.long()).float().mean()

        if dataloader_idx == 0:
            self.log(
                'psuedoval_loss_mixed', loss, 
                on_step=False, on_epoch=True, prog_bar=False, 
                add_dataloader_idx=False
            )
            self.log(
                'psuedoval_acc_mixed', acc, 
                on_step=False, on_epoch=True, prog_bar=True, 
                add_dataloader_idx=False
            )
            self.log(
                'optimal_threshold', optimal_threshold, 
                on_step=False, on_epoch=True, prog_bar=True, 
                add_dataloader_idx=False
            )

            self.dynamic_threshold = optimal_threshold
        elif dataloader_idx == 1:
            self.log(
                'val_loss_mixed', loss, 
                on_step=False, on_epoch=True, prog_bar=True, 
                add_dataloader_idx=False
            )
            self.log(
                'val_acc_mixed', acc, 
                on_step=False, on_epoch=True, prog_bar=True, 
                add_dataloader_idx=False
            )
        elif dataloader_idx == 2:
            # Similar pairs
            self.log(
                'val_loss_similar', loss, 
                on_step=False, on_epoch=True, prog_bar=False, 
                add_dataloader_idx=False
            )
            self.log(
                'val_acc_similar', acc, 
                on_step=False, on_epoch=True, prog_bar=True, 
                add_dataloader_idx=False
            )
        elif dataloader_idx == 3:
            # Dissimilar pairs
            self.log(
                'val_loss_dissimilar', loss, 
                on_step=False, on_epoch=True, prog_bar=False, 
                add_dataloader_idx=False
            )
            self.log(
                'val_acc_dissimilar', acc, 
                on_step=False, on_epoch=True, prog_bar=True, 
                add_dataloader_idx=False
            )
        else:
            raise AssertionError("dataloader_idx out of bounds")

        return {'val_loss': loss, 'val_acc': acc}

    def configure_optimizers(self):
        """
        Configures the optimizer and learning rate scheduler.
        """
        # Choose optimizer
        if self.hparams.use_AdamW:
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.hparams.lr,
                weight_decay=self.hparams.weight_decay
            )
        else:
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.hparams.lr
            )

        # Total training steps
        total_steps = self.trainer.estimated_stepping_batches

        # Warmup steps (e.g., first 10% of total steps)
        warmup_steps = int(self.hparams.warmup_proportion * total_steps)

        # Define the scheduler
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
            eta_min=0
        )

        scheduler_config = {
            'scheduler': scheduler,
            'interval': 'step',  # Step-based scheduling
            'frequency': 1,
            'name': 'cosine_with_warmup'
        }

        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler_config
        }
