import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from einops import rearrange
from data import MIDIDataset

class WraparoundConv3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(WraparoundConv3D, self).__init__()
        self.conv = nn.Conv3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding
        )
        # Initialize weights
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        # x shape: (batch, channels, pitch, octave, time)
        # Wraparound padding on pitch dimension
        pitch_dim = 2
        # Wrap around pitches
        pad_front = x[:, :, -1:, :, :].clone()
        pad_back = x[:, :, :1, :, :].clone()
        x = torch.cat([pad_front, x, pad_back], dim=pitch_dim)
        return self.conv(x)

class SiaViT(pl.LightningModule):
    def __init__(
        self,
        embedding_dim=500,
        data_dir='./data',
        t=128,
        o=8,
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
        cl_margin=1.0
    ):
        super(SiaViT, self).__init__()
        self.save_hyperparameters()
        
        self.conv_layers_params = {
            'num_conv_layers': num_conv_layers,
            'conv_channels': conv_channels,
            'conv_kernel_sizes': conv_kernel_sizes,
            'conv_strides': conv_strides,
            'conv_paddings': conv_paddings,
            'dropout_rates': dropout_rates,
            'maxpool_kernel_sizes': maxpool_kernel_sizes,
        }

        # Build convolutional layers
        conv_layers = []
        in_channels = 1  # Starting from 1 channel after unsqueeze
        for i in range(num_conv_layers):
            out_channels = conv_channels[i]
            kernel_size = conv_kernel_sizes[i]
            stride = conv_strides[i]
            padding = conv_paddings[i]
            conv_layer = WraparoundConv3D(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            )
            conv_layers.append(conv_layer)
            conv_layers.append(nn.BatchNorm3d(out_channels))
            conv_layers.append(nn.ReLU())
            maxpool_kernel_size = maxpool_kernel_sizes[i]
            conv_layers.append(nn.MaxPool3d(kernel_size=maxpool_kernel_size))
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
            # batch_first=False,  # seq_len as first dimension
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
        # Function to compute the output size after convolution and pooling
        def compute_output_size(input_size, kernel_size, stride, padding):
            return (input_size + 2 * padding - kernel_size) // stride + 1

        # Initial sizes
        p_size = 12
        o_size = self.hparams.o
        t_size = self.hparams.t

        for i in range(self.hparams.num_conv_layers):
            # Wraparound padding adds 2 to pitch size
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
        return feature_dim

    def forward_one(self, x):
        # x shape: (batch, 12, o, t)
        x = x.unsqueeze(1)  # Add channel dimension: (batch, 1, 12, o, t)
        x = self.conv(x)
        # Flatten for transformer: (batch, features, seq_len)
        x = rearrange(x, 'b c p o t -> b (c p o) t')
        x = x.permute(2, 0, 1)  # (seq_len, batch, features)
        x = self.feature_projection(x)  # Project features to transformer_d_model
        x = self.transformer(x)
        x = x.mean(dim=0)  # (batch, features)
        x = self.fc(x)
        x = nn.functional.normalize(x, p=2, dim=1)  # Euclidean normalization NOTE lose one degree of freedom in exchange for more stability?
        return x
    
    def compute_distance(self, emb1, emb2):
        return torch.norm(emb1 - emb2, p=2, dim=1)

    def forward(self, x1, x2):
        emb1 = self.forward_one(x1)
        emb2 = self.forward_one(x2)
        return self.compute_distance(emb1, emb2)
    
    def train_dataloader(self):
        train_dataset = MIDIDataset(self.hparams.data_dir, self.hparams.t, split='train')
        return DataLoader(train_dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=0)

    def training_step(self, batch, batch_idx):
        (x1, x2), y = batch
        distance = self.forward(x1, x2)
        y = y.float()
        loss = self.criterion(distance, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def val_dataloader(self):
        val_dataset_similar = MIDIDataset(
            self.hparams.data_dir, self.hparams.t, split='val', pair_type='similar'
        )
        val_dataset_dissimilar = MIDIDataset(
            self.hparams.data_dir, self.hparams.t, split='val', pair_type='dissimilar'
        )
        val_dataset_mixed = MIDIDataset(
            self.hparams.data_dir, self.hparams.t, split='val', pair_type='mixed'
        )
        val_loader_similar = DataLoader(
            val_dataset_similar, batch_size=self.hparams.batch_size, num_workers=0
        )
        val_loader_dissimilar = DataLoader(
            val_dataset_dissimilar, batch_size=self.hparams.batch_size, num_workers=0
        )
        val_loader_mixed = DataLoader(
            val_dataset_mixed, batch_size=self.hparams.batch_size, num_workers=0
        )
        return [val_loader_mixed, val_loader_similar, val_loader_dissimilar]

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        (x1, x2), y = batch
        distance = self.forward(x1, x2)
        y = y.float()
        loss = self.criterion(distance, y)
        
        if dataloader_idx == 0:
            # Mixed pairs
            thresholds = torch.linspace(0, 1, steps=100)  # Sample 100 thresholds from 0 to 1
            accuracies = []

            for threshold in thresholds:
                preds = (distance < threshold).long()
                acc = (preds == y.long()).float().mean()
                accuracies.append(acc.item())

            accuracies = torch.tensor(accuracies)
            max_idx = torch.argmax(accuracies)
            optimal_threshold = thresholds[max_idx].item()
            val_acc_mixed = accuracies[max_idx].item()

            acc = val_acc_mixed

            self.log('val_loss_mixed', loss, on_step=False, on_epoch=True, prog_bar=True, add_dataloader_idx=False)
            self.log('val_acc_mixed', val_acc_mixed, on_step=False, on_epoch=True, prog_bar=True, add_dataloader_idx=False)
            self.log('optimal_threshold', optimal_threshold, on_step=False, on_epoch=True, prog_bar=True, add_dataloader_idx=False)

            self.dynamic_threshold = optimal_threshold
        else:
            preds = (distance < self.dynamic_threshold).long()
            acc = (preds == y.long()).float().mean()

            if dataloader_idx == 1:
                # Similar pairs
                self.log('val_loss_similar', loss, on_step=False, on_epoch=True, prog_bar=True, add_dataloader_idx=False)
                self.log('val_acc_similar', acc, on_step=False, on_epoch=True, prog_bar=True, add_dataloader_idx=False)
            elif dataloader_idx == 2:
                # Dissimilar pairs
                self.log('val_loss_dissimilar', loss, on_step=False, on_epoch=True, prog_bar=True, add_dataloader_idx=False)
                self.log('val_acc_dissimilar', acc, on_step=False, on_epoch=True, prog_bar=True, add_dataloader_idx=False)
            else:
                raise AssertionError("dataloader_idx out of bounds")

        return {'val_loss': loss, 'val_acc': acc}
    
    def configure_optimizers(self):
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

        num_epochs = self.trainer.max_epochs
        num_training_steps = self.trainer.estimated_stepping_batches

        warmup_steps = int(num_training_steps * (5 / num_epochs))
        anneal_start_epoch = max(0, num_epochs - 10)
        anneal_start_steps = int(num_training_steps * (anneal_start_epoch / num_epochs))
        anneal_steps = num_training_steps - anneal_start_steps

        warmup_scheduler = {
            'scheduler': torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1e-6, end_factor=1.0, total_iters=warmup_steps
            ),
            'interval': 'step',
            'frequency': 1
        }

        plateau_scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.1,
                patience=3,
                verbose=True
            ),
            'monitor': 'val_loss_mixed',
            'interval': 'epoch',
            'frequency': 1
        }

        annealing_scheduler = {
            'scheduler': torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1.0, end_factor=0.0, total_iters=anneal_steps
            ),
            'interval': 'step',
            'frequency': 1
        }

        schedulers = [warmup_scheduler, plateau_scheduler, annealing_scheduler]
        return [optimizer], schedulers

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, distance, label):
        return (
            (label) * 0.5 * distance.pow(2) + \
            (1 - label) * 0.5 * torch.clamp(self.margin - distance, min=0.0).pow(2)
        ).mean()
