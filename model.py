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

class MIDIClassifier(pl.LightningModule):
    def __init__(
        self,
        embedding_dim=500,
        data_dir='./data',
        t=128,
        o=8,
        batch_size=32,
        lr=1e-3,
        threshold=0.5,
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
    ):
        super(MIDIClassifier, self).__init__()
        self.save_hyperparameters(ignore=['conv_channels', 'conv_kernel_sizes', 'conv_strides',
                                          'conv_paddings', 'dropout_rates', 'maxpool_kernel_sizes',
                                          'fc_hidden_dims'])
        # Provide default values for lists if None
        if conv_channels is None:
            conv_channels = [16, 32, 64]
        if conv_kernel_sizes is None:
            conv_kernel_sizes = [(7,3,7), (5,3,5), (3,3,3)]
        if conv_strides is None:
            conv_strides = [(1,1,1)] * num_conv_layers
        if conv_paddings is None:
            conv_paddings = [(3,1,3), (2,1,2), (1,1,1)]
        if dropout_rates is None:
            dropout_rates = [0.3] * num_conv_layers
        if maxpool_kernel_sizes is None:
            maxpool_kernel_sizes = [(1,1,2)] * num_conv_layers
        if fc_hidden_dims is None:
            fc_hidden_dims = [256, 512]
        
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
        
        # Initialize feature projection as None
        self.feature_projection = None

        # Transformer encoder with Layer Normalization
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_d_model,
            nhead=transformer_nhead,
            dim_feedforward=2048,
            dropout=0.1,
            activation='relu',
            layer_norm_eps=1e-5,
            batch_first=False,  # seq_len as first dimension
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

        self.criterion = nn.MarginRankingLoss(margin=1.0)

    def forward_one(self, x):
        # x shape: (batch, 12, o, t)
        x = x.unsqueeze(1)  # Add channel dimension: (batch, 1, 12, o, t)
        x = self.conv(x)
        # Flatten for transformer: (batch, features, seq_len)
        x = rearrange(x, 'b c p o t -> b (c p o) t')
        x = x.permute(2, 0, 1)  # (seq_len, batch, features)
        feature_dim = x.size(-1)

        # Initialize feature_projection if not already done
        if self.feature_projection is None:
            self.feature_projection = nn.Linear(feature_dim, self.hparams.transformer_d_model)
            nn.init.xavier_uniform_(self.feature_projection.weight)
            self.feature_projection.to(self.device)

        x = self.feature_projection(x)  # Project features to transformer_d_model
        x = self.transformer(x)
        x = x.mean(dim=0)  # (batch, features)
        x = self.fc(x)
        return x

    def forward(self, x1, x2):
        emb1 = self.forward_one(x1)
        emb2 = self.forward_one(x2)
        # Euclidean distance
        distance = torch.norm(emb1 - emb2, p=2, dim=1)
        return distance, emb1, emb2

    def training_step(self, batch, batch_idx):
        (x1, x2), y = batch
        distance, _, _ = self.forward(x1, x2)
        target = 2 * y - 1  # Convert {0,1} to {-1,1}
        loss = self.criterion(distance, torch.zeros_like(distance), target)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        (x1, x2), y = batch
        distance, _, _ = self.forward(x1, x2)
        preds = (distance < self.hparams.threshold).long()
        acc = (preds == y).float().mean()
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        return acc

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )
        scheduler = {
            'scheduler': ReduceLROnPlateau(optimizer, mode='max', patience=3, verbose=True),
            'monitor': 'val_acc'
        }
        return [optimizer], [scheduler]

    def train_dataloader(self):
        train_dataset = MIDIDataset(self.hparams.data_dir, self.hparams.t, split='train')
        return DataLoader(train_dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=0)

    def val_dataloader(self):
        val_dataset = MIDIDataset(self.hparams.data_dir, self.hparams.t, split='val')
        return DataLoader(val_dataset, batch_size=self.hparams.batch_size, num_workers=0)
