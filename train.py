import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from model import SiaViT

torch.set_float32_matmul_precision('high')

TICKS_PER_MINUTE = 20 * 60

hparams = {
    'embedding_dim': 512,
    'data_dir': './data',
    't': 1 * TICKS_PER_MINUTE,
    'o': 6, 
    'batch_size': 4 * 26,
    'lr': 2e-5,
    'threshold': 0.2,
    'num_conv_layers': 4,
    'conv_channels': [32, 64, 128, 256],
    'conv_kernel_sizes': [(9,3,9), (5,3,5), (3,3,3), (3,3,3)],
    'conv_strides': [(1,1,2), (1,1,2), (1,1,2), (1,1,2)],
    'conv_paddings': [(4,1,4), (2,1,2), (1,1,1), (1,1,1)],
    'dropout_rates': [0.4, 0.3, 0.2, 0.2],
    'maxpool_kernel_sizes': [(1,1,2), (1,1,2), (1,1,2), (1,1,2)],
    'transformer_d_model': 512,
    'transformer_nhead': 8,
    'transformer_num_layers': 4,
    'fc_hidden_dims': [1024, 512],
    'weight_decay': 5e-6,
    'use_AdamW': True
}

if __name__ == '__main__':
    

    # hparams = {
    #     'embedding_dim': 128,
    #     'data_dir': './data',
    #     't': 20 * 180,
    #     'o': 6,  # Number of octaves, based on data shape [12, 6, T]
    #     'batch_size': 32,
    #     'lr': 1e-7,
    #     'threshold': 0.5,
    #     'num_conv_layers': 2,
    #     'conv_channels': [16, 32],
    #     'conv_kernel_sizes': [(5,3,5), (3,3,3)],
    #     'conv_strides': [(1,1,2), (1,1,2)],
    #     'conv_paddings': [(2,1,2), (1,1,1)],
    #     'dropout_rates': [0.3, 0.3],
    #     'maxpool_kernel_sizes': [(1,1,2), (1,1,2)],
    #     'transformer_d_model': 128,
    #     'transformer_nhead': 4,  # transformer_d_model must be divisible by nhead
    #     'transformer_num_layers': 2,
    #     'fc_hidden_dims': [128, 256],
    #     'weight_decay': 1e-5,
    # }

    # Instantiate the model
    model = SiaViT(**hparams)

    # Set up logger
    logger = TensorBoardLogger("tb_logs", name="BIG-MIDIClassifier")

    # Define callbacks
    # early_stop_callback = EarlyStopping(
    #     monitor='val_acc',
    #     min_delta=0.00,
    #     patience=5,
    #     verbose=True,
    #     mode='max'
    # )

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss_mixed',
        filename='{epoch:02d}-{val_loss_mixed:.2f}',
        save_top_k=3,
        save_last=True,
        mode='min',
    )

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=160,
        logger=logger,
        callbacks=[checkpoint_callback],
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        gradient_clip_val=1.0,
        enable_progress_bar=True
    )

    # Train the model
    trainer.fit(model)
