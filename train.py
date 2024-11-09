import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from model import MIDIClassifier

torch.set_float32_matmul_precision('high')

TICKS_PER_MINUTE = 20 * 60

if __name__ == '__main__':
    hparams = {
        'embedding_dim': 256,
        'data_dir': './data',
        't': 4 * TICKS_PER_MINUTE,
        'o': 6, 
        'batch_size': 26,
        'lr': 1e-5,
        'threshold': 1.0,
        'num_conv_layers': 4,
        'conv_channels': [32, 64, 128, 256],
        'conv_kernel_sizes': [(5,3,5), (5,3,5), (3,3,3), (3,3,3)],
        'conv_strides': [(1,1,2), (1,1,2), (1,1,2), (1,1,2)],
        'conv_paddings': [(2,1,2), (2,1,2), (1,1,1), (1,1,1)],
        'dropout_rates': [0.2, 0.2, 0.2, 0.2],
        'maxpool_kernel_sizes': [(1,1,2), (1,1,2), (1,1,2), (1,1,2)],
        'transformer_d_model': 256,
        'transformer_nhead': 8,
        'transformer_num_layers': 4,
        'fc_hidden_dims': [512, 256],
        'weight_decay': 0,
        'use_AdamW': False
    }

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
    model = MIDIClassifier(**hparams)

    # Set up logger
    logger = TensorBoardLogger("tb_logs", name="BIG-MIDIClassifier")

    # Define callbacks
    early_stop_callback = EarlyStopping(
        monitor='val_acc',
        min_delta=0.00,
        patience=5,
        verbose=True,
        mode='max'
    )

    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc_mixed',  # Monitor the combined validation accuracy
        filename='{epoch:02d}-{val_acc_mixed:.2f}',
        save_top_k=3,
        save_last=True,
        mode='max',
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
