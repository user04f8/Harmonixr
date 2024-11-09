import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from model import MIDIClassifier

torch.set_float32_matmul_precision('high')

if __name__ == '__main__':
    # Define hyperparameters
    # hparams = {
    #     'embedding_dim': 500,
    #     'data_dir': './synthetic_data',
    #     't': 128,
    #     'o': 8,  # Number of octaves
    #     'batch_size': 32,
    #     'lr': 1e-6,
    #     'threshold': 0.5,
    #     # 'num_conv_layers': 3,
    #     # 'conv_channels': [16, 32, 64],
    #     # 'conv_kernel_sizes': [(7,3,7), (5,3,5), (3,3,3)],
    #     # 'conv_strides': [(1,1,1), (1,1,1), (1,1,1)],
    #     # 'conv_paddings': [(3,1,3), (2,1,2), (1,1,1)],
    #     # 'dropout_rates': [0.3, 0.3, 0.3],
    #     # 'maxpool_kernel_sizes': [(1,1,2), (1,1,2), (1,1,2)],
    #     'num_conv_layers': 2,
    #     'conv_channels': [16, 32],
    #     'conv_kernel_sizes': [(7,3,7), (5,3,5)],
    #     'conv_strides': [(1,1,1), (1,1,1)],
    #     'conv_paddings': [(3,1,3), (2,1,2)],
    #     'dropout_rates': [0.3, 0.3],
    #     'maxpool_kernel_sizes': [(1,1,2), (1,1,2)],
    #     'transformer_d_model': 256,
    #     'transformer_nhead': 8,
    #     'transformer_num_layers': 2,
    #     'fc_hidden_dims': [256, 512],
    #     'weight_decay': 1e-5,
    # }

    hparams = {
        'embedding_dim': 128,
        'data_dir': './data',
        't': 20 * 180,
        'o': 6,  # Number of octaves, based on data shape [12, 6, T]
        'batch_size': 32,
        'lr': 1e-7,
        'threshold': 0.5,
        'num_conv_layers': 2,
        'conv_channels': [16, 32],
        'conv_kernel_sizes': [(5,3,5), (3,3,3)],
        'conv_strides': [(1,1,2), (1,1,2)],
        'conv_paddings': [(2,1,2), (1,1,1)],
        'dropout_rates': [0.3, 0.3],
        'maxpool_kernel_sizes': [(1,1,2), (1,1,2)],
        'transformer_d_model': 128,
        'transformer_nhead': 4,  # transformer_d_model must be divisible by nhead
        'transformer_num_layers': 2,
        'fc_hidden_dims': [128, 256],
        'weight_decay': 1e-5,
    }

    # Instantiate the model
    model = MIDIClassifier(**hparams)

    # Set up logger
    logger = TensorBoardLogger("tb_logs", name="MIDIClassifier")

    # Define callbacks
    early_stop_callback = EarlyStopping(
        monitor='val_acc',
        min_delta=0.00,
        patience=5,
        verbose=True,
        mode='max'
    )

    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',
        dirpath='checkpoints/',
        filename='MIDIClassifier-{epoch:02d}-{val_acc:.2f}',
        save_top_k=1,
        mode='max',
    )

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=50,
        logger=logger,
        callbacks=[checkpoint_callback],
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        gradient_clip_val=1.0,
        enable_progress_bar=True
    )

    # Train the model
    trainer.fit(model)
