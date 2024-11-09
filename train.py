import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from model import MIDIClassifier

if __name__ == '__main__':
    # Define hyperparameters
    hparams = {
        'embedding_dim': 500,
        'data_dir': './synthetic_data',
        't': 128,
        'o': 8,  # Number of octaves
        'batch_size': 32,
        'lr': 1e-3,
        'threshold': 0.5,
        'num_conv_layers': 3,
        'conv_channels': [16, 32, 64],
        'conv_kernel_sizes': [(7,3,7), (5,3,5), (3,3,3)],
        'conv_strides': [(1,1,1), (1,1,1), (1,1,1)],
        'conv_paddings': [(3,1,3), (2,1,2), (1,1,1)],
        'dropout_rates': [0.3, 0.3, 0.3],
        'maxpool_kernel_sizes': [(1,1,2), (1,1,2), (1,1,2)],
        'transformer_d_model': 512,
        'transformer_nhead': 8,
        'transformer_num_layers': 6,
        'fc_hidden_dims': [256, 512],
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
        callbacks=[early_stop_callback, checkpoint_callback],
        gradient_clip_val=1.0,
    )

    trainer.fit(model)
