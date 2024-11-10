import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from model import SiaViT

torch.set_float32_matmul_precision('high')

TICKS_PER_MINUTE = 20 * 60

if __name__ == '__main__':

    # Instantiate the model
    model = SiaViT(
        embedding_dim=500,
        data_dir='./data',
        t=TICKS_PER_MINUTE,
        o=6,
        batch_size=128,
        lr=1e-5,
        num_conv_layers=3,
        conv_channels=[16, 32, 64],
        conv_kernel_sizes=[(5, 5, 5), (3, 3, 3), (3, 3, 3)],
        conv_strides=[(2, 2, 2), (1, 1, 2), (1, 1, 2)],
        conv_paddings=[(1, 1, 2), (1, 1, 1), (1, 1, 1)],
        dropout_rates=[0.5, 0.5, 0.4],
        maxpool_kernel_sizes=[(2, 1, 2), (2, 1, 2), (2, 1, 2)],
        transformer_d_model=256,
        transformer_nhead=16,
        transformer_num_layers=2,
        fc_hidden_dims=[128],
        weight_decay=1e-5,
        use_AdamW=True,
        cl_margin=1.0,
        cl_margin_dynamic=True,
        warmup_proportion=0.1,
        wraparound_layers=[True, False, False]
    )

    # Set up logger
    logger = TensorBoardLogger("tb_logs", name="SiaViT")

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss_mixed',
        filename='{epoch:02d}-{val_loss_mixed:.2f}',
        save_top_k=3,
        save_last=True,
        mode='min',
    )

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=500,
        logger=logger,
        callbacks=[checkpoint_callback],
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        gradient_clip_val=1.0,
        enable_progress_bar=True
    )

    # Train the model
    trainer.fit(model)
