import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from model import SiaViT
from fire import Fire

TICKS_PER_MINUTE = 20 * 60

def train(
        max_epochs=150,
        devices=1,
        batch_size=128
):

    torch.set_float32_matmul_precision('high')

    # Instantiate the model
    model = SiaViT(
        embedding_dim=128,
        data_dir='./data/expon_decay',
        data_hparams=(0.1, 0.),
        t=TICKS_PER_MINUTE,
        o=6,
        batch_size=batch_size,
        lr=5e-6,
        num_conv_layers=3,
        conv_channels=[32, 64, 128],
        conv_kernel_sizes=[(5, 5, 5), (3, 3, 3), (3, 3, 3)],
        conv_strides=[(2, 2, 2), (1, 1, 2), (1, 1, 1)],
        conv_paddings=[(2, 2, 2), (1, 1, 1), (1, 1, 1)],
        dropout_rates=[0.5, 0.4, 0.4],
        maxpool_kernel_sizes=[(2, 1, 2), (2, 1, 2), (1, 1, 2)],
        use_residual=True,
        transformer_d_model=512,
        transformer_nhead=8,
        transformer_encoder_size=1024,
        transformer_num_layers=3,
        transformer_dropout=0.3,
        fc_hidden_dims=[512, 256],
        weight_decay=1e-5,
        use_AdamW=True,
        cl_margin=0.1,
        cl_margin_dynamic=True,
        warmup_proportion=0.1,
        wraparound_layers=[True, False, False],
        cl_min_start=0.1,
        cl_min_increase_per_epoch=0,
        
    )

    # Set up logger
    logger = TensorBoardLogger("tb_logs", name="SiaViT")

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss_mixed',
        filename='{epoch:02d}-{val_loss_mixed:.5f}',
        save_top_k=3,
        save_last=True,
        mode='min',
    )

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        logger=logger,
        callbacks=[checkpoint_callback],
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=devices,
        gradient_clip_val=1.0,
        enable_progress_bar=True
    )

    # Train the model
    trainer.fit(model)

if __name__ == '__main__':
    Fire(train)