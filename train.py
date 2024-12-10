import datetime

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

import wandb
from data.kvasir_seg_datamodule import KvasirSEGDataModule
from model.model_baseline import ModelBaseline
from model.model_fcn import ModelFCN
from model.model_unet import ModelUNet


def train_model(data_module, model, name, max_epoch=40):
    checkpoint_callback = ModelCheckpoint()
    early_stop_callback = EarlyStopping(monitor="val_loss", mode="min", patience=2, verbose=False)
    wandb_logger = WandbLogger(name=name, project="kvasir_segmentation", log_model="all")

    trainer = L.Trainer(
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=wandb_logger,
        max_epochs=max_epoch,
    )

    trainer.fit(model, data_module)
    trainer.test(model, dataloaders=data_module.test_dataloader())
    trainer.save_checkpoint(f"trained_models/{name}.ckpt")

    wandb.finish()

    return name


def train_baseline(max_epoch=40, batch_size=64, lr=3e-4):
    data_module = KvasirSEGDataModule(batch_size=batch_size)
    model = ModelBaseline(lr=lr)

    name = f"baseline_{datetime.datetime.now().strftime('%y%m%d%H%M')}"

    return train_model(data_module, model, name, max_epoch)


def train_fcn(max_epoch=40, batch_size=16, lr=3e-4):
    data_module = KvasirSEGDataModule(batch_size=batch_size)
    model = ModelFCN(lr=lr)

    name = f"fcn_{datetime.datetime.now().strftime('%y%m%d%H%M')}"

    return train_model(data_module, model, name, max_epoch)


def train_unet(max_epoch=40, batch_size=16, lr=3e-4):
    data_module = KvasirSEGDataModule(batch_size=batch_size)
    model = ModelUNet(lr=lr)

    name = f"unet_{datetime.datetime.now().strftime('%y%m%d%H%M')}"

    return train_model(data_module, model, name, max_epoch)
