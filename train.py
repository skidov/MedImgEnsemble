import datetime

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

import wandb
from data.kvasir_seg_datamodule import KvasirSEGDataModule
from model.model_baseline import ModelBaseline
from model.model_ensemble import ModelEnsemble
from model.model_fcn import ModelFCN
from model.model_tri_unet import ModelTriUNet
from model.model_unet import ModelUNet


def train_model(data_module, model, name, max_epoch=100):
    checkpoint_callback = ModelCheckpoint()
    early_stop_callback = EarlyStopping(monitor="val_loss", mode="min", patience=5, verbose=False)
    wandb_logger = WandbLogger(name=name, project="kvasir_segmentation", log_model="all")

    trainer = L.Trainer(callbacks=[checkpoint_callback, early_stop_callback], logger=wandb_logger, max_epochs=max_epoch, deterministic=True)

    trainer.fit(model, data_module)
    trainer.test(model, dataloaders=data_module.test_dataloader())
    trainer.save_checkpoint(f"trained_models/{name}.ckpt")

    wandb.finish()

    return name


def eval_model(data_module, model, name):
    wandb_logger = WandbLogger(name=name, project="kvasir_segmentation", log_model="all")

    trainer = L.Trainer(logger=wandb_logger, deterministic=True)

    trainer.test(model, dataloaders=data_module.test_dataloader())

    wandb.finish()

    return name


def train_baseline(max_epoch=100, batch_size=64, lr=3e-4):
    data_module = KvasirSEGDataModule(batch_size=batch_size)
    model = ModelBaseline(lr=lr)

    name = f"baseline_{datetime.datetime.now().strftime('%y%m%d%H%M')}"

    return train_model(data_module, model, name, max_epoch)


def train_fcn(max_epoch=100, batch_size=5, lr=3e-4):
    data_module = KvasirSEGDataModule(batch_size=batch_size)
    model = ModelFCN(lr=lr)

    name = f"fcn_{datetime.datetime.now().strftime('%y%m%d%H%M')}"

    return train_model(data_module, model, name, max_epoch)


def train_unet(max_epoch=100, batch_size=32, lr=3e-4):
    data_module = KvasirSEGDataModule(batch_size=batch_size)
    model = ModelUNet(lr=lr)

    name = f"unet_{datetime.datetime.now().strftime('%y%m%d%H%M')}"

    return train_model(data_module, model, name, max_epoch)


def train_tri_unet(max_epoch=100, batch_size=10, lr=3e-4):
    data_module = KvasirSEGDataModule(batch_size=batch_size)
    model = ModelTriUNet(lr=lr)

    name = f"tri_unet_{datetime.datetime.now().strftime('%y%m%d%H%M')}"

    return train_model(data_module, model, name, max_epoch)


def train_ensemble(model_fcn=None, model_unet=None, model_tri_unet=None, max_epoch=100, batch_size=5, lr=3e-4):
    data_module = KvasirSEGDataModule(batch_size=batch_size)
    model = ModelEnsemble(fcn=model_fcn, unet=model_unet, tri_unet=model_tri_unet, lr=lr)

    name = f"ensemble_{datetime.datetime.now().strftime('%y%m%d%H%M')}"

    return train_model(data_module, model, name, max_epoch)
