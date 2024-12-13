import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.model_fcn import ModelFCN
from model.model_tri_unet import ModelTriUNet
from model.model_unet import ModelUNet
from utils.metrics import DiceLoss


class ModelEnsemble(L.LightningModule):
    def __init__(self, fcn=None, unet=None, tri_unet=None, lr=3e-4):
        super().__init__()
        self.lr = lr

        if fcn is None:
            self.fcn = ModelFCN()
        else:
            self.fcn = fcn
        if unet is None:
            self.unet = ModelUNet()
        else:
            self.unet = unet
        if tri_unet is None:
            self.tri_unet = ModelTriUNet()
        else:
            self.tri_unet = tri_unet

        self.criterion = DiceLoss()

    def forward(self, x):
        x1 = self.fcn(x)
        x2 = self.unet(x)
        x3 = self.tri_unet(x)

        out = torch.mean(torch.stack((x1, x2, x3)), dim=0)
        return out

    def training_step(self, batch, batch_idx):
        pred_mask = self(batch[0])
        mask = batch[1].unsqueeze(1)

        loss = self.criterion(pred_mask, mask)
        dice = 1 - loss

        self.log("train_loss", loss, on_epoch=True)
        self.log("train_dice", dice, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        pred_mask = self(batch[0])
        mask = batch[1].unsqueeze(1)

        loss = self.criterion(pred_mask, mask)
        dice = 1 - loss

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_dice", dice, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        pred_mask = self(batch[0])
        mask = batch[1].unsqueeze(1)

        loss = self.criterion(pred_mask, mask)
        dice = 1 - loss

        self.log("test_loss", loss)
        self.log("test_dice", dice)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
