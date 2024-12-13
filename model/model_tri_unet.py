import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.model_unet import ModelUNet
from utils.metrics import DiceLoss


class ModelTriUNet(L.LightningModule):
    def __init__(self, lr=3e-4):
        super().__init__()
        self.lr = lr

        self.unet1 = ModelUNet()
        self.unet2 = ModelUNet()
        self.unet3 = ModelUNet(in_channels=2)

        self.criterion = DiceLoss()

    def forward(self, x):
        x1 = self.unet1(x)
        x2 = self.unet2(x)

        x = torch.cat((x1, x2), 1)
        out = self.unet3(x)
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
