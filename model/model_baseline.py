import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Dice, JaccardIndex
from torchmetrics.classification import BinaryAccuracy

from model.metrics import DiceLoss


class ModelBaseline(L.LightningModule):
    def __init__(self, lr=3e-4):
        super().__init__()
        self.lr = lr

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )

        self.criterion = DiceLoss()
        self.dice = Dice()

    def forward(self, x):
        out = self.conv_block(x)
        return out

    def training_step(self, batch, batch_idx):
        pred_mask = self(batch[0])
        mask = batch[1].unsqueeze(1)

        loss = self.criterion(pred_mask, mask)
        dice = self.dice(pred_mask, mask.type(torch.uint8))

        self.log("train_loss", loss, on_epoch=True)
        self.log("train_dice", dice, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        pred_mask = self(batch[0])
        mask = batch[1].unsqueeze(1)

        loss = self.criterion(pred_mask, mask)
        dice = self.dice(pred_mask, mask.type(torch.uint8))

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_dice", dice, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        pred_mask = self(batch[0])
        mask = batch[1].unsqueeze(1)

        loss = self.criterion(pred_mask, mask)
        dice = self.dice(pred_mask, mask.type(torch.uint8))

        self.log("test_loss", loss)
        self.log("test_dice", dice)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
