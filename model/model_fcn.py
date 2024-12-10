import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Dice, JaccardIndex
from torchmetrics.classification import BinaryAccuracy

from model.metrics import DiceLoss


class ModelFCN(L.LightningModule):
    def __init__(self, lr=3e-4):
        super().__init__()
        self.lr = lr

        self.channels_1 = 64
        self.channels_2 = 128
        self.channels_3 = 256
        self.channels_4 = 512

        self.layer_0 = nn.Sequential(
            nn.Conv2d(3, self.channels_1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channels_1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.layer_1 = nn.Sequential(BasicBlock(self.channels_1, self.channels_1, 1), BasicBlock(self.channels_1, self.channels_1, 1))
        self.upsample1 = nn.Upsample(scale_factor=2, mode="bilinear")
        self.layer_2 = nn.Sequential(BasicBlock(self.channels_1, self.channels_2, 2), BasicBlock(self.channels_2, self.channels_2, 1))
        self.upsample2 = nn.Upsample(scale_factor=4, mode="bilinear")
        self.layer_3 = nn.Sequential(BasicBlock(self.channels_2, self.channels_3, 2), BasicBlock(self.channels_3, self.channels_3, 1))
        self.upsample3 = nn.Upsample(scale_factor=8, mode="bilinear")
        self.layer_4 = nn.Sequential(BasicBlock(self.channels_3, self.channels_4, 2), BasicBlock(self.channels_4, self.channels_4, 1))
        self.upsample4 = nn.Upsample(scale_factor=16, mode="bilinear")

        self.layer_out = nn.Sequential(
            nn.Conv2d(self.channels_1 + self.channels_2 + self.channels_3 + self.channels_4, 1, 1),
            nn.Sigmoid(),
        )

        self.criterion = DiceLoss()
        self.dice = Dice()

    def forward(self, x):
        x = self.layer_0(x)
        x = self.layer_1(x)
        up1 = self.upsample1(x)
        x = self.layer_2(x)
        up2 = self.upsample2(x)
        x = self.layer_3(x)
        up3 = self.upsample3(x)
        x = self.layer_4(x)
        up4 = self.upsample4(x)

        merge = torch.cat([up1, up2, up3, up4], dim=1)
        out = self.layer_out(merge)

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


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(out_channels))

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out
