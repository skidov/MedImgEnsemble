import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.metrics import DiceLoss


class ModelUNet(L.LightningModule):
    def __init__(self, in_channels=3, lr=3e-4):
        super().__init__()
        self.lr = lr

        channels = [64, 128, 256, 512, 1024]

        self.start_conv = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels[0], channels[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
        )

        self.down_block1 = DownBlock(channels[0], channels[1], channels[1])
        self.down_block2 = DownBlock(channels[1], channels[2], channels[2])
        self.down_block3 = DownBlock(channels[2], channels[3], channels[3])
        self.down_block4 = DownBlock(channels[3], channels[4], channels[4] // 2)

        self.up_block1 = UpBlock(channels[4], channels[3] // 2)
        self.up_block2 = UpBlock(channels[3], channels[2] // 2)
        self.up_block3 = UpBlock(channels[2], channels[1] // 2)
        self.up_block4 = UpBlock(channels[1], channels[0])

        self.out_block = nn.Sequential(
            nn.Conv2d(channels[0], 1, kernel_size=1),
            nn.Sigmoid(),
        )

        self.criterion = DiceLoss()

    def forward(self, x):
        d1 = self.start_conv(x)
        d2 = self.down_block1(d1)
        d3 = self.down_block2(d2)
        d4 = self.down_block3(d3)
        d5 = self.down_block4(d4)
        x = self.up_block1(d5, d4)
        x = self.up_block2(x, d3)
        x = self.up_block3(x, d2)
        x = self.up_block4(x, d1)
        out = self.out_block(x)
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


class DownBlock(nn.Module):
    def __init__(self, in_channels, mid_channel, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels, mid_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channel, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
