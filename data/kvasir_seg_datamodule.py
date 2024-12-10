import albumentations as A
import cv2
import lightning as L
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, random_split

from data.kvasir_seg_dataset import KvasirSEGDataset


class KvasirSEGDataModule(L.LightningDataModule):
    def __init__(
        self,
        batch_size=64,
        num_workers=0,
        test_ratio=0.1,
        val_ratio=0.1,
        img_size=(224, 224),
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.test_ratio = test_ratio
        self.val_ratio = val_ratio
        self.img_size = img_size

        self.training_transform = A.Compose(
            [
                A.Resize(*self.img_size, interpolation=cv2.INTER_LANCZOS4),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                ToTensorV2(),
            ]
        )

        self.full_data = None
        self.train_data = None
        self.val_data = None
        self.test_data = None

    def setup(self, stage=None):
        self.full_data = KvasirSEGDataset(transform=self.training_transform)
        self.train_data, self.val_data, self.test_data = random_split(
            self.full_data,
            [1 - self.test_ratio - self.val_ratio, self.test_ratio, self.val_ratio],
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
