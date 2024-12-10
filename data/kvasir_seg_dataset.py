import os
import shutil
import urllib

import cv2
import numpy as np
from torch.utils.data import Dataset


class KvasirSEGDataset(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        self.dir = "data/Kvasir-SEG"
        self.data = list()

        urllib.request.urlretrieve("https://datasets.simula.no/downloads/kvasir-seg.zip", "data/kvasir-seg.zip")
        shutil.unpack_archive("data/kvasir-seg.zip", "data/")

        images = os.listdir(os.path.join(self.dir, "images"))
        masks = os.listdir(os.path.join(self.dir, "masks"))
        for image in images:
            if image in masks:
                self.data.append(image)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = cv2.imread(os.path.join(self.dir, "images", self.data[idx])).astype(np.float32) / 255.0
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(os.path.join(self.dir, "masks", self.data[idx]), 0).astype(np.float32) / 255.0
        mask = cv2.threshold(mask, 0.5, 1, cv2.THRESH_BINARY)[1]

        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        return image, mask
