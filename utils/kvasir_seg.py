import os
import random

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from data.kvasir_seg_dataset import KvasirSEGDataset
from model.model_baseline import ModelBaseline
from model.model_ensemble import ModelEnsemble
from model.model_fcn import ModelFCN
from model.model_tri_unet import ModelTriUNet
from model.model_unet import ModelUNet


def load_model(name):
    if "baseline_" in name:
        return ModelBaseline.load_from_checkpoint(checkpoint_path=f"trained_models/{name}.ckpt")
    if "fcn_" in name:
        return ModelFCN.load_from_checkpoint(checkpoint_path=f"trained_models/{name}.ckpt")
    if "tri_unet_" in name:
        return ModelTriUNet.load_from_checkpoint(checkpoint_path=f"trained_models/{name}.ckpt")
    if "unet_" in name:
        return ModelUNet.load_from_checkpoint(checkpoint_path=f"trained_models/{name}.ckpt")
    if "ensemble_" in name:
        return ModelEnsemble.load_from_checkpoint(checkpoint_path=f"trained_models/{name}.ckpt")


def show_masked_img(dataset, rows):
    ## Figure
    fig = plt.figure(figsize=(9, 3 * rows))
    plt.axis("off")
    plt.title("Kvasir-SEG", fontweight="bold", fontsize=18, y=1.1)

    for i_row in range(rows):
        i_image = random.randrange(0, len(dataset))
        data = dataset[i_image]
        image = data[0]
        mask = data[1].astype(np.uint8) * 255
        mask_inv = cv2.bitwise_not(mask)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        colored_portion = cv2.bitwise_or(image, image, mask=mask)

        gray_portion = cv2.bitwise_or(gray, gray, mask=mask_inv)
        gray_portion = np.stack((gray_portion,) * 3, axis=-1)

        output = colored_portion + gray_portion

        # Img 1
        plt.subplot(rows, 3, i_row * 3 + 1)
        plt.imshow(image)
        plt.axis("off")
        plt.title("Image")
        # Img 2
        plt.subplot(rows, 3, i_row * 3 + 2)
        plt.imshow(mask)
        plt.axis("off")
        plt.title("Mask")
        # Img 3
        plt.subplot(rows, 3, i_row * 3 + 3)
        plt.imshow(output)
        plt.axis("off")
        plt.title("Masked image")

    fig.show()

    if not os.path.exists("images"):
        os.makedirs("images")
    fig.savefig("images/masked_img.png")


def show_predicted_img(model, rows, name):
    transform = A.Compose(
        [
            A.Resize(*(224, 224), interpolation=cv2.INTER_LANCZOS4),
        ]
    )

    dataset = KvasirSEGDataset(transform)

    ## Figure
    fig = plt.figure(figsize=(12, 3 * rows))
    plt.axis("off")
    plt.title("Predicted mask", fontweight="bold", fontsize=18, y=1.1)

    for i_row in range(rows):
        i_image = random.randrange(0, len(dataset))

        data = dataset[i_image]
        image = data[0]
        mask = data[1].astype(np.uint8) * 255

        model.eval()
        with torch.no_grad():
            predicted_mask = model(torch.from_numpy(image).movedim(2, 0).unsqueeze(0)).squeeze(0)
            predicted_mask = predicted_mask.movedim(0, 2).numpy()
            predicted_mask = cv2.threshold(predicted_mask, 0.5, 1, cv2.THRESH_BINARY)[1]
            predicted_mask = predicted_mask.astype(np.uint8) * 255

        mask_inv = cv2.bitwise_not(predicted_mask)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        colored_portion = cv2.bitwise_or(image, image, mask=predicted_mask)

        gray_portion = cv2.bitwise_or(gray, gray, mask=mask_inv)
        gray_portion = np.stack((gray_portion,) * 3, axis=-1)

        masked_image = colored_portion + gray_portion

        # Img 1
        plt.subplot(rows, 4, i_row * 4 + 1)
        plt.imshow(image)
        plt.axis("off")
        plt.title("Image")
        # Img 2
        plt.subplot(rows, 4, i_row * 4 + 2)
        plt.imshow(mask)
        plt.axis("off")
        plt.title("Ground Truth")
        # Img 3
        plt.subplot(rows, 4, i_row * 4 + 3)
        plt.imshow(predicted_mask)
        plt.axis("off")
        plt.title("Predicted Mask")
        # Img 4
        plt.subplot(rows, 4, i_row * 4 + 4)
        plt.imshow(masked_image)
        plt.axis("off")
        plt.title("Masked image")

    fig.show()

    if not os.path.exists("images"):
        os.makedirs("images")
    fig.savefig(f"images/{name}.png")


def show_kvasir_seg(i_image):
    dataset = KvasirSEGDataset()
    show_masked_img(dataset, i_image)
