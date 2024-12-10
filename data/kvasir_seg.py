import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from data.kvasir_seg_dataset import KvasirSEGDataset
from model.model_baseline import ModelBaseline
from model.model_fcn import ModelFCN
from model.model_unet import ModelUNet


def load_model(name):
    if "baseline_" in name:
        return ModelBaseline.load_from_checkpoint(checkpoint_path=f"trained_models/{name}.ckpt")
    if "fcn_" in name:
        return ModelFCN.load_from_checkpoint(checkpoint_path=f"trained_models/{name}.ckpt")
    if "unet_" in name:
        return ModelUNet.load_from_checkpoint(checkpoint_path=f"trained_models/{name}.ckpt")


def show_masked_img(dataset, i_image):
    data = dataset[i_image]
    image = data[0]
    mask = data[1].astype(np.uint8) * 255
    mask_inv = cv2.bitwise_not(mask)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    colored_portion = cv2.bitwise_or(image, image, mask=mask)

    gray_portion = cv2.bitwise_or(gray, gray, mask=mask_inv)
    gray_portion = np.stack((gray_portion,) * 3, axis=-1)

    output = colored_portion + gray_portion

    ## Figure
    plt.figure(figsize=(15, 5))
    plt.axis("off")
    plt.title("Masked Image")
    # Img 1
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.axis("off")
    plt.title("Image")
    # Img 2
    plt.subplot(1, 3, 2)
    plt.imshow(mask)
    plt.axis("off")
    plt.title("Mask")
    # Img 3
    plt.subplot(1, 3, 3)
    plt.imshow(output)
    plt.axis("off")
    plt.title("Masked image")

    plt.show()


def show_predicted_img(model, i_image):
    dataset = KvasirSEGDataset()

    data = dataset[i_image]
    image = data[0]
    mask = data[1].astype(np.uint8) * 255

    model.eval()
    with torch.no_grad():
        predicted_mask = model(torch.from_numpy(image).movedim(2, 0))
        predicted_mask = predicted_mask.movedim(0, 2).numpy()
        predicted_mask = cv2.threshold(predicted_mask, 0.5, 1, cv2.THRESH_BINARY)[1]
        predicted_mask = predicted_mask.astype(np.uint8) * 255

    mask_inv = cv2.bitwise_not(predicted_mask)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    colored_portion = cv2.bitwise_or(image, image, mask=predicted_mask)

    gray_portion = cv2.bitwise_or(gray, gray, mask=mask_inv)
    gray_portion = np.stack((gray_portion,) * 3, axis=-1)

    masked_image = colored_portion + gray_portion

    ## Figure
    plt.figure(figsize=(15, 5))
    plt.axis("off")
    plt.title("Masked Image")
    # Img 1
    plt.subplot(1, 4, 1)
    plt.imshow(image)
    plt.axis("off")
    plt.title("Image")
    # Img 2
    plt.subplot(1, 4, 2)
    plt.imshow(mask)
    plt.axis("off")
    plt.title("Ground Truth")
    # Img 3
    plt.subplot(1, 4, 3)
    plt.imshow(predicted_mask)
    plt.axis("off")
    plt.title("Predicted Mask")
    # Img 4
    plt.subplot(1, 4, 4)
    plt.imshow(masked_image)
    plt.axis("off")
    plt.title("Masked image")

    plt.show()


def show_kvasir_seg(i_image):
    dataset = KvasirSEGDataset()
    show_masked_img(dataset, i_image)
