import random

import train
import wandb
from data import kvasir_seg

random.seed(42)
wandb.login()

name = ""

while True:
    if name == "":
        print("\nCommand list:")
        print("train_baseline:\n\tTrain baseline model")
        print("train_fcn:\n\tTrain FCN model")
        print("train_unet:\n\tTrain UNet model")
        print("show_masked_img i_img:\n\tShow masked image\n\ti_img: Number of the image")
        print("show_predicted_img model_name i_img:\n\tShow masked image\n\tmodel_name: Name of the model checkpoint\n\ti_img: Number of the image")
        print("exit:\n\tExit application")

        choice = input("Enter your command: ")
        cmd = choice.split(" ")

        if len(cmd) == 1 and cmd[0] == "train_baseline":
            print("\n Train baseline model")
            name = train.train_baseline(lr=3e-4)
        elif len(cmd) == 1 and cmd[0] == "train_fcn":
            print("\n Train FCN model")
            name = train.train_fcn(lr=3e-4)
        elif len(cmd) == 1 and cmd[0] == "train_unet":
            print("\n Train UNet model")
            name = train.train_baseline(lr=3e-4)
        elif len(cmd) == 2 and cmd[0] == "show_masked_img":
            print("\n Show masked image")
            kvasir_seg.show_kvasir_seg(int(cmd[1]))
        elif len(cmd) == 3 and cmd[0] == "show_predicted_img":
            print("\n Show predicted image")
            kvasir_seg.show_predicted_img(kvasir_seg.load_model(cmd[1]), int(cmd[2]))
        elif len(cmd) == 1 and cmd[0] == "exit":
            print("\n Exit Application")
            break
        else:
            print("\n Not Valid Choice Try again")
    else:
        print("\nCommand list:")
        print("masked_img i_img:\n\tShow masked image\n\ti_img: Number of the image")
        print("back:\n\tBack to main menu")
        print("exit:\n\tExit application")

        choice = input("Enter your command: ")
        cmd = choice.split(" ")

        if len(cmd) == 2 and cmd[0] == "masked_img":
            print("\n Show masked image")
            kvasir_seg.show_predicted_img(kvasir_seg.load_model(name), int(cmd[1]))
        elif len(cmd) == 1 and cmd[0] == "back":
            print("\n Back to main menu")
            name = ""
        elif len(cmd) == 1 and cmd[0] == "exit":
            print("\n Exit Application")
            break
        else:
            print("\n Not Valid Choice Try again")
