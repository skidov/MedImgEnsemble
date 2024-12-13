import random

from lightning.pytorch import seed_everything

import train
import wandb
from utils import kvasir_seg

wandb.login()


name = ""

while True:
    print("\n")
    random.seed(42)
    seed_everything(42, workers=True)

    if name == "":
        print("\nCommand list:")
        print("train_baseline")
        print("train_fcn")
        print("train_unet")
        print("train_tri_unet")
        print("train_ensemble")
        print("show_masked_img rows\n\trows: Number of rows")
        print("show_predicted_img model_name rows\n\tmodel_name: Name of the model checkpoint\n\trows: Number of rows")
        print("exit")

        choice = input("Enter your command: ")
        cmd = choice.split(" ")

        if len(cmd) == 1 and cmd[0] == "train_baseline":
            print("\n Train baseline model")
            name = train.train_baseline(lr=3e-4, max_epoch=100)

        elif len(cmd) == 1 and cmd[0] == "train_fcn":
            print("\n Train FCN model")
            name = train.train_fcn(lr=3e-4, max_epoch=100)

        elif len(cmd) == 1 and cmd[0] == "train_unet":
            print("\n Train UNet model")
            name = train.train_unet(lr=3e-4, max_epoch=100)

        elif len(cmd) == 1 and cmd[0] == "train_tri_unet":
            print("\n Train TriUNet model")
            name = train.train_tri_unet(lr=3e-4, max_epoch=100)

        elif len(cmd) == 1 and cmd[0] == "train_ensemble":
            print("\n Train Ensemble model")
            name = train.train_ensemble(lr=3e-4, max_epoch=100)

        elif len(cmd) == 2 and cmd[0] == "show_masked_img":
            print("\n Show masked image")
            kvasir_seg.show_kvasir_seg(int(cmd[1]))

        elif len(cmd) == 3 and cmd[0] == "show_predicted_img":
            print("\n Show predicted image")
            kvasir_seg.show_predicted_img(kvasir_seg.load_model(cmd[1]), int(cmd[2]), cmd[1])

        elif len(cmd) == 1 and cmd[0] == "exit":
            print("\n Exit Application")
            break
        else:
            print("\n Not Valid Choice Try again")
    else:
        print("\nCommand list:")
        print("masked_img rows\n\trows: Number of rows")
        print("back")
        print("exit")

        choice = input("Enter your command: ")
        cmd = choice.split(" ")

        if len(cmd) == 2 and cmd[0] == "masked_img":
            print("\n Show masked image")
            kvasir_seg.show_predicted_img(kvasir_seg.load_model(name), int(cmd[1]), name)

        elif len(cmd) == 1 and cmd[0] == "back":
            print("\n Back to main menu")
            name = ""

        elif len(cmd) == 1 and cmd[0] == "exit":
            print("\n Exit Application")
            break
        else:
            print("\n Not Valid Choice Try again")
