import random

from lightning.pytorch import seed_everything

import train
import wandb

wandb.login()

random.seed(42)
seed_everything(42, workers=True)
train.train_baseline(lr=3e-4)

random.seed(42)
seed_everything(42, workers=True)
train.train_fcn(lr=3e-4)

random.seed(42)
seed_everything(42, workers=True)
train.train_unet(lr=3e-4)

random.seed(42)
seed_everything(42, workers=True)
train.train_tri_unet(lr=3e-4)

random.seed(42)
seed_everything(42, workers=True)
train.train_ensemble(lr=3e-4)
