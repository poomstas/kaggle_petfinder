# %%
import os
import torch
from torch._C import device
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import albumentations as A
from src.model import Xception, DenseNet121 # Add more as I go here
import wandb

# %%
config = {
  'gpu_index': 0,
  'model': 'Xception',
  'batch_size': 128,
  'learning_rate': 0.001,
  'epochs': 100,
  'TB_note': ''
  # Add structural design components here, e.g. Number of hidden layers for diff branches
}

# wandb.init(project="PetFinder", entity="poomstas")
# wandb.config = config

# value reference as: wandb.config.epochs

# wandb.log({"loss": loss})

# %%
DATA_DIR = './data/'
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')

df_train = pd.read_csv('./data/train.csv')
df_test  = pd.read_csv('./data/test.csv')
df_train['filename'] = [Id + '.jpg' for Id in df_train['Id']]
df_test['filename'] = [Id + '.jpg' for Id in df_test['Id']]

# %%
if not torch.cuda.is_available():
    print("CUDA IS UNAVAILABLE!! Check connections.")

# %%
DEVICE = torch.device('cuda:{}'.format(config.gpu_index) if torch.cuda.is_available() else 'cpu')
print('Device: {}\nModel: {}'.format(DEVICE, config.model))

if config.model == 'Xception':
    model = Xception().to(DEVICE)
elif config.model == 'DenseNet121':
    model = DenseNet121().to(DEVICE)

# %%

# %%







