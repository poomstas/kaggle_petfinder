# %%
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import albumentations as A
from src.model import Xception #DenseNet121 # Add more as I go here
from src.data import PetDataset
from src.utils import print_config
from torch.utils.data import DataLoader
import wandb

# %%
config = {
  'gpu_index': 0,
  'model': 'Xception',
  'batch_size': 128,
  'drop_last': True,
  'train_shuffle': True,
  'val_shuffle': False,
  'num_workers': 8,
  'learning_rate': 0.001,
  'epochs': 100,
  'gpu_index': 0,
  'TB_note': ''
  # Add structural design components here, e.g. Number of hidden layers for diff branches
}

# wandb.init(project="PetFinder", entity="poomstas")
# wandb.config = config

# value reference as: wandb.config.epochs

# wandb.log({"loss": loss})

# %%
DEVICE = torch.device('cuda:{}'.format(config['gpu_index']) if torch.cuda.is_available() else 'cpu')
print('{}\nDevice: {}\nModel: {}'.format('='*80, DEVICE, config['model']))
print_config(config)

if config['model'] == 'Xception':
    model = Xception().to(DEVICE)
    TARGET_SIZE = 299
    NORMAL_MEAN = [0.5, 0.5, 0.5]
    NORMAL_STD = [0.5, 0.5, 0.5]

elif config['model'] == 'DenseNet121':
    model = DenseNet121().to(DEVICE)
    TARGET_SIZE = 224
    NORMAL_MEAN = [0.485, 0.456, 0.406]
    NORMAL_STD = [0.229, 0.224, 0.225]

# %% Augmentations
AUG_TRAIN = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.OneOf([A.RandomRotate90(p=0.5), A.Rotate(p=0.5)],p=0.5),
    A.ColorJitter (brightness=0.2, contrast=0.2, p=0.3),
    A.ChannelShuffle(p=0.3),
    A.Normalize(NORMAL_MEAN, NORMAL_STD),
    # A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, always_apply=False, p=0.5)
])

AUG_VAL = A.Compose([
    A.Normalize(NORMAL_MEAN, NORMAL_STD),
])

# %% No separate testing data used. "Test" is used as validation set.
dataloader_train = DataLoader(dataset=PetDataset(type_trainval='train', 
                                                 augments=AUG_TRAIN, 
                                                 target_size=TARGET_SIZE),
                              batch_size = config['batch_size'],
                              drop_last = config['drop_last'],
                              shuffle = config['train_shuffle'],
                              num_workers = config['num_workers'])

dataloader_val   = DataLoader(dataset=PetDataset(type_trainval='val',
                                                 augments=AUG_VAL, 
                                                 target_size=TARGET_SIZE),
                              batch_size = config['batch_size'],
                              drop_last = config['drop_last'],
                              shuffle = config['val_shuffle'],
                              num_workers = config['num_workers'])

# %%
temp = iter(dataloader_train)
img, metadata, pawpularity = next(temp)

# %%
# DATA_DIR = './data/'
# TRAIN_DIR = os.path.join(DATA_DIR, 'train')
# TEST_DIR = os.path.join(DATA_DIR, 'test')

# df_train = pd.read_csv('./data/train.csv')
# df_test  = pd.read_csv('./data/test.csv')
# df_train['filename'] = [Id + '.jpg' for Id in df_train['Id']]
# df_test['filename'] = [Id + '.jpg' for Id in df_test['Id']]

