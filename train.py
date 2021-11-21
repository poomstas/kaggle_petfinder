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
from src.utils import print_config, separate_train_val
from torch.utils.data import DataLoader
import wandb

# %%
TRAIN_CSV_PATH = './data/train.csv'
TEST_CSV_PATH = './data/test.csv'
VAL_FRAC = 0.1

separate_train_val(TRAIN_CSV_PATH, val_frac=VAL_FRAC)

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

wandb.init(project='PetFinder', entity='poomstas', mode='disabled')
wandb.config = config # value reference as: wandb.config.epochs

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
    A.OneOf([
            A.RandomRotate90(p=0.5), 
            A.Rotate(p=0.5)],
        p=0.5),
    A.ColorJitter (brightness=0.2, contrast=0.2, p=0.3),
    A.ChannelShuffle(p=0.3),
    A.Normalize(NORMAL_MEAN, NORMAL_STD),
    # A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, always_apply=False, p=0.5)
])

AUG_VALTEST = A.Compose([
    A.Normalize(NORMAL_MEAN, NORMAL_STD),
])

# %% No separate testing data used. "Test" is used as validation set.
dataset_train = PetDataset(csv_file_path = './data/separated_train.csv', 
                           type_trainvaltest='train',
                           augments=AUG_TRAIN, 
                           target_size=TARGET_SIZE),

dataset_val   = PetDataset(csv_file_path = './data/separated_val.csv',
                           type_trainvaltest='val',
                           augments=AUG_VALTEST, 
                           target_size=TARGET_SIZE),

dataset_test  = PetDataset(csv_file_path = './data/test.csv',
                           type_trainvaltest='test',
                           augments=AUG_VALTEST, 
                           target_size=TARGET_SIZE),


dataloader_train = DataLoader(dataset = dataset_train,
                              batch_size = config['batch_size'],
                              drop_last = config['drop_last'],
                              shuffle = config['train_shuffle'],
                              num_workers = config['num_workers'])

dataloader_val   = DataLoader(dataset = dataset_val,
                              batch_size = config['batch_size'],
                              drop_last = config['drop_last'],
                              shuffle = config['val_shuffle'],
                              num_workers = config['num_workers'])

dataloader_test  = DataLoader(dataset = dataset_test,
                              batch_size = config['batch_size'],
                              drop_last = config['drop_last'],
                              shuffle = config['val_shuffle'],
                              num_workers = config['num_workers'])

# %% This throws an error for some reason... need to check.
# temp = iter(dataloader_train)
# img, metadata, pawpularity = next(temp)

# temp = iter(dataloader_val)
# img, metadata, pawpularity = next(temp)

# temp = iter(dataloader_test)
# img, metadata, pawpularity = next(temp)

# %%
# wandb.log({'loss': loss})
