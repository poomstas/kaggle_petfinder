# %%
import time
import torch
import torch.nn as nn
from torchvision import transforms

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
import albumentations as A
from tqdm import tqdm
from src.model import Xception #DenseNet121 # Add more as I go here
from src.data import PetDataset
from src.utils import print_config, separate_train_val, get_writer_name
from torch.utils.data import DataLoader
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from albumentations.pytorch.transforms import ToTensorV2
import wandb

# %%
TRAIN_CSV_PATH = './data/train.csv'
TEST_CSV_PATH = './data/test.csv'
VAL_FRAC = 0.1

MODEL_WEIGHTS_SAVE_PATH = './weights/'

separate_train_val(TRAIN_CSV_PATH, val_frac=VAL_FRAC)

# %%
config = {
  'gpu_index': 0,
  'model': 'Xception',
  'batch_size': 128,
  'drop_last': False,
  'train_shuffle': True,
  'val_shuffle': False,
  'num_workers': 8,
  'learning_rate': 0.001,
  'min_lr': 1e-10,
  'patience': 3,
  'reduce_lr_factor': 0.1,
  'epochs': 30,
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

optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
criterion = nn.MSELoss()
# criterion = ignite.metrics.MeanAbsoluteError
# criterion = nn.L1Loss() # output = loss(input, target)

# %% Albumentation Augmentations
# TRANSFORMS_TRAIN = A.Compose([
#     A.HorizontalFlip(p=0.5),
#     A.VerticalFlip(p=0.5),
#     A.OneOf([
#             A.RandomRotate90(p=0.5), 
#             A.Rotate(p=0.5)],
#         p=0.5),
#     A.ColorJitter (brightness=0.2, contrast=0.2, p=0.3),
#     A.ChannelShuffle(p=0.3),
#     A.Normalize(NORMAL_MEAN, NORMAL_STD),
#     # A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, always_apply=False, p=0.5)
#     ToTensorV2()
# ])

# TRANSFORMS_VALTEST = A.Compose([
#     A.Normalize(NORMAL_MEAN, NORMAL_STD),
#     ToTensorV2()
# ])

TRANSFORMS_TRAIN = transforms.Compose([
    transforms.Normalize(NORMAL_MEAN, NORMAL_STD),
    transforms.ToTensor()
])

TRANSFORMS_VALTEST = transforms.Compose([
    transforms.Normalize(NORMAL_MEAN, NORMAL_STD),
    transforms.ToTensor()
])

# %% No separate testing data used. "Test" is used as validation set.

# %% This throws an error for some reason... need to check.

# %% Setup Logging
TB_name = get_writer_name(config)
wandb.run.name = TB_name
tensorboard = SummaryWriter('./runs/' + TB_name)
model_weights_name_base = TB_name + '_Epoch_'

MODEL_WEIGHTS_SAVE_PATH

lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer = optimizer,
                    mode = 'min',
                    factor = config['reduce_lr_factor'],
                    patience = config['patience'],
                    min_lr = config['min_lr'],
                    verbose = True)

# %%
def train_model(model, dataloaders, criterion, optimizer, lr_scheduler, \
                num_epochs, device, print_samples=True):

    '''Loss functions to keep track of for Regression:
        - R square, Adjusted R square
        - Mean Squared Error (MSE) / Root Mean Squared Error (RMSE)
        - Mean Absolute Error (MAE) '''

    for epoch in range(1, num_epochs+1):
        start_time = time.time()
        best_loss = float('inf')
        best_loss_epoch = None

        for phase in ['Train', 'Val']:
            print("[{}] Epoch: {}/{}".format(phase, epoch, num_epochs))

            if phase == 'Train':
                model.train()
            elif phase == 'Val':
                model.eval()

            running_loss = 0.0
            total_no_data = 0

            for batch_index, (images, metadata, pawpularities) in tqdm(enumerate(dataloaders[phase])):
                bs = images.shape[0]
                images = images.to(device)
                metadata = metadata.to(device)
                pawpularities = pawpularities.to(device)

                with torch.set_grad_enabled(phase=='Train'): # Enable grad only in train
                    pawpulartiy_pred = model(images, metadata)
                    pawpulartiy_pred = torch.squeeze(pawpulartiy_pred) # See if this is necessary

                    loss = criterion(pawpulartiy_pred, pawpularities)
                    running_loss += loss.item() * bs # Will divide later to get an accurate avg
                    total_no_data += bs

                    if phase=='Train':
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()

                    if phase=='Train' and lr_scheduler is not None:
                        lr_scheduler.step()

                    if print_samples and batch_index == 0:
                        print('='*90)
                        print('Image Shape: {}'.format(images.shape))
                        print('Metadata: {}'.format(metadata))
                        print('Pawpularities: {}'.format(pawpularities))
                        print('='*90)

                print(['[Iter. {} of {}] loss: {:.2f}'.format(batch_index, len(dataloaders[phase]), loss)])
            
            running_loss = running_loss / total_no_data
            wandb.log({'MSELoss_{}'.format(phase): running_loss})

            if phase == 'Val' and running_loss < best_loss:
                best_loss = running_loss
                best_loss_epoch = epoch
                Path('./model_save').mkdir(parents=True, exist_ok=True) # Create dir if nonexistent
                model_weights_name = model_weights_name_base + '{}.pth'.format(str(epoch).zfill(3))
                print('Saving Model to : {}'.format(model_weights_name))

            print('Total Training Time So Far: {:.2f} mins'.format((time.time()-start_time)/60))
    
    print('Training Complete. Total Training Time: {:.2f} mins'.format((time.time()-start_time)/60))
    print('Best Loss: {:.5f}'.format(best_loss))
    print('Best Loss Epoch: {}'.format(best_loss_epoch))

# %%
if __name__=='__main__':
    train_model(model, dataloaders, criterion, optimizer, 
                lr_scheduler, config['epochs'], DEVICE, print_samples=True)
