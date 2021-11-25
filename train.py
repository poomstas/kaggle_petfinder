# %%
import os
import sys
import wandb
import time
import argparse
import torch
import torch.nn as nn
import albumentations as A
from src.model import Xception, XceptionImg, DenseNet121 # Add more here later
from src.utils import print_config, separate_train_val, get_writer_name, parse_arguments, get_dict_from_args, create_folder
from torch.utils.data import DataLoader
from pathlib import Path

# %%
TRAIN_CSV_PATH = './data/train.csv'
TEST_CSV_PATH = './data/test.csv'
VAL_FRAC = 0.1

MODEL_SAVE_PATH = './model_save/'

separate_train_val(TRAIN_CSV_PATH, val_frac=VAL_FRAC, random_state=12345, abridge_frac=0.2)

# %% For the case where we retrieve the hyperparameter values from CLI
parser = argparse.ArgumentParser(description='Parse hyperparameter arguments from CLI')
args = parse_arguments(parser) # Reference values like so: args.alpha 
config = get_dict_from_args(args)

wandb.init(project='PetFinder', entity='poomstas', mode='disabled')
wandb.config = config # value reference as: wandb.config.epochs

# %% For the case where we retrieve the hyperparameter values from W&B

# %%
DEVICE = torch.device('cuda:{}'.format(config['gpu_index']) if torch.cuda.is_available() else 'cpu')
print('{}\nDevice: {}\nModel: {}'.format('='*80, DEVICE, config['model']))
print_config(config)

if config['model'].upper() == 'XCEPTION':
    model = Xception().to(DEVICE)
    TARGET_SIZE = 299
    NORMAL_MEAN = [0.5, 0.5, 0.5]
    NORMAL_STD = [0.5, 0.5, 0.5]

elif config['model'].upper() == 'XCEPTIONIMG':
    model = XceptionImg().to(DEVICE)
    TARGET_SIZE = 299
    NORMAL_MEAN = [0.5, 0.5, 0.5]
    NORMAL_STD = [0.5, 0.5, 0.5]

elif config['model'].upper() == 'DENSENET121':
    model = DenseNet121().to(DEVICE)
    TARGET_SIZE = 224
    NORMAL_MEAN = [0.485, 0.456, 0.406]
    NORMAL_STD = [0.229, 0.224, 0.225]

else:
    print('Specified model does not exist.')
    sys.exit()

optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
criterion = nn.MSELoss()
# criterion = ignite.metrics.MeanAbsoluteError
# criterion = nn.L1Loss() # output = loss(input, target)

# %% Albumentation Augmentations
from albumentations.pytorch.transforms import ToTensorV2

TRANSFORMS_TRAIN = A.Compose([
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
    ToTensorV2()
])

TRANSFORMS_VALTEST = A.Compose([
    A.Normalize(NORMAL_MEAN, NORMAL_STD),
    ToTensorV2()
])

# %% No separate testing data used. "Test" is used as validation set.
from src.data import PetDataset
from torch.utils.data import DataLoader

dataset_train = PetDataset(csv_fullpath='./data/separated_train.csv', 
                           img_folder='./data/train', 
                           transform=TRANSFORMS_TRAIN, 
                           target_size=TARGET_SIZE)

dataset_val   = PetDataset(csv_fullpath='./data/separated_val.csv',
                           img_folder='./data/train',
                           transform=TRANSFORMS_VALTEST, 
                           target_size=TARGET_SIZE)

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

dataloaders = {'train': dataloader_train, 'val': dataloader_val}

# %% Setup Logging
case_name = get_writer_name(config)
wandb.run.name = case_name
create_folder(os.path.join(MODEL_SAVE_PATH, case_name))
model_weights_name_base = os.path.join(MODEL_SAVE_PATH, case_name, 'Epoch_')

lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer = optimizer,
                    mode = 'min',
                    factor = config['lr_reduction'],
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

    start_time = time.time()
    best_loss = float('inf')
    best_loss_epoch = None

    for epoch in range(1, num_epochs+1):
        for phase in ['train', 'val']:
            if phase=='train':
                print('\n==================================[Epoch {}/{}]=================================='.format(epoch, num_epochs))
            print('\t[{}]'.format(phase.upper()))

            model.train() if phase=='train' else model.eval()

            running_loss = 0.0
            total_no_data = 0

            for batch_index, (images, metadata, pawpularities) in enumerate(dataloaders[phase]):
                bs = images.shape[0]
                images = images.to(device)
                metadata = metadata.to(device)
                pawpularities = pawpularities.to(device)

                with torch.set_grad_enabled(phase=='train'): # Enable grad only in train
                    pawpulartiy_pred = model(images) # Try with only one input for now
                    # pawpulartiy_pred = model(images, metadata)
                    pawpulartiy_pred = torch.squeeze(pawpulartiy_pred) # See if this is necessary

                    loss = criterion(pawpulartiy_pred, pawpularities)
                    running_loss += loss.item() * bs # Will divide later to get an accurate avg
                    total_no_data += bs

                    if phase=='train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    if print_samples and batch_index == 0:
                        print('='*90)
                        print('Image Shape: {}'.format(images.shape))
                        print('Metadata: {}'.format(metadata))
                        print('Pawpularities: {}'.format(pawpularities))
                        print('='*90)

                print('\t\t[Iter. {} of {}] Loss: {:.2f}'.format(batch_index, len(dataloaders[phase]), running_loss/total_no_data), end='\r')

            if phase=='val' and lr_scheduler is not None:
                lr_scheduler.step(metrics=loss)
            
            running_loss = running_loss / total_no_data
            wandb.log({'MSELoss_{}'.format(phase): running_loss})

            if phase == 'val' and running_loss < best_loss:
                best_loss = running_loss
                best_loss_epoch = epoch
                Path('./model_save').mkdir(parents=True, exist_ok=True) # Create dir if nonexistent
                model_weights_name = model_weights_name_base + '{}.pth'.format(str(epoch).zfill(3))
                print('\n\nSaving Model to : {}'.format(model_weights_name))
                torch.save(model.state_dict(), model_weights_name)

            print('\n\t\tTotal Training Time So Far: {:.2f} mins'.format((time.time()-start_time)/60))
    
    print('='*90)
    print('Training Complete. Total Training Time: {:.2f} mins'.format((time.time()-start_time)/60))
    print('Best Loss: {:.5f}'.format(best_loss))
    print('Best Loss Epoch: {}'.format(best_loss_epoch))

# %%
if __name__=='__main__':
    train_model(model, dataloaders, criterion, optimizer, lr_scheduler, 
                config['epochs'], DEVICE, print_samples=False)
