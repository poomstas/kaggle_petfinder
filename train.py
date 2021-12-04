# %%
import os
import sys
import wandb
import time
import torch
import torch.nn as nn
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt
from src.model import ImgModel
from src.utils import print_config, preprocess_data, get_writer_name, LogCoshLoss, adjustFigAspect, get_lr_suggestion
from pathlib import Path
from albumentations.pytorch.transforms import ToTensorV2
from src.data import PetDataset
from torch.utils.data import DataLoader
from torch_lr_finder import LRFinder

# %%
TRAIN_CSV_PATH = './data/train.csv'
TEST_CSV_PATH = './data/test.csv'
MODEL_SAVE_PATH = './model_save/'

# %% Hyperparameter configuration
config = {
    'model': {
        'backbone':       'swin',         # Backbone Model
        'freeze_backbone':True,           # Freeze backbone model weights (train only fc layers)
        'unfreeze_at':    3,              # Unfreeze backbone at the beginning of this epoch. Irrelevant if freeze_backbone == False
        'dropout':        0,              # Dropout fraction in the fc layers (no dropout if 0)
        'activation_func':'elu',          # Model activation function ['relu', 'tanh', 'leakyrelu', 'elu']
        'n_hidden_nodes': 20,             # Number of hidden node layers on the img side
    },
    'dataloader': {
        'batch_size':     64,             # Batch Size  11GB VRAM -> 32
        'drop_last':      False,          # Drop last mismatched batch
        'train_shuffle':  True,           # Shuffle training data
        'val_shuffle':    False,          # Shuffle validation data
        'num_workers':    6,              # Number of workers for DataLoader
        'abridge_frac':   1.0,            # Fraction of the original training data to be used for train+val
        'val_frac':       0.1,            # Fraction of the training data (abridged or not) to be used for validation set
        'scale_target':   False,          # Scale Pawpularity from 0-100 to 0-1 (set it at False; the model now scales up the output)
    },
    'learning_rate': {
        'lr':             2.31E-04,       # Initial learning rate
        'find_optimal_lr':True,           # Automatically find and apply the optimal learning rate at the beginning
        'new_lr_at_unfr': True,           # Automatically find and apply new learning rate when unfreezing backbone
        'lr_min':         1e-10,          # Minimum bounds for reducing learning rate
        'lr_patience':    2,              # Patience for learning rate plateau detection
        'lr_reduction':   0.33,           # Learning rate reduction factor in case of plateau detection
    },
    'gpu_index':      0,                  # GPU Index, default at 0
    'loss_func':      'MSE',              # Loss function ['MSE', 'MAE', 'LogCosh']
    'epochs':         30,                 # Total number of epochs to train over
    'save_model':     False,              # Save model on validation metric improvement
    'note':           'elu, hidden20, UnfreezeAt5, RardomResizedCrop, nodropout', # Note to leave on W&B
}

wandb.init(config=config, project='PetFinder', entity='poomstas', mode='online') # mode: disabled or online
config = wandb.config # For the case where I use the W&B sweep feature

# %%
preprocess_data(TRAIN_CSV_PATH, 
                val_frac=config['dataloader']['val_frac'], 
                abridge_frac=config['dataloader']['abridge_frac'], 
                scale_target=config['dataloader']['scale_target'])

# %%
DEVICE = torch.device('cuda:{}'.format(config['gpu_index']) if torch.cuda.is_available() else 'cpu')
print('{}\nDevice: {}\nModel: {}'.format('='*80, DEVICE, config['model']['backbone']))
print_config(config)

if config['model']['backbone'].upper() == 'XCEPTIONIMG':
    model = ImgModel(activation=config['model']['activation_func'],
                     n_hidden_nodes=config['model']['n_hidden_nodes'],
                     freeze_backbone=config['model']['freeze_backbone'],
                     dropout=config['model']['dropout']).to(DEVICE)
    TARGET_SIZE = 299
    NORMAL_MEAN = [0.5, 0.5, 0.5]
    NORMAL_STD = [0.5, 0.5, 0.5]

elif config['model']['backbone'].upper() == 'EFFICIENTNET':
    model = ImgModel(backbone='efficientnet',
                     activation=config['model']['activation_func'],
                     n_hidden_nodes=config['model']['n_hidden_nodes'],
                     freeze_backbone=config['model']['freeze_backbone'],
                     dropout=config['model']['dropout']).to(DEVICE)
    TARGET_SIZE = 512
    NORMAL_MEAN = [0.485, 0.456, 0.406] # Imagenet mean
    NORMAL_STD = [0.229, 0.224, 0.225]  # Imagenet std

elif config['model']['backbone'].upper() == 'SWIN':
    model = ImgModel(backbone='swin',
                     activation=config['model']['activation_func'],
                     n_hidden_nodes=config['model']['n_hidden_nodes'],
                     freeze_backbone=config['model']['freeze_backbone'],
                     dropout=config['model']['dropout']).to(DEVICE)
    TARGET_SIZE = 224
    NORMAL_MEAN = [0.485, 0.456, 0.406] # Imagenet mean
    NORMAL_STD = [0.229, 0.224, 0.225]  # Imagenet std

else:
    print('Specified model does not exist.')
    sys.exit()

optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate']['lr'])

# %%
loss_dict = {
    'MSE': nn.MSELoss(), # Mean squared error (calculated by default)
    'MAE': nn.L1Loss(reduction='mean'), # Mean Absolute Error (calculated by default)
    'LogCosh': LogCoshLoss(), 
}
criterion = loss_dict[config['loss_func']]

# %% Albumentation Augmentations
TRANSFORMS_TRAIN = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomResizedCrop(TARGET_SIZE, TARGET_SIZE, scale=(0.85, 1.0)),
    # A.OneOf([
    #         A.RandomRotate90(p=0.5), 
    #         A.Rotate(p=0.5)],
    #     p=0.5),
    A.ColorJitter (brightness=0.1, contrast=0.1, saturation=0.1),
    # A.ChannelShuffle(p=0.3),
    # A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, always_apply=False, p=0.5)
    A.Normalize(NORMAL_MEAN, NORMAL_STD),
    ToTensorV2()
])

TRANSFORMS_VALTEST = A.Compose([
    A.Normalize(NORMAL_MEAN, NORMAL_STD),
    ToTensorV2()
])

# TODO Figure out how to sync the augmentation information with W&B

# %% No separate testing data used. "Test" is used as validation set.
dataset_train = PetDataset(csv_fullpath='./data/separated_train.csv', 
                           img_folder='./data/train', 
                           transform=TRANSFORMS_TRAIN, 
                           target_size=TARGET_SIZE)

dataset_val   = PetDataset(csv_fullpath='./data/separated_val.csv',
                           img_folder='./data/train',
                           transform=TRANSFORMS_VALTEST, 
                           target_size=TARGET_SIZE)

dataloader_train = DataLoader(dataset = dataset_train,
                              batch_size = config['dataloader']['batch_size'],
                              drop_last = config['dataloader']['drop_last'],
                              shuffle = config['dataloader']['train_shuffle'],
                              num_workers = config['dataloader']['num_workers'])

dataloader_val   = DataLoader(dataset = dataset_val,
                              batch_size = config['dataloader']['batch_size'],
                              drop_last = config['dataloader']['drop_last'],
                              shuffle = config['dataloader']['val_shuffle'],
                              num_workers = config['dataloader']['num_workers'])

dataloaders = {'train': dataloader_train, 'val': dataloader_val}

# %% Setup Logging
case_name = get_writer_name(config)
wandb.run.name = case_name
model_weights_name_base = os.path.join(MODEL_SAVE_PATH, case_name, 'Epoch_')

lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer = optimizer,
                    mode = 'min',
                    factor = config['learning_rate']['lr_reduction'],
                    patience = config['learning_rate']['lr_patience'],
                    min_lr = config['learning_rate']['lr_min'],
                    verbose = True)

# %%
def train_model(model, dataloaders, criterion, optimizer, lr_scheduler, \
                config, device, print_samples=True):
    start_time = time.time()
    best_loss = float('inf')
    best_loss_epoch = None
    num_epochs = config['epochs']
    loss_name = config['loss_func']
    loss_MSE = nn.MSELoss()
    loss_MAE = nn.L1Loss(reduction='mean')

    for epoch in range(1, num_epochs+1):
        for phase in ['train', 'val']:
            if epoch==1 and phase=='train' and config['learning_rate']['find_optimal_lr']:
                print('\n==================================[ LR Finder ]=================================='.format(epoch, num_epochs))
                lr_finder_optim = torch.optim.Adam(model.parameters(), lr=1e-7, weight_decay=1e-2) # Should not have a lr scheduler attached
                lr_finder = LRFinder(model, lr_finder_optim, criterion, device=device)
                lr_finder.range_test(dataloader_train, end_lr=1, num_iter=100)
                optimal_lr = get_lr_suggestion(lr_finder)
                optimizer.param_groups[0]['lr'] = optimal_lr
                print("Updated the config's lr value to: {}".format(optimal_lr))
                lr_finder.reset()

                # Not sure if below is needed, but just in case.
                lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                                    optimizer = optimizer,
                                    mode = 'min',
                                    factor = config['learning_rate']['lr_reduction'],
                                    patience = config['learning_rate']['lr_patience'],
                                    min_lr = config['learning_rate']['lr_min'],
                                    verbose = False)
            if phase=='train':
                print('\n==================================[Epoch {}/{}]=================================='.format(epoch, num_epochs))
            if config['model']['freeze_backbone'] and epoch == config['model']['unfreeze_at'] and phase=='train':
                print('\nUnfreezing backbone model parameters...\n')
                for param in model.backbone_model.parameters():
                    param.requires_grad = True
                if config['learning_rate']['new_lr_at_unfr']:
                    print('Finding a new learning rate...')
                    lr_finder_optim = torch.optim.Adam(model.parameters(), lr=1e-7, weight_decay=1e-2) # Should not have a lr scheduler attached
                    lr_finder = LRFinder(model, lr_finder_optim, criterion, device=device)
                    lr_finder.range_test(dataloader_train, end_lr=1, num_iter=100)
                    optimal_lr = get_lr_suggestion(lr_finder)
                    optimizer.param_groups[0]['lr'] = optimal_lr
                    print("Updated the config's lr value to: {}".format(optimal_lr))
                    lr_finder.reset()

            print('\t[{}]'.format(phase.upper()))

            model.train() if phase=='train' else model.eval()

            running_loss = 0.0
            running_loss_MSE = 0.0
            running_loss_MAE = 0.0
            total_no_data = 0
            pawpularities_collect, pawpularities_pred_collect = [], [] # For plotting

            for batch_index, ((images, metadata), pawpularities) in enumerate(dataloaders[phase]):
                bs = images.shape[0]
                images = images.to(device)
                metadata = metadata.to(device)
                pawpularities = pawpularities.to(device)
                pawpularities_collect.extend(pawpularities.tolist())

                with torch.set_grad_enabled(phase=='train'): # Enable grad only in train
                    pawpularities_pred = model((images, metadata))
                    pawpularities_pred = torch.squeeze(pawpularities_pred)
                    pawpularities_pred_collect.extend(pawpularities_pred.tolist())

                    loss = criterion(pawpularities_pred, pawpularities)
                    running_loss += loss.item() * bs # Will divide later to get an accurate avg
                    batch_MSE = loss_MSE(pawpularities_pred, pawpularities)
                    running_loss_MSE += batch_MSE.item() * bs
                    batch_MAE = loss_MAE(pawpularities_pred, pawpularities)
                    running_loss_MAE += batch_MAE.item() * bs
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

                print('\t\t[Iter. {} of {}] {} Loss: {:.4f}\t MSE: {:.4f}\t MAE: {:.4f}'.format(
                    batch_index, len(dataloaders[phase]), loss_name, running_loss/total_no_data, 
                    running_loss_MSE/total_no_data, running_loss_MAE/total_no_data),
                    end='\r')

            running_loss = running_loss / total_no_data
            default_MSE = running_loss_MSE / total_no_data
            default_MAE = running_loss_MAE / total_no_data

            if phase=='val' and lr_scheduler is not None:
                lr_scheduler.step(metrics=running_loss)

            wandb.log({
                'Loss_MSE_{}'.format(phase): default_MSE, # Calculated for every case
                'Loss_MAE_{}'.format(phase): default_MAE, # Calculated for every case
                'Loss_{}_{}'.format(loss_name, phase): running_loss,
                'lr': optimizer.param_groups[0]['lr'],
            })

            Path(os.path.join(MODEL_SAVE_PATH, case_name)).mkdir(parents=True, exist_ok=True) # Create dir if nonexistent
            if phase == 'val' and running_loss < best_loss:
                best_loss = running_loss
                best_loss_epoch = epoch
                if config['save_model']:
                    model_weights_name = model_weights_name_base + '{}.pth'.format(str(epoch).zfill(3))
                    print('\n\nSaving Model to : {}'.format(model_weights_name))
                    torch.save(model.state_dict(), model_weights_name)
                else:
                    print('\n\nBest Loss Observed: {:.4f}'.format(best_loss))

            fig = plt.figure(); adjustFigAspect(fig, aspect=1)
            ax = fig.add_subplot(111)
            ax.scatter(pawpularities_collect, pawpularities_pred_collect)
            ax.set_xlabel('Pawpularity'); ax.set_ylabel('Pawpularity Pred.'); plt.title('Epoch {} {}'.format(epoch, phase))
            ax_maxval = 1 if config['dataloader']['scale_target'] else 100
            ax.plot(np.linspace(0,ax_maxval,100), np.linspace(0,ax_maxval,100), 'r--')
            plt.savefig(os.path.join(MODEL_SAVE_PATH, case_name, 'Epoch_{}_{}.png'.format(str(epoch).zfill(3), phase)))
            plt.close()

            print('\n\t\tTotal Training Time So Far: {:.2f} mins'.format((time.time()-start_time)/60))
    
    print('='*90)
    print('Training Complete. Total Training Time: {:.2f} mins'.format((time.time()-start_time)/60))
    print('Best Loss: {:.5f}'.format(best_loss))
    print('Best Loss Epoch: {}'.format(best_loss_epoch))

# %%
if __name__=='__main__':
    train_model(model, dataloaders, criterion, optimizer, lr_scheduler, 
                config, DEVICE, print_samples=False)
