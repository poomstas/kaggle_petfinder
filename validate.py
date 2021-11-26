# %%
import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
from tqdm import tqdm
from src.model import Xception, DenseNet121, XceptionImg # Add more as I go here
from src.data import PetDataset
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import DataLoader

# %%
test_config = {
    'gpu_index':      0,              # GPU Index, default at 0
    'model':          'XCEPTIONIMG',  # Backbone Model
    'batch_size':     16,             # Batch Size
    'num_workers':    1,              # Number of workers for DataLoader
    'scale_target':   True,           # Scale Pawpularity from 0-100 to 0-1     TODO Use this config in this script
    'model_file_path': './model_save/PetFindr_xceptionimg_LR_0.001_BS_32_nEpoch_30_20211125_235402/Epoch_004.pth'
}

# %%
DEVICE = torch.device('cuda:{}'.format(test_config['gpu_index']) if torch.cuda.is_available() else 'cpu')
print('{}\nDevice: {}\nModel: {}'.format('='*80, DEVICE, test_config['model']))

if test_config['model'].upper() == 'XCEPTION':
    model = Xception().to(DEVICE)
    TARGET_SIZE = 299
    NORMAL_MEAN = [0.5, 0.5, 0.5]
    NORMAL_STD = [0.5, 0.5, 0.5]

elif test_config['model'].upper() == 'XCEPTIONIMG':
    model = XceptionImg().to(DEVICE)
    TARGET_SIZE = 299
    NORMAL_MEAN = [0.5, 0.5, 0.5]
    NORMAL_STD = [0.5, 0.5, 0.5]

elif test_config['model'].upper() == 'DENSENET121':
    model = DenseNet121().to(DEVICE)
    TARGET_SIZE = 224
    NORMAL_MEAN = [0.485, 0.456, 0.406]
    NORMAL_STD = [0.229, 0.224, 0.225]

else:
    print('Specified model does not exist.')
    sys.exit()

# %%
model.load_state_dict(torch.load(test_config['model_file_path']))

# %%
TRANSFORMS_VALTEST = A.Compose([
    A.Normalize(NORMAL_MEAN, NORMAL_STD),
    ToTensorV2()
])

dataset_val  = PetDataset(csv_fullpath='./data/separated_val.csv',
                           img_folder='./data/train',
                           transform=TRANSFORMS_VALTEST, 
                           target_size=TARGET_SIZE,
                           testset=False)

dataloader_val  = DataLoader(dataset = dataset_val,
                              batch_size = test_config['batch_size'],
                              drop_last = False,
                              shuffle = False,
                              num_workers = test_config['num_workers'])

# %%
pawpularities_collect, pawpularities_pred_collect = [], []

for batch_index, (images, metadata, pawpularities) in tqdm(enumerate(dataloader_val), total=len(dataloader_val)):

    pawpularities_collect.extend(pawpularities.tolist())
    pred = model(images.to(DEVICE))
    pawpularities_pred_collect.extend(pred.detach().to('cpu').numpy().tolist())

pawpularities_pred_collect = np.squeeze(pawpularities_pred_collect)

# %%
plt.scatter(pawpularities_collect, pawpularities_pred_collect)
plt.xlabel('Pawpularity'); plt.ylabel('Pawpularity Pred.')
plt.show()

criterion = nn.MSELoss()
loss_MSE = criterion(torch.tensor(pawpularities_pred_collect), torch.tensor(pawpularities_collect))
print('MSE Loss: {:.3f}'.format(loss_MSE))