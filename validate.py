# %%
''' validate.py writing in progress... '''
import torch
from utils import print_config
from src.model import Xception, DenseNet121 # Add more as I go here
import albumentations as A
from src.data import PetDataset
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import DataLoader

# %%
config = {
  'gpu_index': 0,
  'model': 'Xception',
  'batch_size': 32,
  'drop_last': False,
  'train_shuffle': True,
  'val_shuffle': False,
  'num_workers': 1,
  'learning_rate': 0.001,
  'min_lr': 1e-10,
  'patience': 5,
  'lr_reduction': 0.1,
  'epochs': 30,
  'gpu_index': 0,
  'TB_note': ''
  # Add structural design components here, e.g. Number of hidden layers for diff branches
}

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

TRANSFORMS_VALTEST = A.Compose([
    A.Normalize(NORMAL_MEAN, NORMAL_STD),
    ToTensorV2()
])

dataset_test  = PetDataset(csv_fullpath='./data/test.csv',
                           img_folder='./data/test',
                           transform=TRANSFORMS_VALTEST, 
                           target_size=TARGET_SIZE,
                           testset=True)

dataloader_test  = DataLoader(dataset = dataset_test,
                              batch_size = 1,
                              drop_last = False,
                              shuffle = False,
                              num_workers = 1)
