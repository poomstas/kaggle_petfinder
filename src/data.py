import os
import cv2
import torch
import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor, to_pil_image

class PetDataset(Dataset):
    def __init__(self, type_trainval='train', augments=None, target_size=299):
        super(PetDataset, self).__init__()

        assert type_trainval in ['train', 'val'], "Specify either 'train' or 'val' in type_trainval" 
        if type_trainval == 'train':
            csv_file_path = './data/train.csv'
            folder_name = 'train'
        elif type_trainval == 'val':
            csv_file_path = './data/test.csv'
            folder_name = 'test'
        
        self.img_path = os.path.join('./data', folder_name)
        self.df = pd.read_csv(csv_file_path)
        self.df['filename'] = [Id + '.jpg' for Id in self.df['Id']] # Add filename column
        self.augments = augments
        self.target_size = target_size
        
    def __getitem__(self, index):
        img_filename = self.df.loc[index]['filename']
        img_fullpath = os.path.join(self.img_path, img_filename)
        img = cv2.imread(img_fullpath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Read img and convert to RGB

        if (self.target_size, self.target_size) != img.shape[:2]:
            img = cv2.resize(img, (self.target_size, self.target_size))

        metadata = self.df.loc[index].iloc[1:-2].values.astype(np.float32) # -> np.array
        metadata = torch.tensor(metadata)
        pawpularity = self.df.loc[index]['Pawpularity'] # -> np.int64

        if self.augments is not None:
            transformed = self.augments(image=img)
            img = transformed['image']
        img = to_tensor(img)

        return img, metadata, pawpularity

    def __len__(self):
        return len(self.df)

def convert_data(image, label):
    image = to_pil_image(image).convert("RGB")
    label = label.detach().cpu().numpy()

    return image, label
