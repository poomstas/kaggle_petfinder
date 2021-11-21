import os
import cv2
import torch
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class PetDataset(Dataset):
    def __init__(self, csv_file_path, type_trainvaltest='train', transforms=None, target_size=299):
        super(PetDataset, self).__init__()
        
        folder_name = 'test' if type_trainvaltest else 'train'

        self.img_path = os.path.join('./data', folder_name)
        self.df = pd.read_csv(csv_file_path)
        self.df['filename'] = [Id + '.jpg' for Id in self.df['Id']] # Add filename column
        self.transforms = transforms
        self.target_size = target_size

        print('PetDataset\t{}\t{}'.format(type_trainvaltest, self.__len__()))
        
    def __getitem__(self, index):
        img_filename = self.df.loc[index]['filename']
        img_fullpath = os.path.join(self.img_path, img_filename)
        img = cv2.imread(img_fullpath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Read img and convert to RGB

        if (self.target_size, self.target_size) != img.shape[:2]:
            img = cv2.resize(img, (self.target_size, self.target_size))

        if self.transforms is not None:
            img = self.transforms(image=img)["image"]

        metadata = self.df.loc[index].iloc[1:-2].values.astype(np.float32) # -> np.array
        metadata = torch.tensor(metadata)
        pawpularity = self.df.loc[index]['Pawpularity'] # -> np.int64

        img = np.array(img)
        metadata = np.array(metadata)
        pawpularity = np.array(pawpularity)

        return img, metadata, pawpularity

    def __len__(self):
        return len(self.df)
