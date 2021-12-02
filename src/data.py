# %%
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.image as mpimg
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

# %%
class PetDataset(Dataset):
    def __init__(self, csv_fullpath, img_folder, transform=None, target_size=299, testset=False):
        self.df                         = pd.read_csv(csv_fullpath)
        self.img_folder                 = img_folder
        self.transform                  = transform
        self.target_size                = target_size
        self.testset                    = testset
        self.metadata_col_index_start   = self.df.columns.get_loc('Subject Focus')
        self.metadata_col_index_end     = self.df.columns.get_loc('Blur')
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        img_filename = self.df['Id'][index] + '.jpg'
        img_fullpath = os.path.join(self.img_folder, img_filename)

        img = mpimg.imread(img_fullpath)

        if (self.target_size, self.target_size) != img.shape[:2]:
            img = cv2.resize(img, (self.target_size, self.target_size))

        if self.transform is not None:
            img = self.transform(image=img)["image"]
        
        start_indx = self.metadata_col_index_start
        end_indx   = self.metadata_col_index_end
        metadata   = self.df.loc[index].iloc[start_indx:end_indx+1].values.astype(np.int)

        if self.testset:
            return (img, metadata)
        else:
            pawpularity = self.df.loc[index]['Pawpularity'].astype(np.float32) # -> np.float32
            return (img, metadata), pawpularity

# %%
if __name__=='__main__':
    ''' Test to see if the Dataset and DataLoader objects are working correctly. '''
    NORMAL_MEAN = [0.5, 0.5, 0.5]
    NORMAL_STD = [0.5, 0.5, 0.5]
    
    transforms = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.OneOf([
                A.RandomRotate90(p=0.5), 
                A.Rotate(p=0.5)],
            p=0.5),
        A.ColorJitter (brightness=0.2, contrast=0.2, p=0.3),
        A.ChannelShuffle(p=0.3),
        A.Normalize(NORMAL_MEAN, NORMAL_STD),
        ToTensorV2()
    ])

    ds_train = PetDataset('../data/separated_train.csv', img_folder='../data/train', transform=transforms)
    dl_train = DataLoader(dataset=ds_train,
                          shuffle=False,
                          batch_size=4,
                          num_workers=2)

    for images, metadata, pawpularities in dl_train:
        print(images.shape)
        print(metadata)
        print(pawpularities)