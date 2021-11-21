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
    def __init__(self, csv_fullpath, trainvaltest='train', transform=None, target_size=299):
        self.df = pd.read_csv(csv_fullpath)
        self.trainvaltest = trainvaltest
        self.transform = transform
        self.target_size = target_size
        print(self.df)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        img_filename = self.df['Id'][index] + '.jpg'
        img_folder_name = 'test' if self.trainvaltest=='test' else 'train'
        img_fullpath = os.path.join('../data/', img_folder_name, img_filename)

        img = mpimg.imread(img_fullpath) # -> np.ndarray, RGB, uint8

        if (self.target_size, self.target_size) != img.shape[:2]:
            img = cv2.resize(img, (self.target_size, self.target_size))

        if self.transform is not None:
            img = self.transform(image=img)["image"]
        
        metadata = self.df.loc[index].iloc[1:-2].values.astype(np.float32) # -> np.array
        pawpularity = self.df.loc[index]['Pawpularity'] # -> np.int64

        return img, metadata, pawpularity

# %%
if __name__=='__main__':
    NORMAL_MEAN = [0.5, 0.5, 0.5]
    NORMAL_STD = [0.5, 0.5, 0.5]
    
    # transforms = transforms.Compose([
    #     transforms.Normalize(NORMAL_MEAN, NORMAL_STD),
    #     transforms.ToTensor()
    # ])

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
        # A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, always_apply=False, p=0.5)
        ToTensorV2()
    ])

    ds_train = PetDataset('../data/train.csv', trainvaltest='train', transform=transforms)
    dataloader = DataLoader(dataset=ds_train,
                            shuffle=False,
                            batch_size=2,
                            num_workers=1)
    temp = iter(dataloader)
    # temp.next()