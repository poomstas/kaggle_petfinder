import cv2
import numpy as np
import torch

from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor, to_pil_image

class PetDataset(Dataset):
    def __init__(self, image_dir, label_list, augments=None, target_size=299):
        self.image_dir = image_dir
        self.label_list = label_list
        self.augments = augments
        self.target_size = target_size

    def __getitem__(self, index):
        image_path = self.image_dir[index]
        label = self.label_list[index]
        augments = self.augments
        target_size = self.target_size

        bgr_image = cv2.imread(image_path)
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

        image = rgb_image.copy()

        if (target_size, target_size) != image.shape[:2]:
            image = cv2.resize(image, (target_size, target_size))

        if augments is not None:
            transformed = augments(image=image)
            image = transformed["image"]

        image = to_tensor(image)
        label = torch.tensor(np.asarray(label, np.float32))

        return image, label, image_path

    def __len__(self):
        return len(self.image_dir)

def convert_data(image, label):
    image = to_pil_image(image).convert("RGB")
    label = label.detach().cpu().numpy()

    return image, label
