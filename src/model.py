# %%
import torch
import torch.nn.functional as F
import torch.nn as nn

from pretrainedmodels import xception, densenet121, densenet201
from torchvision.models import *

# %%
class Xception(nn.Module):
    def __init__(self, pretrained=True, n_out=1):
        super(Xception, self).__init__()

        self.model = xception(num_classes=1000, pretrained='imagenet' if pretrained else False)
        self.model.last_linear = torch.nn.Linear(self.model.last_linear.in_features, n_out)

    def forward(self, img, label):
        return torch.sigmoid(self.model(img))

# %%
class DenseNet121(nn.Module):
    def __init__(self, pretrained=True):
        super(DenseNet121, self).__init__()

    def forward(self):
        return
