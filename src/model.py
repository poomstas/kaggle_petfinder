# %%
import torch
import torch.nn.functional as F
import torch.nn as nn

from pretrainedmodels import xception, densenet121, densenet201
from torchvision.models import *

# %%
class XceptionImg(nn.Module):
    ''' "Img" at the end indicates an image-only model (doesn't take in metadata).
        For testing purpose.  '''
    def __init__(self, pretrained=True):
        super(XceptionImg, self).__init__()
        self.model = xception(num_classes=1000, pretrained='imagenet' if pretrained else False)
        self.model.last_linear = nn.Linear(self.model.last_linear.in_features, 2048)

        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 128)
        self.fcOut = nn.Linear(128, 1)
        # Use dropout layers somewhere?

    def forward(self, img, metadata): # Take metadata here for consistency but don't use it here.
        out = F.relu(self.model(img))
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = self.fcOut(out)
        return out

# %%
class Xception(nn.Module):
    def __init__(self, pretrained=True, n_out=1):
        super(Xception, self).__init__()

        self.model = xception(num_classes=1000, pretrained='imagenet' if pretrained else False)
        self.model.last_linear = nn.Linear(self.model.last_linear.in_features, n_out)

    def forward(self, img, metadata):
        return torch.sigmoid(self.model(img))

# %%
class DenseNet121(nn.Module):
    def __init__(self, pretrained=True):
        super(DenseNet121, self).__init__()
        self.model = densenet121(num_classes=1000, pretrained='imagenet' if pretrained else False)

    def forward(self):
        return
