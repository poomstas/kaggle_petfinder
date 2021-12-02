# %%
import torch
import torch.nn.functional as F
import torch.nn as nn
from pretrainedmodels import xception, densenet121, densenet201
from torchvision.models import *

# %%
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x

# %%
class XceptionImg(nn.Module):
    def __init__(self, pretrained=True):
        super(XceptionImg, self).__init__()
        self.model = xception(num_classes=1000, pretrained='imagenet' if pretrained else False)
        self.model.last_linear = Identity() # Outputs 2048

        self.img_fc1 = nn.Linear(2048, 10) # The image branch starts here
        self.fc1 = nn.Linear(10, 1) # Takes the concatenated vector (img features + metadata)

    def forward(self, input_tuple):
        img, _ = input_tuple

        out = F.relu(self.model(img))
        out = F.relu(self.img_fc1(out))
        out = self.fc1(out)

        return out

# %%
class Xception(nn.Module):
    def __init__(self, pretrained=True):
        super(Xception, self).__init__()
        self.model = xception(num_classes=1000, pretrained='imagenet' if pretrained else False)
        self.model.last_linear = Identity() # Outputs 2048

        self.img_fc1 = nn.Linear(2048, 1024) # The image branch starts here
        self.img_fc2 = nn.Linear(1024, 512)
        self.img_fc3 = nn.Linear(512, 256)

        self.fc1 = nn.Linear(268, 128) # Takes the concatenated vector (img features + metadata)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, img, metadata):
        out_img = torch.tanh(self.model(img))
        out_img = torch.tanh(self.img_fc1(out_img))
        out_img = torch.tanh(self.img_fc2(out_img)) # Outs 512
        out_img = torch.tanh(self.img_fc3(out_img)) # Outs 256

        out_img = torch.tanh(self.model(img))
        out_img = torch.tanh(self.img_fc1(out_img))
        out_img = torch.tanh(self.img_fc2(out_img)) # Outs 50
        out_img = torch.tanh(self.img_fc3(out_img)) # Outs 256

        out = torch.cat((metadata, out_img), 1) # Outs 50 + 12 = 62
        out = torch.tanh(self.fc1(out))
        out = torch.tanh(self.fc2(out))
        out = self.fc3(out)

        return out_img

# %%
if __name__=='__main__':
    model = XceptionImg()
