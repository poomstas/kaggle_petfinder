# %%
import timm
import torch
import torch.nn.functional as F
import torch.nn as nn
from pretrainedmodels import xception, densenet121, densenet201
from torchvision.models import *

# %%
class ImgModel(nn.Module):
    ''' This class of models takes in only the image as input (no metadata taken as input) '''
    def __init__(self, backbone='xception', pretrained=True, activation='relu', n_hidden_nodes=10, 
                 freeze_backbone=False, dropout=None):
        super(ImgModel, self).__init__()
        self.dropout = dropout

        if backbone=='xception':
            self.backbone_model = xception(num_classes=1000, pretrained='imagenet' if pretrained else False)
            self.backbone_model.last_linear = nn.Identity() # Outputs 2048
            n_backbone_out = 2048
        elif backbone=='efficientnet':
            self.backbone_model = timm.create_model('tf_efficientnet_b0_ns', pretrained=pretrained)
            self.backbone_model.classifier = nn.Identity() # Outputs 1280
            n_backbone_out = 1280
        elif backbone=='swin_A':
            self.backbone_model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=0, in_chans=3)
            n_backbone_out = 768 # Outputs 768
        elif backbone=='swin_B':
            self.backbone_model = timm.create_model('swin_large_patch4_window12_384', pretrained=True, num_classes=0, in_chans=3)
            n_backbone_out = 1536 # Outputs 1536

        if freeze_backbone:
            for param in self.backbone_model.parameters():
                param.requires_grad = False

        self.img_fc1 = nn.Linear(n_backbone_out, n_hidden_nodes)
        self.fc1 = nn.Linear(n_hidden_nodes, 1)

        if dropout:
            self.dropout_layer = nn.Dropout(dropout)
        
        activation_functions = {
            'tanh': torch.tanh,
            'relu': F.relu,
            'leakyrelu': nn.LeakyReLU(negative_slope=0.01, inplace=False), # arg set at default
            'elu': nn.ELU(alpha=0.1, inplace=False), # arg set at default
        }
        self.activation = activation_functions[activation]


    def forward(self, input_tuple):
        img, _ = input_tuple

        out = self.activation(self.backbone_model(img))
        out = self.dropout_layer(out) if self.dropout else out
        out = self.activation(self.img_fc1(out))
        out = self.dropout_layer(out) if self.dropout else out
        out = self.fc1(out)
        out = out * 100 # Scaling for easier convergence

        return torch.squeeze(out)

# %%
# class Xception(nn.Module):
#     def __init__(self, pretrained=True, activation='relu'):
#         super(Xception, self).__init__()
#         self.model = xception(num_classes=1000, pretrained='imagenet' if pretrained else False)
#         self.model.last_linear = nn.Identity() # Outputs 2048

#         self.img_fc1 = nn.Linear(2048, 1024) # The image branch starts here
#         self.img_fc2 = nn.Linear(1024, 512)
#         self.img_fc3 = nn.Linear(512, 256)

#         self.fc1 = nn.Linear(268, 128) # Takes the concatenated vector (img features + metadata)
#         self.fc2 = nn.Linear(128, 64)
#         self.fc3 = nn.Linear(64, 1)

#         activation_functions = {
#             'tanh': torch.tanh,
#             'relu': F.relu,
#         }
#         self.activation = activation_functions[activation]

#     def forward(self, img, metadata):
#         out_img = self.activation(self.model(img))
#         out_img = self.activation(self.img_fc1(out_img))
#         out_img = self.activation(self.img_fc2(out_img)) # Outs 512
#         out_img = self.activation(self.img_fc3(out_img)) # Outs 256

#         out_img = self.activation(self.model(img))
#         out_img = self.activation(self.img_fc1(out_img))
#         out_img = self.activation(self.img_fc2(out_img)) # Outs 50
#         out_img = self.activation(self.img_fc3(out_img)) # Outs 256

#         out = torch.cat((metadata, out_img), 1) # Outs 50 + 12 = 62
#         out = self.activation(self.fc1(out))
#         out = self.activation(self.fc2(out))
#         out = self.fc3(out)

#         return out_img

# %%
if __name__=='__main__':
    model = ImgModel()
