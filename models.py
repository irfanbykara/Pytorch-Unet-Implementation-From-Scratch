
import torch
import torch.nn as nn
from torch.nn import Module
from utils import *
from modules import *
from consts import *
import torch.nn.functional as F
from torchvision import models
class UNet(nn.Module):
  def __init__(self,in_channels=3):
    super(UNet,self).__init__()
    self.in_channels = in_channels

    architecture_shape = RESIDUAL_SHAPES
    channels_for_encoder = architecture_shape.keys()
    # Use map() to divide all elements in the list by the divisor
    channels_for_encoder = list(map(lambda x: divide_by_divisor(x, 2), channels_for_encoder))

    # print(channels_for_encoder)

    self.encoder = Encoder(self.in_channels,channels_for_encoder)
    # print(channels_for_encoder)

    self.bottleneck = Bottleneck(512,1024)
    # print(channels_for_encoder)


    channels_for_decoder = list(architecture_shape.keys())
    channels_for_decoder.reverse()
    # print(channels_for_decoder)


    self.decoder = Decoder(channels_for_decoder)

    self.final = nn.Conv2d(in_channels=64,out_channels=3,stride=1,kernel_size=1)
    self.to(device="cuda:0")

  def forward(self,x):

    x,residuals = self.encoder(x)
    x = self.bottleneck(x)
    x = self.decoder(x,residuals)
    x = self.final(x)

    return x


# Define your custom U-Net model
class UnetReady(nn.Module):
    def __init__(self, num_classes):
        super(UnetReady, self).__init__()
        # Load a pretrained U-Net model from torchvision
        self.base_model = models.segmentation.fcn_resnet50(pretrained=True, progress=True)
        # Modify the final layer to match the number of classes in your dataset
        self.base_model.classifier[-1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        return self.base_model(x)['out']

