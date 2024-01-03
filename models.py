
import torch
import torch.nn as nn
from torch.nn import Module
from utils import *
from modules import *
from consts import *
import torch.nn.functional as F

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

