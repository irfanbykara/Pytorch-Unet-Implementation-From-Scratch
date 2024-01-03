from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from consts import *

class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(BasicConv, self).__init__()

        self.out_channel = out_channel
        self.conv = nn.Conv2d(in_channel, out_channel, stride=1, padding=0, kernel_size=3)
        self.conv2 = nn.Conv2d(out_channel, out_channel, stride=1, padding=0, kernel_size=3)
        self.relu = nn.ReLU()

        # Move the module to CUDA device
        self.to(device="cuda:0")

    def forward(self, x, calculate_residual=False):
        residual = None

        x = self.conv(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)

        if calculate_residual != True:
            return x, residual

        residual_shape = RESIDUAL_SHAPES
        target_size = residual_shape[int(self.out_channel * 2)]
        target_size_tuple = (target_size, target_size)

        crop_height = x.size(2) - target_size_tuple[0]
        crop_width = x.size(3) - target_size_tuple[1]

        residual = x[:, :, crop_height//2:-(crop_height - crop_height//2), crop_width//2:-(crop_width - crop_width//2)]

        return x, residual


class Pooling(nn.Module):
    def __init__(self):
        super(Pooling, self).__init__()

        self.pool = nn.MaxPool2d(stride=2, kernel_size=2)

        # Move the module to CUDA device
        self.to(device="cuda:0")

    def forward(self, x):
        x = self.pool(x)
        return x


class UpConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(UpConv, self).__init__()

        self.conv = nn.ConvTranspose2d(in_channel, out_channel, stride=2, padding=0, kernel_size=2)

        # Move the module to CUDA device
        self.to(device="cuda:0")

    def forward(self, x, downsample=False):
        x = self.conv(x)
        return x


class Encoder(nn.Module):
    def __init__(self, in_channel, channels):
        super(Encoder, self).__init__()

        self.conv_blocks = []
        for i in range(len(channels)):
            conv = BasicConv(in_channel, channels[i])
            in_channel = channels[i]
            self.conv_blocks.append(conv)

        self.pool = Pooling()

        # Move the module to CUDA device
        self.to(device="cuda:0")

    def forward(self, x):
        residuals = []
        for conv in self.conv_blocks:
            x, residual = conv(x, calculate_residual=True)
            residuals.append(residual)
            x = self.pool(x)

        return x, residuals


class Bottleneck(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Bottleneck, self).__init__()
        self.conv = BasicConv(in_channel, out_channel)

        # Move the module to CUDA device
        self.to(device="cuda:0")

    def forward(self, x):
        output, _ = self.conv(x, calculate_residual=False)
        return output


class Decoder(nn.Module):
    def __init__(self, channels=CHANNELS):
        super(Decoder, self).__init__()

        self.up_conv_blocks = []
        self.basic_conv_blocks = []

        for i in range(len(channels)):
            up_conv = UpConv(channels[i], int(channels[i] / 2))
            self.up_conv_blocks.append(up_conv)
            basic_conv = BasicConv(channels[i], int(channels[i] / 2))
            self.basic_conv_blocks.append(basic_conv)

        # Move the module to CUDA device
        self.to(device="cuda:0")

    def forward(self, x, residuals):
        residuals.reverse()
        for i in range(len(self.up_conv_blocks)):
            x = self.up_conv_blocks[i](x)
            x = torch.cat((x, residuals[i]), dim=1)
            x, _ = self.basic_conv_blocks[i](x)

        return x
