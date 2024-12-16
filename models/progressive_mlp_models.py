import sys
import math
import numpy as np
import itertools
import einops
import torch
import torch.nn as nn
import torch.nn.functional as nnf
import torch.utils.checkpoint as checkpoint
from torch.distributions.normal import Normal
from monai.transforms import Resize
from .mlp_models import JointDecoderBlock

def reconstruction_loss(x, y):
    loss = torch.sum(torch.pow(x - y, 2))
    return loss

class Progressive_MLP(nn.Module):
    def __init__(self, in_channels, out_channels, num_channels=16):
        super().__init__()
        self.in_conv = nn.Conv2d(in_channels, num_channels, kernel_size=3, stride=1, padding='same')

        self.encoder1 = ConvBlock(num_channels, num_channels, kernel_size=3, stride=1, padding='same')
        self.encoder2 = ConvBlock(num_channels, num_channels * 2, kernel_size=3, stride=1, padding='same')
        self.encoder3 = ConvBlock(num_channels * 2, num_channels * 4, kernel_size=3, stride=1, padding='same')
        self.encoder4 = ConvBlock(num_channels * 4, num_channels * 8, kernel_size=3, stride=1, padding='same')
        self.middle = ConvBlock(num_channels * 8, num_channels * 16, kernel_size=3, stride=1, padding='same')

        self.down1 = nn.MaxPool2d(2)
        self.down2 = nn.MaxPool2d(2)
        self.down3 = nn.MaxPool2d(2)
        self.down4 = nn.MaxPool2d(2)

        self.decoder1 = JointDecoderBlock(256, 128, out_channels, kernel_size=3, stride=1, padding='same')
        self.decoder2 = JointDecoderBlock(out_channels, 64, out_channels, kernel_size=3, stride=1, padding='same')
        self.decoder3 = JointDecoderBlock(out_channels, 32, out_channels, kernel_size=3, stride=1, padding='same')
        self.decoder4 = JointDecoderBlock(out_channels, 16, out_channels, kernel_size=3, stride=1, padding='same')

        self.loss = reconstruction_loss

    def forward(self, x, real_x):
        x = self.in_conv(x)
        x_1 = self.encoder1(x)

        x = self.down1(x_1)
        x_2 = self.encoder2(x)
        real_2 = Resize((x_2.shape[1], x_2.shape[2], x_2.shape[3]))(real_x)

        x = self.down2(x_2)
        x_3 = self.encoder3(x)
        real_3 = Resize((x_3.shape[1], x_3.shape[2], x_3.shape[3]))(real_x)

        x = self.down3(x_3)
        x_4 = self.encoder4(x)
        real_4 = Resize((x_4.shape[1], x_4.shape[2], x_4.shape[3]))(real_x)

        x = self.down4(x_4)
        x = self.middle(x)

        x_5 = self.decoder1(x, x_4)
        real_5 = Resize((x_5.shape[1], x_5.shape[2], x_5.shape[3]))(real_x)
        x_6 = self.decoder2(x_5, x_3)
        real_6 = Resize((x_6.shape[1], x_6.shape[2], x_6.shape[3]))(real_x)
        x_7 = self.decoder3(x_6, x_2)
        real_7 = Resize((x_7.shape[1], x_7.shape[2], x_7.shape[3]))(real_x)
        x_8 = self.decoder4(x_7, x_1)

        loss = self.loss(x_8, real_x) + 1 / 2 * self.loss(x_7, real_7) + 1 / 4 * self.loss(x_6, real_6) + 1 / 8 * self.loss(
            x_5, real_5)  + 1 / 16 * self.loss(x_4, real_4) + 1 / 32 * self.loss(x_3, real_3) + 1 / 64 * self.loss(x_2, real_2)
        return loss, x_8

    def inference(self, x):
        x = self.in_conv(x)
        x_1 = self.encoder1(x)

        x = self.down1(x_1)
        x_2 = self.encoder2(x)

        x = self.down2(x_2)
        x_3 = self.encoder3(x)

        x = self.down3(x_3)
        x_4 = self.encoder4(x)


        x = self.down4(x_4)
        x = self.middle(x)

        x_5 = self.decoder1(x, x_4)
        x_6 = self.decoder2(x_5, x_3)
        x_7 = self.decoder3(x_6, x_2)
        x_8 = self.decoder4(x_7, x_1)
        return x_8

class Progressive8_MLP(nn.Module):
    def __init__(self, in_channels, out_channels, num_channels=16):
        super().__init__()
        self.in_conv = nn.Conv2d(in_channels, num_channels, kernel_size=3, stride=1, padding='same')

        self.encoder1 = ConvBlock(num_channels, num_channels, kernel_size=3, stride=1, padding='same')
        self.encoder2 = ConvBlock(num_channels, num_channels*2, kernel_size=3, stride=1, padding='same')
        self.encoder3 = ConvBlock(num_channels*2, num_channels*4, kernel_size=3, stride=1, padding='same')
        self.encoder4 = ConvBlock(num_channels*4, num_channels*8, kernel_size=3, stride=1, padding='same')
        self.middle = ConvBlock(num_channels*8, num_channels*16, kernel_size=3, stride=1, padding='same')

        self.down1 = nn.MaxPool2d(2)
        self.down2 = nn.MaxPool2d(2)
        self.down3 = nn.MaxPool2d(2)
        self.down4 = nn.MaxPool2d(2)

        self.decoder1 = JointDecoderBlock(256, 128, out_channels, kernel_size=3, stride=1, padding='same')
        self.decoder2 = JointDecoderBlock(out_channels, 64, out_channels, kernel_size=3, stride=1, padding='same')
        self.decoder3 = JointDecoderBlock(out_channels, 32, out_channels, kernel_size=3, stride=1, padding='same')
        self.decoder4 = JointDecoderBlock(out_channels, 16, out_channels, kernel_size=3, stride=1, padding='same')

        self.loss = reconstruction_loss

    def forward(self, x, real_x):
        x = self.in_conv(x)
        x_1 = self.encoder1(x)
        real_1 = Resize((x_1.shape[1], x_1.shape[2], x_1.shape[3]))(real_x)

        x = self.down1(x_1)
        x_2 = self.encoder2(x)
        real_2 = Resize((x_2.shape[1], x_2.shape[2], x_2.shape[3]))(real_x)

        x = self.down2(x_2)
        x_3 = self.encoder3(x)
        real_3 = Resize((x_3.shape[1], x_3.shape[2], x_3.shape[3]))(real_x)

        x = self.down3(x_3)
        x_4 = self.encoder4(x)
        real_4 = Resize((x_4.shape[1], x_4.shape[2], x_4.shape[3]))(real_x)

        x = self.down4(x_4)
        x = self.middle(x)

        x_5 = self.decoder1(x, x_4)
        real_5 = Resize((x_5.shape[1], x_5.shape[2], x_5.shape[3]))(real_x)
        x_6 = self.decoder2(x_5, x_3)
        real_6 = Resize((x_6.shape[1], x_6.shape[2], x_6.shape[3]))(real_x)
        x_7 = self.decoder3(x_6, x_2)
        real_7 = Resize((x_7.shape[1], x_7.shape[2], x_7.shape[3]))(real_x)
        x_8 = self.decoder4(x_7, x_1)

        # loss = 0.0001 * self.loss(x_8, real_x) + 0.0004 * self.loss(x_7, real_7) + 0.0016 * self.loss(x_6, real_6) + 0.0064 * self.loss(x_5, real_5) \
        #         + 0.0001 * self.loss(x_1, real_1) + 0.0004 * self.loss(x_2, real_2) + 0.0016 * self.loss(x_3, real_3) + 0.0064 * self.loss(x_4, real_4)
        loss = self.loss(x_8, real_x) + 1 / 2 * self.loss(x_7, real_7) + 1 / 4 * self.loss(x_6, real_6) + 1 / 8 * self.loss(
            x_5, real_5) + 1 / 16 * self.loss(x_4, real_4) + 1 / 32 * self.loss(x_3, real_3) + 1 / 64 * self.loss(x_2, real_2) + 1 / 128 * self.loss(x_1, real_1)
        return loss, x_8

    def inference(self, x):
        x = self.in_conv(x)
        x_1 = self.encoder1(x)

        x = self.down1(x_1)
        x_2 = self.encoder2(x)

        x = self.down2(x_2)
        x_3 = self.encoder3(x)

        x = self.down3(x_3)
        x_4 = self.encoder4(x)

        x = self.down4(x_4)
        x = self.middle(x)

        x_5 = self.decoder1(x, x_4)
        x_6 = self.decoder2(x_5, x_3)
        x_7 = self.decoder3(x_6, x_2)
        x_8 = self.decoder4(x_7, x_1)
        return x_8

class Progressive7_MLP(nn.Module):
    def __init__(self, in_channels, out_channels, num_channels=16):
        super().__init__()
        self.in_conv = nn.Conv2d(in_channels, num_channels, kernel_size=3, stride=1, padding='same')

        self.encoder1 = ConvBlock(num_channels, num_channels, kernel_size=3, stride=1, padding='same')
        self.encoder2 = ConvBlock(num_channels, num_channels*2, kernel_size=3, stride=1, padding='same')
        self.encoder3 = ConvBlock(num_channels*2, num_channels*4, kernel_size=3, stride=1, padding='same')
        self.encoder4 = ConvBlock(num_channels*4, num_channels*8, kernel_size=3, stride=1, padding='same')
        self.middle = ConvBlock(num_channels*8, num_channels*16, kernel_size=3, stride=1, padding='same')

        self.down1 = nn.MaxPool2d(2)
        self.down2 = nn.MaxPool2d(2)
        self.down3 = nn.MaxPool2d(2)
        self.down4 = nn.MaxPool2d(2)

        self.decoder1 = JointDecoderBlock(256, 128, out_channels, kernel_size=3, stride=1, padding='same')
        self.decoder2 = JointDecoderBlock(out_channels, 64, out_channels, kernel_size=3, stride=1, padding='same')
        self.decoder3 = JointDecoderBlock(out_channels, 32, out_channels, kernel_size=3, stride=1, padding='same')
        self.decoder4 = JointDecoderBlock(out_channels, 16, out_channels, kernel_size=3, stride=1, padding='same')

        self.loss = reconstruction_loss

    def forward(self, x, real_x):
        x = self.in_conv(x)
        x_1 = self.encoder1(x)

        x = self.down1(x_1)
        x_2 = self.encoder2(x)
        real_2 = Resize((x_2.shape[1], x_2.shape[2], x_2.shape[3]))(real_x)

        x = self.down2(x_2)
        x_3 = self.encoder3(x)
        real_3 = Resize((x_3.shape[1], x_3.shape[2], x_3.shape[3]))(real_x)

        x = self.down3(x_3)
        x_4 = self.encoder4(x)
        real_4 = Resize((x_4.shape[1], x_4.shape[2], x_4.shape[3]))(real_x)

        x = self.down4(x_4)
        x = self.middle(x)

        x_5 = self.decoder1(x, x_4)
        real_5 = Resize((x_5.shape[1], x_5.shape[2], x_5.shape[3]))(real_x)
        x_6 = self.decoder2(x_5, x_3)
        real_6 = Resize((x_6.shape[1], x_6.shape[2], x_6.shape[3]))(real_x)
        x_7 = self.decoder3(x_6, x_2)
        real_7 = Resize((x_7.shape[1], x_7.shape[2], x_7.shape[3]))(real_x)
        x_8 = self.decoder4(x_7, x_1)

        # loss = self.loss(x_8, real_x) + self.loss(x_7, real_7) + self.loss(x_6, real_6) + self.loss(x_5, real_5) \
        #          + 1/2 * self.loss(x_4, real_4) + 1/16 * self.loss(x_3, real_3) * 1/64 * self.loss(x_2, real_2)
        loss = self.loss(x_8, real_x) + 1/2*self.loss(x_7, real_7) + 1/4*self.loss(x_6, real_6) + 1/8*self.loss(x_5, real_5) \
                    + 1/16*self.loss(x_4, real_4) + 1/32*self.loss(x_3, real_3) + 1/64*self.loss(x_2, real_2)
        return loss, x_8

    def inference(self, x):
        x = self.in_conv(x)
        x_1 = self.encoder1(x)

        x = self.down1(x_1)
        x_2 = self.encoder2(x)

        x = self.down2(x_2)
        x_3 = self.encoder3(x)

        x = self.down3(x_3)
        x_4 = self.encoder4(x)

        x = self.down4(x_4)
        x = self.middle(x)

        x_5 = self.decoder1(x, x_4)
        x_6 = self.decoder2(x_5, x_3)
        x_7 = self.decoder3(x_6, x_2)
        x_8 = self.decoder4(x_7, x_1)
        return x_8

class Progressive6_MLP(nn.Module):
    def __init__(self, in_channels, out_channels, num_channels=16):
        super().__init__()
        self.in_conv = nn.Conv2d(in_channels, num_channels, kernel_size=3, stride=1, padding='same')

        self.encoder1 = ConvBlock(num_channels, num_channels, kernel_size=3, stride=1, padding='same')
        self.encoder2 = ConvBlock(num_channels, num_channels*2, kernel_size=3, stride=1, padding='same')
        self.encoder3 = ConvBlock(num_channels*2, num_channels*4, kernel_size=3, stride=1, padding='same')
        self.encoder4 = ConvBlock(num_channels*4, num_channels*8, kernel_size=3, stride=1, padding='same')
        self.middle = ConvBlock(num_channels*8, num_channels*16, kernel_size=3, stride=1, padding='same')

        self.down1 = nn.MaxPool2d(2)
        self.down2 = nn.MaxPool2d(2)
        self.down3 = nn.MaxPool2d(2)
        self.down4 = nn.MaxPool2d(2)

        self.decoder1 = JointDecoderBlock(256, 128, out_channels, kernel_size=3, stride=1, padding='same')
        self.decoder2 = JointDecoderBlock(out_channels, 64, out_channels, kernel_size=3, stride=1, padding='same')
        self.decoder3 = JointDecoderBlock(out_channels, 32, out_channels, kernel_size=3, stride=1, padding='same')
        self.decoder4 = JointDecoderBlock(out_channels, 16, out_channels, kernel_size=3, stride=1, padding='same')

        self.loss = reconstruction_loss

    def forward(self, x, real_x):
        x = self.in_conv(x)
        x_1 = self.encoder1(x)

        x = self.down1(x_1)
        x_2 = self.encoder2(x)

        x = self.down2(x_2)
        x_3 = self.encoder3(x)
        real_3 = Resize((x_3.shape[1], x_3.shape[2], x_3.shape[3]))(real_x)

        x = self.down3(x_3)
        x_4 = self.encoder4(x)
        real_4 = Resize((x_4.shape[1], x_4.shape[2], x_4.shape[3]))(real_x)

        x = self.down4(x_4)
        x = self.middle(x)

        x_5 = self.decoder1(x, x_4)
        real_5 = Resize((x_5.shape[1], x_5.shape[2], x_5.shape[3]))(real_x)
        x_6 = self.decoder2(x_5, x_3)
        real_6 = Resize((x_6.shape[1], x_6.shape[2], x_6.shape[3]))(real_x)
        x_7 = self.decoder3(x_6, x_2)
        real_7 = Resize((x_7.shape[1], x_7.shape[2], x_7.shape[3]))(real_x)
        x_8 = self.decoder4(x_7, x_1)

        # loss = self.loss(x_8, real_x) + self.loss(x_7, real_7) + self.loss(x_6, real_6) + self.loss(x_5, real_5) \
        #          + 1/2 * self.loss(x_4, real_4) + 1/16 * self.loss(x_3, real_3)
        loss = self.loss(x_8, real_x) + 1 / 2 * self.loss(x_7, real_7) + 1 / 4 * self.loss(x_6, real_6) + 1 / 8 * self.loss(
            x_5, real_5) + 1 / 16 * self.loss(x_4, real_4) + 1 / 32 * self.loss(x_3, real_3)
        return loss, x_8

    def inference(self, x):
        x = self.in_conv(x)
        x_1 = self.encoder1(x)

        x = self.down1(x_1)
        x_2 = self.encoder2(x)

        x = self.down2(x_2)
        x_3 = self.encoder3(x)

        x = self.down3(x_3)
        x_4 = self.encoder4(x)

        x = self.down4(x_4)
        x = self.middle(x)

        x_5 = self.decoder1(x, x_4)
        x_6 = self.decoder2(x_5, x_3)
        x_7 = self.decoder3(x_6, x_2)
        x_8 = self.decoder4(x_7, x_1)
        return x_8

class Progressive5_MLP(nn.Module):
    def __init__(self, in_channels, out_channels, num_channels=16):
        super().__init__()
        self.in_conv = nn.Conv2d(in_channels, num_channels, kernel_size=3, stride=1, padding='same')

        self.encoder1 = ConvBlock(num_channels, num_channels, kernel_size=3, stride=1, padding='same')
        self.encoder2 = ConvBlock(num_channels, num_channels*2, kernel_size=3, stride=1, padding='same')
        self.encoder3 = ConvBlock(num_channels*2, num_channels*4, kernel_size=3, stride=1, padding='same')
        self.encoder4 = ConvBlock(num_channels*4, num_channels*8, kernel_size=3, stride=1, padding='same')
        self.middle = ConvBlock(num_channels*8, num_channels*16, kernel_size=3, stride=1, padding='same')

        self.down1 = nn.MaxPool2d(2)
        self.down2 = nn.MaxPool2d(2)
        self.down3 = nn.MaxPool2d(2)
        self.down4 = nn.MaxPool2d(2)

        self.decoder1 = JointDecoderBlock(256, 128, out_channels, kernel_size=3, stride=1, padding='same')
        self.decoder2 = JointDecoderBlock(out_channels, 64, out_channels, kernel_size=3, stride=1, padding='same')
        self.decoder3 = JointDecoderBlock(out_channels, 32, out_channels, kernel_size=3, stride=1, padding='same')
        self.decoder4 = JointDecoderBlock(out_channels, 16, out_channels, kernel_size=3, stride=1, padding='same')

        self.loss = reconstruction_loss

    def forward(self, x, real_x):
        x = self.in_conv(x)
        x_1 = self.encoder1(x)

        x = self.down1(x_1)
        x_2 = self.encoder2(x)

        x = self.down2(x_2)
        x_3 = self.encoder3(x)

        x = self.down3(x_3)
        x_4 = self.encoder4(x)
        real_4 = Resize((x_4.shape[1], x_4.shape[2], x_4.shape[3]))(real_x)

        x = self.down4(x_4)
        x = self.middle(x)

        x_5 = self.decoder1(x, x_4)
        real_5 = Resize((x_5.shape[1], x_5.shape[2], x_5.shape[3]))(real_x)
        x_6 = self.decoder2(x_5, x_3)
        real_6 = Resize((x_6.shape[1], x_6.shape[2], x_6.shape[3]))(real_x)
        x_7 = self.decoder3(x_6, x_2)
        real_7 = Resize((x_7.shape[1], x_7.shape[2], x_7.shape[3]))(real_x)
        x_8 = self.decoder4(x_7, x_1)

        # loss = self.loss(x_8, real_x) + self.loss(x_7, real_7) + self.loss(x_6, real_6) + self.loss(x_5, real_5) \
        #          + 1/2 * self.loss(x_4, real_4)
        loss = self.loss(x_8, real_x) + 1 / 2 * self.loss(x_7, real_7) + 1 / 4 * self.loss(x_6, real_6) + 1 / 8 * self.loss(
            x_5, real_5) + 1 / 16 * self.loss(x_4, real_4)
        return loss, x_8

    def inference(self, x):
        x = self.in_conv(x)
        x_1 = self.encoder1(x)

        x = self.down1(x_1)
        x_2 = self.encoder2(x)

        x = self.down2(x_2)
        x_3 = self.encoder3(x)

        x = self.down3(x_3)
        x_4 = self.encoder4(x)

        x = self.down4(x_4)
        x = self.middle(x)

        x_5 = self.decoder1(x, x_4)
        x_6 = self.decoder2(x_5, x_3)
        x_7 = self.decoder3(x_6, x_2)
        x_8 = self.decoder4(x_7, x_1)
        return x_8

class Progressive4_MLP(nn.Module):
    def __init__(self, in_channels, out_channels, num_channels=16):
        super().__init__()
        self.in_conv = nn.Conv2d(in_channels, num_channels, kernel_size=3, stride=1, padding='same')

        self.encoder1 = ConvBlock(num_channels, num_channels, kernel_size=3, stride=1, padding='same')
        self.encoder2 = ConvBlock(num_channels, num_channels*2, kernel_size=3, stride=1, padding='same')
        self.encoder3 = ConvBlock(num_channels*2, num_channels*4, kernel_size=3, stride=1, padding='same')
        self.encoder4 = ConvBlock(num_channels*4, num_channels*8, kernel_size=3, stride=1, padding='same')
        self.middle = ConvBlock(num_channels*8, num_channels*16, kernel_size=3, stride=1, padding='same')

        self.down1 = nn.MaxPool2d(2)
        self.down2 = nn.MaxPool2d(2)
        self.down3 = nn.MaxPool2d(2)
        self.down4 = nn.MaxPool2d(2)

        self.decoder1 = JointDecoderBlock(256, 128, out_channels, kernel_size=3, stride=1, padding='same')
        self.decoder2 = JointDecoderBlock(out_channels, 64, out_channels, kernel_size=3, stride=1, padding='same')
        self.decoder3 = JointDecoderBlock(out_channels, 32, out_channels, kernel_size=3, stride=1, padding='same')
        self.decoder4 = JointDecoderBlock(out_channels, 16, out_channels, kernel_size=3, stride=1, padding='same')

        self.loss = reconstruction_loss

    def forward(self, x, real_x):
        x = self.in_conv(x)
        x_1 = self.encoder1(x)

        x = self.down1(x_1)
        x_2 = self.encoder2(x)

        x = self.down2(x_2)
        x_3 = self.encoder3(x)

        x = self.down3(x_3)
        x_4 = self.encoder4(x)

        x = self.down4(x_4)
        x = self.middle(x)

        x_5 = self.decoder1(x, x_4)
        real_5 = Resize((x_5.shape[1], x_5.shape[2], x_5.shape[3]))(real_x)
        x_6 = self.decoder2(x_5, x_3)
        real_6 = Resize((x_6.shape[1], x_6.shape[2], x_6.shape[3]))(real_x)
        x_7 = self.decoder3(x_6, x_2)
        real_7 = Resize((x_7.shape[1], x_7.shape[2], x_7.shape[3]))(real_x)
        x_8 = self.decoder4(x_7, x_1)

        loss = self.loss(x_8, real_x) + self.loss(x_7, real_7) + self.loss(x_6, real_6) + self.loss(x_5, real_5) # no weight
        # loss = self.loss(x_8, real_x) + 1/2*self.loss(x_7, real_7) + 1/4*self.loss(x_6, real_6) + 1/8*self.loss(x_5, real_5) # decay weight
        return loss, x_8

    def inference(self, x):
        x = self.in_conv(x)
        x_1 = self.encoder1(x)

        x = self.down1(x_1)
        x_2 = self.encoder2(x)

        x = self.down2(x_2)
        x_3 = self.encoder3(x)

        x = self.down3(x_3)
        x_4 = self.encoder4(x)

        x = self.down4(x_4)
        x = self.middle(x)

        x_5 = self.decoder1(x, x_4)
        x_6 = self.decoder2(x_5, x_3)
        x_7 = self.decoder3(x_6, x_2)
        x_8 = self.decoder4(x_7, x_1)
        return x_8

class Progressive3_MLP(nn.Module):
    def __init__(self, in_channels, out_channels, num_channels=16):
        super().__init__()
        self.in_conv = nn.Conv2d(in_channels, num_channels, kernel_size=3, stride=1, padding='same')

        self.encoder1 = ConvBlock(num_channels, num_channels, kernel_size=3, stride=1, padding='same')
        self.encoder2 = ConvBlock(num_channels, num_channels*2, kernel_size=3, stride=1, padding='same')
        self.encoder3 = ConvBlock(num_channels*2, num_channels*4, kernel_size=3, stride=1, padding='same')
        self.encoder4 = ConvBlock(num_channels*4, num_channels*8, kernel_size=3, stride=1, padding='same')
        self.middle = ConvBlock(num_channels*8, num_channels*16, kernel_size=3, stride=1, padding='same')

        self.down1 = nn.MaxPool2d(2)
        self.down2 = nn.MaxPool2d(2)
        self.down3 = nn.MaxPool2d(2)
        self.down4 = nn.MaxPool2d(2)

        self.decoder1 = JointDecoderBlock(256, 128, out_channels, kernel_size=3, stride=1, padding='same')
        self.decoder2 = JointDecoderBlock(out_channels, 64, out_channels, kernel_size=3, stride=1, padding='same')
        self.decoder3 = JointDecoderBlock(out_channels, 32, out_channels, kernel_size=3, stride=1, padding='same')
        self.decoder4 = JointDecoderBlock(out_channels, 16, out_channels, kernel_size=3, stride=1, padding='same')

        self.loss = reconstruction_loss

    def forward(self, x, real_x):
        x = self.in_conv(x)
        x_1 = self.encoder1(x)

        x = self.down1(x_1)
        x_2 = self.encoder2(x)

        x = self.down2(x_2)
        x_3 = self.encoder3(x)

        x = self.down3(x_3)
        x_4 = self.encoder4(x)

        x = self.down4(x_4)
        x = self.middle(x)

        x_5 = self.decoder1(x, x_4)
        x_6 = self.decoder2(x_5, x_3)
        real_6 = Resize((x_6.shape[1], x_6.shape[2], x_6.shape[3]))(real_x)
        x_7 = self.decoder3(x_6, x_2)
        real_7 = Resize((x_7.shape[1], x_7.shape[2], x_7.shape[3]))(real_x)
        x_8 = self.decoder4(x_7, x_1)

        # loss = self.loss(x_8, real_x) + self.loss(x_7, real_7) + self.loss(x_6, real_6)
        loss = self.loss(x_8, real_x) + 1 / 2 * self.loss(x_7, real_7) + 1 / 4 * self.loss(x_6,real_6)
        return loss, x_8

    def inference(self, x):
        x = self.in_conv(x)
        x_1 = self.encoder1(x)

        x = self.down1(x_1)
        x_2 = self.encoder2(x)

        x = self.down2(x_2)
        x_3 = self.encoder3(x)

        x = self.down3(x_3)
        x_4 = self.encoder4(x)

        x = self.down4(x_4)
        x = self.middle(x)

        x_5 = self.decoder1(x, x_4)
        x_6 = self.decoder2(x_5, x_3)
        x_7 = self.decoder3(x_6, x_2)
        x_8 = self.decoder4(x_7, x_1)
        return x_8

class Progressive2_MLP(nn.Module):
    def __init__(self, in_channels, out_channels, num_channels=16):
        super().__init__()
        self.in_conv = nn.Conv2d(in_channels, num_channels, kernel_size=3, stride=1, padding='same')

        self.encoder1 = ConvBlock(num_channels, num_channels, kernel_size=3, stride=1, padding='same')
        self.encoder2 = ConvBlock(num_channels, num_channels*2, kernel_size=3, stride=1, padding='same')
        self.encoder3 = ConvBlock(num_channels*2, num_channels*4, kernel_size=3, stride=1, padding='same')
        self.encoder4 = ConvBlock(num_channels*4, num_channels*8, kernel_size=3, stride=1, padding='same')
        self.middle = ConvBlock(num_channels*8, num_channels*16, kernel_size=3, stride=1, padding='same')

        self.down1 = nn.MaxPool2d(2)
        self.down2 = nn.MaxPool2d(2)
        self.down3 = nn.MaxPool2d(2)
        self.down4 = nn.MaxPool2d(2)

        self.decoder1 = JointDecoderBlock(256, 128, out_channels, kernel_size=3, stride=1, padding='same')
        self.decoder2 = JointDecoderBlock(out_channels, 64, out_channels, kernel_size=3, stride=1, padding='same')
        self.decoder3 = JointDecoderBlock(out_channels, 32, out_channels, kernel_size=3, stride=1, padding='same')
        self.decoder4 = JointDecoderBlock(out_channels, 16, out_channels, kernel_size=3, stride=1, padding='same')

        self.loss = reconstruction_loss

    def forward(self, x, real_x):
        x = self.in_conv(x)
        x_1 = self.encoder1(x)

        x = self.down1(x_1)
        x_2 = self.encoder2(x)

        x = self.down2(x_2)
        x_3 = self.encoder3(x)

        x = self.down3(x_3)
        x_4 = self.encoder4(x)

        x = self.down4(x_4)
        x = self.middle(x)

        x_5 = self.decoder1(x, x_4)
        x_6 = self.decoder2(x_5, x_3)
        x_7 = self.decoder3(x_6, x_2)
        real_7 = Resize((x_7.shape[1], x_7.shape[2], x_7.shape[3]))(real_x)
        x_8 = self.decoder4(x_7, x_1)

        # loss = self.loss(x_8, real_x) + self.loss(x_7, real_7)
        loss = self.loss(x_8, real_x) + 1 / 2 * self.loss(x_7, real_7)
        return loss, x_8

    def inference(self, x):
        x = self.in_conv(x)
        x_1 = self.encoder1(x)

        x = self.down1(x_1)
        x_2 = self.encoder2(x)

        x = self.down2(x_2)
        x_3 = self.encoder3(x)

        x = self.down3(x_3)
        x_4 = self.encoder4(x)

        x = self.down4(x_4)
        x = self.middle(x)

        x_5 = self.decoder1(x, x_4)
        x_6 = self.decoder2(x_5, x_3)
        x_7 = self.decoder3(x_6, x_2)
        x_8 = self.decoder4(x_7, x_1)
        return x_8


'''******************************************************************************************'''


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.norm(self.conv1(x)))
        x = self.relu(self.norm2(self.conv2(x)))
        return x

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        super().__init__()
        self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear')

        # self.conv1 = nn.Conv2d(in_channels+out_channels, out_channels, kernel_size, stride, padding)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x, skip_x):
        x = self.up_sample(x)
        x = torch.cat([x, skip_x], axis=1)
        x = self.relu(self.norm(self.conv1(x)))
        x = self.relu(self.norm2(self.conv2(x)))
        return x
