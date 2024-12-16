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

class UNet(nn.Module):
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

        self.decoder1 = DecoderBlock(384, 128, kernel_size=3, stride=1, padding='same')
        self.decoder2 = DecoderBlock(192, 100, kernel_size=3, stride=1, padding='same')
        self.decoder3 = DecoderBlock(132, 100, kernel_size=3, stride=1, padding='same')
        self.decoder4 = DecoderBlock(116, 100, kernel_size=3, stride=1, padding='same')

    def forward(self, x):
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

        x = self.decoder1(x, x_4)
        x = self.decoder2(x, x_3)
        x = self.decoder3(x, x_2)
        x = self.decoder4(x, x_1)
        return x

class JointDecoderBlock(nn.Module):
    def __init__(self, in_channels, cat_channels, out_channels, kernel_size=3, stride=1, padding=0):
        super().__init__()
        self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.mlp_block = MLP_Block(out_channels)

        self.conv1 = nn.Conv2d(in_channels+cat_channels, out_channels, kernel_size, stride, padding)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels+cat_channels, out_channels, kernel_size, stride, padding)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x, skip_x):
        x = self.up_sample(x)
        x = torch.cat([x, skip_x], axis=1)
        x_conv = self.relu(self.norm1(self.conv1(x)))
        x_mlp = self.mlp_block(x_conv)
        x_conv = torch.cat([x_mlp, skip_x], axis=1)
        x_conv = self.relu(self.norm2(self.conv2(x_conv)))
        return x_conv



########################################################
# Blocks
########################################################

class MLP_Block(nn.Module):  # input shape: n, c, h, w;  output shape: n, c, h, w

    # def __init__(self, num_channels, block_size=(16, 16), grid_size=(16, 16), # 16/12/8
    # def __init__(self, num_channels, block_size=(12, 12), grid_size=(12, 12),  # 16/12/8
    def __init__(self, num_channels, block_size=(8, 8), grid_size=(8, 8),  # 16/12/8
                 block_gmlp_factor=2, grid_gmlp_factor=2, input_proj_factor=2,
                 channels_reduction=4, lrelu_slope=0.2, dropout_rate=0.0, use_bias=True, use_checkpoint=False):
        super().__init__()

        self.mlpLayer = MultiAxisGmlpLayer(block_size=block_size, grid_size=grid_size,
                                           num_channels=num_channels, input_proj_factor=input_proj_factor,
                                           block_gmlp_factor=block_gmlp_factor, grid_gmlp_factor=grid_gmlp_factor,
                                           dropout_rate=dropout_rate, use_bias=use_bias, use_checkpoint=use_checkpoint)
        self.channel_attention_block = RCAB(num_channels=num_channels, reduction=channels_reduction,
                                            lrelu_slope=lrelu_slope,
                                            use_bias=use_bias, use_checkpoint=use_checkpoint)

    def forward(self, x_in):
        x = x_in.permute(0, 2, 3, 1)  # n,h,w,c
        x = self.mlpLayer(x)
        x = self.channel_attention_block(x)
        x = x.permute(0, 3, 1, 2)  # n,c,h,w

        x_out = x + x_in
        return x_out


class PatchMerging_block(nn.Module):  # input shape: n, c, h, w;  output shape: n, 2c, h/2, w/2
    """Downsampling operation"""

    def __init__(self, embed_dim: int):
        super().__init__()

        self.norm = nn.LayerNorm(embed_dim * 4)
        self.reduction = nn.Linear(embed_dim * 4, embed_dim * 2, bias=False)

    def forward(self, x_in):
        x = einops.rearrange(x_in, 'b c d h -> b d h c')

        b, d, h, c = x.shape
        if (d % 2 == 1) or (h % 2 == 1):
            x = nnf.pad(x, (0, 0, 0, h % 2, 0, d % 2))

        x = torch.cat([x[:, i::2, j::2, :] for i, j in itertools.product(range(2), range(2))], dim=-1)

        x = self.norm(x)
        x = self.reduction(x)
        x_out = einops.rearrange(x, 'b d h c -> b c d h')

        return x_out


########################################################
# Layers
########################################################

class MultiAxisGmlpLayer(nn.Module):  # input shape: n, h, w, c
    """The multi-axis gated MLP block."""

    def __init__(self, block_size, grid_size, num_channels,
                 input_proj_factor=2, block_gmlp_factor=2, grid_gmlp_factor=2,
                 use_bias=True, dropout_rate=0.0, use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint

        self.LayerNorm = nn.LayerNorm(num_channels)
        self.in_project = nn.Linear(num_channels, num_channels * input_proj_factor, bias=use_bias)
        self.gelu = nn.GELU()
        self.GridGmlpLayer = GridGmlpLayer(grid_size=grid_size, num_channels=num_channels * input_proj_factor // 2,
                                           use_bias=use_bias, factor=grid_gmlp_factor)
        self.BlockGmlpLayer = BlockGmlpLayer(block_size=block_size, num_channels=num_channels * input_proj_factor // 2,
                                             use_bias=use_bias, factor=block_gmlp_factor)
        self.out_project = nn.Linear(num_channels * input_proj_factor, num_channels, bias=use_bias)
        self.dropout = nn.Dropout(dropout_rate)

    def forward_run(self, x_in):

        x = self.LayerNorm(x_in)
        x = self.in_project(x)
        x = self.gelu(x)
        c = x.size(-1) // 2
        u, v = torch.split(x, c, dim=-1)

        # grid gMLP
        u = self.GridGmlpLayer(u)

        # block gMLP
        v = self.BlockGmlpLayer(v)

        # out projection
        x = torch.cat([u, v], dim=-1)
        x = self.out_project(x)
        x = self.dropout(x)

        x_out = x + x_in
        return x_out

    def forward(self, x_in):

        if self.use_checkpoint and x_in.requires_grad:
            x_out = checkpoint.checkpoint(self.forward_run, x_in)
        else:
            x_out = self.forward_run(x_in)
        return x_out


class GridGmlpLayer(nn.Module):  # input shape: n, h, w, c
    """Grid gMLP layer that performs global mixing of tokens."""

    def __init__(self, grid_size, num_channels, use_bias=True, factor=2, dropout_rate=0):
        super().__init__()
        self.gh = grid_size[0]
        self.gw = grid_size[1]

        self.LayerNorm = nn.LayerNorm(num_channels)
        self.in_project = nn.Linear(num_channels, num_channels * factor, use_bias)  # c->c*factor
        self.gelu = nn.GELU()
        self.GridGatingUnit = GridGatingUnit(num_channels * factor, n=self.gh * self.gw)  # c*factor->c*factor//2
        self.out_project = nn.Linear(num_channels * factor // 2, num_channels, use_bias)  # c*factor//2->c
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        _, h, w, _ = x.shape

        # padding
        pad_t = pad_d0 = 0
        pad_d1 = (self.gh - h % self.gh) % self.gh
        pad_b = (self.gw - w % self.gw) % self.gw
        x = nnf.pad(x, (0, 0, pad_t, pad_b, pad_d0, pad_d1))

        fh, fw = x.shape[1] // self.gh, x.shape[2] // self.gw
        x = block_images_einops(x, patch_size=(fh, fw))  # n (gh gw) (fh fw) c

        # gMLP: Global (grid) mixing part, provides global grid communication.
        shortcut = x
        x = self.LayerNorm(x)
        x = self.in_project(x)
        x = self.gelu(x)
        x = self.GridGatingUnit(x)
        x = self.out_project(x)
        x = self.dropout(x)
        x = x + shortcut

        x = unblock_images_einops(x, grid_size=(self.gh, self.gw), patch_size=(fh, fw))
        if pad_d1 > 0 or pad_b > 0:
            x = x[:, :h, :w, :].contiguous()

        return x


class BlockGmlpLayer(nn.Module):  # input shape: n, h, w, c
    """Block gMLP layer that performs local mixing of tokens."""

    def __init__(self, block_size, num_channels, use_bias=True, factor=2, dropout_rate=0.0):
        super().__init__()

        self.fh = block_size[0]
        self.fw = block_size[1]

        self.LayerNorm = nn.LayerNorm(num_channels)
        self.in_project = nn.Linear(num_channels, num_channels * factor, use_bias)  # c->c*factor
        self.gelu = nn.GELU()
        self.BlockGatingUnit = BlockGatingUnit(num_channels * factor, n=self.fh * self.fw)  # c*factor->c*factor//2
        self.out_project = nn.Linear(num_channels * factor // 2, num_channels, use_bias)  # c*factor//2->c
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        _, h, w, _ = x.shape

        # padding
        pad_t = pad_d0 = 0
        pad_d1 = (self.fh - h % self.fh) % self.fh
        pad_b = (self.fw - w % self.fw) % self.fw
        x = nnf.pad(x, (0, 0, pad_t, pad_b, pad_d0, pad_d1))

        gh, gw = x.shape[1] // self.fh, x.shape[2] // self.fw
        x = block_images_einops(x, patch_size=(self.fh, self.fw))  # n (gh gw) (fh fw) c

        # gMLP: Local (block) mixing part, provides local block communication.
        shortcut = x
        x = self.LayerNorm(x)
        x = self.in_project(x)
        x = self.gelu(x)
        x = self.BlockGatingUnit(x)
        x = self.out_project(x)
        x = self.dropout(x)
        x = x + shortcut

        x = unblock_images_einops(x, grid_size=(gh, gw), patch_size=(self.fh, self.fw))
        if pad_d1 > 0 or pad_b > 0:
            x = x[:, :h, :w, :].contiguous()

        return x


class GridGatingUnit(nn.Module):  # input shape: n (gh gw) (fh fw) c
    """A SpatialGatingUnit as defined in the gMLP paper."""

    def __init__(self, c, n, use_bias=True):
        super().__init__()

        self.Dense_0 = nn.Linear(n, n, use_bias)
        self.LayerNorm = nn.LayerNorm(c // 2)

    def forward(self, x):
        c = x.size(-1)
        c = c // 2
        u, v = torch.split(x, c, dim=-1)

        v = self.LayerNorm(v)
        v = v.permute(0, 3, 2, 1)  # n, c/2, (fh fw) (gh gw)
        v = self.Dense_0(v)
        v = v.permute(0, 3, 2, 1)  # n (gh gw) (fh fw) c/2

        return u * (v + 1.0)


class BlockGatingUnit(nn.Module):  # input shape: n (gh gw) (fh fw) c
    """A SpatialGatingUnit as defined in the gMLP paper."""

    def __init__(self, c, n, use_bias=True):
        super().__init__()

        self.Dense_0 = nn.Linear(n, n, use_bias)
        self.LayerNorm = nn.LayerNorm(c // 2)

    def forward(self, x):
        c = x.size(-1)
        c = c // 2
        u, v = torch.split(x, c, dim=-1)

        v = self.LayerNorm(v)
        v = v.permute(0, 1, 3, 2)  # n, (gh gw), c/2, (fh fw)
        v = self.Dense_0(v)
        v = v.permute(0, 1, 3, 2)  # n (gh gw) (fh fw) c/2

        return u * (v + 1.0)


class RCAB(nn.Module):  # input shape: n, h, w, c
    """Residual channel attention block. Contains LN,Conv,lRelu,Conv,SELayer."""

    def __init__(self, num_channels, reduction=4, lrelu_slope=0.2, use_bias=True, use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint

        self.LayerNorm = nn.LayerNorm(num_channels)
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, bias=use_bias, padding='same')
        self.leaky_relu = nn.LeakyReLU(negative_slope=lrelu_slope)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, bias=use_bias, padding='same')
        self.channel_attention = CALayer(num_channels=num_channels, reduction=reduction)

    def forward_run(self, x):

        shortcut = x
        x = self.LayerNorm(x)

        x = x.permute(0, 3, 1, 2)  # n,c,h,w
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = x.permute(0, 2, 3, 1)  # n,h,w,c

        x = self.channel_attention(x)
        x_out = x + shortcut

        return x_out

    def forward(self, x):

        if self.use_checkpoint and x.requires_grad:
            x = checkpoint.checkpoint(self.forward_run, x)
        else:
            x = self.forward_run(x)
        return x


class CALayer(nn.Module):  # input shape: n, h, w, c
    """Squeeze-and-excitation block for channel attention."""

    def __init__(self, num_channels, reduction=4, use_bias=True):
        super().__init__()

        self.Conv_0 = nn.Conv2d(num_channels, num_channels // reduction, kernel_size=1, stride=1, bias=use_bias)
        self.relu = nn.ReLU()
        self.Conv_1 = nn.Conv2d(num_channels // reduction, num_channels, kernel_size=1, stride=1, bias=use_bias)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_in):
        x = x_in.permute(0, 3, 1, 2)  # n,c,h,w
        x = torch.mean(x, dim=(2, 3), keepdim=True)
        x = self.Conv_0(x)
        x = self.relu(x)
        x = self.Conv_1(x)
        w = self.sigmoid(x)
        w = w.permute(0, 2, 3, 1)  # n,h,w,c

        x_out = x_in * w
        return x_out


########################################################
# Functions
########################################################

def block_images_einops(x, patch_size):  # n, h, w, c
    """Image to patches."""

    batch, height, width, channels = x.shape
    grid_height = height // patch_size[0]
    grid_width = width // patch_size[1]

    x = einops.rearrange(
        x, "n (gh fh) (gw fw) c -> n (gh gw) (fh fw) c",
        gh=grid_height, gw=grid_width, fh=patch_size[0], fw=patch_size[1])
    return x


def unblock_images_einops(x, grid_size, patch_size):
    """patches to images."""

    x = einops.rearrange(
        x, "n (gh gw) (fh fw) c -> n (gh fh) (gw fw) c",
        gh=grid_size[0], gw=grid_size[1], fh=patch_size[0], fw=patch_size[1])
    return x
