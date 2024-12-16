import monai
import torch
import torch.nn as nn
from .mlp_models import UNet
from .progressive_mlp_models import Progressive8_MLP, Progressive7_MLP, Progressive6_MLP, Progressive5_MLP, Progressive4_MLP, Progressive3_MLP, Progressive2_MLP

def get_model_recons(name, in_channels=1, out_channels=128, spatial_dim=2):
    if name == 'unet':
        model = UNet(in_channels=in_channels, out_channels=out_channels)
    elif name == 'progressive8_mlp':
        model = Progressive8_MLP(in_channels=in_channels, out_channels=out_channels)
    elif name == 'progressive7_mlp':
        model = Progressive7_MLP(in_channels=in_channels, out_channels=out_channels)
    elif name == 'progressive6_mlp':
        model = Progressive6_MLP(in_channels=in_channels, out_channels=out_channels)
    elif name == 'progressive5_mlp':
        model = Progressive5_MLP(in_channels=in_channels, out_channels=out_channels)
    elif name == 'progressive4_mlp':
        model = Progressive4_MLP(in_channels=in_channels, out_channels=out_channels)
    elif name == 'progressive3_mlp':
        model = Progressive3_MLP(in_channels=in_channels, out_channels=out_channels)
    elif name == 'progressive2_mlp':
        model = Progressive2_MLP(in_channels=in_channels, out_channels=out_channels)
    return model
