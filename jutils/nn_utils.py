import torch.nn as nn
from jutils.config_classes import *
def create_encoder(channel_list, config):
    C,H,W  = config.shape
    mods = []
    stride = config.spatial_reduce_factor
    if config.fatten_first:
        mods.append(nn.Conv2d(C, config.fat, kernel_size=3, stride=1, padding=1)
    for C in channel_list:
        mods.append(nn.Conv2d(C_in, C, kernel_size=3, stride=stride, padding=1)
    return nn.Sequential(*mods)

