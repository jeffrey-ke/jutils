import torch
import torch.nn as nn
from jutils.config_classes import *
# discovering that each network is pretty bespoke. I might be able to create a module
# that automatically handles resconnections or skips based on the module's size...
def create_encoder(channel_list, config):
    C_in,H,W  = config.shape
    mods = []
    stride = config.spatial_reduce_factor
    prevC = C_in
    if config.fatten_first:
        mods.append(nn.Conv2d(C_in, config.fat, kernel_size=3, stride=1, padding=1))
        prevC = config.fat

    for C in channel_list:
        mods.append(nn.Conv2d(prevC, C, kernel_size=3, stride=stride, padding=1))
        mods.append(nn.ReLU())
        prevC = C
    return nn.Sequential(*mods)

def xavier_init(module):
    for m in module.modules():
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.xavier_normal_(m.weight)


def batch_size(tensor):
    return tensor.shape[0]

def batchify(single_tensor):
    return single_tensor.unsqueeze(0)

def unbatchify(single_batched_tensor):
    return single_batched_tensor.squeeze(0)

def loraify(model, include=["attn", "linear"], r=4):
    for name, module in model.modules():
        if isinstance(module, nn.Linear) or any(inc in name for inc in include):
            def get_immediate_parent(parent_obj, parent_arr):
                parent = parent_obj
                for p in parent_arr:
                    parent = getattr(parent, p)
                return parent
            *parent_arr, child = name.split(".")
            immediate_parent = get_immediate_parent(model, parent_arr)
            setattr(immediate_parent, child, LoRA(module, r=r))
