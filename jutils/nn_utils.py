from collections import OrderedDict
import torch
from jutils.utils import pdb
from peft import LoraConfig, get_peft_model, TaskType
import torch.nn as nn
from jutils.jnn import LoRA
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

def lorafy(model):
    lora_cfg = LoraConfig(
        r               = 8,
        lora_alpha      = 32,
        lora_dropout    = 0.05,
        bias            = "none",
        task_type       = TaskType.FEATURE_EXTRACTION,   
        target_modules  = [
            "q_proj", "k_proj", "v_proj", "out_proj",    # attention
            "linear1", "linear2",                       # FFN
        ],
    )
    detr_lora = get_peft_model(model, lora_cfg)     
    return detr_lora

def get_state_dict(path):
    # args.frozen state_dict is the path to the checkpoint
    checkpoint = torch.hub.load_state_dict_from_url(path, map_location='cpu')
    return checkpoint['model']

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )
def load_state_dict_up_to_classif_head(detr, args):
    state_dict = get_state_dict(path=args.resume)
    for k,v in list(state_dict.items()):
        if "class_embed" in k:
            state_dict.pop(k)
    detr.load_state_dict(state_dict, strict=False)

def loraify_deprecated(model, include=["attn", "linear"], r=4):
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) or any(inc in name for inc in include):
            def get_immediate_parent(parent_obj, parent_arr):
                parent = parent_obj
                for p in parent_arr:
                    parent = getattr(parent, p)
                return parent
            *parent_arr, child = name.split(".")
            immediate_parent = get_immediate_parent(model, parent_arr)
            setattr(immediate_parent, child, LoRA(module, r=r))
            #TODO: some logging utility would be nice: like logger.log(immediate_parent, child)

def unlorafy_state_dict(state_dict):
    new_sd = OrderedDict()
    for k,v in list(state_dict.items()):
        new_k = k.replace("base_model.model.", "").replace("base_layer.", "")
        new_sd[new_k] = v
    return new_sd

