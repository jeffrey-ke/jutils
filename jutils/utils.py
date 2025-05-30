import torch
import functools
import ipdb as p
import os
import sys
def ensure_same_device(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Collect all torch.Tensor inputs
        tensor_args = [arg for arg in args if isinstance(arg, torch.Tensor)]
        tensor_kwargs = [v for v in kwargs.values() if isinstance(v, torch.Tensor)]
        all_tensors = tensor_args + tensor_kwargs

        if not all_tensors:
            return func(*args, **kwargs)  # If no tensor inputs, proceed normally

        # Get the device of the first tensor
        main_device = all_tensors[0].device

        # Check if all tensors are on the same device
        if any(tensor.device != main_device for tensor in all_tensors):
            raise ValueError(f"All tensors must be on the same device. Expected: {main_device}, but found mixed devices.")

        return func(*args, **kwargs)

    return wrapper


def pdb():
    p.set_trace()


@ensure_same_device
def gradient(outputs, inputs):
    with torch.enable_grad():
        inputs.requires_grad_(True)
        grad =  torch.autograd.grad(outputs,
                                    inputs,
                                    torch.ones_like(outputs, device=inputs.device),
                                    create_graph=True,
                                    retain_graph=True)
    return grad[0]


def get_params(model, exclude=None):
    if exclude==None:
        exclude=[]
    params = [param for name,param in model.named_parameters()
            if not any(excluded in name for excluded in exclude)]
    print(len(params))
    return params


def channel_last(batched_images):
    # precondition: shape is (B,C,H,W)
    t = batched_images.permute((0,2,3,1))
    return t


def load_images(path):
    pass

def add_module_to_path(module_path):
    path_to_module = os.path.abspath(os.path.join(os.getcwd(), module_path))
    sys.path.append(path_to_module)
