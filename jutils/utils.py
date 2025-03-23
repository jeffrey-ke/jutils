import torch
import functools
import pdb as p

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
