import torch

def isin_rowwise(tensor, target_tensor):
    """
    tensor: Nxr
    target_tensor: Mxr
    """
    isin = torch.isin(tensor, target_tensor)
    mask = isin.all(dim=1)
    return mask
    
