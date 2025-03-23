import torch.nn


"""
A VAE encoder. 
Takes an image.
Bottlenecks it using conv layers.
Outputs a latent dim.
Needs to know the input shape (really? I guess I do because I need to know
what the final dense layer looks like)
def VAEEncoder(nn.Module):
    def __init__(self, ): #todo

