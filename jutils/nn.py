import torch.nn as nn


"""
A VAE encoder. 
Takes an image.
Bottlenecks it using conv layers.
Outputs a latent dim.
Needs to know the input shape (really? I guess I do because I need to know
what the final dense layer looks like)
def VAEEncoder(nn.Module):
    def __init__(self, ): #todo
"""
"""
def ConvTDouble(nn.Module):
    def __init__(self, 3d or 2d, in channels, out channels, relu_after=False, batchnorm_after=False):
        layers = []
        if 3d:
            layer append ConvTranspose3d(out, in)
        elif 2d:
            layer append  ConvTranspose2d(out, in)
        
        if batch_norm after:
            layer append batchnorm with out channels
        if relu after:
            layer append relu

        self.me =  Sequential(*layers)

    def forward(self, X):
        return self.me(X)

"""
