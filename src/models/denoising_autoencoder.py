"""
Denoising Autoencoder

Denoising_in_superresolution/src/models
@author: Angel Villar-Corrales
"""

import math

import numpy as np

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F


class ConvAutoencoder(nn.Module):
    """
    Simple fully convolutional autoencoder for image denoising

    Args:
    -----
    num_layers: integer
        number of convolutional layers in encoder and decoder
    num_kernels: list of integers
        number of convolutional kernels in each of the layers
    kernel_size: integer
        size of the convolutional kernels
    """

    def __init__(self, num_layers=4, num_kernels=[64, 128, 256], kernel_size=5):
        """
        Initialization of the model
        """
        assert (len(num_kernels)+1)==num_layers
        super(ConvAutoencoder, self).__init__()

        num_kernels = [3] + num_kernels
        self.num_layers = num_layers
        self.num_kernels = num_kernels
        self.kernel_size = kernel_size

        # encoder
        layers = []
        for i in range(num_layers-1):
            layers.append( nn.Conv2d(num_kernels[i], num_kernels[i+1], kernel_size))
            layers.append( nn.ReLU(True) )
            layers.append( nn.MaxPool2d(2, 2, return_indices=True) )
        self.encoder = nn.Sequential(*layers)

        # decoder
        layers = []
        for i in range(1, num_layers):
            layers.append( nn.MaxUnpool2d(2, 2) )
            layers.append( nn.ConvTranspose2d(num_kernels[-i], num_kernels[-i-1], kernel_size))
            if(i == num_layers-1):
                # layers.append( nn.Sigmoid() )
                pass
            else:
                layers.append( nn.ReLU(True) )
        self.decoder = nn.Sequential(*layers)

        return


    def forward(self, x):
        """
        Forward pass through the autoencoder model
        """

        # flattening the input matrix
        batch = x.shape[0]
        shape = x.shape[1:]
        indices = []
        sizes = []

        # forward through encoder
        for layer in self.encoder:
            if(isinstance(layer, nn.MaxPool2d)):
                sizes.append(x.size())
                x, idx = layer(x)
                indices.append(idx)
            else:
                x = layer(x)

        # forward through decoder
        i = 1
        for layer in self.decoder:
            if(isinstance(layer, nn.MaxUnpool2d)):
                x = layer(x, indices[-i], output_size=sizes[-i])
                i = i + 1
            else:
                x = layer(x)

        return x


class Autoencoder(nn.Module):
    """
    Simple fully connected autoencoder for image denoising

    Args:
    -----
    bottleneck_dim: integer
        dimensionality of the latent space representation
    num_layers: integer
        number of layers in encoder and decoder
    layer_size: list of integers
        list with the dimensionality of each of the layers
    norm: boolean
        If True, latent space representation will be normalized
    size: tuple
        Shape of the images fed to the autoencoder
    """

    def __init__(self, bottleneck_dim=32, num_layers=3, layer_size=[256, 128], norm=False, size=(32,32)):
        """
        Initialization of the model
        """

        assert (len(layer_size)+1)==num_layers
        super(Autoencoder, self).__init__()

        input_size = np.prod(size)
        layer_size = [input_size] + layer_size
        self.num_layers = num_layers
        self.layer_size = layer_size
        self.norm = norm

        # encoder
        layers = []
        for i in range(num_layers-1):
            layers.append( nn.Linear(layer_size[i], layer_size[i+1]) )
            layers.append( nn.ReLU(True) )
        layers.append( nn.Linear(layer_size[-1], bottleneck_dim) )
        self.encoder = nn.Sequential(*layers)

        # decoder
        layers = []
        layers.append( nn.Linear(bottleneck_dim, layer_size[-1]) )
        for i in range(1, num_layers):
            layers.append( nn.ReLU(True) )
            layers.append( nn.Linear(layer_size[-i], layer_size[-i-1]) )
        self.decoder = nn.Sequential(*layers)

        return


    def forward(self, x):
        """
        Forward pass through the autoencoder model
        """

        # flattening the input matrix
        batch = x.shape[0]
        shape = x.shape[1:]
        x = x.view([batch, -1])

        # forward through encoder
        for layer in self.encoder:
            x = layer(x)
        code = x

        # normalization if necessary
        if(self.norm==True):
            code = torch.sigmoid(code)
        x = code.clone()

        # forward through decoder
        for layer in self.decoder:
            x = layer(x)

        # reshaping the output
        y = x.view([batch, *shape])

        return y


#
