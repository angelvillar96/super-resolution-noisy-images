"""
Custom layers to include in neural networks nto already implemented in pyTorch

Denoising_in_superresolution/src/lib
@author: Angel Villar-Corrales
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _quadruple
import numpy as np


class MedianPooling(nn.Module):
    """
    Median pooling works as a max- or average-pooling layer but taking the median of the values
    within the kernel

    Args:
    -----
    kernel_size: integer or 2-tuple
        size of the median pooling kernel
    stride: integer or 2-tuple
        number of columns/rows that we skip between two consecutive kernels
    padding: int or 4-tuple
        number of rows/columns to pad
    same: boolean
        if True, keeps the same dimensions as the input. Overrides padding
    """

    def __init__(self, kernel_size=3, stride=1, padding=0, same=False):
        """
        Initializer of the Median Pooling layer
        """

        super(MedianPooling, self).__init__()
        self.kernel = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _quadruple(padding)
        self.same = same

        return


    def forward(self, x):
        """
        Computing the median for each kernel and stride

        Args:
        -----
        x: torch tensor
            input tensor where the medians are computed
        y: torch tensor
            pooled tensor
        """

        # padding input to desired shape
        pad_dim = self._pad_input(x)
        x = F.pad(x, pad_dim, mode='reflect')

        # breaking input into all possible size-k windows
        windows = x.unfold(dimension=2, size=self.kernel[0], step=1).unfold(dimension=3, size=self.kernel[1], step=1)

        shape = windows.size()[:4] + (-1,)
        y = windows.contiguous().view(shape).median(dim=-1)[0]

        return y


    def _pad_input(self, x):
        """
        Padding the input prior to the media operators given the parameters

        Args:
        -----
        x: torch tensor
            input tensor to be padded
        pad_dim: tuple
            elements to pad the input in each dimension (l, r, t, b)
        """

        if(self.same == True):
            _, _, row, col = x.shape
            x_pad = ( col * (self.stride[0] - 1) + self.kernel[0] - self.stride[0])
            # x_pad = (self.kernel[0]-1)  # this assumes stride=1
            l_pad = x_pad//2
            r_pad = x_pad - l_pad

            y_pad = ( row * (self.stride[1] - 1) + self.kernel[1] - self.stride[1])
            # y_pad = (self.kernel[1]-1)  # this assumes stride=1
            t_pad = y_pad//2
            b_pad = y_pad - t_pad

            pad_dim = (l_pad, r_pad, t_pad, b_pad)

        else:
            pad_dim = self.pad_dim

        return pad_dim



class WienerFilter(nn.Module):
    """
    Custom implementation of the Wiene Filter as a Pytorch NN Module based on Scipy's implementation
    """

    def __init__(self, K=3, device=None):
        """
        Initializer of the Wiener layer

        Args:
        -----
        K: integer or 2-d arraylke
            size of the wiener filter kernels
        """

        super(WienerFilter, self).__init__()
        self.kernel_size = _pair(K)
        self.device = device

        return


    def forward(self, img):
        """
        Forward pass through the denoiser wiener layer
        """

        local_mean, local_var, noise = self._estimate_params(img)

        selection_img = img - local_mean
        selection_img = selection_img * (1 - noise / local_var)
        selection_img = selection_img + local_mean
        # we approximate flat areas with the mean, high-freq areas are left untouched
        denoised_img = torch.where(local_var < noise, local_mean, selection_img)

        return denoised_img


    def _estimate_params(self, img):
        """
        Estimating local means and variances and noise power
        """

        pad_dim = self._pad_input(img)
        weight = torch.ones(1, img.shape[1], *self.kernel_size).float()
        weight = weight.to(self.device)

        # Estimate the local mean (low pass filter)
        local_mean = F.conv2d(
                input=img,
                weight=weight,
                padding=pad_dim
            ) / np.prod(list(weight.shape))

        # Estimate the local variance
        local_var =  F.conv2d(
                input=torch.pow(img,2),
                weight=weight,
                padding=pad_dim
            ) / np.prod(list(weight.shape)) - torch.pow(local_mean,2)

        # noise power
        noise = torch.mean(local_var.view(local_var.shape[0], local_var.shape[1], -1), axis=-1)
        noise = noise[:,np.newaxis, np.newaxis]

        return local_mean, local_var, noise


    def _pad_input(self, x):
        """
        Padding the input prior to the media operators given the parameters
        """
        _, _, row, col = x.shape
        x_pad = (self.kernel_size[0]-1)
        y_pad = (self.kernel_size[1]-1)
        pad_dim = (x_pad//2, y_pad//2)

        return pad_dim

#
