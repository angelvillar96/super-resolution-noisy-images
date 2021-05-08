"""
Methods for computing  and handling metrics, optimizers and other hyper parameters

Denoising_in_superresolution/src/lib
@author: Angel Villar-Corrales
"""

import os
import sys

import numpy as np
import torch


def get_loss_stats(loss_list, message=None):
    """
    Computes loss statistics given a list of loss values

    Args:
    -----
    loss_list: List
        List containing several loss values
    message: string
        Additional message to display
    """

    if(len(loss_list)==0):
        return

    loss_np = torch.stack(loss_list)
    avg_loss = torch.mean(loss_np)
    max_loss = torch.max(loss_np)
    min_loss = torch.min(loss_np)

    if(message is not None):
        print(message)
    print(f"Average loss: {avg_loss} -- Min Loss: {min_loss} -- Max Loss: {max_loss}")
    print("\n")

    return avg_loss


def mean_squared_error(original_img, resoluted_img):
    """
    Computing the mean squared error between original and resoluted images

    Args:
    -----
    original_img: batch of torch Tensors
        batch containing the input images
    resoluted_img: batch of torch Tensors
        batch containing the tensors at the output of the network

    Returns:
    mse: float
        mean squared error of the image
    """

    subs = original_img - resoluted_img
    mse = subs.pow(2).mean()

    return mse


def mean_absoulte_error(original_img, resoluted_img):
    """
    Computing the mean absolute error between original and resoluted images

    Args:
    -----
    original_img: batch of torch Tensors
        batch containing the input images
    resoluted_img: batch of torch Tensors
        batch containing the tensors at the output of the network

    Returns:
    mae: float
        mean absolute error of the image
    """

    subs = original_img - resoluted_img
    vals = torch.abs(subs)
    mae = torch.mean(vals)

    return mae


def psnr(original_img, resoluted_img):
    """
    Computing the peak signal to noise ration between original and resoluted images

    Args:x
    -----
    original_img: batch of torch Tensors
        batch containing the input images
    resoluted_img: batch of torch Tensors
        batch containing the tensors at the output of the network

    Returns:
    psnr: float
        peak to signal noise ratio (in dB)
    """

    # fisrt computing th emse
    resoluted_img = (resoluted_img * 255).round().clamp(0, 255) / 255
    subs = original_img - resoluted_img
    mse = subs.pow(2).mean([-3, -2, -1])

    # formular for psnr (https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio)
    max_val = 1.0
    psnr = 20*torch.log10(torch.tensor(max_val)) - 10*torch.log10(mse)
    idx = torch.where(psnr > 120)[0]  # avoid overflow
    psnr[idx] = 120
    psnr = psnr.mean()

    return psnr


def norm_img(img):
    """ """
    img = (img - 0.5) * 2
    return img

def denorm_img(img):
    """ """
    img = img * 0.5 + 0.5
    return img

#
