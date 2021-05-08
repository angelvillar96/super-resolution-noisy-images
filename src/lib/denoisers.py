"""
Classes and methods to setup denoisers as layers in the neural network
to then remove noise from the input images given the experiment parameters

Denoising_in_superresolution/src/lib
@author: Angel Villar-Corrales
"""

import os

import torch
import torch.nn as nn

from lib.layers import MedianPooling, WienerFilter
import lib.model_setup as model_setup


def introduce_denoising(model, exp_data, exp_path):
    """
    Creating the desired denoising method and combining it with the WDSR model

    Args:
    -----
    model: torch Module
        instanciated WDSR model
    exp_data: dictionary
        parameters corresponding to the given experiment

    Returns:
    --------
    model: torch Module
        WDSR model with integrated denoising method
    """

    denoising_method = exp_data["denoising"]["method"]
    denoising_position = exp_data["denoising"]["denoiser_type"]

    # loading correct denoiser
    if(denoising_method == ""):
        return model

    elif(denoising_method == "median_filter"):
        kernel_size = exp_data["denoising"]["kernel_size"]
        denoiser = MedianPooling(kernel_size=kernel_size, stride=1, same=True)

    elif(denoising_method == "wiener_filter"):
        kernel_size = exp_data["denoising"]["kernel_size"]
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        denoiser = WienerFilter(K=kernel_size, device=device)
        # making sure the kernel weights are fixed and do not update
        for param in denoiser.parameters():
            param.requires_grad = False

    elif(denoising_method == "autoencoder"):
        denoiser = model_setup.load_pretrained_autoencoder(exp_path=exp_path, exp_data=exp_data)

    else:
        print("ERROR! So far, only 'median filter' and 'guided filter' denoising approaches have been implemented ")
        exit()

    # integrating denoiser in the neural network
    # for the prenetwork case, we add the denoisier to the prenetwork block, which is applied
    # before the residual split
    if(denoising_position == "prenetwork"):
        model.prenetwork.add_module(f"{denoising_method}", denoiser)

    # for the innetwork case, we add the denoiser at the very beginning of the skip
    # connection, before the convolutional layer
    elif(denoising_position == "innetwork"):
        skip_layers =  list(model.skip)
        model.skip = nn.Sequential()
        model.skip.add_module(f"{denoising_method}", denoiser)
        for i, layer in enumerate(skip_layers):
            model.skip.add_module(f"{i}", layer)

    else:
        print("ERROR! So far, only 'prenetwork' and 'innetwork' modes have been implemented ")
        exit()

    return model


#
