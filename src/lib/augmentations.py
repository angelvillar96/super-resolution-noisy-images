"""
Methods to augment the input data in several ways:
    - Rotations
    - Shifts
    - Paddings

Denoising_in_superresolution/src/lib
@author: Angel Villar-Corrales
"""

import torch
import torch.nn.functional as F
from torchvision import transforms


def get_augmentations(dataset_name, mode="train", augmentations={}):
    """
    Creating a list of transformations to be applied online when drawing batches

    Args:
    -----
    dataset_name: string
        name of the dataset to augment
    mode: string
        type of dataset to augment ['train', 'test']
    augmentations: dictionary
        dictionary with flags indicating which augmentations to apply
    """

    transform_list = []

    # padding to have 32x32 images
    if(dataset_name == "mnist"):
        pixels_to_pad = (2,2,2,2)
        padding = transforms.Pad(pixels_to_pad, fill=0, padding_mode='constant')
        transform_list.append(padding)

    if("rotation" in list(augmentations.keys()) and augmentations["rotation"]=="True"):
        rotating = transforms.RandomRotation(degrees=30)
        transform_list.append(rotating)

    if("translation" in list(augmentations.keys()) and augmentations["translation"]=="True"):
        translating = transforms.RandomAffine(degrees=0, translate=(1/4,1/4))  # translates in the range [-8, 8] in both axis
        transform_list.append(translating)

    transform_list.append(transforms.ToTensor())
    # converting the list of transforms into a Torchvision object
    transforms_object = transforms.Compose(transform_list)

    return transforms_object


#
