"""
Generating a subset if images given the experiment parameters (dataset, noise, downsampling)
and saving them for inspection purposes

Denoising_in_superresolution/src
@author: Angel Villar-Corrales
"""

import os
from tqdm import tqdm

import numpy as np
from matplotlib import pyplot as plt
import torch

import data
import models as models
import lib.model_setup as model_setup
import lib.utils as utils
import lib.visualizations as visualizations
import lib.metrics as metrics
import lib.arguments as arguments
from config import CONFIG



def main():
    """
    Main logic for the subset generator
    """

    # relevant paths and experiment data
    exp_path = arguments.get_directory_argument()
    plots_path = os.path.join(exp_path, "plots")
    exp_data = utils.load_configuration_file(exp_path)
    dataset_name = exp_data["dataset"]["dataset_name"]

    # loading a batch of the train set
    _, train_loader, _, _, _ = model_setup.load_dataset(exp_data)
    # _, train_loader, _, _, _ = model_setup.load_dataset(exp_data, noisy=True)

    hr_imgs, lr_imgs, labels = next(iter(train_loader))

    # creating figure with first 5 images in the batch
    plt.figure()
    for i in range(5):

        plt.subplot(2,5,i+1)
        hr_img = np.array((hr_imgs[i,:]*0.5)+0.5).transpose(1,2,0).squeeze()
        hr_img = np.clip(hr_img, a_min=0, a_max=1)
        plt.imshow(hr_img)
        plt.title(f"Label: {labels[i]}")

        plt.subplot(2,5,i+6)
        lr_img = np.array((lr_imgs[i,:]*0.5)+0.5).transpose(1,2,0).squeeze()
        lr_img = np.clip(lr_img, a_min=0, a_max=1)
        plt.imshow(lr_img)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_path, "subset_images.png"))

    return


if __name__ == "__main__":

    os.system("clear")
    main()

#
