"""
Loading the model with the best validation performance and using it to denoise
and superresolute a bunch of images from the test set

Denoising_in_superresolution/src
@author: Angel Villar-Corrales
"""

import os
import json
from argparse import Namespace
from tqdm import tqdm

import numpy as np
from matplotlib import pyplot as plt
import torch

import models as models
import data
import lib.utils as utils
import lib.visualizations as visualizations
import lib.metrics as metrics
import lib.arguments as arguments
import lib.model_setup as model_setup
from config import CONFIG


def load_pretrained_model(model_path, exp_data, exp_path):
    """
    Creating a model and loading the pretrained network parameters from the saved
    state dictionary. Setting the model to use a GPU

    Args:
    -----
    model_path: string
        path to the file with the pretrained model weights
    exp_path: dictionary
        parameters of the current experiment
    exp_path: string
        path to the experiment directory

    Returns:
    --------
    model: torch Module
        pretrained model with set pretrained weights and set to cpu or gpu
    device: torch device
        hardware device to be used: cpu or gpu
    """

    # creating model architecture
    # setting up the device
    torch.backends.cudnn.fastest = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # initializing the model and loading the state dicitionary
    model = model_setup.setup_model(exp_data=exp_data, exp_path=exp_path, debug=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)

    return model, device


def main():
    """
    Main orquestrator: sets up the model, loads the pretrained weights, loads dataset, computes
    inference and saves the data
    """

    # processing arguments and relevant paths
    exp_directory, epoch = arguments.get_directory_argument(get_epoch=True)
    exp_data = utils.load_configuration_file(exp_directory)
    exp_data["training"]["batch_size"] = 4
    model_directory = os.path.join(exp_directory, "models")
    model_name = f"model_epoch_{epoch}"
    model_path = os.path.join(model_directory, model_name)
    plots_path = os.path.join(exp_directory, "plots", "test")
    utils.create_directory(plots_path )

    # if model given epoch does not exist, we load the model with the best validation loss
    if(not os.path.exists(model_path)):
        model_name = utils.get_best_model_name(exp_directory)
        model_path = os.path.join(model_directory, model_name)

    # loading test set
    dataset, _, _, test_loader, num_channels = model_setup.load_dataset(exp_data, shuffle_test=True,
                                                                        train=False, test=True)

    # setting up the model
    model, device = load_pretrained_model(model_path, exp_data, exp_directory)

    # processing a batch of images
    model.eval()
    test_mae, test_mse, test_psnr = [], [], []
    with torch.no_grad():
        for i in range(3):
            (hr_imgs, lr_imgs, labels) = next(iter(test_loader))
            hr_imgs = hr_imgs.to(device).float()
            lr_imgs = lr_imgs.to(device).float()

            # recovered_images, out_prenetwork, out_innetwork = model(lr_imgs)
            # pretrained model expects input in range [-0.5, 0.5] and we were using [-1,1]
            recovered_images = model(lr_imgs * 0.5) * 2

            # setting images to the range [0,1]
            hr_imgs, lr_imgs = metrics.denorm_img(hr_imgs), metrics.denorm_img(lr_imgs)
            recovered_images = metrics.denorm_img(recovered_images)

            num_imgs = len(os.listdir(plots_path))
            test_mae.append(metrics.mean_absoulte_error(hr_imgs, recovered_images))
            test_mse.append(metrics.mean_squared_error(hr_imgs, recovered_images))
            cur_psnr = metrics.psnr(hr_imgs, recovered_images)
            test_psnr.append(cur_psnr)

            dataset_name = exp_data["dataset"]["dataset_name"]
            if(dataset_name == "div2k"):
                visualizations.display_images_one_row(
                        hr_imgs, lr_imgs, recovered_images,
                        savepath=os.path.join(plots_path, f"test_figures_{num_imgs}"),
                        dataset_name=dataset_name, psnr=cur_psnr,
                        downscaling=exp_data["corruption"]["downsampling"]["factor"])

            else:
                visualizations.display_images(
                        hr_imgs, lr_imgs, recovered_images,
                        savepath=os.path.join(plots_path, f"test_figures_{num_imgs}"),
                        dataset_name=dataset_name)


    test_mae = torch.stack(test_mae).mean()
    test_mse = torch.stack(test_mse).mean()
    test_psnr = torch.stack(test_psnr).mean()

    # computing the evaluation metrics
    print(f"Batch MAE: {test_mae}")
    print(f"Batch MSE: {test_mse}")
    print(f"Batch PSNR: {test_psnr}")


    return


if __name__ == "__main__":

    os.system("clear")
    main()
