"""
Generating a txt file with the network architecture used in the experiment

Denoising_in_superresolution/src
@author: Angel Villar-Corrales
"""

import os
from tqdm import tqdm

import numpy as np
from matplotlib import pyplot as plt
import torch

import models as models
import lib.model_setup as model_setup
import lib.utils as utils
import lib.arguments as arguments


def main():
    """
    Main logic for getting the network architecture and saving it as txt file
    """

    # relevant paths and experiment data
    exp_path = arguments.get_directory_argument()
    exp_data = utils.load_configuration_file(exp_path)

    model = model_setup.setup_model(exp_data=exp_data, exp_path=exp_path)

    network_file = os.path.join(exp_path, "network_architecture.txt")
    with open(network_file, "w") as file:
        file.write(str(model))

    return


if __name__ == "__main__":

    os.system("clear")
    main()

#
