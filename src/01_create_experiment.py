"""
Creating an experiment, setting up the network and training the model

Denoising_in_superresolution/src
@author: Angel Villar-Corrales
"""

import os
from config import CONFIG

import lib.utils as utils
import lib.arguments as arguments


def create_experiement():
    """
    Creating an experiment: creating directory, creating config file, ....
    """

    # processig command line arguments
    args = arguments.process_create_experiment_arguments()

    # experiment name
    exp_name = f"model_{args.model_name}_epochs_{args.num_epochs}_dataset_{args.dataset_name}_noise_{args.noise}"\
               f"_std_{args.std}_{utils.timestamp()}"
    # exp_name = f"denoiser_{args.denoiser}_type_{args.denoiser_type}_dataset_{args.dataset_name}_std_{args.std}"\
               # f"_{utils.timestamp()}"

    root_path = os.path.dirname(os.getcwd())
    exp_directory = os.path.join(root_path, CONFIG["paths"]["dir_experiments"], args.exp_directory)
    exp_path = os.path.join(exp_directory, exp_name)

    # creating experiment directory and subdirs
    utils.create_directory(exp_path)
    utils.create_directory(exp_path, "models")
    utils.create_directory(exp_path, "plots")

    # creating experiment config file
    utils.create_configuration_file(exp_path=exp_path, config=CONFIG, args=args)

    return


if __name__ == "__main__":

    os.system("clear")
    create_experiement()

#
