"""
Argument parsers for different files

Denoising_in_superresolution/src/lib
@author: Angel Villar-Corrales
"""

import os
import argparse
from config import CONFIG


def process_create_experiment_arguments():
    """
    Processing command line arguments for 01_create_experiment script
    """

    # defining command line arguments
    parser = argparse.ArgumentParser()

    # general arguments
    parser.add_argument("-d", "--exp_directory", help="Directory where the experiment folder will be created", default="test_dir")

    # dataset parameters
    parser.add_argument('--dataset_name', help="Dataset to take the images from [mnist, svhn, div2k]", default="svhn")

    # denoising parameters
    parser.add_argument('--denoiser', help="Type of denoising approach ['', 'median_filter',\
        'wiener_filter', 'autoencoder']", default="")
    parser.add_argument('--denoiser_type', help="Where the denoiser is applied ['', 'prenetwork',\
     'innetwork']", default="")
    parser.add_argument('--kernel_size', help="Size of the median pooling or wiener filter\
        kernel. Must be integer, so far only square kernels are allowed", default="5")
    parser.add_argument('--bottleneck', help="Dimensionality of the blottleneck code for the case of\
        denoising autoencoders", default="128")

    # agumentation parameters
    parser.add_argument('--rotation', help="Binary flag for applying rotation augmentation", default="False")
    parser.add_argument('--translation', help="Binary flag for applying tranlation augmentation", default="False")

    # training parameters
    parser.add_argument('--num_epochs', help="Number of epochs to train for", default="100")
    parser.add_argument('--batch_size', help="Number of examples in each batch", default="12")
    parser.add_argument('--patches_per_image', help="Number of patches to sample per image", default="10")
    parser.add_argument('--optimizer', help="Optimizer used to update the weights ['ADAM']", default="ADAM")
    parser.add_argument('--loss_function', help="Loss function used ['mae', 'mse']", default="mae")
    parser.add_argument('--learning_rate', help="Learning rate", default="3e-4")
    parser.add_argument('--lr_decay', help='Factor by which the learning rate will be decreased during decay', default='0.1')
    parser.add_argument('--patience', help='Number of epochs in which the loss does not decrease before changing lr', default='8')
    parser.add_argument('--validation_size', help="Size of the validation set [0,0.5]", default="0.2")
    parser.add_argument('--save_frequency', help="Number of epochs after which we save a checkpoint", default="10")

    # model parameters
    parser.add_argument('--model_name', help="Name of the model to use ['wdsr_a', 'wdsr_b']", default="wdsr_a")
    parser.add_argument('--num_filters', help="Number of filters in the conv layers of the residual blocks", default="32")
    parser.add_argument('--num_res_blocks', help="Number of residual blocks", default="16")
    parser.add_argument('--num_block_features', help="Number of blocks features", default="256")
    parser.add_argument('--res_scale', help="Weight scale of the residual", default="0.1")
    parser.add_argument('--r_mean', help='Mean of R Channel', default=0.5)
    parser.add_argument('--g_mean', help='Mean of G channel', default=0.5)
    parser.add_argument('--b_mean', help='Mean of B channel', default=0.5)

    # corruption parameters
    parser.add_argument('--noise', help="Type of noise to be used to corrupt the images\
                                        ['', 'gaussian', 'poisson', 'speckle', 'salt_pepper']", default="")
    parser.add_argument('--std', help="Standard deviation of the noise", default=0)
    parser.add_argument('--downscaling', help="Factor by which the images will be downscaled and the upsampled", default=1)
    args = parser.parse_args()


    # formating arguments
    args.kernel_size = int(args.kernel_size)
    args.bottleneck = int(args.bottleneck)

    args.num_epochs = int(args.num_epochs)
    args.batch_size = int(args.batch_size)
    args.patches_per_image = int(args.patches_per_image)
    args.learning_rate = float(args.learning_rate)
    args.lr_decay = float(args.lr_decay)
    args.patience = int(args.patience)
    args.validation_size = float(args.validation_size)
    args.save_frequency = int(args.save_frequency)

    args.num_filters = int(args.num_filters)
    args.num_res_blocks = int(args.num_res_blocks)
    args.num_block_features = int(args.num_block_features)
    args.res_scale = float(args.res_scale)

    args.std = float(args.std)
    args.downscaling = int(args.downscaling)

    # ensuring only known values go through
    assert args.dataset_name in ["mnist", "svhn", "div2k"]

    assert args.denoiser in ["", "median_filter", "wiener_filter", "autoencoder"]
    assert args.denoiser_type in ['', 'prenetwork', 'innetwork']
    assert args.kernel_size > 0
    assert args.bottleneck > 0

    assert args.optimizer in ["ADAM"]
    assert args.loss_function in ["mse", "mae"]
    assert args.learning_rate > 0
    assert args.batch_size > 0
    assert args.patches_per_image > 0
    assert args.validation_size > 0 and args.validation_size < 0.5

    assert args.model_name in ["wdsr_a"]
    assert args.num_filters > 0
    assert args.num_res_blocks > 0

    assert args.noise in ["", "gaussian", "poisson", "speckle", "salt_pepper"]
    assert args.downscaling >= 1

    return args


def get_directory_argument(get_epoch=False, generalization=False, patches=False):
    """
    Reading the directory passed as an argument

    Args:
    -----
    get_epoch: boolean
        if True, an argument is read as parameter in addition to the directory
    generalization: boolean
        if True, reads arguments for evaluating generalization => noise and power
    """

    # defining command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--exp_directory", help="Path to the experiment directory")
    parser.add_argument("-e", "--epoch", help="Epoch of the checkpoint to process", default=-1)
    parser.add_argument('--noise', help="Type of noise to corrupt the test-set images\
                                        ['', 'gaussian', 'poisson', 'speckle', 'salt_pepper']", default="")
    parser.add_argument('--std', help="Standard deviation of the noise", default=0)
    args = parser.parse_args()

    exp_directory = args.exp_directory
    epoch = args.epoch
    noise = args.noise
    std = float(args.std)
    was_relative = False
    assert noise in ["", "gaussian", "poisson", "speckle", "salt_pepper"]

    # for relative paths
    root_path = os.path.dirname(os.getcwd())
    exp_path = os.path.join(root_path, CONFIG["paths"]["dir_experiments"])
    if(exp_path not in exp_directory):
        was_relative = True
        exp_directory = os.path.join(exp_path, exp_directory)

    # meking sure experiment directory exists
    if(not os.path.exists(exp_directory)):
        print(f"ERROR! Experiment directorty {exp_directory} does not exist...")
        print(f"     The given path was: {args.exp_directory}")
        if(was_relative):
            print(f"     It was a relative path. The absolute would be: {exp_directory}")
        print("\n\n")
        exit()

    if(get_epoch==True):
        return exp_directory, epoch
    if(generalization==True):
        return exp_directory, noise, std

    return exp_directory


def get_checkpoint_argument():
    """
    Reading the checkpoint epoch passed as argument, if any
    """

    # defining command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", help="Number of epochs corresponding to the checkpoint to load", default=-1)
    args = parser.parse_args()

    checkpoint = int(args.checkpoint)

    return checkpoint


def get_evaluation_argument(method=False):
    """
    Reading a path to a json file for evaluation purposes
    """

    # defining command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--eval_file", help="Path to the evaluation file")
    parser.add_argument('-m', '--method', help="Method to visually compare ['wiener_filter', \
                        'median_filter', 'autoencoder', 'median_filter wiener_filter']", default="wiener_filter")
    args = parser.parse_args()

    assert args.method in ["median_filter", "wiener_filter", "autoencoder", "median_filter wiener_filter"]

    eval_file = args.eval_file
    eval_files = []

    # for the case when we give several evaluation files separated by spaces
    if(" " in eval_file):
        eval_file = eval_file.split(" ")
    else:
        eval_file = [eval_file]

    for file in eval_file:
        # for relative paths
        root_path = os.path.dirname(os.getcwd())
        exp_path = os.path.join(root_path, CONFIG["paths"]["dir_eval"], "eval_checkpoint_dicts")
        if(exp_path not in file):
            was_relative = True
            file = os.path.join(exp_path, file)

        # meking sure experiment directory exists
        if(not os.path.exists(file)):
            print(f"ERROR! Evaluation file {file} does not exist...")
            print(f"     The given path was: {args.eval_file}")
            if(was_relative):
                print(f"     It was a relative path. The absolute would be: {file}")
            print("\n\n")
            exit()

        eval_files.append(file)

    if(method):
        return eval_files, args.method

    return eval_files


def image_comparison_arguments(method=False):
    """
    Arguments to create a figure for method comparison
    """

    # defining command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset_name', help="Dataset to take the images from [mnist, svhn, div2k]", default="div2k")
    parser.add_argument('-n', '--noise', help="Type of noise used for corruption ['gaussian',\
        'salt_pepper', 'poisson', 'speckle']", default="gaussian")
    parser.add_argument('-s', '--sigma', help="Standard deviation of the noise", default=0.2)
    parser.add_argument('-m', '--method', help="Method to visually compare ['wiener_filter', \
        'median_filter', 'autoencoder']", default="wiener_filter")
    parser.add_argument('--generalization', help="Boolean. If true, the script creates an image\
        comparison of generalization properties", default="False")
    args = parser.parse_args()

    assert args.noise in ["gaussian", "salt_pepper", "poisson", "speckle"]
    assert args.dataset_name in ["mnist", "svhn", "div2k"]
    assert args.method in ["median_filter", "wiener_filter", "autoencoder"]
    assert args.generalization in ["True", "False"]
    generalization = (args.generalization=="True")

    if(method):
        return args.dataset_name, args.noise, float(args.sigma), args.method

    return args.dataset_name, args.noise, float(args.sigma), generalization

#
