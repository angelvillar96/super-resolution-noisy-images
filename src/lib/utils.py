"""
Auxiliary methods to handle files, process experiment configuration file and other
functions that do not belong to any particular class

Denoising_in_superresolution/src/lib
@author: Angel Villar-Corrales
"""

import os
import random
import sys
import json
import datetime

import numpy as np
from matplotlib import pyplot as plt
import torch
sys.path.append("..")
from config import CONFIG


def set_random_seed(random_seed=None):
    """
    Using random seed for numpy and torch
    """
    if(random_seed is None):
        random_seed = CONFIG["random_seed"]
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    return


def create_configuration_file(exp_path, config, args):
    """
    Creating a configuration file for an experiment, including all hyperparameters used
    for the dataset preprocessing and training of the neural network

    Args:
    -----
    exp_path: string
        path to the experiment folder
    config: dictionary
        dictionary with the global configuration parameters: paths, ...
    args: Namespace
        dictionary-like object with the data corresponding to the command line arguments

    Returns:
    --------
    exp_data: dictionary
        dictionary containing all the parameters is the created experiments
    """

    exp_data = {}
    exp_data["exp_created"] = timestamp()
    exp_data["last_modified"] = timestamp()
    exp_data["random_seed"] = config["random_seed"]

    # dataset parameters
    exp_data["dataset"] = {}
    exp_data["dataset"]["dataset_name"] = args.dataset_name
    exp_data["dataset"]["validation_size"] = args.validation_size
    exp_data["dataset"]["patches_per_img"] = args.patches_per_image

    # denoiser parameteres
    exp_data["denoising"] = {}
    exp_data["denoising"]["method"] = args.denoiser
    exp_data["denoising"]["denoiser_type"] = args.denoiser_type
    exp_data["denoising"]["kernel_size"] = args.kernel_size
    exp_data["denoising"]["bottleneck"] = args.bottleneck

    # augmentation parameters
    exp_data["augmentations"] = {}
    exp_data["augmentations"]["rotation"] = args.rotation
    exp_data["augmentations"]["translation"] = args.translation

    # corruption parameters
    exp_data["corruption"] = {}
    exp_data["corruption"]["noise"] = {}
    exp_data["corruption"]["noise"]["noise_type"] = args.noise
    exp_data["corruption"]["noise"]["std"] = args.std
    exp_data["corruption"]["compression"] = {}
    exp_data["corruption"]["downsampling"] = {}
    exp_data["corruption"]["downsampling"]["factor"] = args.downscaling

    # network parameters
    exp_data["model"] = {}
    exp_data["model"]["model_name"] = args.model_name
    exp_data["model"]["res_scale"] = args.res_scale
    exp_data["model"]["num_res_blocks"] = args.num_res_blocks
    exp_data["model"]["num_filters"] = args.num_filters
    exp_data["model"]["num_block_features"] = args.num_block_features
    exp_data["model"]["r_mean"] = args.r_mean
    exp_data["model"]["g_mean"] = args.g_mean
    exp_data["model"]["b_mean"] = args.b_mean

    # training parameters
    exp_data["training"] = {}
    exp_data["training"]["batch_size"] = args.batch_size
    exp_data["training"]["optimizer"] = args.optimizer
    exp_data["training"]["loss_function"] = args.loss_function
    exp_data["training"]["learning_rate"] = args.learning_rate
    exp_data["training"]["lr_decay"] = args.lr_decay
    exp_data["training"]["patience"] = args.patience
    exp_data["training"]["epochs"] = args.num_epochs
    exp_data["training"]["save_frequency"] = args.save_frequency

    # creating file and saving it in the experiment folder
    exp_data_file = os.path.join(exp_path, "experiment_parameters.json")
    with open(exp_data_file, "w") as file:
        json.dump(exp_data, file)

    return exp_data


def load_configuration_file(path):
    """
    Loading the experiment hyperparameters as a dictionary

    Args:
    -----
    path: string
        path to the experiment directory

    Returns:
    --------
    exp_data: dictionary
        dictionary containing all experiment hyperparameters
    """

    exp_data_file = os.path.join(path, "experiment_parameters.json")

    if(not os.path.exists(exp_data_file)):
        return None

    with open(exp_data_file, "r") as file:
        exp_data = json.load(file)

    return exp_data


def create_directory(path, name=None):
    """
    Checking if a directory already exists and creating it if necessary

    Args:
    -----
    path: string
        path/name where the directory will be created
    name: string
        name fo the directory to be created
    """

    if(name is not None):
        path = os.path.join(path, name)

    if(not os.path.exists(path)):
        os.makedirs(path)

    return


def create_train_logs(path):
    """
    Creating a dictionary/json file with logs that are updated every epoch during training

    Args:
    -----
    path: string
        path/name where the json file will be stored

    Returns:
    --------
    train_logs: dictionary
        dict with the initialized structure of the training logs
    """

    train_logs = {}
    train_logs["training_start"] = timestamp()
    train_logs["last_modified"] = timestamp()

    # loss information
    train_logs["loss"] = {}
    train_logs["loss"]["train"] = []
    train_logs["loss"]["valid"] = []

    # metrics information
    train_logs["metrics"] = {}
    train_logs["metrics"]["mae"] = {}
    train_logs["metrics"]["mae"]["train"] = []
    train_logs["metrics"]["mae"]["valid"] = []
    train_logs["metrics"]["mse"] = {}
    train_logs["metrics"]["mse"]["train"] = []
    train_logs["metrics"]["mse"]["valid"] = []
    train_logs["metrics"]["psnr"] = {}
    train_logs["metrics"]["psnr"]["train"] = []
    train_logs["metrics"]["psnr"]["valid"] = []

    # creating file and saving it in the experiment folder
    train_logs_file = os.path.join(path, "training_logs.json")
    with open(train_logs_file, "w") as file:
        json.dump(train_logs, file)

    return train_logs


def load_train_logs(path):
    """
    Loading train logs
    """

    train_logs_file = os.path.join(path, "training_logs.json")

    if(not os.path.exists(train_logs_file)):
        return None

    with open(train_logs_file) as file:
        train_logs = json.load(file)

    return train_logs


def create_generalization_logs(path):
    """
    Creating a json file to save generalization evaluations
    """

    gen_logs = {}
    gen_logs["training_start"] = timestamp()
    gen_logs["last_modified"] = timestamp()

    # creating file and saving it in the experiment folder
    gen_logs_file = os.path.join(path, "generalization_logs.json")
    with open(gen_logs_file, "w") as file:
        json.dump(gen_logs, file)

    return gen_logs


def load_generalization_logs(path):
    """
    Loading generalization logs
    """

    gen_logs_file = os.path.join(path, "generalization_logs.json")

    if(not os.path.exists(gen_logs_file)):
        return None

    with open(gen_logs_file) as file:
        gen_logs = json.load(file)

    return gen_logs


def load_autoencoder_logs(path):
    """
    Loading autoencoder train logs
    """

    train_logs_file = os.path.join(path, "autoencoder_logs.json")
    with open(train_logs_file) as file:
        train_logs = json.load(file)

    return train_logs


def update_logs(path, plot_path, train_loss, valid_loss, train_mae, valid_mae,
                train_mse, valid_mse, train_psnr, valid_psnr):
    """
    Updating training logs
    """

    train_logs_file = os.path.join(path, "training_logs.json")
    train_logs = load_train_logs(path)

    train_logs["last_modified"] = timestamp()

    # updating loss
    train_logs["loss"]["train"].append(float(train_loss))
    train_logs["loss"]["valid"].append(float(valid_loss))

    # updating metrics
    train_logs["metrics"]["mae"]["train"].append(float(train_mae))
    train_logs["metrics"]["mae"]["valid"].append(float(valid_mae))
    train_logs["metrics"]["mse"]["train"].append(float(train_mse))
    train_logs["metrics"]["mse"]["valid"].append(float(valid_mse))
    train_logs["metrics"]["psnr"]["train"].append(float(train_psnr))
    train_logs["metrics"]["psnr"]["valid"].append(float(valid_psnr))

    # saving figure with loss over epochs
    plt.figure()
    epochs = np.arange(len(train_logs["loss"]["train"]))
    plt.plot(epochs[1:], train_logs["loss"]["train"][1:], label="Train Loss")
    plt.plot(epochs[1:], train_logs["loss"]["valid"][1:], label="Valid Loss")
    plt.legend(loc="best")
    fig_path = os.path.join(plot_path, "loss_landscape.png")
    plt.savefig(fig_path)

    # saving updated logs
    with open(train_logs_file, "w") as file:
        json.dump(train_logs, file)

    return


def timestamp():
    """
    Obtaining the current timestamp in an human-readable way

    Returns:
    --------
    timestamp: string
        current timestamp in format hh-mm-ss
    """

    timestamp = str(datetime.datetime.now()).split('.')[0].replace(' ', '_').replace(':', '-')

    return timestamp


def get_best_model_name(exp_path, autoencoder=False):
    """
    Checking the training logs to find at which epoch the validation loss was the smallest

    Args:
    -----
    exp_path: string
        root path of the experiment directory
    autoencoder: boolean
        if true, returns the name of the best autoencoder

    Returns:
    --------
    model_name: string
        name of the best model
    """

    # relevant path
    if(autoencoder==False):
        models_path = os.path.join(exp_path, "models")
        logs_path = os.path.join(exp_path, "training_logs.json")#
    else:
        models_path = os.path.join(exp_path, "models")
        models_path = os.path.join(models_path, "autoencoder")
        logs_path = os.path.join(exp_path, "autoencoder_logs.json")

    # loading autoencoder_logs
    if(not os.path.exists(logs_path)):
        print("ERROR! Training logs do not exist...")
        exit()
    with open(logs_path) as file:
        logs = json.load(file)

    if(autoencoder == False):
        valid_loss = logs["loss"]["valid"]
    else:
        valid_loss = logs["valid_loss"]

    # obtaining the model with the smallest validation löo
    models = os.listdir(models_path)
    min_epoch = 0
    min_loss = 1e8
    model_name = ""
    for model in models:
        if("model_" not in model and "autoencoder_" not in model):
            continue
        if(model.split("_")[-1] == "trained"):
            epoch = -1
        else:
            epoch = int(model.split("_")[-1])
        loss = valid_loss[epoch]
        if(loss < min_loss):
            min_loss = loss
            min_epoch = epoch
            model_name = model

    print(f"Loading model: {model_name} with validation loss {min_loss}")

    return model_name


def load_untrained(exp_path):
    """
    Loading an untrained model to compare as a baseline
    """

    models_path = os.path.join(exp_path, "models")
    models = os.listdir(models_path)
    for model in models:
        if(model == "model_epoch_0"):
            model_name = model
            break

    return model_name

#
