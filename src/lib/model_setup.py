"""
Common methods necessary for loading data and setting up the model

Denoising_in_superresolution/src/lib
@author: Angel Villar-Corrales
"""

import os
import sys
from argparse import Namespace
import torch

sys.path.append("..")

import data
import models as models
import lib.denoisers as denoisers
import lib.utils as utils
from config import CONFIG


def load_dataset(exp_data, noisy=False, shuffle_test=False, train=True, test=True, test_patch=False):
    """
    Loading dataset and computing data loaders

    Args:
    -----
    exp_data: dictionary
        parameters corresponding to the given experiment
    noisy: boolean
        If true, loads dataset to train a denoiser. Otherwise, loads the superresolution dataset

    Returns:
    --------
    dataset: Torch Dataset
        current dataset object
    train_loader: DataLoader
        data loader for the training set
    valid_loader: DataLoader
        data loader for the validation set
    test_loader: DataLoader
        data loader for the test set
    num_channels: integer
        number of color channels for the images in the given dataset
    """

    dataset = data.Data(dataset_name=exp_data["dataset"]["dataset_name"], config=CONFIG,
                        test=test, train=train, noise=exp_data["corruption"]["noise"]["noise_type"],
                        std=exp_data["corruption"]["noise"]["std"],
                        downscaling=exp_data["corruption"]["downsampling"]["factor"],
                        valid_size=exp_data["dataset"]["validation_size"],
                        augmentations=exp_data["augmentations"], noisy=noisy,
                        patches_per_img=exp_data["dataset"]["patches_per_img"])

    train_loader, valid_loader = dataset.get_train_validation_loaders(batch_size=exp_data["training"]["batch_size"])
    test_loader = dataset.get_data_loader(batch_size=exp_data["training"]["batch_size"], set="test",
                                          shuffle=shuffle_test, test_patch=test_patch)

    if(exp_data["dataset"]["dataset_name"]=="mnist"):
        num_channels = 1
    else:
        num_channels = 3

    return dataset, train_loader, valid_loader, test_loader, num_channels


def load_generalization_dataset(exp_data, noise, std, shuffle_test=False, test_patch=False,
                                generalization=False, savefig=False):
    """
    Loading dataset and computing data loaders

    Args:
    -----
    exp_data: dictionary
        parameters corresponding to the given experiment
    noise: string
        type of noise used to corrup the test images
    std: float
        power of the corruption noise
    shuffle_test: boolean
        If true, test set images are sampled randomly

    Returns:
    --------
    dataset: Torch Dataset
        current dataset object
    test_loader: DataLoader
        data loader for the test set
    num_channels: integer
        number of color channels for the images in the given dataset
    """

    dataset = data.Data(dataset_name=exp_data["dataset"]["dataset_name"], config=CONFIG,
                        test=True, train=False, noise=noise, std=std,
                        downscaling=exp_data["corruption"]["downsampling"]["factor"],
                        augmentations=exp_data["augmentations"], noisy=False,
                        patches_per_img=exp_data["dataset"]["patches_per_img"])

    test_loader = dataset.get_data_loader(batch_size=exp_data["training"]["batch_size"], set="test",
                                          shuffle=shuffle_test, test_patch=test_patch,
                                          generalization=generalization, savefig=savefig)

    num_channels = 1 if(exp_data["dataset"]["dataset_name"]=="mnist") else 3

    return dataset, test_loader, num_channels


def setup_model(exp_data, dataset=None, exp_path="", debug=False):
    """
    Initializing the model given the experiment parameters

    Args:
    -----
    exp_data: dictionary
        parameters corresponding to the given experiment
    dataset: Dataset object
        train or test set object. Necessary to sample guiding images for the guided filters
    exp_path: string
        path to a experiment directory. It is necessary to load a pretrained denoiser autoencoder

    Returns:
    --------
    model: Torch Model
        model to be trainined or loaded
    """

    if(exp_data["dataset"]["dataset_name"]=="mnist"):
        num_channels = 1
    else:
        num_channels = 3

    # Initializing the model
    model_args = {}
    model_args["scale"] = exp_data["corruption"]["downsampling"]["factor"]
    model_args["res_scale"] = exp_data["model"]["res_scale"]
    model_args["n_resblocks"] = exp_data["model"]["num_res_blocks"]
    model_args["n_feats"] = exp_data["model"]["num_filters"]
    model_args["block_feats"] = exp_data["model"]["num_block_features"]
    model_args["n_colors"] = num_channels
    model_args["r_mean"] = exp_data["model"]["r_mean"]
    model_args["g_mean"] = exp_data["model"]["g_mean"]
    model_args["b_mean"] = exp_data["model"]["b_mean"]
    model_args["r_mean"] = 0
    model_args["g_mean"] = 0
    model_args["b_mean"] = 0
    model_args["n_colors"] = num_channels
    model_args["debug"] = debug

    if(exp_data["model"]["model_name"]=="wdsr_a"):
        model = models.WDSR(Namespace(**model_args))
        # model = models.WDSR_A(Namespace(**model_args))  # old version
    elif(exp_data["model"]["model_name"]=="wdsr_b"):
        # model = models.WDSR_B(Namespace(**model_args))
        raise NotImplementedError(f"Only 'WDSR_A' model is allowed")
    else:
        raise NotImplementedError(f"Only 'WDSR_A' model is allowed")

    # loading weight from the pretrained WDSR model
    downscaling = exp_data["corruption"]["downsampling"]["factor"]
    root_path = os.path.dirname(os.getcwd())
    weights_path = os.path.join( root_path, CONFIG["paths"]["weights_path"], f"wdsr_x{downscaling}.pth")

    if torch.cuda.is_available():
        print("CUDA Avaliable")
        print(f"Found {torch.cuda.device_count()} devices")
        checkpoint = torch.load(weights_path, map_location=torch.device("cuda"))
    else:
        print("CUDA NOT Avaliable")
        checkpoint = torch.load(weights_path, map_location=torch.device("cpu"))
        model.load_state_dict(checkpoint['model_state_dict'])

    # introduce denoiser into the model
    model = denoisers.introduce_denoising(model=model, exp_data=exp_data, exp_path=exp_path)

    return model


def setup_autoencoder(exp_data, dataset=None):
    """
    Setting up an autoencoder for denoising purposes

    Args:
    -----
    exp_data: dictionary
        parameters corresponding to the given experiment

    Returns:
    --------
    autoencoder: torch Model
        instance of an autoencoder ready to train
    """

    bottleneck_dim = exp_data["denoising"]["bottleneck"]
    if(exp_data["dataset"]["dataset_name"]=="mnist"):
        num_channels = 1
    else:
        num_channels = 3
    width = 32//exp_data["corruption"]["downsampling"]["factor"]
    size = (num_channels, width, width)

    # autoencoder = models.Autoencoder(bottleneck_dim, size=size, layer_size=[256, 128])
    autoencoder = models.ConvAutoencoder(num_layers=4, num_kernels=[64, 128, 256], kernel_size=5)
    # autoencoder = models.ConvAutoencoder(num_layers=5, num_kernels=[64, 128, 256, 256], kernel_size=5)

    return autoencoder


def load_pretrained_autoencoder(exp_path, exp_data):
    """
    Loading a pretrained autoencoder to use as denoiser

    Args:
    -----
    exp_path: string
        root path of the experiment directory
    exp_data: dictionary
        parameters corresponding to the given experiment

    Returns:
    --------
    autoencoder: torch Model
        best pretrained autoencoder model with its parameters frozen to avoid updates
    """

    # relevant paths
    models_path = os.path.join(exp_path, "models")
    autoencoders_path = os.path.join(models_path, "autoencoder")

    # getting the name of the best autoencoder
    model_name = utils.get_best_model_name(exp_path, autoencoder=True)
    path_to_model = os.path.join(autoencoders_path, model_name)

    # instanciating the model
    autoencoder = setup_autoencoder(exp_data, dataset=None)

    # loading the parameters and freezing the weights
    autoencoder.load_state_dict(torch.load(path_to_model))
    for param in autoencoder.parameters():
        param.requires_grad = False

    return autoencoder


def hyperparameter_setup(exp_data, model):
    """
    Loading correct optimizer and loss function given experiment parameters

    Args:
    -----
    exp_data: dictionary
        parameters corresponding to the given experiment
    model: Torch Model
        model to be trainined or loaded

    Returns:
    --------
    optimizer: torch optim Object
        object corresponding to the optimizer used for updating networks parameters ['Adam']
    loss_function: torch nn Object
        Loss function ['cross_entropy', 'mse']
    """

    if(exp_data["denoising"]["method"] == "autoencoder"):
        aut = True
    else:
        aut = False

    # selecting correct optimizer
    lr = exp_data["training"]["learning_rate"]
    if(exp_data["training"]["optimizer"] == "ADAM"):
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)

    # selecting correct loss function
    if(exp_data["training"]["loss_function"] == "cross_entropy"):
        loss_function = torch.nn.CrossEntropyLoss()
    elif(exp_data["training"]["loss_function"] == "mse"):
        loss_function = torch.nn.MSELoss()
    elif(exp_data["training"]["loss_function"] == "mae"):
        loss_function = torch.nn.L1Loss()
    else:
        raise NotImplementedError(f"""Loss function {exp_data['training']['loss_function']}
                                   is not recognized""")

    # setting up correct scheduler. MultiStepLR(50,75) worked fine in our experiments
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=exp_data["training"]["lr_decay"],
                                                           # patience=exp_data["training"]["patience"])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     (50, 75),
                                                     exp_data["training"]["lr_decay"])

    # loading parameters from the WDSR pretrained on real data as a starting point
    if(not aut):
        downscaling = exp_data["corruption"]["downsampling"]["factor"]
        root_path = os.path.dirname(os.getcwd())
        weights_path = os.path.join( root_path, CONFIG["paths"]["weights_path"], f"wdsr_x{downscaling}.pth")
        checkpoint = torch.load(weights_path)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])

    return optimizer, loss_function, scheduler


#
