"""
Father class including all methods and properties common to all datasets in the
denoising in super-resolution pipeline

Denoising_in_superresolution/src/data
@author: Angel Villar-Corrales
"""

import os
import sys
import shutil

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import imageio
from torch.utils.data import Dataset
from torchvision import datasets, transforms

sys.path.append("..")

from lib.noisers import Noiser, Blur
import lib.utils as utils


class SrDataset(Dataset):
    """
    Father class including all methods and properties common to all datasets in the
    denoising in super-resolution pipeline

    Args:
    -----
    dataset_name: string
        name of the dataset used for training [MNIST, SVHN]
    split: string
        decides whether this dataset contains a trainig or test set
    config: dictionary
        dictionary with the global configuration parameters
    noise: string
        name of the noise to be used to corrupt the images
    std: float
        standard deviation of the noise
    downscaling: integer
        factor by which the images will be downscaled and the upsampled
    transform: Torchvision Transform
        Transform object with the transformatins and augmentations to apply to the data
    noisy: boolean
        If true, loads dataset to train a denoiser. Otherwise, loads the superresolution dataset
    patches_per_img: integer
        For the DIV2K dataset, number of image patches to sample in every batch during training
    """

    def __init__(self, dataset_name, split, config, noise, std, downscaling, transform,
                 noisy=False, patches_per_img=1):
        """
        Initializer of the super-resolution dataset object
        """

        super(SrDataset, self).__init__()

        self.dataset_name = dataset_name
        self.split = split
        self.config = config
        self.transform = transform
        self.noisy = noisy

        self.MODE = "TRAIN"
        self.savefig = False

        # patches per image only makes sense in DIV2K
        self.patches_per_img = patches_per_img
        if(self.dataset_name == "mnist" or self.dataset_name == "svhn"):
            self.patches_per_img = 1

        # noise parameteres and methods
        self.noise = noise
        self.std = std
        self.downscaling = downscaling
        self.add_noise = Noiser(noise=self.noise, std=self.std)
        self.blur = Blur(downscaling=self.downscaling)

        # relevant paths
        cur_path = os.getcwd()
        self.root_path = os.path.dirname(cur_path)
        self.dir_data = os.path.join(self.root_path, self.config["paths"]["dir_data"])

        self.data = None
        self.hr_images = []
        self.labels = []

        self.load_data()

        return


    def __len__(self):
        """ Returns the number of examples in  the dataset """
        length = len(self.hr_images)
        return length


    def __getitem__(self, idx):
        """
        Gets the data corresponding to the position idx of the dataset

        Args:
        -----
        idx: integer
            position to take the data from

        Returns:
        --------
        hr_img: numpy array
            original image corresponding to position idx of the dataset
        lr_image: numpy array
            low resolution image corresponding to downgraded version of the original one
        cur_label: integer
            label corresponding to the idx image of the dataset
        """

        if self.dataset_name == "div2k":  # for DIV2K we sample image patches
            # print(self.hr_images[idx])
            hr_img, lr_img_ = self.get_patches(idx)
            hr_img, lr_img_ = hr_img / 255, lr_img_ / 255
            hr_img = (hr_img - 0.5) / 0.5
            cur_label = 0
        else:  # for MNIST or SVHN we sample the full image
            hr_img = self.hr_images[idx]
            cur_label = self.labels[idx]
            # ensuring that images are in the range [-1,1]
            if(np.max(hr_img) > 1):
                hr_img = hr_img / 255
            # downgrading the quality of the image by smooth downsampling
            lr_img_ = self.blur(hr_img)
            hr_img = (hr_img - 0.5) / 0.5

        # adding noise to the downsampled image
        lr_img = self.add_noise(lr_img_)

        # reshaping to (Channel, Height, Width)
        hr_img = np.transpose(hr_img, (2, 0, 1))
        lr_img_ = np.transpose(lr_img_, (2, 0, 1))
        lr_img = np.transpose(lr_img, (2, 0, 1))

        # if noisy flag is set to true, we yield the images just for denoising
        if(self.noisy is True):
            return lr_img_, lr_img, cur_label

        # Otherwise we return images to denoise and superresolution
        return hr_img, lr_img, cur_label


    def load_data(self):
        """
        Loading data corresponding to the given dataset and split
        """

        # loading MNIST dataset
        if self.dataset_name == "mnist":
            self.data = datasets.MNIST(self.dir_data, train=(self.split == "train"), transform=self.transform, download=True)
            self.hr_images = [np.transpose(self.data[i][0].numpy(), (1, 2, 0)) for i in range(len(self.data))]
            self.labels = [self.data[i][1] for i in range(len(self.data))]

        # loading SVHN dataset
        elif self.dataset_name == "svhn":
            self.data = datasets.SVHN(self.dir_data, split=self.split, transform=self.transform, download=True)
            self.hr_images = [np.transpose(self.data[i][0].numpy(), (1, 2, 0)) for i in range(len(self.data))]
            self.labels = [self.data[i][1] for i in range(len(self.data))]

        # loading DIV2K dataset
        elif self.dataset_name == "div2k":
            self.data = DIV2K(self.dir_data, split=self.split, transform=self.transform,
                              downscaling=self.downscaling, method="bicubic", patches_per_img=self.patches_per_img)
            self.get_patches = lambda idx : self.data._getitem(idx, self.MODE)
            self.hr_images = self.data.hr_images
            self.labels = [0 for i in range(len(self.data))]

        return


    def eval(self):
        """ Setting to eval mode """

        self.MODE = "EVAL"
        if self.dataset_name == "div2k":
            self.get_patches = lambda idx : self.data._getitem(idx, self.MODE)

        return

    def train(self):
        """ Setting to train mode"""

        self.MODE = "TRAIN"
        if self.dataset_name == "div2k":
            self.get_patches = lambda idx : self.data._getitem(idx, self.MODE)

        return


class DIV2K():
    """
    Class for loading and handling the DIV2K dataset
    """

    def __init__(self, dir_data, split, transform, downscaling=2, method="bicubic", patches_per_img=5):
        """
        Initializer of the dataset
        """

        if(split=="test" or split=="validation"):
            split = "valid"

        self.split = split
        self.transform = transform
        self.downscaling = downscaling
        self.method = method
        self.patches_per_img = patches_per_img
        self.savefig = False

        self.hr_data_path = os.path.join(dir_data, "NTIRE", f"DIV2K_{split}_HR")
        self.lr_data_path = os.path.join(dir_data, "NTIRE", f"DIV2K_{split}_LR_{method}", f"X{downscaling}")

        if(not os.path.exists(self.hr_data_path) or not os.path.exists(self.lr_data_path)):
            raise FileExistsError(f"""DIV2K dataset could not be found. It must be located in
                                    {self.hr_data_path} and {self.lr_data_path}""")

        self.hr_images, self.lr_images = self._get_image_names()

        return


    def __len__(self):
        """ Getting number of elements in the dataset """

        n_elements = len(self.hr_images)

        return n_elements


    def _getitem(self, idx, mode="TRAIN"):
        """
        Method for loading a HR image with its LR counterpart and extracting patches

        Args:
        -----
        idx: integer
            index of the image to extract the patches from

        Returns:
        --------
        hr_patch, lr_patch: numpy arrays
            patches extracted from a HR image and its LR counterpart
        """

        hr_img, lr_img  = self._load_file(idx)
        if(mode == "TRAIN"):
            hr_patch, lr_patch = self._extract_patches(hr_img, lr_img)
        else:
            hr_patch, lr_patch = hr_img, lr_img
        hr_patch, lr_patch = np.array(hr_patch), np.array(lr_patch)

        return hr_patch, lr_patch


    def _extract_patches(self, hr_img, lr_img, patch_size=96):
        """
        Extracting patches from the HR and LR images
        """

        height, width = hr_img.shape[:2]
        idx_x = np.random.randint(low=0, high=width - patch_size)
        idx_y = np.random.randint(low=0, high=height - patch_size)

        idx_x_lr, idx_y_lr = idx_x // self.downscaling, idx_y // self.downscaling
        patch_size_lr = patch_size // self.downscaling

        hr_patch = hr_img[idx_y:idx_y+patch_size, idx_x:idx_x+patch_size]
        lr_patch = lr_img[idx_y_lr:idx_y_lr+patch_size_lr, idx_x_lr:idx_x_lr+patch_size_lr]

        return hr_patch, lr_patch


    def _load_file(self, idx):
        """ Loading an image given the index position """

        hr_name = self.hr_images[idx]
        lr_name = self.lr_images[idx]

        hr_img = imageio.imread(hr_name)
        lr_img = imageio.imread(lr_name)

        return hr_img, lr_img


    def _get_image_names(self):
        """ Obtaining the names of the HR and LR images for online loading """
        names_hr = sorted(os.listdir(self.hr_data_path))
        names_lr = sorted(os.listdir(self.lr_data_path))

        names_hr = [os.path.join(self.hr_data_path, img) for img in names_hr] * self.patches_per_img
        names_lr = [os.path.join(self.lr_data_path, img) for img in names_lr] * self.patches_per_img

        return names_hr, names_lr


    def _check_and_load(self, ext, img, f, verbose=True):
        """ Method for making binary files for the images """

        if not os.path.isfile(f) or ext.find('reset') >= 0:
            if verbose:
                print('Making a binary: {}'.format(f))
            with open(f, 'wb') as _f:
                pickle.dump(imageio.imread(img), _f)

        return


#
