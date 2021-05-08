"""
Creating a noisy dataset for debuggin purposes

Denoising_in_superresolution/src/data
@author: Angel Villar-Corrales
"""

import os
import sys

import numpy as np

from torch.utils.data import Dataset
from torchvision import datasets, transforms

sys.path.append("..")

from lib.noisers import Noiser, Blur


class NoisyDataset(Dataset):
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
    """

    def __init__(self, dataset_name, split, config, noise, std, downscaling, transform):
        """
        Initializer of the super-resolution dataset object
        """
        super(NoisyDataset, self).__init__()

        self.dataset_name = dataset_name
        self.split = split
        self.config = config
        self.transform = transform

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

        hr_img = self.hr_images[idx]
        cur_label = self.labels[idx]

        # ensuring that images are in the range [0,1] (or more due to equalizations)
        # if(np.max(hr_img)>1):
            # hr_img = hr_img/255

        # downgrading the quality of the image by smooth downsampling
        lr_img_ = self.blur(hr_img)

        # adding noise to the downsampled image
        lr_img = self.add_noise(lr_img_)

        # reshaping to (Channel, Height, Width)
        lr_img_ = np.transpose(lr_img_, (2, 0, 1))
        lr_img = np.transpose(lr_img, (2, 0, 1))

        return lr_img_, lr_img, cur_label


    def load_data(self):
        """
        Loading data corresponding to the given dataset and split
        """

        # loading MNIST dataset
        if self.dataset_name == "mnist":
            self.data = datasets.MNIST(self.dir_data, train=(self.split == "train"), transform=self.transform, download=True)
            self.hr_images = [np.transpose(self.data[i][0].numpy(), (1, 2, 0)) for i in range(len(self.data))]
            self.labels = [self.data[i][1] for i in range(len(self.data))]

        elif self.dataset_name == "svhn":
            self.data = datasets.SVHN(self.dir_data, split=self.split, transform=self.transform, download=True)

            self.hr_images = [np.transpose(self.data[i][0].numpy(), (1, 2, 0)) for i in range(len(self.data))]
            self.labels = [self.data[i][1] for i in range(len(self.data))]

        return

#
