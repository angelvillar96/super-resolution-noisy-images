"""
Finding, loading and preprocessing the data used for training and evauation of the
denoising and superresolution pipeline

Denoising_in_superresolution/src/data
@author: Angel Villar-Corrales
"""

import numpy as np
from torch.utils.data import DataLoader, sampler

from .sr_dataset import SrDataset
import lib.augmentations as augment


class Data:
    """
    Class corresponding to data handler. This class includes loaders for train and test sets

    Args:
    -----
    dataset_name: string
        name of the dataset used for training [MNIST, SVHN]
    config: dictionary
        dictionary with the global configuration parameters
    train: boolean
        loading the train set
    test: boolean
        loading the train set
    noise: string
        name of the noise to be used to corrupt the images
    std: float
        standard deviation of the noise
    downscaling: integer
        factor by which the images will be downscaled and the upsampled
    valid_size: float
        percentage of the training set used for validation purposes
    augmentations: dictionary
        dictionary with flags indicating which augmentations to apply
    noisy: boolean
        If true, loads dataset to train a denoiser. Otherwise, loads the superresolution dataset
    patches_per_img: integer
        For the DIV2K dataset, number of image patches to sample in every batch during training
    """

    def __init__(self, dataset_name, config, train=True, test=False, noise="", std=0,
                 downscaling=1, valid_size=0.2, augmentations={}, noisy=False, patches_per_img=1):
        """
        Initializer of the data object
        """

        # dataset parameters
        self.dataset_name = dataset_name
        self.train_loader = None
        self.test_loader = None
        self.train = train
        self.test = test

        # dataset and validation split parameters
        self.valid_size = valid_size
        self.patches_per_img = patches_per_img
        self.shuffle = True

        # noise parameteres
        self.noise = noise
        self.std = std

        # config
        self.config = config

        if train:
            # obtaining transformations and augmentations to be applied
            train_transforms = augment.get_augmentations(dataset_name=dataset_name,
                                                         mode="train",
                                                         augmentations=augmentations)
            # loading train dataset
            self.train_set = SrDataset(dataset_name=dataset_name, split="train", config=config, noise=self.noise,
                                       std=self.std, downscaling=downscaling, transform=train_transforms, noisy=noisy,
                                       patches_per_img=patches_per_img)
        else:
            self.train_set = None

        if test:
            # obtaining transformations and augmentations to be applied
            test_transforms = augment.get_augmentations(dataset_name=dataset_name,
                                                        mode="test",
                                                        augmentations=augmentations)
            # loading test dataset
            self.test_set = SrDataset(dataset_name=dataset_name, split="test", config=config, noise=self.noise,
                                      std=self.std, downscaling=downscaling, transform=test_transforms, noisy=noisy,
                                      patches_per_img=patches_per_img)
        else:
            self.test_set = None
        return

    def get_data_loader(self, batch_size, set="train", shuffle=False, test_patch=False,
                        generalization=False, savefig=False):
        """
        Computing a data loader to iterate over a data set given a certain dataset

        Args:
        -----
        batch_size: integer
            number of elements to have in each batch
        set: string
            set to compute the data loader of ['train', 'test']
        shuffle: boolean
            if true, the images are drawn randomly
        test_patch: boolean
            if true, patch is sampled even for the test set
        savefig: boolean
            if true, hr_img is the whole image and not a patch (even when test_patch is true).
            We use this to generate the paper figures

        Returns:
        --------
        data_loader: Data Loader
            torch data loader to iterate the given set
        """

        if(set == "train"):
            dataset = self.train_set
            if(dataset is not None):
                dataset.train()
        elif(set == "test"):
            dataset = self.test_set
            if(dataset is not None):
                if(test_patch):
                    dataset.train()
                else:
                    dataset.eval()
                if(savefig):
                    dataset.set_savefig(mode=True, generalization=generalization)
        else:
            assert False

        if(generalization):
            data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                     num_workers=1)
        else:
            data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                     num_workers=self.config["num_workers"])

        return data_loader

    def get_train_validation_loaders(self, batch_size, test_patch=False):
        """
        Creates a train/validation split of the training set and returns the data loaders for both

        Args:
        -----
        batch_size: integer
            number of elements to have in each batch

        Returns:
        --------
        train_loader: Data Loader
            torch data loader to iterate over the train set
        valid_loader: Data Loader
            torch data loader to iterate over the validation set
        """

        if(not self.train):
            return None, None

        # special case fo DIV2K dataset
        if(self.dataset_name == "div2k"):
            train_loader = self.get_data_loader(batch_size=batch_size, set="train",
                                                shuffle=True, test_patch=test_patch)
            valid_loader = self.get_data_loader(batch_size=batch_size, set="test",
                                                shuffle=True, test_patch=test_patch)
            print("\n")
            print(f"{len(self.train_set) // self.patches_per_img} images in training split")
            print(f"{len(self.test_set) // self.patches_per_img} images in validation split")
            print(f"{self.patches_per_img} patches sampled from each image")
            print("\n")
            return train_loader, valid_loader

        num_train = len(self.train_set)
        indices = list(range(num_train))
        split = int(np.floor(self.valid_size * num_train))

        # randomizing train and validation set
        if(self.shuffle):
            np.random.seed(self.config["random_seed"])
            np.random.shuffle(indices)

        # getting idx for train and validation
        train_idx, valid_idx = indices[split:], indices[:split]
        train_sampler = sampler.SubsetRandomSampler(train_idx)
        valid_sampler = sampler.SubsetRandomSampler(valid_idx)

        # creating data loaders
        train_loader = DataLoader(self.train_set, batch_size=batch_size, sampler=train_sampler,
                                  num_workers=self.config["num_workers"])

        valid_loader = DataLoader(self.train_set, batch_size=batch_size, sampler=valid_sampler,
                                  num_workers=self.config["num_workers"])

        self.train_examples = len(train_idx)
        self.valid_examples = len(valid_idx)

        print("\n")
        print(f"{self.train_examples} images in training split")
        print(f"{self.valid_examples} images in validation split")
        print("\n")

        return train_loader, valid_loader


#
