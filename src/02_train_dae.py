"""
Training a denoiser autoencoder to later include it as a denoising methood within
the super resolution network

Denoising_in_superresolution/src
@author: Angel Villar-Corrales
"""

import os
import json
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



class AutoencoderTrainer:
    """
    Class for instanciating, training and evaluating an autoencoder
    """

    def __init__(self, exp_path, epochs_autoencoder=50):
        """
        Initializer of the trainer object

        Args:
        -----
        exp_path: string
            path to the experiment directory
        epochs_autoencoder: integer
            number of epochs to train the autoencoder for
        """

        self.exp_path = exp_path
        self.plots_path = os.path.join(self.exp_path, "plots", "autoencoder")
        self.models_path = os.path.join(self.exp_path, "models", "autoencoder")
        self.exp_data = utils.load_configuration_file(self.exp_path)
        self.dataset_name = self.exp_data["dataset"]["dataset_name"]
        self.num_epochs = epochs_autoencoder

        utils.create_directory(self.plots_path)
        utils.create_directory(self.models_path)

        # different metrics
        self.loss_list = []

        self.train_loss = 1e18
        self.valid_loss = 1e18

        return


    def load_dataset(self):
        """
        Loading dataset and data loaders given the experiment parameters
        """

        dataset, train_loader, valid_loader, _, num_channels = model_setup.load_dataset(self.exp_data, noisy=True)
        self.dataset = dataset
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.num_channels = num_channels

        return


    def setup_training(self):
        """
        Seting up the model, the hardware and model hyperparameters
        """

        # setting up the device
        torch.backends.cudnn.fastest = True
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # initializing the model
        model = model_setup.setup_autoencoder(self.exp_data, dataset=self.dataset.train_set)
        self.model = model.to(self.device)

        # setting up model hyper-parameters
        self.optimizer, self.loss_function, self.scheduler = model_setup.hyperparameter_setup(self.exp_data, self.model)

        return


    def training_loop(self):
        """
        Iteratively running training and validation epoch while saving intermediate models
        and training logs
        """

        # initializing logs
        self.autoencoder_logs_path = os.path.join(self.exp_path, "autoencoder_logs.json")
        self.autoencoder_logs = {}
        self.autoencoder_logs["experiment_started"] = utils.timestamp()
        self.autoencoder_logs["train_loss"] = []
        self.autoencoder_logs["valid_loss"] = []

        self.save_frequency = 5

        for epoch in range(self.num_epochs):
            print(f"########## Epoch {epoch+1}/{self.num_epochs} ##########")
            self.validation_epoch(epoch)
            self.train_epoch(epoch)
            self.scheduler.step(self.valid_loss)

            # saving logs
            self.autoencoder_logs["train_loss"].append(float(self.train_loss))
            self.autoencoder_logs["valid_loss"].append(float(self.valid_loss))
            with open(self.autoencoder_logs_path, "w") as file:
                json.dump(self.autoencoder_logs, file)

            # saving a checkpoint
            if(epoch % self.save_frequency == 0):
                save_path = os.path.join(self.models_path, f"autoencoder_epoch_{epoch}")
                torch.save(self.model.state_dict(), save_path)

        # saving trained model
        save_path = os.path.join(self.models_path, f"autoencoder_trained")
        torch.save(self.model.state_dict(), save_path)

        return


    def train_epoch(self, epoch):
        """
        Computing training epoch
        """

        self.model.train()
        loss_list = []

        # for i, (imgs, noisy_imgs, _) in enumerate(tqdm(self.train_loader)):
        for i, (imgs, noisy_imgs, _) in enumerate(tqdm(self.valid_loader)):

            imgs = imgs.to(self.device).float()
            noisy_imgs = noisy_imgs.to(self.device).float()
            self.optimizer.zero_grad()

            # noisy_imgs = noisy_imgs/127.5
            denoised_imgs = self.model(noisy_imgs * 0.5) * 2
            denoised_imgs = metrics.denorm_img(denoised_imgs)
            # denoised_imgs = denoised_imgs*127.5

            loss = self.loss_function(imgs, denoised_imgs)
            loss_list.append(loss)

            # Backward and optimizer
            loss.backward()
            self.optimizer.step()

        loss = metrics.get_loss_stats(loss_list, message=f"Training epoch {epoch+1}")
        self.train_loss = loss

        return

    @torch.no_grad()
    def validation_epoch(self, epoch):
        """
        Computing a validation epoch
        """

        self.model.eval()
        loss_list = []

        for i, (imgs, noisy_imgs, _) in enumerate(tqdm(self.valid_loader)):
            # validate on 100 batches
            if(i>100):
                break

            imgs = imgs.to(self.device).float()
            noisy_imgs = noisy_imgs.to(self.device).float()

            denoised_imgs = self.model(noisy_imgs * 0.5) * 2
            denoised_imgs = metrics.denorm_img(denoised_imgs)

            loss = self.loss_function(imgs, denoised_imgs)
            loss_list.append(loss)

            # saving some images every 5 epochs to check if it learns
            # if(i == 0 and epoch%5 == 0 ):
            if(i < 3 and epoch%3 == 0 ):
                # visualizations.display_images(imgs, noisy_imgs, denoised_imgs,
                #                      savepath=os.path.join(self.plots_path, f"valid_plot_epoch_{epoch}.png"),
                #                      dataset_name=self.dataset_name)
                visualizations.display_images_one_row(imgs, noisy_imgs, denoised_imgs,
                                     savepath=os.path.join(self.plots_path, f"valid_plot_epoch_{epoch}_{i}"),
                                     dataset_name=self.dataset_name, psnr=None,
                                     downscaling=self.exp_data["corruption"]["downsampling"]["factor"])

        loss = metrics.get_loss_stats(loss_list, message=f"Validation epoch {epoch+1}")
        self.loss_list.append(loss)
        self.valid_loss = loss

        return



if __name__ == "__main__":

    os.system("clear")

    epochs_autoencoder = 80
    exp_directory = arguments.get_directory_argument()

    autoencoder_trainer = AutoencoderTrainer(exp_path=exp_directory, epochs_autoencoder=epochs_autoencoder)

    print("Loading dataset...")
    autoencoder_trainer.load_dataset()

    print("Setting up model...")
    autoencoder_trainer.setup_training()

    autoencoder_trainer.training_loop()

    #
