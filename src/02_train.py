"""
Training a deep learning model for denoising and super-resolution purposes

Denoising_in_superresolution/src
@author: Angel Villar-Corrales
"""

import os
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


class Trainer:
    """
    Class that creates and trains a deep neural network for denoising and super-resolution

    Args:
    -----
    exp_path: string
        path to the experiment directory
    """

    def __init__(self, exp_path):
        """
        Initializer of the trainer object
        """

        self.exp_path = exp_path
        self.plots_path = os.path.join(self.exp_path, "plots")
        self.models_path = os.path.join(self.exp_path, "models")
        self.exp_data = utils.load_configuration_file(self.exp_path)
        self.dataset_name = self.exp_data["dataset"]["dataset_name"]

        self.valid_plot_path = os.path.join(self.plots_path, "valid_plots")
        utils.create_directory(self.valid_plot_path)

        # different metrics
        self.loss_list = []

        self.train_loss = 1e18
        self.valid_loss = 1e18
        self.train_mse = 1e18
        self.valid_mse = 1e18
        self.train_mae = 1e18
        self.valid_mae = 1e18
        self.train_psnr = 1e18
        self.valid_psnr = 1e18

        return


    def load_dataset(self):
        """
        Loading dataset and data loaders given the experiment parameters
        """

        self.dataset, self.train_loader, self.valid_loader, _,\
            self.num_channels = model_setup.load_dataset(self.exp_data)

        return


    def setup_training(self):
        """
        Seting up the model, the hardware and model hyperparameters
        """

        # setting up the device
        torch.backends.cudnn.fastest = True
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # initializing the model
        model = model_setup.setup_model(exp_data=self.exp_data, dataset=self.dataset.train_set, exp_path=self.exp_path)
        self.model = model.to(self.device)

        # setting up model hyper-parameters
        self.optimizer, self.loss_function, self.scheduler = model_setup.hyperparameter_setup(self.exp_data, self.model)

        return


    def training_loop(self):
        """
        Iteratively running training and validation epoch while saving intermediate models
        and training logs
        """

        self.train_logs = utils.create_train_logs(self.exp_path)
        self.num_epochs = self.exp_data["training"]["epochs"]
        self.save_frequency = self.exp_data["training"]["save_frequency"]

        for epoch in range(self.num_epochs):
            print(f"########## Epoch {epoch+1}/{self.num_epochs} ##########")
            self.validation_epoch(epoch)
            self.train_epoch(epoch)
            self.scheduler.step()

            # updating training_logs
            utils.update_logs(path=self.exp_path, plot_path=self.plots_path, train_loss=self.train_loss,
                              valid_loss=self.valid_loss, train_mae=self.train_mae, valid_mae=self.valid_mae,
                              train_mse=self.train_mse, valid_mse=self.valid_mse, train_psnr=self.train_psnr,
                              valid_psnr=self.valid_psnr)

            # saving a checkpoint
            if(epoch % self.save_frequency == 0):
                save_path = os.path.join(self.models_path, f"model_epoch_{epoch}")
                torch.save(self.model.state_dict(), save_path)

        # saving trained model
        save_path = os.path.join(self.models_path, f"model_trained")
        torch.save(self.model.state_dict(), save_path)

        return


    def train_epoch(self, epoch):
        """
        Computing training epoch
        """

        self.model.train()
        loss_list = []
        mae_list = []
        mse_list = []
        psnr_list = []

        progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
        for i, (hr_imgs, lr_imgs, labels) in progress_bar:

            hr_imgs = hr_imgs.to(self.device).float()
            lr_imgs = lr_imgs.to(self.device).float()
            self.optimizer.zero_grad()

            recovered_images = self.model(lr_imgs * 0.5) * 2  # pretrained model expects input in range [-0.5, 0.5]

            # setting images to the range [0,1]
            hr_imgs = metrics.denorm_img(hr_imgs)
            lr_imgs = metrics.denorm_img(lr_imgs)
            recovered_images = metrics.denorm_img(recovered_images)

            loss = self.loss_function(hr_imgs, recovered_images)
            loss_list.append(loss)
            mae_list.append( metrics.mean_absoulte_error(hr_imgs, recovered_images) )
            mse_list.append( metrics.mean_squared_error(hr_imgs, recovered_images) )
            psnr_list.append( metrics.psnr(hr_imgs, recovered_images) )

            # Backward and optimizer
            loss.backward()
            self.optimizer.step()

            progress_bar.set_description(f"Iter {i+1}: loss {loss.item():.5f}. ")

        loss = metrics.get_loss_stats(loss_list, message=f"Training epoch {epoch+1}")
        self.train_loss = loss
        self.train_mae = torch.mean(torch.stack(mae_list))
        self.train_mse = torch.mean(torch.stack(mse_list))
        self.train_psnr = torch.mean(torch.stack(psnr_list))

        return

    @torch.no_grad()
    def validation_epoch(self, epoch):
        """
        Computing a validation epoch
        """

        self.model.eval()
        loss_list = []
        mae_list = []
        mse_list = []
        psnr_list = []

        for i, (hr_imgs, lr_imgs, labels) in enumerate(tqdm(self.valid_loader)):

            hr_imgs = hr_imgs.to(self.device).float()
            lr_imgs = lr_imgs.to(self.device).float()

            # pretrained model expects input in range [-0.5, 0.5] and we were using [-1,1]
            recovered_images = self.model(lr_imgs * 0.5) * 2

            # setting images to the range [0,1]
            hr_imgs = metrics.denorm_img(hr_imgs)
            lr_imgs = metrics.denorm_img(lr_imgs)
            recovered_images = metrics.denorm_img(recovered_images)

            loss = self.loss_function(hr_imgs, recovered_images)
            loss_list.append(loss)
            mae_list.append( metrics.mean_absoulte_error(hr_imgs, recovered_images) )
            mse_list.append( metrics.mean_squared_error(hr_imgs, recovered_images) )
            cur_psnr = metrics.psnr(hr_imgs, recovered_images)
            psnr_list.append( cur_psnr )

            # saving some images every 5 epochs to check learning progress
            if(i < 3 and epoch%5 == 0 ):
                if(self.dataset_name == "div2k"):
                    visualizations.display_images_one_row(
                            hr_imgs, lr_imgs, recovered_images,
                            savepath=os.path.join(self.valid_plot_path, f"test_figures_{epoch}_{i}"),
                            dataset_name=self.dataset_name, psnr=cur_psnr,
                            downscaling=self.exp_data["corruption"]["downsampling"]["factor"]
                        )
                else:
                    visualizations.display_images(
                            hr_imgs, lr_imgs, recovered_images,
                            savepath=os.path.join(self.valid_plot_path, f"valid_plot_epoch_{epoch}.png"),
                            dataset_name=self.dataset_name
                        )

            # we only use 30 images for validation
            if(i==30):
                break

        loss = metrics.get_loss_stats(loss_list, message=f"Validation epoch {epoch+1}")
        self.loss_list.append(loss)
        self.valid_loss = loss
        self.valid_mae = torch.mean(torch.stack(mae_list))
        self.valid_mse = torch.mean(torch.stack(mse_list))
        self.valid_psnr = torch.mean(torch.stack(psnr_list))

        return



if __name__ == "__main__":

    os.system("clear")
    exp_directory = arguments.get_directory_argument()

    trainer = Trainer(exp_path=exp_directory)

    print("Loading dataset...")
    trainer.load_dataset()

    print("Setting up model...")
    trainer.setup_training()

    trainer.training_loop()

#
