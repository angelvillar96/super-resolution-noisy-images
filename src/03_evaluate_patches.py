"""
Loading a pretrained model and using it for superresolution and denoising purposes
on patches of the images from the test-set. The results for the image are the average
of the patch-results

Denoising_in_superresolution/src
@author: Angel Villar-Corrales
"""

import os
import json
from argparse import Namespace
from tqdm import tqdm

import numpy as np
from matplotlib import pyplot as plt
import torch

import models as models
import data
import lib.utils as utils
import lib.metrics as metrics
import lib.arguments as arguments
import lib.model_setup as model_setup
from config import CONFIG


class EvaluatePatches:
    """
    Loading the model and performing superresolution in a few images
    """

    def __init__(self, exp_path, checkpoint=-1):
        """
        Initializer of the evaluator object

        Args:
        -----
        exp_path: string
            path to the experiment directory
        checkpoint: integer
            number of epochs corresponding to the checkpoint to load. -1 means trained model
        """

        self.exp_path = exp_path
        self.models_path = os.path.join(self.exp_path, "models")
        self.plots_path = os.path.join(self.exp_path, "plots")

        self.exp_data = utils.load_configuration_file(self.exp_path)
        self.exp_data["training"]["batch_size"] = 1
        self.exp_data["dataset"]["patches_per_img"] = 10
        self.train_logs = utils.load_train_logs(self.exp_path)

        self.checkpoint = checkpoint

        return


    def load_dataset(self):
        """
        Loading dataset and data loaders
        """

        self.dataset, _, _,\
            self.test_loader, self.num_channels = model_setup.load_dataset(self.exp_data, test_patch=True)

        return


    def load_generalization_dataset(self, noise, std):
        """
        Loading dataset and data loaders to evaluate the generalization capabilities of a model

        Args:
        -----
        noise: string
            type of noise used to corrup the test images
        std: float
            power of the corruption noise
        """

        self.dataset,self.test_loader,\
            self.num_channels = model_setup.load_generalization_dataset(exp_data=self.exp_data, noise=noise,
                                                                        std=std, test_patch=True, savefig=False)

        return


    def load_model(self):
        """
        Creating a model and loading the pretrained network parameters from the saved
        state dictionary. Setting the model to use a GPU
        """

        # getting model name given checkpoint
        if(self.checkpoint<0):
            model_name = "model_trained"
        else:
            model_name = f"model_epoch_{self.checkpoint}"
        path_to_model = os.path.join(self.models_path, model_name)

        # making sure the model exists
        if(not os.path.exists(path_to_model)):
            print("ERROR!")
            print(f"Model: {model_name} was not found in path {self.models_path}")
            exit()

        # creating model architecture
        # setting up the device
        torch.backends.cudnn.fastest = True
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # initializing the model and loading the state dicitionary
        model = model_setup.setup_model(exp_data=self.exp_data, exp_path=self.exp_path)
        model.load_state_dict(torch.load(path_to_model))
        self.model = model.to(self.device)

        # setting up model hyper-parameters
        self.optimizer, self.loss_function, self.scheduler = model_setup.hyperparameter_setup(self.exp_data, self.model)

        return

    with torch.no_grad()
    def test_model(self):
        """
        Using the pretrained model to perform denoising and superresolution in the test
        set measuring the metric values

        Returns:
        --------
        test_loss: float
            loss computed on the test set
        test_mae: float
            Mean Absolute Error computed on the test set
        test_mse: float
            Mean Squared Error computed on the test set
        test_psnr: float
            Peak signal-to-noise error computed on the test set
        """

        self.model.eval()
        loss_list = []
        mae_list = []
        mse_list = []
        psnr_list = []

        for i, (hr_imgs, lr_imgs, labels) in enumerate(tqdm(self.test_loader)):

            hr_imgs = hr_imgs.to(self.device).float()
            lr_imgs = lr_imgs.to(self.device).float()

            # pretrained model expects input in range [-0.5, 0.5] and we were using [-1,1]
            recovered_images = self.model(lr_imgs * 0.5) * 2

            # setting images to the range [0,1]
            hr_imgs, lr_imgs = metrics.denorm_img(hr_imgs), metrics.denorm_img(lr_imgs)
            recovered_images = metrics.denorm_img(recovered_images)

            loss = self.loss_function(hr_imgs, recovered_images)
            loss_list.append(loss)
            mae_list.append( metrics.mean_absoulte_error(hr_imgs, recovered_images) )
            mse_list.append( metrics.mean_squared_error(hr_imgs, recovered_images) )
            psnr_list.append( metrics.psnr(hr_imgs, recovered_images) )

        loss = metrics.get_loss_stats(loss_list, message=f"Test Loss Stats")
        test_loss = loss
        test_mae = torch.mean(torch.stack(mae_list))
        test_mse = torch.mean(torch.stack(mse_list))
        test_psnr = torch.mean(torch.stack(psnr_list))

        return test_loss, test_mae, test_mse, test_psnr



if __name__ == "__main__":

    os.system("clear")

    exp_directory, noise, std = arguments.get_directory_argument(generalization=True)
    checkpoint = 0

    evaluator = EvaluatePatches(exp_path=exp_directory, checkpoint=checkpoint)

    if(noise==""):
        evaluator.load_dataset()
    else:
        evaluator.load_generalization_dataset(noise=noise, std=std)

    evaluator.load_model()
    test_loss, test_mae, test_mse, test_psnr = evaluator.test_model()

    print(f"Test Loss: {test_loss}")
    print(f"Test MAE: {test_mae}")
    print(f"Test MSE: {test_mse}")
    print(f"Test PSNR: {test_psnr}")

    # creating/saving generalization results
    if(noise!=""):
        gen_logs_path = os.path.join(exp_directory, "generalization_logs.json")
        if(not os.path.exists(gen_logs_path)):
            gen_logs = utils.create_generalization_logs(exp_directory)
        else:
            gen_logs = utils.load_generalization_logs(exp_directory)
        exp_name = f"noise={noise}__std={std}"
        gen_logs[exp_name] = {}
        gen_logs[exp_name]["MAE"] = float(test_mae)
        gen_logs[exp_name]["MSE"] = float(test_mse)
        gen_logs[exp_name]["PSNR"] = float(test_psnr)

        with open(gen_logs_path, "w") as file:
            json.dump(gen_logs, file)




#
