"""
Given an experiment directory, testing all checkpoints and monitoring the metrics on the
test set. Then, saving these values of the training_logs json file

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


Evaluator = __import__('03_evaluate')
EvaluatorPatches = __import__('03_evaluate_patches')

def main():
    """
    Main logic for testing the different checkpoints saved during training
    """

    patches = False

    # loading exp_directory and reading all checkpoints
    exp_directory = arguments.get_directory_argument()
    model_directory = os.path.join(exp_directory, "models")

    model_list = os.listdir(model_directory)
    model_list = sorted(model_list)
    print(f"Checkpoints found: {model_list}")

    # for saving results in training logs
    train_logs = utils.load_train_logs(exp_directory)
    train_logs["evaluation"] = {}
    train_logs["evaluation"]["epochs"] = []
    train_logs["evaluation"]["loss"] = []
    train_logs["evaluation"]["mae"] = []
    train_logs["evaluation"]["mse"] = []
    train_logs["evaluation"]["psnr"] = []
    train_logs["evaluation"]["ssim"] = []

    # iterating all checpoints testing them
    for i, model in enumerate(model_list):

        print(f"Processing: {model}")
        if("model_") not in model:
            continue

        # obtaining the epoch number of the checkpoint
        epoch = model.split("_")[-1]
        if(epoch=="trained"):
            epoch = -1
        else:
            epoch = int(epoch)

        if(patches):
            evaluator = EvaluatorPatches.EvaluatePatches(exp_path=exp_directory, checkpoint=epoch)
        else:
            evaluator = Evaluator.Evaluate(exp_path=exp_directory, checkpoint=epoch)
        evaluator.load_dataset()
        evaluator.load_model()
        test_loss, test_mae, test_mse, test_psnr = evaluator.test_model()

        if(epoch==-1): epoch = 100  # for final epoch (model_final.pth)
        train_logs["evaluation"]["epochs"].append(int(epoch))
        train_logs["evaluation"]["loss"].append(float(test_loss))
        train_logs["evaluation"]["mae"].append(float(test_mae))
        train_logs["evaluation"]["mse"].append(float(test_mse))
        train_logs["evaluation"]["psnr"].append(float(test_psnr))

        train_logs_file = os.path.join(exp_directory, "training_logs.json")
        with open(train_logs_file, "w") as file:
            json.dump(train_logs, file)

    return


if __name__ == "__main__":

    os.system("clear")
    main()


#
