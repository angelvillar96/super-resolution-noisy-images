"""
Computing plots for the different metrics monitored during training and validation

Denoising_in_superresolution/src
@author: Angel Villar-Corrales
"""

import os

import numpy as np
from matplotlib import pyplot as plt

import lib.utils as utils
import lib.arguments as arguments


def main():
    """
    Main logic for loading the data and creating the plots for all the monitored metrics
    using linear and logarithmic scales for the Y axis
    """

    # loading training logs
    exp_directory = arguments.get_directory_argument()
    train_logs = utils.load_train_logs(exp_directory)
    plots_path = os.path.join(exp_directory, "plots", "metrics")
    utils.create_directory(plots_path)

    ##################################
    # computing plots for all metrics
    ##################################

    epochs = np.arange(len(train_logs["loss"]["train"]))

    # loading test data, if any
    evaluation = False
    if("evaluation" in list(train_logs.keys())):
        evaluation = True
        test_epochs = np.sort(train_logs["evaluation"]["epochs"])
        idx_sorted = np.argsort(train_logs["evaluation"]["epochs"])

        test_loss = np.array(train_logs["evaluation"]["loss"])[idx_sorted]
        test_mae = np.array(train_logs["evaluation"]["mae"])[idx_sorted]
        test_mse = np.array(train_logs["evaluation"]["mse"])[idx_sorted]
        test_psnr = np.array(train_logs["evaluation"]["psnr"])[idx_sorted]


    # 1.- Linear loss plot (except epoch 0)
    train_loss = train_logs["loss"]["train"][1:]
    valid_loss = train_logs["loss"]["valid"][1:]
    plt.figure()
    plt.plot(epochs[1:], train_loss, label="Train Loss")
    plt.plot(epochs[1:], valid_loss, label="Validation Loss")
    if(evaluation):
        plt.scatter(test_epochs, test_loss, label="Test Loss", c="g", marker="x")
        plt.plot(test_epochs, test_loss, "--", c="g")
    plt.legend(loc="best")
    plt.title("Loss (MSE)")
    plt.savefig(os.path.join(plots_path, "plot_loss.png"))

    # 2.- Logscale loss plot
    train_loss = train_logs["loss"]["train"]
    valid_loss = train_logs["loss"]["valid"]
    plt.figure()
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, valid_loss, label="Validation Loss")
    if(evaluation):
        plt.scatter(test_epochs, test_loss, label="Test Loss", c="g", marker="x")
        plt.plot(test_epochs, test_loss, "--", c="g")
    plt.yscale("log")
    plt.legend(loc="best")
    plt.title("Loss (MSE)")
    plt.savefig(os.path.join(plots_path, "plot_loss_log.png"))

    # 3.- Linear mae plot
    train_mae = train_logs["metrics"]["mae"]["train"][1:]
    valid_mae = train_logs["metrics"]["mae"]["valid"][1:]
    plt.figure()
    plt.plot(epochs[1:], train_mae, label="Train MAE")
    plt.plot(epochs[1:], valid_mae, label="Validation MAE")
    if(evaluation):
        plt.scatter(test_epochs, test_mae, label="Test MAE", c="g", marker="x")
        plt.plot(test_epochs, test_mae, "--", c="g")
    plt.legend(loc="best")
    plt.title("Mean Average Error (MAE)")
    plt.savefig(os.path.join(plots_path, "plot_metric_mae.png"))

    # 4.- Log mae plot
    train_mae = train_logs["metrics"]["mae"]["train"]
    valid_mae = train_logs["metrics"]["mae"]["valid"]
    plt.figure()
    plt.plot(epochs, train_mae, label="Train MAE")
    plt.plot(epochs, valid_mae, label="Validation MAE")
    if(evaluation):
        plt.scatter(test_epochs, test_mae, label="Test MAE", c="g", marker="x")
        plt.plot(test_epochs, test_mae, "--", c="g")
    plt.legend(loc="best")
    plt.yscale("log")
    plt.title("Mean Average Error (MAE)")
    plt.savefig(os.path.join(plots_path, "plot_metric_mae_log.png"))

    # 5.- Linear psnr plot
    train_psnr = train_logs["metrics"]["psnr"]["train"]
    valid_psnr = train_logs["metrics"]["psnr"]["valid"]
    plt.figure()
    plt.plot(epochs, train_psnr, label="Train PSNR")
    plt.plot(epochs, valid_psnr, label="Validation PSNR")
    if(evaluation):
        plt.scatter(test_epochs, test_psnr, label="Test PSNR", c="g", marker="x")
        plt.plot(test_epochs, test_psnr, "--", c="g")
    plt.legend(loc="best")
    plt.title("Peak Signal-to-Noise Ratio (PSNR)")
    plt.savefig(os.path.join(plots_path, "plot_metric_psnr.png"))

    # 6.- Log psnr plot
    train_psnr = train_logs["metrics"]["psnr"]["train"]
    valid_psnr = train_logs["metrics"]["psnr"]["valid"]
    plt.figure()
    plt.plot(epochs, train_psnr, label="Train PSNR")
    plt.plot(epochs, valid_psnr, label="Validation PSNR")
    if(evaluation):
        plt.scatter(test_epochs, test_psnr, label="Test PSNR", c="g", marker="x")
        plt.plot(test_epochs, test_psnr, "--", c="g")
    plt.legend(loc="best")
    plt.yscale("log")
    plt.title("Peak Signal-to-Noise Ratio (PSNR)")
    plt.savefig(os.path.join(plots_path, "plot_metric_psnr_log.png"))


    return


if __name__ == "__main__":

    os.system("clear")
    main()
