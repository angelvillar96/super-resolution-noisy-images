# Deep Learning Architectural Designs for <br> Super-Resolution of Noisy Images

This repository contains the code for reproducing the results of our ICASSP 2021 paper *Deep Learning Architectural Designs for Super-Resolution of Noisy Images*.

[Paper](http://www.angelvillarcorrales.com/templates/others/Publications/Deep_Learning_Architectures_for_Super_Resolution_of_NoisyImages_ICASSP.pdf)

#### Codebase Coming Soon!


## Contents

 * [1. Getting Started](#getting-started)
 * [2. Directory Structure](#directory-structure)
 * [3. Use Guide](#use-guide)
 * [4. Citing](#citing)
 * [5. Contact](#contact)


## Getting Started

To download the code, fork the repository or clone it using the following command:

```
  git clone https://github.com/angelvillar96/super-resolution-noisy-images.git
```

### Prerequisites

To get the repository running, you will need several python packages, e.g., numpy, torch and matplotlib.

You can install them all easily and avoiding dependency issues by installing the conda environment file included in the repository. To do so, run the following command from the Conda Command Window:

**TODO:** Export and add environment

```shell
$ conda env create -f environment.yml
$ activate denoising_superresolution
```

*__Note__:* This step might take a few minutes


## Directory Structure

The following tree diagram displays the detailed directory structure of the project:

```
denoising_in_superresolution
├── eval_results/
│   ├── eval_checkpoint_dicts/
│   ├── eval_image_comparison/
│   └── eval_metric_plots/
|
├── experiments/
│   ├── baseline/
│   ├── median_filter_exps/
│   ├── wiener_filter_exps/
│   └── denoising_autoencoder_exps/
|
├── src/
│   |── data/
|   |    ├── __init__.py
|   |    └── sr_dataset.py
│   |── lib/
|   |    ├── arguments.py
|   |    ├── augmentations.py
|   |    ├── denoisers.py
|   |    └── layers.py
|   |    ...
│   ├── models/
|   |    ├── denoising_autoencoder.py
|   |    └── wdsr_a.py      
│   |── 01_create_experiment.py
│   |── 02_train_model.py
│   |── 03_evaluate_model.py
|       ...
|
├── environment.yml
├── README.md
```
Now, we give a short overview of the different directories:

- **eval_results/**: This directory contains the final evaluation results, comparing the performance of different methods, noise schemes and hyper-parameters.

  - **eval_checkpoint_dicts/**: *JSON* files containing the comprehensive evaluations for different experiments.

  - **eval_image_comparison/**: images comparing reconstruction quality of a degraded image using different methods.

  - **eval_metric_plots/**: plots displaying the evaluation metrics (MSE, MAE and PSNR) as a function of the power of the noise.

- **experiments/**: Directory containing the experiment folders. New experiments created are placed automatically under this directory. We provide pre-trained models for *pre-network* and *in-network* architectures using no denoiser, median filters, wiener filters and denoising autoencoders respectively.

- **src/**: Code for the experiments. For a more detailed description of the code structure,  click [here](https://github.com/angelvillar96/denoising_in_superresolution/blob/denoising/src/README.md)

  - **data/**: Methods for data loading and preprocessing.

  - **models/**: Classes corresponding to the WDSR super-resolution model and for the denoising autoencoders

  - **lib/**: Library methods for different purposes, such as command-line arguments handling, implementation of denoisers as neural network layers or evaluation metrics.


## Use Guide

### Creating an Experiment

#### Usage

```shell
$ python 01_create_experiment.py [-h] [-d EXP_DIRECTORY]
                               [--dataset_name DATASET_NAME]
                               [--denoiser DENOISER]
                               [--denoiser_type DENOISER_TYPE]
                               [--kernel_size KERNEL_SIZE]
                               [--bottleneck BOTTLENECK] [--rotation ROTATION]
                               [--translation TRANSLATION]
                               [--num_epochs NUM_EPOCHS]
                               [--batch_size BATCH_SIZE]
                               [--optimizer OPTIMIZER]
                               [--loss_function LOSS_FUNCTION]
                               [--learning_rate LEARNING_RATE]
                               [--lr_decay LR_DECAY] [--patience PATIENCE]
                               [--validation_size VALIDATION_SIZE]
                               [--save_frequency SAVE_FREQUENCY]
                               [--model_name MODEL_NAME]
                               [--num_filters NUM_FILTERS]
                               [--num_res_blocks NUM_RES_BLOCKS]
                               [--num_block_features NUM_BLOCK]
                               [--res_scale RES_SCALE] [--r_mean R_MEAN]
                               [--g_mean G_MEAN] [--b_mean B_MEAN]
                               [--noise NOISE] [--std STD]
                               [--downscaling DOWNSCALING]
```  

Creating an experiment automatically generates a directory in the specified EXP_DIRECTORY, containing a *JSON* file with the experiment parameters and subdirectories for the models and plots.

**Note**: run `python 01_create_experiment.py --help` to display a detailed description of the parameters.

#### Example

The following example creates in the directory */experiments/example_experiment* an experiment using a Wiener filter denoiser and the in-network architecture for the MNIST dataset. Next, the network architecture is exported to a *.txt*  file and a subset of corrupted images is saved under the */plots* directory.

```shell
$ python 01_create_experiment.py -d example_experiment --dataset_name mnist \
    --denoiser wiener_filter --denoiser_type innetwork --noise gaussian \
    --std 0.3 --downscaling 2
$ python aux_generate_network_txt.py -d example_experiment/denoiser_wiener_filter_type_innetwork_dataset_mnist_std_0.3_2020-05-11_09-44-51
$ python aux_generate_subset_data.py -d example_experiment/denoiser_wiener_filter_type_innetwork_dataset_mnist_std_0.3_2020-05-11_09-44-51
```


### Training and Evaluation

Once the experiment is initialized, the models can be first trained to jointly perform denoising and super-resolution and then their performance can be evaluated by using the following commands.

```shell
$ CUDA_VISIBLE_DEVICES=0 python 02_train_model.py -d YOUR_EXP_DIRECTORY
$ CUDA_VISIBLE_DEVICES=0 python 03_test_accross_epochs.py -d YOUR_EXP_DIRECTORY
$ CUDA_VISIBLE_DEVICES=0 python 04_generate_metric_plots.py -d YOUR_EXP_DIRECTORY
```

First, the model will be trained and validated for the number of epochs specified in the configuration file (100 by default). Every 10 epochs, a model checkpoint will be saved under the */models* directory, and every epoch the current loss and metrics will be stored in a *training_logs.json* file.

Then, each of the checkpoints will be evaluated on the test set. These evaluation results are also saved in the *training_logs.json* file.

Finally, the third command generates plots displaying the loss and metric landscapes for training, validation and test.


## Citing

Please consider citing if you find our findings or our repository helpful.
```
@article{villar2021deep,
  title={Deep learning architectural designs for super-resolution of noisy images},
  author={Villar-Corrales, Angel and Schirrmacher, Franziska and Riess, Christian},
  journal={2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year={2021}
}
```

## Contact

This work has been developed by [Angel Villar-Corrales](http://angelvillarcorrales.com/templates/home.php), supervised by
 [M. Sc. Franziska Schirrmacher](https://www.cs1.tf.fau.de/person/franziska-schirrmacher/).

In case of any questions or problems regarding the project or repository, do not hesitate to contact me at villar@ais.uni-bonn.de.
