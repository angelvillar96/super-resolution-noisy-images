# Code Structure

This README describes the functionality of the different scripts included in the project.

All classes and methods in the project have their corresponding *docstring* explaining their functionality, input arguments and returned values. I refer to the *docstring* for particular documentation of each method.


* [1. Necessary Scripts](#necessary-scripts)
* [2. Metrics and Comparison Scripts](#metrics-and-comparison-scripts)
* [3. Auxiliary Scripts](#auxiliary-scripts)
* [4. Hints](#hints)



## Necessary Scripts

The files explained in this section correspond to the ones whose execution is necessary for some step of the super-resolution and denoising pipeline.

 - **config.py**: Determines the paths where experiments, datasets and evaluations will be saved. Furthermore, it defines the seed for the random number generator.

 - **01_create_experiment.py**: Creates an experiment folder under the */experiments* directory. The experiment parameters are given as command line arguments and then saved in the experiment directory as a JSON file.

 - **02_train_denoiser_autoencoder.py**: If the denoising method selected in the experiment is *denoiser_autoencoder*, this script must be run prior to training the SR model. This file instanciates a denoising autoencoder and trains it to denoise low-resolution images. The trained model is then saved in the */models/autoencoder* folder in the experiment directory.

 - **02_train_model.py**: Trains the SR model, joined with the corresponding denoiser, for the number of epochs specified in the
*experiment_parameters* JSON file. Furthermore, the train and validation loss and metrics will be saved (each epoch) to the *training_logs.json* file, checkpoint models will be saved every 10 epochs, and a figure assessing the quality of reconstruction will be created in the */plots/valid_plots* directory every 5 epochs.

- **03_evaluate_model.py**: Loads a model checkpoint given the epoch number and evaluates its performance on the test set.

- **03_test_accross_epochs.py**: Iteratively loads the model checkpoints and evaluates them on the test set. The test loss and metrics are stored in the *training_logs.json*. This file iteratively calls *03_evaluate_model.py*.

- **04_apply_pretrained.py**: Loads the checkpoint with the best validation loss and performs denoising and super-resolution for a few test-set images, which are then saved in the */plots/test_plots* directory. This file illustrates how to load one of our pretrained models to use it in practice.

- **aux_generate_network.txt**: Generates a *.txt* file with the architecture and a detail description of a network. This can be used as a sanity check.

- **aux_generate_subset_data.txt**: Generates a *.png* with a few HR images and their noisy LR counterparts. Can be used to check the difficulty of the dataset prior to training.


## Hints

   - Run `METHOD.__doc__` or `help(METHOD)` to display the documentation of a method or class.

 ```python
 >>> print(noisers.Noiser.__call__.__doc__)

         Method that adds the noise to the original image

         Args:
         -----
         img: numpy array
             original image to which the noise will be added

         Returns:
         --------
         corrupted_img: numpy array
             Image to which the noise has been added.
             It has the same shape as the input image
 ```

- Run `python FILE_NAME.py --help` to display the command line arguments of the given script.

  ```shell
    $ python 03_evaluate_model.py --help

       usage: 03_evaluate_model.py [-h] [-d EXP_DIRECTORY] [-e EPOCH]
                                   [--noise NOISE]  [--std STD]

       optional arguments:
         -h, --help            
               show this help message and exit
         -d EXP_DIRECTORY, --exp_directory EXP_DIRECTORY
               Path to the experiment directory
         -e EPOCH, --epoch EPOCH
               Epoch of the checkpoint to process
         --noise NOISE         
               Type of noise to corrupt the test-set images ['', 'gaussian', 'poisson', 'speckle', 'salt_pepper']
         --std STD
               Standard deviation of the noise
 ```
