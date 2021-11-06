"""
Classes and methods to add noise and impurities to the original images before training

Denoising_in_superresolution/src/lib
@author: Angel Villar-Corrales
"""

import numpy as np
from skimage.transform import rescale
from skimage.filters import gaussian


class Noiser:
    """
    Class that adds noise to the original images

    Args:
    -----
    noise: string
        name of the noise to be used to corrupt the images
    std: float
        standard deviation of the noise
    """

    def __init__(self, noise="", std=0):
        """
        Initializer of the Noiser object
        """
        self.noise = noise
        self.std = std
        return

    def __call__(self, img):
        """
        Method that adds the noise to the original image

        Args:
        -----
        img: numpy array
            original image to which the noise will be added

        Returns:
        --------
        corrupted_img: numpy array
            Image to which the noise has been added. It has the same shape as the input image
        """

        size = img.shape

        if(self.noise == ""):
            corrupted_img = img

        elif(self.noise == "gaussian"):
            noise = np.random.normal(loc=0, scale=self.std, size=size)
            corrupted_img = img + noise

        elif(self.noise == "salt_pepper"):
            probs = np.random.uniform(low=0, high=1, size=size[0]*size[1])
            salt = np.where(probs > (1-self.std/2))[0]
            pepper = np.where(probs < self.std/2)[0]
            salt = self._to_matrix(salt, shape=size[:-1])
            pepper = self._to_matrix(pepper, shape=size[:-1])
            corrupted_img = np.copy(img)
            corrupted_img[salt[:, 0], salt[:, 1], :] = 1
            corrupted_img[pepper[:, 0], pepper[:, 1], :] = 0

        elif(self.noise == "poisson"):
            noise = np.random.poisson(lam=self.std, size=size)
            corrupted_img = img + noise

        elif(self.noise == "speckle"):
            noise = np.random.normal(loc=0, scale=self.std, size=size)
            corrupted_img = img + img * noise

        else:
            message = (f"""So far, only ['', 'gaussian', 'poisson', 'speckle', 'salt&pepper'] noise
                        distributions are allowed and you have requested {self.noise}...""")
            raise NotImplementedError(message)

        # converting the corrupted image to range [-1,1]
        corrupted_img = self._transform_values(corrupted_img)
        return corrupted_img

    def _to_matrix(self, idx, shape):
        """
        Converting flattened indices into 2d matrix indicies

        Args:
        -----
        idx: numpy array
            array with indicies in the range [0, N*M]
        shape: tuple
            dimensions of the matrix (N,M) we convert the indices to

        Returns:
        --------
        matrix_idx: numpy array
            input indices converted to desired matrix shape
        """
        matrix_idx = np.zeros((idx.shape[0], 2))
        matrix_idx[:, 0] = idx % shape[0]
        matrix_idx[:, 1] = idx // shape[1]
        return matrix_idx.astype(int)

    def _transform_values(self, img, mean=0.5, var=0.5):
        """
        Transforming the noise values to be in the range [-1,1]
        """
        img = np.clip(img, 0, 1)
        img = (img - mean) / var
        return img


class Blur:
    """
    Class to blurr and downscale the original images

    Args:
    -----
    downscaling: integer
        factor by which the images will be downscaled and the upsampled
    """

    def __init__(self, downscaling=1):
        """
        Initializer of the blur object
        """
        self.downscaling = downscaling
        return

    def __call__(self, hr_img):
        """
        Blurring the image by the given factor using a gaussian filter

        Args:
        -----
        hr_img: numpy array
            image from the dataset (possibly noisy and quantized), but of high resolution

        Returns:
        --------
        lr_img: numpy array
            low resolution image
        """

        img = gaussian(hr_img, sigma=self.downscaling/2, multichannel=True)
        lr_img = rescale(img, 1/self.downscaling, multichannel=True)

        return lr_img


#
