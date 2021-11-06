"""
Methods for computing  and handling metrics, optimizers and other hyper parameters

Denoising_in_superresolution/src/lib
@author: Angel Villar-Corrales
"""

import torch
from pytorch_msssim import ssim as _ssim
from pytorch_msssim import ms_ssim as _ms_ssim


def get_loss_stats(loss_list, message=None):
    """
    Computes loss statistics given a list of loss values

    Args:
    -----
    loss_list: List
        List containing several loss values
    message: string
        Additional message to display
    """

    if(len(loss_list) == 0):
        return

    loss_np = torch.stack(loss_list)
    avg_loss = torch.mean(loss_np)
    max_loss = torch.max(loss_np)
    min_loss = torch.min(loss_np)

    if(message is not None):
        print(message)
    print(f"Average loss: {avg_loss} -- Min Loss: {min_loss} -- Max Loss: {max_loss}")
    print("\n")

    return avg_loss


def compute_metrics(original_img, resoluted_img):
    """
    Computing differnt evaluation metrics between original and resoluted images.
    Computes MSE, MAE, PSNR, SSIM & MS_SSIM

    Args:
    -----
    original_img: batch of torch Tensors
        batch containing the input images
    resoluted_img: batch of torch Tensors
        batch containing the tensors at the output of the network

    Returns:
    --------
    metrics: dictionary
        dict with the mean value for each metric
    """
    mse_val = mean_squared_error(original_img, resoluted_img)
    mae_val = mean_absoulte_error(original_img, resoluted_img)
    psnr_val = psnr(original_img, resoluted_img)
    ssim_val = ssim(original_img, resoluted_img)
    msssim_val = ms_ssim(original_img, resoluted_img)
    metrics = {
            "mse": mse_val,
            "mae": mae_val,
            "psnr": psnr_val,
            "ssim": ssim_val,
            "ms_ssim": msssim_val
        }
    return metrics


def mean_squared_error(original_img, resoluted_img):
    """
    Computing the mean squared error between original and resoluted images

    Args:
    -----
    original_img: batch of torch Tensors
        batch containing the input images
    resoluted_img: batch of torch Tensors
        batch containing the tensors at the output of the network

    Returns:
    --------
    mse: float
        mean squared error of the image
    """

    subs = original_img - resoluted_img
    mse = subs.pow(2).mean()

    return mse


def mean_absoulte_error(original_img, resoluted_img):
    """
    Computing the mean absolute error between original and resoluted images

    Args:
    -----
    original_img: batch of torch Tensors
        batch containing the input images
    resoluted_img: batch of torch Tensors
        batch containing the tensors at the output of the network

    Returns:
    --------
    mae: float
        mean absolute error of the image
    """

    subs = original_img - resoluted_img
    vals = torch.abs(subs)
    mae = torch.mean(vals)

    return mae


def psnr(original_img, resoluted_img):
    """
    Computing the peak signal to noise ration between original and resoluted images

    Args:
    -----
    original_img: batch of torch Tensors
        batch containing the input images
    resoluted_img: batch of torch Tensors
        batch containing the tensors at the output of the network

    Returns:
    --------
    psnr: float
        peak to signal noise ratio (in dB)
    """

    # fisrt computing th emse
    resoluted_img = (resoluted_img * 255).round().clamp(0, 255) / 255
    subs = original_img - resoluted_img
    mse = subs.pow(2).mean([-3, -2, -1])

    # formular for psnr (https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio)
    max_val = 1.0
    psnr = 20*torch.log10(torch.tensor(max_val)) - 10*torch.log10(mse)
    idx = torch.where(psnr > 120)[0]  # avoid overflow
    psnr[idx] = 120
    psnr = psnr.mean()

    return psnr


def ssim(original_img, resoluted_img):
    """
    Computing the Structural Similarity (SSIM) between original and resoluted images
    See https://github.com/VainF/pytorch-msssim

    Args:
    -----
    original_img: batch of torch Tensors
        batch containing the input images
    resoluted_img: batch of torch Tensors
        batch containing the tensors at the output of the network

    Returns:
    --------
    ssim_val: float
        mean structural similarity value
    """
    ssim_vals = _ssim(original_img, resoluted_img, data_range=255)
    ssim_val = ssim_vals.mean()
    return ssim_val


def ms_ssim(original_img, resoluted_img):
    """
    Computing the Multi-Scale Structural Similarity (MS-SSIM) between original and resoluted images
    See https://github.com/VainF/pytorch-msssim
    Args:
    -----
    original_img: batch of torch Tensors
        batch containing the input images
    resoluted_img: batch of torch Tensors
        batch containing the tensors at the output of the network

    Returns:
    --------
    msssim_val: float
        mean multi-scale structural similarity value
    """
    msssim_vals = _ms_ssim(original_img, resoluted_img, data_range=255)
    msssim_val = msssim_vals.mean()
    return msssim_val


def norm_img(img):
    """ """
    img = (img - 0.5) * 2
    return img


def denorm_img(img):
    """ """
    img = img * 0.5 + 0.5
    return img

#
