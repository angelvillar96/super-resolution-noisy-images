"""
Methods for denormalizing and visualizing images

Denoising_in_superresolution/src/utils
@author: Angel Villar-Corrales
"""

import numpy as np
from matplotlib import pyplot as plt


def display_images_one_row(hr_imgs, lr_imgs, recovered_imgs, savepath, dataset_name,
                           psnr, downscaling=2):
    """
    Displaying a row with images: original, corrupted, reconstructed
    """

    plt.figure()
    hr_img = denormalize_images(hr_imgs[0, :].cpu(), dataset_name)
    lr_img = denormalize_images(lr_imgs[0, :].cpu(), dataset_name)
    recovered_img = denormalize_images(recovered_imgs[0, :].cpu(), dataset_name)

    plt.subplot(2, 3, 1)
    plt.imshow(hr_img)
    plt.title("HR Image")
    plt.axis("off")

    plt.subplot(2, 3, 2)
    plt.imshow(lr_img)
    plt.title("LR Image")
    plt.axis("off")

    plt.subplot(2, 3, 3)
    plt.imshow(recovered_img)
    plt.title("Recovered")
    plt.axis("off")

    hr_patch, lr_patch, recovered_patch = extract_patches(hr_img, lr_img, recovered_img,
                                                          downscaling=downscaling)

    plt.subplot(2, 3, 4)
    plt.imshow(hr_patch)
    plt.title("HR Image")
    plt.axis("off")

    plt.subplot(2, 3, 5)
    plt.imshow(lr_patch)
    plt.title("LR Image")
    plt.axis("off")

    plt.subplot(2, 3, 6)
    plt.imshow(recovered_patch)
    plt.title("Recovered")
    plt.axis("off")

    if(psnr is not None):
        plt.suptitle(f"PSNR: {round(psnr.item(),2)}")

    plt.tight_layout()
    plt.savefig(savepath)

    return


def extract_patches(hr_img, lr_img, recovered_img, patch_size=96, downscaling=2):
    """
    Extracting patches from the HR and LR images
    """

    height, width = hr_img.shape[:2]
    idx_x = np.random.randint(low=0, high=width - patch_size)
    idx_y = np.random.randint(low=0, high=height - patch_size)

    idx_x_lr, idx_y_lr = idx_x // downscaling, idx_y // downscaling
    patch_size_lr = patch_size // downscaling

    hr_patch = hr_img[idx_y:idx_y+patch_size, idx_x:idx_x+patch_size]
    recovered_patch = recovered_img[idx_y:idx_y+patch_size, idx_x:idx_x+patch_size]
    lr_patch = lr_img[idx_y_lr:idx_y_lr+patch_size_lr, idx_x_lr:idx_x_lr+patch_size_lr]

    return hr_patch, lr_patch, recovered_patch


def display_images(hr_imgs, lr_imgs, recovered_imgs, savepath, dataset_name):
    """
    Displaying a grid with images: original, corrupted, reconstructed
    """

    plt.figure()
    for j in range(3):

        hr_img = denormalize_images(hr_imgs[j, :].cpu(), dataset_name)
        lr_img = denormalize_images(lr_imgs[j, :].cpu(), dataset_name)
        recovered_img = denormalize_images(recovered_imgs[j, :].cpu(), dataset_name)
        plt.subplot(3, 3, j+1)
        plt.imshow(hr_img)
        plt.axis("off")
        plt.subplot(3, 3, j+4)
        plt.imshow(lr_img)
        plt.axis("off")
        plt.subplot(3, 3, j+7)
        plt.imshow(recovered_img)
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(savepath)

    return


def visualize_method_comparison(hr_img, lr_img, recovered_imgs, index,
                                titles, savepath, dataset_name):
    """
    Displaying a comparison between the different methods using one image
    """

    fontsize = 10

    plt.figure()
    plt.subplot(2, 5, 1)
    plt.imshow(denormalize_images(hr_img, dataset_name))
    plt.axis("off")
    plt.title(titles[0], fontsize=fontsize)
    plt.subplot(2, 5, 2)
    plt.imshow(denormalize_images(lr_img, dataset_name))
    plt.axis("off")
    plt.title(titles[1], fontsize=fontsize)
    for i in range(len(recovered_imgs)):
        plt.subplot(2, 5, 3+i)
        plt.title(titles[2+i], fontsize=fontsize)
        plt.imshow(denormalize_images(recovered_imgs[i][index, :].cpu(), dataset_name))
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(savepath, bbox_inches='tight')

    return


def visualize_hyperparam_comparison(hr_img, lr_img, recovered_imgs, index,
                                    titles, savepath, dataset_name):
    """
    Displaying a comparison between the different methods using one image
    """

    fontsize = 10

    plt.figure()
    plt.subplot(2, 4, 1)
    plt.imshow(denormalize_images(hr_img, dataset_name))
    plt.axis("off")
    plt.title(titles[0], fontsize=fontsize)
    plt.subplot(2, 4, 2)
    plt.imshow(denormalize_images(lr_img, dataset_name))
    plt.axis("off")
    plt.title(titles[1], fontsize=fontsize)
    for i in range(len(recovered_imgs)):
        plt.subplot(2, 4, 3+i)
        plt.title(titles[2+i], fontsize=fontsize)
        plt.imshow(denormalize_images(recovered_imgs[i][index, :].cpu(), dataset_name))
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(savepath)

    return


def denormalize_images(img, dataset_name):
    """
    Destandarizing the images in order to be able to plot them

    Args:
    -----
    images: torch tensor
        tensor with the standarized images
    """

    img = img.numpy().transpose(1, 2, 0).squeeze()
    img = np.clip(img, a_min=0, a_max=1)

    return img


#
