"""
Wide activation for super resolution network

Denoising_in_superresolution/src/models

Adapted from: https://github.com/ychfan/wdsr
"""

import math

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim


class MODEL(nn.Module):

  def __init__(self, params):
    super(MODEL, self).__init__()
    self.temporal_size = None
    self.image_mean = 0
    kernel_size = 3
    skip_kernel_size = 5
    weight_norm = torch.nn.utils.weight_norm
    num_inputs = params.n_colors

    num_outputs = params.scale * params.scale * params.n_colors

    # prenetwork
    prenetwork = []
    self.prenetwork = nn.Sequential(*prenetwork)

    body = []
    conv = weight_norm(
        nn.Conv2d(
            num_inputs,
            params.n_feats,
            kernel_size,
            padding=kernel_size // 2))
    init.ones_(conv.weight_g)
    init.zeros_(conv.bias)
    body.append(conv)
    for _ in range(params.n_resblocks):
      body.append(
          Block(
              params.n_feats,
              kernel_size,
              4,
              weight_norm=weight_norm,
              res_scale=1 / math.sqrt(params.n_resblocks),
          ))
    conv = weight_norm(
        nn.Conv2d(
            params.n_feats,
            num_outputs,
            kernel_size,
            padding=kernel_size // 2))
    init.ones_(conv.weight_g)
    init.zeros_(conv.bias)
    body.append(conv)
    self.body = nn.Sequential(*body)

    skip = []
    if num_inputs != num_outputs:
      conv = weight_norm(
          nn.Conv2d(
              num_inputs,
              num_outputs,
              skip_kernel_size,
              padding=skip_kernel_size // 2))
      init.ones_(conv.weight_g)
      init.zeros_(conv.bias)
      skip.append(conv)
    self.skip = nn.Sequential(*skip)

    shuf = []
    if params.scale > 1:
      shuf.append(nn.PixelShuffle(params.scale))
    self.shuf = nn.Sequential(*shuf)

  def forward(self, x):

      x = self.prenetwork(x)
      # output_denoiser_prenetwork = x.clone()

      x = self.body(x) + self.skip(x)
      x = self.shuf(x)

      return x


class Block(nn.Module):

  def __init__(self,
               num_residual_units,
               kernel_size,
               width_multiplier=1,
               weight_norm=torch.nn.utils.weight_norm,
               res_scale=1):
    super(Block, self).__init__()
    body = []
    conv = weight_norm(
        nn.Conv2d(
            num_residual_units,
            int(num_residual_units * width_multiplier),
            kernel_size,
            padding=kernel_size // 2))
    init.constant_(conv.weight_g, 2.0)
    init.zeros_(conv.bias)
    body.append(conv)
    body.append(nn.ReLU(True))
    conv = weight_norm(
        nn.Conv2d(
            int(num_residual_units * width_multiplier),
            num_residual_units,
            kernel_size,
            padding=kernel_size // 2))
    init.constant_(conv.weight_g, res_scale)
    init.zeros_(conv.bias)
    body.append(conv)

    self.body = nn.Sequential(*body)

  def forward(self, x):
    x = self.body(x) + x
    return x
