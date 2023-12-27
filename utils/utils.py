"""
MIT License:
Copyright (c) 2023 Muhammad Umer

Utils for PyTorch Lightning
"""

import sys

import numpy as np
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import TQDMProgressBar
from torch.utils.data import DataLoader
from tqdm import tqdm


def numpy_collate(batch):
    """
    Collate function to use PyTorch datalaoders
    Reference:
    https://jax.readthedocs.io/en/latest/notebooks/Neural_Network_and_Data_Loading.html
    """
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


class SimplifiedProgressBar(TQDMProgressBar):
    """
    Simplified progress bar for non-interactive terminals.
    """

    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        if not sys.stdout.isatty():
            bar.disable = True
        return bar

    def init_predict_tqdm(self):
        bar = super().init_predict_tqdm()
        if not sys.stdout.isatty():
            bar.disable = True
        return bar

    def init_test_tqdm(self):
        bar = super().init_test_tqdm()
        if not sys.stdout.isatty():
            bar.disable = True
        return bar


def get_mean_std(loader: DataLoader):
    """
    Calculate the mean and standard deviation of the data in the loader.

    Args:
        loader (DataLoader): The DataLoader for which the mean and std are calculated.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The mean and standard deviation of the data in the loader.
    """
    # var[X] = E[X**2] - E[X]**2
    channels_sum, channels_sqrd_sum, num_batches = 0, 0, 0

    for data, _ in tqdm(loader):
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_sqrd_sum += torch.mean(data**2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_sqrd_sum / num_batches - mean**2) ** 0.5

    return mean, std


class RayleighChannel(nn.Module):
    """A class used to represent the Rayleigh Channel.

    Attributes:
        sigma (float): a scaling factor for the noise added to the input.
    """

    def __init__(self, sigma):
        """Constructs all the necessary attributes for the Rayleigh Channel object.

        Args:
            sigma (float): a scaling factor for the noise added to the input.
        """
        super().__init__()
        self.sigma = sigma

    def forward(self, x):
        """Applies the Rayleigh Channel transformation to the input tensor.

        Args:
            x (torch.Tensor): input tensor.

        Returns:
            torch.Tensor: output tensor after adding Rayleigh noise.
        """
        return x + self.sigma * torch.abs(
            (torch.randn_like(x) + 1j * torch.randn_like(x))
            / torch.sqrt(torch.tensor(2.0))
        )
