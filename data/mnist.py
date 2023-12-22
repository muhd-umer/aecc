"""
Functions for loading MNIST dataset.
"""

import torch
import torchvision
import torchvision.transforms.v2 as v2
from termcolor import colored
from torch.utils.data.dataset import random_split

mnist_settings = {
    "default": {
        "mean": [0.1307],
        "std": [0.3081],
    },
    "norm_0to1": {"mean": [0.0], "std": [1.0]},
    "norm_neg1to1": {"mean": [0.5], "std": [0.5]},
}


def get_mnist_transform(cfg):
    """
    Get MNIST transforms.

    Args:
        cfg (dict): Configuration dictionary.

    Returns:
        tuple: (train_transform, test_transform)
    """

    cfg.mean, cfg.std = (
        mnist_settings[cfg.normalize]["mean"],
        mnist_settings[cfg.normalize]["std"],
    )

    train_transform = v2.Compose(
        [
            v2.ToImage(),
            v2.RandomHorizontalFlip(p=0.5),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=cfg.mean, std=cfg.std),
        ]
    )
    test_transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=cfg.mean, std=cfg.std),
        ]
    )
    return train_transform, test_transform


def get_mnist_loaders(
    root,
    train_transform,
    test_transform,
    batch_size=128,
    num_workers=4,
    val_size=0.1,
    shuffle=True,
    return_steps=False,
):
    """
    Get MNIST dataset.

    Args:
        root (string): Root directory of dataset.
        train_transform (callable): A function/transform that takes in an PIL image and returns a transformed version.
        test_transform (callable): A function/transform that takes
        in an PIL image and returns a transformed version.
        batch_size (int, optional): Number of samples per batch. Default is 128.
        num_workers (int, optional): Number of subprocesses to use for data loading. Default is 4.
        val_size (float, optional): If float, should be between 0.0 and 1.0
        and represent the proportion of the dataset to include in the validation split. Default is 0.1.
        shuffle (bool, optional): If True, the data will be split randomly. Default is True.
        dataset_type (str, optional): Type of dataset. Default is "default".
        return_steps (bool, optional): If True, return number of steps per epoch. Default is False.

    Returns:
        tuple: (train_dataset, val_dataset, test_dataset)
    """

    train_dataset = torchvision.datasets.MNIST(
        root=root,
        train=True,
        transform=train_transform,
        download=True,
    )
    test_dataset = torchvision.datasets.MNIST(
        root=root,
        train=False,
        transform=test_transform,
        download=True,
    )

    # Split train dataset into train and validation
    if val_size > 0.0:
        val_size = int(len(train_dataset) * val_size)
        train_size = len(train_dataset) - val_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    else:
        val_dataset = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )

    if val_size == 0.0:
        val_loader = None
        print(colored("No validation set used.", "yellow"))
    else:
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    if return_steps:
        steps_per_epoch = len(train_loader)
        return train_loader, val_loader, test_loader, steps_per_epoch
    else:
        return train_loader, val_loader, test_loader
