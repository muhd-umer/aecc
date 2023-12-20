"""
Dataset class for custom Imagenette dataset.
"""
import os

import numpy as np
import torch
import torchvision.transforms.v2 as v2
from PIL import Image
from termcolor import colored
from torch.utils.data import Dataset
from torch.utils.data.dataset import random_split


class Imagenette(Dataset):
    """
    Create custom Imagenette uncategorized dataset.

    Training set: 9103 images
    Test set: 3791 images

    Total: 12894 images
    """

    def __init__(self, root, train=True, transform=None, noise_factor=0.2):
        """
        Initialize dataset.

        Args:
            root (str): Root directory path.
            train (bool, optional): If True, use 'train' mode, else 'test'. Default is True.
            transform (callable, optional): Optional transform to be applied on a sample.
            noise_factor (float, optional): Noise factor. Default is 0.2.
        """
        self.root = os.path.expanduser(root)
        self.mode = "train" if train else "test"
        self.transform = transform
        self.noise_factor = noise_factor

        self.imgs = []

        self._init_dataset()

    def __len__(self):
        return len(self.imgs)

    def _add_noise(self, img):
        # Convert image to NumPy array
        img_array = np.array(img)

        # Generate Gaussian noise
        noise = np.random.normal(loc=0, scale=self.noise_factor, size=img_array.shape)

        # Add noise to the image
        noisy_img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)

        # Convert back to Pillow image
        noisy_img = Image.fromarray(noisy_img_array)

        return noisy_img, img

    def __getitem__(self, idx):
        img_path = self.imgs[idx]

        img = Image.open(img_path).convert("RGB")
        noisy_img, img = self._add_noise(img)

        if self.transform is not None:
            noisy_img = self.transform(noisy_img)
            img = self.transform(img)

        return noisy_img, img

    def _init_dataset(self):
        """
        Initialize dataset.
        """
        self.imgs = []

        if self.mode == "train":
            self.imgs = self._get_img_paths(os.path.join(self.root, "train"))
        elif self.mode == "test":
            self.imgs = self._get_img_paths(os.path.join(self.root, "test"))

    def _get_img_paths(self, root):
        """
        Get image paths.
        """
        img_paths = []
        for img_name in os.listdir(root):
            img_path = os.path.join(root, img_name)
            img_paths.append(img_path)

        return img_paths


def get_imagenette_dataset(
    root, train_transform=None, test_transform=None, val_size=0.1, noise_factor=0.2
):
    """
    Get Imagenette dataset.

    Args:
        root (string): Root directory of dataset.
        train_transform (callable, optional): A function/transform that takes
        in an PIL image and returns a transformed version. Default is None.
        test_transform (callable, optional): A function/transform that takes
        in an PIL image and returns a transformed version. Default is None.
        val_size (float, optional): If float, should be between 0.0 and 1.0
        and represent the proportion of the dataset to include in the validation split. Default is 0.1.
        noise_factor (float, optional): Noise factor. Default is 0.2.

    Returns:
        tuple: (train_dataset, val_dataset, test_dataset)
    """
    train_dataset = Imagenette(
        root, train=True, transform=train_transform, noise_factor=noise_factor
    )
    test_dataset = Imagenette(
        root, train=False, transform=test_transform, noise_factor=noise_factor
    )

    if val_size > 0.0:
        val_size = int(len(train_dataset) * val_size)
        train_size = len(train_dataset) - val_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    else:
        val_dataset = None

    return train_dataset, val_dataset, test_dataset


def get_imagenette_transform(cfg):
    """
    Returns the transformations to be applied on the training and testing datasets.

    The transformations include conversion to image, dtype conversion, resizing,
    random rotation (for training set), random autocontrast (for training set),
    dtype conversion, and normalization.

    Args:
        cfg (dict): Config dictionary.

    Returns:
        tuple: (train_transform, test_transform)
    """

    # Transformations
    train_transform, test_transform = (
        v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.uint8, scale=True),
                v2.Resize(
                    (cfg.img_size, cfg.img_size),
                    interpolation=v2.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                v2.RandomRotation(degrees=(-30, 30)),
                v2.RandomAutocontrast(p=0.25),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=cfg.mean, std=cfg.std),
            ]
        ),
        v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.uint8, scale=True),
                v2.Resize(
                    (cfg.img_size, cfg.img_size),
                    interpolation=v2.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=cfg.mean, std=cfg.std),
            ]
        ),
    )

    return train_transform, test_transform


def get_imagenette_loaders(
    root,
    train_transform,
    test_transform,
    batch_size=128,
    num_workers=4,
    val_size=0.1,
    shuffle=True,
    noise_factor=0.2,
    return_steps=False,
):
    """
    Get CIFAR100 dataset.

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
        noise_factor (float, optional): Noise factor. Default is 0.2.
        return_steps (bool, optional): If True, return number of steps per epoch. Default is False.

    Returns:
        tuple: (train_dataset, val_dataset, test_dataset)
    """

    train_dataset, val_dataset, test_dataset = get_imagenette_dataset(
        root, train_transform, test_transform, val_size, noise_factor
    )

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
        steps_per_epoch = len(train_dataset) // batch_size
        return train_loader, val_loader, test_loader, steps_per_epoch
    else:
        return train_loader, val_loader, test_loader
