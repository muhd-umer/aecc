"""
Dataset class for CIFAR10.
"""
import os
import pickle

import numpy as np
import torch
import torchvision.transforms.v2 as v2
from PIL import Image
from termcolor import colored
from torch.utils.data import Dataset
from torch.utils.data.dataset import random_split


class CIFAR10(Dataset):
    """
    Dataset class for CIFAR10.

    CIFAR10 dataset is expected to have the following directory structure.
    root
    ├── cifar-10-batches-py
    │   ├── batches.meta
    │   ├── data_batch_1
    │   ├── data_batch_2
    │   ├── data_batch_3
    │   ├── data_batch_4
    │   ├── data_batch_5
    │   ├── readme.html
    └── └── test_batch

    Args:
        root (string): Root directory of dataset.
        train (bool, optional): If True, creates dataset from training set,
        otherwise creates from test set.
        transform (callable, optional): A function/transform that takes in an
        PIL image and returns a transformed version.
        noise_factor (float, optional): Noise factor.
    """

    def __init__(self, root, train=True, transform=None, noise_factor=0.2):
        self.root = os.path.expanduser(root)
        self.train = train
        self.transform = transform
        self.noise_factor = noise_factor

        if self.train:
            self.train_data, self.train_labels = self._load_data("train")
        else:
            self.test_data, self.test_labels = self._load_data("test")

        self.classes = self._load_meta()[b"label_names"]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, _ = self.train_data[index], self.train_labels[index]
        else:
            img, _ = self.test_data[index], self.test_labels[index]

        img = Image.fromarray(img).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        noisy_img = img + self.noise_factor * torch.randn_like(img)
        noisy_img = torch.clamp(noisy_img, 0.0, 1.0)

        return noisy_img, img

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _load_data(self, train_or_test):
        """
        Load data from file.
        """
        if train_or_test == "train":
            data = []
            labels = []
            for i in range(1, 6):
                data_dict = self._unpickle(
                    os.path.join(
                        self.root, "cifar-10-batches-py", "data_batch_" + str(i)
                    )
                )
                data.append(data_dict[b"data"])
                labels.append(data_dict[b"labels"])

            data = np.concatenate(data)
            labels = np.concatenate(labels)
        else:
            data_dict = self._unpickle(
                os.path.join(self.root, "cifar-10-batches-py", "test_batch")
            )
            data = data_dict[b"data"]
            labels = data_dict[b"labels"]

        data = np.vstack(data).reshape(-1, 3, 32, 32)
        data = data.transpose((0, 2, 3, 1))

        return data, labels

    def _load_meta(self):
        """
        Load meta data from file.
        """
        return self._unpickle(
            os.path.join(self.root, "cifar-10-batches-py", "batches.meta")
        )

    def _unpickle(self, file):
        """
        Unpickle file.
        """
        with open(file, "rb") as fo:
            dict = pickle.load(fo, encoding="bytes")
        return dict


def get_cifar10_dataset(
    root, train_transform=None, test_transform=None, val_size=0.1, noise_factor=0.2
):
    """
    Get CIFAR10 dataset.

    Args:
        root (string): Root directory of dataset.
        train_transform (callable, optional): A function/transform that takes
        in an PIL image and returns a transformed version.
        test_transform (callable, optional): A function/transform that takes
        in an PIL image and returns a transformed version.
        val_size (float, optional): If float, should be between 0.0 and 1.0
        and represent the proportion of the dataset to include in the validation split.
        noise_factor (float, optional): Noise factor.
    Returns:
        tuple: (train_dataset, val_dataset, test_dataset)
    """
    train_dataset = CIFAR10(
        root, train=True, transform=train_transform, noise_factor=0.2
    )
    test_dataset = CIFAR10(
        root, train=False, transform=test_transform, noise_factor=0.2
    )

    if val_size > 0.0:
        val_size = int(len(train_dataset) * val_size)
        train_size = len(train_dataset) - val_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    else:
        val_dataset = None

    return train_dataset, val_dataset, test_dataset


cifar10_settings = {
    "standard": {
        "mean": [0.4914, 0.4822, 0.4465],
        "std": [0.2023, 0.1994, 0.2010],
    },
    "default": {"mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0]},
    "neg1to1": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
}


def get_cifar10_transform(cfg):
    """
    Get CIFAR10 transforms.

    Args:
        cfg (dict): Configuration dictionary.

    Returns:
        tuple: (train_transform, test_transform)
    """

    cfg.mean, cfg.std = (
        cifar10_settings[cfg.normalize]["mean"],
        cifar10_settings[cfg.normalize]["std"],
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


def get_cifar10_loaders(
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
    Get CIFAR10 dataset.

    Args:
        root (string): Root directory of dataset.
        train_transform (callable): A function/transform that takes
        in an PIL image and returns a transformed version.
        test_transform (callable): A function/transform that takes
        in an PIL image and returns a transformed version.
        val_size (float, optional): If float, should be between 0.0 and 1.0
        and represent the proportion of the dataset to include in the validation split.
        shuffle (bool, optional): If True, the data will be split randomly.
        noise_factor (float, optional): Noise factor.
        return_steps (bool, optional): If True, return number of steps per epoch.

    Returns:
        tuple: (train_dataset, val_dataset, test_dataset)
    """

    train_dataset, val_dataset, test_dataset = get_cifar10_dataset(
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
