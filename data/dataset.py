"""
Dataset class for custom Imagenette dataset.
"""
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data.dataset import random_split
from torchvision import transforms


class Imagenette(Dataset):
    """
    Create custom Imagenette uncategorized dataset.

    Training set: 9103 images
    Test set: 3791 images

    Total: 12894 images
    """

    def __init__(self, root, mode, transform, noise_factor=0.2):
        """
        Args:
            root (str): Root directory path.
            mode (str): 'train' or 'test'.
            transform (callable, optional): Optional transform to be applied on a sample.
            noise_factor (float): Noise factor.
        """
        self.root = root
        self.mode = mode
        self.transform = transform
        self.noise_factor = noise_factor

        self.imgs = []

        self._init_dataset()

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]

        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        noisy_img = img + torch.randn_like(img) * self.noise_factor
        noisy_img = torch.clamp(noisy_img, 0.0, 1.0)

        return img, noisy_img

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


def get_dataloader(
    root,
    mode,
    batch_size,
    num_workers,
    shuffle=True,
    normalize=True,
    noise_factor=0.2,
    val_split=0.2,
):
    """
    Get dataloader.
    """
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            # transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ]
    )

    if normalize:
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                # transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
                ),  # ImageNet stats
            ]
        )

    dataset = Imagenette(root, mode, transform=transform, noise_factor=noise_factor)

    if mode == "train":
        train_size = int((1 - val_split) * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )

        return dataloader, val_dataloader

    elif mode == "test":
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )

        return dataloader

    else:
        raise ValueError("Invalid mode. Expected 'train' or 'test'.")
