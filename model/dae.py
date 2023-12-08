"""
Denoising Autoencoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vit import ViT


class Decoder(nn.Module):
    def __init__(self, latent_dim, num_channels):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 8 * 7 * 7)
        self.conv1 = nn.ConvTranspose2d(
            8, 8, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.conv2 = nn.ConvTranspose2d(
            8, 16, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.conv3 = nn.Conv2d(16, num_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu(self.fc(x))
        x = x.view(-1, 8, 7, 7)  # reshape operation
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.sigmoid(self.conv3(x))
        return x


class DenoisingAutoencoder(nn.Module):
    def __init__(
        self,
        img_size,
        patch_size,
        num_channels,
        proj_dim,
        num_heads,
        dim_feedforward,
        blocks,
        mlp_units,
        latent_dim,
    ):
        super().__init__()  # Call parent's __init__ method first

        self.encoder = ViT(
            img_size,
            patch_size,
            num_channels,
            proj_dim,
            num_heads,
            dim_feedforward,
            blocks,
            mlp_units,
            latent_dim,
        )

        self.decoder = Decoder(latent_dim, num_channels)

    def forward(self, batch):
        batch = self.encoder(batch)
        batch = self.decoder(batch)
        return batch
