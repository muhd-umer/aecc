"""
MIT License:
Copyright (c) 2023 Muhammad Umer

Vision Transformer Decoder
"""

import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import Block


class ViTDecoder(nn.Module):
    """
    Vision Transformer Decoder class.

    Attributes:
        pos_embedding (nn.Parameter): Positional embedding.
        transformer (nn.Sequential): Transformer block sequence.
        head (nn.Linear): Linear layer for the output.
        patch2img (Rearrange): Layer to rearrange patches to image.
    """

    def __init__(
        self,
        in_channels=3,
        img_size=224,
        patch_size=16,
        emb_dim=768,
        num_layer=12,
        num_head=12,
        gate=nn.Sigmoid,
    ) -> None:
        """
        Initialize the ViTDecoder.

        Args:
            in_channels (int, optional): Number of input channels.
            img_size (int, optional): Size of the input image.
            patch_size (int, optional): Size of the patches.
            emb_dim (int, optional): Embedding dimension.
            num_layer (int, optional): Number of transformer layers.
            num_head (int, optional): Number of transformer heads.
            gate (nn.Module, optional): Gate function.
        """
        super().__init__()

        self.pos_embedding = nn.Parameter(
            torch.zeros((img_size // patch_size) ** 2 + 1, 1, emb_dim)
        )

        self.transformer = nn.Sequential(
            *[Block(emb_dim, num_head) for _ in range(num_layer)]
        )

        self.head = nn.Linear(emb_dim, in_channels * patch_size**2)
        self.patch2img = Rearrange(
            "(h w) b (c p1 p2) -> b c (h p1) (w p2)",
            p1=patch_size,
            p2=patch_size,
            h=img_size // patch_size,
        )
        self.gate = gate()

        self._init_weight()

    def _init_weight(self):
        """
        Initialize the weights of the pos_embedding.
        """
        trunc_normal_(self.pos_embedding, std=0.02)

    def forward(self, features):
        """
        Forward pass of the ViTDecoder.

        Args:
            features (torch.Tensor): Input features.

        Returns:
            torch.Tensor: Image after the forward pass.
        """
        features = features + self.pos_embedding

        features = rearrange(features, "t b c -> b t c")
        features = self.transformer(features)
        features = rearrange(features, "b t c -> t b c")
        features = features[1:]  # remove global feature

        patches = self.head(features)
        img = self.patch2img(patches)

        # Gate
        img = self.gate(img)

        return img
