"""
MIT License:
Copyright (c) 2023 Muhammad Umer

Vision Transformer Encoder
"""

import torch
import torch.nn as nn
from einops import rearrange
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import Block


class ViTEncoder(nn.Module):
    """
    Vision Transformer Encoder class.

    Attributes:
        cls_token (nn.Parameter): Class token.
        pos_embedding (nn.Parameter): Positional embedding.
        patchify (nn.Conv2d): Convolutional layer to create patches.
        transformer (nn.Sequential): Transformer block sequence.
        layer_norm (nn.LayerNorm): Layer normalization.
    """

    def __init__(
        self,
        in_channels=3,
        img_size=224,
        patch_size=16,
        emb_dim=768,
        num_layer=12,
        num_head=12,
    ) -> None:
        """
        Initialize the ViTEncoder.

        Args:
            in_channels (int, optional): Number of input channels.
            img_size (int, optional): Size of the input image.
            patch_size (int, optional): Size of the patches.
            emb_dim (int, optional): Embedding dimension.
            num_layer (int, optional): Number of transformer layers.
            num_head (int, optional): Number of transformer heads.
        """
        super().__init__()

        # self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = nn.Parameter(
            torch.zeros((img_size // patch_size) ** 2, 1, emb_dim)
        )

        self.patchify = nn.Conv2d(in_channels, emb_dim, patch_size, patch_size)

        self.transformer = nn.Sequential(
            *[Block(emb_dim, num_head) for _ in range(num_layer)]
        )

        self.layer_norm = nn.LayerNorm(emb_dim)

        self._init_weight()

    def _init_weight(self):
        """
        Initialize the weights of the cls_token and pos_embedding.
        """
        trunc_normal_(self.cls_token, std=0.02)
        trunc_normal_(self.pos_embedding, std=0.02)

    def forward(self, img):
        """
        Forward pass of the ViTEncoder.

        Args:
            img (torch.Tensor): Input image.

        Returns:
            torch.Tensor: Features after the forward pass.
        """
        patches = self.patchify(img)
        patches = rearrange(patches, "b c h w -> (h w) b c")
        patches = patches + self.pos_embedding

        # patches = torch.cat(
        #     [self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0
        # )
        patches = rearrange(patches, "t b c -> b t c")
        features = self.layer_norm(self.transformer(patches))
        features = rearrange(features, "b t c -> t b c")

        return features
