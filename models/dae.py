"""
MIT License:
Copyright (c) 2023 Muhammad Umer

Denoising Autoencoder Vision Transformer (DAE-ViT) models.

The configurations follow the same as that described in the paper
"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale".

Reference:
https://arxiv.org/abs/2010.11929

The only difference is that the decoder part is added to the model.
"""

import torch.nn as nn

from .decoder import ViTDecoder
from .encoder import ViTEncoder


class DAEViT(nn.Module):
    """
    Denoising Autoencoder Vision Transformer class.

    Attributes:
        encoder (ViTEncoder): Encoder part of the autoencoder.
        decoder (ViTDecoder): Decoder part of the autoencoder.
    """

    def __init__(
        self,
        in_channels=3,
        img_size=224,
        patch_size=16,
        emb_dim=768,
        encoder_layer=12,
        encoder_head=12,
        decoder_layer=8,
        decoder_head=16,
    ) -> None:
        """
        Initialize the DAEViT.

        Args:
            in_channels (int, optional): Number of input channels.
            img_size (int, optional): Size of the input image.
            patch_size (int, optional): Size of the patches.
            emb_dim (int, optional): Embedding dimension.
            encoder_layer (int, optional): Number of encoder transformer layers.
            encoder_head (int, optional): Number of encoder transformer heads.
            decoder_layer (int, optional): Number of decoder transformer layers.
            decoder_head (int, optional): Number of decoder transformer heads.
        """
        super().__init__()

        self.encoder = ViTEncoder(
            in_channels,
            img_size,
            patch_size,
            emb_dim,
            encoder_layer,
            encoder_head,
        )
        self.decoder = ViTDecoder(
            in_channels,
            img_size,
            patch_size,
            emb_dim,
            decoder_layer,
            decoder_head,
        )

    def forward(self, img):
        """
        Forward pass of the DAEViT.

        Args:
            img (torch.Tensor): Input image.

        Returns:
            torch.Tensor: Predicted image after the forward pass.
        """
        features = self.encoder(img)
        predicted_img = self.decoder(features)
        return predicted_img


def dae_vit_tiny_patch16_224(**kwargs):
    model = DAEViT(
        patch_size=16,
        emb_dim=192,
        encoder_layer=12,
        encoder_head=3,
        decoder_layer=8,
        decoder_head=6,
        **kwargs
    )
    return model


def dae_vit_small_patch16_224(**kwargs):
    model = DAEViT(
        patch_size=16,
        emb_dim=384,
        encoder_layer=12,
        encoder_head=6,
        decoder_layer=8,
        decoder_head=8,
        **kwargs
    )
    return model


def dae_vit_base_patch16_224(**kwargs):
    model = DAEViT(
        patch_size=16,
        emb_dim=768,
        encoder_layer=12,
        encoder_head=12,
        decoder_layer=8,
        decoder_head=16,
        **kwargs
    )
    return model


def dae_vit_large_patch16_224(**kwargs):
    model = DAEViT(
        patch_size=16,
        emb_dim=1024,
        encoder_layer=24,
        encoder_head=16,
        decoder_layer=16,
        decoder_head=16,
        **kwargs
    )
    return model


def dae_vit_huge_patch16_224(**kwargs):
    model = DAEViT(
        patch_size=16,
        emb_dim=1280,
        encoder_layer=32,
        encoder_head=16,
        decoder_layer=16,
        decoder_head=16,
        **kwargs
    )
    return model


dae_vit_models = {
    "dae_vit_tiny": dae_vit_tiny_patch16_224,
    "dae_vit_small": dae_vit_small_patch16_224,
    "dae_vit_base": dae_vit_base_patch16_224,
    "dae_vit_large": dae_vit_large_patch16_224,
    "dae_vit_huge": dae_vit_huge_patch16_224,
}
