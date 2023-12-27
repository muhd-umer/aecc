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

import torch
import torch.nn as nn

from utils import RayleighChannel

from .decoder import ViTDecoder
from .encoder import ViTEncoder


class DAEViT(nn.Module):
    """
    DAEViT is a Denoising AutoEncoder model that leverages the Vision Transformer (ViT)
    architecture. It consists of an encoder and a decoder, both of which are based on the ViT model.

    Attributes:
        encoder (ViTEncoder): The encoder part of the autoencoder. It is responsible for
        transforming the input image into a lower-dimensional representation.
        decoder (ViTDecoder): The decoder part of the autoencoder. It reconstructs the original image from the lower-dimensional representation produced by the encoder.

    The DAEViT model is initialized with several parameters that define the structure and behavior
    of the encoder and decoder. These parameters include the number of input channels, the size of
    the input image, the size of the patches, the embedding dimension, the number of transformer
    layers and heads in the encoder and decoder, the gate function for the decoder, and the noise
    factor for the input image.
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
        gate=nn.Sigmoid,
        noise_factor=0.2,
    ) -> None:
        """
        Initializes the DAEViT model with the given parameters.

        Args:
            in_channels (int, optional): The number of channels in the input image.
            img_size (int, optional): The size (height and width) of the input image in pixels.
            patch_size (int, optional): The size of the patches that the image is divided into for
            the ViT model.
            emb_dim (int, optional): The dimension of the embeddings in the ViT model.
            encoder_layer (int, optional): The number of transformer layers in the encoder.
            encoder_head (int, optional): The number of transformer heads in the encoder.
            decoder_layer (int, optional): The number of transformer layers in the decoder.
            decoder_head (int, optional): The number of transformer heads in the decoder.
            gate (nn.Module, optional): The gate function used in the decoder.
            noise_factor (float, optional): The factor by which the input image is noised before
            being passed to the encoder.
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
            gate,
        )

        self.rayleigh = RayleighChannel(noise_factor)

    def forward(self, img):
        """
        Forward pass of the DAEViT.

        Args:
            img (torch.Tensor): Input image.

        Returns:
            torch.Tensor: Predicted image after the forward pass.
        """
        features = self.encoder(img)
        noisy_features = self.rayleigh(features)
        predicted_img = self.decoder(noisy_features)

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
