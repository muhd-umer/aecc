"""
MIT License:
Copyright (c) 2023 Muhammad Umer

ResNet Blocks
"""

import torch.nn as nn

from .layers import (
    DecoderBottleneckLayer,
    DecoderResidualLayer,
    EncoderBottleneckLayer,
    EncoderResidualLayer,
)


class EncoderResidualBlock(nn.Module):
    """
    A class used to represent the Encoder Residual Block in a neural network.
    """

    def __init__(
        self,
        in_channels,
        hidden_channels,
        up_channels,  # placeholder
        layers,
        downsample_method="conv",
    ):
        """
        Initialize the EncoderResidualBlock.
        """
        super(EncoderResidualBlock, self).__init__()

        if downsample_method == "conv":
            self._add_layers(
                in_channels,
                hidden_channels,
                layers,
                EncoderResidualLayer,
                downsample_first=True,
            )
        elif downsample_method == "pool":
            self.add_module(
                "00 MaxPooling", nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
            self._add_layers(
                in_channels,
                hidden_channels,
                layers,
                EncoderResidualLayer,
                downsample_first=False,
            )

    def _add_layers(
        self, in_channels, hidden_channels, layers, layer_type, downsample_first
    ):
        """
        Add layers to the block.
        """
        for i in range(layers):
            downsample = True if i == 0 and downsample_first else False
            layer = layer_type(
                in_channels=in_channels if i == 0 else hidden_channels,
                hidden_channels=hidden_channels,
                downsample=downsample,
            )
            self.add_module(f"{i:02d} {layer_type.__name__}", layer)

    def forward(self, x):
        """
        Forward pass through the block.
        """
        for _, layer in self.named_children():
            x = layer(x)

        return x


class EncoderBottleneckBlock(nn.Module):
    """
    A class used to represent the Encoder Bottleneck Block in a neural network.
    """

    def __init__(
        self,
        in_channels,
        hidden_channels,
        up_channels,
        layers,
        downsample_method="conv",
    ):
        """
        Initialize the EncoderBottleneckBlock.
        """
        super(EncoderBottleneckBlock, self).__init__()

        if downsample_method == "conv":
            self._add_layers(
                in_channels,
                hidden_channels,
                up_channels,
                layers,
                EncoderBottleneckLayer,
                downsample_first=True,
            )
        elif downsample_method == "pool":
            self.add_module(
                "00 MaxPooling", nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
            self._add_layers(
                in_channels,
                hidden_channels,
                up_channels,
                layers,
                EncoderBottleneckLayer,
                downsample_first=False,
            )

    def _add_layers(
        self,
        in_channels,
        hidden_channels,
        up_channels,
        layers,
        layer_type,
        downsample_first,
    ):
        """
        Add layers to the block.
        """
        for i in range(layers):
            downsample = True if i == 0 and downsample_first else False
            layer = layer_type(
                in_channels=in_channels if i == 0 else up_channels,
                hidden_channels=hidden_channels,
                up_channels=up_channels,
                downsample=downsample,
            )
            self.add_module(f"{i:02d} {layer_type.__name__}", layer)

    def forward(self, x):
        """
        Forward pass through the block.
        """
        for _, layer in self.named_children():
            x = layer(x)

        return x


class DecoderResidualBlock(nn.Module):
    """
    A class used to represent the Decoder Residual Block in a neural network.
    """

    def __init__(self, in_channels, hidden_channels, inter_channels, layers):
        """
        Initialize the DecoderResidualBlock.
        """
        super(DecoderResidualBlock, self).__init__()

        self._add_layers(hidden_channels, inter_channels, layers, DecoderResidualLayer)

    def _add_layers(self, hidden_channels, inter_channels, layers, layer_type):
        """
        Add layers to the block.
        """
        for i in range(layers):
            upsample = True if i == layers - 1 else False
            layer = layer_type(
                hidden_channels=hidden_channels,
                inter_channels=inter_channels if upsample else hidden_channels,
                upsample=upsample,
            )
            self.add_module(f"{i:02d} {layer_type.__name__}", layer)

    def forward(self, x):
        """
        Forward pass through the block.
        """
        for _, layer in self.named_children():
            x = layer(x)

        return x


class DecoderBottleneckBlock(nn.Module):
    """
    A class used to represent the Decoder Bottleneck Block in a neural network.
    """

    def __init__(self, in_channels, hidden_channels, inter_channels, layers):
        """
        Initialize the DecoderBottleneckBlock.
        """
        super(DecoderBottleneckBlock, self).__init__()

        self._add_layers(
            in_channels, hidden_channels, inter_channels, layers, DecoderBottleneckLayer
        )

    def _add_layers(
        self, in_channels, hidden_channels, inter_channels, layers, layer_type
    ):
        """
        Add layers to the block.
        """
        for i in range(layers):
            upsample = True if i == layers - 1 else False
            layer = layer_type(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                inter_channels=inter_channels if upsample else in_channels,
                upsample=upsample,
            )
            self.add_module(f"{i:02d} {layer_type.__name__}", layer)

    def forward(self, x):
        """
        Forward pass through the block.
        """
        for _, layer in self.named_children():
            x = layer(x)

        return x
