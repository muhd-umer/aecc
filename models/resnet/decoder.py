"""
MIT License:
Copyright (c) 2023 Muhammad Umer

ResNet Decoder
"""

import torch.nn as nn

from .blocks import DecoderBottleneckBlock, DecoderResidualBlock


class ResNetDecoder(nn.Module):
    """
    This class represents the ResNet Decoder module.
    It consists of several convolutional layers and a final gating layer.
    """

    def __init__(self, cfg, bottleneck=False):
        """
        Initialize the ResNetDecoder.

        Args:
            cfg (list): A list of configurations for each layer.
            bottleneck (bool): If True, use bottleneck blocks. Otherwise, use residual blocks.
        """
        super(ResNetDecoder, self).__init__()

        if len(cfg) != 4:
            raise ValueError("Only 4 layers can be configured")

        self._initialize_layers(cfg, bottleneck)

        self.gate = nn.Sigmoid()

    def _initialize_layers(self, cfg, bottleneck):
        """
        Initialize the convolutional layers based on the given configurations.

        Args:
            cfg (list): A list of configurations for each layer.
            bottleneck (bool): If True, use bottleneck blocks. Otherwise, use residual blocks.
        """
        block = DecoderBottleneckBlock if bottleneck else DecoderResidualBlock

        self.conv1 = block(
            in_channels=2048 if bottleneck else 512,
            hidden_channels=512,
            down_channels=1024 if bottleneck else 256,
            layers=cfg[0],
        )
        self.conv2 = block(
            in_channels=1024 if bottleneck else 256,
            hidden_channels=256,
            down_channels=512 if bottleneck else 128,
            layers=cfg[1],
        )
        self.conv3 = block(
            in_channels=512 if bottleneck else 128,
            hidden_channels=128,
            down_channels=256 if bottleneck else 64,
            layers=cfg[2],
        )
        self.conv4 = block(
            in_channels=256 if bottleneck else 64,
            hidden_channels=64,
            down_channels=64 if bottleneck else 64,
            layers=cfg[3],
        )

        self.conv5 = nn.Sequential(
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                in_channels=64,
                out_channels=3,
                kernel_size=7,
                stride=2,
                padding=3,
                output_padding=1,
                bias=False,
            ),
        )

    def forward(self, x):
        """
        Forward pass through the ResNetDecoder.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.gate(x)

        return x
