"""
MIT License:
Copyright (c) 2023 Muhammad Umer

ResNet Encoder
"""

import torch.nn as nn

from .blocks import EncoderBottleneckBlock, EncoderResidualBlock


class ResNetEncoder(nn.Module):
    """
    This class represents the ResNet Encoder module.
    It consists of several convolutional layers.
    """

    def __init__(self, cfg, bottleneck=False, in_channels=3):
        """
        Initialize the ResNetEncoder.

        Args:
            cfg (list): A list of configurations for each layer.
            bottleneck (bool): If True, use bottleneck blocks. Otherwise, use residual blocks.
            in_channels (int): The number of input channels.
        """
        super(ResNetEncoder, self).__init__()

        if len(cfg) != 4:
            raise ValueError("Only 4 layers can be configured")

        self.in_channels = in_channels
        self._initialize_layers(cfg, bottleneck)

    def _initialize_layers(self, cfg, bottleneck):
        """
        Initialize the convolutional layers based on the given configurations.

        Args:
            cfg (list): A list of configurations for each layer.
            bottleneck (bool): If True, use bottleneck blocks. Otherwise, use residual blocks.
        """
        block = EncoderBottleneckBlock if bottleneck else EncoderResidualBlock

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
        )

        self.conv2 = block(
            in_channels=64,
            hidden_channels=64,
            up_channels=256 if bottleneck else 64,
            layers=cfg[0],
            downsample_method="pool",
        )
        self.conv3 = block(
            in_channels=256 if bottleneck else 64,
            hidden_channels=128,
            up_channels=512 if bottleneck else 128,
            layers=cfg[1],
            downsample_method="conv",
        )
        self.conv4 = block(
            in_channels=512 if bottleneck else 128,
            hidden_channels=256,
            up_channels=1024 if bottleneck else 256,
            layers=cfg[2],
            downsample_method="conv",
        )
        self.conv5 = block(
            in_channels=1024 if bottleneck else 256,
            hidden_channels=512,
            up_channels=2048 if bottleneck else 512,
            layers=cfg[3],
            downsample_method="conv",
        )

    def forward(self, x):
        """
        Forward pass through the ResNetEncoder.

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

        return x
