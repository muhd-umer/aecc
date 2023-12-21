"""
MIT License:
Copyright (c) 2023 Muhammad Umer

ResNet Layers
"""

import torch.nn as nn


class EncoderResidualLayer(nn.Module):
    """
    This class represents a residual layer for an encoder in a ResNet architecture.
    """

    def __init__(self, in_channels, hidden_channels, downsample):
        """
        Initialize the EncoderResidualLayer.

        Args:
            in_channels (int): Number of input channels.
            hidden_channels (int): Number of hidden channels.
            downsample (bool): Whether to downsample the input.
        """
        super(EncoderResidualLayer, self).__init__()

        if downsample:
            self.weight_layer1 = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=hidden_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
            )
        else:
            self.weight_layer1 = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=hidden_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
            )

        self.weight_layer2 = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=hidden_channels),
        )

        if downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=hidden_channels,
                    kernel_size=1,
                    stride=2,
                    padding=0,
                    bias=False,
                ),
                nn.BatchNorm2d(num_features=hidden_channels),
            )
        else:
            self.downsample = None

        self.relu = nn.Sequential(nn.ReLU(inplace=True))

    def forward(self, x):
        """
        Forward pass of the EncoderResidualLayer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        identity = x

        x = self.weight_layer1(x)
        x = self.weight_layer2(x)

        if self.downsample is not None:
            identity = self.downsample(identity)

        x = x + identity

        x = self.relu(x)

        return x


class EncoderBottleneckLayer(nn.Module):
    """
    This class represents a bottleneck layer for an encoder in a ResNet architecture.
    """

    def __init__(self, in_channels, hidden_channels, up_channels, downsample):
        """
        Initialize the EncoderBottleneckLayer.

        Args:
            in_channels (int): Number of input channels.
            hidden_channels (int): Number of hidden channels.
            up_channels (int): Number of upsampled channels.
            downsample (bool): Whether to downsample the input.
        """
        super(EncoderBottleneckLayer, self).__init__()

        if downsample:
            self.weight_layer1 = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=hidden_channels,
                    kernel_size=1,
                    stride=2,
                    padding=0,
                    bias=False,
                ),
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
            )
        else:
            self.weight_layer1 = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=hidden_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                ),
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
            )

        self.weight_layer2 = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=hidden_channels),
            nn.ReLU(inplace=True),
        )

        self.weight_layer3 = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_channels,
                out_channels=up_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=up_channels),
        )

        if downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=up_channels,
                    kernel_size=1,
                    stride=2,
                    padding=0,
                    bias=False,
                ),
                nn.BatchNorm2d(num_features=up_channels),
            )
        elif in_channels != up_channels:
            self.downsample = None
            self.up_scale = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=up_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                ),
                nn.BatchNorm2d(num_features=up_channels),
            )
        else:
            self.downsample = None
            self.up_scale = None

        self.relu = nn.Sequential(nn.ReLU(inplace=True))

    def forward(self, x):
        """
        Forward pass of the EncoderBottleneckLayer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        identity = x

        x = self.weight_layer1(x)
        x = self.weight_layer2(x)
        x = self.weight_layer3(x)

        if self.downsample is not None:
            identity = self.downsample(identity)
        elif self.up_scale is not None:
            identity = self.up_scale(identity)

        x = x + identity

        x = self.relu(x)

        return x


class DecoderResidualLayer(nn.Module):
    """
    This class represents a residual layer for a decoder in a ResNet architecture.
    """

    def __init__(self, hidden_channels, output_channels, upsample):
        """
        Initialize the DecoderResidualLayer.

        Args:
            hidden_channels (int): Number of hidden channels.
            output_channels (int): Number of output channels.
            upsample (bool): Whether to upsample the input.
        """
        super(DecoderResidualLayer, self).__init__()

        self.weight_layer1 = nn.Sequential(
            nn.BatchNorm2d(num_features=hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
        )

        if upsample:
            self.weight_layer2 = nn.Sequential(
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(
                    in_channels=hidden_channels,
                    out_channels=output_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                    bias=False,
                ),
            )
        else:
            self.weight_layer2 = nn.Sequential(
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    in_channels=hidden_channels,
                    out_channels=output_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                ),
            )

        if upsample:
            self.upsample = nn.Sequential(
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(
                    in_channels=hidden_channels,
                    out_channels=output_channels,
                    kernel_size=1,
                    stride=2,
                    output_padding=1,
                    bias=False,
                ),
            )
        else:
            self.upsample = None

    def forward(self, x):
        """
        Forward pass of the DecoderResidualLayer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        identity = x

        x = self.weight_layer1(x)
        x = self.weight_layer2(x)

        if self.upsample is not None:
            identity = self.upsample(identity)

        x = x + identity

        return x


class DecoderBottleneckLayer(nn.Module):
    """
    This class represents a bottleneck layer for a decoder in a ResNet architecture.
    """

    def __init__(self, in_channels, hidden_channels, down_channels, upsample):
        """
        Initialize the DecoderBottleneckLayer.

        Args:
            in_channels (int): Number of input channels.
            hidden_channels (int): Number of hidden channels.
            down_channels (int): Number of downsampled channels.
            upsample (bool): Whether to upsample the input.
        """
        super(DecoderBottleneckLayer, self).__init__()

        self.weight_layer1 = nn.Sequential(
            nn.BatchNorm2d(num_features=in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=hidden_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
        )

        self.weight_layer2 = nn.Sequential(
            nn.BatchNorm2d(num_features=hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
        )

        if upsample:
            self.weight_layer3 = nn.Sequential(
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(
                    in_channels=hidden_channels,
                    out_channels=down_channels,
                    kernel_size=1,
                    stride=2,
                    output_padding=1,
                    bias=False,
                ),
            )
        else:
            self.weight_layer3 = nn.Sequential(
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    in_channels=hidden_channels,
                    out_channels=down_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                ),
            )

        if upsample:
            self.upsample = nn.Sequential(
                nn.BatchNorm2d(num_features=in_channels),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=down_channels,
                    kernel_size=1,
                    stride=2,
                    output_padding=1,
                    bias=False,
                ),
            )
        elif in_channels != down_channels:
            self.upsample = None
            self.down_scale = nn.Sequential(
                nn.BatchNorm2d(num_features=in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=down_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                ),
            )
        else:
            self.upsample = None
            self.down_scale = None

    def forward(self, x):
        """
        Forward pass of the DecoderBottleneckLayer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        identity = x

        x = self.weight_layer1(x)
        x = self.weight_layer2(x)
        x = self.weight_layer3(x)

        if self.upsample is not None:
            identity = self.upsample(identity)
        elif self.down_scale is not None:
            identity = self.down_scale(identity)

        x = x + identity

        return x
