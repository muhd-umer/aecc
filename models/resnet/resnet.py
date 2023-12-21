"""
MIT License:
Copyright (c) 2023 Muhammad Umer

ResNet AutoEncoder
"""

import torch
import torch.nn as nn

from .decoder import ResNetDecoder
from .encoder import ResNetEncoder


class DAEResNet(nn.Module):
    """
    A ResNet Denoising AutoEncoder model.
    """

    def __init__(self, cfg, bottleneck):
        """
        Initializes the ResNet AutoEncoder model.

        Parameters:
        cfg (list): The configuration of the ResNet model.
        bottleneck (bool): The bottleneck flag of the ResNet model.
        """
        super(DAEResNet, self).__init__()

        self.encoder = ResNetEncoder(cfg=cfg, bottleneck=bottleneck)
        self.decoder = ResNetDecoder(cfg=cfg[::-1], bottleneck=bottleneck)

    def forward(self, x):
        """
        Defines the computation performed at every call.

        Parameters:
        x (Tensor): The input tensor.

        Returns:
        Tensor: The output tensor.
        """
        x = self.encoder(x)
        x = self.decoder(x)

        return x


class ResNet(nn.Module):
    """
    A ResNet model.
    """

    def __init__(self, cfg, bottleneck=False, num_classes=1000):
        """
        Initializes the ResNet model.

        Parameters:
        cfg (list): The configuration of the ResNet model.
        bottleneck (bool): The bottleneck flag of the ResNet model.
        num_classes (int): The number of classes for the model to classify.
        """
        super(ResNet, self).__init__()

        self.encoder = ResNetEncoder(cfg, bottleneck)
        self.avpool = nn.AdaptiveAvgPool2d((1, 1))

        if bottleneck:
            self.fc = nn.Linear(in_features=2048, out_features=num_classes)
        else:
            self.fc = nn.Linear(in_features=512, out_features=num_classes)

        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initializes the weights of the model.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Defines the computation performed at every call.

        Parameters:
        x (Tensor): The input tensor.

        Returns:
        Tensor: The output tensor.
        """
        x = self.encoder(x)
        x = self.avpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def dae_resnet_18(**kwargs):
    """
    Constructs a ResNet-18 model.
    """
    return DAEResNet(cfg=[2, 2, 2, 2], bottleneck=False)


def dae_resnet_34(**kwargs):
    """
    Constructs a ResNet-34 model.
    """
    return DAEResNet(cfg=[3, 4, 6, 3], bottleneck=False)


def dae_resnet_50(**kwargs):
    """
    Constructs a ResNet-50 model.
    """
    return DAEResNet(cfg=[3, 4, 6, 3], bottleneck=True)


def dae_resnet_101(**kwargs):
    """
    Constructs a ResNet-101 model.
    """
    return DAEResNet(cfg=[3, 4, 23, 3], bottleneck=True)


def dae_resnet_152(**kwargs):
    """
    Constructs a ResNet-152 model.
    """
    return DAEResNet(cfg=[3, 8, 36, 3], bottleneck=True)


dae_resnet_models = {
    "dae_resnet_18": dae_resnet_18,
    "dae_resnet_34": dae_resnet_34,
    "dae_resnet_50": dae_resnet_50,
    "dae_resnet_101": dae_resnet_101,
    "dae_resnet_152": dae_resnet_152,
}
