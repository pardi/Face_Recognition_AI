from torch.nn import Module
import torch.nn as nn
import torch


class BlockConv(Module):
    def __init__(self, input_channels: int, output_channels: int, kernel_size: int = 3, stride: int = 2, padding: int = 1, batch_norm: bool = False):
        super(BlockConv, self).__init__()

        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              bias=False)
        self.bn2d = nn.BatchNorm2d(output_channels)
        self.activation = nn.LeakyReLU(0.2)

        self.batch_norm = batch_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.batch_norm:
            x = self.bn2d(x)

        return self.activation(x)


class Discriminator(Module):
    def __init__(self, input_channels: int = 3, hidden_channels: int = 32, output_channels: int = 1):
        super(Discriminator, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels

        self.d_block = nn.Sequential(
            BlockConv(self.input_channels, self.hidden_channels, kernel_size=4, padding=1, stride=2, batch_norm=False),
            BlockConv(self.hidden_channels, self.hidden_channels * 2, kernel_size=4, padding=1, stride=2,
                      batch_norm=True),
            BlockConv(self.hidden_channels * 2, self.hidden_channels * 4, kernel_size=4, padding=1, stride=2,
                      batch_norm=True),
            BlockConv(self.hidden_channels * 4, self.hidden_channels * 8, kernel_size=4, padding=1, stride=2,
                      batch_norm=True),
            BlockConv(self.hidden_channels * 8, 1, kernel_size=4, padding=1, stride=4, batch_norm=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.d_block(x)
        return x


class BlockDeconv(Module):
    def __init__(self, input_channels: int, output_channels: int, kernel_size: int = 3, stride: int = 2,
                 padding: int = 0, batch_norm: bool = False, activation=True):
        super(BlockDeconv, self).__init__()

        self.deconv = nn.ConvTranspose2d(input_channels, output_channels, kernel_size=kernel_size, padding=padding,
                                         stride=stride, bias=False)
        self.bn2d = nn.BatchNorm2d(output_channels)

        self.activation = nn.ReLU()
        self.batch_norm = batch_norm
        self.activation_flag = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.deconv(x)

        if self.batch_norm:
            x = self.bn2d(x)

        if self.activation_flag:
            x = self.activation(x)

        return x


class Generator(Module):
    def __init__(self, latent_dim: int, conv_features: int = 64):
        super(Generator, self).__init__()

        self.conv_features = conv_features
        self.latent_dim = latent_dim

        self.g_block = nn.Sequential(
            BlockDeconv(self.latent_dim, self.conv_features * 8, kernel_size=4, padding=0, batch_norm=False),
            BlockDeconv(self.conv_features * 8, self.conv_features * 4, kernel_size=4, padding=1, batch_norm=True),
            BlockDeconv(self.conv_features * 4, self.conv_features * 2, kernel_size=4, padding=1, batch_norm=True),
            BlockDeconv(self.conv_features * 2, self.conv_features, kernel_size=4, padding=1, batch_norm=True),
            BlockDeconv(self.conv_features, 3, kernel_size=4, padding=1, batch_norm=False, activation=False),
            nn.Tanh())

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.g_block(x)

        return x
