from torch.nn import Module
import torch.nn as nn
import torch


class BlockConv(Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, stride=2, padding=1, batch_norm=False):
        super(BlockConv, self).__init__()

        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              bias=False)
        self.bn2d = nn.BatchNorm2d(output_channels)
        self.activation = nn.LeakyReLU(0.2)

        self.batch_norm = batch_norm

    def forward(self, x: torch.Tensor):
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
