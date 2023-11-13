import torch
import torch.nn as nn

class VGG19(nn.Module):
    def __init__(self):
        super().__init__()
        self.first_conv_block = ConvReluPoolBlock(in_channels = 3, out_channels = 64, kernel_size = 3, stride = 1, padding = 0)
        self.second_conv_block = ConvReluPoolBlock(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1, padding = 0, pool = True)
        self.third_conv_block = ConvReluPoolBlock(in_channels = 64, out_channels = 128, kernel_size = 3, stride = 1, padding = 0)
        self.fourth_conv_block = ConvReluPoolBlock(in_channels = 128, out_channels = 128, kernel_size = 3, stride = 1, padding = 0)
        self.fifth_conv_block = ConvReluPoolBlock(in_channels = 128, out_channels = 256, kernel_size = 3, stride = 1, padding = 0, pool = True)
        # TODO: Add the rest of the layers

    def forward(self, x):
        pass


class ConvReluPoolBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, pool = False):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU(),
        ]
        if pool:
            layers.append(nn.MaxPool2d(kernel_size = 2, stride = 2))

        self.block = nn.Sequential(*layers)

