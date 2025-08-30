import torch
import torch.nn as nn

import re
import argparse


class DenseLayer(nn.Module):

    def __init__(self, in_channels, growth_rate):
        """
        First 1x1 convolution generating 4 * growth_rate number of channels irrespective of the total number of input channels.
        First 3x3 convolution generating growth_rate number of channels from the 4 * growth_rate number of input channels.

        Args:
        in_channels (int) : # input channels to the Dense Layer
        """
        super().__init__()

        self.bn1 = nn.BatchNorm1d(in_channels)
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=4 * growth_rate, kernel_size=1, stride=1, padding=0, bias=False)

        self.bn2 = nn.BatchNorm1d(4 * growth_rate)
        self.conv2 = nn.Conv1d(in_channels=4 * growth_rate, out_channels=growth_rate, kernel_size=3, stride=1, padding=1, bias=False)

        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Bottleneck DenseLayer with following operations
        (i) batch_norm -> relu -> 1x1 conv
        (ii) batch_norm -> relu -> 3x3 conv

        Concatenation of input and output tensor which is the main idea of DenseNet.

        Args:
            x (tensor) : input tensor to be passed through the dense layer

        Attributes:
            x (tensor) : output tensor
        """
        xin = x

        x = self.conv1(self.relu(self.bn1(x)))
        x = self.conv2(self.relu(self.bn2(x)))

        out = torch.cat([xin, x], 1)

        return out


class DenseBlock(nn.Module):

    def __init__(self, num_layers, in_channels, growth_rate):
        """
        Looping through total number of layers in the denseblock.
        Adding growth_rate number of channels in each loop as each layer generates tensor with growth_rate channels.

        Args:
            num_layers (int) : total number of dense layers in the dense block
            in_channels (int) : input number of channels
            growth_rate (int) : growth rate of the dense block
        """
        super().__init__()

        self.num_layers = num_layers
        self.deep_nn = nn.ModuleList()

        for num in range(num_layers):
            self.deep_nn.add_module(f"DenseLayer_{num + 1}", DenseLayer(in_channels + growth_rate * num, growth_rate))

    def forward(self, x):
        # xin = x

        for layer in self.deep_nn:
            x = layer(x)

        return x


class TransitionLayer(nn.Module):

    def __init__(self, in_channels, compression_factor):
        """
        1x1 conv used to change output channels using the compression_factor (default = 0.5).
        avgpool used to downsample the feature map resolution

        Args:
            compression_factor (float) : output_channels/input_channels
            in_channels (int) : input number of channels
        """
        super().__init__()

        self.bn = nn.BatchNorm1d(in_channels)
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=int(in_channels * compression_factor), kernel_size=1, stride=1, padding=0, bias=False)
        self.avgpool = nn.AvgPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.bn(x)
        x = self.conv1(x)
        x = self.avgpool(x)

        return x


class DenseNet1D(nn.Module):

    def __init__(self, densenet_variant, growth_rate, compression_factor, in_channels, num_classes):
        """
        Creating an initial 7x7 convolution followed by 3 DenseBlock and 3 Transition layers. Concluding this with 4th DenseBlock, 7x7 global average pool and FC layer
        for classification

        Args:
            densenet_variant (list) : list containing the total number of layers in a dense block
            in_channels (int) : input number of channels
            num_classes (int) : Total number of output classes
        """
        super().__init__()

        # 7x7 Convolution with maxpool
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)

        # Add 3 DenseBlocks and 3 Transition Layers
        self.deep_nn = nn.ModuleList()
        dense_block_in_channels = 64

        for num in range(len(densenet_variant))[:-1]:
            self.deep_nn.add_module(f"DenseBlock_{num + 1}", DenseBlock(densenet_variant[num], dense_block_in_channels, growth_rate))
            dense_block_in_channels = int(dense_block_in_channels + growth_rate * densenet_variant[num])

            self.deep_nn.add_module(f'TransitionLayer_{num + 1}', TransitionLayer(dense_block_in_channels, compression_factor))
            dense_block_in_channels = int(dense_block_in_channels * compression_factor)

        # Add the 4th final DenseBlock
        self.deep_nn.add_module(f"DenseBlock_{len(densenet_variant)}", DenseBlock(densenet_variant[-1], dense_block_in_channels, growth_rate))
        dense_block_in_channels = int(dense_block_in_channels + growth_rate * densenet_variant[-1])

        self.bn2 = nn.BatchNorm1d(dense_block_in_channels)
        self.average_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(dense_block_in_channels, num_classes)

    def forward(self, x):
        if x.ndim == 2 and self.conv1.in_channels == 1:
            x = x.unsqueeze(1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        for layer in self.deep_nn:
            x = layer(x)
        x = self.relu(self.bn2(x))
        x = self.average_pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x


def build_densenet_1d(args):

    # DenseNetX = (growth_rate, compression_factor, Bottleneck_layer)
    densenet_parameters = {
        'densenet121': (32, 0.5, [6, 12, 24, 16]),
        'densenet169': (32, 0.5, [6, 12, 32, 32]),
        'densenet201': (32, 0.5, [6, 12, 48, 32]),
        'densenet264': (32, 0.5, [6, 12, 64, 48])
    }

    # DenseNet121 -> densenet121
    match = re.match(r'([A-Za-z_-]*)DenseNet(\d+)', args.model_name, re.IGNORECASE)
    if match:
        base_model_key = f"densenet{match.group(2)}"
    else:
        raise ValueError(f"Invalid model name: {args.model_name}")

    if base_model_key in densenet_parameters:
        # Model instantiation
        growth_rate, compression_factor, densenet_variant = densenet_parameters[base_model_key]

        return DenseNet1D(
            densenet_variant=densenet_variant,
            growth_rate=growth_rate,
            compression_factor=compression_factor,
            in_channels=args.in_channels,
            num_classes=args.num_classes
        )
    else:
        raise ValueError(f"Invalid model name: {args.model_name}")


if __name__ == '__main__':
    args = {
        'model_name': 'DenseNet121',
        'in_channels': 1,
        'num_bins': 15000,
        'num_classes': 3
    }
    args = argparse.Namespace(**args)

    model = build_densenet_1d(args)

    # print models structure
    print(model)

    x = torch.randn(1, args.num_bins)
    output = model(x)
    print(output)


