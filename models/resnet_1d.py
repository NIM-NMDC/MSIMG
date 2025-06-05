import torch
import torch.nn as nn

import re
import argparse


class Bottleneck(nn.Module):

    def __init__(self, in_channels, intermediate_channels, expansion, is_Bottleneck, stride):
        """
        Creates a Bottleneck with conv 1x1->3x3->1x1 layers.

        Note:
            1. Addition of feature maps occur at just before the final ReLU with the input feature maps
            2. if input size is different from output, select projected mapping or else identity
            3. if is_Bottleneck=False (3x3->3x3) are used else (1x1->3x3->1x1). Bottleneck is required for resnet-50/101/152
        Args:
            in_channels (int): input channels to the Bottleneck
            intermediate_channels (int): number of channels to 3x3 conv
            expansion (int): factor by which the input #channels are increased
            stride (int): stride applied in the 3x3 conv. 2 for first Bottleneck of the block and 1 for remaining

        Attributes:
            Layer consisting of conv->batchnorm->relu
        """

        super(Bottleneck, self).__init__()

        self.expansion = expansion
        self.in_channels = in_channels
        self.intermediate_channels = intermediate_channels
        self.is_Bottleneck = is_Bottleneck

        # i.e. if dim(x) == dim(F) => Identity function
        if self.in_channels == self.intermediate_channels * self.expansion:
            self.identity = True
        else:
            self.identity = False
            projection_layer = []
            projection_layer.append(
                nn.Conv1d(
                    in_channels=self.in_channels,
                    out_channels=self.intermediate_channels * self.expansion,
                    kernel_size=1,
                    stride=stride,
                    padding=0,
                    bias=False
                )
            )
            projection_layer.append(
                nn.BatchNorm1d(
                    self.intermediate_channels * self.expansion
                )
            )
            # Only conv->BN and no ReLU
            # projection_layer.append(nn.ReLU())
            self.projection = nn.Sequential(*projection_layer)

        # commonly used relu
        self.relu = nn.ReLU()

        # is_Bottleneck = True for all ResNet 50+
        if self.is_Bottleneck:
            # bottleneck
            # 1x1
            self.conv1_1x1 = nn.Conv1d(
                in_channels=self.in_channels,
                out_channels=self.intermediate_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            )
            self.batchnorm1 = nn.BatchNorm1d(self.intermediate_channels)

            # 3x3
            self.conv2_3x3 = nn.Conv1d(
                in_channels=self.intermediate_channels,
                out_channels=self.intermediate_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False
            )
            self.batchnorm2 = nn.BatchNorm1d(self.intermediate_channels)

            # 1x1
            self.conv3_1x1 = nn.Conv1d(
                in_channels=self.intermediate_channels,
                out_channels=self.intermediate_channels * self.expansion,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            )
            self.batchnorm3 = nn.BatchNorm1d(self.intermediate_channels * self.expansion)

        else:
            # basicblock
            # 3x3
            self.conv1_3x3 = nn.Conv1d(
                in_channels=self.in_channels,
                out_channels=self.intermediate_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False
            )
            self.batchnorm1 = nn.BatchNorm1d(self.intermediate_channels)

            # 3x3
            self.conv2_3x3 = nn.Conv1d(
                in_channels=self.intermediate_channels,
                out_channels=self.intermediate_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            )
            self.batchnorm2 = nn.BatchNorm1d(self.intermediate_channels)

    def forward(self, x):
        # input stored to be added before the final relu
        in_x = x

        if self.is_Bottleneck:
            # conv1x1->BN->relu
            x = self.relu(self.batchnorm1(self.conv1_1x1(x)))

            # conv3x3->BN->relu
            x = self.relu(self.batchnorm2(self.conv2_3x3(x)))

            # conv1x1->BN
            x = self.batchnorm3(self.conv3_1x1(x))

        else:
            # conv3x3->BN->relu
            x = self.relu(self.batchnorm1(self.conv1_3x3(x)))

            # conv3x3->BN
            x = self.batchnorm2(self.conv2_3x3(x))

        # identity or projected mapping
        if self.identity:
            x += in_x
        else:
            x += self.projection(in_x)

        # final relu
        x = self.relu(x)

        return x


class ResNet1D(nn.Module):

    def __init__(self, resnet_variant, in_channels, num_classes):
        """
        Creates the ResNet architecture based on the provided variant. 18/34/50/101 etc.
        Based on the input parameters, define the channels list, repetition list along with expansion factor(4) and stride(3/1)
        using _make_blocks method, create a sequence of multiple Bottlenecks
        Average Pool at the end before the FC layer

        Args:
            resnet_variant (list): eg. [[64,128,256,512],[3,4,6,3],4,True]
            in_channels (int): input data channels
            num_classes (int): output classes

        Attributes:
            Layer consisting of conv->batchnorm->relu
        """
        super().__init__()

        self.channels_list = resnet_variant[0]
        self.repetition_list = resnet_variant[1]
        self.expansion = resnet_variant[2]
        self.is_Bottleneck = resnet_variant[3]

        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()

        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.block1 = self._make_blocks(64, self.channels_list[0], self.repetition_list[0], self.expansion,
                                        self.is_Bottleneck, stride=1)
        self.block2 = self._make_blocks(self.channels_list[0] * self.expansion, self.channels_list[1],
                                        self.repetition_list[1], self.expansion, self.is_Bottleneck, stride=2)
        self.block3 = self._make_blocks(self.channels_list[1] * self.expansion, self.channels_list[2],
                                        self.repetition_list[2], self.expansion, self.is_Bottleneck, stride=2)
        self.block4 = self._make_blocks(self.channels_list[2] * self.expansion, self.channels_list[3],
                                        self.repetition_list[3], self.expansion, self.is_Bottleneck, stride=2)

        self.average_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(self.channels_list[3] * self.expansion, num_classes)

    def _make_blocks(self, in_channels, intermediate_channels, num_repeats, expansion, is_Bottleneck, stride):
        """
        Args:
            in_channels: channels of the Bottleneck input
            intermediate_channels: channels of the 3x3 in the Bottleneck
            num_repeats: Bottlenecks in the block
            expansion: factor by which intermediate_channels are multiplied to create the output channels
            is_Bottleneck: status if Bottleneck in required
            stride: stride to be used in the first Bottleneck conv 3x3

            Attributes:
                Sequence of Bottleneck layers
        """
        layers = []

        layers.append(
            Bottleneck(
                in_channels=in_channels,
                intermediate_channels=intermediate_channels,
                expansion=expansion,
                is_Bottleneck=is_Bottleneck,
                stride=stride
            )
        )
        for num in range(1, num_repeats):
            layers.append(
                Bottleneck(
                    in_channels=intermediate_channels * expansion,
                    intermediate_channels=intermediate_channels,
                    expansion=expansion,
                    is_Bottleneck=is_Bottleneck,
                    stride=1
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        if x.ndim == 2 and self.conv1.in_channels == 1:
            x = x.unsqueeze(1)
        x = self.relu(self.batchnorm1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.average_pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        return x


def build_resnet_1d(args):

    # ResNetX = (Num of channels, repetition, Bottleneck_expansion, Bottleneck_layer)
    resnet_parameters = {
        'resnet18': ([64, 128, 256, 512], [2, 2, 2, 2], 1, False),
        'resnet34': ([64, 128, 256, 512], [3, 4, 6, 3], 1, False),
        'resnet50': ([64, 128, 256, 512], [3, 4, 6, 3], 4, True),
        'resnet101': ([64, 128, 256, 512], [3, 4, 23, 3], 4, True),
        'resnet152': ([64, 128, 256, 512], [3, 8, 36, 3], 4, True)
    }


    # ResNet50 -> resnet50
    match = re.match(r'([A-Za-z_]*)ResNet(\d+)', args.model_name, re.IGNORECASE)
    if match:
        base_model_key = args.model_name.lower()
    else:
        raise ValueError(f"Invalid model name: {args.model_name}")

    if base_model_key in resnet_parameters:
        # Model instantiation
        resnet_variant = resnet_parameters[base_model_key]

        return ResNet1D(
            resnet_variant=resnet_variant,
            in_channels=args.in_channels,
            num_classes=args.num_classes
        )
    else:
        raise ValueError(f"Invalid model name: {args.model_name}")


if __name__ == '__main__':
    args = {
        'model_name': 'ResNet50',
        'in_channels': 1,
        'spectrum_dim': 15000,
        'embedding_channels': 256,
        'embedding_dim': 1024,
        'num_classes': 3
    }
    args = argparse.Namespace(**args)
    model = build_resnet_1d(args)

    # print models structure
    print(model)

    x = torch.randn(1, args.spectrum_dim)

    output = model(x)
    print(output)