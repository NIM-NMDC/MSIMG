import torch
import torch.nn as nn


class SEBlock(nn.Module):
    """
    squeeze-and-excitation block
    """
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.in_channels = in_channels
        self.reduction = reduction

        self.squeeze = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        :param x: input tensor of shape (B, C, H, W)
        :return: output tensor of shape (B, C, H, W)
        """
        B, C, H, W = x.shape
        # Squeeze
        y = self.squeeze(x).view(B, C)
        y = self.excitation(y).view(B, C, 1, 1)  # (B, C, 1, 1)
        return x * y.expand_as(x)  # Scale the input tensor
