import torch.nn as nn
from torchvision import models


def build_resnet(model_name, num_classes, in_channels=1, pretrained=False):
    """
    Build the ResNet model.

    :param model_name: The model name.
    :param num_classes: The number of classes.
    :param in_channels: The number of input channels.
    :param pretrained: Whether to load the pretrained model.
    :return: The ResNet model.
    """
    model = getattr(models, model_name)(pretrained=pretrained)
    if in_channels != 3:
        model.conv1 = nn.Conv2d(in_channels, model.conv1.out_channels, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
