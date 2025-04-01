import torch
import torch.nn as nn
from torchvision import transforms as T
from torchvision.transforms.functional import erase as f_erase

import random


class MinMaxNormalize(nn.Module):
    """
    Min-Max normalization for the input tensor.
    """
    def forward(self, img):
        min_val = torch.min(img)
        max_val = torch.max(img)
        if max_val > min_val:
            img = (img - min_val) / (max_val - min_val)
        else:
            img = img - min_val
        return img


class ZeroMeanNormalize(nn.Module):
    """
    Zero mean normalization for the input tensor. [0, 1] -> [-1, 1]
    """
    def forward(self, img):
        return img * 2 - 1


class NoiseInjection(nn.Module):
    """
    Mass spectrometry specific noise injection.
    """
    def __init__(self, noise_level=0.05, spike_prob=0.02):
        super().__init__()
        self.noise_level = noise_level
        self.spike_prob = spike_prob

    def forward(self, img):
        noise = torch.randn_like(img) * self.noise_level

        # Spike noise (simulate mass spectrometry artifact)
        if random.random() < self.spike_prob:
            h, w = img.shape[-2:]
            x = random.randint(0, w - 1)
            y = random.randint(0, h - 1)
            noise[y:y + 3, x:x + 3] += 0.5

        return torch.clamp(img + noise, 0., 1.)


class RandomErase(torch.nn.Module):
    """
    Randomly erase a rectangular area in the MS image.
    """
    def __init__(self, p=0.5, scale=(0.02, 0.2), value=0.0):
        """
        :param p: The probability that the random erasing operation will be performed.
        :param scale: (min_area, max_area) of the erased area.
        :param value: The value to fill the erased area.
        """
        super().__init__()
        self.p = p
        self.scale = scale
        self.value = value

    def forward(self, img):
        # img: Tensor[C, H, W]
        if random.random() < self.p:
            C, H, W = img.size()
            area = H * W
            target_area = random.uniform(self.scale[0], self.scale[1]) * area
            erase_side = int(round(target_area ** 0.5))

            y = random.randint(0, H - erase_side)
            x = random.randint(0, W - erase_side)

            v_tensor = torch.tensor(self.value, dtype=img.dtype, device=img.device)
            img = f_erase(
                img, y, x, erase_side, erase_side, v=v_tensor, inplace=True
            )
        return img


def build_base_transform(config):
    """
    Build the base transform for the dataset.
    :param config: Configuration dictionary.
        {
            "resize": tuple,
            "min_max_norm": bool,
            "zero_mean_norm": bool
        }
    :return: A torchvision.transforms.Compose object.
    """
    transforms = []

    if config.get('resize'):
        transforms.append(T.Resize(config['resize']))

    transforms.append(T.ToTensor())

    if config.get('min_max_norm', True):
        transforms.append(MinMaxNormalize())

    if config.get('zero_mean_norm', False):
        transforms.append(ZeroMeanNormalize())

    return T.Compose(transforms)


def build_aug_transform(config):
    """
    Build the augmentation transform for the dataset.
    :param config: Configuration dictionary.
        {
            "noise_level": float,
            "spike_prob": float,
            "random_erase": float
        }
    :return: A torchvision.transforms.Compose object.
    """
    transforms = []

    if config.get('noise_level', 0) > 0:
        transforms.append(NoiseInjection(
            noise_level=config['noise_level'],
            spike_prob=config.get('spike_prob', 0.02)
        ))

    if config.get('random_erase', 0) > 0:
        transforms.append(RandomErase(
            p=config['random_erase_prob'],
            scale=config.get('random_erase_scale', (0.02, 0.2)),
            value=config.get('random_erase_value', 0.0)
        ))

    return T.Compose(transforms)


def build_dynamic_transform(config, aug_prob):
    """
    Build the dynamic transform for the dataset.

    :param config: Configuration dictionary.
    :param aug_prob: float, The probability of applying the augmentation transform.
    :return: A torchvision.transforms.Compose object.
    """
    base = build_base_transform(config)
    aug = build_aug_transform(config)

    return T.Compose([
        base,
        T.RandomApply([aug], p=aug_prob)
    ])
