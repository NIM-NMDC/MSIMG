import torch
from torchvision import transforms as T
from torchvision.transforms.functional import erase as f_erase

import random


class MinMaxNormalize(torch.nn.Module):

    def forward(self, img):
        min_val = torch.min(img)
        max_val = torch.max(img)
        if max_val > min_val:
            img = (img - min_val) / (max_val - min_val)
        else:
            img = img - min_val
        return img


class RandomNoise(torch.nn.Module):
    """
    Add random noise to the input tensor.
    """
    def __init__(self, noise_std=0.05):
        super().__init__()
        self.noise_std = noise_std

    def forward(self, img):
        noise = torch.randn_like(img) * self.noise_std
        return img + noise


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


def ms_img_transform_pipeline(min_max_norm=True, noise_std=0.0, erase_p=0.0, resize=None):
    """
    Create a transformation pipeline for MS 2D images.

    :param min_max_norm: bool, Whether to perform min-max normalization.
    :param noise_std: float, Standard deviation of the random noise.
    :param erase_p: float, Probability of random erasing.
    :param resize: Optional tuple=(H, W), Whether to resize the image to the specified size.
    :return: A torchvision.transforms.Compose object.
    """
    transform_pipeline = []

    if min_max_norm:
        transform_pipeline.append(MinMaxNormalize())

    if noise_std > 0:
        transform_pipeline.append(RandomNoise(noise_std))

    if erase_p > 0:
        transform_pipeline.append(RandomErase(p=erase_p, scale=(0.02, 0.2), value=0.0))

    if resize:
        transform_pipeline.append(T.Resize(size=resize))

    return T.Compose(transform_pipeline)

