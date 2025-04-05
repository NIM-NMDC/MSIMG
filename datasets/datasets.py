import torch
from torch.utils.data import Dataset

import os
from PIL import Image


class MS2DImgDataset(Dataset):

    def __init__(self, file_paths, label_mapping, get_label_function=None, transform=None):
        self.file_paths = file_paths
        self.label_mapping = label_mapping
        self.transform = transform

        self.img_samples = []
        for file_path in file_paths:
            if get_label_function:
                label = get_label_function(file_path)
            else:
                raise ValueError('get_label_function is required.')
            self.img_samples.append(
                {
                    'file_path': file_path,
                    'label': self.label_mapping[label]
                }
            )

    def __len__(self):
        return len(self.img_samples)

    def __getitem__(self, idx):
        img_sample = self.img_samples[idx]
        img = Image.open(img_sample['file_path']).convert('L')

        if self.transform:
            img = self.transform(img)

        return img, img_sample['label']
