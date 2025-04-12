import torch
from torch.utils.data import Dataset

import os
from PIL import Image


class MS2DIMGDataset(Dataset):

    def __init__(self, file_paths, label_mapping, transform=None):
        self.file_paths = file_paths
        self.label_mapping = label_mapping
        self.transform = transform

        self.samples = []
        for class_name, file_path in file_paths:
            self.samples.append(
                {
                    'file_path': file_path,
                    'label': self.label_mapping[class_name]
                }
            )

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]['file_path'], self.samples[idx]['label']
        img = Image.open(file_path).convert('L')
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.samples)


class MS2DIMGDomainDataset(Dataset):
    """
    MS2D Image Multiple Domain Dataset.

    Dataset Structure:\n
    dataset_dir/
        Domain A/
            - Class A/ [IMGs]
            - Class B/ [IMGs]
        Domain B/
            - Class A/ [IMGs]
            - Class B/ [IMGs]

    :param file_paths: List of file paths to the images (e.g., [(domain_name, class_name, file_path), ...]).
    :param transform: Transform to be applied to the images.
    :param task: Task type. If 'domain_discrepancy_validation', use the domain as the label.
    """
    def __init__(self, file_paths, label_mapping, transform=None, task=None):
        self.file_paths = file_paths
        self.label_mapping = label_mapping
        self.transform = transform
        self.task = task

        self.samples = []
        self._load_samples()

    def _get_label(self, domain_name, class_name):
        if self.task == 'domain_discrepancy_validation':
            return domain_name
        elif self.task == 'classification':
            return class_name
        else:
            raise ValueError(f"Unknown task: {self.task}")

    def _load_samples(self):
        for domain_name, class_name, file_path in self.file_paths:
            self.samples.append(
                {
                    'file_path': file_path,
                    'label': self.label_mapping[self._get_label(domain_name=domain_name, class_name=class_name)]
                }
            )

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]['file_path'], self.samples[idx]['label']
        img = Image.open(file_path).convert('L')
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.samples)


