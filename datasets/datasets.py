import torch
from torch.utils.data import Dataset
import numpy as np


class MS2DIMGDataset(Dataset):
    """
    MS 2D Image Dataset.

    Dataset Structure:\n
    dataset_dir/
        - Class A/ [IMGs]
        - Class B/ [IMGs]

    :param dataset: List of tuples containing (class_name, file_path).
    :param label_mapping: Mapping from class names to labels.
    :param transform: Transform to be applied to the images.
    """
    def __init__(self, dataset, label_mapping, transform=None):
        self.dataset = dataset
        self.label_mapping = label_mapping
        self.transform = transform

    def __getitem__(self, idx):
        sample = np.load(self.dataset[idx]['file_path'])
        patches = sample['patches']
        positions = sample['positions']
        padding_mask = sample['padding_mask']
        label = self.label_mapping[self.dataset[idx]['class_name']]
        if self.transform:
            patches = self.transform(patches)
        return patches, positions, padding_mask, label

    def __len__(self):
        return len(self.dataset)


class MS2DIMGDomainDataset(Dataset):
    """
    MS 2D Image Multiple Domain Dataset.

    Dataset Structure:\n
    dataset_dir/
        Domain A/
            - Class A/ [IMGs]
            - Class B/ [IMGs]
        Domain B/
            - Class A/ [IMGs]
            - Class B/ [IMGs]

    :param dataset: List of tuples containing (domain_name, class_name, file_path).
    :param label_mapping: Mapping from class names to labels.
    :param transform: Transform to be applied to the images.
    :param task: Task type. If 'domain_discrepancy_validation', use the domain as the label.
    """
    def __init__(self, dataset, label_mapping, transform=None, task=None):
        self.dataset = dataset
        self.label_mapping = label_mapping
        self.transform = transform
        self.task = task

    def _get_label(self, domain_name, class_name, label_mapping):
        if self.task == 'domain_discrepancy_validation':
            return label_mapping[domain_name]
        elif self.task == 'classification':
            return label_mapping[class_name]
        else:
            raise ValueError(f"Unknown task: {self.task}")

    def __getitem__(self, idx):
        sample = np.load(self.dataset[idx]['file_path'])
        patches = sample['patches']
        positions = sample['positions']
        padding_mask = sample['padding_mask']
        label = self._get_label(
            domain_name=self.dataset[idx]['domain_name'],
            class_name=self.dataset[idx]['class_name'],
            label_mapping=self.label_mapping
        )
        if self.transform:
            patches = self.transform(patches)
        return patches, positions, padding_mask, label

    def __len__(self):
        return len(self.dataset)


