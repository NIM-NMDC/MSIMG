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
    :param preload: If True, load all data into memory.
    """
    def __init__(self, dataset, label_mapping, transform=None, preload=False):
        self.dataset = dataset
        self.label_mapping = label_mapping
        self.transform = transform

        if preload:
            self.samples = []
            for sample_info in dataset:
                sample = np.load(sample_info['file_path'])
                patches = torch.tensor(sample['patches'], dtype=torch.float32)
                positions = torch.tensor(sample['positions'], dtype=torch.float32)
                padding_mask = torch.tensor(sample['padding_mask'], dtype=torch.bool)
                label = torch.tensor(self.label_mapping[sample_info['class_name']], dtype=torch.long)

                if self.transform:
                    patches = self.transform(patches)

                self.samples.append((patches, positions, padding_mask, label))

    def __getitem__(self, idx):
        if hasattr(self, 'samples'):
            return self.samples[idx]
        else:
            sample = np.load(self.dataset[idx]['file_path'])
            patches = torch.tensor(sample['patches'], dtype=torch.float32)
            positions = torch.tensor(sample['positions'], dtype=torch.float32)
            padding_mask = torch.tensor(sample['padding_mask'], dtype=torch.bool)
            label = torch.tensor(self.label_mapping[self.dataset[idx]['class_name']], dtype=torch.long)

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
    :param preload: If True, load all data into memory.
    """
    def __init__(self, dataset, label_mapping, transform=None, task=None, preload=False):
        self.dataset = dataset
        self.label_mapping = label_mapping
        self.transform = transform
        self.task = task

        if preload:
            self.samples = []
            for sample_info in dataset:
                sample = np.load(sample_info['file_path'])
                patches = torch.tensor(sample['patches'], dtype=torch.float32)
                positions = torch.tensor(sample['positions'], dtype=torch.float32)
                padding_mask = torch.tensor(sample['padding_mask'], dtype=torch.bool)
                label = self._get_label(
                    domain_name=sample_info['domain_name'],
                    class_name=sample_info['class_name'],
                    label_mapping=label_mapping
                )
                label = torch.tensor(label, dtype=torch.long)

                if self.transform:
                    patches = self.transform(patches)

                self.samples.append((patches, positions, padding_mask, label))

    def _get_label(self, domain_name, class_name, label_mapping):
        if self.task == 'domain_discrepancy_validation':
            return label_mapping[domain_name]
        elif self.task == 'classification':
            return label_mapping[class_name]
        else:
            raise ValueError(f"Unknown task: {self.task}")

    def __getitem__(self, idx):
        if hasattr(self, 'samples'):
            return self.samples[idx]
        else:
            sample = np.load(self.dataset[idx]['file_path'])
            patches = torch.tensor(sample['patches'], dtype=torch.float32)
            positions = torch.tensor(sample['positions'], dtype=torch.float32)
            padding_mask = torch.tensor(sample['padding_mask'], dtype=torch.long)
            label = self._get_label(
                domain_name=self.dataset[idx]['domain_name'],
                class_name=self.dataset[idx]['class_name'],
                label_mapping=self.label_mapping
            )
            label = torch.tensor(label, dtype=torch.long)

            if self.transform:
                patches = self.transform(patches)

            return patches, positions, padding_mask, label

    def __len__(self):
        return len(self.dataset)


