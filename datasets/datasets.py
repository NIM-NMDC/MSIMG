import torch
from torch.utils.data import Dataset

import numpy as np


class MSDataset(Dataset):
    """
    Mass Spectrometry Quantitative peaks table Dataset.
    """
    def __init__(self, X, y, transform=False):
        """
        :param X: numpy array of shape (n_samples, n_features)
        :param y: numpy array of shape (n_samples,)
        :param transform: If True, apply standard scaling to the features.
        """
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        _X = self.X[idx]
        _y = self.y[idx]
        if self.transform:
            min_val = _X.min()
            max_val = _X.max()
            if max_val > min_val:
                _X = (_X - min_val) / (max_val - min_val)
        return torch.tensor(_X, dtype=torch.float32), torch.tensor(_y, dtype=torch.long)


class MSIMGDataset(Dataset):
    """
    Mass Spectrometry 2D Image Dataset.
    """
    def __init__(self, patches_list, positions_list, padding_mask_list, labels, return_positions=False, transform=None):
        """
        :param patches_list: List of patches (numpy arrays).
        :param positions_list: List of positions (numpy arrays).
        :param padding_mask_list: List of padding masks (numpy arrays).
        :param labels: List of labels (numpy arrays).
        :param return_positions: If True, return positions along with patches and labels.
        :param transform: Optional transform to be applied on the patches.
        """
        self.patches_list = patches_list
        self.positions_list = positions_list
        self.padding_mask_list = padding_mask_list
        self.labels = labels
        self.return_positions = return_positions
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        patches = self.patches_list[idx]
        label = self.labels[idx]
        patches = torch.tensor(patches, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)

        if self.transform:
            patches = self.transform(patches)

        if self.return_positions:
            positions = self.positions_list[idx]
            padding_mask = self.padding_mask_list[idx]
            return patches, \
                   torch.tensor(positions, dtype=torch.float32), \
                   torch.tensor(padding_mask, dtype=torch.bool), \
                   label
        else:
            return patches, \
                   label



