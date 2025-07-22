import re
import numpy as np
import pandas as pd

from utils.bin_ms import binning


def load_ms_dataset(dataset, label_mapping, mz_min, mz_max, bin_size):
    """
    Prepare the mass spectrometry dataset using the specified m/z range and bin size.

    :param dataset: List of dictionaries containing 'file_path' and 'class_name'.
    :param label_mapping: Mapping from class names to labels.
    :param mz_min: Minimum m/z value for binning.
    :param mz_max: Maximum m/z value for binning.
    :param bin_size: Size of each bin in Da.
    :return: List of binned spectra.
    """
    binned_spectra = []
    labels = []
    for sample in dataset:
        file_path = sample['file_path']
        class_name = sample['class_name']

        df = pd.read_csv(file_path)
        mz_column, intensity_column = None, None
        for column in df.columns:
            if re.search(r'\bm\/?z\b', column, re.IGNORECASE):
                mz_column = column
            elif re.search(r'peak height', column, re.IGNORECASE):
                intensity_column = column

        if mz_column is None or intensity_column is None:
            raise ValueError(f"Could not find m/z or intensity columns in {file_path}")
        mz_array = df[mz_column].values
        intensity_array = df[intensity_column].values
        binned_spectrum = binning(mz_array, intensity_array, mz_min, mz_max, bin_size)
        binned_spectra.append(binned_spectrum)
        labels.append(label_mapping[class_name])
    return np.array(binned_spectra), np.array(labels)


def load_ms_img_dataset(dataset, label_mapping):
    """
    Prepare the mass spectrometry image dataset.

    :param dataset: List of dictionaries containing 'file_path' and 'class_name'.
    :param label_mapping: Mapping from class names to labels.
    """
    patches_list = []
    positions_list = []
    padding_mask_list = []
    labels = []

    for sample in dataset:
        file_path = sample['file_path']
        class_name = sample['class_name']

        patched_data = np.load(file_path)
        patches = patched_data['patches']
        positions = patched_data['positions']
        patches_list.append(patches)
        positions_list.append(positions)
        labels.append(label_mapping[class_name])
    return np.array(patches_list), np.array(positions_list), np.array(labels)





