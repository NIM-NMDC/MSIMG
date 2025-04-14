import os
import numpy as np
import pandas as pd

from tqdm import tqdm
from scipy import sparse
from functools import partial
from multiprocessing import Pool
from pyteomics import mzml, mzxml


def binning(mz_array, intensity_array, mz_min, mz_max, bin_size=0.01):
    """
    Binning the ms data.

    :param mz_array: (np.array) A list of m/z values.
    :param intensity_array: (np.array) A list of intensity values (same length as mz_list).
    :param mz_min: The minimum value of m/z bin.
    :param mz_max: The maximum value of m/z bin.
    :param bin_size: The Da of every bin in m/z binning.
    :return: A table of binned m/z and intensity.
    """
    mz_bins = (mz_max - mz_min) // bin_size
    bin_index = round((pd.Series(mz_array.tolist()) - mz_min) / bin_size)
    bin_table = pd.DataFrame({'index': bin_index, 'intensity': intensity_array.tolist()})
    bin_table['index'] = bin_table['index'].astype(int)
    bin_table = bin_table.groupby('index').sum()
    full_index = range(int(mz_bins))
    bin_table = bin_table.reindex(full_index)
    bin_table = bin_table.fillna(0)
    return bin_table['intensity']


def parse_spec(spec, mz_min, mz_max, bin_size):
    """
    Parse the spectrum data.

    :param spec: The mass spectrum data.
    :param mz_min: The minimum value of m/z bin.
    :param mz_max: The maximum value of m/z bin.
    :param bin_size: The Da of every bin in m/z binning.
    :return: A binned spectrum.
    """
    scan = spec.get('id', spec.get('num'))  # Support both mzML and mzXML
    mz_array = spec['m/z array']
    intensity_array = spec['intensity array']
    bin_spec = binning(mz_array=mz_array, intensity_array=intensity_array, mz_min=mz_min, mz_max=mz_max, bin_size=bin_size)
    bin_spec.name = scan
    return bin_spec


def parse_ms(ms_file_path, prefix, mz_min, mz_max, bin_size):
    try:
        save_dir = os.path.dirname(ms_file_path)
        file_name = os.path.splitext(os.path.basename(ms_file_path))[0]
        save_path = os.path.join(save_dir, f'{prefix}_{file_name}.npz')

        if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
            return True

        if ms_file_path.endswith('.mzML'):
            reader = mzml.read(ms_file_path)
            ms_level_key = 'ms level'
        elif ms_file_path.endswith('.mzXML'):
            reader = mzxml.read(ms_file_path)
            ms_level_key = 'msLevel'
        else:
            raise ValueError(f"Unsupported file format: {ms_file_path}")

        pseudo_ms_image = []
        for spec in reader:
            if spec.get(ms_level_key, 0) == 1:
                binned_spec = parse_spec(spec, mz_min, mz_max, bin_size)
                pseudo_ms_image.append(binned_spec)

        pseudo_ms_image = pd.DataFrame(pseudo_ms_image).T  # pseudo_ms_image: (scans, mz_bins) -> (mz_bins, scans)
        pseudo_ms_image = pseudo_ms_image.to_numpy()

        sparse_table = sparse.csr_matrix(pseudo_ms_image, dtype=np.float32)
        sparse.save_npz(save_path, sparse_table)
        del pseudo_ms_image
        del reader
        return True
    except Exception as e:
        raise RuntimeError(f"Error processing {ms_file_path}: {e}")


def parallel_parse_ms(ms_file_paths, prefix, mz_min, mz_max, bin_size, workers=4):
    """
    Generate pseudo 2D MS images from raw MS data (.mzML, .mzXML).

    :param ms_file_paths: The list of ms file paths.
    :param prefix: Prefix for the save path pattern.
    :param mz_min: The minimum value of m/z bin.
    :param mz_max: The maximum value of m/z bin.
    :param bin_size: The Da of every bin in m/z binning.
    :param workers: Number of worker processes to use.
    """
    with Pool(processes=workers) as pool:
        worker = partial(
            parse_ms,
            prefix=prefix,
            mz_min=mz_min,
            mz_max=mz_max,
            bin_size=bin_size
        )

        results = list(
            tqdm(
                pool.imap_unordered(worker, ms_file_paths),
                total=len(ms_file_paths),
                desc='Rasterizing ms files'
            )
        )

    success_rate = sum(results) / len(results)
    print(f"Binning completed. Success rate: {success_rate:.2%}")
