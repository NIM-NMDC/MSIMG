import os
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import sparse
from pathlib import Path
from functools import partial
from multiprocessing import Pool
from pyteomics import mzml, mzxml


def _create_save_path(file_path, prefix):
    """
    Create a save path based on the original file path and a prefix.
    """
    p = Path(file_path)

    base_dir = p.parent.parent
    class_name = p.parent.name
    save_dir = base_dir / prefix / class_name
    save_dir.mkdir(parents=True, exist_ok=True)
    new_file_name = p.with_suffix('.npz').name
    save_path = save_dir / new_file_name
    return save_path


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
    mz_array = np.array(mz_array)
    intensity_array = np.array(intensity_array)
    num_bins = math.ceil((mz_max - mz_min) / bin_size)

    valid_mask = (mz_array >= mz_min) & (mz_array <= mz_max)
    filtered_mz_array = mz_array[valid_mask]
    filtered_intensity_array = intensity_array[valid_mask]

    bin_indices = np.floor((filtered_mz_array - mz_min) / bin_size).astype(int)
    bin_table = pd.DataFrame({'bin_index': bin_indices, 'intensity': filtered_intensity_array.tolist()})
    aggregated_intensities = bin_table.groupby('bin_index')['intensity'].sum()
    full_index = pd.RangeIndex(start=0, stop=num_bins, step=1)
    feature_vector = aggregated_intensities.reindex(full_index, fill_value=0.0)
    return feature_vector


def parse_spec(spec, mz_min, mz_max, bin_size):
    """
    Parse the spectrum data.

    :param spec: The mass spectrum data.
    :param mz_min: The minimum value of m/z bin.
    :param mz_max: The maximum value of m/z bin.
    :param bin_size: The Da of every bin in m/z binning.
    :return: A binned spectrum.
    """
    mz_array = spec['m/z array']
    intensity_array = spec['intensity array']
    bin_spec = binning(mz_array=mz_array, intensity_array=intensity_array, mz_min=mz_min, mz_max=mz_max, bin_size=bin_size)
    return bin_spec


def parse_ms(ms_file_path, prefix, mz_min, mz_max, bin_size):
    try:
        if not (os.path.exists(ms_file_path) and os.path.getsize(ms_file_path) > 0):
            raise FileNotFoundError(f"File not found or empty: {ms_file_path}")

        save_path = _create_save_path(ms_file_path, prefix)
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

        # pseudo_ms_image = pd.DataFrame(pseudo_ms_image).T  # pseudo_ms_image: (scans, mz_bins) -> (mz_bins, scans)
        pseudo_ms_image = pd.DataFrame(pseudo_ms_image)
        pseudo_ms_image = pseudo_ms_image.to_numpy()  # pseudo_ms_image: (scans, mz_bins)
        # print(f"shape: {pseudo_ms_image.shape}")

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
                desc='Binning ms files'
            )
        )

    success_rate = sum(results) / len(results)
    print(f"Binning completed. Success rate: {success_rate:.2%}")


def process_binning(args, ms_file_paths):
    """
    Process the binning of MS files.
    """
    # Binning MS files
    print('Binning MS Files...')

    parallel_parse_ms(
        ms_file_paths=ms_file_paths,
        prefix=args.bin_prefix,
        mz_min=args.mz_min,
        mz_max=args.mz_max,
        bin_size=args.bin_size,
        workers=args.num_workers
    )

    print('Binning Process Completed.')


if __name__ == '__main__':
    import argparse
    from utils.file_utils import get_file_paths
    parser = argparse.ArgumentParser(description='Dataset Process Workflow')
    parser.add_argument('--dataset_dir', type=str, required=True, help='Path to the dataset directory')
    parser.add_argument('--suffix', type=str, default='mzML', help='File suffix to filter (e.g., .mzML, .mzXML)')
    parser.add_argument('--mz_min', type=float, required=True, help='Minimum m/z value for binning')
    parser.add_argument('--mz_max', type=float, required=True, help='Maximum m/z value for binning')
    parser.add_argument('--bin_size', type=float, required=True, help='Bin size for m/z binning')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker processes to use')

    args = parser.parse_args()

    if '.' not in args.suffix:
        args.suffix = '.' + args.suffix

    args.bin_prefix = f'mz_{args.mz_min}-{args.mz_max}_bin_size_{args.bin_size}'
    ms_file_paths = get_file_paths(base_dir=args.dataset_dir, suffix=args.suffix)
    process_binning(args=args, ms_file_paths=ms_file_paths)

#     import re
#     import pandas as pd
#     file_path = r"E:\msdata\ST000923\HMP2_C8-pos\C8p_rawData\Quantitative Table\CD\0024_XAV_iHMP2_LIP_SM-6CAJC_CD_peaks.csv"
#     df = pd.read_csv(file_path)
#     mz_column, intensity_column = None, None
#     for column in df.columns:
#         if re.search(r'\bm\/?z\b', column, re.IGNORECASE):
#             mz_column = column
#         elif re.search(r'peak height', column, re.IGNORECASE):
#             intensity_column = column
#     print(f'mz_column: {mz_column}, intensity_column: {intensity_column}')
#
#     if mz_column is None or intensity_column is None:
#         raise ValueError(f"Could not find m/z or intensity columns in {file_path}")
#     mz_array = df[mz_column].values
#     intensity_array = df[intensity_column].values
#     binned_spectrum = binning(mz_array, intensity_array, 200, 1100, 0.01)
#     print(f"Binned spectrum shape: {binned_spectrum.shape}")
#     print(f"Binned spectrum: {binned_spectrum[:20]}")  # Print first 10 bins for verification