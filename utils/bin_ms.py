import os
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import sparse
from pathlib import Path
from pyteomics import mzml
from functools import partial
from multiprocessing import Pool


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
        else:
            raise ValueError(f"Unsupported file format: {ms_file_path}")

        ms_matrix = []
        rt_list = []
        for spec in reader:
            if spec.get(ms_level_key, 0) == 1:
                rt_value = None
                scan_list = spec.get('scanList')
                if scan_list:
                    scan = scan_list.get('scan')
                    if scan:
                        rt_value = scan[0].get('scan start time')

                if rt_value is None:
                    continue

                binned_spec = parse_spec(spec, mz_min, mz_max, bin_size)
                ms_matrix.append(binned_spec)
                rt_list.append(rt_value)

        ms_matrix = pd.DataFrame(ms_matrix)
        ms_matrix = ms_matrix.to_numpy()  # (rt_scans, mz_bins)
        # print(f"ms_matrix.shape: {ms_matrix.shape}")
        # print(f"rt_list.shape: {len(rt_list)}")
        # print(f"rt_list: {rt_list}")
        sparse_ms_matrix = sparse.csr_matrix(ms_matrix, dtype=np.float32)
        np.savez_compressed(save_path, sparse_ms_matrix=sparse_ms_matrix, rt_list=rt_list)
        del ms_matrix, reader
        return True
    except Exception as e:
        raise RuntimeError(f"Error processing {ms_file_path}: {e}")


def parallel_parse_ms(ms_file_paths, prefix, mz_min, mz_max, bin_size, workers=4):
    """
    Generate pseudo 2D MS images from raw MS data (.mzML).

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
    # ms_file_path = r"S:\msdata\ST001937\mzML\Benign SPNS\2JY8.mzML"
    # parse_ms(ms_file_path=ms_file_path, mz_min=50.0, mz_max=500.0, bin_size=0.01, prefix='test_prefix')
    import argparse
    from utils.file_utils import get_file_paths
    parser = argparse.ArgumentParser(description='Dataset Process Workflow')
    parser.add_argument('--dataset_dir', type=str, required=True, help='Path to the dataset directory')
    parser.add_argument('--suffix', type=str, default='.mzML', help='File suffix to filter (e.g., .mzML, .mzXML)')
    parser.add_argument('--mz_min', type=float, required=True, help='Minimum m/z value for binning')
    parser.add_argument('--mz_max', type=float, required=True, help='Maximum m/z value for binning')
    parser.add_argument('--bin_size', type=float, required=True, help='Bin size for m/z binning')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker processes to use')

    args = parser.parse_args()

    if '.' not in args.suffix:
        args.suffix = '.' + args.suffix

    args.bin_prefix = f'MZ_{args.mz_min}-{args.mz_max}_BIN_SIZE_{args.bin_size}'
    ms_file_paths = get_file_paths(base_dir=args.dataset_dir, suffix=args.suffix)
    process_binning(args=args, ms_file_paths=ms_file_paths)