import os
import pickle
import shutil
import argparse
import numpy as np

from collections import defaultdict

from utils.rasterize_ms import parallel_parse_ms
from utils.patch_ms import parallel_generate_patches
from utils.score_patches import parallel_calculate_patches_scores, calculate_average_scores_and_indices
from utils.select_patches import parallel_select_top_k_patches_per_file, parallel_select_top_k_patches_per_class


def find_files(dataset_dir, suffix):
    """
    Find all files in the dataset directory with the specified suffix.

    :param dataset_dir: Directory to search for files.
    :param suffix: suffix to filter files (e.g., .mzML, .mzXML).
    :return: A dictionary with class names as keys and lists of file paths as values.
    """
    dataset_files = defaultdict(list)
    for class_name in os.listdir(dataset_dir):
        class_dir = os.path.join(dataset_dir, class_name)
        if os.path.isdir(class_dir):
            for file_name in os.listdir(class_dir):
                if file_name.endswith(suffix):
                    dataset_files[class_name].append(os.path.join(class_dir, file_name))

    return dataset_files


def process_binning(args):
    """
    Process the binning of MS files.

    :param args: Arguments containing parameters for binning.
    """
    # Binning MS files
    print('Binning MS Files...')

    dataset_files = find_files(args.dataset_dir, args.suffix)

    for class_name, file_paths in dataset_files.items():
        print(f"Binning class: {class_name}")

        parallel_parse_ms(
            ms_file_paths=file_paths,
            prefix=args.bin_prefix,
            mz_min=args.mz_min,
            mz_max=args.mz_max,
            bin_size=args.bin_size,
            workers=args.num_workers
        )

    print('Dataset Binning Process Completed.')
    print(f'Finished Step: Binning. Binned dataset directory: {os.path.join(args.dataset_dir, args.bin_prefix)}')


def process_patching(args):
    """
    Process the patching of binned MS files.

    :param args: Arguments containing parameters for patching.
    """
    # Patching binned MS files
    print('Patching binned MS Files...')

    bin_dir = os.path.join(args.dataset_dir, args.bin_prefix)
    binned_dataset_files = find_files(bin_dir, '.npz')

    for class_name, file_paths in binned_dataset_files.items():
        print(f"Patching Class: {class_name}")

        parallel_generate_patches(
            binned_file_paths=file_paths,
            prefix=args.patch_prefix,
            patch_strategy=args.patch_strategy,
            max_peaks=args.max_peaks,
            min_distance=args.min_distance,
            intensity_threshold=args.intensity_threshold,
            smoothing_sigma=args.smoothing_sigma,
            pnds_overlap=args.pnds_overlap,
            patch_width=args.patch_width,
            patch_height=args.patch_height,
            overlap_col=args.overlap_col,
            overlap_row=args.overlap_row,
            padding_value=args.padding_value,
            workers=args.num_workers
        )

    print('Dataset Patching Process Completed.')
    patch_dir = f'{args.patch_prefix}_{args.bin_prefix}'
    print(f'Finished Step: Patching. Patched dataset directory: {os.path.join(args.dataset_dir, patch_dir)}')


def process_score_calculation(args):
    """
    Process the calculation of scores for the patches (save inplace).

    :param args: Arguments containing parameters for score calculation.
    """
    # Calculate Scores for Patches
    print('Calculating Scores for Patches...')

    patch_dir = os.path.join(args.dataset_dir, f'{args.patch_prefix}_{args.bin_prefix}')
    patched_dataset_files = find_files(patch_dir, '.npz')

    for class_name, file_paths in patched_dataset_files.items():
        print(f"Calculating scores for Class: {class_name}")

        parallel_calculate_patches_scores(
            patched_file_paths=file_paths,
            score_strategy=args.score_strategy,
            workers=args.num_workers
        )

    print('Dataset Score Calculation Process Completed.')
    print(f'Finished Step: Score Calculation. Scored and patched dataset directory: {patch_dir}')


def process_patch_selection(args):
    """
    Process the selection of top K patches from the patched MS files.

    :param args: Arguments containing parameters for patch selection.
    """
    # Selecting Top K Patches
    print('Selecting Top K Patches...')

    patch_dir = os.path.join(args.dataset_dir, f'{args.patch_prefix}_{args.bin_prefix}')
    patched_dataset_files = find_files(patch_dir, '.npz')

    if args.selection_strategy == 'class_average':
        print(f'Generating shared indices per class using {args.score_strategy} scores...')
        sorted_indices_dict = {}
        sorted_indices_file_path = os.path.join(patch_dir, f'{args.selection_strategy}_{args.score_strategy}_sorted_indices.pkl')

        if os.path.exists(sorted_indices_file_path):
            try:
                with open(sorted_indices_file_path, 'rb') as f:
                    sorted_indices_dict = pickle.load(f)
            except Exception as e:
                raise RuntimeError(f"Error loading indices from {sorted_indices_file_path}: {e}")

        if not sorted_indices_dict:
            for class_name, file_paths in patched_dataset_files.items():
                print(f'Calculating average {args.score_strategy} scores for class {class_name}...')
                class_indices = calculate_average_scores_and_indices(
                    patched_file_paths=file_paths,
                    score_strategy=args.score_strategy,
                )
                sorted_indices_dict[class_name] = class_indices

        try:
            with open(sorted_indices_file_path, 'wb') as f:
                pickle.dump(sorted_indices_dict, f)
        except Exception as e:
            raise RuntimeError(f"Error saving indices to {sorted_indices_file_path}: {e}")

        for class_name, file_paths in patched_dataset_files.items():
            print(f"Selecting Top K Patches for Class: {class_name}")

            parallel_select_top_k_patches_per_class(
                patched_file_paths=file_paths,
                prefix=args.select_prefix,
                selection_strategy=args.selection_strategy,
                sorted_indices=sorted_indices_dict[class_name],
                top_k=args.top_k,
                workers=args.num_workers,
            )

    elif args.selection_strategy == 'per_file':
        print(f'Executing per file selection using {args.score_strategy} scores...')

        for class_name, file_paths in patched_dataset_files.items():
            print(f"Selecting Top K Patches for Class: {class_name}")

            parallel_select_top_k_patches_per_file(
                patched_file_paths=file_paths,
                prefix=args.select_prefix,
                selection_strategy=args.selection_strategy,
                score_strategy=args.score_strategy,
                top_k=args.top_k,
                workers=args.num_workers
            )

    print('Dataset Top K Patches Selection Process Completed.')
    select_dir = f'{args.select_prefix}_{args.patch_prefix}_{args.bin_prefix}'
    print(f'Finished Step: Patch Selection. Selected patches dataset directory: {os.path.join(args.dataset_dir, select_dir)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataset Process Workflow')
    parser.add_argument('--step', type=str, choices=['all', 'binning', 'patching', 'score_calculation', 'patch_selection'], default='all', help='Processing steps to execute')
    parser.add_argument('--dataset_dir', type=str, required=True, help='Path to the dataset directory')
    parser.add_argument('--suffix', type=str, default='mzML', help='File suffix to filter (e.g., .mzML, .mzXML)')
    parser.add_argument('--mz_min', type=float, required=True, help='Minimum m/z value for binning')
    parser.add_argument('--mz_max', type=float, required=True, help='Maximum m/z value for binning')
    parser.add_argument('--bin_size', type=float, required=True, help='Bin size for m/z binning')
    parser.add_argument('--patch_strategy', type=str, required=True, choices=['pcp', 'grid'], help='Strategy to generate patches (e.g., pcp, grid)')
    parser.add_argument('--max_peaks', type=int, default=-1, help='Maximum number of peaks to be extracted')
    parser.add_argument('--min_distance', type=int, default=16, help='Minimum distance between peaks (PCP-PNDS)')
    parser.add_argument('--intensity_threshold', type=float, default=0.1, help='Intensity threshold for peak extraction')
    parser.add_argument('--smoothing_sigma', type=float, default=0.05, help='Gaussian smoothing sigma for peak extraction')
    parser.add_argument('--pnds_overlap', type=float, default=0.2, help='Overlap ratio for PNDS sampling.')
    parser.add_argument('--patch_width', type=int, default=32, help='Width of the patches to be extracted')
    parser.add_argument('--patch_height', type=int, default=32, help='Height of the patches to be extracted')
    parser.add_argument('--overlap_col', type=int, default=0, help='Number of overlapping pixels between patches in the column direction')
    parser.add_argument('--overlap_row', type=int, default=0, help='Number of overlapping pixels between patches in the row direction')
    parser.add_argument('--padding_value', type=float, default=0.0, help='Padding value for the patches')
    parser.add_argument('--score_strategy', type=str, default='entropy', choices=['entropy', 'mean'], help='Strategy to calculate patch scores (e.g. Entropy: 1D image entropy, Mean: mean intensity, Random: random selection)')
    parser.add_argument('--selection_strategy', type=str, choices=['per_file', 'class_average'], help='Strategy for selecting patches (e.g., per_file, class_average)')
    parser.add_argument('--top_k', type=int, default=256, help='Number of patches to be selected')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker processes to use')
    parser.add_argument('--random_seed', type=int, default=3407, help='Random seed for reproducibility')

    args = parser.parse_args()

    # np.random.seed(args.random_seed)

    if '.' not in args.suffix:
        args.suffix = '.' + args.suffix

    args.bin_prefix = f'mz_{args.mz_min}-{args.mz_max}_bin_size_{args.bin_size}'
    if args.patch_strategy == 'pcp':
        args.selection_strategy = 'per_file'
        if args.smoothing_sigma == 0:
            args.patch_prefix = f'{args.patch_strategy}_patch_{args.patch_width}x{args.patch_height}_pnds_{args.pnds_overlap}_threshold_{args.intensity_threshold}'
        else:
            args.patch_prefix = f'{args.patch_strategy}_patch_{args.patch_width}x{args.patch_height}_pnds_{args.pnds_overlap}_threshold_{args.intensity_threshold}_sigma_{args.smoothing_sigma}'
    elif args.patch_strategy == 'grid':
        args.selection_strategy = 'class_average'
        args.patch_prefix = f'{args.patch_strategy}_patch_{args.patch_width}x{args.patch_height}_overlap_{args.overlap_col}x{args.overlap_row}'
    else:
        raise ValueError(f"Invalid patch strategy: {args.patch_strategy}. Choose either 'pcp' or 'grid'.")
    args.select_prefix = f'{args.score_strategy}_top_{args.top_k}' if args.score_strategy != 'random' else f'{args.score_strategy}_{args.top_k}'

    print('=' * 50)
    print('Starting Dataset Processing Workflow...')

    if args.step in ['all', 'binning']:
        process_binning(args)

    if args.step in ['all', 'patching']:
        if not os.path.exists(os.path.join(args.dataset_dir, args.bin_prefix)):
            raise FileNotFoundError(f'Binned dataset directory {os.path.join(args.dataset_dir, args.bin_prefix)} does not exist. Please run the binning step first.')

        process_patching(args)

    if args.step in ['all', 'patching', 'score_calculation', 'patch_selection']:
        if not os.path.exists(os.path.join(args.dataset_dir, f'{args.patch_prefix}_{args.bin_prefix}')):
            raise FileNotFoundError(f'Patched dataset directory {os.path.join(args.dataset_dir, f"{args.patch_prefix}_{args.bin_prefix}")} does not exist. Please run the patching step first.')

        process_score_calculation(args)

    if args.step in ['all', 'patching', 'score_calculation', 'patch_selection']:
        if not os.path.exists(os.path.join(args.dataset_dir, f'{args.patch_prefix}_{args.bin_prefix}')):
            raise FileNotFoundError(f"Patched dataset directory {os.path.join(args.dataset_dir, f'{args.patch_prefix}_{args.bin_prefix}')} does not exist. Please run the patching step first.")

        process_patch_selection(args)

    print('Dataset Process Workflow Completed.')
    print("=" * 50)
