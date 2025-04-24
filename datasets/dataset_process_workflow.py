import os
import pickle
import shutil
import argparse
import numpy as np

from collections import defaultdict

from utils.rasterize_ms import parallel_parse_ms
from utils.patch_ms import (
    parallel_generate_patches,
    parallel_calculate_patches_scores,
    parallel_get_patches_numbers,
    parallel_select_top_k_patches
)


def process_binning(args):
    """
    Process the binning of MS files.

    :param args: Arguments containing parameters for binning.
    """
    # Binning MS files
    print('Binning MS Files...')

    # mzML or mzXML
    dataset_files = defaultdict(list)
    for class_name in os.listdir(args.dataset_dir):
        class_dir = os.path.join(args.dataset_dir, class_name)
        if os.path.isdir(class_dir):
            for file_name in os.listdir(class_dir):
                if file_name.endswith(args.suffix):
                    dataset_files[class_name].append(os.path.join(class_dir, file_name))

    binned_dataset_files = defaultdict(list)
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
    return os.path.join(args.dataset_dir, args.bin_prefix)


def process_patching(args):
    """
    Process the patching of binned MS files.

    :param args: Arguments containing parameters for binning.
    """
    # Patching binned MS files
    print('Patching binned MS Files...')

    bin_dir = os.path.join(args.dataset_dir, args.bin_prefix)

    binned_dataset_files = defaultdict(list)
    for class_name in os.listdir(bin_dir):
        class_dir = os.path.join(bin_dir, class_name)
        if os.path.isdir(class_dir):
            for file_name in os.listdir(class_dir):
                if file_name.endswith('.npz'):
                    binned_dataset_files[class_name].append(os.path.join(class_dir, file_name))

    patched_dataset_files = defaultdict(list)
    for class_name, file_paths in binned_dataset_files.items():
        print(f"Patching Class: {class_name}")

        parallel_generate_patches(
            binned_file_paths=file_paths,
            prefix=args.patch_prefix,
            method=args.select_method,
            patch_width=args.patch_width,
            patch_height=args.patch_height,
            overlap_col=args.overlap_col,
            overlap_row=args.overlap_row,
            workers=args.num_workers
        )

    print('Dataset Patching Process Completed.')
    patch_dir = f"{args.patch_prefix}_{args.bin_prefix}"
    return os.path.join(args.dataset_dir, patch_dir)


def process_sorted_indices_calculation(args):
    """
    Calculate the sorted indices for the patches.

    :param args: Arguments containing parameters for binning.
    """
    # Calculate Sorted Indices
    print('Calculating Sorted Indices...')

    patch_dir = os.path.join(args.dataset_dir, f'{args.patch_prefix}_{args.bin_prefix}')

    patched_dataset_files = defaultdict(list)
    for class_name in os.listdir(patch_dir):
        class_dir = os.path.join(patch_dir, class_name)
        if os.path.isdir(class_dir):
            for file_name in os.listdir(class_dir):
                if file_name.endswith('.npz'):
                    patched_dataset_files[class_name].append(os.path.join(class_dir, file_name))

    sorted_indices_file_name = f"{args.select_method}_sorted_indices.pkl"
    sorted_indices_file_path = os.path.join(patch_dir, sorted_indices_file_name)

    recalculate = False
    if os.path.exists(sorted_indices_file_path):
        print(f"Sorted indices file already exists: {sorted_indices_file_path}")
        try:
            with open(sorted_indices_file_path, 'rb') as f:
                sorted_indices_dict = pickle.load(f)
            # Basic validation
            if not isinstance(sorted_indices_dict, dict) or not all(isinstance(v, np.ndarray) for v in sorted_indices_dict.values()):
                print("Warning: Invalid sorted indices file format. Recalculating...")
                recalculate = True
            elif set(sorted_indices_dict.keys()) != set(patched_dataset_files.keys()):
                print("Warning: Class names in sorted indices file do not match the patched dataset. Recalculating...")
                recalculate = True
            else:
                print("Sorted indices file is valid. Skipping calculation.")
        except Exception as e:
            print(f"Warning:Error loading sorted indices file: {e}. Recalculating...")
            recalculate = True
    else:
        print(f"Sorted indices file does not exist: {sorted_indices_file_path}. Recalculating...")
        recalculate = True

    if recalculate:
        print(f"Calculating patches {args.select_method} and get sorted indices...")
        sorted_indices_dict = {}
        for class_name, file_paths in patched_dataset_files.items():
            print(f"Calculating {args.select_method} scores for Class: {class_name}")

            sorted_indices = parallel_calculate_patches_scores(
                patched_file_paths=file_paths,
                method=args.select_method,
                workers=args.num_workers
            )

            sorted_indices_dict[class_name] = sorted_indices

        # Save sorted indices to a file
        try:
            with open(sorted_indices_file_path, 'wb') as f:
                pickle.dump(sorted_indices_dict, f)
        except IOError as e:
            raise IOError(f"Error saving sorted indices file: {e}")

        print(f"Sorted indices file saved: {sorted_indices_file_path}")


def process_random_indices(args):
    """
    Generate random indices for the patches.

    :param args: Arguments containing parameters for binning.
    """
    # Generate Random Indices
    print('Generating Random Indices...')

    patch_dir = os.path.join(args.dataset_dir, f'{args.patch_prefix}_{args.bin_prefix}')

    patched_dataset_files = defaultdict(list)
    for class_name in os.listdir(patch_dir):
        class_dir = os.path.join(patch_dir, class_name)
        if os.path.isdir(class_dir):
            for file_name in os.listdir(class_dir):
                if file_name.endswith('.npz'):
                    patched_dataset_files[class_name].append(os.path.join(class_dir, file_name))

    random_indices_file_name = f"{args.select_method}_indices.pkl"
    random_indices_file_path = os.path.join(patch_dir, random_indices_file_name)

    get_random_indices = False
    if os.path.exists(random_indices_file_path):
        print(f"Random indices file already exists: {random_indices_file_path}")
        try:
            with open(random_indices_file_path, 'rb') as f:
                random_indices_dict = pickle.load(f)
            # Basic validation
            if not isinstance(random_indices_dict, dict) or not all(
                    isinstance(v, np.ndarray) for v in random_indices_dict.values()):
                print("Warning: Invalid random indices file format. Get random indices...")
                get_random_indices = True
            elif set(random_indices_dict.keys()) != set(patched_dataset_files.keys()):
                print("Warning: Class names in random indices file do not match the patched dataset. Get random indices...")
                get_random_indices = True
            else:
                print("Random indices file is valid. Skipping get random indices.")
        except Exception as e:
            print(f"Warning:Error loading random indices file: {e}. Get random indices...")
            get_random_indices = True
    else:
        print(f"Random indices file does not exist: {random_indices_file_path}. Get random indices...")
        get_random_indices = True

    if get_random_indices:
        print(f"Get random indices...")
        random_indices_dict = {}
        for class_name, file_paths in patched_dataset_files.items():
            print(f"Generating random indices for Class: {class_name}")

            patches_numbers = parallel_get_patches_numbers(
                patched_file_paths=file_paths,
                workers=args.num_workers
            )

            # Get min patches number
            min_patches_number = min(patches_numbers)
            random_indices = np.random.permutation(np.arange(min_patches_number))
            random_indices_dict[class_name] = random_indices

        # Save sorted indices to a file
        try:
            with open(random_indices_file_path, 'wb') as f:
                pickle.dump(random_indices_dict, f)
        except IOError as e:
            raise IOError(f"Error saving random indices file: {e}")

        print(f"Random indices file saved: {random_indices_file_path}")


def process_patch_selection(args):
    """
    Process the selection of top K patches from the patched MS files.

    :param args: Arguments containing parameters for binning.
    """
    # Selecting Top K Patches
    print('Selecting Top K Patches...')

    patch_dir = os.path.join(args.dataset_dir, f'{args.patch_prefix}_{args.bin_prefix}')

    # Load top_k_indices_dict from the file
    if args.select_method != 'random':
        sorted_indices_file_path = os.path.join(patch_dir, f"{args.select_method}_sorted_indices.pkl")
    else:
        sorted_indices_file_path = os.path.join(patch_dir, f"{args.select_method}_indices.pkl")
    with open(sorted_indices_file_path, 'rb') as f:
        sorted_indices_dict = pickle.load(f)

    if args.top_k > len(sorted_indices_dict[list(sorted_indices_dict.keys())[0]]):
        raise ValueError(f"Top K value {args.top_k} exceeds the number of patches available in the dataset.")

    patched_dataset_files = defaultdict(list)
    for class_name in os.listdir(patch_dir):
        class_dir = os.path.join(patch_dir, class_name)
        if os.path.isdir(class_dir):
            for file_name in os.listdir(class_dir):
                if file_name.endswith('.npz'):
                    patched_dataset_files[class_name].append(os.path.join(class_dir, file_name))

    selected_patches_dataset_files = defaultdict(list)

    for class_name, file_paths in patched_dataset_files.items():
        print(f"Selecting Top K Patches for Class: {class_name}")

        top_k_indices = sorted_indices_dict[class_name][:args.top_k]
        parallel_select_top_k_patches(
            patched_file_paths=file_paths,
            prefix=args.select_prefix,
            top_k_indices=top_k_indices,
            workers=args.num_workers
        )

    print('Dataset Top K Patches Selection Process Completed.')
    select_dir = f'{args.select_prefix}_{args.patch_prefix}_{args.bin_prefix}'
    return os.path.join(args.dataset_dir, select_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataset Process Workflow')
    parser.add_argument('--step', type=str, choices=['all', 'binning', 'patching', 'score_calculation', 'patch_selection'], default='all', help='Processing steps to execute')
    parser.add_argument('--dataset_dir', type=str, required=True, help='Path to the dataset directory')
    parser.add_argument('--suffix', type=str, default='mzML', help='File suffix to filter (e.g., .mzML, .mzXML)')
    parser.add_argument('--mz_min', type=float, required=True, help='Minimum m/z value for binning')
    parser.add_argument('--mz_max', type=float, required=True, help='Maximum m/z value for binning')
    parser.add_argument('--bin_size', type=float, required=True, help='Bin size for m/z binning')
    parser.add_argument('--select_method', type=str, default='entropy', help='Method to calculate (e.g. Entropy: 1D image entropy, Mean: mean intensity, Random: random selection)')
    parser.add_argument('--patch_width', type=int, default=224, help='Width of the patches to be extracted')
    parser.add_argument('--patch_height', type=int, default=224, help='Height of the patches to be extracted')
    parser.add_argument('--overlap_col', type=int, default=0, help='Number of overlapping pixels between patches in the column direction')
    parser.add_argument('--overlap_row', type=int, default=0, help='Number of overlapping pixels between patches in the row direction')
    parser.add_argument('--top_k', type=int, default=512, help='Number of patches to be selected')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker processes to use')
    parser.add_argument('--random_seed', type=int, default=3407, help='Random seed for reproducibility')

    args = parser.parse_args()

    # np.random.seed(args.random_seed)

    if '.' not in args.suffix:
        args.suffix = '.' + args.suffix

    args.bin_prefix = f'mz_{args.mz_min}-{args.mz_max}_bin_size_{args.bin_size}'
    args.patch_prefix = f'patch_{args.patch_width}x{args.patch_height}_overlap_{args.overlap_col}x{args.overlap_row}'
    args.select_prefix = f'{args.select_method}_top_{args.top_k}' if args.select_method != 'random' else f'{args.select_method}_{args.top_k}'

    if args.step in ['all', 'binning']:
        bin_dir = process_binning(args)
        print(f'Binned dataset directory: {bin_dir}')

    if args.step in ['all', 'patching']:
        if args.step == 'patching' and not os.path.exists(
            os.path.join(args.dataset_dir, args.bin_prefix)
        ):
            raise FileNotFoundError(f'Binned dataset directory {os.path.join(args.dataset_dir, args.bin_prefix)} does not exist. Please run the binning step first.')

        patch_dir = process_patching(args)
        print(f'Patched dataset directory: {patch_dir}')

    if args.step in ['all', 'score_calculation', 'patch_selection']:
        if args.step == 'score_calculation' and not os.path.exists(
            os.path.join(args.dataset_dir, f'{args.patch_prefix}_{args.bin_prefix}')
        ):
            raise FileNotFoundError(f'Patched dataset directory {os.path.join(args.dataset_dir, f"{args.patch_prefix}_{args.bin_prefix}")} does not exist. Please run the patching step first.')

        if args.select_method != 'random':
            process_sorted_indices_calculation(args)
            print(f'Sorted indices calculation completed.')
        else:
            process_random_indices(args)
            print(f'Random indices generation completed.')

    if args.step in ['all', 'patch_selection']:
        if args.step == 'patch_selection' and not os.path.exists(
            os.path.join(args.dataset_dir, f'{args.patch_prefix}_{args.bin_prefix}')
        ):
            raise FileNotFoundError(f"Patched dataset directory {os.path.join(args.dataset_dir, f'{args.patch_prefix}_{args.bin_prefix}')} does not exist. Please run the patching step first.")

        select_dir = process_patch_selection(args)
        print(f'Selected patches dataset directory: {select_dir}')

    print('Dataset Process Workflow Completed.')