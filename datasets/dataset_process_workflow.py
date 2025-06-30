import os
import yaml
import pickle
import shutil
import argparse
import numpy as np

from collections import defaultdict

from utils.split_utils import split_dataset_files_by_class_stratified
from utils.rasterize_ms import parallel_parse_ms
from utils.patch_ms import parallel_get_anchor_points, parallel_generate_patches
from utils.score_patches import parallel_calculate_patches_scores, calculate_average_scores_and_indices
from utils.select_patches import parallel_select_top_k_patches


def load_params_from_yaml(file_path, key=None):
    """
    Load parameters from a YAML file.

    :param file_path: Path to the YAML file.
    :return: Dictionary containing the parameters.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"YAML file {file_path} does not exist.")

    with open(file_path, 'r') as file:
        params = yaml.safe_load(file)

    if not isinstance(params, dict):
        raise ValueError("YAML file must contain a dictionary of parameters.")

    if key:
        if key not in params:
            raise KeyError(f"Key '{key}' not found in the YAML file.")
        return params.get(key)
    else:
        return params


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
    # binned_file_paths = [file_path for file_paths in binned_dataset_files.values() for file_path in file_paths]
    binned_train_set, _ = split_dataset_files_by_class_stratified(
        dataset_dir=bin_dir,
        suffix='.npz',
        train_size=0.8,
        test_size=0.2,
        random_seed=args.random_seed
    )
    binned_train_file_paths = [item['file_path'] for item in binned_train_set]

    if args.patch_strategy == 'ics':
        print('Using Information Content Sampling (ICS) for patch generation, creating global anchor points for patching.')
        global_anchor_points_path = os.path.join(
            bin_dir,
            f"{args.patch_params.get('information_metric')}_detection_window_{args.patch_params.get('detection_window_height')}x{args.patch_params.get('detection_window_width')}_"
            f"step_{args.patch_params.get('step_height')}x{args.patch_params.get('step_width')}_{args.patch_params.get('detection_metric_type')}_{args.patch_params.get('metric_threshold')}_"
            f"global_anchor_points.pkl"
        )
        if os.path.exists(global_anchor_points_path):
            print(f"Loading existing global anchor points from {global_anchor_points_path}")
            with open(global_anchor_points_path, 'rb') as f:
                global_anchor_points = pickle.load(f)
        else:
            print("Calculating global anchor points for ICS patching...")
            anchor_points_lists = parallel_get_anchor_points(
                binned_file_paths=binned_train_file_paths,
                patch_params=args.patch_params,
                workers=args.num_workers
            )

            print("Aggregating and deduplicate anchor points...")
            global_anchor_points_set = set()
            for anchor_points_list in anchor_points_lists:
                global_anchor_points_set.update(tuple(anchor_point) for anchor_point in anchor_points_list)

            global_anchor_points = np.array(list(global_anchor_points_set))
            print(f"Saving global anchor points to {global_anchor_points_path}")
            with open(global_anchor_points_path, 'wb') as f:
                pickle.dump(global_anchor_points, f)

            print(f"Total unique global anchor points: {len(global_anchor_points)}")

    for class_name, file_paths in binned_dataset_files.items():
        print(f"Patching Class: {class_name}")

        parallel_generate_patches(
            binned_file_paths=file_paths,
            prefix=args.patch_prefix,
            patch_strategy=args.patch_strategy,
            patch_params=args.patch_params,
            global_anchor_points=global_anchor_points if args.patch_strategy == 'ics' else None,
            workers=args.num_workers
        )

    print('Dataset Patching Process Completed.')
    patch_dir = f'{args.patch_prefix}_{args.bin_prefix}'
    print(f'Finished Step: Patching. Patched dataset directory: {os.path.join(args.dataset_dir, patch_dir)}')


def process_patch_selection(args):
    """
    Process the score calculation and selection of top K patches from the patched MS files.

    :param args: Arguments containing parameters for patch selection.
    """
    # Calculate Scores for Patches
    print('Calculating Scores for Patches...')

    patch_dir = os.path.join(args.dataset_dir, f'{args.patch_prefix}_{args.bin_prefix}')
    patched_dataset_files = find_files(patch_dir, '.npz')
    # patched_file_paths = [file_path for file_paths in patched_dataset_files.values() for file_path in file_paths]
    patched_train_set, _ = split_dataset_files_by_class_stratified(
        dataset_dir=patch_dir,
        suffix='.npz',
        train_size=0.8,
        test_size=0.2,
        random_seed=args.random_seed
    )
    patched_train_file_paths = [item['file_path'] for item in patched_train_set]

    for class_name, file_paths in patched_dataset_files.items():
        print(f"Calculating scores for Class: {class_name}")

        parallel_calculate_patches_scores(
            patched_file_paths=file_paths,
            score_strategy=args.score_strategy,
            workers=args.num_workers
        )

    print('Dataset Score Calculation Process Completed.')
    print(f'Finished Step: Score Calculation. Scored dataset directory: {patch_dir}')

    # Selecting Top K Patches
    print('Selecting Top K Patches...')
    print(f'Generating global sorted indices using {args.score_strategy} scores...')
    global_sorted_indices_file_path = os.path.join(
        patch_dir,
        f"{args.patch_params.get('information_metric')}_detection_window_{args.patch_params.get('detection_window_height')}x{args.patch_params.get('detection_window_width')}_"
        f"step_{args.patch_params.get('step_height')}x{args.patch_params.get('step_width')}_{args.patch_params.get('detection_metric_type')}_{args.patch_params.get('metric_threshold')}_"
        f"global_{args.score_strategy}_sorted_indices.pkl"
    )

    global_sorted_indices = None
    if os.path.exists(global_sorted_indices_file_path):
        print(f"Loading existing global sorted indices from {global_sorted_indices_file_path}")
        try:
            with open(global_sorted_indices_file_path, 'rb') as f:
                global_sorted_indices = pickle.load(f)
        except Exception as e:
            raise RuntimeError(f"Error loading indices from {global_sorted_indices_file_path}: {e}")

    if global_sorted_indices is None:
        print(f'Calculating global average {args.score_strategy} scores for {len(patched_train_file_paths)} files...')
        global_sorted_indices = calculate_average_scores_and_indices(
            patched_file_paths=patched_train_file_paths,
            score_strategy=args.score_strategy,
        )

        print(f"Saving global average {args.score_strategy} sorted indices to {global_sorted_indices_file_path}")
        with open(global_sorted_indices_file_path, 'wb') as f:
            pickle.dump(global_sorted_indices, f)

    # print(f"Selecting Top K Patches for {len(patched_file_paths)} files...")
    for class_name, file_paths in patched_dataset_files.items():
        print(f"Calculating scores for Class: {class_name}")
        parallel_select_top_k_patches(
            patched_file_paths=file_paths,
            prefix=args.select_prefix,
            sorted_indices=global_sorted_indices,
            top_k=args.top_k,
            workers=args.num_workers,
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
    parser.add_argument('--patch_strategy', type=str, required=True, choices=['grid', 'ics'], help='Strategy to generate patches (e.g., grid, ics)')
    parser.add_argument('--score_strategy', type=str, default='entropy', choices=['entropy', 'mean'], help='Strategy to calculate patch scores (e.g. Entropy: 1D image entropy, Mean: mean intensity, Random: random selection)')
    parser.add_argument('--top_k', type=int, default=256, help='Number of patches to be selected')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker processes to use')
    parser.add_argument('--random_seed', type=int, default=3407, help='Random seed for reproducibility')

    args = parser.parse_args()

    if '.' not in args.suffix:
        args.suffix = '.' + args.suffix

    args.bin_prefix = f'mz_{args.mz_min}-{args.mz_max}_bin_size_{args.bin_size}'

    if args.patch_strategy == 'grid':
        patch_params = load_params_from_yaml('../configs/patch_config.yaml', key=args.patch_strategy)
        args.patch_prefix = f"{args.patch_strategy}_patch_{patch_params.get('patch_height')}x{patch_params.get('patch_width')}_overlap_{patch_params.get('overlap_row')}x{patch_params.get('overlap_col')}"
        args.patch_params = patch_params
    elif args.patch_strategy == 'ics':
        patch_params = load_params_from_yaml('../configs/patch_config.yaml', key=args.patch_strategy)
        args.patch_prefix = f"{patch_params.get('information_metric')}_{args.patch_strategy}_patch_{patch_params.get('patch_height')}x{patch_params.get('patch_width')}_{patch_params.get('detection_metric_type')}_{patch_params.get('metric_threshold')}"
        args.patch_params = patch_params
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

    if args.step in ['all', 'patch_selection']:
        if not os.path.exists(os.path.join(args.dataset_dir, f'{args.patch_prefix}_{args.bin_prefix}')):
            raise FileNotFoundError(f"Patched dataset directory {os.path.join(args.dataset_dir, f'{args.patch_prefix}_{args.bin_prefix}')} does not exist. Please run the patching step first.")

        process_patch_selection(args)

    print('Dataset Process Workflow Completed.')
    print("=" * 50)
