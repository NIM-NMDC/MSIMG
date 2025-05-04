import os
import numpy as np
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool


def select_top_k_patches_per_class(patched_file_path, prefix, sorted_indices, top_k, padding_value=0.0):
    try:
        if not (os.path.exists(patched_file_path) and os.path.getsize(patched_file_path) > 0):
            raise FileNotFoundError(f"File not found or empty: {patched_file_path}")

        previous_prefix = os.path.basename(os.path.dirname(os.path.dirname(patched_file_path)))
        class_name = os.path.basename(os.path.dirname(patched_file_path))
        file_name = os.path.basename(patched_file_path)

        dataset_dir = os.path.abspath(os.path.join(patched_file_path, '../../..'))
        save_dir = os.path.join(dataset_dir, f'{prefix}_{previous_prefix}', class_name)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'{prefix}_{file_name}')

        if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
            return True

        with np.load(patched_file_path) as patched_file_data:
            if 'patches' not in patched_file_data or 'positions' not in patched_file_data:
                raise ValueError(f"'patches' or 'positions' not found in {patched_file_path}")

            patches = patched_file_data['patches']
            positions = patched_file_data['positions']

        num_patches = len(patches)
        top_k_indices = sorted_indices[:top_k]

        padding_patch = np.full(patches[0].shape, padding_value, dtype=patches[0].dtype)
        padding_position = np.array([-1, -1], dtype=positions[0].dtype)

        selected_patches = []
        selected_positions = []
        padding_mask = []
        for idx in top_k_indices:
            selected_patches.append(patches[idx])
            selected_positions.append(positions[idx])
            padding_mask.append(False)

        num_selected = len(selected_patches)
        if num_selected < top_k:
            num_padding = top_k - num_selected

            selected_patches.extend([padding_patch] * num_padding)
            selected_positions.extend([padding_position] * num_padding)
            padding_mask.extend([True] * num_padding)

        selected_patches = np.array(selected_patches)
        selected_positions = np.array(selected_positions)
        padding_mask = np.array(padding_mask)

        np.savez_compressed(save_path, patches=selected_patches, positions=selected_positions, padding_mask=padding_mask)

        return True
    except Exception as e:
        raise RuntimeError(f"Error processing {patched_file_path}: {e}")


def parallel_select_top_k_patches_per_class(patched_file_paths, prefix, selection_strategy, sorted_indices, top_k, workers=4):
    """
    Select top K patches based on the provided indices and save them.

    :param patched_file_paths: A list of file paths to the pseudo MS images.
    :param prefix: Prefix for the save path pattern.
    :param selection_strategy: Strategy for selecting patches ('class_average' or 'per_file)
    :param sorted_indices: Sorted indices of patches to select from.
    :param top_k: Number of top patches to select.
    :param workers: Number of worker processes to use.
    """
    with Pool(processes=workers) as pool:
        worker = partial(
            select_top_k_patches_per_class,
            prefix=prefix,
            sorted_indices=sorted_indices,
            top_k=top_k
        )

        results = list(
            tqdm(
                pool.imap_unordered(worker, patched_file_paths),
                total=len(patched_file_paths),
                desc=f'Selecting {selection_strategy} top K patches',
            )
        )

    success_rate = sum(results) / len(results)
    print(f"Patch Selection completed. Success rate: {success_rate:.2%}")


def select_top_k_patches_per_file(patched_file_path, prefix, score_strategy, top_k, padding_value=0.0):
    try:
        if not (os.path.exists(patched_file_path) and os.path.getsize(patched_file_path) > 0):
            raise FileNotFoundError(f"File not found or empty: {patched_file_path}")

        previous_prefix = os.path.basename(os.path.dirname(os.path.dirname(patched_file_path)))
        class_name = os.path.basename(os.path.dirname(patched_file_path))
        file_name = os.path.basename(patched_file_path)

        dataset_dir = os.path.abspath(os.path.join(patched_file_path, '../../..'))
        save_dir = os.path.join(dataset_dir, f'{prefix}_{previous_prefix}', class_name)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'{prefix}_{file_name}')

        if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
            return True

        with np.load(patched_file_path) as patched_file_data:
            if 'patches' not in patched_file_data or 'positions' not in patched_file_data:
                raise ValueError(f"'patches' or 'positions' not found in {patched_file_path}")

            patches = patched_file_data['patches']
            positions = patched_file_data['positions']
            num_patches = len(patches)

            top_k_indices = np.array([], dtype=int)
            if score_strategy not in patched_file_data or len(patched_file_data[score_strategy]) != num_patches:
                raise ValueError(f"Missing {score_strategy} scores in {patched_file_path} or mismatch between number of patches {num_patches}. Please recalculate.")

            scores = patched_file_data[score_strategy]

            if len(scores) != num_patches:
                raise ValueError(f"Mismatch between number of patches {num_patches} and scores {len(scores)} in {patched_file_path}")
            sorted_indices = np.argsort(scores)[::-1]
            top_k_indices = sorted_indices[:top_k]

        padding_patch = np.full(patches[0].shape, padding_value, dtype=patches[0].dtype)
        padding_position = np.array([-1, -1], dtype=positions[0].dtype)

        selected_patches = []
        selected_positions = []
        padding_mask = []
        for idx in top_k_indices:
            selected_patches.append(patches[idx])
            selected_positions.append(positions[idx])
            padding_mask.append(False)

        num_selected = len(selected_patches)
        if num_selected < top_k:
            num_padding = top_k - num_selected

            selected_patches.extend([padding_patch] * num_padding)
            selected_positions.extend([padding_position] * num_padding)
            padding_mask.extend([True] * num_padding)

        selected_patches = np.array(selected_patches)
        selected_positions = np.array(selected_positions)
        padding_mask = np.array(padding_mask)

        np.savez_compressed(save_path, patches=selected_patches, positions=selected_positions, padding_mask=padding_mask)

        return True
    except Exception as e:
        raise RuntimeError(f"Error processing {patched_file_path}: {e}")


def parallel_select_top_k_patches_per_file(patched_file_paths, prefix, selection_strategy, score_strategy, top_k, workers=4):
    """
    Select top K patches for each file based on the score strategy and save them.

    :param patched_file_paths: A list of file paths to the pseudo MS images.
    :param prefix: Prefix for the save path pattern.
    :param selection_strategy: Strategy for selecting patches ('class_average' or 'per_file').
    :param score_strategy: Strategy for scoring patches ('entropy', 'mean', or 'random').
    :param top_k: Number of top patches to select.
    :param workers: Number of worker processes to use.
    """
    with Pool(processes=workers) as pool:
        worker = partial(
            select_top_k_patches_per_file,
            prefix=prefix,
            score_strategy=score_strategy,
            top_k=top_k,
        )

        results = list(
            tqdm(
                pool.imap_unordered(worker, patched_file_paths),
                total=len(patched_file_paths),
                desc=f'Selecting {selection_strategy} top K patches',
            )
        )

    success_rate = sum(results) / len(results)
    print(f"Patch Selection completed. Success rate: {success_rate:.2%}")


