import os
import numpy as np
from tqdm import tqdm
from pathlib import Path
from functools import partial
from multiprocessing import Pool


def _create_save_path(file_path, prefix):
    """
    Create a save path based on the original file path and a prefix.
    """
    p = Path(file_path)

    file_name = p.name
    class_name = p.parent.name
    class_parent_dir_name = p.parent.parent.name
    dataset_dir = p.parent.parent.parent

    new_dir_name = f"{prefix}_{class_parent_dir_name}"
    save_dir = dataset_dir / new_dir_name / class_name
    save_dir.mkdir(parents=True, exist_ok=True)

    new_file_name = f"{prefix}_{file_name}"
    save_path = save_dir / new_file_name
    return save_path


def select_top_k_patches(patched_file_path, prefix, sorted_indices, top_k, padding_value=0.0):
    try:
        if not (os.path.exists(patched_file_path) and os.path.getsize(patched_file_path) > 0):
            raise FileNotFoundError(f"File not found or empty: {patched_file_path}")

        save_path = _create_save_path(patched_file_path, prefix)
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


def parallel_select_top_k_patches(patched_file_paths, prefix, sorted_indices, top_k, workers=4):
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
            select_top_k_patches,
            prefix=prefix,
            sorted_indices=sorted_indices,
            top_k=top_k
        )

        results = list(
            tqdm(
                pool.imap_unordered(worker, patched_file_paths),
                total=len(patched_file_paths),
                desc=f'Selecting top K patches',
            )
        )

    success_rate = sum(results) / len(results)
    print(f"Patch Selection completed. Success rate: {success_rate:.2%}")


