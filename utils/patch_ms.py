import os
import numpy as np

from tqdm import tqdm
from scipy import sparse
from functools import partial
from multiprocessing import Pool


def extract_patches(image, patch_width=224, patch_height=224, overlap_col=0, overlap_row=0):
    """
    extract patches from the pseudo MS image.

    :param image: The pseudo MS image to extract patches from (2D array).
    :param patch_width: The width of the patches to be extracted.
    :param patch_height: The height of the patches to be extracted.
    :param overlap_col: The number of overlapping pixels between patches in the column direction.
    :param overlap_row: The number of overlapping pixels between patches in the row direction.
    :return: patches (np.ndarray): Extracted patches of shape (num_patches, patch_height, patch_width).
    :return: positions (np.ndarray): Corresponding positions of patches as (num_patches, 2) where each row is (row_idx, col_idx).
    """
    assert len(image.shape) == 2, "Image should be a 2D array"
    H, W = image.shape

    step_h = patch_height - overlap_row
    step_w = patch_width - overlap_col

    patches = []
    positions = []

    for y in range(0, H - patch_height + 1, step_h):
        for x in range(0, W - patch_width + 1, step_w):
            patch = image[y:y + patch_height, x:x + patch_width]
            patches.append(patch)
            positions.append((y, x))

    return np.array(patches), np.array(positions)


def calculate_entropy(image):
    """
    Calculate the 1D entropy of an image.

    :param image: Image matrix with 2D (e.g., 224x224).
    :return: 1D Image entropy
    """
    assert image.min() >= 0 and image.max() <= 1, "Image values should be in the range [0, 1]"

    image = np.array(image)
    if image.sum() == 0:
        return 0

    hist, _ = np.histogram(image, bins=256, range=(0, 1))
    P = hist / hist.sum()
    P[P == 0] = 1  # log2(1) = 0, so we avoid log(0)
    entropy = -np.sum(P * np.log2(P))
    return entropy


def calculate_mean(image):
    """
    Calculate the mean of an image.

    :param image: Image matrix with 2D (e.g., 224x224).
    :return: pool_int: Mean value of the image.
    """
    image = np.array(image)
    mean = np.mean(image)
    return mean


def calculate_patches_top_k(file_path, prefix, method, patch_width, patch_height, overlap_col, overlap_row):
    try:
        save_dir = os.path.dirname(file_path)
        file_name = os.path.basename(file_path)
        save_path = os.path.join(save_dir, f'{prefix}_{file_name}')

        if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
            return np.load(save_path)['scores']

        sparse_matrix = sparse.load_npz(file_path)
        raw_image = sparse_matrix.toarray()
        raw_image = raw_image / raw_image.max()  # Normalize the image to [0, 1]

        patches, positions = extract_patches(
            image=raw_image,
            patch_width=patch_width,
            patch_height=patch_height,
            overlap_col=overlap_col,
            overlap_row=overlap_row
        )

        if method == 'entropy':
            scores = np.array([calculate_entropy(patch) for patch in patches])
        elif method == 'mean':
            scores = np.array([calculate_mean(patch) for patch in patches])
        else:
            raise ValueError('Invalid method. Choose either "entropy" or "mean".')

        np.savez_compressed(save_path, patches=patches, positions=positions, scores=scores)

        return scores
    except Exception as e:
        raise RuntimeError(f"Error processing {file_path}: {e}")


def parallel_calculate_patches_top_k(file_paths, prefix, method='entropy', patch_width=224, patch_height=224, overlap_col=0, overlap_row=0, top_k=512, workers=4):
    """
    Calculate the top k patches based on the specified method (entropy or mean) from the pseudo MS images.

    :param file_paths: List of file paths to the pseudo MS images.
    :param prefix: Prefix for the save path pattern.
    :param method: Method to calculate (e.g. Entropy: 1D image entropy, Mean: mean intensity).
    :param patch_width: The width of the patches to be extracted.
    :param patch_height: The height of the patches to be extracted.
    :param overlap_col: The number of overlapping pixels between patches in the column direction.
    :param overlap_row: The number of overlapping pixels between patches in the row direction.
    :param top_k: The number of patches to be selected.
    :param workers: Number of worker processes to use.
    :return top_k_indices: Indices of the top k patches.
    """
    if method not in ['entropy', 'mean']:
        raise ValueError('Invalid method. Choose either "entropy" or "mean".')

    with Pool(processes=workers) as pool:
        worker = partial(
            calculate_patches_top_k,
            prefix=prefix,
            method=method,
            patch_width=patch_width,
            patch_height=patch_height,
            overlap_col=overlap_col,
            overlap_row=overlap_row
        )

        results = list(
            tqdm(
                pool.imap_unordered(worker, file_paths),
                total=len(file_paths),
                desc='Calculating patches scores',
            )
        )

    valid_results = [result for result in results if len(result) > 0]
    if not valid_results:
        raise RuntimeError("No valid files processed")

    """
    When processing mass spectrometry data, the resulting matrix has a shape of (mz_bins, scans), 
    where mz_bins is fixed, but the number of scans may vary depending on how many spectra were collected in each file. 
    To ensure comparability during patch scoring and selection, we normalize the number of scores across all files. 
    Specifically, we truncate all score arrays to the same minimum length, retaining only the initial portion of patches for each file. 
    Any extra patches in files with more scans are discarded, which does not compromise the fairness or consistency of the overall evaluation.
    """
    min_len = min(len(scores) for scores in valid_results)
    valid_results_trimmed = [scores[:min_len] for scores in valid_results]

    patch_scores = sum(valid_results_trimmed)
    avg_scores = patch_scores / len(valid_results_trimmed)
    top_k_indices = np.argsort(avg_scores)[-top_k:][::-1]  # Get top k indices

    return top_k_indices


def select_top_k_patches(file_path, prefix, top_k_indices):
    try:
        save_dir = os.path.dirname(file_path)
        file_name = os.path.basename(file_path)
        save_path = os.path.join(save_dir, f"{prefix}_{file_name}")

        if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
            return True

        patched_file = np.load(file_path)
        patches = patched_file['patches']
        positions = patched_file['positions']

        num_patches = len(patches)

        patch_height, patch_width = patches[0].shape[:2]

        zero_padding = np.zeros((patch_width, patch_height), dtype=patches[0].dtype)

        selected_patches = []
        selected_positions = []

        for idx in top_k_indices:
            if idx < num_patches:
                selected_patches.append(patches[idx])
                selected_positions.append(positions[idx])
            else:
                selected_patches.append(zero_padding)
                selected_positions.append((-1, -1))

        selected_patches = np.array(selected_patches)
        selected_positions = np.array(selected_positions)
        padding_mask = (selected_positions == -1).all(axis=1)

        np.savez_compressed(save_path, patches=selected_patches, positions=selected_positions, padding_mask=padding_mask)

        return True
    except Exception as e:
        raise RuntimeError(f"Error processing {file_path}: {e}")


def parallel_select_top_k_patches(file_paths, prefix, top_k_indices, workers=4):
    """
    Select top K patches based on the provided indices and save them.

    :param file_paths: A list of file paths to the pseudo MS images.
    :param prefix: Prefix for the save path pattern.
    :param top_k_indices: Indices of the top K patches.
    :param workers: Number of worker processes to use.
    """
    with Pool(processes=workers) as pool:
        worker = partial(
            select_top_k_patches,
            prefix=prefix,
            top_k_indices=top_k_indices,
        )

        results = list(
            tqdm(
                pool.imap_unordered(worker, file_paths),
                total=len(file_paths),
                desc='Selecting top K patches',
            )
        )

    success_rate = sum(results) / len(results)
    print(f"Selection completed. Success rate: {success_rate:.2%}")