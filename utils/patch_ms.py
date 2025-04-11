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
    patches = []
    positions = []

    step_col = patch_width - overlap_col
    step_row = patch_height - overlap_row

    for row_idx, i in enumerate(range(0, H - patch_height + 1, step_row)):
        for col_idx, j in enumerate(range(0, W - patch_width + 1, step_col)):
            patch = image[i:i + patch_height, j:j + patch_width]
            patches.append(patch)
            positions.append((row_idx, col_idx))

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

    save_dir = os.path.dirname(file_path)
    file_name = os.path.basename(file_path)
    save_path = os.path.join(save_dir, f'{prefix}_{file_name}')
    np.savez_compressed(save_path, patches=patches, positions=positions)

    return scores


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

    patch_scores = sum(valid_results)
    avg_score = patch_scores / len(valid_results)
    top_k_indices = np.argsort(avg_score)[-top_k:][::-1]  # Get top k indices

    return top_k_indices


def select_top_k_patches(file_path, prefix, top_k_indices):
    try:
        patched_file = np.load(file_path)
        patches = patched_file['patches']
        positions = patched_file['positions']

        selected_patches = patches[top_k_indices]
        selected_positions = positions[top_k_indices]

        save_dir = os.path.dirname(file_path)
        file_name = os.path.basename(file_path)
        save_path = os.path.join(save_dir, f"{prefix}_{file_name}")
        np.savez_compressed(save_path, patches=selected_patches, positions=selected_positions)

        return True
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False


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