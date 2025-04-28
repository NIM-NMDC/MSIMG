import os
import numpy as np
from tqdm import tqdm
from scipy import sparse
from functools import partial
from multiprocessing import Pool
from skimage.feature import peak_local_max
from skimage.filters import gaussian


def extract_grid_patches(image, patch_width=224, patch_height=224, overlap_col=0, overlap_row=0):
    """
    Extracts fixed-size patches from a pseudo MS image using a grid-based approach.

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


def detect_peaks(image, max_peaks=2048, min_distance=10, intensity_threshold=0.1, smoothing_sigma=None):
    """
    Detects peaks (local maxima) in the pseudo MS image.

    :param image: The pseudo MS image to detect peaks in (2D array, normalized).
    :param max_peaks: The maximum number of peaks to detect.
    :param min_distance: The minimum distance (pixels) between peaks.
    :param intensity_threshold: The minimum intensity for a peak.
    :param smoothing_sigma: Standard deviation for Gaussian kernel for optional smoothing. Set to None or 0 to disable smoothing.
    :return: np.ndarray: Coordinates of detected peaks as (num_peaks, 2) where each row is (row_idx, col_idx).
    """
    assert len(image.shape) == 2, "Image should be a 2D array"

    if smoothing_sigma is not None and smoothing_sigma > 0:
        image_processed = gaussian(image, sigma=smoothing_sigma, preserve_range=True)
    else:
        image_processed = image

    # Find coordinates of local maxima
    # coordinates is an array of shape (N, 2) where N is the number of peaks
    coordinates = peak_local_max(
        image_processed,
        min_distance=min_distance,
        threshold_abs=intensity_threshold,
    )

    coordinates = coordinates[:max_peaks]  # Limit to max_peaks

    return coordinates


def extract_pcp_patches(image, peak_coords, patch_width=224, patch_height=224, padding_value=0.0):
    """
    Extracts fixed-size patches centered around detected peak coordinates using PCP (Peak-Centric Patching) Algorithm.

    :param image: The pseudo MS image to extract patches from (2D array, normalized).
    :param peak_coords: Coordinates of detected peaks as (num_peaks, 2) from detect_peaks function.
    :param patch_width: The width of the patches to be extracted.
    :param patch_height: The height of the patches to be extracted.
    :param padding_value: Value to use for padding if patch goes out of image bounds.
    :return patches (np.ndarray): Extracted patches of shape (num_peaks, patch_height, patch_width).
    :return positions (np.ndarray): Corresponding *center* positions (peak coordinates) of patches as (num_peaks, 2) where each row is (peak_row_idx, peak_col_idx).
    """
    assert len(image.shape) == 2, "Image should be a 2D array"
    H, W = image.shape
    patches = []
    positions = []  # Store the *center* coordinates (peak coordinates)

    if peak_coords.shape[0] == 0:
        raise ValueError("No peaks detected in the image.")

    h_half_floor = patch_height // 2
    w_half_floor = patch_width // 2
    # Use ceil for end calculation if needed, adjust for 0-based index and slice exclusivity
    h_half_ceil = patch_height - h_half_floor
    w_half_ceil = patch_width - w_half_floor

    for mz_idx, scan_idx in peak_coords:
        # Calculate patch boundaries centered at the peak
        mz_start = mz_idx - h_half_floor
        mz_end = mz_idx + h_half_ceil
        scan_start = scan_idx - w_half_floor
        scan_end = scan_idx + w_half_ceil

        # Create an empty patch with padding value
        patch = np.full((patch_height, patch_width), padding_value, dtype=image.dtype)

        # Determine the valid intersection range in the original image
        mz_valid_start = max(0, mz_start)
        mz_valid_end = min(H, mz_end)
        scan_valid_start = max(0, scan_start)
        scan_valid_end = min(W, scan_end)

        # Determine where to paste the valid data in the patch
        paste_mz_start = mz_valid_start - mz_start
        paste_mz_end = mz_valid_end - mz_start
        paste_scan_start = scan_valid_start - scan_start
        paste_scan_end = scan_valid_end - scan_start

        # Copy the valid data if there is an intersection
        if mz_valid_start < mz_valid_end and scan_valid_start < scan_valid_end:
            patch[paste_mz_start:paste_mz_end, paste_scan_start:paste_scan_end] = \
                image[mz_valid_start:mz_valid_end, scan_valid_start:scan_valid_end]

        patches.append(patch)
        positions.append((mz_idx, scan_idx))

    return np.array(patches), np.array(positions)


def generate_patches(binned_file_path, prefix, patch_method, peak_detection_params, patch_params):
    try:
        if not (os.path.exists(binned_file_path) and os.path.getsize(binned_file_path) > 0):
            raise FileNotFoundError(f"File not found or empty: {binned_file_path}")

        # /dataset_dir/{bin_prefix}/{class_name}/{bin_prefix}_file_name.npz
        # /dataset_dir/{previous_prefix}/{class_name}/{bin_prefix}_file_name.npz
        previous_prefix = os.path.basename(os.path.dirname(os.path.dirname(binned_file_path)))
        class_name = os.path.basename(os.path.dirname(binned_file_path))
        file_name = os.path.basename(binned_file_path)

        dataset_dir = os.path.abspath(os.path.join(binned_file_path, '../../..'))
        save_dir = os.path.join(dataset_dir, f'{prefix}_{previous_prefix}', class_name)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'{prefix}_{file_name}')

        if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
            return True

        sparse_matrix = sparse.load_npz(binned_file_path)
        raw_image = sparse_matrix.toarray()
        raw_image = raw_image / raw_image.max()  # Normalize the image to [0, 1]

        if patch_method == 'pcp':
            # Detect peaks in the pseudo MS image
            peak_coords = detect_peaks(
                image=raw_image,
                max_peaks=peak_detection_params.get('max_peaks', 2048),
                min_distance=peak_detection_params.get('min_distance', 10),
                intensity_threshold=peak_detection_params.get('intensity_threshold', 0.1),
                smoothing_sigma=peak_detection_params.get('smoothing_sigma', 1)
            )

            patches, positions = extract_pcp_patches(
                image=raw_image,
                peak_coords=peak_coords,
                patch_width=patch_params.get('patch_width', 224),
                patch_height=patch_params.get('patch_height', 224),
                padding_value=patch_params.get('padding_value', 0.0)
            )

            np.savez_compressed(save_path, patches=patches, positions=positions)
        elif patch_method == 'grid':
            patches, positions = extract_grid_patches(
                image=raw_image,
                patch_width=patch_params.get('patch_width', 224),
                patch_height=patch_params.get('patch_height', 224),
                overlap_col=patch_params.get('overlap_col', 0),
                overlap_row=patch_params.get('overlap_row', 0)
            )

            np.savez_compressed(save_path, patches=patches, positions=positions)

        return True
    except Exception as e:
        raise RuntimeError(f"Error processing {binned_file_path}: {e}")


def parallel_generate_patches(
        binned_file_paths, prefix, patch_method,
        max_peaks=2048, min_distance=10, intensity_threshold=0.1, smoothing_sigma=None,
        patch_width=224, patch_height=224, overlap_col=0, overlap_row=0, padding_value=0.0, workers=4
):
    """
    Generate patches from pseudo MS images in parallel.

    :param binned_file_paths: List of file paths to the pseudo MS images.
    :param prefix: Prefix for the save path pattern.
    :param patch_method: Method to extract patches ('grid' or 'pcp').
    :param max_peaks: The maximum number of peaks to detect.
    :param min_distance: The minimum distance (pixels) between peaks.
    :param intensity_threshold: The minimum intensity for a peak.
    :param smoothing_sigma: Standard deviation for Gaussian kernel for optional smoothing. Set to None or 0 to disable smoothing.
    :param patch_width: The width of the patches to be extracted.
    :param patch_height: The height of the patches to be extracted.
    :param overlap_col: The number of overlapping pixels between patches in the column direction.
    :param overlap_row: The number of overlapping pixels between patches in the row direction.
    :param padding_value: Value to use for padding if patch goes out of image bounds.
    :param workers: Number of worker processes to use.
    """
    if patch_method not in ['grid', 'pcp']:
        raise ValueError('Invalid method. Choose either "grid" or "pcp".')

    peak_detection_params = {
        'max_peaks': max_peaks,
        'min_distance': min_distance,
        'intensity_threshold': intensity_threshold,
        'smoothing_sigma': smoothing_sigma
    }

    patch_params = {
        'patch_width': patch_width,
        'patch_height': patch_height,
        'overlap_col': overlap_col,
        'overlap_row': overlap_row,
        'padding_value': padding_value
    }

    with Pool(processes=workers) as pool:
        worker = partial(
            generate_patches,
            prefix=prefix,
            patch_method=patch_method,
            peak_detection_params=peak_detection_params,
            patch_params=patch_params
        )

        results = list(
            tqdm(
                pool.imap_unordered(worker, binned_file_paths),
                total=len(binned_file_paths),
                desc='Generating patches',
            )
        )

    success_rate = sum(results) / len(results)
    print(f"Patching completed. Success rate: {success_rate:.2%}")


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


def calculate_patches_scores(patched_file_path, method='entropy'):
    try:
        if not (os.path.exists(patched_file_path) and os.path.getsize(patched_file_path) > 0):
            raise FileNotFoundError(f"File not found or empty: {patched_file_path}")

        patched_file = np.load(patched_file_path)
        patches = patched_file['patches']

        if patches.size == 0:
            raise ValueError(f"No patches found in the file: {patched_file_path}")

        if method == 'entropy':
            scores = np.array([calculate_entropy(patch) for patch in patches])
        elif method == 'mean':
            scores = np.array([calculate_mean(patch) for patch in patches])
        else:
            raise ValueError('Invalid method. Choose either "entropy" or "mean".')

        return scores
    except Exception as e:
        raise RuntimeError(f"Error processing {patched_file_path}: {e}")


def parallel_calculate_patches_scores(patched_file_paths, method='entropy', workers=4):
    """
    Calculate scores for patches in parallel and return the full sorted indices.

    :param patched_file_paths: A list of file paths to the pseudo MS images.
    :param method: Method to calculate (e.g. Entropy: 1D image entropy, Mean: mean intensity).
    :param workers: Number of worker processes to use.
    :return: sorted_indices: Indices of the patches sorted by score.
    """
    if method not in ['entropy', 'mean']:
        raise ValueError('Invalid method. Choose either "entropy" or "mean".')

    with Pool(processes=workers) as pool:
        worker = partial(
            calculate_patches_scores,
            method=method
        )

        results = list(
            tqdm(
                pool.imap_unordered(worker, patched_file_paths),
                total=len(patched_file_paths),
                desc=f'Calculating patches {method} scores',
            )
        )

    valid_results = [result for result in results if len(result) > 0]

    if not valid_results:
        raise RuntimeError("No valid scores calculated.")

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

    # Get the indices sorted by score in descending order (highest score first)
    sorted_indices = np.argsort(avg_scores)[::-1]

    return sorted_indices


def get_patches_number(patched_file_path):
    try:
        if not (os.path.exists(patched_file_path) and os.path.getsize(patched_file_path) > 0):
            raise FileNotFoundError(f"File not found or empty: {patched_file_path}")

        patched_file = np.load(patched_file_path)
        patches = patched_file['patches']

        return len(patches)
    except Exception as e:
        raise RuntimeError(f"Error processing {patched_file_path}: {e}")


def parallel_get_patches_numbers(patched_file_paths, workers=4):
    """
    Get the number of patches in multiple patched files in parallel.

    :param patched_file_paths: A list of file paths to the patched files.
    :param workers: Number of worker processes to use.
    :return: List of numbers of patches for each file.
    """
    with Pool(processes=workers) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(get_patches_number, patched_file_paths),
                total=len(patched_file_paths),
                desc='Getting number of patches',
            )
        )

    return results


def select_top_k_patches(patched_file_path, prefix, top_k_indices):
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

        patched_file = np.load(patched_file_path)
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
        raise RuntimeError(f"Error processing {patched_file_path}: {e}")


def parallel_select_top_k_patches(patched_file_paths, prefix, top_k_indices, workers=4):
    """
    Select top K patches based on the provided indices and save them.

    :param patched_file_paths: A list of file paths to the pseudo MS images.
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
                pool.imap_unordered(worker, patched_file_paths),
                total=len(patched_file_paths),
                desc='Selecting top K patches',
            )
        )

    success_rate = sum(results) / len(results)
    print(f"Selection completed. Success rate: {success_rate:.2%}")