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


def detect_peaks(image, max_peaks=0, min_distance=10, intensity_threshold=0.1, smoothing_sigma=None):
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
    if max_peaks == 0:
        coordinates = peak_local_max(
            image_processed,
            min_distance=min_distance,
            threshold_abs=intensity_threshold,
        )
    else:
        coordinates = peak_local_max(
            image_processed,
            min_distance=min_distance,
            threshold_abs=intensity_threshold,
            num_peaks=max_peaks,
        )

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


def generate_patches(binned_file_path, prefix, patch_strategy, peak_detection_params, patch_params):
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

        if patch_strategy == 'pcp':
            # Detect peaks in the pseudo MS image
            peak_coords = detect_peaks(
                image=raw_image,
                max_peaks=peak_detection_params.get('max_peaks', 0),
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
        elif patch_strategy == 'grid':
            patches, positions = extract_grid_patches(
                image=raw_image,
                patch_width=patch_params.get('patch_width', 224),
                patch_height=patch_params.get('patch_height', 224),
                overlap_col=patch_params.get('overlap_col', 0),
                overlap_row=patch_params.get('overlap_row', 0)
            )
        else:
            raise ValueError(f'Invalid patch strategy: {patch_strategy}. Choose either "grid" or "pcp".')

        np.savez_compressed(save_path, patches=patches, positions=positions)
        return True
    except Exception as e:
        raise RuntimeError(f"Error processing {binned_file_path}: {e}")


def parallel_generate_patches(
        binned_file_paths, prefix, patch_strategy,
        max_peaks=0, min_distance=10, intensity_threshold=0.1, smoothing_sigma=1,
        patch_width=224, patch_height=224, overlap_col=0, overlap_row=0, padding_value=0.0, workers=4
):
    """
    Generate patches from pseudo MS images in parallel.

    :param binned_file_paths: List of file paths to the pseudo MS images.
    :param prefix: Prefix for the save path pattern.
    :param patch_strategy: Strategy to extract patches ('pcp' or 'grid').
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
    if patch_strategy not in ['pcp', 'grid']:
        raise ValueError('Invalid strategy. Choose either "pcp" or "grid".')

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

    print(f"Starting parallel patch generation for {len(binned_file_paths)} files using '{patch_strategy}' strategy...")
    with Pool(processes=workers) as pool:
        worker = partial(
            generate_patches,
            prefix=prefix,
            patch_strategy=patch_strategy,
            peak_detection_params=peak_detection_params,
            patch_params=patch_params
        )

        results = list(
            tqdm(
                pool.imap_unordered(worker, binned_file_paths),
                total=len(binned_file_paths),
                desc=f'Generating {patch_strategy} patches',
            )
        )

    success_rate = sum(results) / len(results)
    print(f"Patch generation completed. Success rate: {success_rate:.2%}")


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
