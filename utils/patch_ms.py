import os
import numpy as np
import itertools
from tqdm import tqdm
from scipy import sparse, ndimage
from functools import partial
from multiprocessing import Pool


def get_normalized_image(raw_image):
    """
    Normalize the raw image to the range [0, 1].

    :param raw_image: The raw pseudo MS image (2D array).
    :return: Normalized image (2D array).
    """
    non_zero_pixels = raw_image[raw_image > 0]
    if non_zero_pixels.size == 0:
        raise ValueError("No non-zero pixels found in the image.")

    v_max = np.percentile(non_zero_pixels, 99.9)
    v_min = 0
    if v_max <= v_min:
        raise ValueError(f"Invalid value range: v_min={v_min}, v_max={v_max}")

    normalized_image = (raw_image - v_min) / (v_max - v_min)
    normalized_image = np.clip(normalized_image, 0, 1)

    return normalized_image


def calculate_local_variance_map(image, window_size):
    """
    Use uniform filter to efficiently calculate local variance maps.
    """
    mean_sq = ndimage.uniform_filter(image ** 2, size=window_size, mode='reflect')
    mean_val = ndimage.uniform_filter(image, size=window_size, mode='reflect')
    sq_mean = mean_val ** 2
    variance_map = mean_sq - sq_mean
    variance_map = np.maximum(variance_map, 0)
    return variance_map


def calculate_signal_mask_map(image, window_size, intensity_threshold=0.01):
    """
    Calculate the signal mask map of the image.
    The occupancy map is a binary map where pixels above the intensity threshold are marked as occupied.
    """
    signal_mask = (image > intensity_threshold).astype(np.float32)
    return signal_mask
    # occupancy_map = ndimage.uniform_filter(signal_mask, size=window_size, mode='reflect')
    # return occupancy_map


def information_content_sampling(information_map, detection_window_height, detection_window_width, step_height, step_width, detection_metric_type='mean', metric_threshold=0.05):
    """
    Perform Information Content Sampling (ICS) on the information map to select patches.
    Using a sliding window on the information map to find the center point of the area where the information content is above the threshold.

    :param information_map: 2D array representing the information content of the image.
    :param detection_window_height: Height of the sampling window.
    :param detection_window_width: Width of the sampling window.
    :param step_height: Vertical step size for sliding the window.
    :param step_width: Horizontal step size for sliding the window.
    :param detection_metric_type: Metric to use for sampling ('mean', 'max', 'min').
    :param metric_threshold: Threshold value for the detection metric.
    :return: List of coordinates where patches are sampled.
    """
    assert detection_metric_type in ['mean', 'max', 'min'], "metric_type must be one of ['mean', 'max', 'min']"

    H, W = information_map.shape
    anchor_point_coords = []

    for row_start in range(0, H - detection_window_height + 1, step_height):
        for col_start in range(0, W - detection_window_width + 1, step_width):
            row_end = row_start + detection_window_height
            col_end = col_start + detection_window_width

            window_on_info_map = information_map[row_start:row_end, col_start:col_end]

            metric_value = 0
            if detection_metric_type == 'mean':
                metric_value = np.mean(window_on_info_map)
            elif detection_metric_type == 'max':
                metric_value = np.max(window_on_info_map)
            elif detection_metric_type == 'min':
                metric_value = np.min(window_on_info_map)
            else:
                raise ValueError(f"Unsupported metric type: {detection_metric_type}. Choose 'mean' or 'max'.")

            if metric_value >= metric_threshold:
                row_center = row_start + detection_window_height // 2
                col_center = col_start + detection_window_width // 2
                anchor_point_coords.append((row_center, col_center))

    return np.array(anchor_point_coords, dtype=int)


def get_anchor_points(binned_file_path, patch_params):
    """
    Extracts anchor points from a binned file based on the patch parameters.
    """
    try:
        if not os.path.exists(binned_file_path):
            raise FileNotFoundError(f"File not found: {binned_file_path}")

        sparse_matrix = sparse.load_npz(binned_file_path)
        raw_image = sparse_matrix.toarray()
        normalized_image = get_normalized_image(raw_image)

        information_metric = patch_params.get('information_metric')
        information_map_window_size = patch_params.get('information_map_window_size', 16)
        # print(f"Calculating information map using {information_metric} with window size {information_map_window_size}x{information_map_window_size}")
        information_map = None
        if information_metric == 'local_variance':
            information_map = calculate_local_variance_map(
                image=normalized_image,
                window_size=(information_map_window_size, information_map_window_size)
            )

            map_min, map_max = np.min(information_map), np.max(information_map)
            if map_max > map_min:
                information_map = (information_map - map_min) / (map_max - map_min)
            else:
                raise ValueError(f"Information map is constant in {binned_file_path}")
        elif information_metric == 'signal_mask':
            information_map = calculate_signal_mask_map(
                image=normalized_image,
                window_size=(information_map_window_size, information_map_window_size),
                intensity_threshold=patch_params.get('intensity_threshold', 0.01)
            )
        else:
            raise ValueError(f"Unsupported information metric: {information_metric}. Choose 'local_variance' or 'signal_occupancy'.")

        detection_window_height = patch_params.get('detection_window_height', 32)
        detection_window_width = patch_params.get('detection_window_width', 32)
        step_height = patch_params.get('step_height')
        step_width = patch_params.get('step_width')
        if step_height == 0 or step_height < 0:
            step_height = detection_window_height
        if step_width == 0 or step_width < 0:
            step_width = detection_window_width

        anchor_points = information_content_sampling(
            information_map=information_map,
            detection_window_height=detection_window_height,
            detection_window_width=detection_window_width,
            step_height=step_height,
            step_width=step_width,
            detection_metric_type=patch_params.get('detection_metric_type', 'mean'),
            metric_threshold=patch_params.get('metric_threshold', 0.05)
        )

        return anchor_points
    except Exception as e:
        raise RuntimeError(f"Error processing {binned_file_path}: {e}")


def parallel_get_anchor_points(binned_file_paths, patch_params, workers=4):
    """
    Extract anchor points from multiple binned files in parallel.
    """
    print(f"Starting parallel anchor point extraction for {len(binned_file_paths)} files...")
    with Pool(processes=workers) as pool:
        worker = partial(
            get_anchor_points,
            patch_params=patch_params
        )

        results = list(
            tqdm(
                pool.imap_unordered(worker, binned_file_paths),
                total=len(binned_file_paths),
                desc='Calculating Information Content Sampling (ICS) anchor points'
            )
        )

    return results


def grid_patching(image, patch_height, patch_width, overlap_row, overlap_col):
    """
    Extracts fixed-size patches from a pseudo MS image using a grid-based approach.

    :param image: The pseudo MS image to extract patches from (2D array).
    :param patch_width: The width of the patches to be extracted.
    :param patch_height: The height of the patches to be extracted.
    :param overlap_row: The number of overlapping pixels between patches in the row direction.
    :param overlap_col: The number of overlapping pixels between patches in the column direction.
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


def point_patching(image, point_coords, patch_height, patch_width, padding_value):
    """
    Extracts fixed-size patches centered around detected point coordinates.

    :param image: The pseudo MS image to extract patches from (2D array, normalized).
    :param point_coords: Coordinates of detected peaks as (num_peaks, 2) from detect_peaks function.
    :param patch_height: The height of the patches to be extracted.
    :param patch_width: The width of the patches to be extracted.
    :param padding_value: Value to use for padding if patch goes out of image bounds.
    :return patches (np.ndarray): Extracted patches of shape (num_peaks, patch_height, patch_width).
    :return positions (np.ndarray): Corresponding *center* positions (peak coordinates) of patches as (num_peaks, 2) where each row is (peak_row_idx, peak_col_idx).
    """
    assert len(image.shape) == 2, "Image should be a 2D array"
    H, W = image.shape
    patches = []
    positions = []  # Store the *center* coordinates (peak coordinates)

    if point_coords.shape[0] == 0:
        raise ValueError("No peaks detected in the image.")

    h_half_floor = patch_height // 2
    w_half_floor = patch_width // 2
    # Use ceil for end calculation if needed, adjust for 0-based index and slice exclusivity
    h_half_ceil = patch_height - h_half_floor
    w_half_ceil = patch_width - w_half_floor

    for row_center, col_center in point_coords:
        # Calculate patch boundaries centered at the peak
        row_start = row_center - h_half_floor
        row_end = row_center + h_half_ceil
        col_start = col_center - w_half_floor
        col_end = col_center + w_half_ceil

        # Create an empty patch with padding value
        patch = np.full((patch_height, patch_width), padding_value, dtype=image.dtype)

        # Determine the valid intersection range in the original image
        img_row_valid_start = max(0, row_start)
        img_row_valid_end = min(H, row_end)
        img_col_valid_start = max(0, col_start)
        img_col_valid_end = min(W, col_end)

        # Determine where to paste the valid data in the patch
        patch_row_start = img_row_valid_start - row_start
        patch_row_end = img_row_valid_end - row_start
        patch_col_start = img_col_valid_start - col_start
        patch_col_end = img_col_valid_end - col_start

        # Copy the valid data if there is an intersection
        if img_row_valid_start < img_row_valid_end and img_col_valid_start < img_col_valid_end:
            patch[patch_row_start:patch_row_end, patch_col_start:patch_col_end] = \
                image[img_row_valid_start:img_row_valid_end, img_col_valid_start:img_col_valid_end]

            patches.append(patch)
            positions.append((row_center, col_center))  # Store the center coordinates

    return np.array(patches), np.array(positions)


def generate_patches(binned_file_path, prefix, patch_strategy, patch_params, global_anchor_points=None):
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
        normalized_image = get_normalized_image(raw_image)

        if patch_strategy == 'grid':
            patches, positions = grid_patching(
                image=normalized_image,
                patch_height=patch_params.get('patch_height'),
                patch_width=patch_params.get('patch_width'),
                overlap_row=patch_params.get('overlap_row'),
                overlap_col=patch_params.get('overlap_col')
            )
        elif patch_strategy == 'ics':
            # ics_window_size = patch_params.get('ics_window_size', 16)
            #
            # information_map = calculate_local_variance_map(image=normalized_image, window_size=(ics_window_size, ics_window_size))
            # map_min, map_max = np.min(information_map), np.max(information_map)
            # if map_max > map_min:
            #     normalized_information_map = (information_map - map_min) / (map_max - map_min)
            # else:
            #     raise ValueError(f"Information map is constant in {binned_file_path}")
            #
            # detection_window_height = patch_params.get('detection_window_height', 32)
            # detection_window_width = patch_params.get('detection_window_width', 32)
            # step_height = patch_params.get('step_height')
            # step_width = patch_params.get('step_width')
            # if step_height == 0 or step_height < 0:
            #     step_height = detection_window_height
            # if step_width == 0 or step_width < 0:
            #     step_width = detection_window_width
            #
            # anchor_points = information_content_sampling(
            #     information_map=normalized_information_map,
            #     detection_window_height=detection_window_height,
            #     detection_window_width=detection_window_width,
            #     step_height=step_height,
            #     step_width=step_width,
            #     detection_metric_type=patch_params.get('detection_metric_type', 'mean'),
            #     threshold=patch_params.get('threshold', 0.3)
            # )

            if global_anchor_points is not None:
                patches, positions = point_patching(
                    image=normalized_image,
                    point_coords=global_anchor_points,
                    patch_height=patch_params.get('patch_height'),
                    patch_width=patch_params.get('patch_width'),
                    padding_value=patch_params.get('padding_value')
                )
            else:
                raise ValueError("Global anchor points must be provided for ICS patching.")
        else:
            raise ValueError(f'Invalid patch strategy: {patch_strategy}. Choose either "grid" or "pcp".')

        np.savez_compressed(save_path, patches=patches, positions=positions)
        return True
    except Exception as e:
        raise RuntimeError(f"Error processing {binned_file_path}: {e}")


def parallel_generate_patches(
        binned_file_paths, prefix, patch_strategy, patch_params, global_anchor_points=None, workers=4
):
    """
    Generate patches from pseudo MS images in parallel.

    :param binned_file_paths: List of file paths to the pseudo MS images.
    :param prefix: Prefix for the save path pattern.
    :param patch_strategy: Strategy to extract patches ('grid' or 'ics').
    :param patch_params: Dictionary containing parameters for patch generation.
    :param global_anchor_points: Global anchor points for ICS patching, if applicable.
    :param workers: Number of worker processes to use.
    """
    if patch_strategy not in ['grid', 'ics']:
        raise ValueError('Invalid strategy. Choose either "grid" or "ics".')

    print(f"Starting parallel patch generation for {len(binned_file_paths)} files using '{patch_strategy}' strategy...")
    with Pool(processes=workers) as pool:
        worker = partial(
            generate_patches,
            prefix=prefix,
            patch_strategy=patch_strategy,
            patch_params=patch_params,
            global_anchor_points=global_anchor_points
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
