import os
import numpy as np
import pickle
from tqdm import tqdm
from scipy import sparse, ndimage
from pathlib import Path
from functools import partial
from multiprocessing import Pool
from sklearn.cluster import DBSCAN


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
    save_path = save_dir / file_name
    return save_path


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


def calculate_density_map(image, window_size, intensity_threshold=0):
    """
    Calculate a density map from a pseudo 2D ms image.
    The density at a pixel is the number of signal pixels (above intensity_threshold) within a window of 'window_size' centered at that pixel.

    :param image: Normalized pseudo MS image (2D array).
    :param window_size: Sliding window size in pixels.
    :param intensity_threshold: The threshold to consider a pixel as signal.
    :return: A 2D array representing the density map.
    """
    signal_mask = (image > 0).astype(np.float32) if intensity_threshold <= 0 else (image > intensity_threshold).astype(np.float32)
    # uniform_filter calculates the mean, so multiply by window area to get the sum (count)
    window_area = window_size * window_size
    density_map = ndimage.uniform_filter(signal_mask, size=window_size, mode='reflect') * window_area
    return density_map


def non_maximum_suppression(density_map, num_patches, suppression_window_size, min_density_threshold=1):
    """
    Performs Non-Maximum Suppression on a density map to find the top N densest patch centers.

    :param density_map: 2D array where each value is the density score.
    :param num_patches: The maximum number of patches (peaks) to find.
    :param suppression_window_size: A tuple (height, width) of the area to suppress around a found peak. This should typically be the patch size.
    :param min_density_threshold: The minimum density score for a peak to be considered.
    :return: A numpy array of shape (N, 2) containing the (row, col) coordinates of the peaks.
    """
    H, W = suppression_window_size
    h_half_floor = H // 2
    w_half_floor = W // 2
    h_half_ceil = H - h_half_floor
    w_half_ceil = W - w_half_floor

    temp_map = np.copy(density_map)
    peak_coords = []

    for _ in range(num_patches):
        max_val = np.max(temp_map)

        if max_val < min_density_threshold:
            break

        # find the coordinates of the peak
        coords = np.unravel_index(np.argmax(temp_map), temp_map.shape)
        peak_coords.append(coords)

        # suppress the area around the peak
        row_center, col_center = coords
        row_start = max(0, row_center - h_half_floor)
        row_end = min(temp_map.shape[0], row_center + h_half_ceil)
        col_start = max(0, col_center - w_half_floor)
        col_end = min(temp_map.shape[1], col_center + w_half_ceil)

        temp_map[row_start:row_end, col_start:col_end] = 0
    return np.array(peak_coords, dtype=int)


def get_peak_coords(binned_file_path, patch_params):
    """
    Extracts anchor points from a binned file based on the patch parameters.
    """
    try:
        if not os.path.exists(binned_file_path):
            raise FileNotFoundError(f"File not found: {binned_file_path}")

        sparse_matrix = sparse.load_npz(binned_file_path)
        raw_image = sparse_matrix.toarray()
        normalized_image = get_normalized_image(raw_image)
        density_map = calculate_density_map(
            image=normalized_image,
            window_size=patch_params.get('window_size'),
            intensity_threshold=patch_params.get('intensity_threshold', 0)
        )
        peak_coords = non_maximum_suppression(
            density_map=density_map,
            num_patches=patch_params['num_patches'],
            suppression_window_size=(patch_params['patch_height'], patch_params['patch_width']),
            min_density_threshold=patch_params.get('min_density_threshold', 1)
        )

        return peak_coords
    except Exception as e:
        raise RuntimeError(f"Error processing {binned_file_path}: {e}")


def parallel_get_peak_coords(binned_file_paths, patch_params, workers=4):
    """
    Extract anchor points from multiple binned files in parallel.
    """
    print(f"Starting parallel anchor point extraction for {len(binned_file_paths)} files...")
    with Pool(processes=workers) as pool:
        worker = partial(
            get_peak_coords,
            patch_params=patch_params
        )

        results = list(
            tqdm(
                pool.imap_unordered(worker, binned_file_paths),
                total=len(binned_file_paths),
                desc='Calculating peak coordinates'
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


def generate_patches(binned_file_path, prefix, patch_strategy, patch_params, peak_coords=None):
    try:
        if not (os.path.exists(binned_file_path) and os.path.getsize(binned_file_path) > 0):
            raise FileNotFoundError(f"File not found or empty: {binned_file_path}")

        save_path = _create_save_path(binned_file_path, prefix)
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
        elif patch_strategy == 'dnms':
            if peak_coords is not None:
                patches, positions = point_patching(
                    image=normalized_image,
                    point_coords=peak_coords,
                    patch_height=patch_params.get('patch_height'),
                    patch_width=patch_params.get('patch_width'),
                    padding_value=patch_params.get('padding_value')
                )
            else:
                raise ValueError("Peak Coordinates must be provided for DNMS patching.")
        else:
            raise ValueError(f'Invalid patch strategy: {patch_strategy}. Choose either "grid" or "dnms".')

        np.savez_compressed(save_path, patches=patches, positions=positions)
        return True
    except Exception as e:
        raise RuntimeError(f"Error processing {binned_file_path}: {e}")


def parallel_generate_patches(binned_file_paths, prefix, patch_strategy, patch_params, peak_coords=None, workers=4):
    """
    Generate patches from pseudo MS images in parallel.

    :param binned_file_paths: List of file paths to the pseudo MS images.
    :param prefix: Prefix for the save path pattern.
    :param patch_strategy: Strategy to extract patches ('grid' or 'dnms').
    :param patch_params: Dictionary containing parameters for patch generation.
    :param peak_coords: Peak coordinates for point patching, if applicable.
    :param workers: Number of worker processes to use.
    """
    if patch_strategy not in ['grid', 'dnms']:
        raise ValueError('Invalid strategy. Choose either "grid" or "dnms".')

    print(f"Starting parallel patch generation for {len(binned_file_paths)} files using '{patch_strategy}' strategy...")
    with Pool(processes=workers) as pool:
        worker = partial(
            generate_patches,
            prefix=prefix,
            patch_strategy=patch_strategy,
            patch_params=patch_params,
            peak_coords=peak_coords,
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


def filter_coords_by_dbscan(coords, eps, min_samples=1):
    """
    Filter coordinates using DBSCAN clustering to remove noise.

    :param coords: Array of coordinates to filter.
    :param eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    :param min_samples: The number of samples in a neighborhood for a point to be considered as a core point.
    :return: Filtered coordinates after applying DBSCAN.
    """
    if coords.shape[0] == 0:
        raise ValueError('Empty array')

    db = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
    labels = db.labels_

    merged_coords = []
    for label in set(labels):
        points_in_cluster_mask = (labels == label)
        points_in_cluster = coords[points_in_cluster_mask]
        centroid = np.mean(points_in_cluster, axis=0)
        merged_coords.append(centroid)

    return np.array(merged_coords, dtype=int)


def process_patching(args, binned_dataset_dir, binned_file_paths):
    """
    Process the patching of binned MS files.
    """
    # Patching binned MS files
    print('Patching binned MS Files...')

    if args.patch_strategy == 'dnms':
        if args.generate_strategy == 'per_file':
            print('Using per-file patch generation strategy.')
            for binned_file_path in tqdm(binned_file_paths, desc="Generating Patches"):
                peak_coords = get_peak_coords(
                    binned_file_path=binned_file_path,
                    patch_params=args.patch_params
                )
                generate_patches(
                    binned_file_path=binned_file_path,
                    prefix=args.patch_prefix,
                    patch_strategy=args.patch_strategy,
                    patch_params=args.patch_params,
                    peak_coords=peak_coords,
                )
        elif args.generate_strategy == 'global':
            print('Using Density Map with Non-Maximum Suppression, creating global peak coordinates for patching.')
            global_peak_coords_path = os.path.join(
                binned_dataset_dir,
                f"{args.patch_strategy}_intensity_thr_{args.patch_params.get('intensity_threshold')}_min_density_thr_{args.patch_params.get('min_density_threshold')}_"
                f"min_peak_dist_{args.patch_params.get('min_peak_distance')}_global_peak_coords.pkl"
            )

            if os.path.exists(global_peak_coords_path):
                print(f"Loading existing global peak coords from {global_peak_coords_path}")
                with open(global_peak_coords_path, 'rb') as f:
                    global_peak_coords = pickle.load(f)
                print(f"Total unique global peak_coords loaded: {len(global_peak_coords)}")
            else:
                print("Calculating global peak coords for patching...")
                peak_coords_lists = parallel_get_peak_coords(
                    binned_file_paths=binned_file_paths,
                    patch_params=args.patch_params,
                    workers=args.num_workers
                )

                print("Aggregating peak coords...")
                aggregate_peak_coords = np.vstack(peak_coords_lists)
                global_peak_coords = filter_coords_by_dbscan(
                    coords=aggregate_peak_coords,
                    eps=args.patch_params.get('min_peak_distance', 20),
                    min_samples=args.patch_params.get('min_samples', 1)
                )
                print(f"Total unique global peak_coords after DBSCAN filtering: {len(global_peak_coords)}")

                print(f"Saving global peak_coords to {global_peak_coords_path}")
                with open(global_peak_coords_path, 'wb') as f:
                    pickle.dump(global_peak_coords, f)

                print(f"Total unique global peak_coords: {len(global_peak_coords)}")

            parallel_generate_patches(
                binned_file_paths=binned_file_paths,
                prefix=args.patch_prefix,
                patch_strategy=args.patch_strategy,
                patch_params=args.patch_params,
                peak_coords=global_peak_coords,
                workers=args.num_workers
            )
        else:
            raise ValueError(f"Unknown patch generation strategy '{args.generate_strategy}' for DNMS patching.")
    elif args.patch_strategy == 'grid':
        parallel_generate_patches(
            binned_file_paths=binned_file_paths,
            prefix=args.patch_prefix,
            patch_strategy=args.patch_strategy,
            patch_params=args.patch_params,
            peak_coords=None,
            workers=args.num_workers
        )
    else:
        raise ValueError(f"Unknown patch strategy '{args.patch_strategy}'")
    print('Patching Process Completed.')
