import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from functools import partial
from multiprocessing import Pool
from scipy.spatial import cKDTree


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


def get_normalized_matrix(raw_matrix):
    """
    Normalize the matrix to the range [0, 1].

    :param raw_matrix: The raw matrix (2D array).
    :return: Normalized matrix (2D array).
    """
    non_zero_elements = raw_matrix[raw_matrix > 0]
    if non_zero_elements.size == 0:
        raise ValueError("No non-zero elements found in the matrix.")

    v_max = np.percentile(non_zero_elements, 99.9)
    v_min = 0
    if v_max <= v_min:
        raise ValueError(f"Invalid value range: v_min={v_min}, v_max={v_max}")

    normalized_matrix = (raw_matrix - v_min) / (v_max - v_min)
    normalized_matrix = np.clip(normalized_matrix, 0, 1)
    return normalized_matrix


def get_peaks(peak_list_file_paths):
    peaks_df = []
    for file_path in tqdm(peak_list_file_paths):
        if not (os.path.exists(file_path) and os.path.isfile(file_path)):
            raise ValueError(f"File {file_path} does not exist or is not a file.")

        df = pd.read_csv(file_path)
        mz_col = [col for col in df.columns if 'm/z' in col][0]
        rt_col = [col for col in df.columns if 'RT' in col][0]
        temp_df = df[[mz_col, rt_col]].copy()
        temp_df.columns = ['m/z', 'RT']
        peaks_df.append(temp_df)
    peaks_df = pd.concat(peaks_df, ignore_index=True)
    return peaks_df


def greedy_density_peaks_selection(peaks, patch_height, patch_width, n_candidates=1000, min_peaks_in_patch=10):
    """

    :param peaks: (N, 2) array of ms peak (m/z, rt) coordinates.
    :param patch_height: The height of the patch on the RT axis.
    :param patch_width: The width of the patch on the m/z axis.
    :param n_candidates: The number of candidate center peaks to evaluate in each iteration.
    :param min_peaks_in_patch: Minimum number of peaks required to extract patch.
    """
    kdtree = cKDTree(peaks)
    available_mask = np.ones(peaks.shape[0], dtype=bool)
    selected_peaks = []

    iteration = 0
    while np.sum(available_mask) > min_peaks_in_patch:
        iteration += 1

        available_indices = np.where(available_mask)[0]
        if len(available_indices) > n_candidates:
            candidate_indices = np.random.choice(available_indices, n_candidates, replace=False)
        else:
            candidate_indices = available_indices

        best_candidate_index = -1
        max_score = -1

        for idx in candidate_indices:
            peak = peaks[idx]
            radius = np.sqrt((patch_height / 2) ** 2 + (patch_width / 2) ** 2)
            indices_in_ball = kdtree.query_ball_point(peak, r=radius)

            score = np.sum(available_mask[indices_in_ball])
            if score > max_score:
                max_score = score
                best_candidate_index = idx

        if max_score < min_peaks_in_patch:
            break

        selected_peak = peaks[best_candidate_index]
        selected_peaks.append(selected_peak)

        final_indices_in_ball = kdtree.query_ball_point(selected_peak, r=radius)
        available_mask[final_indices_in_ball] = False
    return np.array(selected_peaks, dtype=int)


def get_patch_points(peaks, mz_min, bin_size, rt_list):
    """
    Convert peaks (m/z, rt) to matrix indices based on the given m/z and rt parameters.

    :param peaks: Array of point coordinates as (num_peaks, 2) where each row is (m/z, rt).
    :param mz_min: Minimum m/z value.
    :param bin_size: Size of the m/z bin.
    :param rt_list: List of retention times corresponding to the rows in the matrix.
    :return: Array of matrix indices as (num_peaks, 2) where each row is (row_idx, col_idx).
    """
    mzs = peaks[:, 0]
    col_indices = np.floor((mzs - mz_min) / bin_size).astype(int)
    rt_kdtree = cKDTree(rt_list.reshape(-1, 1))
    rts = peaks[:, 1]
    _, row_indices = rt_kdtree.query(rts.reshape(-1, 1), k=1)
    patch_points = np.stack((row_indices, col_indices), axis=1)
    return patch_points


def grid_patching(matrix, patch_height, patch_width, overlap_row, overlap_col):
    """
    Extracts fixed-size patches from a matrix using a grid-based approach.

    :param matrix: The matrix to extract patches from (2D array).
    :param patch_width: The width of the patches to be extracted.
    :param patch_height: The height of the patches to be extracted.
    :param overlap_row: The number of overlapping pixels between patches in the row direction.
    :param overlap_col: The number of overlapping pixels between patches in the column direction.
    :return: patches (np.ndarray): Extracted patches of shape (num_patches, patch_height, patch_width).
    :return: positions (np.ndarray): Corresponding positions of patches as (num_patches, 2) where each row is (row_idx, col_idx).
    """
    assert len(matrix.shape) == 2, "Matrix should be a 2D array"
    H, W = matrix.shape

    step_h = patch_height - overlap_row
    step_w = patch_width - overlap_col

    patches = []
    positions = []

    for y in range(0, H - patch_height + 1, step_h):
        for x in range(0, W - patch_width + 1, step_w):
            patch = matrix[y:y + patch_height, x:x + patch_width]
            patches.append(patch)
            positions.append((y, x))

    return np.array(patches), np.array(positions)


def point_patching(matrix, patch_points, patch_height, patch_width, padding_value):
    """
    Extracts fixed-size patches centered around detected point coordinates.

    :param matrix: The matrix to extract patches from (2D array).
    :param patch_points: Coordinates of patch center point as (num_peaks, 2).
    :param patch_height: The height of the patches to be extracted.
    :param patch_width: The width of the patches to be extracted.
    :param padding_value: Value to use for padding if patch goes out of image bounds.
    :return patches (np.ndarray): Extracted patches of shape (num_peaks, patch_height, patch_width).
    :return positions (np.ndarray): Corresponding *center* positions (peak coordinates) of patches as (num_peaks, 2) where each row is (peak_row_idx, peak_col_idx).
    """
    assert len(matrix.shape) == 2, "Matrix should be a 2D array"
    H, W = matrix.shape
    patches = []
    positions = []  # Store the patch center point coordinates

    if patch_points.shape[0] == 0:
        raise ValueError("No patch points provided.")

    h_half_floor = patch_height // 2
    w_half_floor = patch_width // 2
    # Use ceil for end calculation if needed, adjust for 0-based index and slice exclusivity
    h_half_ceil = patch_height - h_half_floor
    w_half_ceil = patch_width - w_half_floor

    for row_index, col_index in patch_points:
        # Calculate patch boundaries centered at the patch center point
        row_start = row_index - h_half_floor
        row_end = row_index + h_half_ceil
        col_start = col_index - w_half_floor
        col_end = col_index + w_half_ceil

        # Create an empty patch with padding value
        patch = np.full((patch_height, patch_width), padding_value, dtype=matrix.dtype)

        # Determine the valid intersection range in the matrix
        matrix_row_valid_start = max(0, row_start)
        matrix_row_valid_end = min(H, row_end)
        matrix_col_valid_start = max(0, col_start)
        matrix_col_valid_end = min(W, col_end)

        # Determine where to paste the valid data in the patch
        patch_row_start = matrix_row_valid_start - row_start
        patch_row_end = matrix_row_valid_end - row_start
        patch_col_start = matrix_col_valid_start - col_start
        patch_col_end = matrix_col_valid_end - col_start

        # Copy the valid data if there is an intersection
        if matrix_row_valid_start < matrix_row_valid_end and matrix_col_valid_start < matrix_col_valid_end:
            patch[patch_row_start:patch_row_end, patch_col_start:patch_col_end] = \
                matrix[matrix_row_valid_start:matrix_row_valid_end, matrix_col_valid_start:matrix_col_valid_end]

            patches.append(patch)
            positions.append((row_index, col_index))  # Store the patch center point coordinates

    return np.array(patches), np.array(positions)


def generate_patches(binned_file_path, prefix, patch_strategy, patch_params, dataset_params, peaks=None):
    try:
        if not (os.path.exists(binned_file_path) and os.path.getsize(binned_file_path) > 0):
            raise FileNotFoundError(f"File not found or empty: {binned_file_path}")

        save_path = _create_save_path(binned_file_path, prefix)
        if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
            return True

        binned_data = np.load(binned_file_path, allow_pickle=True)
        sparse_ms_matrix = binned_data['sparse_ms_matrix'].item()
        ms_matrix = sparse_ms_matrix.toarray()
        normalized_ms_matrix = get_normalized_matrix(ms_matrix)
        rt_list = binned_data['rt_list']

        if patch_strategy == 'GP':
            patches, positions = grid_patching(
                matrix=normalized_ms_matrix,
                patch_height=patch_params.get('patch_height'),
                patch_width=patch_params.get('patch_width'),
                overlap_row=patch_params.get('overlap_row'),
                overlap_col=patch_params.get('overlap_col')
            )
        elif patch_strategy == 'DAPS':
            if peaks is not None:
                patch_points = get_patch_points(
                    peaks=peaks,
                    mz_min=dataset_params.get('mz_min'),
                    bin_size=dataset_params.get('bin_size'),
                    rt_list=rt_list
                )
                patches, positions = point_patching(
                    matrix=normalized_ms_matrix,
                    patch_points=patch_points,
                    patch_height=patch_params.get('patch_height'),
                    patch_width=patch_params.get('patch_width'),
                    padding_value=patch_params.get('padding_value')
                )
            else:
                raise ValueError("Peaks must be provided for point patching.")
        else:
            raise ValueError(f'Invalid patch strategy: {patch_strategy}.')

        np.savez_compressed(save_path, patches=patches, positions=positions)
        return True
    except Exception as e:
        raise RuntimeError(f"Error processing {binned_file_path}: {e}")


def parallel_generate_patches(binned_file_paths, prefix, patch_strategy, patch_params, dataset_params, peaks=None, workers=4):
    """
    Generate patches from pseudo MS images in parallel.

    :param binned_file_paths: List of file paths to the pseudo MS images.
    :param prefix: Prefix for the save path pattern.
    :param patch_strategy: Strategy to extract patches.
    :param patch_params: Dictionary containing parameters for patch generation.
    :param dataset_params: Dictionary containing dataset parameters (e.g., mz_min, bin_size, rt_scan).
    :param peaks: Peaks for point patching, if applicable.
    :param workers: Number of worker processes to use.
    """
    print(f"Starting parallel patch generation for {len(binned_file_paths)} files using '{patch_strategy}' strategy...")
    with Pool(processes=workers) as pool:
        worker = partial(
            generate_patches,
            prefix=prefix,
            patch_strategy=patch_strategy,
            patch_params=patch_params,
            dataset_params=dataset_params,
            peaks=peaks
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


def process_patching(args, peak_list_file_paths, binned_dataset_dir, binned_file_paths):
    """
    Process the patching of binned MS files.
    """
    # Patching binned MS files
    print('Patching binned MS Files...')
    if args.patch_strategy == 'GP':
        parallel_generate_patches(
            binned_file_paths=binned_file_paths,
            prefix=args.patch_prefix,
            patch_strategy=args.patch_strategy,
            patch_params=args.patch_params,
            dataset_params=args.dataset_params,
            peaks=None,
            workers=args.num_workers
        )
    elif args.patch_strategy == 'DAPS':
        selected_peaks_file_path = os.path.join(
            binned_dataset_dir,
            f"PATCH_{args.patch_params.get('patch_height')}x{args.patch_params.get('patch_width')}_MIN_PKS_{args.patch_params.get('min_peaks_in_patch')}_"
            f"RT_SCAN_{args.dataset_params.get('rt_scan')}_BIN_SIZE_{args.dataset_params.get('bin_size')}.pkl"
        )

        if os.path.exists(selected_peaks_file_path):
            print(f"Loading existing selected peaks from {selected_peaks_file_path}")
            with open(selected_peaks_file_path, 'rb') as f:
                selected_peaks = pickle.load(f)
            print(f"Total selected peaks loaded: {len(selected_peaks)}")
        else:
            print("Calculating selected peaks for patching...")
            peaks = get_peaks(
                peak_list_file_paths=peak_list_file_paths,
            )
            selected_peaks = greedy_density_peaks_selection(
                peaks=peaks.to_numpy(),
                patch_height=args.patch_params.get('patch_height') * args.dataset_params.get('rt_scan'),
                patch_width=args.patch_params.get('patch_width') * args.dataset_params.get('bin_size'),
                n_candidates=args.patch_params.get('n_candidates', 1000),
                min_peaks_in_patch=args.patch_params.get('min_peaks_in_patch', 10)
            )
            print(f"Saving selected peaks to {selected_peaks_file_path}")
            with open(selected_peaks_file_path, 'wb') as f:
                pickle.dump(selected_peaks, f)
            print(f"Total selected peaks saved: {len(selected_peaks)}")

        parallel_generate_patches(
            binned_file_paths=binned_file_paths,
            prefix=args.patch_prefix,
            patch_strategy=args.patch_strategy,
            patch_params=args.patch_params,
            dataset_params=args.dataset_params,
            peaks=selected_peaks,
            workers=args.num_workers
        )
    else:
        raise ValueError(f"Unknown patch strategy '{args.patch_strategy}'")
    print('Patching Process Completed.')
