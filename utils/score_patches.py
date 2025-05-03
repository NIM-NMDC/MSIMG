import os
import numpy as np
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool


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


def calculate_patches_scores(patched_file_path, score_strategy='entropy'):
    try:
        if not (os.path.exists(patched_file_path) and os.path.getsize(patched_file_path) > 0):
            raise FileNotFoundError(f"File not found or empty: {patched_file_path}")

        try:
            with np.load(patched_file_path) as patched_file_data:
                existing_keys = list(patched_file_data.keys())
                if score_strategy in existing_keys and len(patched_file_data[score_strategy]) > 0:
                    return True
                if 'patches' not in existing_keys or 'positions' not in existing_keys:
                    raise ValueError(f"Invalid file format: {patched_file_path}")
                if len(patched_file_data['patches']) == 0 or len(patched_file_data['positions']) == 0:
                    raise ValueError(f"No patches or positions found in the file: {patched_file_path}")

                patches = patched_file_data['patches']
                _patched_file_data = dict(patched_file_data)
        except Exception as e:
            raise RuntimeError(f"Error loading file {patched_file_path}: {e}")

        if score_strategy == 'entropy':
            scores = np.array([calculate_entropy(patch) for patch in patches])
        elif score_strategy == 'mean':
            scores = np.array([calculate_mean(patch) for patch in patches])
        else:
            raise ValueError('Invalid strategy. Choose either "entropy" or "mean".')

        _patched_file_data[score_strategy] = scores

        # Use temp files and rename to increase atomicity and prevent interruptions from causing file corruption,
        # temp_save_path = patched_file_path + '.tmp'
        base, ext = os.path.split(patched_file_path)
        temp_save_path = os.path.join(os.path.dirname(base), f"{os.path.basename(base)}_temp{ext}")
        try:
            np.savez_compressed(temp_save_path, **_patched_file_data)
            os.replace(temp_save_path, patched_file_path)
        except Exception as save_e:
            os.remove(temp_save_path)
            raise RuntimeError(f"Error saving file {patched_file_path}: {save_e}")

        return True
    except Exception as e:
        raise RuntimeError(f"Error processing {patched_file_path}: {e}")


def parallel_calculate_patches_scores(patched_file_paths, score_strategy='entropy', workers=4):
    """
    Calculate scores for patches in parallel and return the full sorted indices.

    :param patched_file_paths: A list of file paths to the pseudo MS images.
    :param score_strategy: Strategy to calculate score (e.g. Entropy: 1D image entropy, Mean: mean intensity).
    :param workers: Number of worker processes to use.
    """
    if score_strategy not in ['entropy', 'mean']:
        raise ValueError('Invalid strategy. Choose either "entropy" or "mean".')

    print(f'Starting parallel calculation of {score_strategy} scores for {len(patched_file_paths)} files.')
    with Pool(processes=workers) as pool:
        worker = partial(
            calculate_patches_scores,
            score_strategy=score_strategy,
        )

        results = list(
            tqdm(
                pool.imap_unordered(worker, patched_file_paths),
                total=len(patched_file_paths),
                desc=f'Calculating patches {score_strategy} scores',
            )
        )

    success_rate = sum(results) / len(results)
    print(f"Score calculation completed. Success rate: {success_rate:.2%}")


def calculate_average_scores_and_indices(patched_file_paths, score_strategy):
    """
    Calculate the average scores across all files and return the indices of the top K patches.

    :param patched_file_paths: A list of file paths to the pseudo MS images.
    :param score_strategy: Strategy to calculate (e.g. Entropy: 1D image entropy, Mean: mean intensity).
    """
    scores_list = []
    for file_path in patched_file_paths:
        try:
            if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                with np.load(file_path) as patched_file_data:
                    if score_strategy in patched_file_data and len(patched_file_data[score_strategy]) > 0:
                        scores_list.append(patched_file_data[score_strategy])
                    else:
                        raise ValueError(f"Missing {score_strategy} scores in {file_path}. Please recalculate.")
            else:
                raise FileNotFoundError(f"File not found or empty: {file_path}")
        except Exception as e:
            raise RuntimeError(f"Error loading file {file_path}: {e}")

    """
    When processing mass spectrometry data, the resulting matrix has a shape of (mz_bins, scans),
    where mz_bins is fixed, but the number of scans may vary depending on how many spectra were collected in each file.
    To ensure comparability during patch scoring and selection, we normalize the number of scores across all files.
    Specifically, we truncate all score arrays to the same minimum length, retaining only the initial portion of patches for each file.
    Any extra patches in files with more scans are discarded, which does not compromise the fairness or consistency of the overall evaluation.
    """
    min_len = min(len(scores) for scores in scores_list)

    if min_len == 0:
        raise ValueError("Minimum patch length is zero after loading scores.")

    trimmed_scores = [scores[:min_len] for scores in scores_list]
    avg_scores = np.sum(trimmed_scores, axis=0) / len(trimmed_scores)

    # Get the indices sorted by score in descending order (highest score first)
    sorted_indices = np.argsort(avg_scores)[::-1]

    return sorted_indices