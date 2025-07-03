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

        with np.load(patched_file_path) as patched_file_data:
            if 'patches' not in patched_file_data:
                raise ValueError(f"Missing 'patches' in {patched_file_path}. Please check the file format.")
            patches = patched_file_data['patches']

        if score_strategy == 'entropy':
            scores = np.array([calculate_entropy(patch) for patch in patches])
        elif score_strategy == 'mean':
            scores = np.array([calculate_mean(patch) for patch in patches])
        else:
            raise ValueError('Invalid strategy. Choose either "entropy" or "mean".')

        return scores
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

    scores_list = []
    print(f'Starting parallel calculation of {score_strategy} scores for {len(patched_file_paths)} files.')
    with Pool(processes=workers) as pool:
        worker = partial(
            calculate_patches_scores,
            score_strategy=score_strategy,
        )

        for scores in tqdm(
                pool.imap_unordered(worker, patched_file_paths),
                total=len(patched_file_paths),
                desc=f'Calculating patches {score_strategy} scores'
        ):
            if len(scores) > 0:
                scores_list.append(scores)

    success_rate = len(scores_list) / len(patched_file_paths)
    print(f"Score calculation completed. Success rate: {success_rate:.2%}")
    return scores_list


def calculate_average_scores(scores_list):
    """
    Calculate the average scores across all files and return the indices of the top K patches.

    :param scores_list: A list of arrays containing information scores for each file.
    """
    min_len = min(len(scores) for scores in scores_list)

    if min_len == 0:
        raise ValueError("Minimum patch length is zero after loading scores.")

    trimmed_scores = [scores[:min_len] for scores in scores_list]
    avg_scores = np.mean(trimmed_scores, axis=0)

    return avg_scores
