import torch
import os
import json
import logging
import numpy as np
import pandas as pd

from collections import defaultdict
from tqdm import tqdm


logging.basicConfig(
  format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


def compute_forgetfulness(correctness_trend):
    """
    Given a epoch-wise trend of train predictions, compute frequency with which
    an example is forgotten, i.e. predicted incorrectly _after_ being predicted correctly.
    Based on: https://arxiv.org/abs/1812.05159
    """
    # Example is never predicted correctly, or learnt!
    if not any(correctness_trend):
        return 1000
    learnt = False  # Predicted correctly in the current epoch
    times_forgotten = 0
    for is_correct in correctness_trend:
        if (not learnt and not is_correct) or (learnt and is_correct):
            # Nothing changed.
            continue
        elif learnt and not is_correct:
            # Forgot after learning at some point!
            learnt = False
            times_forgotten += 1
        elif not learnt and is_correct:
            # Learnt!
            learnt = True
    return times_forgotten


def compute_correctness(trend):
    """
    Aggregate #times an example is predicted correctly during all training epochs.
    """
    return sum(trend)


def compute_training_dynamics_metrics(args, training_dynamics):
    """
    Given the training dynamics (logits for each training instance across epochs), compute metrics
    based on it, for data map coordinates.
    Computed metrics are: confidence, variability, correctness, forgetfulness, threshold_closeness---
    the last two being baselines from prior work
    (Example Forgetting: https://arxiv.org/abs/1812.05159 and Active Bias: https://arxiv.org/abs/1704.07433 respectively).
    Returns:
    - DataFrame with these metrics.
    - DataFrame with more typical training evaluation metrics, such as accuracy / loss.
    """
    confidence_ = {}
    variability_ = {}
    threshold_closeness_ = {}
    correctness_ = {}
    forgetfulness_ = {}

    variability_func = lambda conf: np.std(conf)
    if args.confidence_interval:
        variability_func = lambda conf: np.sqrt(np.var(conf) + np.var(conf) * np.var(conf) / (len(conf)-1))
    threshold_closeness_func = lambda conf: conf * (1 - conf)

    num_total_epochs = len(list(training_dynamics.values())[0]['logits'])
    if args.burn_out < num_total_epochs:
        logger.info(f'Computing training dynamics. Burning out at {args.burn_out} of {num_total_epochs}.')
    else:
        logger.info(f'Computing training dynamics across {num_total_epochs}.')
    logger.info('Metrics computed: confidence, variability, correctness, forgetfulness, threshold_closeness.')

    logits = {i: [] for i in range(num_total_epochs)}
    targets = {i: [] for i in range(num_total_epochs)}
    training_accuracy = defaultdict(float)

    for uuid in tqdm(training_dynamics):
        correctness_trend = []
        true_probs_trend = []

        record = training_dynamics[uuid]
        for i, epoch_logits in enumerate(record['logits']):
            probs = torch.nn.functional.softmax(torch.Tensor(epoch_logits), dim=-1)
            true_probs = float(probs[record['label']])
            true_probs_trend.append(true_probs)

            prediction = np.argmax(epoch_logits)
            is_correct = (prediction == record['label']).item()
            correctness_trend.append(is_correct)

            training_accuracy[i] += is_correct
            logits[i].append(epoch_logits)
            targets[i].append(record['label'])

        if args.burn_out < num_total_epochs:
            correctness_trend = correctness_trend[:args.burn_out]
            true_probs_trend = true_probs_trend[:args.burn_out]

        correctness_[uuid] = compute_correctness(correctness_trend)
        confidence_[uuid] = np.mean(true_probs_trend)
        variability_[uuid] = variability_func(true_probs_trend)
        forgetfulness_[uuid] = compute_forgetfulness(correctness_trend)
        threshold_closeness_[uuid] = threshold_closeness_func(confidence_[uuid])

    # Should not affect ranking, so ignoring.
    epsilon_var = np.mean(list(variability_.values()))

    column_names = ['uuid', 'index', 'threshold_closeness', 'confidence', 'variability', 'correctness', 'forgetfulness']
    df = pd.DataFrame([
        [uuid, i, threshold_closeness_[uuid], confidence_[uuid], variability_[uuid], correctness_[uuid], forgetfulness_[uuid]]
        for i, uuid in enumerate(correctness_)
    ], columns=column_names)

    return df


def consider_ascending_order(filtering_metric):
    """
    Determine if the metric values' sorting order to get the most `valuable` examples for training.
    """
    if filtering_metric == 'variability':
        return False
    elif filtering_metric == 'confidence':
        return True
    elif filtering_metric == 'threshold_closeness':
        return False
    elif filtering_metric == 'forgetfulness':
        return False
    elif filtering_metric == 'correctness':
        return True
    else:
        raise NotImplemented(f'Filtering based on {filtering_metric} not implemented!')


# def write_filtered_data(args, training_dynamics_metrics):
#     """
#     Filter data based on the given metric, and write it in TSV format to train classifier.
#     """
#     # First save the args for filtering, to keep track of which model was used for filtering.
#     argparse_dict = vars(args)
#     with open(os.path.join(args.filtering_output_dir, f'filtering_configs.json'), 'w') as file:
#         file.write(json.dumps(argparse_dict, indent=4, sort_keys=True) + '\n')
#
#     # Determine whether to sort data in ascending order or not, based on the metric.
#     is_ascending = consider_ascending_order(args.metric)
#     if args.worst:
#         is_ascending = not is_ascending
#
#     # Sort by selection
#     sorted_scores = training_dynamics_metrics.sort_values(by=[args.metric], ascending=is_ascending)
#
#     original_train_file = os.path.join(os.path.join(args.data_dir, args.task_name), f'train.tsv')

