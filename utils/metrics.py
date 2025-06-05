import os
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.utils import resample as bootstrap_resample
from scipy.stats import wilcoxon


def calculate_accuracy(y_true, y_pred):
    """
    Calculate accuracy.

    :param y_true: true labels
    :param y_pred: predicted labels
    """
    return accuracy_score(y_true=y_true, y_pred=y_pred)


def calculate_precision(y_true, y_pred, average='macro', zero_division=0):
    """
    Calculate precision.

    :param y_true: true labels
    :param y_pred: predicted labels
    """
    return precision_score(y_true=y_true, y_pred=y_pred, average=average, zero_division=zero_division)


def calculate_recall(y_true, y_pred, average='macro'):
    """
    Calculate recall.

    :param y_true: true labels
    :param y_pred: predicted labels
    """
    return recall_score(y_true=y_true, y_pred=y_pred, average=average)


def calculate_f1_score(y_true, y_pred, average='macro'):
    """
    Calculate f1 score.

    :param y_true: true labels
    :param y_pred: predicted labels
    """
    return f1_score(y_true=y_true, y_pred=y_pred, average=average)


def calculate_confusion_matrix(y_true, y_pred, label_mapping):
    """
    Calculate confusion matrix.

    :param y_true: true labels
    :param y_pred: predicted labels
    """
    class_labels = [label for label, _ in sorted(label_mapping.items(), key=lambda item: item[1])]
    targets_mapped = [class_labels[target] for target in y_true]
    predicts_mapped = [class_labels[predict] for predict in y_pred]
    cm = confusion_matrix(targets_mapped, predicts_mapped, labels=class_labels)
    return cm, class_labels


def calculate_specificity(cm, label_mapping):
    """
    Calculate specificity

    :param cm: confusion matrix
    :param label_mapping: mapping from label to class name
    """
    specificity = {}
    for i, label in enumerate(label_mapping):
        tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
        fp = cm[:, i].sum() - cm[i, i]
        fn = cm[i, :].sum() - cm[i, i]
        tp = cm[i, i]
        specificity[label] = round(tn / (tn + fp), 4)
    return specificity


def calculate_roc_auc(y_true, y_pred, num_classes):
    """
    Calculate ROC-AUC score

    :param y_true: true labels
    :param y_pred: predicted probabilities
    :param num_classes: number of classes
    """
    targets_one_hot = np.eye(num_classes)[y_true]

    fpr, tpr = {}, {}
    roc_auc = {}

    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(targets_one_hot[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    return fpr, tpr, roc_auc


def calculate_bootstrap_ci(samples, n_bootstraps=1000, alpha=0.05, random_seed=3407):
    """
    Calculate bootstrap confidence interval.
    """
    if len(samples) < 2:
        return np.nan, np.nan
    if random_seed:
        np.random.seed(random_seed)
    bootstrap_means = []
    for _ in range(n_bootstraps):
        sample = bootstrap_resample(samples)
        bootstrap_means.append(np.mean(sample))
    lower_bound = np.percentile(bootstrap_means, 100 * (alpha / 2.0))
    upper_bound = np.percentile(bootstrap_means, 100 * (1 - alpha / 2.0))
    return lower_bound, upper_bound


def calculate_wilcoxon_test_p_value(model_alpha_metrics_file, model_beta_metrics_file, metric_to_compare='Accuracy'):
    """
    Calculate the Wilcoxon signed-rank test p-value between two models
    based on their K-fold metrics from CSV files.
    Assume CSV files have compared fold-wise performance for the specified metric.
    """
    try:
        df_a = pd.read_csv(model_alpha_metrics_file)
        df_b = pd.read_csv(model_beta_metrics_file)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found: {e.filename}. Please check paths: '{model_alpha_metrics_file}', '{model_beta_metrics_file}'")

    values_a = df_a[metric_to_compare].dropna().values
    values_b = df_b[metric_to_compare].dropna().values

    if len(values_a) != len(values_b):
        raise ValueError(f"Arrays for metric '{metric_to_compare}' and '{metric_to_compare}' have mismatched lengths.")

    result = wilcoxon(values_a, values_b, alternative='two-sided', zero_method='wilcox', correction=False)
    return result.pvalue