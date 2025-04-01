import numpy as np

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc
)


def compute_accuracy(targets, predicts):
    """
    Compute accuracy

    Args:
        targets: true labels
        predicts: predicted labels

    Returns:
        accuracy score
    """
    # targets = targets
    # predicts = predicts.argmax(axis=-1)
    return accuracy_score(targets, predicts)


def compute_precision(targets, predicts, average='macro', zero_division=0):
    """
    Compute precision

    Args:
        targets: true labels
        predicts: predicted labels
        average: averaging strategy

    Returns:
        precision score
    """
    return precision_score(targets, predicts, average=average, zero_division=zero_division)


def compute_recall(targets, predicts, average='macro'):
    """
    Compute recall

    Args:
        targets: true labels
        predicts: predicted labels
        average: averaging strategy

    Returns:
        recall score
    """
    return recall_score(targets, predicts, average=average)


def compute_f1(targets, predicts, average='macro'):
    """
    Compute f1 score

    Args:
        targets: true labels
        predicts: predicted labels
        average: averaging strategy

    Returns:
        f1 score
    """
    return f1_score(targets, predicts, average=average)


def compute_confusion_matrix(targets, predicts, label_mapping):
    """
    Compute confusion matrix

    Args:
        targets: true labels
        predicts: predicted labels
        label_mapping: mapping from label to class name

    Returns:
        confusion matrix
    """
    class_labels = [label for label, _ in sorted(label_mapping.items(), key=lambda item: item[1])]
    targets_mapped = [class_labels[target] for target in targets]
    predicts_mapped = [class_labels[predict] for predict in predicts]
    cm = confusion_matrix(targets_mapped, predicts_mapped, labels=class_labels)
    return cm, class_labels


def compute_specificity(cm, label_mapping):
    """
    Compute specificity

    Args:
        cm: confusion matrix
        label_mapping: mapping from label to class name

    Returns:
        specificity
    """
    specificity = {}
    for i, label in enumerate(label_mapping):
        tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
        fp = cm[:, i].sum() - cm[i, i]
        fn = cm[i, :].sum() - cm[i, i]
        tp = cm[i, i]
        specificity[label] = round(tn / (tn + fp), 4)
    return specificity


def compute_roc_auc(targets, predicts, num_classes):
    """
    Compute ROC-AUC score

    Args:
        targets: true labels
        predicts: predicted probabilities
        num_classes: number of classes

    Returns:
        ROC-AUC score
    """
    targets_one_hot = np.eye(num_classes)[targets]

    fpr, tpr = {}, {}
    roc_auc = {}

    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(targets_one_hot[:, i], predicts[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    return fpr, tpr, roc_auc

def calculate_metrics_statistics(metrics_list):
    """
    Calculate statistics of metrics

    Args:
        metrics_list: [(accuracy, precision, recall, f1 score), (...), ...]

    Returns:

    """
    # metrics_array = np.array(metrics_list)
    #
    # mean_metrics = np.mean(metrics_array, axis=0)
    # std_metrics = np.std(metrics_array, axis=0)
    #
    # # generate mean ± std string
    # metrics_statistics = {
    #     'Accuracy': f"{mean_metrics[0]:.2f} ± {std_metrics[0]:.2f}",
    #     'Precision': f"{mean_metrics[1]:.2f} ± {std_metrics[1]:.2f}",
    #     'Recall': f"{mean_metrics[2]:.2f} ± {std_metrics[2]:.2f}",
    #     'F1 Score': f"{mean_metrics[3]:.2f} ± {std_metrics[3]:.2f}"
    # }

    accuracies = [metrics['Accuracy'] for metrics in metrics_list]
    precisions = [metrics['Precision'] for metrics in metrics_list]
    recalls = [metrics['Recall'] for metrics in metrics_list]
    f1_scores = [metrics['F1 Score'] for metrics in metrics_list]

    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)

    mean_precision = np.mean(precisions)
    std_precision = np.std(precisions)

    mean_recall = np.mean(recalls)
    std_recall = np.std(recalls)

    mean_f1_score = np.mean(f1_scores)
    std_f1_score = np.std(f1_scores)

    metrics_statistics = {
        'Accuracy': f"{mean_accuracy:.3f} ± {std_accuracy:.2f}",
        'Precision': f"{mean_precision:.3f} ± {std_precision:.2f}",
        'Recall': f"{mean_recall:.3f} ± {std_recall:.2f}",
        'F1 Score': f"{mean_f1_score:.3f} ± {std_f1_score:.2f}"
    }

    return metrics_statistics

