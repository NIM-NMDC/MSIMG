import os
import seaborn as sns
import matplotlib.pyplot as plt

from datetime import datetime

from utils.metrics import calculate_confusion_matrix, calculate_roc_auc


def plot_metrics(save_dir, plot_title, metrics, titles):
    n = len(metrics)
    plt.figure(figsize=(15, 5 * (n // 2 + n % 2)))
    for i, (train_metrics, valid_metrics) in enumerate(metrics):
        plt.subplot(n // 2 + n % 2, 2, i + 1)
        plt.plot(train_metrics, label='Train')
        plt.plot(valid_metrics, label='Valid')
        plt.title(titles[i])
        plt.xlabel('Epoch')
        plt.ylabel(titles[i])
        plt.legend()
    plt.suptitle(plot_title.split('.')[0])
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    time_stamp = datetime.now().strftime('%Y%m%d%H%M%S')
    metrics_file_path = os.path.join(save_dir, f"{plot_title}_train_valid_metrics_plot_{time_stamp}.png")
    plt.savefig(metrics_file_path)
    plt.show()
    plt.close()


def plot_confusion_matrix(targets, predicts, label_mapping, plot_title, save_dir, cm=None):
    if cm is None:
        cm, class_labels = calculate_confusion_matrix(targets, predicts, label_mapping)
    else:
        class_labels = [label for label, _ in sorted(label_mapping.items(), key=lambda item: item[1])]

    # visualize confusion matrix
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted')
    plt.ylabel('Target')
    plt.title(f"{plot_title.split('.')[0]} Confusion Matrix")

    time_stamp = datetime.now().strftime('%Y%m%d%H%M%S')
    cm_file_path = os.path.join(save_dir, f"{plot_title.split('.')[0]}_confusion_matrix_plot_{time_stamp}.png")
    plt.savefig(cm_file_path)
    plt.show()
    plt.close()


def plot_roc_auc_curve(targets, predicts, label_mapping, plot_title, save_dir):
    """
    Plot ROC curve

    Args:
        fpr: false positive rate
        tpr: true positive rate
        auc_scores: area under the curve
        num_classes: number of classes

    Returns:
        None
    """
    num_classes = len(label_mapping)
    fpr, tpr, auc_scores = calculate_roc_auc(targets, predicts, num_classes)
    class_labels = [label for label, _ in sorted(label_mapping.items(), key=lambda item: item[1])]

    plt.figure(figsize=(8, 6))
    for i, class_label in enumerate(class_labels):
        plt.plot(fpr[i], tpr[i], label=f'{class_label} (AUC = {auc_scores[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', linewidth=0.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f"{plot_title.split('.')[0]} ROC Curve")
    plt.legend(loc='lower right')

    time_stamp = datetime.now().strftime('%Y%m%d%H%M%S')
    roc_auc_file_path = os.path.join(save_dir, f"{plot_title}_roc_auc_plot_{time_stamp}.png")
    plt.savefig(roc_auc_file_path)
    plt.show()
    plt.close()