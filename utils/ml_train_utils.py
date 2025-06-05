import os
import joblib

from utils.metrics import calculate_accuracy, calculate_precision, calculate_recall, calculate_f1_score, calculate_confusion_matrix, calculate_specificity
from visualization.metrics_visualization import plot_confusion_matrix, plot_roc_auc_curve


def train_test_ml(model, train_set, test_set, label_mapping, exp_base_dir, exp_model_name, metrics_visualization=True):
    """
    Train a machine learning model and evaluate it on the test set.
    """
    X_train, y_train = train_set
    X_test, y_test = test_set

    model.fit(X_train, y_train)
    save_model_path = os.path.join(exp_base_dir, f'{exp_model_name}.pkl')
    joblib.dump(model, save_model_path)

    y_pred = model.predict(X_test)

    test_accuracy = calculate_accuracy(y_test, y_pred)
    test_precision = calculate_precision(y_test, y_pred)
    test_recall = calculate_recall(y_test, y_pred)
    test_f1 = calculate_f1_score(y_test, y_pred)
    print('==========================================================================================')
    print(f'Test Accuracy: {test_accuracy:.4f} | Test Precision: {test_precision:.4f} | '
          f'Test Recall (Sensitivity): {test_recall:.4f} | Test F1: {test_f1:.4f}')

    confusion_matrix, _ = calculate_confusion_matrix(y_test, y_pred, label_mapping)
    # print(f'Confusion Matrix: \n{matrix}')
    specificity = calculate_specificity(confusion_matrix, label_mapping)
    print(f'Specificity for each class: {specificity}')
    print('==========================================================================================')

    if metrics_visualization:
        plot_confusion_matrix(y_test, y_pred, label_mapping, exp_model_name, exp_base_dir, cm=confusion_matrix)
        plot_roc_auc_curve(y_test, model.predict_proba(X_test), label_mapping, exp_model_name, exp_base_dir)

    return test_accuracy, test_precision, test_recall, test_f1