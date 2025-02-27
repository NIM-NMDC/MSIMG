import torch

import os
import numpy as np
import pandas as pd

from tqdm import tqdm
from datetime import datetime

from utils.metrics import compute_precision, compute_recall, compute_f1, compute_confusion_matrix, compute_specificity
from utils.metrics_visualization import plot_metrics, plot_confusion_matrix, plot_roc_auc_curve


def train(
        exp_dir, exp_model_name, model, train_loader, valid_loader, criterion, optimizers,
        schedulers, early_stopping, epochs, device, use_early_stopping=False, metrics_visualization=True
):
    train_losses = []
    train_accuracies = []
    valid_losses = []
    valid_accuracies = []

    min_loss = np.inf

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        total_train_correct = 0
        total_train_samples = 0

        epoch_digits = len(str(epochs))
        epoch_str = f'{epoch + 1}'.zfill(epoch_digits)
        train_progress_bar = tqdm(train_loader, desc=f'Epoch: {epoch_str} Training', leave=True)
        for X, y in train_progress_bar:
            X, y = X.to(device), y.to(device)

            for optimizer in optimizers:
                optimizer.zero_grad()

            logits = model(X)
            loss = criterion(logits, y)

            loss.backward()
            for optimizer in optimizers:
                optimizer.step()

            total_train_loss += (loss.item() * y.size(0))
            total_train_correct += (logits.argmax(dim=1) == y).sum().item()
            total_train_samples += y.size(0)

            train_progress_bar.set_postfix({
                'Train Loss': f'{(total_train_loss / total_train_samples):.4f}',
                'Accuracy': f'{(total_train_correct / total_train_samples):.4f}'
            })

        model.eval()
        total_valid_loss = 0
        total_valid_correct = 0
        total_valid_samples = 0
        valid_progress_bar = tqdm(valid_loader, desc=f'Epoch: {epoch_str} Evaluating', leave=True)
        with torch.no_grad():
            for X, y in valid_progress_bar:
                X, y = X.to(device), y.to(device)

                logits = model(X)
                loss = criterion(logits, y)

                total_valid_loss += (loss.item() * y.size(0))
                total_valid_correct += (logits.argmax(dim=1) == y).sum().item()
                total_valid_samples += y.size(0)

                valid_progress_bar.set_postfix({
                    'Valid Loss': f'{(total_valid_loss / total_valid_samples):.4f}',
                    'Accuracy': f'{(total_valid_correct / total_valid_samples):.4f}'
                })

        epoch_train_loss = total_train_loss / total_train_samples
        epoch_train_accuracy = total_train_correct / total_train_samples
        epoch_valid_loss = total_valid_loss / total_valid_samples
        epoch_valid_accuracy = total_valid_correct / total_valid_samples

        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_accuracy)
        valid_losses.append(epoch_valid_loss)
        valid_accuracies.append(epoch_valid_accuracy)
        print('Epoch: {} | Train Loss: {:.4f} | Train Acc: {:.4f} | Valid Loss: {:.4f} | Valid Acc: {:.4f}'.format(
            epoch_str, epoch_train_loss, epoch_train_accuracy, epoch_valid_loss, epoch_valid_accuracy
        ))

        # scheduler.step(valid_loss)
        for scheduler in schedulers:
            scheduler.step(epoch_valid_loss)

        if use_early_stopping:
            early_stopping(epoch_valid_loss, model, save_model_dir=exp_dir, save_model_name=exp_model_name)
            if early_stopping.early_stop:
                print('Early Stopping.')

                if metrics_visualization:
                    metrics = [
                        (train_losses, valid_losses),
                        (train_accuracies, valid_accuracies)
                    ]
                    plot_metrics(exp_dir, exp_model_name, metrics, ['Loss', 'Accuracy'])

                break

        else:
            save_model_path = os.path.join(exp_dir, f'{exp_model_name}.pth')

            if epoch_valid_loss < min_loss:
                min_loss = epoch_valid_loss
                torch.save(model.state_dict(), str(save_model_path))
                print(f"Model saved at epoch {epoch + 1}, validation loss: {min_loss:.4f}")

    # save train and valid accuracy
    metrics_df = pd.DataFrame({
        'Epoch': list(range(1, len(train_accuracies) + 1)),
        'Train Loss': train_losses,
        'Valid Loss': valid_losses,
        'Train Accuracy': train_accuracies,
        'Valid Accuracy': valid_accuracies
    })
    time_stamp = datetime.now().strftime('%Y%m%d%H%M%S')
    metrics_df.to_csv(os.path.join(exp_dir, f'{exp_model_name}_metrics_{time_stamp}.csv'), index=False)
    print(f'Metrics saved to {exp_dir}')

    if metrics_visualization:
        metrics = [
            (train_losses, valid_losses),
            (train_accuracies, valid_accuracies),
        ]
        plot_metrics(exp_dir, exp_model_name, metrics, ['Loss', 'Accuracy'])


def test(
        exp_dir, exp_model_name, model, test_loader, criterion, label_mapping,
        device, metrics_visualization=True
):
    save_model_path = os.path.join(exp_dir, f'{exp_model_name}.pth')

    if not os.path.exists(save_model_path):
        raise FileNotFoundError('No models checkpoint file found in the directory.')

    model.load_state_dict(torch.load(str(save_model_path), weights_only=True))

    model.eval()
    total_test_loss = 0
    total_test_correct = 0
    total_test_samples = 0
    all_labels = []
    all_predicts = []
    all_predicts_proba = []
    test_progress_bar = tqdm(test_loader, desc=f'Testing', leave=True)
    with torch.no_grad():
        for X, y in test_progress_bar:
            X, y = X.to(device), y.to(device)

            logits = model(X)
            loss = criterion(logits, y)

            total_test_loss += (loss.item() * y.size(0))
            total_test_correct += (logits.argmax(dim=1) == y).sum().item()
            total_test_samples += y.size(0)

            all_labels.extend(y.cpu().numpy())
            all_predicts.extend(logits.argmax(dim=1).cpu().numpy())
            all_predicts_proba.extend(torch.softmax(logits, dim=1).cpu().numpy())

            test_progress_bar.set_postfix({
                'Test Loss': f'{(total_test_loss / total_test_samples):.4f}',
                'Accuracy': f'{(total_test_correct / total_test_samples):.4f}'
            })

    test_loss = total_test_loss / total_test_samples
    test_accuracy = total_test_correct / total_test_samples
    test_precision = compute_precision(all_labels, all_predicts)
    test_recall = compute_recall(all_labels, all_predicts)
    test_f1 = compute_f1(all_labels, all_predicts)
    print('==========================================================================================')
    print('Test Loss: {:.4f} | Test Acc: {:.4f}'.format(test_loss, test_accuracy))
    print('Test Precision: {:.4f} | Test Recall (Sensitivity): :{:.4f} | Test F1: {:.4f}'.format(
        test_precision, test_recall, test_f1)
    )

    confusion_matrix, _ = compute_confusion_matrix(all_labels, all_predicts, label_mapping)
    # print(f'Confusion Matrix: \n{matrix}')
    specificity = compute_specificity(confusion_matrix, label_mapping)
    print(f'Specificity for each class: {specificity}')
    print('==========================================================================================')

    if metrics_visualization:
        plot_confusion_matrix(all_labels, all_predicts, label_mapping, exp_model_name, exp_dir, cm=confusion_matrix)
        plot_roc_auc_curve(np.array(all_labels), np.array(all_predicts_proba), label_mapping, exp_model_name, exp_dir)

    return test_accuracy, test_precision, test_recall, test_f1