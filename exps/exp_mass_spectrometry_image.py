import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import os
import math
import time
import yaml
import numpy as np
import pandas as pd
import argparse
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

from utils.split_utils import split_dataset_files_by_class_stratified
from datasets.prepare_datasets import prepare_ms_img_dataset
from datasets.datasets import MSIMGDataset
from models.resnet_2d import build_resnet_2d
from callbacks.early_stopping import EarlyStopping
from utils.train_utils import train, test
from utils.metrics import calculate_bootstrap_ci


# os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"


def load_params_from_yaml(file_path, key=None):
    """
    Load parameters from a YAML file.

    :param file_path: Path to the YAML file.
    :return: Dictionary containing the parameters.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"YAML file {file_path} does not exist.")

    with open(file_path, 'r') as file:
        params = yaml.safe_load(file)

    if not isinstance(params, dict):
        raise ValueError("YAML file must contain a dictionary of parameters.")

    if key:
        if key not in params:
            raise KeyError(f"Key '{key}' not found in the YAML file.")
        return params.get(key)
    else:
        return params


def set_seeds(seed):
    """
    Set random seeds for reproducibility.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _create_dataset(patches_list, positions_list, padding_mask_list, labels, return_positions, transform=None):
    return MSIMGDataset(
        patches_list=patches_list,
        positions_list=positions_list,
        padding_mask_list=padding_mask_list,
        labels=labels,
        return_positions=return_positions,
        transform=transform
    )


def run_experiment(args):
    set_seeds(args.random_seed)
    print(f"Dataset directory: {args.dataset_dir}")
    train_set, test_set = split_dataset_files_by_class_stratified(
        dataset_dir=args.dataset_dir,
        suffix='.npz',
        train_size=0.8,
        test_size=0.2,
        random_seed=args.random_seed
    )
    train_patches_list, train_positions_list, train_padding_mask_list, train_labels = prepare_ms_img_dataset(dataset=train_set, label_mapping=args.label_mapping)
    test_patches_list, test_positions_list, test_padding_mask_list, test_labels = prepare_ms_img_dataset(dataset=test_set, label_mapping=args.label_mapping)

    exp_dir_name = (f"{args.model_name}_{args.dataset_name}_num_classes_{args.num_classes}_"
                    f"in_channels_{args.top_k}_patch_{args.patch_height}x{args.patch_width}_batch_size_{args.batch_size}")
    print(exp_dir_name)
    exp_base_dir = os.path.join(args.save_dir, exp_dir_name)

    if not os.path.exists(exp_base_dir):
        os.makedirs(exp_base_dir)

    skf = StratifiedKFold(n_splits=args.k_folds, shuffle=True, random_state=args.random_seed)
    fold_test_metrics_list = []

    for fold_idx, (train_fold_indices, valid_fold_indices) in enumerate(skf.split(train_patches_list, train_labels)):
        print(f"{args.model_name} Fold {fold_idx + 1}/{args.k_folds}")
        exp_model_name = f"{args.model_name}_kfold_{fold_idx + 1}_trained_on_{args.dataset_name}_{args.num_classes}"

        train_fold_patches_list, train_fold_positions_list, train_fold_padding_mask_list, train_fold_labels = \
            train_patches_list[train_fold_indices], train_positions_list[train_fold_indices], train_padding_mask_list[train_fold_indices], train_labels[train_fold_indices]
        valid_fold_patches_list, valid_fold_positions_list, valid_fold_padding_mask_list, valid_fold_labels = \
            train_patches_list[valid_fold_indices], train_positions_list[valid_fold_indices], train_padding_mask_list[valid_fold_indices], train_labels[valid_fold_indices]
        print(f'X_train.shape: {train_fold_patches_list.shape}, y_train.shape: {train_fold_labels.shape}')
        print(f'X_valid.shape: {valid_fold_patches_list.shape}, y_valid.shape: {valid_fold_labels.shape}')
        print(f'X_test.shape: {test_patches_list.shape}, y_test.shape: {test_labels.shape}')

        exp_model_name = f"{args.model_name}_{args.dataset_name}_num_classes_{args.num_classes}_in_channels_{args.top_k}"
        model = None
        return_positions = False
        if 'ResNet' in args.model_name:
            model = build_resnet_2d(args)
            return_positions = False
        elif 'ViT' in args.model_name:
            return_positions = True

        if args.use_multi_gpu and torch.cuda.device_count() > 1:
            print(f'Using {torch.cuda.device_count()} GPUs for training.')
            print("DataParallel typically expects model on primary GPU (cuda:0). Moving model to cuda:0 before DataParallel.")
            model = model.to(args.device)
            model = nn.DataParallel(model)  # Wrap the models with DataParallel for multi-GPU support
        else:
            model = model.to(args.device)

        class_weights = compute_class_weight('balanced', classes=np.array(list(args.label_mapping.values())), y=train_fold_labels)
        class_weights = torch.tensor(class_weights, dtype=torch.float32, device=args.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        # criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, min_lr=1e-32)

        if args.use_early_stopping:
            early_stopping = EarlyStopping(patience=args.patience)
        else:
            early_stopping = None

        train_loader = DataLoader(
            _create_dataset(
                patches_list=train_fold_patches_list,
                positions_list=train_fold_positions_list,
                padding_mask_list=train_fold_padding_mask_list,
                labels=train_fold_labels,
                return_positions=return_positions,
                transform=None
            ),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers
        )
        valid_loader = DataLoader(
            _create_dataset(
                patches_list=valid_fold_patches_list,
                positions_list=valid_fold_positions_list,
                padding_mask_list=valid_fold_padding_mask_list,
                labels=valid_fold_labels,
                return_positions=return_positions,
                transform=None
            ),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers
        )
        test_loader = DataLoader(
            _create_dataset(
                patches_list=test_patches_list,
                positions_list=test_positions_list,
                padding_mask_list=test_padding_mask_list,
                labels=test_labels,
                return_positions=return_positions,
                transform=None
            ),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers
        )

        train(
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            early_stopping=early_stopping,
            epochs=args.epochs,
            device=args.device,
            exp_base_dir=exp_base_dir,
            exp_model_name=exp_model_name,
            metrics_visualization=True
        )

        accuracy, precision, recall, f1_score = test(
            model=model,
            test_loader=test_loader,
            criterion=criterion,
            label_mapping=args.label_mapping,
            device=args.device,
            exp_base_dir=exp_base_dir,
            exp_model_name=exp_model_name,
            metrics_visualization=True
        )

        fold_test_metrics_list.append({
            'Fold': fold_idx + 1,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1_score
        })

    fold_test_metrics_df = pd.DataFrame(fold_test_metrics_list)
    mean_metrics = fold_test_metrics_df[['Accuracy', 'Precision', 'Recall', 'F1 Score']].mean()
    std_metrics = fold_test_metrics_df[['Accuracy', 'Precision', 'Recall', 'F1 Score']].std()

    summary_stats_list = []
    for metric in ['Accuracy', 'Precision', 'Recall', 'F1 Score']:
        metric_values = fold_test_metrics_df[metric].values
        ci_lower, ci_upper = calculate_bootstrap_ci(metric_values, random_seed=args.random_seed)
        summary_stats_list.append({
            'Metric': metric,
            'Mean_Test_on_Holdout': mean_metrics.get(metric, np.nan),
            'Std_Test_on_Holdout': std_metrics.get(metric, np.nan),
            '95% CI Lower': ci_lower,
            '95% CI Upper': ci_upper
        })

    print(exp_dir_name)
    summary_stats_df = pd.DataFrame(summary_stats_list)
    print("Summary Statistics (Mean, Std, 95% Bootstrap CI from K-Fold models tested on hold-out):")
    print(summary_stats_df)

    time_stamp = datetime.now().strftime('%Y%m%d%H%M%S')
    fold_test_metrics_csv_path = os.path.join(exp_base_dir, f"{args.model_name}_{args.dataset_name}_num_classes_{args.num_classes}_in_channels_{args.top_k}_kfold_tested_on_holdout_metrics_{time_stamp}.csv")
    summary_stats_csv_path = os.path.join(exp_base_dir, f"{args.model_name}_{args.dataset_name}_num_classes_{args.num_classes}_in_channels_{args.top_k}_kfold_tested_on_holdout_summary_stats_{time_stamp}.csv")
    fold_test_metrics_df.to_csv(fold_test_metrics_csv_path, index=False)
    summary_stats_df.to_csv(summary_stats_csv_path, index=False)


def main():
    parser = argparse.ArgumentParser(description='Mass Spectra 2D Image')
    parser.add_argument('--root_dir', type=str, default='../', help='Root directory')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--model_name', type=str, default='ResNet50', help='Model name')
    parser.add_argument('--dataset_name', type=str, help='Dataset to use')
    parser.add_argument('--label_maps', nargs='+', help='List of label maps to use (e.g. HC=0 CD=1 UC=2)')

    parser.add_argument('--patch_strategy', type=str, required=True, choices=['grid', 'ics'], help='Strategy to generate patches (e.g., grid, ics)')
    parser.add_argument('--score_strategy', type=str, default='entropy', choices=['entropy', 'mean'], help='Strategy to calculate patch scores (e.g. Entropy: 1D image entropy, Mean: mean intensity, Random: random selection)')
    parser.add_argument('--top_k', type=int, default=256, help='Number of patches to be selected')
    parser.add_argument('--patch_height', type=int, default=224, help='Height of the patches to be extracted')
    parser.add_argument('--patch_width', type=int, default=224, help='Width of the patches to be extracted')
    parser.add_argument('--bin_size', type=float, default=0.01, help='Bin size for m/z binning')

    parser.add_argument('--k_folds', type=int, default=6, help='Number of patches to be selected')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=64, help='Number of epochs')
    parser.add_argument('--device', type=str, default=None, help='Device to use')
    parser.add_argument('--num_workers', type=int, default=32, help='Number of workers for DataLoader')
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained model')
    parser.add_argument('--use_multi_gpu', action='store_true', help='Use multiple GPUs')
    parser.add_argument('--use_early_stopping', action='store_true', help='Use early stopping')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--random_seed', type=int, default=3407, help='Random seed for reproducibility')

    args = parser.parse_args()

    if args.device is None:
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.use_multi_gpu:
        args.device = torch.device("cuda:0")

    # Set save directory
    save_dir = os.path.join(args.root_dir, args.save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    args.save_dir = save_dir

    patch_params = load_params_from_yaml(file_path=os.path.join(args.root_dir, 'configs/patch_config.yaml'), key=args.patch_strategy)
    # print(f"Patch parameters for strategy '{args.patch_strategy}': {patch_params}")
    if args.patch_strategy == 'ics':
        dataset_sub_dir = f"{args.score_strategy}_top_{args.top_k}_{patch_params.get('information_metric')}_{args.patch_strategy}_patch_{args.patch_height}x{args.patch_width}_{patch_params.get('detection_metric_type')}_{patch_params.get('metric_threshold')}_bin_size_{args.bin_size}"
    elif args.patch_strategy == 'grid':
        dataset_sub_dir = f"{args.score_strategy}_top_{args.top_k}_{args.patch_strategy}_patch_{args.patch_height}x{args.patch_width}_overlap_{patch_params.get('overlap_row')}x{patch_params.get('overlap_col')}_bin_size_{args.bin_size}"
    else:
        raise ValueError(f"Invalid patch strategy: {args.patch_strategy}. Supported strategies are 'ics' and 'grid'.")
    dataset_dict = {
        'ST000923-C8-pos': f"datasets/MS-IMG/ST000923-C8-pos/{dataset_sub_dir}",
        # 'ST000923-C18-neg': f"{dataset_parent_dir}/ST000923-C18-neg",
        # 'ST000923-HILIC-pos': f"{dataset_parent_dir}/ST000923-HILIC-pos",
        # 'ST000923-HILIC-neg': f"{dataset_parent_dir}/ST000923-HILIC-neg",
        # 'ST001000-C8-pos': f"{dataset_parent_dir}/ST001000-C8-pos",
        # 'ST001000-C18-neg': f"{dataset_parent_dir}/ST001000-C18-neg",
        # 'ST001000-HILIC-pos': f"{dataset_parent_dir}/ST001000-HILIC-pos",
        # 'ST001000-HILIC-neg': f"{dataset_parent_dir}/ST001000-HILIC-neg",
        # 'ST003161': f"{dataset_parent_dir}/ST003161",
        'ST003313': f"datasets/MS-IMG/ST003313/{dataset_sub_dir}",
        # 'PXD10371': f"{dataset_parent_dir}/PXD10371",
        # 'MSV000089237': f"{dataset_parent_dir}/MSV000089237",
    }
    dataset_dir = os.path.join(args.root_dir, dataset_dict[args.dataset_name])
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset directory {dataset_dir} does not exist.")
    args.dataset_dir = dataset_dir

    label_mapping = {}
    if args.label_maps:
        for pair in args.label_maps:
            label, value = pair.split('=')
            if value.isdigit():
                label_mapping[label] = int(value)
            else:
                raise ValueError(f"Invalid label mapping: {pair}. Value must be an integer.")
    else:
        label_mapping = {'HC': 0, 'CD': 1, 'UC': 2}

    args.label_mapping = label_mapping
    args.num_classes = len(label_mapping)
    args.in_channels = args.top_k

    run_experiment(args)


if __name__ == '__main__':
    main()