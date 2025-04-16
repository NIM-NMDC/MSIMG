import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time
import os
import random
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from utils.split_utils import split_dataset_files_by_class_stratified
from datasets.datasets import MS2DIMGDataset
from models.resnet import build_resnet
from callbacks.early_stopping import EarlyStopping
from utils.train_utils import train, test


os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3, 4, 5, 6, 7"


def _get_dataset_dirs(root_dir, dataset_dict, dataset_names):
    dataset_dirs = []
    for dataset_name in dataset_names:
        # If exact match found in dataset_dict
        if dataset_name in dataset_dict:
            dataset_dirs.append(os.path.join(root_dir, dataset_dict[dataset_name]))
        else:
            # If partial match found in dataset_dict
            matched = [os.path.join(root_dir, v) for k, v in dataset_dict.items() if k.startswith(dataset_name)]
            if not matched:
                raise ValueError(f"Dataset {dataset_name} not found in dataset_dict.")
            dataset_dirs.extend(matched)
    return dataset_dirs


def _get_dataset(dataset_dir, random_seed):
    random.seed(random_seed)

    # file_dict: {class_name: [file_path1, file_path2, ...]}
    file_dict = defaultdict(list)

    for class_name in os.listdir(dataset_dir):
        class_dir = os.path.join(dataset_dir, class_name)
        if not os.path.exists(class_dir):
            continue

        file_paths = [
            os.path.join(class_dir, file)
            for file in os.listdir(class_dir)
            if file.lower().endswith('.npz')
        ]

        file_dict[class_name] = file_paths

    dataset = []
    for class_name, file_paths in file_dict.items():
        dataset.extend([{"class_name": class_name, "file_path": file_path} for file_path in file_paths])

    random.shuffle(dataset)

    return dataset


def _create_dataset(dataset, label_mapping, transform=None):
    return MS2DIMGDataset(
        dataset=dataset,
        label_mapping=label_mapping,
        transform=transform
    )


def exp(args):
    train_set = []
    valid_set = []
    test_set = []
    for dataset_dir in args.dataset_dirs['train']:
        train_set.extend(_get_dataset(dataset_dir, args.random_seed))
    for dataset_dir in args.dataset_dirs['test']:
        valid_test_set = _get_dataset(dataset_dir, args.random_seed)
        _valid_set, _test_set = train_test_split(
            valid_test_set,
            test_size=0.5,
            random_state=args.random_seed,
            stratify=[args.label_mapping[sample['class_name']] for sample in valid_test_set]
        )
        valid_set.extend(_valid_set)
        test_set.extend(_test_set)

    train_loader = DataLoader(
        _create_dataset(train_set, args.label_mapping),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    valid_loader = DataLoader(
        _create_dataset(valid_set, args.label_mapping),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    test_loader = DataLoader(
        _create_dataset(test_set, args.label_mapping),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    print(f"{args.model_name}_{args.dataset_info}_num_classes_{args.num_classes}_in_channels_{args.top_k}")
    exp_dir_name = f"{args.model_name}_{args.dataset_info}_num_classes_{args.num_classes}_in_channels_{args.top_k}"
    exp_dir = os.path.join(args.save_dir, exp_dir_name)

    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    exp_model_name = f"{args.model_name}_{args.dataset_info}_num_classes_{args.num_classes}_in_channels_{args.top_k}"

    model = build_resnet(
        model_name=args.model_name.lower(),
        num_classes=args.num_classes,
        in_channels=args.top_k,
        pretrained=args.pretrained
    )

    if args.use_multi_gpu and torch.cuda.device_count() > 1:
        print(f'Using {torch.cuda.device_count()} GPUs for training.')
        device = torch.device('cuda:0')
        model = model.to(device)
        model = nn.DataParallel(model)
    else:
        model = model.to(args.device)

    train_labels = [args.label_mapping[sample['class_name']] for sample in train_set]
    class_weights = compute_class_weight('balanced', classes=np.array(list(args.label_mapping.values())), y=np.array(train_labels))
    class_weights = torch.tensor(class_weights, dtype=torch.float32, device=args.device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.02)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, min_lr=1e-32)

    if args.use_early_stopping:
        early_stopping = EarlyStopping(patience=args.patience)
    else:
        early_stopping = None

    train(
        exp_dir=exp_dir,
        exp_model_name=exp_model_name,
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        criterion=criterion,
        optimizers=[optimizer],
        schedulers=[scheduler],
        early_stopping=early_stopping,
        epochs=args.epochs,
        device=args.device,
        use_early_stopping=args.use_early_stopping,
        metrics_visualization=True
    )

    accuracy, precision, recall, f1_score = test(
        exp_dir=exp_dir,
        exp_model_name=exp_model_name,
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        label_mapping=args.label_mapping,
        device=args.device,
        metrics_visualization=True
    )

    metrics_result = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1_score
    }

    # Save metrics to CSV using pandas
    time_stamp = datetime.now().strftime('%Y%m%d%H%M%S')
    csv_file = os.path.join(exp_dir, f"{exp_dir_name}_metrics_{time_stamp}.csv")
    df = pd.DataFrame([metrics_result])
    df.to_csv(csv_file, index=False)

    return exp_dir, exp_model_name, metrics_result


def main():
    parser = argparse.ArgumentParser(description='Mass Spectra 2D Image')
    parser.add_argument('--root_dir', type=str, default='../', help='Root directory')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--model_name', type=str, default='ResNet50', help='Model name')
    parser.add_argument('--train_dataset_names', nargs='+', required=True, help='Dataset to use for training (e.g. ST000923-C8-pos)')
    parser.add_argument('--test_dataset_names', nargs='+', required=True, help='Dataset to use for testing (e.g. ST000923-C18-neg)')
    parser.add_argument('--label_maps', nargs='+', help='List of label maps to use (e.g. HC=0 CD=1 UC=2)')

    parser.add_argument('--top_k', type=int, default=512, help='Number of patches to be selected')
    parser.add_argument('--patch_width', type=int, default=224, help='Width of the patches to be extracted')
    parser.add_argument('--patch_height', type=int, default=224, help='Height of the patches to be extracted')
    parser.add_argument('--overlap_col', type=int, default=0, help='Number of overlapping pixels between patches in the column direction')
    parser.add_argument('--overlap_row', type=int, default=0, help='Number of overlapping pixels between patches in the row direction')
    parser.add_argument('--bin_size', type=float, default=0.01, help='Bin size for m/z binning')

    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=128, help='Number of epochs')
    parser.add_argument('--device', type=str, default=None, help='Device to use')
    parser.add_argument('--preload', action='store_true', help='Preload dataset into memory')
    parser.add_argument('--num_workers', type=int, default=32, help='Number of workers for DataLoader')
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained model')
    parser.add_argument('--use_multi_gpu', action='store_true', help='Use multiple GPUs')
    parser.add_argument('--use_augmentation', action='store_true', help='Use augmentation')
    parser.add_argument('--use_early_stopping', action='store_true', help='Use early stopping')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    parser.add_argument('--random_seed', type=int, default=3407, help='Random seed for reproducibility')

    args = parser.parse_args()

    if args.device is None:
        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.use_early_stopping:
        if args.patience is None:
            args.patience = 10

    if args.preload:
        args.num_workers = 0  # Set to 0 to avoid issues with DataLoader

    # Set save directory
    save_dir = os.path.join(args.root_dir, args.save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    args.save_dir = save_dir

    dataset_parent_dir = f"datasets/IBD_2D/top_{args.top_k}_patch_{args.patch_width}x{args.patch_height}_overlap_{args.overlap_col}x{args.overlap_row}_bin_size_{args.bin_size}"
    dataset_dict = {
        'ST000923-C8-pos': f"{dataset_parent_dir}/ST000923-C8-pos",
        'ST000923-C18-neg': f"{dataset_parent_dir}/ST000923-C18-neg",
        'ST000923-HILIC-pos': f"{dataset_parent_dir}/ST000923-HILIC-pos",
        'ST000923-HILIC-neg': f"{dataset_parent_dir}/ST000923-HILIC-neg",
        'ST001000-C8-pos': f"{dataset_parent_dir}/ST001000-C8-pos",
        'ST001000-C18-neg': f"{dataset_parent_dir}/ST001000-C18-neg",
        'ST001000-HILIC-pos': f"{dataset_parent_dir}/ST001000-HILIC-pos",
        'ST001000-HILIC-neg': f"{dataset_parent_dir}/ST001000-HILIC-neg",
        'ST003161': f"{dataset_parent_dir}/ST003161",
        'ST003313': f"{dataset_parent_dir}/ST003313",
        'PXD10371': f"{dataset_parent_dir}/PXD10371",
        'MSV000089237': f"{dataset_parent_dir}/MSV000089237",
    }
    dataset_dirs = {}
    dataset_dirs['train'] = _get_dataset_dirs(root_dir=args.root_dir, dataset_dict=dataset_dict, dataset_names=args.train_dataset_names)
    dataset_dirs['test'] = _get_dataset_dirs(root_dir=args.root_dir, dataset_dict=dataset_dict, dataset_names=args.test_dataset_names)
    args.dataset_dirs = dataset_dirs

    args.dataset_info = f"train_{'_'.join(args.train_dataset_names)}_test_{'_'.join(args.test_dataset_names)}"

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

    exp_dir, trained_model_name, metrics_results = exp(args)

    print(f"{args.model_name}_{args.dataset_info}_num_classes_{args.num_classes}_in_channels_{args.top_k}")

    for metric, result in metrics_results.items():
        print(f'{metric}: {result:.4f}')


if __name__ == '__main__':
    main()