import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import os
import numpy as np
import pandas as pd
import argparse
from datetime import datetime
from sklearn.utils.class_weight import compute_class_weight

from utils.split_utils import split_dataset_files_by_class_stratified
from datasets.datasets import MS2DIMGDataset
from models.resnet import build_resnet
from callbacks.early_stopping import EarlyStopping
from utils.train_utils import train, test


os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3, 4, 5, 6, 7"


def create_dataset(dataset, label_mapping, transform=None):
    return MS2DIMGDataset(
        dataset=dataset,
        label_mapping=label_mapping,
        transform=transform
    )


def exp(args):
    train_set = []
    valid_set = []
    test_set = []

    for dataset_dir in args.dataset_dirs:
        _train_set, _valid_set, _test_set = split_dataset_files_by_class_stratified(
            root_dir=dataset_dir,
            train_size=0.8,
            test_size=0.1,
            random_seed=args.random_seed
        )
        train_set.extend(_train_set)
        valid_set.extend(_valid_set)
        test_set.extend(_test_set)

    train_loader = DataLoader(
        create_dataset(train_set, args.label_mapping),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    valid_loader = DataLoader(
        create_dataset(valid_set, args.label_mapping),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    test_loader = DataLoader(
        create_dataset(valid_set, args.label_mapping),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    print(f"{args.model_name}_{args.dataset_name}_num_classes_{args.num_classes}_in_channels_{args.top_k}")
    exp_dir_name = f"{args.model_name}_{args.dataset_name}_num_classes_{args.num_classes}_in_channels_{args.top_k}"
    exp_dir = os.path.join(args.save_dir, exp_dir_name)

    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    exp_model_name = f"{args.model_name}_train_{args.dataset_name}_num_classes_{args.num_classes}_in_channels_{args.top_k}"

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
    parser.add_argument('--dataset_names', nargs='+', required=True, help='List of datasets to use (e.g. ST000923-C8-pos ST000923-C18-neg)')

    parser.add_argument('--top_k', type=int, default=512, help='Number of patches to be selected')
    parser.add_argument('--patch_width', type=int, default=224, help='Width of the patches to be extracted')
    parser.add_argument('--patch_height', type=int, default=224, help='Height of the patches to be extracted')
    parser.add_argument('--overlap_col', type=int, default=0, help='Number of overlapping pixels between patches in the column direction')
    parser.add_argument('--overlap_row', type=int, default=0, help='Number of overlapping pixels between patches in the row direction')
    parser.add_argument('--bin_size', type=float, default=0.01, help='Bin size for m/z binning')

    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=128, help='Number of epochs')
    parser.add_argument('--device', type=str, default=None, help='Device to use')
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

    # Set save directory
    save_dir = os.path.join(args.root_dir, args.save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    args.save_dir = save_dir

    dataset_dict = {
        'ST000923-C8-pos': f'datasets/IBD_2D/top_{args.top_k}_patch_{args.patch_width}x{args.patch_height}_overlap_{args.overlap_col}x{args.overlap_row}_bin_size_{args.bin_size}/ST000923-C8-pos',
        'ST000923-C18-neg': f'datasets/IBD_2D/top_{args.top_k}_patch_{args.patch_width}x{args.patch_height}_overlap_{args.overlap_col}x{args.overlap_row}_bin_size_{args.bin_size}/ST000923-C18-neg',
        'ST000923-HILIC-pos': f'datasets/IBD_2D/top_{args.top_k}_patch_{args.patch_width}x{args.patch_height}_overlap_{args.overlap_col}x{args.overlap_row}_bin_size_{args.bin_size}/ST000923-HILIC-pos',
        'ST000923-HILIC-neg': f'datasets/IBD_2D/top_{args.top_k}_patch_{args.patch_width}x{args.patch_height}_overlap_{args.overlap_col}x{args.overlap_row}_bin_size_{args.bin_size}/ST000923-HILIC-neg'
    }
    dataset_dirs = []
    if args.dataset_names == ['.'] or args.dataset_names == ['all'] or args.dataset_names == ['ALL'] or args.dataset_names == ['All']:
        args.dataset_names = dataset_dict.keys()

    for dataset_name in args.dataset_names:
        if dataset_name in dataset_dict:
            dataset_dir = os.path.join(args.root_dir, dataset_dict[args.dataset_name])
            if not os.path.exists(dataset_dir):
                raise FileNotFoundError(f"Dataset directory {dataset_dir} does not exist.")
            dataset_dirs.append(dataset_dict[dataset_name])
        else:
            raise ValueError(f"Dataset name {dataset_name} not found in dataset_dict.")
    args.dataset_dirs = dataset_dirs

    label_mapping = {'nonIBD': 0, 'CD': 1, 'UC': 2}
    args.label_mapping = label_mapping
    args.num_classes = len(label_mapping)

    exp_dir, trained_model_name, metrics_results = exp(args)

    print(f"{args.model_name}_{args.dataset_name}_num_classes_{args.num_classes}_in_channels_{args.top_k}")

    for metric, result in metrics_results.items():
        print(f'{metric}: {result:.4f}')


if __name__ == '__main__':
    main()