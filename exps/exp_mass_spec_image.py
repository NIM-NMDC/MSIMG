import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import os
import numpy as np
import pandas as pd
import argparse
from datetime import datetime
from sklearn.utils.class_weight import compute_class_weight

from datasets.prepare_datasets import prepare_ibd_ms_img_dataset
from models.resnet import build_resnet
from callbacks.early_stopping import EarlyStopping
from trainer import train, test


def exp(exp_args, save_dir, label_mapping, device, use_multi_gpu=False):

    model = None

    train_dataset, valid_dataset, test_dataset = prepare_ibd_ms_img_dataset(exp_args, label_mapping)
    train_loader = DataLoader(
        train_dataset,
        batch_size=exp_args['batch_size'],
        shuffle=True,
        num_workers=exp_args['num_workers']
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=exp_args['batch_size'],
        shuffle=False,
        num_workers=exp_args['num_workers']
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=exp_args['batch_size'],
        shuffle=False,
        num_workers=exp_args['num_workers']
    )

    print(f"{exp_args['model_name']} {exp_args['dataset_name']}_dataset num_classes {exp_args['num_classes']}")
    exp_dir_name = f"{exp_args['model_name']}_{exp_args['dataset_name']}_dataset_num_classes_{exp_args['num_classes']}"
    exp_dir = os.path.join(save_dir, exp_dir_name)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    exp_model_name = f"{exp_args['model_name']}_train_{exp_args['dataset_name']}_dataset"

    model = build_resnet(
        model_name=exp_args['model_name'].lower(),
        num_classes=exp_args['num_classes'],
        in_channels=exp_args['in_channels'],
        pretrained=exp_args['pretrained']
    )

    if use_multi_gpu and torch.cuda.device_count() > 1:
        print(f'Using {torch.cuda.device_count()} GPUs for training.')
        device = torch.device('cuda:0')
        model = model.to(device)
        model = nn.DataParallel(model)
    else:
        model = model.to(device)

    train_labels = [sample['label'] for sample in train_dataset.img_samples]
    class_weights = compute_class_weight('balanced', classes=np.array(list(label_mapping.values())), y=np.array(train_labels))
    class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, min_lr=1e-32)

    if exp_args['use_early_stopping']:
        early_stopping = EarlyStopping(patience=exp_args['patience'])
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
        epochs=exp_args['epochs'],
        device=device,
        use_early_stopping=exp_args['use_early_stopping'],
        metrics_visualization=True
    )

    accuracy, precision, recall, f1_score = test(
        exp_dir=exp_dir,
        exp_model_name=exp_model_name,
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        label_mapping=label_mapping,
        device=device,
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
    parser.add_argument('--dataset_name', type=str, default='IBD_2D_Full_MZ', help='Dataset to use')

    parser.add_argument('--in_channels', type=int, default=1, help='Number of input channels')
    parser.add_argument('--num_classes', type=int, default=3, help='Number of classes')

    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=128, help='Number of epochs')
    parser.add_argument('--device', type=str, default=None, help='Device to use')
    parser.add_argument('--use_multi_gpu', action='store_true', help='Use multiple GPUs')
    parser.add_argument('--use_augmentation', action='store_true', help='Use augmentation')
    parser.add_argument('--use_normalization', action='store_true', help='Use normalization')
    parser.add_argument('--use_early_stopping', action='store_true', help='Use early stopping')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')

    args = parser.parse_args()

    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    if args.use_early_stopping:
        if args.patience is None:
            args.patience = 10

    # Set save directory
    save_dir = os.path.join(args.root_dir, args.save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    dataset_dict = {
        'IBD_2D_Full_MZ': 'datasets/IBD_2D/full_mz',
    }
    dataset_dir = os.path.join(args.root_dir, dataset_dict[args.dataset_name])
    label_mapping = {'HC': 0, 'CD': 1, 'UC': 2}

    exp_args = {
        'model_name': args.model_name,
        'dataset_name': args.dataset_name,
        'root_dir': args.root_dir,
        'dataset_dir': dataset_dir,

        'in_channels': args.in_channels,
        'num_classes': args.num_classes,
        'num_workers': 4,
        'batch_size': args.batch_size,
        'epochs': args.epochs,

        'resize': (256, 256),
        'min_max_norm': True,
        'zero_mean_norm': False,
        'noise_level': 0.05,
        'spike_prob': 0.02,
        'random_erase_prob': 0.5,
        'random_erase_scale': (0.02, 0.2),
        'random_erase_value': 0.0,
        'pretrained': False,
        'use_augmentation': args.use_augmentation,
        'use_normalization': args.use_normalization,
        'use_early_stopping': args.use_early_stopping,
        'patience': args.patience,
        'random_seed': 3407,
    }

    exp_dir, trained_model_name, metrics_results = exp(
        exp_args=exp_args,
        save_dir=save_dir,
        label_mapping=label_mapping,
        device=device,
        use_multi_gpu=args.use_multi_gpu
    )

    print(f"{exp_args['model_name']} {exp_args['dataset']}_dataset num_classes {exp_args['num_classes']}")

    for metric, result in metrics_results.items():
        print(f'{metric}: {result:.4f}')


if __name__ == '__main__':
    main()





