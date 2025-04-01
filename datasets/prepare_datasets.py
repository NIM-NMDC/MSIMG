from utils.file_split import split_dataset_files_by_label_stratified
from datasets.transforms import build_base_transform, build_dynamic_transform
from datasets.datasets import MS2DImgDataset


def prepare_ibd_ms_img_dataset(exp_args, label_mapping):
    """
    Prepare the IBD mass spectrometry image dataset.
    """
    train_set, valid_set, test_set = split_dataset_files_by_label_stratified(
        root_dir=exp_args['ms_img_dir'],
        train_size=0.8,
        test_size=0.1,
        random_seed=exp_args['random_seed']
    )

    transform_config = {
        'resize': exp_args['resize'],
        'min_max_norm': exp_args.get('min_max_norm', True),
        'zero_mean_norm': exp_args.get('zero_mean_norm', False),
        'noise_level': exp_args.get('noise_level', 0.05),
        'spike_prob': exp_args.get('spike_prob', 0.02),
        'random_erase_prob': exp_args.get('random_erase_prob', 0.5),
        'random_erase_scale': exp_args.get('random_erase_scale', (0.02, 0.2)),
        'random_erase_value': exp_args.get('random_erase_value', 0.0)
    }

    if exp_args.get('use_dynamic_transform', False):
        train_transform = build_dynamic_transform(config=transform_config, aug_prob=0.5)
    else:
        train_transform = build_base_transform(config=transform_config)

    valid_transform = build_base_transform(config=transform_config)
    test_transform = build_base_transform(config=transform_config)

    train_dataset = MS2DImgDataset(file_paths=train_set, label_mapping=label_mapping, transform=train_transform)
    valid_dataset = MS2DImgDataset(file_paths=valid_set, label_mapping=label_mapping, transform=valid_transform)
    test_dataset = MS2DImgDataset(file_paths=test_set, label_mapping=label_mapping, transform=test_transform)

    return train_dataset, valid_dataset, test_dataset




