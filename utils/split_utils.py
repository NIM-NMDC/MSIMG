import os
import random

from collections import defaultdict
from sklearn.model_selection import train_test_split


def split_dataset_files_by_class_stratified(root_dir, train_size=0.8, test_size=0.1, random_seed=3407):
    """
    Dataset Structure:
    - dataset_dir/
        - Class A/ [IMGs]
        - Class B/ [IMGs]
        - Class C/ [IMGs]
    """
    random.seed(random_seed)

    # file_dict: {class_name: [file_path1, file_path2, ...]}
    file_dict = defaultdict(list)

    for class_name in os.listdir(root_dir):
        class_dir = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        file_paths = [
            os.path.join(class_dir, file)
            for file in os.listdir(class_dir)
            if file.lower().endswith('.npz')
        ]

        file_dict[class_name] = file_paths

    train_set = []
    valid_set = []
    test_set = []

    for class_name, file_paths in file_dict.items():
        train_file_paths, remaining_file_paths = train_test_split(
            file_paths,
            train_size=train_size,
            random_state=random_seed,
            shuffle=True
        )
        test_file_paths, valid_file_paths = train_test_split(
            remaining_file_paths,
            test_size=test_size / (1 - train_size),
            random_state=random_seed,
            shuffle=True
        )

        # train_set.extend([(class_name, file_path) for file_path in train_file_paths])
        train_set.extend([{'class_name': class_name, 'file_path': file_path} for file_path in train_file_paths])
        valid_set.extend([{'class_name': class_name, 'file_path': file_path} for file_path in valid_file_paths])
        test_set.extend([{'class_name': class_name, 'file_path': file_path} for file_path in test_file_paths])

    if len(valid_set) == 0:
        raise ValueError('The validation set is empty. Please adjust the split strategy.')

    random.shuffle(train_set)
    random.shuffle(valid_set)
    random.shuffle(test_set)

    return train_set, valid_set, test_set


def split_dataset_files_by_domain_class_stratified(root_dir, train_size=0.8, test_size=0.1, random_seed=3407):
    """
    Dataset Structure:
    - dataset_dir/
        - Domain A/
            - Class A/ [IMGs]
            - Class B/ [IMGs]
        - Domain B/
            - Class A/ [IMGs]
            - Class B/ [IMGs]
    """
    random.seed(random_seed)

    # file_dict: {(domain, class): [file_path1, file_path2, ...]}
    file_dict = defaultdict(list)

    for domain_name in os.listdir(root_dir):
        domain_dir = os.path.join(root_dir, domain_name)
        if not os.path.isdir(domain_dir):
            continue

        for class_name in os.listdir(domain_dir):
            class_dir = os.path.join(domain_dir, class_name)
            if not os.path.isdir(class_dir):
                continue

            file_paths = [
                os.path.join(class_dir, file)
                for file in os.listdir(class_dir)
                if file.lower().endswith('.npz')
            ]
            file_dict[(domain_name, class_name)] = file_paths

    train_set = []
    valid_set = []
    test_set = []

    for (domain_name, class_name), file_paths in file_dict.items():
        train_file_paths, remaining_file_paths = train_test_split(
            file_paths,
            train_size=train_size,
            random_state=random_seed,
            shuffle=True
        )
        test_file_paths, valid_file_paths = train_test_split(
            remaining_file_paths,
            test_size=test_size / (1 - train_size),
            random_state=random_seed,
            shuffle=True
        )

        train_set.extend([(domain_name, class_name, file_path) for file_path in train_file_paths])
        valid_set.extend([(domain_name, class_name, file_path) for file_path in valid_file_paths])
        test_set.extend([(domain_name, class_name, file_path) for file_path in test_file_paths])

    if len(valid_set) == 0:
        raise ValueError('The validation set is empty. Please adjust the split strategy.')

    random.shuffle(train_set)
    random.shuffle(valid_set)
    random.shuffle(test_set)

    return train_set, valid_set, test_set





