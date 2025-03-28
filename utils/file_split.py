import os
import random

from collections import defaultdict


def split_dataset_files_by_label_stratified(root_dir, train_size=0.8, test_size=0.1, random_seed=3407):
    """
    - root_dir/
        - HC/
            - ST000923/ [IMG]
            - ST001000/ [IMG]
        - CD/
            - ST000923/ [IMG]
            - ST001000/ [IMG]
        - UC/
            - ST000923/ [IMG]
            - ST001000/ [IMG]
    """
    random.seed(random_seed)

    # file_dict: {(class, sub_dataset): [file_path1, file_path2, ...]}
    file_dict = defaultdict(list)

    for class_name in os.listdir(root_dir):
        class_dir = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        for sub_dataset_name in os.listdir(class_dir):
            sub_dataset_dir = os.path.join(class_dir, sub_dataset_name)
            if not os.path.isdir(sub_dataset_dir):
                continue

            file_paths = [
                os.path.join(sub_dataset_dir, file)
                for file in os.listdir(sub_dataset_dir)
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
            ]
            file_dict[(class_name, sub_dataset_name)] = file_paths

    train_set = []
    valid_set = []
    test_set = []

    for key, files in file_dict.items():
        # class_name, sub_dataset_name = key
        random.shuffle(files)
        file_count = len(files)

        test_count = max(1, int(round(file_count * test_size)))
        valid_count = max(1, int(round(file_count * (1 - train_size - test_size))))
        train_count = file_count - test_count - valid_count

        if train_count <= 0:
            train_count += valid_count
            valid_count = 0

        test = files[:test_count]
        valid = files[test_count:test_count + valid_count]
        train = files[test_count + valid_count:]

        train_set.extend(train)
        valid_set.extend(valid)
        test_set.extend(test)

    if len(valid_set) == 0:
        raise ValueError('The validation set is empty. Please adjust the split strategy.')

    random.shuffle(train_set)
    random.shuffle(valid_set)
    random.shuffle(test_set)

    return train_set, valid_set, test_set





