import os
import shutil


def get_file_paths(base_dir, suffix):
    file_paths = []

    if os.path.exists(base_dir):
        for root, dirs, files in os.walk(base_dir):
            for file in files:
                if file.endswith(suffix):
                    file_path = os.path.join(root, file)
                    file_paths.append(file_path)
        return file_paths
    else:
        raise FileNotFoundError(f"Directory {base_dir} not found.")


def move_files(base_dir, suffix, target_labels, get_label_function=None):
    files_path = get_file_paths(base_dir=base_dir, suffix=suffix)

    print(f'Found {len(files_path)} {suffix} files.')

    for file_path in files_path:
        file_name = os.path.basename(file_path)
        label = get_label_function(file_name)
        if label in target_labels:
            target_dir = os.path.join(base_dir, label)
            os.makedirs(target_dir, exist_ok=True)
            target_path = os.path.join(target_dir, file_name)
            shutil.move(file_path, target_path)
            print(f'Moved {file_name} to {target_path}')


if __name__ == '__main__':

    def get_label_from_filename(filename):
        filename_without_ext = os.path.splitext(filename)[0]
        parts = filename_without_ext.split('_')
        return parts[-1] if parts else None

    base_dir = input('Please input the mass spectrometry files directory: ')
    target_labels_input = input('Please input the target labels (e.g., HC, CD, UC): ')
    target_labels = [label.strip() for label in target_labels_input.split(',') if label.strip()]
    print(target_labels)
    suffix = input('Please input the file suffix (e.g., .mzML, .raw): ')
    if not suffix.startswith('.'):
        suffix = '.' + suffix

    # ST000923
    move_files(
        base_dir=base_dir,
        suffix=suffix,
        target_labels=target_labels,
        get_label_function=get_label_from_filename
    )

    # move_files(
    #     base_dir=r'E:\msdata\ST001000\HILIC-neg',
    #     target_labels=['HC', 'CD', 'UC'],
    #     suffix='.raw',
    #     get_label_function=get_label_from_filename
    # )