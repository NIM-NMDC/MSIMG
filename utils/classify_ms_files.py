import os
import shutil


def move_mzml_files(base_dir, target_labels, suffix='.mzML', get_label_function=None):
    mzml_files_path = []

    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(suffix):
                file_path = os.path.join(root, file)
                mzml_files_path.append(file_path)

    print(f'Found {len(mzml_files_path)} mzML files.')

    for file_path in mzml_files_path:
        filename = os.path.basename(file_path)
        label = get_label_function(filename)
        if label in target_labels:
            target_dir = os.path.join(base_dir, label)
            os.makedirs(target_dir, exist_ok=True)
            target_path = os.path.join(target_dir, filename)
            shutil.move(file_path, target_path)
            print(f'Moved {filename} to {target_path}')


if __name__ == '__main__':
    # ST000923
    base_dir = 'E:\msdata\ST000923\HMP2_HILIC-neg\HILn_rawData'

    def get_label_from_filename(filename):
        filename_without_ext = os.path.splitext(filename)[0]
        parts = filename_without_ext.split('_')
        return parts[-1] if parts else None

    move_mzml_files(
        base_dir=base_dir,
        target_labels=['CD', 'UC', 'nonIBD'],
        get_label_function=get_label_from_filename
    )