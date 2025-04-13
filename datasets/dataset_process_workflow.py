import os
import shutil
import argparse

from collections import defaultdict

from utils.rasterize_ms import parallel_parse_ms
from utils.patch_ms import parallel_calculate_patches_top_k, parallel_select_top_k_patches


def move_dataset(dataset_dir, save_dir, dataset_files):
    save_dir = os.path.join(dataset_dir, save_dir)
    os.makedirs(save_dir, exist_ok=True)

    for class_name, file_paths in dataset_files.items():
        save_class_dir = os.path.join(save_dir, class_name)
        os.makedirs(save_class_dir, exist_ok=True)

        for file_path in file_paths:
            src_path = file_path
            dest_path = os.path.join(save_class_dir, os.path.basename(file_path))

            shutil.move(src_path, dest_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataset Process Workflow')
    parser.add_argument('--dataset_dir', type=str, required=True, help='Path to the dataset directory')
    parser.add_argument('--suffix', type=str, default='mzML', help='File suffix to filter (e.g., .mzML, .mzXML)')
    parser.add_argument('--mz_min', type=float, required=True, help='Minimum m/z value for binning')
    parser.add_argument('--mz_max', type=float, required=True, help='Maximum m/z value for binning')
    parser.add_argument('--bin_size', type=float, required=True, help='Bin size for m/z binning')
    parser.add_argument('--select_method', type=str, default='entropy', help='Method to calculate (e.g. Entropy: 1D image entropy, Mean: mean intensity)')
    parser.add_argument('--patch_width', type=int, default=224, help='Width of the patches to be extracted')
    parser.add_argument('--patch_height', type=int, default=224, help='Height of the patches to be extracted')
    parser.add_argument('--overlap_col', type=int, default=0, help='Number of overlapping pixels between patches in the column direction')
    parser.add_argument('--overlap_row', type=int, default=0, help='Number of overlapping pixels between patches in the row direction')
    parser.add_argument('--top_k', type=int, default=512, help='Number of patches to be selected')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker processes to use')

    args = parser.parse_args()

    if '.' not in args.suffix:
        args.suffix = '.' + args.suffix

    dataset_files = defaultdict(list)

    for class_name in os.listdir(args.dataset_dir):
        class_dir = os.path.join(args.dataset_dir, class_name)
        if os.path.isdir(class_dir):
            for file_name in os.listdir(class_dir):
                if file_name.endswith(args.suffix):
                    dataset_files[class_name].append(os.path.join(class_dir, file_name))

    print('Beginning Dataset Processing Workflow...')
    print('Binning MS Files...')
    binned_dataset_files = defaultdict(list)
    bin_prefix = f'mz_{args.mz_min}-{args.mz_max}_bin_size_{args.bin_size}'
    for class_name, file_paths in dataset_files.items():
        print(f"Processing class: {class_name}")

        parallel_parse_ms(
            ms_file_paths=file_paths,
            prefix=bin_prefix,
            mz_min=args.mz_min,
            mz_max=args.mz_max,
            bin_size=args.bin_size,
            workers=args.num_workers
        )

        binned_dataset_files[class_name] = [
            os.path.dirname(file_path) + '/' + bin_prefix + '_' + os.path.splitext(os.path.basename(file_path))[0] + '.npz'
            for file_path in file_paths
        ]

    print('Dataset Binning Process Completed.')
    # print(binned_dataset_files)

    print('Patching MS Files...')
    patched_dataset_files = defaultdict(list)
    top_k_indices_dict = {}
    patch_prefix = f'patch_{args.patch_width}x{args.patch_height}_overlap_{args.overlap_col}x{args.overlap_row}'
    for class_name, file_paths in binned_dataset_files.items():
        print(f"Patching Class: {class_name}")
        top_k_indices = parallel_calculate_patches_top_k(
            file_paths=file_paths,
            prefix=patch_prefix,
            method=args.select_method,
            patch_width=args.patch_width,
            patch_height=args.patch_height,
            overlap_col=args.overlap_col,
            overlap_row=args.overlap_row,
            top_k=args.top_k,
            workers=args.num_workers
        )
        top_k_indices_dict[class_name] = top_k_indices

        patched_dataset_files[class_name] = [
            os.path.dirname(file_path) + '/' + patch_prefix + '_' + os.path.basename(file_path)
            for file_path in file_paths
        ]

    print('Dataset Patching Process Completed.')
    # print(patched_dataset_files)
    # print(top_k_indices_dict)

    print('Selecting Top K Patches...')
    selected_patches_dataset_files = defaultdict(list)
    select_prefix = f'top_{args.top_k}'
    for class_name, file_paths in patched_dataset_files.items():
        print(f"Selecting Top K Patches for Class: {class_name}")

        top_k_indices = top_k_indices_dict[class_name]
        parallel_select_top_k_patches(
            file_paths=file_paths,
            prefix=select_prefix,
            top_k_indices=top_k_indices,
            workers=args.num_workers
        )

        selected_patches_dataset_files[class_name] = [
            os.path.dirname(file_path) + '/' + select_prefix + '_' + os.path.basename(file_path)
            for file_path in file_paths
        ]

    print('Top K Patches Selection Process Completed.')
    # print(selected_patches_dataset_files)

    # Move processed files to a new directory
    move_dataset(dataset_dir=args.dataset_dir, save_dir=bin_prefix, dataset_files=binned_dataset_files)
    move_dataset(dataset_dir=args.dataset_dir, save_dir=f'{patch_prefix}_{bin_prefix}', dataset_files=patched_dataset_files)
    move_dataset(dataset_dir=args.dataset_dir, save_dir=f'{select_prefix}_{patch_prefix}_{bin_prefix}', dataset_files=selected_patches_dataset_files)

    print('Dataset Process Workflow Completed.')