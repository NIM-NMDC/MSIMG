import os
import shutil
import subprocess

from pathlib import Path

# Set PYTHONPATH
project_root = r"D:\Codes\MSIMG"
os.environ["PYTHONPATH"] = project_root

patch_strategy = 'ics'
score_strategy = 'entropy'
top_k = 4
step = 'patch_selection'


def move_dataset_files(src_dir, dest_dir):
    """
    Move all subdirectories and files from src_dir to dst_dir.
    """
    if not os.path.exists(src_dir):
        print(f"Source directory {src_dir} does not exist.")
        return
    if not os.path.isdir(src_dir):
        print(f"Source path {src_dir} is not a directory.")
        return

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
        print(f"Destination directory {dest_dir} created.")
    elif not os.path.isdir(dest_dir):
        print(f"Destination path {dest_dir} is not a directory.")
        return

    print(f"Moving subdirectories and files from {src_dir} to {dest_dir}...")

    try:
        items = os.listdir(src_dir)
    except OSError as e:
        print(f"Error accessing source directory: {e}")
        return

    moved_count = 0
    skipped_count = 0
    errors_during_move = False

    for item_name in items:
        source_item_path = os.path.join(src_dir, item_name)
        dest_item_path = os.path.join(dest_dir, item_name)

        if os.path.isdir(source_item_path):
            try:
                shutil.move(source_item_path, dest_item_path)
                print(f"Moved directory: {source_item_path} to {dest_item_path}")
                moved_count += 1
            except shutil.Error as e:
                print(f"Shutil Error moving directory {source_item_path}: {e}")
                errors_during_move = True
            except Exception as e:
                print(f"Error moving directory {source_item_path}: {e}")
                errors_during_move = True
        else:
            print(f"Skipped non-directory item: {source_item_path}")
            skipped_count += 1

    print("-" * 20)
    print(f"Moved {moved_count} directories.")
    print(f"Skipped {skipped_count} non-directory items.")

    print(f"Deleting source directory: {src_dir}...")
    try:
        shutil.rmtree(src_dir)
        print(f"Source directory {src_dir} deleted.")
    except OSError as e:
        print(f"Error removing source directory: {e}")

    if errors_during_move:
        print("Some errors occurred during the move operation. Source directory may not be completely moved.")
    else:
        print("All directories moved successfully.")

commands = {
    # 'ST000923-C8-pos': fr'python dataset_process_workflow.py --dataset_dir E:\msdata\ST000923\HMP2_C8-pos\C8p_rawData --suffix .mzML --mz_min 200 --mz_max 1100 --bin_size 0.01 --num_workers 1 --patch_strategy {patch_strategy} --score_strategy {score_strategy} --top_k {top_k} --step {step}',
    # 'ST000923-C18-neg': fr'python dataset_process_workflow.py --dataset_dir E:\msdata\ST000923\HMP2_C18-neg\C18n_rawData --suffix .mzML --mz_min 70 --mz_max 850 --bin_size 0.01 --num_workers 4 --step patch_selection --select_method {select_method} --top_k {top_k}',
    # 'ST000923-HILIC-pos': fr'python dataset_process_workflow.py --dataset_dir E:\msdata\ST000923\HMP2_HILIC-pos\HILp_rawData --suffix .mzML --mz_min 70 --mz_max 800 --bin_size 0.01 --num_workers 4 --step patch_selection --select_method {select_method} --top_k {top_k}',
    # 'ST000923-HILIC-neg': fr'python dataset_process_workflow.py --dataset_dir E:\msdata\ST000923\HMP2_HILIC-neg\HILn_rawData --suffix .mzML --mz_min 70 --mz_max 750 --bin_size 0.01 --num_workers 4 --step patch_selection --select_method {select_method} --top_k {top_k}',
    # 'ST001000-C8-pos': fr'python dataset_process_workflow.py --dataset_dir E:\msdata\ST001000\C8-pos --suffix .mzML --mz_min 200 --mz_max 1100 --bin_size 0.01 --num_workers 4 --step patch_selection --select_method {select_method} --top_k {top_k}',
    # 'ST001000-C18-neg': fr'python dataset_process_workflow.py --dataset_dir E:\msdata\ST001000\C18-neg --suffix .mzML --mz_min 70 --mz_max 650 --bin_size 0.01 --num_workers 4 --step patch_selection --select_method {select_method} --top_k {top_k}',
    # 'ST001000-HILIC-pos': fr'python dataset_process_workflow.py --dataset_dir E:\msdata\ST001000\HILIC-pos --suffix .mzML --mz_min 70 --mz_max 800 --bin_size 0.01 --num_workers 4 --step patch_selection --select_method {select_method} --top_k {top_k}',
    # 'ST001000-HILIC-neg': fr'python dataset_process_workflow.py --dataset_dir E:\msdata\ST001000\HILIC-neg --suffix .mzML --mz_min 70 --mz_max 750 --bin_size 0.01 --num_workers 4 --step patch_selection --select_method {select_method} --top_k {top_k}',
    # 'ST003161': fr'python dataset_process_workflow.py --dataset_dir E:\msdata\ST003161_Rawfiles\SOURCE_Israel_Stool_2022_mzml --suffix .mzML --mz_min 65 --mz_max 1010 --bin_size 0.01 --num_workers 4  --patch_strategy {patch_strategy} --score_strategy {score_strategy} --top_k {top_k} --step {step}',
    'ST003313': fr'python dataset_process_workflow.py --dataset_dir E:\msdata\ST003313_Rawdata\mzml_stool --suffix .mzML --mz_min 65 --mz_max 1010 --bin_size 0.01 --num_workers 1 --patch_strategy {patch_strategy} --score_strategy {score_strategy} --top_k {top_k} --step {step}',
    # 'PXD010371': fr'python dataset_process_workflow.py --dataset_dir E:\msdata\PXD010371\mzML --suffix .mzML --mz_min 350 --mz_max 2000 --bin_size 0.01 --num_workers 4 --step patch_selection --select_method {select_method} --top_k {top_k}',
    # 'MSV000089237': fr'python dataset_process_workflow.py --dataset_dir E:\msdata\MSV000089237\mzML --suffix .mzML --mz_min 330 --mz_max 1650 --bin_size 0.01 --num_workers 4 --step patch_selection --select_method {select_method} --top_k {top_k}',
}


for i, key in enumerate(commands.keys(), 1):
    cmd = commands[key]
    print(f"\n===> Running command {i}/{len(commands)}:\n{cmd}\n")
    result = subprocess.run(cmd, shell=True)
    if result.returncode == 0:
        print(f"[✓] Command {i} finished successfully.\n")
    else:
        print(f"[✗] Command {i} failed with return code {result.returncode}.\n")


# if select_method != 'random':
#     dir_prefix = f'{select_method}_top_{top_k}'
# else:
#     dir_prefix = f'{select_method}_{top_k}'
# source_dir_dict = {
#     'ST000923-C8-pos': fr'E:\msdata\ST000923\HMP2_C8-pos\C8p_rawData\{dir_prefix}_patch_224x224_overlap_0x0_mz_200.0-1100.0_bin_size_0.01',
#     'ST000923-C18-neg': fr'E:\msdata\ST000923\HMP2_C18-neg\C18n_rawData\{dir_prefix}_patch_224x224_overlap_0x0_mz_70.0-850.0_bin_size_0.01',
#     'ST000923-HILIC-pos': fr'E:\msdata\ST000923\HMP2_HILIC-pos\HILp_rawData\{dir_prefix}_patch_224x224_overlap_0x0_mz_70.0-800.0_bin_size_0.01',
#     'ST000923-HILIC-neg': fr'E:\msdata\ST000923\HMP2_HILIC-neg\HILn_rawData\{dir_prefix}_patch_224x224_overlap_0x0_mz_70.0-750.0_bin_size_0.01',
#     'ST001000-C8-pos': fr'E:\msdata\ST001000\C8-pos\{dir_prefix}_patch_224x224_overlap_0x0_mz_200.0-1100.0_bin_size_0.01',
#     'ST001000-C18-neg': fr'E:\msdata\ST001000\C18-neg\{dir_prefix}_patch_224x224_overlap_0x0_mz_70.0-650.0_bin_size_0.01',
#     'ST001000-HILIC-pos': fr'E:\msdata\ST001000\HILIC-pos\{dir_prefix}_patch_224x224_overlap_0x0_mz_70.0-800.0_bin_size_0.01',
#     'ST001000-HILIC-neg': fr'E:\msdata\ST001000\HILIC-neg\{dir_prefix}_patch_224x224_overlap_0x0_mz_70.0-750.0_bin_size_0.01',
#     'ST003161': fr'E:\msdata\ST003161_Rawfiles\SOURCE_Israel_Stool_2022_mzml\{dir_prefix}_patch_224x224_overlap_0x0_mz_65.0-1010.0_bin_size_0.01',
#     'ST003313': fr'E:\msdata\ST003313_Rawdata\mzml_stool\{dir_prefix}_patch_224x224_overlap_0x0_mz_65.0-1010.0_bin_size_0.01',
#     'PXD010371': fr'E:\msdata\PXD010371\mzML\{dir_prefix}_patch_224x224_overlap_0x0_mz_350.0-2000.0_bin_size_0.01',
#     'MSV000089237': fr'E:\msdata\MSV000089237\mzML\{dir_prefix}_patch_224x224_overlap_0x0_mz_330.0-1650.0_bin_size_0.01',
# }
#
# destination_dir_dict = {
#     'ST000923-C8-pos': fr'E:\IBD_2D\{dir_prefix}_patch_224x224_overlap_0x0_bin_size_0.01\ST000923-C8-pos',
#     'ST000923-C18-neg': fr'E:\IBD_2D\{dir_prefix}_patch_224x224_overlap_0x0_bin_size_0.01\ST000923-C18-neg',
#     'ST000923-HILIC-pos': fr'E:\IBD_2D\{dir_prefix}_patch_224x224_overlap_0x0_bin_size_0.01\ST000923-HILIC-pos',
#     'ST000923-HILIC-neg': fr'E:\IBD_2D\{dir_prefix}_patch_224x224_overlap_0x0_bin_size_0.01\ST000923-HILIC-neg',
#     'ST001000-C8-pos': fr'E:\IBD_2D\{dir_prefix}_patch_224x224_overlap_0x0_bin_size_0.01\ST001000-C8-pos',
#     'ST001000-C18-neg': fr'E:\IBD_2D\{dir_prefix}_patch_224x224_overlap_0x0_bin_size_0.01\ST001000-C18-neg',
#     'ST001000-HILIC-pos': fr'E:\IBD_2D\{dir_prefix}_patch_224x224_overlap_0x0_bin_size_0.01\ST001000-HILIC-pos',
#     'ST001000-HILIC-neg': fr'E:\IBD_2D\{dir_prefix}_patch_224x224_overlap_0x0_bin_size_0.01\ST001000-HILIC-neg',
#     'ST003161': fr'E:\IBD_2D\{dir_prefix}_patch_224x224_overlap_0x0_bin_size_0.01\ST003161',
#     'ST003313': fr'E:\IBD_2D\{dir_prefix}_patch_224x224_overlap_0x0_bin_size_0.01\ST003313',
#     'PXD010371': fr'E:\IBD_2D\{dir_prefix}_patch_224x224_overlap_0x0_bin_size_0.01\PXD010371',
#     'MSV000089237': fr'E:\IBD_2D\{dir_prefix}_patch_224x224_overlap_0x0_bin_size_0.01\MSV000089237',
# }

# for key in source_dir_dict.keys():
#     source_dir = source_dir_dict[key]
#     dest_dir = destination_dir_dict[key]
#     move_dataset_files(source_dir, dest_dir)
