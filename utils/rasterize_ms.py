import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyteomics import mzml, mzxml
from scipy import sparse
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm


def binning(mz_array, intensity_array, mz_min, mz_max, bin_size=0.01):
    """
    Binning the ms data.

    :param mz_array: (np.array) A list of m/z values.
    :param intensity_array: (np.array) A list of intensity values (same length as mz_list).
    :param mz_min: The minimum value of m/z bin.
    :param mz_max: The maximum value of m/z bin.
    :param bin_size: The Da of every bin in m/z binning.
    :return: A table of binned m/z and intensity.
    """
    mz_bins = (mz_max - mz_min) // bin_size
    bin_index = round((pd.Series(mz_array.tolist()) - mz_min) / bin_size)
    bin_table = pd.DataFrame({'index': bin_index, 'intensity': intensity_array.tolist()})
    bin_table['index'] = bin_table['index'].astype(int)
    bin_table = bin_table.groupby('index').sum()
    full_index = range(int(mz_bins))
    bin_table = bin_table.reindex(full_index)
    bin_table = bin_table.fillna(0)
    return bin_table['intensity']


# def parse_spec(spec, mz_min, mz_max, bin_size):
#     """
#     Parse the spectrum data.
#
#     :param spec: The mass spectrum data.
#     :param mz_min: The minimum value of m/z bin.
#     :param mz_max: The maximum value of m/z bin.
#     :param bin_size: The Da of every bin in m/z binning.
#     :return: A binned spectrum.
#     """
#     scan = spec.get('id', spec.get('num'))  # Support both mzML and mzXML
#     mz_array = spec['m/z array']
#     intensity_array = spec['intensity array']
#     bin_table = binning(mz_array=mz_array, intensity_array=intensity_array, mz_min=mz_min, mz_max=mz_max, bin_size=bin_size)
#     bin_table.columns = [scan]
#     return bin_table


def parse_spec(spec, mz_min, mz_max, bin_size):
    """
    Parse the spectrum data.

    :param spec: The mass spectrum data.
    :param mz_min: The minimum value of m/z bin.
    :param mz_max: The maximum value of m/z bin.
    :param bin_size: The Da of every bin in m/z binning.
    :return: A binned spectrum.
    """
    scan = spec.get('id', spec.get('num'))  # Support both mzML and mzXML
    mz_array = spec['m/z array']
    intensity_array = spec['intensity array']
    bin_spec = binning(mz_array=mz_array, intensity_array=intensity_array, mz_min=mz_min, mz_max=mz_max, bin_size=bin_size)
    bin_spec.name = scan
    return bin_spec


# def parse_ms(ms_file_paths, save_pattern, mz_min, mz_max, bin_size, threads=6):
#     """
#     Generate pseudo 2D MS images from raw MS data (.mzML, .mzXML).
#
#     :param ms_file_paths: The list of ms file paths.
#     :param save_pattern: The pattern to save the binned data.
#     :param mz_min: The minimum value of m/z bin.
#     :param mz_max: The maximum value of m/z bin.
#     :param bin_size: The Da of every bin in m/z binning.
#     :param threads: The number of threads to use for parallel processing.
#     """
#     with tqdm(total=len(ms_file_paths), desc="Rasterizing ms files") as progress_bar:
#         for file_path in ms_file_paths:
#             try:
#                 reader = mzml.read(file_path) if file_path.endswith('.mzML') else mzxml.read(file_path)
#             except Exception as e:
#                 print(f"Error reading file {file_path}: {e}")
#                 continue
#             file_name = os.path.splitext(os.path.basename(file_path))[0]
#             progress_bar.set_description(f'Binning {file_name}')
#
#             pool = ThreadPool(threads)
#             parse_spec_partial = partial(parse_spec, mz_min=mz_min, mz_max=mz_max, bin_size=bin_size)
#             pseudo_ms_image = pool.map(parse_spec_partial, list(reader))  # pseudo_ms_image: (scans, mz_bins)
#             pseudo_ms_image = np.array(pseudo_ms_image).T  # pseudo_ms_image: (mz_bins, scans)
#             pool.close()
#             pool.join()
#             sparse_table = sparse.csr_matrix(pseudo_ms_image, dtype=np.float32)
#
#             save_dir = os.path.dirname(file_path)
#             save_path = os.path.join(save_dir, f'{file_name}_{save_pattern}')
#             sparse.save_npz(save_path, sparse_table)
#
#             del pseudo_ms_image
#             del reader
#
#             progress_bar.update(1)


def parse_ms(ms_file_path, prefix, mz_min, mz_max, bin_size):
    try:
        reader = mzml.read(ms_file_path) if ms_file_path.endswith('.mzML') else mzxml.read(ms_file_path)

        pseudo_ms_image = []
        for spec in reader:
            binned_spec = parse_spec(spec, mz_min, mz_max, bin_size)
            pseudo_ms_image.append(binned_spec)

        pseudo_ms_image = pd.DataFrame(pseudo_ms_image).T  # pseudo_ms_image: (mz_bins, scans)
        pseudo_ms_image = pseudo_ms_image.to_numpy()

        sparse_table = sparse.csr_matrix(pseudo_ms_image, dtype=np.float32)

        save_dir = os.path.dirname(ms_file_path)
        file_name = os.path.splitext(os.path.basename(ms_file_path))[0]
        save_path = os.path.join(save_dir, f'{prefix}_{file_name}.npz')
        sparse.save_npz(save_path, sparse_table)
        del pseudo_ms_image
        del reader
        return True
    except Exception as e:
        print(f"Error processing file {ms_file_path}: {e}")
        return False


def parallel_parse_ms(ms_file_paths, prefix, mz_min, mz_max, bin_size, workers=4):
    """
    Generate pseudo 2D MS images from raw MS data (.mzML, .mzXML).

    :param ms_file_paths: The list of ms file paths.
    :param prefix: Prefix for the save path pattern.
    :param mz_min: The minimum value of m/z bin.
    :param mz_max: The maximum value of m/z bin.
    :param bin_size: The Da of every bin in m/z binning.
    :param workers: Number of worker processes to use.
    """
    with Pool(processes=workers) as pool:
        worker = partial(
            parse_ms,
            prefix=prefix,
            mz_min=mz_min,
            mz_max=mz_max,
            bin_size=bin_size
        )

        results = list(
            tqdm(
                pool.imap_unordered(worker, ms_file_paths),
                total=len(ms_file_paths),
                desc='Rasterizing ms files'
            )
        )

    success_rate = sum(results) / len(results)
    print(f"Binning completed. Success rate: {success_rate:.2%}")


def plot_pseudo_ms_image(npz_file):
    # Load the sparse matrix from the .npz file
    sparse_matrix = sparse.load_npz(npz_file)

    # Extract the row and column indices of non-zero entries
    row, col = sparse_matrix.nonzero()
    values = sparse_matrix.data

    # Create a scatter plot (for sparse visualization)
    plt.figure(figsize=(10, 8))
    plt.scatter(col, row, c=values, cmap='viridis', s=1)  # s=1 for smaller dots
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    mz_list = np.array([
        143.74027418, 150.9925338 , 151.78174021, 161.51779207,
        176.22627342, 176.77479698, 201.75184462, 201.86904981,
        209.48442371, 210.49347505, 220.05214306, 221.88865572,
        224.22467716, 235.54788739, 236.69232447, 255.55234014,
        261.51824175, 263.10981018, 263.28748065, 272.78656884,
        278.27743785, 291.20583488, 296.92432913, 297.38044699,
        314.03460473, 328.90537181, 335.54733192, 348.92650254,
        348.9832194 , 369.49242576, 370.22326722, 383.77748178,
        393.16018191, 398.63190298, 400.26834841, 405.50789388,
        419.77598208, 431.73188828, 444.91942081, 466.35765143,
        479.18245674, 487.89914362, 513.61660977, 540.39440224,
        551.10070782, 551.74007574, 554.95860704, 572.50813796,
        581.28123815, 581.84715279, 601.06164865, 602.06059471,
        630.5165327 , 631.54586323, 641.7856251 , 649.6637182 ,
        652.46919724, 657.06456439, 661.36643005, 669.1579834 ,
        699.32118188, 714.05810282, 720.55875055, 726.41439559,
        728.25356765, 730.79765544, 734.55925838, 744.10415428,
        753.75974441, 763.88643212, 766.27190607, 783.34424577,
        787.9463843 , 801.14475238, 801.27429824, 802.06243576,
        822.56153045, 841.22459703, 846.70326806, 847.30616612,
        855.76836041, 872.25481404, 873.2577998 , 875.06405461,
        888.92886337, 895.88151541, 900.30327408, 905.02660199,
        918.85893858, 922.11296019, 923.28431309, 934.72630109,
        944.29192037, 961.7115442 , 962.82779001, 964.86419938,
        967.66040633, 969.87087374, 979.34510262, 990.58998013
    ])

    intensity_list = np.array([
        636.76387802,  64.69331441, 184.5536952 , 751.88249241,
        695.61931486, 456.71462867, 219.30255848, 908.97351548,
        722.41516769, 959.57974631, 625.04714912, 661.61892058,
        845.46332086, 640.31551319,  79.09899505, 699.79881373,
        407.65690634, 291.90527877, 732.09108391, 876.44626575,
        138.03638975, 826.42378533, 433.0054348 , 322.08120791,
        343.45388458, 189.29455806, 639.47356192, 392.23795542,
        604.75994442, 984.3173989 , 573.82692741, 332.5388819 ,
        186.7562786 , 930.82408263,  15.18600489, 568.08673215,
        275.50442851, 224.71234321,  47.45981741, 689.92962118,
        976.67918562, 765.76983365, 141.6340282 , 367.23342983,
         76.32409516, 464.80325008, 105.37906295, 114.01364766,
        689.65679517, 892.29810342,  67.42138231, 556.94701237,
        725.24369081, 842.97675796, 844.35673195, 300.01272697,
        910.05613577, 300.3521806 , 616.70466038, 389.18590338,
        132.23159909, 970.15318857, 314.10143028,  40.63810692,
        189.98290339,  91.91524053, 497.84447831, 173.03095161,
        179.76884136, 918.70940137,  91.29111323, 828.24278707,
        705.1877406 , 590.68042172, 178.50707492, 168.15189549,
        165.22431304, 189.91652352, 530.61983984, 543.74540744,
        739.63273906, 630.34363622, 782.68609921, 158.87173575,
        727.12162414, 353.59864671, 265.87172696, 111.90089336,
        584.93469735, 993.82270122, 558.72659016, 897.68065255,
        616.82701754, 784.46943211, 937.47186443, 837.31333485,
        344.38563127, 567.53435092, 952.32179804, 451.78836852
    ])

    # print(binning(mz_list, intensity_list, 100, 900, 10))

    # parse_ms(
    #     ms_file_dir='./mzML',
    #     suffix='.mzML',
    #     mz_min=200,
    #     mz_max=1100,
    #     bin_size=0.01
    # )

