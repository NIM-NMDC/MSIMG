import os
import math
import numpy as np
import pyopenms
import matplotlib.pyplot as plt


def rasterize_ms_to_image(ms_file_path, image_path=None, image_size=(500, 500), start_mz=None, end_mz=None, rt_limit=None):
    """
    Rasterize mass spectrometry data (MS1) to an image representation.

    :param ms_file_path: Path to the mass spectrometry file.
    :param image_path: Path where the output image will be saved.
    :param image_size: Tuple representing the size of the output image (default is (1000, 1000)).
    :param start_mz: Optional float, the lower bound of m/z range for imaging.
    :param end_mz: Optional float, the upper bound of m/z range for imaging.
    :param rt_limit: Optional float, if provided, spectra with rt > rt_limit will be skipped.
    :return: None
    """
    # Load the mass spectrometry file.
    exp = pyopenms.MSExperiment()
    file_suffix = ms_file_path.split('.')[-1]
    if file_suffix == 'mzML':
        pyopenms.MzMLFile().load(ms_file_path, exp)
    elif file_suffix == 'mzXML':
        pyopenms.MzXMLFile().load(ms_file_path, exp)
    else:
        raise ValueError(f'Unsupported file format: {file_suffix}')

    # Initiative variables to store the min and max values for RT and M/Z
    min_rt = np.inf
    max_rt = -np.inf
    min_mz = np.inf
    max_mz = -np.inf
    max_intensity = -np.inf

    rt_list = []
    mz_arrays, intensity_arrays = [], []

    exp.sortSpectra(True)

    # Iterate over all spectra in the experiment to find the min and max values for RT and M/Z
    for spectrum in exp.getSpectra():
        if spectrum.getMSLevel() == 1:
            rt = spectrum.getRT()

            if rt_limit and rt > rt_limit:
                continue

            mz_array, intensity_array = spectrum.get_peaks()

            if len(mz_array) == 0:
                continue

            # Update the min and max values
            min_rt = min(min_rt, rt)
            max_rt = max(max_rt, rt)
            min_mz = min(min_mz, np.min(mz_array))
            max_mz = max(max_mz, np.max(mz_array))
            max_intensity = max(max_intensity, np.max(intensity_array))

            rt_list.append(rt)
            mz_arrays.append(mz_array)
            intensity_arrays.append(intensity_array)

    if len(rt_list) == 0:
        print('No MS1 spectra in the selected RT range (or file is empty).')
        return

    if start_mz:
        min_mz = max(min_mz, start_mz)

    if end_mz:
        max_mz = min(max_mz, end_mz)

    image = np.zeros(image_size)
    # Set the retention time and m/z ranges.
    rt_scale = np.linspace(min_rt, max_rt, image_size[0])  # y-axis (Retention Time)
    mz_scale = np.linspace(min_mz, max_mz, image_size[1])  # x-axis (M/Z)

    for i in tqdm(range(len(rt_list))):
        rt = rt_list[i]
        mz_array = mz_arrays[i]
        intensity_array = intensity_arrays[i]

        # Skip if there are no peaks with non-zero intensity.
        if len(intensity_array) == 0:
            continue

        valid_idx = (mz_array >= min_mz) & (mz_array <= max_mz) & (intensity_array > 0)
        mz_array = mz_array[valid_idx]
        intensity_array = intensity_array[valid_idx]

        # Find the corresponding retention time and m/z indices.
        y_idx = np.argmin(np.abs(rt_scale - rt))
        x_idx = np.digitize(mz_array, mz_scale) - 1

        # Map the intensity values to the image.
        for j in range(len(mz_array)):
            # Normalize intensity values
            gamma = 0.15
            intensity_scaled = np.power(intensity_array[j] / max_intensity, gamma)
            image[y_idx, x_idx[j]] = intensity_scaled  # In grayscale, 0 is black and 1 is white.

    plt.imshow(image, cmap='gray', aspect='auto', origin='lower', extent=(min_mz, max_mz, min_rt, max_rt))
    plt.axis('off')
    # Remove any whitespace around the image.
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    if image_path:
        # Save image with no extra padding.
        plt.savefig(image_path, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.show()
    plt.close()


if __name__ == '__main__':

    from tqdm import tqdm

    ms_files_dir = r'E:\msdata\ST000923\HMP2_C8-pos\C8p_rawData\test'
    ms_files_path = []
    if os.path.exists(ms_files_dir):
        for root, dirs, files in os.walk(ms_files_dir):
            for file in files:
                if file.endswith('.mzML'):
                    file_path = os.path.join(root, file)
                    ms_files_path.append(file_path)

    for ms_file_path in ms_files_path:
        image_path = os.path.splitext(ms_file_path)[0] + '.png'
        rasterize_ms_to_image(ms_file_path, image_path=image_path, image_size=(500, 500), start_mz=300, end_mz=600, rt_limit=600)