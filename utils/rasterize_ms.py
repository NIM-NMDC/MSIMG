import os
import numpy as np
import pyopenms
import matplotlib.pyplot as plt

from tqdm import tqdm

from file_utils import get_files_path


def rasterize_ms_to_image(ms_file_path, image_path=None, image_size=(500, 500), start_mz=None, end_mz=None, start_rt=None, end_rt=None):
    """
    Rasterize mass spectrometry data (MS1) to an image representation.

    :param ms_file_path: Path to the mass spectrometry file.
    :param image_path: Path where the output image will be saved.
    :param image_size: Tuple representing the size of the output image (default is (1000, 1000)).
    :param start_mz: Optional float, the lower bound of m/z range for imaging.
    :param end_mz: Optional float, the upper bound of m/z range for imaging.
    :param start_rt: Optional float, the lower bound of RT range for imaging.
    :param end_rt: Optional float, the upper bound of RT range for imaging.
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

            if start_rt and rt < start_rt:
                continue
            if end_rt and rt > end_rt:
                continue

            mz_array, intensity_array = spectrum.get_peaks()

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
    # Set the m/z ranges and retention time.
    mz_scale = np.linspace(min_mz, max_mz, image_size[1])  # x-axis (M/Z)
    rt_scale = np.linspace(min_rt, max_rt, image_size[0])  # y-axis (Retention Time)

    for i in tqdm(range(len(rt_list))):
        rt = rt_list[i]
        mz_array = mz_arrays[i]
        intensity_array = intensity_arrays[i]

        valid_idx = (mz_array >= min_mz) & (mz_array <= max_mz) & (intensity_array > 0)
        mz_array = mz_array[valid_idx]
        intensity_array = intensity_array[valid_idx]

        # Find the corresponding retention time and m/z indices.
        x_idx = np.digitize(mz_array, mz_scale) - 1
        y_idx = np.argmin(np.abs(rt_scale - rt))

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
    # plt.show()
    plt.close()


if __name__ == '__main__':

    def get_int_input(prompt, default):
        while True:
            user_input = input(prompt)
            if user_input.strip() == '':
                return default
            try:
                value = int(user_input)
                return value
            except ValueError:
                print('Invalid input. Please enter an integer value.')

    ms_files_dir = input('Please input the directory of mass spectrometry files: ')
    suffix = input('Please input the file suffix (e.g., .mzML, .raw): ')
    if not suffix.startswith('.'):
        suffix = '.' + suffix
    image_size_width = get_int_input('Please input the width of the image (default 500): ', 500)
    image_size_height = get_int_input('Please input the height of the image (default 500): ', 500)
    image_size = (image_size_width, image_size_height)

    if os.path.exists(ms_files_dir):
        ms_files_path = get_files_path(base_dir=ms_files_dir, suffix=suffix)

        for ms_file_path in ms_files_path:
            images_dir = os.path.join(ms_files_dir, 'images')
            os.makedirs(images_dir, exist_ok=True)
            image_path = os.path.join(images_dir, os.path.splitext(os.path.basename(ms_file_path))[0] + '.png')
            rasterize_ms_to_image(ms_file_path=ms_file_path, image_path=image_path, image_size=image_size)
    else:
        raise FileNotFoundError(f"Directory {ms_files_dir} not found.")