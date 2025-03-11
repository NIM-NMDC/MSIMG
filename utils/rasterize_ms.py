import os
import math
import numpy as np
import pyopenms
import matplotlib.pyplot as plt


def rasterize_ms_to_image(ms_file_path, image_path=None, image_size=(500, 500)):
    """
    Rasterize mass spectrometry data to an image representation.

    :param ms_file_path: Path to the mass spectrometry file.
    :param image_path: Path where the output image will be saved.
    :param image_size: Tuple representing the size of the output image (default is (1000, 1000)).
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

    image = np.zeros(image_size)

    # Set the retention time and m/z ranges.
    rt_scale = np.linspace(min_rt, max_rt, image_size[0])  # y-axis (Retention Time)
    mz_scale = np.linspace(min_mz, max_mz, image_size[1])  # x-axis (M/Z)

    for i in range(len(rt_list)):
        rt = rt_list[i]
        mz_array = mz_arrays[i]
        intensity_array = intensity_arrays[i]

        # Filter out peaks where intensity is zero.
        non_zero_idx = intensity_array > 0
        mz_array = mz_array[non_zero_idx]
        intensity_array = intensity_array[non_zero_idx]

        # Skip if there are no peaks with non-zero intensity.
        if len(mz_array) == 0:
            continue

        # Find the corresponding retention time and m/z indices.
        y_idx = np.argmin(np.abs(rt_scale - rt))
        x_idx = np.digitize(mz_array, mz_scale) - 1

        # Map the intensity values to the image.
        for j in range(len(mz_array)):
            # Normalize intensity values
            gamma = 0.3
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
