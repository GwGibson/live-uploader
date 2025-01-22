import numba as nb
import numpy as np


@nb.njit
def create_amp(I, Q):
    return np.sqrt(I**2 + Q**2)


@nb.njit
def create_phase(I, Q):
    return np.arctan2(Q, I)


def calculate_filtered_min_max(data_arrays, axis, threshold=3.0, scale_factor=0.6745):
    """
    Calculate the filtered minimum and maximum values for each NumPy array in a list
    along a specified axis, excluding outliers determined by the robust Z-Score method.

    Params:
        data_arrays (list of numpy.ndarray): A list of NumPy arrays where each array,
            along the specified axis, will be processed to find the filtered min and max values.
        axis (int): The axis along which to calculate the min and max values.
        threshold (float, optional): The Z-Score threshold to determine outliers.
            The default is 3.0, corresponding to about three standard deviations in
            a normal distribution Lowering the threshold increases the sensitivity
            of outlier detection, potentially identifying more points as outliers.
            Raising the threshold decreases sensitivity, allowing more variability
            in the data without classifying points as outliers.
        scale_factor (float, optional): Scale factor to normalize the MAD to the standard
            deviation of a normally distributed dataset. Keep the scale factor at 0.6745 for
            normal-like distributions, or slightly increase it if the data has heavier tails.
    Raises:
        ValueError: If all data points in any array are considered outliers,
            indicating no valid data to calculate min and max.

    Returns:
        tuple:
            - numpy.ndarray: An array with the minimum values of each array in data_arrays
                along the specified axis, excluding outliers.
            - numpy.ndarray: An array with the maximum values of each array in data_arrays
                along the specified axis, excluding outliers.
    """

    # Calculate medians and MADs across the specified axis
    medians = np.median(data_arrays, axis=axis, keepdims=True)
    mads = np.median(np.abs(data_arrays - medians), axis=axis, keepdims=True)
    # Handle the case where MAD is zero (highly unlikely)
    mads[mads == 0] = np.inf
    robust_z_scores = scale_factor * (data_arrays - medians) / mads
    valid_mask = np.abs(robust_z_scores) <= threshold
    valid_data = np.where(valid_mask, data_arrays, np.nan)
    return np.nanmin(valid_data, axis=axis), np.nanmax(valid_data, axis=axis)


def get_amplitudes(path):
    data = np.load(path)
    timestamps = data[0].real
    complex_data = data[1:]
    amplitudes = create_amp(complex_data.real, complex_data.imag)
    sensor_means = np.mean(amplitudes, axis=1, keepdims=True)
    amplitudes = amplitudes - sensor_means
    # Trimming after processing here to ensure spikes are more visible
    start_idx = len(amplitudes[0]) // 4  # Start at ~ 15s
    amplitudes = amplitudes[:, start_idx:]
    timestamps = timestamps[start_idx:]
    return amplitudes, timestamps


def main():
    amplitudes, timestamps = get_amplitudes("data/test_tstream_1736805171_001.npy")

    global_min = np.min(amplitudes)
    global_max = np.max(amplitudes)

    sensor_min, time_min = np.unravel_index(np.argmin(amplitudes), amplitudes.shape)
    sensor_max, time_max = np.unravel_index(np.argmax(amplitudes), amplitudes.shape)

    start_time = timestamps[0]
    relative_times = timestamps - start_time

    print(
        f"Global min: {global_min:.2f} from sensor {sensor_min} at {relative_times[time_min]:.2f} seconds"
    )
    print(
        f"Global max: {global_max:.2f} from sensor {sensor_max} at {relative_times[time_max]:.2f} seconds"
    )


if __name__ == "__main__":
    main()
