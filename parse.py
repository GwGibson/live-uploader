import numba as nb
import numpy as np


@nb.njit
def create_amp(I, Q):
    return np.sqrt(I**2 + Q**2)


@nb.njit
def create_phase(I, Q):
    return np.arctan2(Q, I)


class DataProcessingError(Exception):
    pass


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


def validate_data_format(data):
    if not isinstance(data, np.ndarray):
        raise DataProcessingError("Data must be a numpy array")
    if data.ndim != 2:
        raise DataProcessingError("Data must be a 2D array")
    if len(data) < 2:  # At least timestamps and one row of data
        raise DataProcessingError(
            "Data must contain timestamps and at least one row of measurements"
        )


def process_raw_data(data):
    validate_data_format(data)
    try:
        timestamps = data[0].real
        complex_data = data[1:]
        return timestamps, complex_data
    except Exception as e:
        raise DataProcessingError(f"Failed to process data: {str(e)}") from e


def process_measurements(
    complex_data,
    transform_func,
    center=True,
    filter_outliers=False,
    threshold=3.0,
    scale_factor=0.6745,
):
    """
    Process complex measurements with optional centering and outlier capping.

    Parameters:
        complex_data (numpy.ndarray): Complex input data
        transform_func (callable): Function to transform complex data (e.g., create_amp or create_phase)
        center (bool, optional): Whether to center the measurements by subtracting the mean. Defaults to True.
        filter_outliers (bool, optional): Whether to cap outliers at valid min/max values. Defaults to False.
        threshold (float, optional): Z-score threshold for outlier detection. Defaults to 3.0.
        scale_factor (float, optional): Scale factor for MAD normalization. Defaults to 0.6745.

    Returns:
        numpy.ndarray: Processed measurements with outliers capped at valid min/max values
    """
    try:
        measurements = transform_func(complex_data.real, complex_data.imag)

        # Center the measurements if requested
        if center:
            sensor_means = np.mean(measurements, axis=1, keepdims=True)
            measurements = measurements - sensor_means

        # Cap outliers if requested
        if filter_outliers:
            # Calculate medians and MADs across time axis (axis=1)
            medians = np.median(measurements, axis=1, keepdims=True)
            mads = np.median(np.abs(measurements - medians), axis=1, keepdims=True)
            mads[mads == 0] = np.inf  # Handle zero MAD case (unlikely)
            robust_z_scores = scale_factor * (measurements - medians) / mads
            valid_mask = np.abs(robust_z_scores) <= threshold

            # For each sensor, find valid min and max values
            for i, measurement in enumerate(measurements):
                valid_values = measurement[valid_mask[i]]
                if len(valid_values) > 0:  # Only proceed if we have valid values
                    valid_min = np.min(valid_values)
                    valid_max = np.max(valid_values)
                    # Cap the values outside the valid range
                    measurements[i] = np.clip(measurement, valid_min, valid_max)

        return measurements

    except Exception as e:
        raise DataProcessingError(f"Failed to process measurements: {str(e)}") from e


def get_amplitudes(
    data, center=False, filter_outliers=False, threshold=3.0, scale_factor=0.674
):
    timestamps, complex_data = process_raw_data(data)
    amplitudes = process_measurements(
        complex_data,
        create_amp,
        center=center,
        filter_outliers=filter_outliers,
        threshold=threshold,
        scale_factor=scale_factor,
    )
    return amplitudes, timestamps


def get_phases(
    data, center=False, filter_outliers=False, threshold=3.0, scale_factor=0.674
):
    timestamps, complex_data = process_raw_data(data)
    phases = process_measurements(
        complex_data,
        create_phase,
        center=center,
        filter_outliers=filter_outliers,
        threshold=threshold,
        scale_factor=scale_factor,
    )
    return phases, timestamps


def main():
    raw_data = np.load("data/test_tstream_1736805171_001.npy")
    amplitudes, timestamps = get_phases(raw_data)

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
