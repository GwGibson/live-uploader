import numba as nb
import numpy as np


@nb.njit
def create_amp(I, Q):
    return np.sqrt(I**2 + Q**2)


def get_amplitudes(path):
    data = np.load(path)
    timestamps = data[0].real
    complex_data = data[1:]
    amplitudes = create_amp(complex_data.real, complex_data.imag)
    sensor_means = np.mean(amplitudes, axis=1, keepdims=True)
    amplitudes = amplitudes - sensor_means
    # Trimming after processing here to ensure spikes are more visible
    start_idx = len(amplitudes[0])//4 # Start at ~ 15s
    amplitudes = amplitudes[:, start_idx:]
    timestamps = timestamps[start_idx:]
    return amplitudes, timestamps


def main():
    amplitudes, timestamps = get_amplitudes("data/test_tstream_1736805171_001")

    sensor_means = np.mean(amplitudes, axis=1, keepdims=True)
    amplitudes = amplitudes - sensor_means

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
