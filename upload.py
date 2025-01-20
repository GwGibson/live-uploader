import argparse
from enum import Enum
import numpy as np
from live_uploader import LiveUploader
from parse import get_amplitudes


class UploadType(Enum):
    TEST = 0


def random_datastream_data_dict():
    num_channels = 11000
    num_measurements = 2000

    return {
        "RANDOM": (
            # 2D -> each channel has a number of measurements
            # i.e. Time series data for each channel
            np.random.rand(num_channels, num_measurements),
        )
    }


def test_datastream_data_dict():
    amplitudes, _ = get_amplitudes("data/test_tstream_1736805171_001.npy")

    return {"TEST": (amplitudes)}


def start_live_upload(upload_type):
    live_uploader = LiveUploader()

    database_name = "ocs_feeds"
    measurement_name = "LIVE_MEASUREMENTS"
    host = "localhost"
    port = 8086

    if upload_type == UploadType.TEST:
        datastream_data_dict = test_datastream_data_dict()
    else:
        raise ValueError(f"Upload type: {upload_type} is an invalid upload type.")

    live_uploader.set_database_details(database_name, measurement_name, host, port)
    # Assumes same number of measurements across all channels
    num_data_points = len(datastream_data_dict[next(iter(datastream_data_dict))][0])
    # In seconds
    upload_interval = 0.2
    # Actual spacing between data points
    data_point_interval = upload_interval / 2
    # January 22, 2022 at 15:30:00 UTC
    # start_time = datetime(2022, 1, 22, 15, 30, 0, tzinfo=timezone.utc)
    # TODO: Add option to use actual timesteps here.
    live_uploader.set_timestamps(num_data_points, data_point_interval)
    live_uploader.clear_measurements()
    live_uploader.upload(datastream_data_dict, upload_interval)


def parse_upload_type(value):
    try:
        return UploadType(int(value))
    except ValueError:
        try:
            return UploadType[value.upper()]
        except KeyError as exc:
            raise argparse.ArgumentTypeError(f"Invalid upload type: {value}") from exc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Upload data to the live uploader.",
        epilog="Example usage: python upload.py 0",
    )
    parser.add_argument(
        "upload_type",
        type=parse_upload_type,
    )
    args = parser.parse_args()
    start_live_upload(args.upload_type)
