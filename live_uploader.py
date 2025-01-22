from datetime import datetime, timedelta, timezone
import time
from colorama import Fore
import numpy as np
from influxdb import InfluxDBClient
from influxdb.exceptions import InfluxDBClientError
from tqdm import tqdm


class LiveUploader:
    def __init__(self, influxdb_client=None):
        self.influxdb_client = influxdb_client if influxdb_client else InfluxDBClient
        self.timestamps_set = False
        self.database_details_set = False
        self.database_cleared = False
        self.pause_requested = False
        self.resume_requested = False

        self.logger = None
        self.timestamps = None

        self.database_name = None
        self.measurement_name = None
        self.host = None
        self.port = None

        self.current_index = 0
        self._last_processed_index = 0

        # Members for upload state
        self.client = None
        self.processed_data = None
        self.points_per_upload = None
        self.upload_interval = None
        self.initialized = False

    @property
    def current_index(self):
        return self._current_index

    @current_index.setter
    def current_index(self, value):
        self._current_index = value
        # Reset the last processed index when position changes
        self._last_processed_index = value

    def pause_upload(self):
        self.pause_requested = True
        self.resume_requested = False

    def resume_upload(self):
        self.resume_requested = True
        self.pause_requested = False

    def _initialize_upload(self, datastream, upload_interval):
        """Initialize upload parameters if not already initialized."""
        if not self.initialized:
            if not self.timestamps_set:
                raise RuntimeError("Timestamps not set.")
            if not self.database_details_set:
                raise RuntimeError("Database details not set.")
            if not self.timestamps or len(self.timestamps) <= 1:
                raise ValueError("Timestamps list must contain at least two elements.")

            # Calculate intervals and process data
            data_point_interval = (
                self.timestamps[1] - self.timestamps[0]
            ).total_seconds()
            if data_point_interval > upload_interval:
                raise ValueError("Data point frequency must be <= to upload frequency.")

            # Initialize client
            self.client = self.influxdb_client(self.host, self.port)
            self.client.switch_database(self.database_name)

            # Store upload parameters
            self.upload_interval = upload_interval
            self.points_per_upload = int(upload_interval / data_point_interval)
            self.processed_data = self._preprocess_data(
                datastream.data, self.points_per_upload
            )
            self.initialized = True

    def upload(self, datastream, upload_interval, progress_callback=None):
        """Start continuous upload process."""
        self._initialize_upload(datastream, upload_interval)

        if not self.database_cleared:
            print("Warning: Database not cleared before upload.")
        self.database_cleared = False

        self._perform_live_upload(progress_callback)

    def upload_single_point(self, progress_callback=None):
        """Upload a single point at the current index."""
        if not self.initialized:
            return

        def single_point_progress(start_index, end_index):
            if progress_callback:
                message = f"Uploading point {self.current_index}"
                progress_callback(message, self.current_index)

        self._process_single_upload(single_point_progress)

    def _perform_live_upload(self, progress_callback=None):
        self.pause_requested = False
        self.resume_requested = False
        total_points = len(self.timestamps)

        if progress_callback is None:
            with tqdm(
                total=total_points,
                bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.CYAN, Fore.RESET),
                ascii=False,
                dynamic_ncols=True,
            ) as pbar:
                self._process_uploads(
                    progress_handler=lambda start_index, end_index: (
                        pbar.set_description(
                            f"Uploading the mean of {end_index - start_index} data point(s) "
                            f"from {self.timestamps[start_index]} to {self.timestamps[end_index-1]}"
                        ),
                        pbar.update(end_index - start_index),
                    ),
                )
        else:
            def progress_handler(start_index, end_index):
                message = (
                    f"Uploading data points {start_index} to {end_index-1} of {total_points} "
                    f"({(end_index/total_points*100):.1f}%) - "
                    f"From {self.timestamps[start_index]} to {self.timestamps[end_index-1]}"
                )
                progress_callback(message, start_index)

            self._process_uploads(progress_handler=progress_handler)

    def _process_uploads(self, progress_handler):
        total_points = len(self.timestamps)

        while self._current_index < total_points:
            while self.pause_requested and not self.resume_requested:
                time.sleep(0.1)
                if self._current_index != self._last_processed_index:
                    self._last_processed_index = self._current_index

            start_time = datetime.now()
            self._process_single_upload(progress_handler)
            self._sleep_until_next_interval(start_time, self.upload_interval)

    def _process_single_upload(self, progress_handler):
        """Process a single upload chunk at the current index."""
        total_points = len(self.timestamps)

        aligned_position = (
            self._current_index // self.points_per_upload
        ) * self.points_per_upload
        end_index = min(aligned_position + self.points_per_upload, total_points)

        data_index = aligned_position // self.points_per_upload
        if data_index < len(self.processed_data):
            progress_handler(aligned_position, end_index)

            avg_timestamp_ns = self._calculate_average_timestamp(
                aligned_position, end_index
            )
            avg_data = self.processed_data[data_index]
            live_data_points = self._create_live_data_points(
                avg_data,
                avg_timestamp_ns,
            )

            self._write_live_data_points(self.client, live_data_points)

            if self._current_index == self._last_processed_index:
                self._current_index = end_index
                self._last_processed_index = self._current_index

    def set_actual_timestamps(self, timestamps):
        self.timestamps_set = True
        self.timestamps = timestamps

    def set_timestamps(
        self,
        num_data_points,
        data_point_interval,
        start_date=datetime.now(timezone.utc),
    ):
        """
        Sets up timestamps based on the number of data points, their frequency,
        and the starting date. Must be called prior to uploading data to the
        database.

        Params:
            num_data_points (int): The number of data points for which
                timestamps are generated and should match the number of
                data points in all the data files.
            data_point_interval (float): The interval in seconds at which data
                points are generated. If multiple data points occur within a
                single upload interval, they will be averaged and the result
                will be uploaded to the InfluxDB.
            start_date (datetime, optional): The starting datetime for the
                timestamps. Defaults to the current UTC time.
        """
        self.timestamps_set = True
        self.timestamps = [
            start_date + timedelta(seconds=i * data_point_interval)
            for i in range(num_data_points)
        ]

    def set_database_details(
        self,
        database_name,
        measurement_name,
        host,
        port,
    ):
        """
        Sets the database connection details and verifies the connection.

        Params:
            database_name (str): The name of the database.
            measurement_name (str): The name of the measurement.
            host (str): The host address of the InfluxDB server.
            port (int): The port number of the InfluxDB server.

        Raises:
            ValueError: If any parameter does not meet the expected
                format or range.
            ConnectionError: If connection to the specified database fails.
        """
        if not all(
            isinstance(x, str) and x.strip()
            for x in [database_name, measurement_name, host]
        ):
            raise ValueError("Database details must be non-empty strings.")
        if not isinstance(port, int) or not 1 <= port <= 65535:
            raise ValueError(
                "Port must be an integer within the TCP port range 1-65535."
            )

        # Ensure the database details are valid
        try:
            client = self.influxdb_client(host=host, port=port, timeout=3)
            # This will throw an error if the database does not exist
            client.switch_database(database_name)
        except InfluxDBClientError as e:
            raise ConnectionError(f"Failed to connect to the database: {e}") from e
        finally:
            if "client" in locals():
                client.close()

        # Set database details if all checks pass
        self.database_details_set = True
        self.database_name = database_name
        self.measurement_name = measurement_name
        self.host = host
        self.port = port

    def clear_measurements(self):
        """
        Clears the live measurement data from the database. This method should be
        called before uploading new data to the database. If the measurement
        does not exist, this method will not raise an error.

        Raises:
            RuntimeError: If the database details are not set prior to calling
                this method.
            InfluxDBClientError: If there is an error during the deletion of
            the measurement.
        """
        if not self.database_details_set:
            raise RuntimeError("Database details not set.")
        try:
            with self.influxdb_client(host=self.host, port=self.port) as client:
                client.switch_database(self.database_name)
                client.query(f'DROP MEASUREMENT "{self.measurement_name}"')
                self.database_cleared = True
        except (ConnectionError, InfluxDBClientError) as e:
            print(f"Failed to clear the measurement: {e}")

    def _preprocess_data(self, data_array, points_per_upload):
        return np.array(
            [
                np.round(
                    np.mean(data_array[:, i : i + points_per_upload], axis=1),
                    decimals=2,
                )
                for i in range(0, data_array.shape[1], points_per_upload)
            ]
        )

    def _calculate_average_timestamp(self, start_index, end_index):
        avg_timestamp = sum(
            ts.timestamp() for ts in self.timestamps[start_index:end_index]
        ) / (end_index - start_index)
        return int(
            datetime.fromtimestamp(avg_timestamp, timezone.utc).timestamp() * 1e9
        )

    def _sleep_until_next_interval(self, start_time, upload_interval):
        elapsed_time = (datetime.now() - start_time).total_seconds()
        remaining_sleep_time = max(0, upload_interval - elapsed_time)

        # Break sleep into smaller chunks to check for pause condition
        chunk_size = 0.04  # Check every 40ms
        sleep_chunks = int(remaining_sleep_time / chunk_size)
        remaining_fraction = remaining_sleep_time % chunk_size

        for _ in range(sleep_chunks):
            if self.pause_requested and not self.resume_requested:
                while self.pause_requested and not self.resume_requested:
                    time.sleep(0.1)
            time.sleep(chunk_size)

        if remaining_fraction > 0:
            time.sleep(remaining_fraction)

    def _create_live_data_points(self, data, timestamp):
        def escape_quotes(s):
            return s.replace('"', '\\"')

        return [
            f'{self.measurement_name} channel_data="{escape_quotes(str(data.tolist()).replace(" ", ""))}" {timestamp}'
        ]

    def _write_live_data_points(self, client, data_points):
        all_data_points = "\n".join(data_points)
        client.write_points(all_data_points, protocol="line")

    def cleanup(self):
        """Cleanup resources."""
        if self.client:
            self.client.close()
            self.client = None
        self.initialized = False
        self.processed_data = None
        self.points_per_upload = None
        self.upload_interval = None
