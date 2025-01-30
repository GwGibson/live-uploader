from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum, auto
import sys
import numpy as np
from PyQt6 import QtWidgets, QtCore
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QCursor

from live_uploader import LiveUploader
from parse import DataProcessingError, get_amplitudes, get_phases
from plot import PlotCanvas

from gui.database_thread import DatabaseThread
from gui.styles import Styles
from gui.ui_components import UIComponents
from gui.database_dialog import DatabaseSettingsDialog
from gui.uploader_thread import UploaderThread


# run python -m gui.live_uploader_gui from root directory


class StreamType(Enum):
    AMPLITUDES = auto()
    PHASES = auto()
    # DFS = auto()

    def __str__(self):
        return self.name


@dataclass
class Datastream:
    stream: StreamType
    data: np.ndarray

    def __post_init__(self):
        if isinstance(self.stream, StreamType):
            self.stream = self.stream.name


class LiveUploaderGUI(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.live_uploader = (
            LiveUploader()
        )  # should inject this, eventually want to skip the db and go file -> live_viewer
        self.current_index = 0
        self.is_playing = False
        self.data_loaded = False
        self.datastream = None
        self.db_settings = None
        self.is_user_sliding = False

        # Thread management
        self.uploader_thread = None
        self.database_thread = None

        # Initialize Control UI elements
        # TODO: These should be in a container or something to make it easier to manage
        self.data_type_group = None
        self.amplitude_radio = None
        self.phase_radio = None
        self.load_button = None
        self.unload_button = None
        self.settings_button = None

        self.process_button = None
        self.center_mean_check = None
        self.filter_check = None
        self.threshold_spin = None
        self.scale_factor_spin = None
        self.min_value_label = None
        self.max_value_label = None
        self.avg_min_value_label = None
        self.avg_max_value_label = None

        self.actual_timestamps_check = None
        self.timestamp_interval_spin = None
        self.upload_interval_spin = None

        self.prev_button = None
        self.play_button = None
        self.next_button = None
        self.timeline_slider = None

        # Labels
        self.progress_label = None
        self.status_label = None

        # Data storage
        self.raw_data = None
        self.timestamps = None

        self.plot_canvas = None

        self.init_ui()
        self._set_initial_control_states()

    def init_ui(self):
        """Initialize the main UI components."""
        self.setWindowTitle("Live Uploader")
        self.setGeometry(100, 100, 800, 500)

        main_widget = QtWidgets.QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QtWidgets.QVBoxLayout()

        # Top input controls
        main_layout.addLayout(self._create_top_section())
        # Rewind, Play/Pause, Fast Forward
        main_layout.addLayout(self._create_playback_section())
        # Slider / Canvas
        visualization_container = self._create_visualization_section()
        main_layout.addWidget(visualization_container)
        main_layout.addWidget(self.plot_canvas)
        # Status text
        main_layout.addWidget(self._create_status_section())

        main_widget.setLayout(main_layout)

    def toggle_timestamp_mode(self, state):
        use_real_timestamps = bool(state)
        self.timestamp_interval_spin.setEnabled(not use_real_timestamps)

    def show_database_settings(self):
        dialog = DatabaseSettingsDialog(self, self.db_settings)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            new_settings = dialog.get_settings()
            self.database_thread = DatabaseThread(self.live_uploader, new_settings)

            self.database_thread.success.connect(
                lambda: self._handle_settings_success(new_settings)
            )
            self.database_thread.error.connect(self._handle_settings_error)
            self.database_thread.finished.connect(self._cleanup_db_thread)

            self.status_label.setText("Connecting to database...")
            self.disable_all_controls()

            QtWidgets.QApplication.setOverrideCursor(QCursor(Qt.CursorShape.WaitCursor))

            self.database_thread.start()
            loop = QtCore.QEventLoop()
            self.database_thread.finished.connect(loop.quit)
            loop.exec()

            QtWidgets.QApplication.restoreOverrideCursor()

    def _cleanup_db_thread(self):
        """Clean up thread after database operation"""
        if self.database_thread:
            self.database_thread.deleteLater()
            del self.database_thread

    def _handle_settings_success(self, new_settings):
        """Handle successful database settings update"""
        self.db_settings = new_settings
        self.status_label.setText("Database settings updated successfully")
        for control in [
            self.play_button,
            self.timeline_slider,
            self.unload_button,
            self.settings_button,
            self.process_button,
            self.center_mean_check,
            self.filter_check,
        ]:
            control.setEnabled(True)

        if self.filter_check.isChecked():
            self.threshold_spin.setEnabled(True)
            self.scale_factor_spin.setEnabled(True)

    def _handle_settings_error(self, error_msg):
        """Handle database settings error"""
        self.status_label.setText(f"Database error: {error_msg}")
        self._set_initial_control_states()

    # Lets assume this is a closed set with a max of 3 (missing df's)
    # Otherwise we should rework this so adding more data types can be done in one place
    def get_selected_data_type(self):
        """Get the currently selected data type."""
        if self.amplitude_radio.isChecked():
            return StreamType.AMPLITUDES
        elif self.phase_radio.isChecked():
            return StreamType.PHASES

    def load_data(self):
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open Data File", "", "NumPy Files (*.npy);;All Files (*)"
        )

        if file_name:
            try:
                self.raw_data = np.load(file_name)
                self._process_data()

                # Update UI states after successful loading
                self.disable_all_controls()
                self.enable_data_processing_controls()
                for control in [
                    self.unload_button,
                    self.settings_button,
                    self.process_button,  # Enable the process button
                ]:
                    control.setEnabled(True)

                self.status_label.setText(
                    f"{self.datastream.stream} data loaded successfully"
                )
            except (ValueError, OSError, RuntimeError) as e:
                self.status_label.setText(f"Error loading file: {str(e)}")
                self.unload_data()

    def _process_data(self):
        """Process data with current settings and set up the uploader."""
        try:
            # Get current processing settings
            center = self.center_mean_check.isChecked()
            filter_outliers = self.filter_check.isChecked()
            threshold = self.threshold_spin.value() if filter_outliers else None
            scale_factor = self.scale_factor_spin.value() if filter_outliers else None

            stream_type = self.get_selected_data_type()

            if stream_type == StreamType.AMPLITUDES:
                data, timestamps = get_amplitudes(
                    self.raw_data,
                    center=center,
                    filter_outliers=filter_outliers,
                    threshold=threshold,
                    scale_factor=scale_factor,
                )
            elif stream_type == StreamType.PHASES:
                data, timestamps = get_phases(
                    self.raw_data,
                    center=center,
                    filter_outliers=filter_outliers,
                    threshold=threshold,
                    scale_factor=scale_factor,
                )
            else:
                raise DataProcessingError("Unsupported data type")

            self.datastream = Datastream(stream_type, data)
            self.timestamps = timestamps
            self._update_min_max_display(data)
            self.plot_canvas.update_plot(data)

        except Exception as e:
            raise RuntimeError(f"Failed to process data: {str(e)}") from e

    def _setup_uploader(self):
        """Set up the uploader with current data"""
        try:
            if self.datastream is None or self.datastream.data is None:
                raise ValueError("No data available")

            # Get number of data points from first row/channel of data
            if len(self.datastream.data.shape) < 2:
                raise ValueError("Data must be 2-dimensional")

            num_data_points = len(self.datastream.data[0])
            if num_data_points == 0:
                raise ValueError("No data points available")

            if self.actual_timestamps_check.isChecked():
                if self.timestamps is None:
                    raise ValueError(
                        "No timestamps available for actual timestamp mode"
                    )
                self.live_uploader.set_actual_timestamps(
                    [
                        datetime.fromtimestamp(ts, tz=timezone.utc)
                        for ts in self.timestamps
                    ]
                )
                self.live_uploader.timestamps_set = True
            else:
                data_point_interval = self.timestamp_interval_spin.value()
                if data_point_interval <= 0:
                    raise ValueError("Invalid data point interval")
                self.live_uploader.set_timestamps(
                    num_data_points,
                    data_point_interval,
                    start_date=datetime.now(timezone.utc),
                )

            self.timeline_slider.setMaximum(num_data_points - 1)
            self.status_label.setText(f"Loaded: {self.datastream.stream}")
            self.data_loaded = True

        except Exception as e:
            raise RuntimeError(f"Failed to setup uploader: {str(e)}") from e

    def unload_data(self):
        if self.uploader_thread:
            self.uploader_thread.stop()
            self.uploader_thread = None

        self._set_initial_control_states()
        self.current_index = 0
        self.datastream = None
        self.data_loaded = False
        self.timeline_slider.setMaximum(0)
        self._update_min_max_display(None)
        self.raw_data = None
        self.timestamps = None
        self.progress_label.setText("")

        self.filter_check.setChecked(False)
        self.center_mean_check.setChecked(False)

        if self.plot_canvas:
            self.plot_canvas.update_plot(None)

        if self.live_uploader:
            self.live_uploader.cleanup()
            self.live_uploader = LiveUploader()
        self.status_label.setText("Data unloaded")

    def toggle_play(self):
        if not self.db_settings:
            self.status_label.setText("Database settings not configured")
            return

        self.is_playing = not self.is_playing
        self.play_button.setText("⏸" if self.is_playing else "▶")

        if self.is_playing:
            self.disable_all_controls()
            self.play_button.setEnabled(True)  # Allow pausing

            try:
                self._setup_uploader()

                if self.uploader_thread and self.uploader_thread.isRunning():
                    # Resume existing upload, but sync to current slider position first
                    self.uploader_thread.set_index(self.current_index)
                    self.uploader_thread.resume()
                    self.status_label.setText("Upload resumed")
                else:
                    # Start new upload
                    self.uploader_thread = UploaderThread(
                        self.live_uploader,
                        self.datastream,
                        self.upload_interval_spin.value(),
                    )
                    # Set initial position from slider before starting
                    self.uploader_thread.set_index(self.current_index)
                    self.uploader_thread.error.connect(self.handle_upload_error)
                    self.uploader_thread.finished.connect(self.handle_upload_finished)
                    self.uploader_thread.progress.connect(self.handle_progress)
                    self.uploader_thread.index_updated.connect(
                        self.update_slider_position
                    )
                    self.uploader_thread.start()
                    self.status_label.setText("Upload started")
            except (ValueError, RuntimeError, ConnectionError) as e:
                self.handle_upload_error(f"Failed to start upload: {str(e)}")
                return
        else:
            # Pause the upload if it's running
            if self.uploader_thread and self.uploader_thread.isRunning():
                self.uploader_thread.pause()
                # Sync GUI with thread's actual position from the live uploader
                self.current_index = self.uploader_thread.get_current_index()
                self.timeline_slider.setValue(self.current_index)
                self.progress_label.setText("")
                self.status_label.setText(
                    f"Upload paused at index: {self.current_index}"
                )

            # Re-enable controls
            for control in [
                self.prev_button,
                self.play_button,
                self.next_button,
                self.unload_button,
                self.timeline_slider,
            ]:
                control.setEnabled(True)

    def handle_progress(self, message):
        self.progress_label.setText(message)

    def handle_upload_finished(self):
        self.progress_label.setText("Upload complete")
        self.unload_data()
        self.play_button.setText("▶")

    def handle_upload_error(self, error_msg):
        self.status_label.setText(f"Upload error: {error_msg}")
        self.is_playing = False
        self.unload_button.setEnabled(True)  # Allow user to unload after an error.
        self.play_button.setText("▶")

    def update_slider_position(self, position):
        """Update the slider position based on upload progress."""
        self.timeline_slider.setValue(position)
        self.current_index = position

    def slider_moved(self, position):
        """Handle slider movement, only update status for user interactions"""
        if not self.data_loaded:
            return

        if self.is_user_sliding:  # Only show status message if user is dragging
            self.status_label.setText(
                f"Position: {position}/{self.timeline_slider.maximum()}"
            )
            # Update the uploader thread position if it exists and is running
            if self.uploader_thread and self.uploader_thread.isRunning():
                self.uploader_thread.set_index(position)

        self.current_index = position
        self.timeline_slider.setValue(position)

    def _move_to_index(self, new_index):
        if not self.data_loaded or self.is_playing:
            return

        max_index = self.timeline_slider.maximum()
        new_index = max(0, min(max_index, new_index))

        if new_index != self.current_index:
            self.current_index = new_index
            self.timeline_slider.setValue(new_index)
            self.live_uploader.upload_single_point()
            self.status_label.setText(f"Position: {new_index}/{max_index}")

    def move_previous(self):
        self.live_uploader.clear_measurements()
        self._move_to_index(self.current_index - 1)

    def move_next(self):
        self._move_to_index(self.current_index + 1)

    def _set_initial_control_states(self):
        self.disable_all_controls()

        self.load_button.setEnabled(True)
        self.actual_timestamps_check.setEnabled(True)
        self.timestamp_interval_spin.setEnabled(True)
        self.upload_interval_spin.setEnabled(True)
        self.data_type_group.setEnabled(True)

    def enable_data_processing_controls(self):
        for control in [
            self.process_button,
            self.center_mean_check,
            self.filter_check,
        ]:
            control.setEnabled(True)

    def disable_all_controls(self):
        for control in [
            self.load_button,
            self.unload_button,
            self.settings_button,
            self.process_button,
            self.data_type_group,
            self.center_mean_check,
            self.filter_check,
            self.threshold_spin,
            self.scale_factor_spin,
            self.actual_timestamps_check,
            self.timestamp_interval_spin,
            self.upload_interval_spin,
            self.prev_button,
            self.play_button,
            self.next_button,
            self.timeline_slider,
        ]:
            if control is not None:
                control.setEnabled(False)

    def _create_top_section(self):
        """Create the top section with input, processing, and upload settings."""
        top_section = QtWidgets.QHBoxLayout()
        top_section.setSpacing(20)

        # Add the three main sections
        top_section.addWidget(self._create_input_settings())
        top_section.addWidget(self._create_upload_settings())
        top_section.addWidget(self._create_data_processing())

        return top_section

    def _create_input_settings(self):
        group = UIComponents.create_group_box("Input Settings", Styles.GROUP_BOX)
        layout = QtWidgets.QVBoxLayout()
        layout.setSpacing(10)
        layout.setContentsMargins(15, 10, 15, 10)

        # Data Type Selection
        self.data_type_group = QtWidgets.QGroupBox("Data Type")
        radio_layout = QtWidgets.QHBoxLayout()
        self.amplitude_radio = QtWidgets.QRadioButton("Amplitudes")
        self.phase_radio = QtWidgets.QRadioButton("Phases")
        self.amplitude_radio.setChecked(True)
        radio_layout.addWidget(self.amplitude_radio)
        radio_layout.addWidget(self.phase_radio)
        self.data_type_group.setLayout(radio_layout)
        layout.addWidget(self.data_type_group)

        # Load Data File button
        self.load_button = QtWidgets.QPushButton("Load Data File")
        self.load_button.setMinimumWidth(150)
        self.load_button.setFixedHeight(30)
        self.load_button.clicked.connect(self.load_data)
        layout.addWidget(self.load_button)

        # Unload Data File button
        self.unload_button = QtWidgets.QPushButton("Unload Data File")
        self.unload_button.setMinimumWidth(150)
        self.unload_button.setFixedHeight(30)
        self.unload_button.clicked.connect(self.unload_data)
        layout.addWidget(self.unload_button)

        group.setLayout(layout)
        return group

    def _create_data_processing(self):
        group = UIComponents.create_group_box("Data Processing", Styles.GROUP_BOX)
        group.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Fixed
        )

        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(10, 5, 10, 5)
        layout.setSpacing(5)

        # Process Data button at the top
        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addStretch()

        self.process_button = QtWidgets.QPushButton("Process Data")
        self.process_button.setFixedHeight(25)
        self.process_button.setFixedWidth(100)
        self.process_button.setEnabled(False)
        self.process_button.clicked.connect(self._process_data)
        button_layout.addWidget(self.process_button)

        button_layout.addStretch()
        layout.addLayout(button_layout)

        # Main settings grid
        grid_widget = QtWidgets.QWidget()
        grid_layout = QtWidgets.QGridLayout()
        grid_layout.setContentsMargins(5, 0, 5, 0)
        grid_layout.setSpacing(10)

        # Left column: Checkboxes
        self.filter_check = QtWidgets.QCheckBox("Filter Outliers")
        self.filter_check.setChecked(False)
        self.filter_check.stateChanged.connect(self._toggle_filter_settings)
        grid_layout.addWidget(self.filter_check, 0, 0)

        self.center_mean_check = QtWidgets.QCheckBox("Center Around Mean")
        self.center_mean_check.setChecked(False)
        grid_layout.addWidget(self.center_mean_check, 1, 0)

        # Right column: Threshold and Scale Factor controls
        threshold_label = QtWidgets.QLabel("Threshold:")
        grid_layout.addWidget(threshold_label, 0, 1)
        self.threshold_spin = QtWidgets.QSpinBox()
        self.threshold_spin.setRange(1, 50)
        self.threshold_spin.setValue(3)
        self.threshold_spin.setSingleStep(1)
        self.threshold_spin.setEnabled(False)
        self.threshold_spin.setFixedWidth(70)
        grid_layout.addWidget(self.threshold_spin, 0, 2)

        scale_label = QtWidgets.QLabel("Scale Factor:")
        grid_layout.addWidget(scale_label, 1, 1)
        self.scale_factor_spin = QtWidgets.QDoubleSpinBox()
        self.scale_factor_spin.setRange(0, 1.0)
        self.scale_factor_spin.setValue(0.67)
        self.scale_factor_spin.setSingleStep(0.01)
        self.scale_factor_spin.setDecimals(2)
        self.scale_factor_spin.setEnabled(False)
        self.scale_factor_spin.setFixedWidth(70)
        grid_layout.addWidget(self.scale_factor_spin, 1, 2)

        grid_widget.setLayout(grid_layout)
        layout.addWidget(grid_widget)

        # Horizontal line
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        line.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        layout.addWidget(line)

        # Values section
        values_widget = QtWidgets.QWidget()
        values_layout = QtWidgets.QVBoxLayout()
        values_layout.setSpacing(2)  # Reduced overall vertical spacing

        # Max Values Row (Raw and Avg Max)
        max_values = QtWidgets.QHBoxLayout()
        max_values.setSpacing(20)
        max_values.setContentsMargins(0, 0, 0, 2)

        # Raw Max Value
        raw_max_layout = QtWidgets.QHBoxLayout()
        raw_max_label = QtWidgets.QLabel("Raw Max:")
        self.max_value_label = QtWidgets.QLineEdit()
        self.max_value_label.setReadOnly(True)
        self.max_value_label.setFixedWidth(100)
        self.max_value_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        raw_max_layout.addWidget(raw_max_label)
        raw_max_layout.addWidget(self.max_value_label)
        max_values.addLayout(raw_max_layout)

        # Average Max Value
        avg_max_layout = QtWidgets.QHBoxLayout()
        avg_max_label = QtWidgets.QLabel("Avg Max:")
        self.avg_max_value_label = QtWidgets.QLineEdit()
        self.avg_max_value_label.setReadOnly(True)
        self.avg_max_value_label.setFixedWidth(100)
        self.avg_max_value_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        avg_max_layout.addWidget(avg_max_label)
        avg_max_layout.addWidget(self.avg_max_value_label)
        max_values.addLayout(avg_max_layout)

        values_layout.addLayout(max_values)

        # Min Values Row (Raw and Avg Min)
        min_values = QtWidgets.QHBoxLayout()
        min_values.setSpacing(20)
        min_values.setContentsMargins(0, 0, 0, 0)

        # Raw Min Value
        raw_min_layout = QtWidgets.QHBoxLayout()
        raw_min_label = QtWidgets.QLabel("Raw Min:")
        self.min_value_label = QtWidgets.QLineEdit()
        self.min_value_label.setReadOnly(True)
        self.min_value_label.setFixedWidth(100)
        self.min_value_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        raw_min_layout.addWidget(raw_min_label)
        raw_min_layout.addWidget(self.min_value_label)
        min_values.addLayout(raw_min_layout)

        # Average Min Value
        avg_min_layout = QtWidgets.QHBoxLayout()
        avg_min_label = QtWidgets.QLabel("Avg Min:")
        self.avg_min_value_label = QtWidgets.QLineEdit()
        self.avg_min_value_label.setReadOnly(True)
        self.avg_min_value_label.setFixedWidth(100)
        self.avg_min_value_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        avg_min_layout.addWidget(avg_min_label)
        avg_min_layout.addWidget(self.avg_min_value_label)
        min_values.addLayout(avg_min_layout)

        values_layout.addLayout(min_values)

        values_widget.setLayout(values_layout)
        layout.addWidget(values_widget)

        group.setLayout(layout)
        return group

    def _update_min_max_display(self, data):
        if data is not None:
            # Raw min/max values (across all data points)
            raw_min = np.nanmin(data)
            raw_max = np.nanmax(data)
            self.min_value_label.setText(f"{raw_min:.4f}")
            self.max_value_label.setText(f"{raw_max:.4f}")

            # Average line min/max values (same calculation as used in plot)
            average_line = np.nanmean(data, axis=0)
            avg_min = np.nanmin(average_line)
            avg_max = np.nanmax(average_line)
            self.avg_min_value_label.setText(f"{avg_min:.4f}")
            self.avg_max_value_label.setText(f"{avg_max:.4f}")
        else:
            # Clear all displays when no data is present
            self.min_value_label.setText("")
            self.max_value_label.setText("")
            self.avg_min_value_label.setText("")
            self.avg_max_value_label.setText("")

    def _create_upload_settings(self):
        group = UIComponents.create_group_box("Upload Settings", Styles.GROUP_BOX)
        layout = QtWidgets.QVBoxLayout()
        layout.setSpacing(10)
        layout.setContentsMargins(15, 10, 15, 10)

        # Actual timestamps checkbox
        self.actual_timestamps_check = QtWidgets.QCheckBox("Use Actual Timestamps")
        self.actual_timestamps_check.setChecked(False)
        self.actual_timestamps_check.stateChanged.connect(self.toggle_timestamp_mode)
        layout.addWidget(self.actual_timestamps_check)

        # Data point interval
        layout.addWidget(
            self._create_interval_widget(
                "Data Point Interval (s):", self._create_data_point_spinbox()
            )
        )

        # Upload interval
        layout.addWidget(
            self._create_interval_widget(
                "Upload Interval (s):", self._create_upload_spinbox()
            )
        )

        # Database Settings button
        self.settings_button = QtWidgets.QPushButton("Database Settings")
        self.settings_button.setFixedHeight(30)
        self.settings_button.clicked.connect(self.show_database_settings)
        layout.addWidget(self.settings_button)

        group.setLayout(layout)
        return group

    def _toggle_filter_settings(self, state):
        enabled = bool(state)
        self.threshold_spin.setEnabled(enabled)
        self.scale_factor_spin.setEnabled(enabled)

    def _create_data_point_spinbox(self):
        self.timestamp_interval_spin = UIComponents.create_spin_box(
            0.01, 5.0, 0.05, 0.1
        )
        self.timestamp_interval_spin.valueChanged.connect(
            self._validate_interval_values
        )
        return self.timestamp_interval_spin

    def _create_upload_spinbox(self):
        self.upload_interval_spin = UIComponents.create_spin_box(0.05, 5.0, 0.05, 0.1)
        self.upload_interval_spin.valueChanged.connect(self._validate_interval_values)
        return self.upload_interval_spin

    def _validate_interval_values(self, _):
        """
        Validate that timestamp interval doesn't exceed upload interval.
        Automatically adjusts timestamp interval if it exceeds upload interval.
        """
        if self.timestamp_interval_spin and self.upload_interval_spin:
            timestamp_value = self.timestamp_interval_spin.value()
            upload_value = self.upload_interval_spin.value()

            if timestamp_value > upload_value:
                # Set timestamp interval to match upload interval
                self.timestamp_interval_spin.setValue(upload_value)

            # Update maximum value of timestamp interval
            self.timestamp_interval_spin.setMaximum(upload_value)

    def _create_interval_widget(self, label_text, spin_box):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        label = QtWidgets.QLabel(label_text)
        layout.addWidget(label)
        layout.addWidget(spin_box)

        widget.setLayout(layout)
        return widget

    def _create_playback_section(self):
        layout = QtWidgets.QHBoxLayout()
        layout.addStretch()

        # Previous frame button
        self.prev_button = QtWidgets.QPushButton("←")
        self.prev_button.setFixedWidth(40)
        self.prev_button.clicked.connect(self.move_previous)
        layout.addWidget(self.prev_button)

        # Play/Pause button
        self.play_button = QtWidgets.QPushButton("▶")
        self.play_button.setFixedWidth(40)
        self.play_button.clicked.connect(self.toggle_play)
        layout.addWidget(self.play_button)

        # Next frame button
        self.next_button = QtWidgets.QPushButton("→")
        self.next_button.setFixedWidth(40)
        self.next_button.clicked.connect(self.move_next)
        layout.addWidget(self.next_button)

        layout.addStretch()
        return layout

    def _create_visualization_section(self):
        container = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()
        layout.setSpacing(0)
        layout.setContentsMargins(20, 0, 20, 0)

        self.timeline_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.timeline_slider.setMinimum(0)
        self.timeline_slider.setMaximum(100)
        self.timeline_slider.sliderPressed.connect(self._slider_pressed)
        self.timeline_slider.sliderReleased.connect(self._slider_released)
        self.timeline_slider.valueChanged.connect(self.slider_moved)
        layout.addWidget(self.timeline_slider)

        # Create and add plot
        self.plot_canvas = PlotCanvas(self, width=6, height=2)
        layout.addWidget(self.plot_canvas)

        container.setLayout(layout)
        return container

    def _slider_pressed(self):
        self.is_user_sliding = True

    def _slider_released(self):
        self.disable_all_controls()  # Disable controls during the delay
        self.status_label.setText("Moving through time...")

        timer = QtCore.QTimer()
        timer.setSingleShot(True)

        def finish_slider_release():
            self.live_uploader.clear_measurements()
            self.is_user_sliding = False
            self.prev_button.setEnabled(True)
            self.play_button.setEnabled(True)
            self.next_button.setEnabled(True)
            self.timeline_slider.setEnabled(True)
            self.unload_button.setEnabled(True)
            self.progress_label.setText("")
            self.status_label.setText("Ready to resume")
            timer.deleteLater()

        # Connect the timer timeout to our finish function
        timer.timeout.connect(finish_slider_release)
        timer.start(300)

    def _create_status_section(self):
        status_widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()

        self.progress_label = QtWidgets.QLabel("")
        self.progress_label.setWordWrap(True)
        layout.addWidget(self.progress_label)

        self.status_label = QtWidgets.QLabel("No data loaded")
        layout.addWidget(self.status_label)

        status_widget.setLayout(layout)
        return status_widget


def main():
    app = QtWidgets.QApplication(sys.argv)
    gui = LiveUploaderGUI()
    gui.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
