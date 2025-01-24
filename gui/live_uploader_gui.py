from dataclasses import dataclass
from datetime import datetime, timezone
import sys
import numpy as np
from PyQt6 import QtWidgets, QtCore
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QCursor

from live_uploader import LiveUploader
from parse import get_amplitudes

from gui.database_thread import DatabaseThread
from gui.styles import Styles
from gui.ui_components import UIComponents
from gui.database_dialog import DatabaseSettingsDialog
from gui.uploader_thread import UploaderThread


# run python -m gui.live_uploader_gui from root directory


@dataclass
class Datastream:
    stream: str
    data: np.ndarray


class LiveUploaderGUI(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.live_uploader = None  # should inject this
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
        self.load_button = None
        self.unload_button = None
        self.settings_button = None
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

        self.init_ui()
        self._set_initial_control_states()

    def init_ui(self):
        """Initialize the main UI components."""
        self.setWindowTitle("Live Uploader")
        self.setGeometry(100, 100, 900, 300)

        main_widget = QtWidgets.QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QtWidgets.QVBoxLayout()

        # Create and add UI sections
        main_layout.addLayout(self._create_top_section())
        main_layout.addLayout(self._create_playback_section())
        main_layout.addWidget(self._create_timeline_section())
        main_layout.addWidget(self._create_status_section())

        main_widget.setLayout(main_layout)

    def toggle_timestamp_mode(self, state):
        """Enable/disable interval input based on checkbox state"""
        use_real_timestamps = bool(state)
        self.timestamp_interval_spin.setEnabled(not use_real_timestamps)

    def show_database_settings(self):
        """Show the database settings dialog"""
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
        ]:
            control.setEnabled(True)

    def _handle_settings_error(self, error_msg):
        """Handle database settings error"""
        self.status_label.setText(f"Database error: {error_msg}")
        self._set_initial_control_states()

    def load_data(self):
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open Data File", "", "NumPy Files (*.npy);;All Files (*)"
        )

        if file_name:
            try:
                self.live_uploader = LiveUploader()
                amplitudes, timestamps = get_amplitudes(file_name)
                # TODO: Allow amplitudes, phases, dfs
                self.datastream = Datastream(stream="AMPLITUDES", data=amplitudes)

                # Setup timestamps based on checkbox state
                num_data_points = len(self.datastream.data[0])

                if self.actual_timestamps_check.isChecked() and timestamps is not None:
                    # Use actualy timestamps if available
                    self.live_uploader.set_actual_timestamps(
                        [
                            datetime.fromtimestamp(ts, tz=timezone.utc)
                            for ts in timestamps
                        ]
                    )
                    self.live_uploader.timestamps_set = True
                else:
                    # Generate timestamps using interval
                    data_point_interval = self.timestamp_interval_spin.value()
                    self.live_uploader.set_timestamps(
                        num_data_points,
                        data_point_interval,
                        start_date=datetime.now(timezone.utc),
                    )

                self.timeline_slider.setMaximum(num_data_points - 1)
                self.status_label.setText(f"Loaded: {file_name}")
                self.data_loaded = True
                # Should rework this - basically want to enable play controls and slider and disable upload settings
                # Should an unload data option so user can adjust those.
                self.disable_all_controls()
                for control in [
                    self.unload_button,
                    self.settings_button,
                ]:
                    control.setEnabled(True)

            # TODO, should not worry about parsine errors here. Need this to happen when user selects
            # amplitude/phase etc.
            except (ValueError, OSError, RuntimeError, np.AxisError) as e:
                self.status_label.setText(f"Error loading file: {str(e)}")

    def unload_data(self):
        self._set_initial_control_states()
        self.current_index = 0
        self.datastream = None
        self.data_loaded = False
        self.timeline_slider.setMaximum(0)
        if self.live_uploader:
            if self.db_settings:
                self.live_uploader.clear_measurements()
            self.live_uploader.cleanup()
            self.live_uploader = None
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
            if self.uploader_thread and self.uploader_thread.isRunning():
                # Resume existing upload, but sync to current slider position first
                self.uploader_thread.set_index(self.current_index)
                self.uploader_thread.resume()
                self.status_label.setText("Upload resumed")
            else:
                # Start new upload
                try:
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
                except Exception as e:
                    self.handle_upload_error(f"Failed to start upload: {str(e)}")
                    return
        else:
            # Pause the upload if it's running
            if self.uploader_thread and self.uploader_thread.isRunning():
                self.uploader_thread.pause()
                # Sync GUI with thread's actual position from the live uploader
                self.current_index = self.uploader_thread.get_current_index()
                self.timeline_slider.setValue(self.current_index)
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
        """Set the initial enabled/disabled states of all controls."""
        self.disable_all_controls()

        self.load_button.setEnabled(True)
        self.actual_timestamps_check.setEnabled(True)
        self.timestamp_interval_spin.setEnabled(True)
        self.upload_interval_spin.setEnabled(True)

    def disable_all_controls(self):
        for control in [
            self.load_button,
            self.unload_button,
            self.settings_button,
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
        """Create the top section with input and upload settings."""
        top_section = QtWidgets.QHBoxLayout()
        top_section.setSpacing(20)

        top_section.addWidget(self._create_input_settings())
        top_section.addStretch(1)
        top_section.addWidget(self._create_upload_settings())

        return top_section

    def _create_input_settings(self):
        """Create the input settings group."""
        group = UIComponents.create_group_box("Input Settings", Styles.GROUP_BOX)
        layout = QtWidgets.QVBoxLayout()
        layout.setSpacing(10)
        layout.setContentsMargins(15, 10, 15, 10)

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

        # Database Settings button
        self.settings_button = QtWidgets.QPushButton("Database Settings")
        self.settings_button.setFixedHeight(30)
        self.settings_button.clicked.connect(self.show_database_settings)
        layout.addWidget(self.settings_button)

        group.setLayout(layout)
        return group

    def _create_upload_settings(self):
        """Create the upload settings group."""
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

        group.setLayout(layout)
        return group

    def _create_data_point_spinbox(self):
        """Create the data point interval spin box."""
        self.timestamp_interval_spin = UIComponents.create_spin_box(
            0.01, 5.0, 0.05, 0.1
        )
        return self.timestamp_interval_spin

    def _create_upload_spinbox(self):
        """Create the upload interval spin box."""
        self.upload_interval_spin = UIComponents.create_spin_box(0.05, 5.0, 0.05, 0.1)
        return self.upload_interval_spin

    def _create_interval_widget(self, label_text, spin_box):
        """Create a widget combining a label and spin box."""
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
        """Create the playback controls section with fine control buttons."""
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

    def _create_timeline_section(self):
        """Create the timeline slider section."""
        self.timeline_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.timeline_slider.setMinimum(0)
        self.timeline_slider.setMaximum(100)

        self.timeline_slider.sliderPressed.connect(self._slider_pressed)
        self.timeline_slider.sliderReleased.connect(self._slider_released)
        self.timeline_slider.valueChanged.connect(self.slider_moved)

        return self.timeline_slider

    def _slider_pressed(self):
        self.is_user_sliding = True

    def _slider_released(self):
        """Handle slider release with a delay to ensure upload completion"""
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
            self.status_label.setText("Ready to resume")
            timer.deleteLater()

        # Connect the timer timeout to our finish function
        timer.timeout.connect(finish_slider_release)
        timer.start(300)

    def _create_status_section(self):
        """Create the status and progress section."""
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
