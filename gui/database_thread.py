from PyQt6 import QtCore


class DatabaseThread(QtCore.QThread):  # Changed from QObject to QThread
    finished = QtCore.pyqtSignal()
    error = QtCore.pyqtSignal(str)
    success = QtCore.pyqtSignal()

    def __init__(self, live_uploader, settings):
        super().__init__()
        self.live_uploader = live_uploader
        self.settings = settings
        self._is_running = True

    def stop(self):
        self._is_running = False
        self.wait()  # Wait for thread to finish

    def run(self):
        """Run the database connection and setup process."""
        try:
            client = self._connect_to_database()
            if self._setup_database(client):
                self.success.emit()
        finally:
            self._cleanup()

    def _connect_to_database(self):
        """Create and test the database connection."""
        try:
            client = self.live_uploader.influxdb_client(
                host=self.settings["host"], port=self.settings["port"]
            )
            client.ping()  # Test connection
            return client
        except (ValueError, ConnectionError, TimeoutError) as e:
            self.error.emit(f"Configuration error: {str(e)}")
            raise

    def _setup_database(self, client):
        """Configure database settings and clear measurements."""
        try:
            self._update_database_settings()
            self._try_clear_measurements()
            return True
        except ConnectionError as e:
            self.error.emit(f"Database connection failed: {str(e)}")
            return False
        finally:
            client.close()

    def _update_database_settings(self):
        """Update the database settings in live_uploader."""
        self.live_uploader.set_database_details(
            database_name=self.settings["database_name"],
            measurement_name=self.settings["measurement_name"],
            host=self.settings["host"],
            port=self.settings["port"],
        )

    def _try_clear_measurements(self):
        """Attempt to clear measurements, logging warning on failure."""
        try:
            self.live_uploader.clear_measurements()
        except ConnectionError as e:
            print(f"Warning: Could not clear measurements: {e}")

    def _cleanup(self):
        """Perform cleanup tasks."""
        if self._is_running:
            self.finished.emit()
