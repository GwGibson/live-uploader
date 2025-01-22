
from PyQt6 import QtWidgets

class DatabaseSettingsDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, current_settings=None):
        super().__init__(parent)
        self.current_settings = current_settings or {
            "database_name": "ocs_feeds",
            "measurement_name": "LIVE_MEASUREMENTS",
            "host": "localhost",
            "port": 8086,
        }
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Database Settings")
        layout = QtWidgets.QVBoxLayout()

        form_layout = QtWidgets.QFormLayout()

        # Database name input
        self.db_name_input = QtWidgets.QLineEdit()
        self.db_name_input.setText(self.current_settings["database_name"])
        form_layout.addRow("Database Name:", self.db_name_input)

        # Measurement name input
        self.measurement_input = QtWidgets.QLineEdit()
        self.measurement_input.setText(self.current_settings["measurement_name"])
        form_layout.addRow("Measurement Name:", self.measurement_input)

        # Host input
        self.host_input = QtWidgets.QLineEdit()
        self.host_input.setText(self.current_settings["host"])
        form_layout.addRow("Host:", self.host_input)

        # Port input
        self.port_input = QtWidgets.QSpinBox()
        self.port_input.setRange(1, 65535)
        self.port_input.setValue(self.current_settings["port"])
        form_layout.addRow("Port:", self.port_input)

        layout.addLayout(form_layout)

        # Add status label for validation messages
        self.status_label = QtWidgets.QLabel("")
        self.status_label.setStyleSheet("color: red")
        layout.addWidget(self.status_label)

        # Add buttons
        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.validate_and_accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.setLayout(layout)

    def validate_and_accept(self):
        if not self.db_name_input.text().strip():
            self.status_label.setText("Database name cannot be empty")
            return
        if not self.measurement_input.text().strip():
            self.status_label.setText("Measurement name cannot be empty")
            return
        if not self.host_input.text().strip():
            self.status_label.setText("Host cannot be empty")
            return

        self.accept()

    def get_settings(self):
        return {
            "database_name": self.db_name_input.text().strip(),
            "measurement_name": self.measurement_input.text().strip(),
            "host": self.host_input.text().strip(),
            "port": self.port_input.value(),
        }
