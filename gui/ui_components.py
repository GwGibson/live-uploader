from PyQt6 import QtWidgets


class UIComponents:
    @staticmethod
    def create_group_box(title, style):
        group = QtWidgets.QGroupBox(title)
        group.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed
        )
        group.setStyleSheet(style)
        return group

    @staticmethod
    def create_spin_box(min_val, max_val, step, default, width=70):
        spin_box = QtWidgets.QDoubleSpinBox()
        spin_box.setRange(min_val, max_val)
        spin_box.setSingleStep(step)
        spin_box.setValue(default)
        spin_box.setFixedWidth(width)
        return spin_box
