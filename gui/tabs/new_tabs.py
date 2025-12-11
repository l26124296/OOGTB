from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
class SettingsTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("<h2>Global Settings</h2>"))
        layout.addStretch()