from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
class IsingTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("<h2>Something coming soon</h2>"))
        layout.addStretch()