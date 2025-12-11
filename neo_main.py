# main.py
import sys
from PyQt6.QtWidgets import QApplication
from gui.main_window import LatticeStudio

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    window = LatticeStudio()
    window.show()
    
    sys.exit(app.exec())