from PyQt6.QtWidgets import QFrame
class HOLine(QFrame):
    def __init__(self):
        super(HOLine, self).__init__()
        self.setFrameShape(QFrame.Shape.HLine)
        self.setFrameShadow(QFrame.Shadow.Sunken)