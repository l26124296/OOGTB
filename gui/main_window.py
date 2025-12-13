# gui/main_window.py

from PyQt6.QtWidgets import QMainWindow, QTabWidget
from PyQt6.QtCore import QSize
from PyQt6.QtGui import QIcon
from gui.utils import *

# 引入所有分頁 Tabs
from gui.tabs.generator_tabs import LatticeGeneratorTab
from gui.tabs.hmodel_tabs import PhysicsCalculationTab
from gui.tabs.butterfly_tabs import ButterflyTab
from gui.tabs.ising_tabs import IsingTab
from gui.tabs.transmission_tabs import TaransmissionTab
class LatticeStudio(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TheSisTer(--Ver. pre alpha--)")
        self.resize(1400, 900)

        # 1. 建立 Tab Widget
        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.TabPosition.West) # 分頁在左側
        self.setCentralWidget(self.tabs)
        self.tabs.setObjectName("MainSideBar")
        
        # 2. 設定圖示大小 (讓圖示大一點比較好看)
        self.tabs.setIconSize(QSize(50, 50))

        # 3. 建立各個分頁的實例
        self.tab_generator = LatticeGeneratorTab()
        self.tab_physics = PhysicsCalculationTab(generator_tab=self.tab_generator)
        self.tab_butterfly = ButterflyTab(generator_tab=self.tab_generator)
        self.tab_ising = IsingTab()
        self.tab_tansmission = TaransmissionTab()

        icon_geo = QIcon('icon/geometry.png')
        icon_phy = QIcon('icon/phys.png')
        icon_butterfly = QIcon('icon/butterfly.png')
        icon_magnet = QIcon('icon/magnet.png')
        icon_trans = QIcon('icon/transmission.png')

        self.tabs.addTab(self.tab_generator, icon_geo, "")
        self.tabs.addTab(self.tab_physics, icon_phy, "")
        self.tabs.addTab(self.tab_butterfly, icon_butterfly, "")
        self.tabs.addTab(self.tab_ising, icon_magnet, "")
        self.tabs.addTab(self.tab_tansmission, icon_trans, "")

        # 5. 美化 Tab 樣式 (CSS)
        # 這裡將左側 Tab 欄位設計成深色背景，選中時變亮
        with open('./gui/tabs/qss/main_tabs.qss', 'r') as f:
            self.tabs.setStyleSheet(f.read()) 