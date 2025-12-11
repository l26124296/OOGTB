# gui/main_window.py

from PyQt6.QtWidgets import QMainWindow, QTabWidget, QApplication
from PyQt6.QtCore import QSize
from PyQt6.QtGui import QIcon
from gui.utils import *
import qtawesome as qta
import sys

# 引入剛剛拆分出去的 Tabs
from gui.tabs.generator_tabs import LatticeGeneratorTab
from gui.tabs.hmodel_tabs import PhysicsCalculationTab
from gui.tabs.new_tabs import SettingsTab
class LatticeStudio(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Lattice Physics Studio")
        self.resize(1400, 900)

        # 1. 建立 Tab Widget
        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.TabPosition.West) # 分頁在左側
        self.setCentralWidget(self.tabs)

        # 2. 設定圖示大小 (讓圖示大一點比較好看)
        self.tabs.setIconSize(QSize(50, 50))

        # 3. 建立各個分頁的實例
        self.tab_generator = LatticeGeneratorTab()
        self.tab_physics = PhysicsCalculationTab(generator_tab=self.tab_generator)
        self.tab_settings = SettingsTab()

        icon_geo = QIcon('icon/geometry.png')
        icon_phy = QIcon('icon/phys.png')
        icon_set = QIcon('icon/gear.png')

        self.tabs.addTab(self.tab_generator, icon_geo, "") # 加空格排版比較好看
        self.tabs.addTab(self.tab_physics, icon_phy, "")
        self.tabs.addTab(self.tab_settings, icon_set, "")
        
        # 5. 美化 Tab 樣式 (CSS)
        # 這裡將左側 Tab 欄位設計成深色背景，選中時變亮
        self.tabs.setStyleSheet("""
            QTabWidget::pane { 
                border: 0px solid #444; 
                background: #f0f0f0; /* 右側內容區背景色 */
            }
            QTabBar::tab { 
                height: 50px; 
                width: 50px; /* 設定足夠寬度以容納圖示+文字 */
                font-size: 14px;
                font-weight: bold;
                color: #aaa;
                background: #c3dbfb; /* 側邊欄背景色 */
                border: none;
                padding-left: 10px;
                text-align: left; /* 文字靠左對齊 */
            }
            QTabBar::tab:selected { 
                color: #ffffff;
                background: #b0cfff; /* 選中時的背景 */
                border-left: 4px solid #2e7dff; /* 左側亮條 */
            }
            QTabBar::tab:hover {
                background: #333333;
            }
        """)