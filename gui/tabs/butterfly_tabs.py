from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QStackedWidget,
                             QPushButton, QSplitter, QScrollArea, QGroupBox, 
                             QComboBox, QDoubleSpinBox, QSpinBox, QLineEdit, 
                             QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView,
                             QGridLayout, QMessageBox, QFileDialog)
from PyQt6.QtCore import Qt 
from PyQt6.QtGui import QFont
import numpy as np
import os
import traceback
import pickle

# --- Matplotlib Integration ---
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.colors import LinearSegmentedColormap

# 後端
from core.PhysicsModel import HamiltonianModel 
from core.GeometryData import SimplicialComplex 
from core.Butterfly import Entomologist, ButterflyCage

# 輔助工具
from gui.utils import HOLine
from asteval import Interpreter
aeval = Interpreter()

class ButterflyTab(QWidget):
    def __init__(self, generator_tab=None, physics_tab=None):
        super().__init__()
        
        # --- 1. 修正 Tab 連結邏輯 ---
        # 優先使用直接傳入的 generator_tab
        self.generator_tab = generator_tab
        self.physics_tab = physics_tab
        
        # 如果沒傳 generator_tab 但有 physics_tab，嘗試從 physics_tab 獲取
        if self.generator_tab is None and self.physics_tab is not None:
            if hasattr(self.physics_tab, 'generator_tab'):
                self.generator_tab = self.physics_tab.generator_tab

        # --- State Data ---
        self.hamiltonian_model = None
        self.hopping_widgets = {} 
        self.worker = None        
        self.last_cage = None     
        
        self.init_ui()

    def init_ui(self):
        main_layout = QHBoxLayout(self)

        # Left Panel (Controls)
        self.scrollarea = QScrollArea()
        self.scrollarea.setWidgetResizable(True)
        self.scrollarea.setFixedWidth(320)
        
        self.controls_container = QWidget()
        self.controls_layout = QVBoxLayout(self.controls_container)
        self.controls_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.scrollarea.setWidget(self.controls_container)

        title = QLabel("<h2>Butterfly Simulation</h2>")
        title.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        self.controls_layout.addWidget(title)
        self.controls_layout.addWidget(HOLine())
        # Groups
        self._init_group_geometry()
        self._init_group_model()
        self._init_group_sweep()
        self._init_group_exec()
        self._init_group_graphic_params()
        self.controls_layout.addStretch()
        self._init_group_sl()

        # Right Panel
        self.plotter = ButterflyPlotter()

        # Splitter
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.addWidget(self.scrollarea)
        self.splitter.addWidget(self.plotter)
        
        main_layout.addWidget(self.splitter)

    # =================================================
    #  UI Initialization
    # =================================================

    def _init_group_geometry(self):
        self.group_geometry = QGroupBox("1. Geometry Source") # 存為成員變數以便控制
        layout = QVBoxLayout(self.group_geometry)

        self.lbl_geo_info = QLabel("Status: No Geometry")
        self.lbl_geo_info.setStyleSheet("color: #FF5555; font-weight: bold;")
        layout.addWidget(self.lbl_geo_info)

        h_btns = QHBoxLayout()
        btn_import = QPushButton("Import from Generator")
        btn_import.clicked.connect(self.load_geometry_from_generator)
        
        btn_load = QPushButton("Load File (.npz)")
        btn_load.clicked.connect(self.load_geometry_from_file)
        
        h_btns.addWidget(btn_import)
        h_btns.addWidget(btn_load)
        layout.addLayout(h_btns)
        
        self.controls_layout.addWidget(self.group_geometry)

    def _init_group_model(self):
        self.group_model = QGroupBox("2. Model Parameters")
        layout = QVBoxLayout(self.group_model)

        # On-site
        r_onsite = QHBoxLayout()
        r_onsite.addWidget(QLabel("On-site:"))
        self.combo_onsite = QComboBox()
        self.combo_onsite.addItems(['Bipartite', 'Laplace', 'Random'])
        r_onsite.addWidget(self.combo_onsite)
        self.spin_onsite_val = QDoubleSpinBox()
        self.spin_onsite_val.setRange(-50, 50); self.spin_onsite_val.setSingleStep(0.1)
        r_onsite.addWidget(self.spin_onsite_val)
        layout.addLayout(r_onsite)

        layout.addWidget(QLabel("Hopping (Complex):"))
        self.container_hopping = QWidget()
        self.grid_hopping = QGridLayout(self.container_hopping)
        self.grid_hopping.setContentsMargins(0,0,0,0)
        layout.addWidget(self.container_hopping)
        
        # 預設提示
        self.grid_hopping.addWidget(QLabel("<font color='#FF5555'>Load Geometry to define hopping</font>"), 0, 0)

        self.controls_layout.addWidget(self.group_model)

    def _init_group_sweep(self):
        self.group_sweep = QGroupBox("3. Sweep Configuration")
        layout = QVBoxLayout(self.group_sweep)
        
        self.tab_sweep = QTabWidget()

        # --- Tab 1: 1D Sweep ---
        w_1d = QWidget()
        l_1d = QGridLayout(w_1d)
        self.spin_flux_min = QDoubleSpinBox(); self.spin_flux_min.setRange(-100, 100); self.spin_flux_min.setValue(-3.14); self.spin_flux_min.setSingleStep(0.01)
        self.spin_flux_max = QDoubleSpinBox(); self.spin_flux_max.setRange(-100, 100); self.spin_flux_max.setValue(3.14); self.spin_flux_max.setSingleStep(0.01)
        
        l_1d.addWidget(QLabel("Min Flux Density:"), 0, 0); l_1d.addWidget(self.spin_flux_min, 0, 1)
        l_1d.addWidget(QLabel("Max Flux Density:"), 1, 0); l_1d.addWidget(self.spin_flux_max, 1, 1)
        self.tab_sweep.addTab(w_1d, "1D Sweep")
        
        # --- Tab 2: 2D Path ---
        w_2d = QWidget()
        l_2d = QVBoxLayout(w_2d)
        
        # Areas
        h_area = QHBoxLayout()
        self.edit_area1 = QLineEdit("sin(pi/5)"); self.edit_area1.setPlaceholderText("A_thin")
        self.edit_area2 = QLineEdit("sin(2*pi/5)"); self.edit_area2.setPlaceholderText("A_thick")
        h_area.addWidget(QLabel("Area 1:")); h_area.addWidget(self.edit_area1)
        h_area.addWidget(QLabel("Area 2:")); h_area.addWidget(self.edit_area2)
        l_2d.addLayout(h_area)
        
        # Path Table
        l_2d.addWidget(QLabel("Flux Path (Φ1,Φ2):"))
        self.table_path = QTableWidget(2, 2)
        self.table_path.setHorizontalHeaderLabels(["Flux 1 ", "Flux 2 "])
        self.table_path.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table_path.setItem(0,0, QTableWidgetItem("0.0")); self.table_path.setItem(0,1, QTableWidgetItem("0.0"))
        self.table_path.setItem(1,0, QTableWidgetItem("1.0")); self.table_path.setItem(1,1, QTableWidgetItem("1.0"))
        l_2d.addWidget(self.table_path)
        
        # Table Controls
        h_tbl = QHBoxLayout()
        btn_add = QPushButton("+"); btn_add.clicked.connect(self._add_path_row)
        btn_rem = QPushButton("-"); btn_rem.clicked.connect(self._rem_path_row)
        h_tbl.addWidget(btn_add); h_tbl.addWidget(btn_rem)
        l_2d.addLayout(h_tbl)
        
        self.tab_sweep.addTab(w_2d, "2D Path")
        
        layout.addWidget(self.tab_sweep)
        
        # Global Resolution
        h_res = QHBoxLayout()
        h_res.addWidget(QLabel("Total Resolution (Steps):"))
        self.spin_resolution = QSpinBox()
        self.spin_resolution.setRange(10, 10000); self.spin_resolution.setValue(200)
        h_res.addWidget(self.spin_resolution)
        layout.addLayout(h_res)

        self.controls_layout.addWidget(self.group_sweep)

    def _init_group_exec(self):
        group = QGroupBox("4. Execution")
        layout = QVBoxLayout(group)
        
        h_act = QHBoxLayout()
        self.btn_run = QPushButton("Run Calculation")
        self.btn_run.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold; padding: 4px; width: 50px;")
        self.btn_run.clicked.connect(self.run_calculation)
        
        self.btn_abort = QPushButton("Abort")
        self.btn_abort.setStyleSheet("background-color: #F44336; color: white; font-weight: bold; padding: 4px;width: 50px;")
        self.btn_abort.setEnabled(False)
        self.btn_abort.clicked.connect(self.abort_calculation)
        
        h_act.addWidget(self.btn_run)
        h_act.addWidget(self.btn_abort)
        layout.addLayout(h_act)
        
        self.progress_lbl = QLabel("Ready")
        self.progress_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.progress_lbl)
        
        self.controls_layout.addWidget(group)

    def _init_group_graphic_params(self):
    
        group = QGroupBox("5. Graphic Parameters")
        layout = QVBoxLayout(group)
        
        h_res = QHBoxLayout()
        
        self.combo_plot_modes = QComboBox()
        self.combo_plot_modes.addItems(['Plot Lines', 'Plot Density'])
        h_res.addWidget(QLabel("Plot Mode:"))
        h_res.addWidget(self.combo_plot_modes)

        layout.addLayout(h_res)

        self.plot_mode_stack = QStackedWidget()
        layout.addWidget(self.plot_mode_stack)

        # === Plot Lines page ===
        plot_lines_widget = QWidget()
        plot_lines_layout = QHBoxLayout(plot_lines_widget)
        plot_lines_layout.addStretch()
        plot_lines_layout.addWidget(QLabel("Line width:"))

        self.spin_line_width = QDoubleSpinBox()
        self.spin_line_width.setRange(0, 10); self.spin_line_width.setSingleStep(0.1); self.spin_line_width.setValue(0.6)    
        self.spin_line_width.setDecimals(1); self.spin_line_width.setStyleSheet("width: 50px;")
        plot_lines_layout.addWidget(self.spin_line_width)

        self.plot_mode_stack.addWidget(plot_lines_widget)

        # === Plot Density page ===
        plot_density_widget = QWidget()
        plot_density_layout = QHBoxLayout(plot_density_widget)
        plot_density_layout.addStretch()
        plot_density_layout.addWidget(QLabel("Bins(x,y):"))

        self.spin_bins_L = QSpinBox()
        self.spin_bins_L.setRange(0, 2048); self.spin_bins_L.setSingleStep(16); self.spin_bins_L.setValue(128)
        self.spin_bins_L.setStyleSheet("width: 50px;") ; self.spin_bins_L

        self.spin_bins_R = QSpinBox()
        self.spin_bins_R.setRange(0, 2048); self.spin_bins_R.setSingleStep(16); self.spin_bins_R.setValue(128)
        self.spin_bins_R.setStyleSheet("width: 50px;")
        
        plot_density_layout.addWidget(self.spin_bins_L)
        plot_density_layout.addWidget(self.spin_bins_R)

        self.plot_mode_stack.addWidget(plot_density_widget)

        # ---------- ComboBox 控制 Stack ----------
        self.combo_plot_modes.currentIndexChanged.connect(
            self.plot_mode_stack.setCurrentIndex)
        self.controls_layout.addWidget(group)
        
    def _init_group_sl(self):
        # -----------------------------------------------------
        # [新增] 資料管理區域 (Data Management Group)
        # -----------------------------------------------------
        data_group = QGroupBox()
        data_layout = QHBoxLayout()
        
        # 儲存按鈕
        self.btn_save = QPushButton("Save Butterfly")
        self.btn_save.setStyleSheet("padding: 6px;")
        self.btn_save.clicked.connect(self.on_save_clicked)
        self.btn_save.setEnabled(False) # 一開始沒有數據，先鎖住
        
        # 讀取按鈕
        self.btn_load = QPushButton("Load Butterfly")
        self.btn_load.setStyleSheet("padding: 6px;")
        self.btn_load.clicked.connect(self.on_load_clicked)
        
        # 刷新按鈕
        self.btn_refresh = QPushButton("⟲")
        self.btn_refresh.setMaximumSize(30,30)
        self.btn_refresh.clicked.connect(self.on_refresh_clicked)

        data_layout.addWidget(self.btn_save)
        data_layout.addWidget(self.btn_load)
        data_layout.addWidget(self.btn_refresh)
        data_group.setLayout(data_layout)
        
        self.controls_layout.addWidget(data_group)
    # =================================================
    #  Logic: Geometry Import (Fixed)
    # =================================================

    def load_geometry_from_generator(self):
        """從 Generator Tab 匯入 Mesh (已修復連結檢查)"""
        print("[DEBUG] Import button clicked.")
        
        # 1. 檢查 Tab 連結
        if self.generator_tab is None:
            print("[ERROR] self.generator_tab is None.")
            QMessageBox.critical(self, "Link Error", 
                "Generator Tab not linked.\nPlease check main.py initialization: ButterflyTab(generator_tab=...)")
            return

        # 2. 檢查 Mesh 屬性
        if not hasattr(self.generator_tab, 'current_complex'):
            print("[ERROR] Generator Tab has no 'current_complex' attribute.")
            QMessageBox.critical(self, "Data Error", "Generator Tab object has no 'mesh' attribute.")
            return

        mesh_data = self.generator_tab.current_complex
        
        # 3. 檢查 Mesh 內容
        if mesh_data is None:
            print("[ERROR] Generator Tab found, but mesh is None.")
            QMessageBox.warning(self, "Empty Data", 
                "Generator Tab has no mesh data yet.\nDid you press 'Generate' in Tab 1?")
            return

        print(f"[SUCCESS] Mesh loaded. Type: {type(mesh_data)}")
        
        # 4. 統一處理載入邏輯
        self._process_loaded_geometry(mesh_data, "Generator")

    def load_geometry_from_file(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Load Geometry", "", "Data (*.npz *.pkl)")
        if not fname: return
        try:
            geo_data = None
            if fname.endswith('.pkl'):
                with open(fname, 'rb') as f: 
                    geo_data = pickle.load(f)
            elif fname.endswith('.npz'):
                geo_data = SimplicialComplex.load(fname)
            
            if geo_data:
                self._process_loaded_geometry(geo_data, fname.split('/')[-1])
            
        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, "Load Error", str(e))

    def _process_loaded_geometry(self, geometry_data, source_name):
        """統一處理幾何載入後的初始化工作"""
        try:
            # 1. 初始化 Hamiltonian Model (用於獲取 Edge Types 和檢查 Bipartite)
            self.hamiltonian_model = HamiltonianModel(geometry_data)
            if not hasattr(self.hamiltonian_model, 'geo'):
                raise AttributeError("HamiltonianModel object missing 'geo' attribute.")
            
            geo = self.hamiltonian_model.geo
            n = len(geo.positions)

            # 更新 UI 狀態
            self.lbl_geo_info.setText(f"Loaded: {source_name} (N={n})")
            self.lbl_geo_info.setStyleSheet("color: #4CAF50; font-weight: bold;")
            
            # 重建 Hopping UI (使用 geo 中的 edge_type_map)
            edge_map = getattr(geo, 'edge_type_map', {0: 'Default'})
            self._rebuild_hopping_ui(edge_map)
            
            # 啟用 Model Group
            self.group_model.setEnabled(True)

        except Exception as e:
            traceback.print_exc()
            self.hamiltonian_model = None # 載入失敗則清空
            QMessageBox.critical(self, "Init Error", f"Failed to initialize model:\n{str(e)}")

    def _rebuild_hopping_ui(self, edge_map):
        # Clear old widgets
        while self.grid_hopping.count():
            item = self.grid_hopping.takeAt(0)
            if item.widget(): item.widget().deleteLater()
        self.hopping_widgets.clear()
        
        # Header
        self.grid_hopping.addWidget(QLabel("Type"), 0, 0)
        self.grid_hopping.addWidget(QLabel("Re"), 0, 1)
        self.grid_hopping.addWidget(QLabel("Im"), 0, 2)
        
        row = 1
        for type_id, name in edge_map.items():
            self.grid_hopping.addWidget(QLabel(str(name)), row, 0)
            
            s_re = QDoubleSpinBox(); s_re.setRange(-20, 20); s_re.setSingleStep(0.1)
            s_im = QDoubleSpinBox(); s_im.setRange(-20, 20); s_im.setSingleStep(0.1)
            
            if row == 1: s_re.setValue(1.0) # Default t=1.0
            
            self.grid_hopping.addWidget(s_re, row, 1)
            self.grid_hopping.addWidget(s_im, row, 2)
            self.hopping_widgets[type_id] = (s_re, s_im)
            row += 1

    # =================================================
    #  Logic: Execution
    # =================================================

    def run_calculation(self):
        # 檢查 Model 是否存在
        if not self.hamiltonian_model:
            QMessageBox.warning(self, "Error", "Load geometry first.")
            return

        try:
            # 1. 收集並轉換資料
            control_vars = self._gather_control_vars()
            sweep_config = self._gather_sweep_config()
            
            # 2. 鎖定 UI
            self._set_ui_locked(True)
            self.progress_lbl.setText("Initializing Entomologist...")
            self.progress_lbl.setStyleSheet("color: orange;")
            
            # 3. 啟動 Worker
            # 傳入 self.hamiltonian_model
            self.worker = Entomologist(self.hamiltonian_model, control_vars, sweep_config)
            
            self.worker.progress_sig.connect(lambda v: self.progress_lbl.setText(f"Progress: {v}%"))
            self.worker.finished_sig.connect(self._on_worker_finished)
            self.worker.error_sig.connect(self._on_worker_error)
            self.worker.start()
            
        except Exception as e:
            traceback.print_exc()
            QMessageBox.warning(self, "Input Error", str(e))
            self._set_ui_locked(False)

    def abort_calculation(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.progress_lbl.setText("Aborting...")

    def _on_worker_finished(self, cage: ButterflyCage):

        self.last_cage = cage
        self._set_ui_locked(False)
        steps = len(cage.params)
        self.progress_lbl.setText(f"Done. (Steps={steps})")
        self.progress_lbl.setStyleSheet("color: green;")
        
        self.btn_save.setEnabled(True)
        # [新增] 呼叫 Plotter 繪圖
        print(f"[UI] Plotting {steps} steps...")
        if self.combo_plot_modes.currentText() == 'Plot Lines':
            mode = 'Plot Lines'
            linewidth = self.spin_line_width.value()
            self.plotter.update_plot(cage, plot_mode=mode, linewidth=linewidth)
        else:
            mode = 'Plot Density'
            bins = (self.spin_bins_L.value(), self.spin_bins_R.value())
            self.plotter.update_plot(cage, plot_mode=mode, bins=bins)

    def _on_worker_error(self, msg):
        self._set_ui_locked(False)
        self.progress_lbl.setText("Error")
        self.progress_lbl.setStyleSheet("color: red;")
        QMessageBox.critical(self, "Calculation Error", msg)

    # =================================================
    #  Helpers
    # =================================================

    def _gather_control_vars(self):
        """
        收集 UI 參數並轉換為 HamiltonianModel.construct 的輸入格式
        
        Returns:
            dict: 包含 'ttdict', 'vec_pot_mode', 'onsite_config'
        """
        if not self.hamiltonian_model:
            raise ValueError("No model loaded.")

        geo = self.hamiltonian_model.geo
        
        # 1. Hopping 轉換: UI (int ID) -> Backend (str Name)
        # 對應 construct(ttdict=...)
        ttdict = {}
        for tid, (w_re, w_im) in self.hopping_widgets.items():
            # 嘗試從 map 獲取名稱，若無則用 ID 字串
            name = geo.edge_type_map.get(tid, str(tid))
            val = complex(w_re.value(), w_im.value())
            ttdict[name] = val
            
        # 2. On-site 轉換: UI Selection -> Backend Config
        # 對應 construct(onsite_config=...)
        ui_onsite = self.combo_onsite.currentText()
        scale = self.spin_onsite_val.value()
        
        onsite_config = None
        if ui_onsite == 'Bipartite':
            onsite_config = {'type': 'bipartite', 'scale': scale}
        elif ui_onsite == 'Random':
            onsite_config = {'type': 'random', 'scale': scale}
        elif ui_onsite == 'Laplace':
            onsite_config = {'type': 'laplace', 'scale': scale}
        # 3. Vector Potential Mode
        # 對應 construct(vec_pot_mode=...)
        vec_pot_mode = 'Landau'

        # 打包回傳
        return {
            'ttdict': ttdict,
            'onsite_config': onsite_config,
            'vec_pot_mode': vec_pot_mode
        }

    def _gather_sweep_config(self):
        cfg = {}
        cfg['resolution'] = self.spin_resolution.value()
        
        if self.tab_sweep.currentIndex() == 0: # 1D
            cfg['mode'] = '1D'
            cfg['min'] = self.spin_flux_min.value()
            cfg['max'] = self.spin_flux_max.value()
        else: # 2D
            cfg['mode'] = '2D'
            a1 = self._parse_math(self.edit_area1.text())
            a2 = self._parse_math(self.edit_area2.text())
            if a1 is None or a2 is None: raise ValueError("Invalid Area expression")
            cfg['areas'] = (a1, a2)
            
            path = []
            for r in range(self.table_path.rowCount()):
                try:
                    p1 = float(self.table_path.item(r, 0).text())
                    p2 = float(self.table_path.item(r, 1).text())
                    path.append((p1, p2))
                except: raise ValueError(f"Invalid path value at row {r+1}")
            cfg['path_points'] = path
        return cfg

    def _parse_math(self, txt):
        try:
            if aeval: return float(aeval(txt))
            return float(eval(txt, {"__builtins__": None}, {"sqrt": np.sqrt, "pi": np.pi}))
        except: return None

    def _set_ui_locked(self, locked):
        self.group_geometry.setEnabled(not locked)
        self.group_model.setEnabled(not locked)
        self.group_sweep.setEnabled(not locked)
        self.btn_run.setEnabled(not locked)
        self.btn_abort.setEnabled(locked)

    def _add_path_row(self):
        r = self.table_path.rowCount()
        self.table_path.insertRow(r)
        self.table_path.setItem(r,0, QTableWidgetItem("0.0"))
        self.table_path.setItem(r,1, QTableWidgetItem("0.0"))

    def _rem_path_row(self):
        if self.table_path.rowCount() > 2:
            self.table_path.removeRow(self.table_path.rowCount()-1)

        # ---------------------------------------------------------

    def on_save_clicked(self):
        if self.last_cage is None:
            return

        # 開啟檔案儲存對話框
        file_path, filter_type = QFileDialog.getSaveFileName(
            self,
            "Save Butterfly Cage",
            os.getcwd(),
            "NumPy Zip (*.npz);;Pickle Object (*.pkl)"
        )

        if file_path:
            # 根據使用者選的副檔名自動補齊 (如果沒打的話)
            if filter_type.startswith("NumPy") and not file_path.endswith('.npz'):
                file_path += '.npz'
            elif filter_type.startswith("Pickle") and not file_path.endswith('.pkl'):
                file_path += '.pkl'

            success = self.last_cage.save(file_path)
            
            if success:
                QMessageBox.information(self, "Success", f"Data saved to:\n{file_path}")
            else:
                QMessageBox.critical(self, "Error", "Failed to save data.")

    def on_load_clicked(self):
        # 開啟檔案讀取對話框
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Butterfly Cage",
            os.getcwd(),
            "Butterfly Files (*.npz *.pkl);;All Files (*)"
        )

        if file_path:
            try:
                # 調用靜態方法載入
                cage = ButterflyCage.load(file_path)
                
                if isinstance(cage, ButterflyCage):
                    self.last_cage = cage
                    
                    # 1. 更新繪圖 (這裡會自動讀取 cage.path_ticks 畫出漂亮的 2D 軸)
                    if self.combo_plot_modes.currentText() == 'Plot Lines':
                        mode = 'Plot Lines'
                        linewidth = self.spin_line_width.value()
                        self.plotter.update_plot(cage, plot_mode=mode, linewidth=linewidth)
                    else:
                        mode = 'Plot Density'
                        bins = (self.spin_bins_L.value(), self.spin_bins_R.value())
                        self.plotter.update_plot(cage, plot_mode=mode, bins=bins)
                    
                    # 2. 啟用儲存按鈕 (讀進來的檔案也可以另存)
                    self.btn_save.setEnabled(True)
                    
                    # 3. (選用) 更新 UI 狀態以匹配檔案內容
                    # 例如顯示載入的模式是 1D 還是 2D
                    mode = cage.mode
                    info_text = f"Loaded {mode} Cage.\nSteps: {len(cage.eigenvalues)}"
                    QMessageBox.information(self, "Loaded", info_text)
                else:
                    QMessageBox.warning(self, "Error", "File loaded but It's not a butterfly.")
                    
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load file:\n{str(e)}")

    def on_refresh_clicked(self):

        if self.combo_plot_modes.currentText() == 'Plot Lines':
            print(f"[UI] Plotting {len(self.last_cage.eigenvalues)} steps...")
            mode = 'Plot Lines'
            linewidth = self.spin_line_width.value()
            self.plotter.update_plot(self.last_cage, plot_mode=mode, linewidth=linewidth)
        else:
            print(f"[UI] Plotting {len(self.last_cage.eigenvalues)} steps...")
            mode = 'Plot Density'
            bins = (self.spin_bins_L.value(), self.spin_bins_R.value())
            self.plotter.update_plot(self.last_cage, plot_mode=mode, bins=bins)

class ButterflyPlotter(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # 1. 建立 Layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # 2. 設定深色主題風格 (Dark Background Style)
        # 使用 matplotlib 的 style 或 rcParams 來設定
        self.figure = Figure(figsize=(10, 4), dpi=100)
        self.figure.patch.set_facecolor('#1e1e1e') # 背景色

        self.canvas = FigureCanvas(self.figure)
        self.canvas.setStyleSheet("background-color: #1e1e1e;")
        
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        
        # 3. 初始化 Axes
        self.ax = self.figure.add_subplot(111)
        self._init_axes_style()

        # 特殊 cmap Blues帶透明度
        self.cmap_neo = LinearSegmentedColormap.from_list(
            'custom_cmap', [(0, "#00000000"), (0.35, "#3858C2AF") ,  (1, "#0000FF")], N=256
        )
        
    def _init_axes_style(self):
        """設定座標軸樣式 (白色文字、深色背景)"""
        self.ax.set_facecolor('#1e1e1e')
        self.ax.tick_params(axis='x', colors='white')
        self.ax.tick_params(axis='y', colors='white')
        self.ax.xaxis.label.set_color('white')
        self.ax.yaxis.label.set_color('white')
        
        # 邊框顏色
        for spine in self.ax.spines.values():
            spine.set_edgecolor('#555')

        self.ax.grid(True, color='#333', linestyle='--', linewidth=0.5)
        self.figure.subplots_adjust( top=0.949,bottom=0.074,left=0.075,right=0.972,hspace=0.2,wspace=0.2)

    def update_plot(self, cage , plot_mode=None, linewidth=None, bins=None):
        """
        繪製蝴蝶圖
        Args:
            cage (ButterflyCage): 包含 params (flux) 和 eigenvalues
        """
        self.ax.clear()
        self._init_axes_style()
        
        if cage is None or len(cage.params) == 0:
            self.ax.text(0.5, 0.5, "No Data", color='#777', 
                         ha='center', va='center', transform=self.ax.transAxes)
            self.canvas.draw()
            return

        # 準備數據
        if cage.mode == '1D':
            x = cage.params  # (Steps,)
        elif cage.mode == '2D':
            x = np.array(range(len(cage.params)))  # (Steps,)
        # eigenvalues 可能是 List of arrays，轉為 2D numpy array (Steps, N_modes)
        y = np.array(cage.eigenvalues) 

        # 繪製(根據 plot_mode)
        if plot_mode == 'Plot Lines':
            self.ax.plot(x, y, color='#00E5FF', alpha=0.8, linewidth=linewidth)
        elif plot_mode == 'Plot Density':
            x = np.kron(x, np.ones(y.shape[1])) 
            y = y.flatten()
            self.ax.hist2d(x, y, cmap=self.cmap_neo, bins=bins)

        # 設定標籤
        self.ax.set_ylabel('Energy ($E/t$)')
        self.ax.set_title('Hofstadter Butterfly', color='white', pad=10)
        if cage.mode == '1D':
            self.ax.set_xlabel('Magnetic Flux ($\Phi/\Phi_0$)')
        elif cage.mode == '2D':
            self.ax.set_xticks(list(cage.path_ticks.keys()), list(cage.path_ticks.values()))
            self.ax.set_xlabel('Flux Path (Φ1,Φ2)')
        
        # 自動調整範圍 (保留一點邊距)
        self.ax.set_xlim(x.min(), x.max())
        
        self.canvas.draw()
