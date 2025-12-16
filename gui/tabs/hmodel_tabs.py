# gui/tabs/physics_tab.py

import datetime
import traceback
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSpinBox,
                             QPushButton, QSplitter, QTextEdit, QScrollArea, 
                             QGroupBox, QFileDialog, QMessageBox, QComboBox, 
                             QDoubleSpinBox, QGridLayout , QCheckBox)
from PyQt6.QtCore import Qt 
from PyQt6.QtGui import QFont , QTransform
import pyqtgraph as pg
from gui.utils import *
# 引入後端
from core.LatticeGenerators import *
from core.PhysicsModel import *
from core.analyzer import *

class PhysicsCalculationTab(QWidget):
    def __init__(self, generator_tab=None):
        super().__init__()
        # 保存對 Generator Tab 的引用，以便直接讀取記憶體中的數據
        self.generator_tab = generator_tab 
        
        # --- State ---
        self.current_geometry = None
        self.hamiltonian_model = None

        # 儲存 Hopping 輸入框的參照: { type_id: (spin_real, spin_imag) }
        self.hopping_widgets = {}

        self.module_widgets = [] # 用來儲存物理量計算的 UI 區塊，方便統一鎖定/解鎖
        self.init_ui()

    def set_generator_tab(self, tab_instance):
        """用於在主程式中注入 Generator Tab 的實例"""
        self.generator_tab = tab_instance

    def init_ui(self):
        main_layout = QHBoxLayout(self)
        
        # =================================================
        # 1. 左側控制面板
        # =================================================
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFixedWidth(400)
        
        
        self.controls_container = QWidget()

        self.controls_layout = QVBoxLayout(self.controls_container)
        self.controls_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.scroll.setWidget(self.controls_container)

        title = QLabel("<h2>Hamiltonian Model</h2>")
        title.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        self.controls_layout.addWidget(title)
        self.controls_layout.addWidget(HOLine())
        # -------------------------------------------------
        # 1.1 資料來源區塊 (Geometry Source)
        # -------------------------------------------------
        self.group_source = QGroupBox("1. Geometry Source")
        self.group_source.setStyleSheet("QGroupBox { font-weight: bold; border: 1px solid #555; margin-top: 10px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }")
        layout_source = QVBoxLayout()

        # 1.1.1狀態顯示 Label
        self.lbl_status = QLabel("Status: No Geometry Loaded")
        self.lbl_status.setStyleSheet("color: #FF5555; font-weight: bold;")
        self.lbl_info = QLabel("Sites: 0 | Edges: 0")
        self.lbl_info.setStyleSheet("color: #888;")

        # 1.1.2按鈕：從 Tab 1 匯入
        self.btn_import = QPushButton("Import from Generator Tab")
        self.btn_import.clicked.connect(self.load_from_memory)
        
        # 1.1.3按鈕：從檔案讀取
        self.btn_load_file = QPushButton("Load from .npz File")
        self.btn_load_file.clicked.connect(self.load_from_file)

        layout_source.addWidget(self.lbl_status)
        layout_source.addWidget(self.lbl_info)
        layout_source.addWidget(self.btn_import)
        layout_source.addWidget(self.btn_load_file)
        
        self.group_source.setLayout(layout_source)
        self.controls_layout.addWidget(self.group_source)

        # -------------------------------------------------
        # 1.2 Hamiltonian 設定區塊 (預設鎖定)
        # -------------------------------------------------
        self.group_hamiltonian = QGroupBox("2. Hamiltonian Settings")
        self.group_hamiltonian.setStyleSheet("QGroupBox { font-weight: bold; border: 1px solid #555; margin-top: 10px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }")
        
        # 1.2.1使用 QVBoxLayout 作為主要容器
        self.layout_ham_main = QVBoxLayout()
        
        # --- Row 1: On-site Term ---
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("On-site:"))
        
        self.combo_onsite = QComboBox()
        self.combo_onsite.addItems(['Random', 'Laplace', 'Bipartite'])
        row1.addWidget(self.combo_onsite, 2) # Stretch factor 2
        
        row1.addWidget(QLabel("Value:"))
        self.spin_onsite_val = QDoubleSpinBox()
        self.spin_onsite_val.setRange(-100, 100)
        self.spin_onsite_val.setSingleStep(0.1)
        row1.addWidget(self.spin_onsite_val, 1)
        
        self.layout_ham_main.addLayout(row1)

        # --- Row 2: Magnetic Field ---
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Gauge:"))
        
        self.combo_gauge = QComboBox()
        self.combo_gauge.addItems(['Landau', 'Circular', 'Off'])
        row2.addWidget(self.combo_gauge, 2)
        
        row2.addWidget(QLabel("Flux B:"))
        self.spin_flux = QDoubleSpinBox()
        self.spin_flux.setRange(-50, 50)
        self.spin_flux.setSingleStep(0.01)
        self.spin_flux.setDecimals(4)
        row2.addWidget(self.spin_flux, 1)
        
        self.layout_ham_main.addLayout(row2)
        
        self.layout_ham_main.addWidget(HOLine())

        # --- Row 3+: Hopping Parameters (Dynamic Area) ---
        self.layout_ham_main.addWidget(QLabel("<b>Hopping Parameters (Complex):</b>"))
        
        # 1.2.2放置一個容器，之後動態填入 Grid
        self.container_hopping = QWidget()
        self.layout_hopping = QGridLayout(self.container_hopping)
        self.layout_hopping.setContentsMargins(0,0,0,0)
        self.layout_ham_main.addWidget(self.container_hopping)

        # --- Update Button ---
        self.btn_update_h = QPushButton("Construct and Solve H")
        self.btn_update_h.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold;")
        self.btn_update_h.clicked.connect(self.run_construct_and_solve)
        self.layout_ham_main.addWidget(self.btn_update_h)

        self.group_hamiltonian.setLayout(self.layout_ham_main)
        self.group_hamiltonian.setEnabled(False) # 預設鎖定
        self.controls_layout.addWidget(self.group_hamiltonian)

        # self.controls_layout.addStretch()

        # -------------------------------------------------------------
        # 1.3 Electronic State & Analysis 
        # -------------------------------------------------------------
        self.group_state = QGroupBox("3. Electronic State & Analysis")
        self.group_state.setStyleSheet("QGroupBox { font-weight: bold; border: 1px solid #555; margin-top: 10px; } " \
        "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }")
        layout_state = QVBoxLayout()

        # === 階段 A: 設定佔據態 (Occupancy) ===
        # 1.3.1使用 Frame 包起來，強調這是第一步
        frame_occ = QFrame()
        frame_occ.setStyleSheet("background-color: #2a2a2a; border-radius: 5px;")
        layout_occ = QVBoxLayout(frame_occ)
        
        # --- Row 1: 輸入與按鈕
        layout_occ.addWidget(QLabel("<b>Occupied Energy Window:</b>",styleSheet="color: #FFFFFF; font-weight: bold;font-size: 12px;"))
        row_range = QHBoxLayout()
        
        self.spin_emin = QDoubleSpinBox()
        self.spin_emin.setRange(-1000.00, 1000.00); self.spin_emin.setValue(-4.0); self.spin_emin.setPrefix("Min: ")
        self.spin_emin.setStyleSheet("background-color: #EFEFEF; color: #333; font-weight: bold;")
        self.spin_emin.setSingleStep(0.1) ; self.spin_emin.setDecimals(2)
        self.spin_emax = QDoubleSpinBox()
        self.spin_emax.setRange(-1000.00, 1000.00); self.spin_emax.setValue(0.0); self.spin_emax.setPrefix("Max: ")
        self.spin_emax.setStyleSheet("background-color: #EFEFEF; color: #333; font-weight: bold;")
        self.spin_emax.setSingleStep(0.1) ; self.spin_emax.setDecimals(2)

        self.btn_set_projector = QPushButton("Set")
        self.btn_set_projector.setFixedWidth(80)
        with open("gui/tabs/qss/PhyCal_set_button.qss", "r") as f:
            self.btn_set_projector.setStyleSheet(f.read())
        self.btn_set_projector.clicked.connect(self.run_set_projector)
        
        row_range.addWidget(self.spin_emin)
        row_range.addWidget(self.spin_emax)
        row_range.addWidget(self.btn_set_projector)
        layout_occ.addLayout(row_range)
        
        # --- Row 2: 狀態資訊
        self.lbl_occ_info = QLabel("Status: Projector not set")
        self.lbl_occ_info.setStyleSheet("color: #aaa; font-size: 11px; margin-left: 5px;")
        layout_occ.addWidget(self.lbl_occ_info)
        
        layout_state.addWidget(frame_occ)
        layout_state.addWidget(HOLine())

        # === 階段 B: 物理量計算模組 (Analysis Modules) ===
        # 1.3.2這裡放置縱向排列的功能塊
        self.layout_modules = QVBoxLayout()
        self.layout_modules.setSpacing(10)

        # 1.3.21 Charge Density
        self._add_analysis_module(
            title="Charge Density",
            btn_text="Plot ρ(r)",
            callback=self.run_calc_charge
        )

        # 1.3.22 Chern Marker
        self._add_analysis_module(
            title="Chern Marker",
            btn_text="Calc & Plot C(r)",
            callback=self.run_calc_chern
        )
        # 1.3.23 Angular Momentum (需要額外參數 Cn)
        # 我們自訂這個區塊因為它有參數輸入
        frame_am = QGroupBox("Angular Momentum")
        frame_am.setStyleSheet("QGroupBox { border: 1px solid #444; margin-top: 5px; } QGroupBox::title { color: #aaa; }")
        l_am = QHBoxLayout(frame_am)
        l_am.addWidget(QLabel("Sym (C<sub>n</sub>):"))
        self.spin_cn = QSpinBox(); self.spin_cn.setRange(1, 20); self.spin_cn.setValue(5)
        l_am.addWidget(self.spin_cn)
        l_am.addStretch()
        l_am.addWidget(QLabel("Bins:"))
        self.spin_am_bins = QSpinBox()
        self.spin_am_bins.setRange(10, 500)
        self.spin_am_bins.setValue(100) # AM 預設 100
        self.spin_am_bins.setFixedWidth(60)
        l_am.addWidget(self.spin_am_bins)

        btn_am = QPushButton("Plot L")
        btn_am.clicked.connect(self.run_calc_am)
        btn_am.setFixedWidth(100)
        l_am.addWidget(btn_am)
        self.layout_modules.addWidget(frame_am)
        # 註冊到 module_widgets 以便統一啟用/禁用
        self.module_widgets.append(frame_am)

        # 1.3.24 Structure Factor (新增)
        frame_sf = QGroupBox("Structure Factor S(q)")
        frame_sf.setStyleSheet("QGroupBox { border: 1px solid #444; margin-top: 5px; } QGroupBox::title { color: #aaa; }")
        l_sf = QVBoxLayout(frame_sf)
        
        # --- Row 1: 參數
        r_sf_param = QHBoxLayout()
        r_sf_param.addWidget(QLabel("Range ±q:"))
        self.spin_qrange = QDoubleSpinBox(); self.spin_qrange.setRange(1, 100); self.spin_qrange.setValue(20.0)
        r_sf_param.addWidget(self.spin_qrange)
        
        r_sf_param.addWidget(QLabel("Resolution(per axis):"))
        self.spin_qres = QSpinBox(); self.spin_qres.setRange(50, 2000); self.spin_qres.setValue(400); self.spin_qres.setSingleStep(50)
        r_sf_param.addWidget(self.spin_qres)
        l_sf.addLayout(r_sf_param)
        
        # --- Row 2: GPU 選項與按鈕
        r_sf_action = QHBoxLayout()
        self.check_gpu = QCheckBox("Use GPU"); self.check_gpu.setChecked(True) # 預設開啟
        self.check_gpu.setStyleSheet("color: #aaa;")
        r_sf_action.addWidget(self.check_gpu)
        
        r_sf_action.addStretch()

        btn_sf = QPushButton("Plot S(q)")
        btn_sf.clicked.connect(self.run_calc_sf)
        btn_sf.setFixedWidth(100)
        r_sf_action.addWidget(btn_sf)
        l_sf.addLayout(r_sf_action)
        
        self.layout_modules.addWidget(frame_sf)
        self.module_widgets.append(frame_sf)

        # 1.3.25 Density of States
        # 原本只是一個按鈕，現在改成一個 GroupBox 或 Frame 以容納 SpinBox
        frame_dos = QGroupBox("Density of States (DOS)")
        frame_dos.setStyleSheet("QGroupBox { border: 1px solid #444; margin-top: 5px; } \
                                QGroupBox::title { color: #aaa; }")
        l_dos = QHBoxLayout(frame_dos)
        l_dos.setContentsMargins(5, 15, 5, 5)

        # Bins 參數
        l_dos.addStretch()
        l_dos.addWidget(QLabel("Bins:"))
        self.spin_dos_bins = QSpinBox()
        self.spin_dos_bins.setRange(10, 1000)
        self.spin_dos_bins.setValue(200) # DOS 預設 200，解析度高一點
        self.spin_dos_bins.setFixedWidth(60)
        l_dos.addWidget(self.spin_dos_bins)
        
        # 按鈕
        btn_back_dos = QPushButton("Plot DOS")
        btn_back_dos.clicked.connect(self._plot_dos_histogram)
        btn_back_dos.setFixedWidth(100)
        l_dos.addWidget(btn_back_dos)

        self.layout_modules.addWidget(frame_dos)
        self.module_widgets.append(frame_dos)

        layout_state.addLayout(self.layout_modules)

        # 初始設定：Hamiltonian 未解前鎖定整個大區塊
        self.group_state.setLayout(layout_state)
        self.group_state.setEnabled(False)
        self.controls_layout.addWidget(self.group_state)
        
        # 初始設定：物理量模組先鎖定，直到按了 "Set" 才能用
        self._set_modules_enabled(False)

        self.controls_layout.addStretch()

        # =================================================
        # 2. 右側顯示區
        # =================================================
        right_splitter = QSplitter(Qt.Orientation.Vertical)
        right_splitter.setHandleWidth(5)

        # 2.1 Plot Widget
        self.plot_widget = pg.PlotWidget(background='#1e1e1e') # 給個預設值就好
        right_splitter.addWidget(self.plot_widget)

        # 2.2 Terminal
        self.terminal = QTextEdit()
        self.terminal.setReadOnly(True)
        self.terminal.setStyleSheet("background-color: #0c0c0c; color: #ccc; font-family: Consolas;")
        
        term_container = QWidget()
        term_layout = QVBoxLayout(term_container)
        term_layout.setContentsMargins(0,0,0,0); term_layout.setSpacing(0)
        term_layout.addWidget(QLabel(" OUTPUT / CONSOLE", styleSheet="background:#252526; color:#bbb; font-weight:bold;"))
        term_layout.addWidget(self.terminal)
        right_splitter.addWidget(term_container)
        
        right_splitter.setStretchFactor(0, 7)
        right_splitter.setStretchFactor(1, 3)

        # Main Splitter
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_splitter.addWidget(self.scroll)
        main_splitter.addWidget(right_splitter)
        main_splitter.setStretchFactor(1, 1)
        
        main_layout.addWidget(main_splitter)

    # =================================================
    #  -- Methods --
    # =================================================
    
    def log(self, message, level="INFO"):
        """輸出到 Terminal"""
        color = {"INFO": "#4CAF50", "WARN": "#FFC107", "ERROR": "#F44336"}.get(level, "#ccc")
        time = datetime.datetime.now().strftime("%H:%M:%S")
        self.terminal.append(f'<span style="color:#555;">[{time}]</span> <span style="color:{color};"><b>[{level}]</b></span> {message}')
        self.terminal.verticalScrollBar().setValue(self.terminal.verticalScrollBar().maximum())

    def load_from_memory(self):
        """從 Generator Tab 讀取"""
        if self.generator_tab is None:
            self.log("Generator tab reference missing.", "ERROR")
            return

        data = self.generator_tab.current_complex
        if data is None:
            QMessageBox.warning(self, "No Data", "No lattice generated in Geometry tab yet.")
            self.log("Import failed: No data in Geometry tab.", "WARN")
            return
            
        self.log("Importing geometry from memory...", "INFO")
        self._initialize_hamiltonian(data)

    def load_from_file(self):
        """從 .npz 檔案讀取"""
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Lattice Data", "", "NumPy Compressed (*.npz)")
        if file_path:
            try:
                self.log(f"Loading file: {file_path}...", "INFO")
                data = SimplicialComplex.load(file_path)
                self._initialize_hamiltonian(data)
            except Exception as e:
                self.log(f"Load failed: {str(e)}", "ERROR")
                traceback.print_exc()

    def _initialize_hamiltonian(self, geometry_data):
        """初始化並動態生成輸入欄位"""
        try:
            self.current_geometry = geometry_data
            self.lbl_status.setText("Geometry Loaded")
            self.lbl_status.setStyleSheet("color: #4CAF50;")
            
            # 1. 建立 Model
            self.hamiltonian_model = HamiltonianModel(geometry_data)

            # 2. Bipartite 檢查
            tags, is_bipartite = self.hamiltonian_model._get_bipartite_tags(target_type_id=0)
            if is_bipartite:
                self.log("Bipartite Check: SUCCESS", "INFO")
                # 可以在這裡自動把 On-site 選單切換到 Bipartite (Optional)
            else:
                self.log("Bipartite Check: FAILED (Geometry is frustrated)", "WARN")
                self.log(" -> 'Bipartite' on-site potential will be unavailable.", "WARN")

            n_sites = len(geometry_data.positions)
            n_edges = len(geometry_data.edges)
            
            self.lbl_status.setText("Geometry Loaded")
            self.lbl_status.setStyleSheet("color: #4CAF50; font-weight: bold;")
            self.lbl_info.setText(f"Sites: {n_sites} | Edges: {n_edges}")
            # 3. 動態生成 Hopping 輸入框
            self._rebuild_hopping_ui(geometry_data.edge_type_map)
            
            # 4. 解鎖
            self.group_hamiltonian.setEnabled(True)
            self.log(f"Geometry loaded. {len(geometry_data.positions)} sites.", "INFO")
            
        except Exception as e:
            self.log(f"Init Error: {e}", "ERROR")
            traceback.print_exc()

    def _rebuild_hopping_ui(self, edge_type_map):
        """根據 geometry 的 edge_type_map 動態生成 Grid Layout"""
        # 1. 清除舊元件
        while self.layout_hopping.count():
            item = self.layout_hopping.takeAt(0)
            widget = item.widget()
            if widget: widget.deleteLater()
        self.hopping_widgets.clear()

        # 2. 建立標題列
        # Grid: [Name] [Real Input] [Imag Input]
        self.layout_hopping.addWidget(QLabel("Type Name"), 0, 0)
        self.layout_hopping.addWidget(QLabel("Real (Re)"), 0, 1)
        self.layout_hopping.addWidget(QLabel("Imag (Im)"), 0, 2)

        # 3. 遍歷並生成
        if not edge_type_map:
            # 如果沒有 map (例如 Penrose 預設)，給一個 Default
            edge_type_map = {0: 'Default'}

        row = 1
        for type_id, name in edge_type_map.items():
            # Label
            lbl = QLabel(f"{name} (id:{type_id})")
            lbl.setStyleSheet("color: #bbb;")
            
            # Real Input
            spin_re = QDoubleSpinBox()
            spin_re.setRange(-100, 100)
            spin_re.setSingleStep(0.1)
            spin_re.setValue(1.0 if row == 1 else 0.0) # 預設第一個是 1.0
            
            # Imag Input
            spin_im = QDoubleSpinBox()
            spin_im.setRange(-100, 100)
            spin_im.setSingleStep(0.1)
            spin_im.setValue(0.0)
            
            # 排版
            self.layout_hopping.addWidget(lbl, row, 0)
            self.layout_hopping.addWidget(spin_re, row, 1)
            self.layout_hopping.addWidget(spin_im, row, 2)
            
            # 存儲引用，以便後續讀取
            self.hopping_widgets[type_id] = (spin_re, spin_im)
            
            row += 1
    
    def run_construct_and_solve(self):
        """建構 H -> 估算資源 -> 對角化 -> 畫 DOS"""
        if self.hamiltonian_model is None: return
        # 防呆邏輯：開始前先鎖定並重置狀態 
        self.group_state.setEnabled(False)        # 鎖定整個區塊
        self._set_modules_enabled(False)          # 鎖定物理量計算按鈕 (Charge, Chern...)
        self.lbl_occ_info.setText("Status: Projector not set") 
        self.lbl_occ_info.setStyleSheet("color: #aaa; font-size: 11px; margin-left: 5px;")
        try:
            # -----------------------------------------------------
            # 1. 讀取參數 & 建構 Hamiltonian (Sparse)
            # -----------------------------------------------------
            # (...參數讀取邏輯同前...)
            onsite_mode = self.combo_onsite.currentText().lower()
            onsite_val = self.spin_onsite_val.value()
            onsite_config = None
            if onsite_mode == 'random': onsite_config = {'type': 'random', 'scale': onsite_val}
            elif onsite_mode == 'laplace': onsite_config = {'type': 'laplace', 'scale': onsite_val}
            elif onsite_mode == 'bipartite': onsite_config = {'type': 'bipartite', 'scale': onsite_val}

            gauge_mode = self.combo_gauge.currentText()
            b_flux = self.spin_flux.value()
            if gauge_mode == 'Off': b_flux = 0.0; gauge_mode = 'Landau'

            ttdict = {}
            type_map = self.current_geometry.edge_type_map
            if not type_map: type_map = {0: 'Default'}

            for type_id, widgets in self.hopping_widgets.items():
                re_val = widgets[0].value()
                im_val = widgets[1].value()
                name = type_map.get(type_id, f"Unknown_{type_id}")
                ttdict[name] = re_val + 1j * im_val
            
            self.log("Constructing Hamiltonian...", "INFO")
            # 構建
            H = self.hamiltonian_model.construct(
                ttdict=ttdict, b_field=b_flux, vec_pot_mode=gauge_mode, onsite_config=onsite_config
            )
            
            N = H.shape[0]
            sparsity = H.nnz / (N**2) * 100
            self.log(f"H Constructed. Size: {N}x{N}, Sparsity: {sparsity:.4f}%", "INFO")

            # -----------------------------------------------------
            # 2. 資源預估與防呆
            # -----------------------------------------------------
            mem_gb, time_str, is_heavy = self.estimate_resources(N)
            
            msg = f"Resource Estimate for Diagonalization:\n - Memory (Dense): {mem_gb:.3f} GB\n - Time (Approx): {time_str}"
            self.log(msg, "WARN" if is_heavy else "INFO")
            
            if is_heavy:
                reply = QMessageBox.question(self, "Heavy Calculation", 
                    f"System size N={N} is large.\n{msg}\n\nContinue?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
                if reply == QMessageBox.StandardButton.No:
                    self.log("Calculation aborted by user.", "WARN")
                    return

            # -----------------------------------------------------
            # 3. 初始化 Analyzer 並求解 (Diagonalize)
            # -----------------------------------------------------
            self.analyzer = ElectronicState(self.hamiltonian_model)
            
            start_time = time.time()
            self.log("Starting diagonalization (this may freeze UI)...", "INFO")
            
            # 強制處理 UI 事件，讓 Log 顯示出來
            from PyQt6.QtWidgets import QApplication
            QApplication.processEvents()
            
            # 執行計算
            self.analyzer.diagonalize()
            
            elapsed = time.time() - start_time
            self.log(f"Diagonalization complete in {elapsed:.4f} sec.", "SUCCESS")

            # -----------------------------------------------------
            # 4. 繪製 DOS Histogram
            # -----------------------------------------------------
            self._plot_dos_histogram()
            
            # 解鎖 Energy range 及 set 按鈕
            self.group_state.setEnabled(True)
            self._set_modules_enabled(False)       
            self.log("Hamiltonian solved. You can now set occupied states.", "INFO")

        except Exception as e:
            self.log(f"Process Error: {e}", "ERROR")
            traceback.print_exc()
            # 如果失敗，確保區塊保持鎖定
            self.group_state.setEnabled(False)

    def _plot_dos_histogram(self):
        if self.analyzer.eigenvalues is None: return
        
        evals = self.analyzer.eigenvalues
        
        # --- 1. 動態設定樣式 (DOS 專用風格) ---
        self._configure_plot(
            title="Density of States (DOS)", 
            x_label="Energy (eV)", 
            y_label="Density / Count",
            background="#05040E",  # 深色背景
            text_color="#FFFCD8"   # 淺灰文字
        )
        
        # --- 2. 讀取 UI bins ---
        n_bins = self.spin_dos_bins.value()
        
        # --- 3. 計算 Histogram 數據 ---
        y, x_edges = np.histogram(evals, bins=n_bins)
        x_centers = (x_edges[:-1] + x_edges[1:]) / 2
        width = x_edges[1] - x_edges[0]
        
        # --- 3. 繪製 ---
        # 針對深色背景的配色
        fill_brush = pg.mkBrush("#3c90df") # 亮藍色半透明
        outline_pen = pg.mkPen(color="#337cc0", width=1)
        
        bar_item = pg.BarGraphItem(
            x=x_centers, height=y, width=width*0.98,
            brush=fill_brush ,
            pen=outline_pen
        )
        self.plot_widget.addItem(bar_item)
        self.plot_widget.autoRange()

    def _plot_scatter_heatmap(self, pos, values, cmap='viridis', adjust_range: Union[bool, Tuple[float, float]] = False):
        """
        輔助函數：畫散點熱圖 (包含 Hover Data 功能)
        """
        # 用最開始的兩個點距離來估計scatter的大小
        recommand_dot_size = 0.5 * np.linalg.norm(self.current_geometry.positions[1] - self.current_geometry.positions[0])
        recommand_dot_size = max(0.2, recommand_dot_size)
        self.plot_widget.clear()
        
        # 1. 數值正規化 (用於計算顏色)
        if adjust_range == True:
            limit = max(abs(np.min(values)), abs(np.max(values)))
            if limit == 0: limit = 1
            norm_values = (values + limit) / (2 * limit)
        elif isinstance(adjust_range, tuple):
            v_min, v_max = adjust_range
            if v_max == v_min: norm_values = np.zeros_like(values)
            else: norm_values = (values - v_min) / (v_max - v_min)
        else:
            v_min, v_max = np.min(values), np.max(values)
            if v_max == v_min: norm_values = np.zeros_like(values)
            else: norm_values = (values - v_min) / (v_max - v_min)

        # 2. 產生顏色筆刷
        import matplotlib.cm as cm
        try:
            colormap = cm.get_cmap(cmap)
        except:
            colormap = cm.viridis # Fallback

        brushes = []
        for val in norm_values:
            rgba = colormap(val) 
            color = [int(c * 255) for c in rgba]
            brushes.append(pg.mkBrush(color[0], color[1], color[2], 255))
            
        # 3. 建立 Scatter Plot Item
        scatter = pg.ScatterPlotItem(
            x=pos[:, 0], 
            y=pos[:, 1],
            size=recommand_dot_size,
            brush=brushes,
            pen=None,

            #像素點大小模式: True=以像素為單位，False=以數值為單位(隨座標縮放)
            pxMode=False,
            #傳入原始數據
            data=values,  
            
            #啟用懸停功能
            hoverable=True,
            hoverSymbol='o',    # 懸停時的樣式
            hoverSize=1.3*recommand_dot_size,        # 懸停時變大 (可選)
        )
        
        # 關鍵修改 C: 連接信號
        # 當滑鼠懸停時，呼叫我們寫好的 _on_scatter_hovered
        scatter.sigHovered.connect(self._on_scatter_hovered)
        
        self.plot_widget.addItem(scatter)
        
        # 加入簡單的數值範圍 Log
        self.log(f"Value Range: [{np.min(values):.4f}, {np.max(values):.4f}]", "DATA")
        self.plot_widget.autoRange()

    def _plot_histogram(self, data, color , bins):
        """輔助函數：畫直方圖"""
        self.plot_widget.clear()
        if len(data) == 0: return
        
        n_bins = bins

        y, x_edges = np.histogram(data, bins=n_bins)
        x_centers = (x_edges[:-1] + x_edges[1:]) / 2
        width = x_edges[1] - x_edges[0]
        
        bar = pg.BarGraphItem(x=x_centers, height=y, width=width*0.9, brush=color, pen=None)
        self.plot_widget.addItem(bar)
        self.plot_widget.autoRange()

    def _add_analysis_module(self, title, btn_text, callback):
        """輔助函數：快速建立標準的物理量計算區塊"""
        group = QGroupBox(title)
        group.setStyleSheet("QGroupBox { border: 1px solid #444; margin-top: 5px; } QGroupBox::title { color: #999; }")
        layout = QHBoxLayout(group)
        layout.setContentsMargins(5, 15, 5, 5) # 上邊距留給 Title
        
        btn = QPushButton(btn_text)
        btn.clicked.connect(callback)
        layout.addWidget(btn)
        
        self.layout_modules.addWidget(group)
        self.module_widgets.append(group)
        return group

    def _set_modules_enabled(self, enabled: bool):
        """統一啟用或禁用所有物理量計算模組"""
        for widget in self.module_widgets:
            widget.setEnabled(enabled)

    def run_set_projector(self):
        """Step 1: 設定佔據態並生成 Projector"""
        if self.analyzer is None: return
        
        try:
            e_min = self.spin_emin.value()
            e_max = self.spin_emax.value()
            
            self.log(f"Setting occupied state: [{e_min}, {e_max}]...", "INFO")
            
            # 1. 計算 P
            self.analyzer.set_filled_states(e_min, e_max)
            
            # 2. 更新 UI 顯示
            n_occ = np.sum(self.analyzer.occupancy_mask)
            n_total = len(self.analyzer.eigenvalues)
            ratio = (n_occ/n_total)*100
            self.lbl_occ_info.setText(f"Occupied: {n_occ} / {n_total} ({ratio:.1f}%)")
            self.lbl_occ_info.setStyleSheet("color: #4CAF50; font-weight: bold; margin-left: 5px;")
            
            # 3. 解鎖物理量計算區塊
            self._set_modules_enabled(True)
            self.log("Projector ready. Choose a quantity to calculate.", "INFO")
            
        except Exception as e:
            self.log(f"Set Projector Error: {e}", "ERROR")
            traceback.print_exc()
            self._set_modules_enabled(False) # 失敗則鎖定

    def _visualize_generic(self, mode, **kwargs):
        """核心繪圖調度邏輯 (合併原本的 visualize_quantity)"""
        if self.analyzer is None or self.analyzer.Projector is None:
            self.log("Projector missing. Please click 'Set' first.", "WARN")
            return

        try:
            pos = self.current_geometry.positions
            
            if mode == "charge":
                self.log("Calculating Charge Density...", "INFO")
                values = self.analyzer.calculate_charge_density()
                self._configure_plot("Charge Density ρ(r)", "X", "Y", background="#9C9C9C", text_color="#000000", lock_aspect=True)
                self._plot_scatter_heatmap(pos, values, cmap='Blues', adjust_range=(0,1))
                
            elif mode == "chern":
                self.log("Calculating Local Chern Marker...", "INFO")
                values = self.analyzer.calculate_chern_marker()
                self._configure_plot("Local Chern Marker C(r)", "X", "Y", background="#9C9C9C", text_color="#000000", lock_aspect=True)
                self._plot_scatter_heatmap(pos, values, cmap='bwr', adjust_range=True)
                
            elif mode == "am":
                cn = kwargs.get('cn', 5)
                bins = kwargs.get('bins', 100)
                self.log(f"Calculating AM Spectrum (C{cn})...", "INFO")
                l_vals = self.analyzer.get_am_spectrum(Cn=cn)
                self._configure_plot(f"AM Spectrum (C{cn})", "Lz", "Count", background='#121212', text_color='#CCC')
                self._plot_histogram(l_vals, color='#FF5722', bins=bins)

            elif mode == "sf":
                q_r = kwargs.get('q_range', 20.0)
                res = kwargs.get('res', 400)
                gpu = kwargs.get('use_gpu', True)
                
                self.log(f"Calculating Structure Factor (Range: ±{q_r}, Res: {res})...", "INFO")
                
                # 呼叫後端 (回傳 2D array 和 extent)
                sf_data, extent = self.analyzer.calculate_structure_factor(q_range=q_r, resolution=res, use_gpu=gpu)
                
                # 設定繪圖區
                self._configure_plot(
                    "Structure Factor S(q)", "qx", "qy", 
                    background='#000000', # S(q) 通常背景全黑看起來比較清楚
                    text_color='#CCC',
                    lock_aspect=True      # q空間也需要鎖定比例
                )
                
                # 呼叫專用的 Image 繪圖函式
                self._plot_image_heatmap(sf_data, extent, cmap='inferno') # inferno 適合看亮點

        except Exception as e:
            self.log(f"Calc Error ({mode}): {e}", "ERROR")
            traceback.print_exc()

    def estimate_resources(self, N: int):
        """
        估算資源消耗
        Returns: (memory_gb, time_sec_str, is_heavy)
        """
        # 1. 記憶體估算 (Dense Matrix Complex128)
        # N^2 * 16 bytes
        mem_bytes = (N ** 2) * 16
        mem_gb = mem_bytes / (1024**3)
        
        # 2. 時間估算 (O(N^3))
        # 基準：Ryzen 5900X, N=2000 ~ 1 sec (視 NumPy 優化而定)
        # t ~ (N/2000)^3
        t_est = (N / 2000.0) ** 3
        
        if t_est < 1:
            time_str = "< 1 sec"
        elif t_est < 60:
            time_str = f"~ {t_est:.1f} sec"
        else:
            time_str = f"~ {t_est/60:.1f} min"
            
        is_heavy = (N > 5000) # 閾值設為 5000 (約需 15秒, 400MB)
        return mem_gb, time_str, is_heavy
    
    def log(self, message, level="INFO"):
        """
        將訊息輸出到右下角的 Terminal
        Args:
            message (str): 訊息內容
            level (str): INFO, WARN, ERROR
        """
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        
        # 根據等級設定顏色 (HTML formatting)
        color_map = {
            "INFO": "#4CAF50",  # Green
            "WARN": "#FFC107",  # Yellow
            "ERROR": "#F44336", # Red
            "DATA": "#2196F3"   # Blue
        }
        color = color_map.get(level, "#cccccc")
        
        formatted_msg = f'<span style="color:#555;">[{timestamp}]</span> <span style="color:{color}; font-weight:bold;">[{level}]</span> {message}'
        
        self.terminal.append(formatted_msg)
        
        # 自動捲動到底部
        scrollbar = self.terminal.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _configure_plot(self, title, x_label, y_label, background='#121212', text_color='#AAAAAA', lock_aspect=False):
            """
            通用繪圖設定：在每次繪圖前呼叫此函數來重置樣式。
            """
            # 1. 設定背景
            self.plot_widget.setBackground(background)
            self.plot_widget.setAspectLocked(lock_aspect)
            # 2. 設定標題
            self.plot_widget.setTitle(title, color=text_color, size='14pt')
            
            # 3. 設定軸標籤樣式
            label_style = {'color': text_color, 'font-size': '12pt', 'font-weight': 'bold'}
            self.plot_widget.setLabel('bottom', x_label, **label_style)
            self.plot_widget.setLabel('left', y_label, **label_style)
            
            # 4. 設定軸線與刻度顏色 (確保與背景有對比度)
            axis_pen = pg.mkPen(color=text_color, width=1)
            
            # 底部軸 (X)
            ax_b = self.plot_widget.getAxis('bottom')
            ax_b.setPen(axis_pen)
            ax_b.setTextPen(text_color)
            
            # 左側軸 (Y)
            ax_l = self.plot_widget.getAxis('left')
            ax_l.setPen(axis_pen)
            ax_l.setTextPen(text_color)
            
            # 5. 清除舊內容
            self.plot_widget.clear()
    
    def _on_scatter_hovered(self, obj, points):
        """
        當滑鼠懸停在散點上時觸發。
        obj: 觸發事件的 ScatterPlotItem
        points: 被懸停的點的列表 (list of SpotItem)
        """
        if len(points) > 0:
            # 取得第一個被指到的點 (通常滑鼠只會指到一個)
            point = points[0]
            
            # 讀取座標
            pos = point.pos()
            x, y = pos.x(), pos.y()
            
            # 讀取我們存入的數據 (透過 .data() 方法)
            val = point.data()
            
            # 設定 Tooltip 顯示資訊
            # 格式: X: 1.234, Y: 5.678, Val: 0.9876
            tooltip_text = f"X: {x:.4f}\nY: {y:.4f}\nValue: {val:.4e}"
            
            # 將 Tooltip 設定給 PlotWidget
            self.plot_widget.setToolTip(tooltip_text)
        else:
            # 如果滑鼠移開了，清除 Tooltip 或顯示預設訊息
            self.plot_widget.setToolTip("")

    def run_calc_sf(self):
            """觸發結構因子計算"""
            q_range = self.spin_qrange.value()
            res = self.spin_qres.value()
            use_gpu = self.check_gpu.isChecked()
            
            self._visualize_generic("sf", q_range=q_range, res=res, use_gpu=use_gpu)

    def _plot_image_heatmap(self, data, extent, cmap='inferno' , adjust_range: Union[None, Tuple[float, float]] = None):
        """
        繪製規則網格熱圖 (For Structure Factor)
        Args:
            data: 2D numpy array
            extent: [xmin, xmax, ymin, ymax]
            cmap: matplotlib colormap name
            adjust_range: (Tuple[float, float] or None)是否根據數據自動調整顏色範圍
        """
        self.plot_widget.clear()
        
        # 1. 處理 Colormap
        import matplotlib.cm as cm
        try:
            # 獲取 colormap (0~1 -> RGBA)
            colormap = cm.get_cmap(cmap)
            # 建立 lookup table (256x4)
            lut = (colormap(np.linspace(0, 1, 256)) * 255).astype(np.uint8)
        except:
            lut = None # Fallback to default grey
            
        # 2. 建立 ImageItem
        # PyQtGraph 的 ImageItem 預設是 (width, height)，且轉置的
        # 我們通常需要轉置 data 以符合直覺 (x軸對應 column, y軸對應 row)
        # 注意: data.T 或是保持原樣取決於後端計算時 x, y 的順序
        # 假設 calculate_structure_factor 產出的是 [ix, iy]，則不需要轉置
        
        img_item = pg.ImageItem(data)
        
        # 套用 Colormap
        if lut is not None:
            img_item.setLookupTable(lut)
            
        # 3. 設定座標範圍 (Transform)
        # 預設 ImageItem 畫在 [0, width] x [0, height]
        # 我們需要將其縮放和平移到 extent 指定的範圍
        xmin, xmax, ymin, ymax = extent
        
        width = xmax - xmin
        height = ymax - ymin
        data_width = data.shape[0]
        data_height = data.shape[1]
        
        # 計算縮放比例
        scale_x = width / data_width
        scale_y = height / data_height
        
        # 設定變換矩陣: 先縮放，再平移
        tr = QTransform()
        tr.translate(xmin, ymin)
        tr.scale(scale_x, scale_y)
        img_item.setTransform(tr)
        
        self.plot_widget.addItem(img_item)
        
        # 加入簡單的 Colorbar 說明 (Log)
        v_max = np.max(data)
        self.log(f"Max Intensity: {v_max:.4e}", "DATA")
        
        # 加入 Hover 顯示數值 (Optional, 針對 ImageItem 比較複雜，這裡先略過或簡單實作)
        self.plot_widget.autoRange()

        # 儲存目前的 image item 引用，供 hover event 使用
        self.current_image_item = img_item 
        
        # 連接滑鼠移動事件 (如果還沒連接過)
        if not hasattr(self, '_hover_connected'):
            self.plot_widget.scene().sigMouseMoved.connect(self._on_image_hover)
            self._hover_connected = True

    def _on_image_hover(self, pos):
        """處理 ImageItem 的滑鼠懸停數值顯示"""
        # 檢查是否當前是畫 Image 模式
        if not hasattr(self, 'current_image_item') or self.current_image_item is None:
            return

        # 將視窗座標轉為 Plot 座標
        mouse_point = self.plot_widget.plotItem.vb.mapSceneToView(pos)
        x, y = mouse_point.x(), mouse_point.y()
        
        # 嘗試從 ImageItem 獲取對應的 pixel value
        # mapFromScene 比較複雜，這裡用簡單的座標反推
        # 假設我們知道 extent 和 data shape... 
        # 為了簡化，pyqtgraph 提供了 mapFromView 嗎？沒有直接的。
        
        # 這裡用更簡單的方法：利用我們存在 plot_widget 裡的 data (如果有的話)
        # 或者直接顯示座標就好，數值看顏色
        self.plot_widget.setToolTip(f"qx: {x:.2f}, qy: {y:.2f}")

    # --- 獨立的計算觸發函數 ---

    def run_calc_charge(self):
        self._visualize_generic("charge")

    def run_calc_chern(self):
        self._visualize_generic("chern")

    def run_calc_am(self):
        # 讀取當前的 Cn 設定
        cn_val = self.spin_cn.value()
        bins_val = self.spin_am_bins.value()
        self._visualize_generic("am", cn=cn_val)
