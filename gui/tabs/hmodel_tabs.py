# gui/tabs/physics_tab.py

import datetime
import traceback
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSpinBox,
                             QPushButton, QSplitter, QTextEdit, QScrollArea, 
                             QGroupBox, QFileDialog, QMessageBox, QComboBox, 
                             QDoubleSpinBox, QGridLayout)
from PyQt6.QtCore import Qt 
from PyQt6.QtGui import QFont 
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
        self.controls_layout.addWidget(title)
        title.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        # -------------------------------------------------
        # A. 資料來源區塊 (Geometry Source)
        # -------------------------------------------------
        self.group_source = QGroupBox("1. Geometry Source")
        self.group_source.setStyleSheet("QGroupBox { font-weight: bold; border: 1px solid #555; margin-top: 10px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }")
        layout_source = QVBoxLayout()

        # 狀態顯示 Label
        self.lbl_status = QLabel("Status: No Geometry Loaded")
        self.lbl_status.setStyleSheet("color: #FF5555; font-weight: bold;")
        self.lbl_info = QLabel("Sites: 0 | Edges: 0")
        self.lbl_info.setStyleSheet("color: #888;")

        # 按鈕：從 Tab 1 匯入
        self.btn_import = QPushButton("Import from Generator Tab")
        self.btn_import.clicked.connect(self.load_from_memory)
        
        # 按鈕：從檔案讀取
        self.btn_load_file = QPushButton("Load from .npz File")
        self.btn_load_file.clicked.connect(self.load_from_file)

        layout_source.addWidget(self.lbl_status)
        layout_source.addWidget(self.lbl_info)
        layout_source.addWidget(self.btn_import)
        layout_source.addWidget(self.btn_load_file)
        
        self.group_source.setLayout(layout_source)
        self.controls_layout.addWidget(self.group_source)

        # -------------------------------------------------
        # B. Hamiltonian 設定區塊 (預設鎖定)
        # -------------------------------------------------
        self.group_hamiltonian = QGroupBox("2. Hamiltonian Settings")
        self.group_hamiltonian.setStyleSheet("QGroupBox { font-weight: bold; border: 1px solid #555; margin-top: 10px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }")
        
        # 使用 QVBoxLayout 作為主要容器
        self.layout_ham_main = QVBoxLayout()
        
        # --- Row 1: On-site Term ---
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("On-site:"))
        
        self.combo_onsite = QComboBox()
        self.combo_onsite.addItems(['None', 'Laplace', 'Bipartite'])
        row1.addWidget(self.combo_onsite, 2) # Stretch factor 2
        
        row1.addWidget(QLabel("Val:"))
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
        
        # 這裡放置一個容器，之後動態填入 Grid
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
        # 3. Electronic State & Analysis 
        # -------------------------------------------------------------
        self.group_state = QGroupBox("3. Electronic State & Analysis")
        self.group_state.setStyleSheet("QGroupBox { font-weight: bold; border: 1px solid #555; margin-top: 10px; } " \
        "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }")
        layout_state = QVBoxLayout()

        # === 階段 A: 設定佔據態 (Occupancy) ===
        # 使用 Frame 包起來，強調這是第一步
        frame_occ = QFrame()
        frame_occ.setStyleSheet("background-color: #2a2a2a; border-radius: 5px;")
        layout_occ = QVBoxLayout(frame_occ)
        
        # Row 1: 輸入與按鈕
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
        self.btn_set_projector.setFixedWidth(50)
        self.btn_set_projector.setStyleSheet("background-color: #E65100; color: white; font-weight: bold;")
        self.btn_set_projector.clicked.connect(self.run_set_projector)
        
        row_range.addWidget(QLabel("Range:",styleSheet=" color:#FFFFFF; font-weight:bold;"))
        row_range.addWidget(self.spin_emin)
        row_range.addWidget(self.spin_emax)
        row_range.addWidget(self.btn_set_projector)
        layout_occ.addLayout(row_range)
        
        # Row 2: 狀態資訊
        self.lbl_occ_info = QLabel("Status: Projector not set")
        self.lbl_occ_info.setStyleSheet("color: #aaa; font-size: 11px; margin-left: 5px;")
        layout_occ.addWidget(self.lbl_occ_info)
        
        layout_state.addWidget(frame_occ)
        layout_state.addWidget(HOLine())

        # === 階段 B: 物理量計算模組 (Analysis Modules) ===
        # 這裡放置縱向排列的功能塊
        self.layout_modules = QVBoxLayout()
        self.layout_modules.setSpacing(10)

        # 1. Charge Density
        self._add_analysis_module(
            title="Charge Density",
            btn_text="Plot ρ(r)",
            callback=self.run_calc_charge
        )

        # 2. Angular Momentum (需要額外參數 Cn)
        # 我們自訂這個區塊因為它有參數輸入
        frame_am = QGroupBox("Angular Momentum")
        frame_am.setStyleSheet("QGroupBox { border: 1px solid #444; margin-top: 5px; } QGroupBox::title { color: #ccc; }")
        l_am = QHBoxLayout(frame_am)
        l_am.addWidget(QLabel("Sym (Cn):"))
        self.spin_cn = QSpinBox(); self.spin_cn.setRange(1, 20); self.spin_cn.setValue(5)
        l_am.addWidget(self.spin_cn)
        btn_am = QPushButton("Plot Spectrum")
        btn_am.clicked.connect(self.run_calc_am)
        l_am.addWidget(btn_am)
        self.layout_modules.addWidget(frame_am)
        # 註冊到 module_widgets 以便統一啟用/禁用
        self.module_widgets.append(frame_am)

        # 3. Chern Marker
        self._add_analysis_module(
            title="Chern Marker",
            btn_text="Calc & Plot C(r)",
            callback=self.run_calc_chern
        )
        
        # 4. 回到 DOS (Optional)
        btn_back_dos = QPushButton("Back to DOS View")
        btn_back_dos.clicked.connect(self._plot_dos_histogram)
        self.layout_modules.addWidget(btn_back_dos)

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

        # Plot Widget
        self.plot_widget = pg.PlotWidget(background='#1e1e1e') # 給個預設值就好
        right_splitter.addWidget(self.plot_widget)

        # Terminal
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
    #  Logic Methods
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
        
        try:
            # -----------------------------------------------------
            # 1. 讀取參數 & 建構 Hamiltonian (Sparse)
            # -----------------------------------------------------
            # (...參數讀取邏輯同前...)
            onsite_mode = self.combo_onsite.currentText().lower()
            onsite_val = self.spin_onsite_val.value()
            onsite_config = None
            if onsite_mode == 'laplace': onsite_config = {'type': 'laplace', 'scale': onsite_val}
            elif onsite_mode == 'bipartite': onsite_config = {'type': 'bipartite', 's': onsite_val}

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
            
            # 解鎖 Electronic State 區塊
            self.group_state.setEnabled(True)
            self.log("Hamiltonian solved. You can now set occupied states.", "INFO")
        except Exception as e:
            self.log(f"Process Error: {e}", "ERROR")
            traceback.print_exc()

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
        
        # --- 2. 計算 Histogram 數據 ---
        n_bins = int(np.sqrt(len(evals))) + 10
        if n_bins > 500: n_bins = 500
        
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

    def _plot_scatter_heatmap(self, pos, values, cmap='viridis', symmetrical=False):
        """
        輔助函數：畫散點熱圖
        """
        self.plot_widget.clear()
        
        # 1. 數值正規化 (Normalize)
        if symmetrical:
            # 對稱範圍 (例如 -1 到 1)，適合看 Chern Marker
            limit = max(abs(np.min(values)), abs(np.max(values)))
            if limit == 0: limit = 1
            norm_values = (values + limit) / (2 * limit) # 映射到 0~1
        else:
            # 一般範圍 (0 到 Max)，適合看 Charge
            v_min, v_max = np.min(values), np.max(values)
            if v_max == v_min: norm_values = np.zeros_like(values)
            else: norm_values = (values - v_min) / (v_max - v_min)

        # 2. 產生顏色筆刷 (Brushes)
        # PyQtGraph 內建了一些 colormap，也可以自己定義
        # 這裡我們手動用 matplotlib 的 colormap (如果有的話) 或簡單插值
        import matplotlib.cm as cm
        colormap = cm.get_cmap(cmap)
        
        # 轉換為 QColor
        brushes = []
        for val in norm_values:
            # colormap 回傳 (r, g, b, a) 範圍 0-1
            rgba = colormap(val) 
            color = [int(c * 255) for c in rgba]
            brushes.append(pg.mkBrush(color[0], color[1], color[2], 255))
            
        # 3. 繪製
        # 為了效能，PyQtGraph 的 setData 支援 list of brushes
        self.plot_widget.addItem(pg.ScatterPlotItem(
            x=pos[:, 0], y=pos[:, 1],
            size=8, # 點的大小
            brush=brushes,
            pen=None, # 無邊框
            hoverable=True # 滑鼠懸停顯示數值 (進階功能，需額外實作)
        ))
        
        # 加入 Colorbar (雖然 PyQtGraph 加 Colorbar 比較麻煩，這裡先用 Log 顯示範圍)
        self.log(f"Value Range: [{np.min(values):.4f}, {np.max(values):.4f}]", "DATA")
        self.plot_widget.autoRange()

    def _plot_histogram(self, data, color):
        """輔助函數：畫直方圖"""
        self.plot_widget.clear()
        if len(data) == 0: return
        
        # 自動判定bins數量
        n_bins = int(np.sqrt(len(data)))
        if n_bins > 500: n_bins = 500

        y, x_edges = np.histogram(data, bins=n_bins)
        x_centers = (x_edges[:-1] + x_edges[1:]) / 2
        width = x_edges[1] - x_edges[0]
        
        bar = pg.BarGraphItem(x=x_centers, height=y, width=width*0.9, brush=color, pen=None)
        self.plot_widget.addItem(bar)
        self.plot_widget.autoRange()

    def _add_analysis_module(self, title, btn_text, callback):
        """輔助函數：快速建立標準的物理量計算區塊"""
        group = QGroupBox(title)
        group.setStyleSheet("QGroupBox { border: 1px solid #444; margin-top: 5px; } QGroupBox::title { color: #ccc; }")
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
            
            # 3. ★ 解鎖物理量計算區塊 ★
            self._set_modules_enabled(True)
            self.log("Projector ready. Choose a quantity to calculate.", "INFO")
            
        except Exception as e:
            self.log(f"Set Projector Error: {e}", "ERROR")
            traceback.print_exc()
            self._set_modules_enabled(False) # 失敗則鎖定

    # --- 獨立的計算觸發函數 ---

    def run_calc_charge(self):
        self._visualize_generic("charge")

    def run_calc_chern(self):
        self._visualize_generic("chern")

    def run_calc_am(self):
        # 讀取當前的 Cn 設定
        cn_val = self.spin_cn.value()
        self._visualize_generic("am", cn=cn_val)

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
                self._configure_plot("Charge Density ρ(r)", "X", "Y", background='#FFFFFF', text_color='#000000')
                self._plot_scatter_heatmap(pos, values, cmap='viridis')
                
            elif mode == "chern":
                self.log("Calculating Local Chern Marker...", "INFO")
                values = self.analyzer.calculate_chern_marker()
                self._configure_plot("Local Chern Marker C(r)", "X", "Y", background='#FFFFFF', text_color='#000000')
                self._plot_scatter_heatmap(pos, values, cmap='bwr', symmetrical=True)
                
            elif mode == "am":
                cn = kwargs.get('cn', 5)
                self.log(f"Calculating AM Spectrum (C{cn})...", "INFO")
                l_vals = self.analyzer.get_am_spectrum(Cn=cn)
                self._configure_plot(f"AM Spectrum (C{cn})", "Lz", "Count", background='#121212', text_color='#CCC')
                self._plot_histogram(l_vals, color='#FF5722')

        except Exception as e:
            self.log(f"Calc Error ({mode}): {e}", "ERROR")
            traceback.print_exc()

    # =================================================
    #  Helper Methods
    # =================================================
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

    def _configure_plot(self, title, x_label, y_label, background='#121212', text_color='#AAAAAA'):
            """
            通用繪圖設定：在每次繪圖前呼叫此函數來重置樣式。
            """
            # 1. 設定背景
            self.plot_widget.setBackground(background)
            
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