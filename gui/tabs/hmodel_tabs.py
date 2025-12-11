# gui/tabs/physics_tab.py

import datetime
import traceback
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
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

        self.controls_layout.addStretch()

        # =================================================
        # 2. 右側顯示區
        # =================================================
        right_splitter = QSplitter(Qt.Orientation.Vertical)
        right_splitter.setHandleWidth(5)

        # Plot Widget
        self.plot_widget = pg.PlotWidget(background='#1e1e1e')
        self.plot_widget.showGrid(x=True, y=True)
        self.graph_item = pg.PlotDataItem()
        self.plot_widget.addItem(self.graph_item)
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

    def run_construction(self):
        """讀取 UI 數值並呼叫 Model 建構矩陣"""
        if self.hamiltonian_model is None: return
        
        try:
            # 1. 讀取 On-site
            onsite_mode = self.combo_onsite.currentText().lower() # none, laplace, bipartite
            onsite_val = self.spin_onsite_val.value()
            
            onsite_config = None
            if onsite_mode == 'laplace':
                onsite_config = {'type': 'laplace', 'scale': onsite_val}
            elif onsite_mode == 'bipartite':
                onsite_config = {'type': 'bipartite', 's': onsite_val}

            # 2. 讀取 Magnetic Field
            gauge_mode = self.combo_gauge.currentText() # Landau, Circular, Off
            b_flux = self.spin_flux.value()
            if gauge_mode == 'Off': 
                b_flux = 0.0
                gauge_mode = 'Landau' # Avoid error in backend

            # 3. 讀取 Hopping Parameters (ttdict)
            ttdict = {}
            # 注意: 這裡我們需要反查 name，因為 construct() 預期 ttdict[name] = value
            # 但我們的 UI 是基於 type_id 建立的
            
            type_map = self.current_geometry.edge_type_map
            if not type_map: type_map = {0: 'Default'}

            for type_id, widgets in self.hopping_widgets.items():
                re_val = widgets[0].value()
                im_val = widgets[1].value()
                complex_val = re_val + 1j * im_val
                
                name = type_map.get(type_id, f"Unknown_{type_id}")
                ttdict[name] = complex_val
                
            self.log(f"Constructing H with ttdict={ttdict}...", "INFO")

            # 4. 呼叫後端
            H = self.hamiltonian_model.construct(
                ttdict=ttdict,
                b_field=b_flux,
                vec_pot_mode=gauge_mode,
                onsite_config=onsite_config
            )
            
            # 5. 回饋
            sparsity = H.nnz / (H.shape[0]**2) * 100
            self.log(f"Hamiltonian Constructed. Shape: {H.shape}, Sparsity: {sparsity:.4f}%", "INFO")
            
            # (Optional) 可以在此處可視化 H 的 sparsity pattern
            # self.graph_item.clear()
            # self.plot_widget.setTitle("Hamiltonian Sparsity Pattern")
            # rows, cols = H.nonzero()
            # self.graph_item.setData(x=cols, y=rows, pen=None, symbol='s', symbolSize=2, brush='w')
            # self.plot_widget.invertY(True) # 矩陣習慣 (0,0) 在左上
        except RuntimeError as re:
            # Bipartite 失敗的錯誤
            if "Bipartite check failed" in str(re):
                self.log(f"Construction Aborted: {re}", "ERROR")
                QMessageBox.critical(self, "Physical Error", 
                    "Cannot apply Bipartite Potential.\n\n"
                    "The lattice structure is not bipartite under the defined Nearest Neighbor connections.\n"
                    "Please check your geometry or choose 'None'/'Laplace' for On-site term.")
            else:
                self.log(f"Runtime Error: {re}", "ERROR")

        except Exception as e:
            self.log(f"Construction Error: {e}", "ERROR")
            traceback.print_exc()
    
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

        except Exception as e:
            self.log(f"Process Error: {e}", "ERROR")
            traceback.print_exc()

    def _plot_dos_histogram(self):
        """利用 Analyzer 數據畫直方圖"""
        if self.analyzer.eigenvalues is None: return
        
        evals = self.analyzer.eigenvalues
        
        # 1. 計算 Histogram 數據
        # bins 可以自動決定，或者讓使用者設定。這裡先用 sqrt(N) 或固定值
        n_bins = int(np.sqrt(len(evals))) + 10
        if n_bins > 200: n_bins = 200
        
        y, x_edges = np.histogram(evals, bins=n_bins)
        x_centers = (x_edges[:-1] + x_edges[1:]) / 2
        width = x_edges[1] - x_edges[0]
        
        # 2. 清除舊圖並繪製
        self.plot_widget.clear()
        
        # 使用 BarGraphItem
        # brush='b' (藍色填充), pen='w' (白色邊框)
        bar_item = pg.BarGraphItem(x=x_centers, height=y, width=width, brush='#4a90e2', pen=None)
        self.plot_widget.addItem(bar_item)
        
        # 自動調整視角
        self.plot_widget.autoRange()
        
        # 標記一些統計資訊 (Optional)
        # self.log(f"Energy Range: [{evals.min():.4f}, {evals.max():.4f}]", "DATA")
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

