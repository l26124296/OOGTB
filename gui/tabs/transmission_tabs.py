import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QProgressBar,
                            QPushButton, QFrame, QSplitter, QComboBox, QScrollArea,
                            QDoubleSpinBox, QSpinBox, QGroupBox, QListWidget,
                            QAbstractItemView, QListWidgetItem
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QFont
from gui.utils import HOLine

# 呼叫子視窗類別
from gui.tabs.transmission_subtab.trans_subtabs import TransmissionAnalyzerWindow
# 呼叫後端
from core.Transmission import GFCalculator , Analyzer
class TransmissionTab(QWidget):
    sig_energy_selected = pyqtSignal(float)
    sig_site_clicked = pyqtSignal(int)

    def __init__(self, parent=None, *args, **kwargs):
        super().__init__(parent)
        # 1. 接收傳入的 physics_tab
        self.physics_tab = kwargs.get('physics_tab', None)
        self.analyzer_subtab = kwargs.get('analyzer_tab', None)

        self.transtab_layout = QHBoxLayout(self)
        # 2. Leads
        self.lead_candidates = {}
        self.lead_label_items = []
        self.deviceInput = None
        # --- Lead Model 參數儲存 ---
        self.lead_model_type = "WBL"  # 選項: "WBL", "1D-Chain"
        self.val_gamma = 0.5          # WBL Gamma
        self.val_t_coupling = -1.0    # 1D Chain coupling to system
        self.val_t_chain = -1.0       # 1D Chain hopping inside lead
        # 暫存變數
        self.site_positions = None # 儲存晶格座標
        self.current_arrows_group = None
        self.analyzer_window = None
        self.last_gf_bank = None # 儲存最後的 Green's Function Bank
        # 初始化 UI
        self._init_left_panel()
        self._init_right_panel()
        
        # 嘗試初始刷新狀態
        self.refresh_model_status()
    
    def _init_left_panel(self):
        """左側控制面板"""
        self.left_panel = QScrollArea()
        self.left_panel.setFrameShape(QFrame.Shape.StyledPanel)
        self.left_panel.setFixedWidth(320)
        
        left_layout = QVBoxLayout(self.left_panel)

        title = QLabel("<h2>Transport</h2>")
        title.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        left_layout.addWidget(title)
        left_layout.addWidget(HOLine())
        left_layout.addWidget(HOLine())
        
        # --- 0. System Status (新增區塊) ---

        subtitle = QLabel("<h3>Device Settings</h3>")
        subtitle.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        left_layout.addWidget(subtitle)

        status_group = QGroupBox("System Status:")
        status_layout = QVBoxLayout(status_group)
        
        self.lbl_stat_sites = QLabel("Sites (N): -")
        self.lbl_stat_nnz = QLabel("Non-zeros: -")
        self.lbl_stat_sparsity = QLabel("Sparsity: -")
        self.lbl_status_msg = QLabel("") # 用於顯示警告訊息
        self.lbl_status_msg.setStyleSheet("color: red; font-size: 11px;")
        self.lbl_status_msg.setWordWrap(True)
        
        status_layout.addWidget(self.lbl_stat_sites)
        status_layout.addWidget(self.lbl_stat_nnz)
        status_layout.addWidget(self.lbl_stat_sparsity)
        status_layout.addWidget(self.lbl_status_msg)
        
        
        # 增加一個手動刷新按鈕 (方便除錯或強制同步)
        btn_refresh = QPushButton("Sync from Tab-2")
        btn_refresh.clicked.connect(self.refresh_model_status)
        status_layout.addWidget(btn_refresh)
        
        left_layout.addWidget(status_group)

        # --- Lead Definition ---
        lead_group = QGroupBox("Lead Definition")
        self.lead_layout = QVBoxLayout(lead_group)
        
        # Lead Mode(WBL or 1D Chain)
        self._init_lead_config_ui()
        self.lead_layout.addSpacing(10)

        # Leads List
        self.lead_layout.addWidget(QLabel("<b>Contact Sites</b>"))
        self.lead_layout.addWidget(QLabel("Click sites in plot to add/remove:\n" \
        "(Double-click list to edit names)"))
        self.list_leads = QListWidget()
        self.list_leads.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        # 允許使用者雙擊列表項目來編輯名稱
        self.list_leads.itemChanged.connect(self._on_lead_name_edited)
        self.lead_layout.addWidget(self.list_leads)
        
        # 按鈕：清除所有
        btn_clear_leads = QPushButton("Clear All Candidates")
        btn_clear_leads.clicked.connect(self._clear_all_leads)

        self.lead_layout.addWidget(btn_clear_leads)
        left_layout.addWidget(lead_group)
        
        self.transtab_layout.addWidget(self.left_panel)

        # --- Simulation ---
        left_layout.addWidget(HOLine())
        subtitle = QLabel("<h3>Simulation Settings</h3>")
        subtitle.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        left_layout.addWidget(subtitle)

        left_layout.addWidget(QLabel("Energy(min, max, step):"))

        E_setting = QHBoxLayout()
        self.spin_emin = QDoubleSpinBox(); self.spin_emin.setRange(-10.0, 10.0); self.spin_emin.setSingleStep(0.1) ; self.spin_emin.setValue(-5.0)
        self.spin_emax = QDoubleSpinBox(); self.spin_emax.setRange(-10.0, 10.0); self.spin_emax.setSingleStep(0.1) ; self.spin_emax.setValue(5.0)
        self.spin_step = QSpinBox(); self.spin_step.setRange(10, 1000); self.spin_step.setSingleStep(10) ; self.spin_step.setValue(50) 
        E_setting.addWidget(self.spin_emin)
        E_setting.addWidget(self.spin_emax)
        E_setting.addWidget(self.spin_step)
        left_layout.addLayout(E_setting)

        self.prog_bar = QProgressBar()
        self.prog_bar.setRange(0, 100)
        self.prog_bar.setValue(0)
        left_layout.addWidget(self.prog_bar)

        self.lbl_cal_gf_msg = QLabel("")
        left_layout.addWidget(self.lbl_cal_gf_msg)

        self.btn_run_gf = QPushButton("Calculate Green's Function")
        self.btn_run_gf.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold;")
        self.btn_run_gf.setFixedWidth(180)
        self.btn_run_gf.setFixedHeight(40)
        self.btn_run_gf.clicked.connect(self._on_run_gf)
        left_layout.addWidget(self.btn_run_gf, alignment=Qt.AlignmentFlag.AlignCenter)

        left_layout.addWidget(HOLine())
        subtitle = QLabel("<h3>Source Settings</h3>")
        subtitle.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        left_layout.addWidget(subtitle)
        self.btn_open_analyzer = QPushButton("Open Transmission Analyzer")
        self.btn_open_analyzer.setEnabled(False) # 預設停用，直到算出 GF 為止
        self.btn_open_analyzer.setObjectName("cal_TE")
        with open('./gui/tabs/qss/open_analyzer_button.qss', 'r') as f:
            self.btn_open_analyzer.setStyleSheet(f.read())
        self.btn_open_analyzer.clicked.connect(self._open_analyzer_window)
        self.btn_open_analyzer.setFixedWidth(180)
        self.btn_open_analyzer.setFixedHeight(40)
        left_layout.addWidget(self.btn_open_analyzer, alignment=Qt.AlignmentFlag.AlignCenter)

    def _init_right_panel(self):
        """右側繪圖區"""
        splitter = QSplitter(Qt.Orientation.Vertical)
        
        # --- Top Plot: Lattice ---
        self.plot_lattice = pg.PlotWidget(title="Lattice Geometry (Click to select Leads)")
        self.plot_lattice.setBackground('#1e1e1e')
        self.plot_lattice.setAspectLocked(True)
        self.plot_lattice.showGrid(x=True, y=True, alpha=0.5)
        
        # 1. Bonds (GraphItem) - 用於畫靜態邊
        self.bond_graph_item = pg.GraphItem()
        self.plot_lattice.addItem(self.bond_graph_item)

        # 3. Sites (Scatter) - 這是主要交互對象
        # 設定 hoverable=True 讓滑鼠懸停有反應
        self.site_scatter = pg.ScatterPlotItem(
            pxMode=False, 
            hoverable=True, 
            hoverBrush='w', 
            hoverSize=15
        )
        self.site_scatter.sigClicked.connect(self._on_site_clicked)
        self.site_scatter.setZValue(10)
        self.plot_lattice.addItem(self.site_scatter)
        
        # ... (Bottom Plot Transmission 部分省略，維持原樣) ...
        self.plot_trans = pg.PlotWidget(title="Transmission T(E)")
        self.plot_trans.showGrid(x=True, y=False, alpha=0.5)
        self.plot_trans.setLabel('bottom', "Energy (t)")
        self.plot_trans.setLabel('left', "Transmission")

        # 創建垂直標記線 (InfiniteLine)
        # pen設定顏色與樣式
        self.vline = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('r', width=2, style=pg.QtCore.Qt.PenStyle.DashLine))
        self.plot_trans.addItem(self.vline)
        
        # 3. 綁定滑鼠點擊事件
        # 注意：是綁定在 scene() 上
        self.plot_trans.scene().sigMouseClicked.connect(self._on_plot_trans_clicked)
        
        # 用於儲存當前的數據 (這部分由您的 Calculate 完成後填入)
        self.current_energies = None 
        self.selected_energy_index = None
        
        # 鎖定 Y 軸範圍 0-1.1 (符合物理 T <= N_channels)
        self.plot_trans.setYRange(-0.05, 1.1)
        self.plot_trans.setMouseEnabled(x=True, y=False) # 只允許 X 軸縮放/拖曳

        splitter.addWidget(self.plot_lattice)
        splitter.addWidget(self.plot_trans)

        # 設定比例 (70% : 30%)
        splitter.setStretchFactor(0, 7)
        splitter.setStretchFactor(1, 3)
        self.transtab_layout.addWidget(splitter)

        # 在 MainWindow 的 __init__ 中加入：
        self.ant_timer = pg.QtCore.QTimer()
        self.ant_timer.timeout.connect(self.update_ant_animation)
        self.ant_scatter = pg.ScatterPlotItem(pxMode=True, pen=None) # 用於繪製移動的點
        self.plot_lattice.addItem(self.ant_scatter) # 預先加入，之後只更新數據
        self.ant_data = {
            'starts': None, # 起點座標數組
            'diffs': None,  # 向量差 (終點-起點)
            'sizes': None,  # 點的大小
            'phases': None, # 當前進度 (0.0 ~ 1.0)
            'speeds': None  # (可選) 移動速度
        }

    def _init_lead_config_ui(self):
        """
        [新增] 建構 Lead Model 設定區塊 (WBL vs 1D-Chain)
        並將其加入到 parent_layout 中
        """
        
        layout = self.lead_layout
        
        # 1. 模式選擇
        layout.addWidget(QLabel("<b>Lead Model Type</b>"))
        self.combo_lead_mode = QComboBox()
        self.combo_lead_mode.addItems(["Wide-Band Limit (WBL)", "1D Semi-Infinite Chain"])
        self.combo_lead_mode.currentIndexChanged.connect(self._on_lead_mode_changed)
        layout.addWidget(self.combo_lead_mode)
        
        # 建立兩個容器，分別放 WBL 和 Chain 的參數，透過 hide/show 切換
        
        # --- A. WBL 參數區 ---
        self.container_wbl = QWidget()
        wbl_layout = QHBoxLayout(self.container_wbl)
        wbl_layout.setContentsMargins(0, 0, 0, 0)
        
        lbl_gamma = QLabel("Γ (Broadening):") 
        self.spin_gamma = QDoubleSpinBox()
        self.spin_gamma.setRange(0.0, 100.0)
        self.spin_gamma.setSingleStep(0.1)
        self.spin_gamma.setValue(self.val_gamma)
        self.spin_gamma.valueChanged.connect(lambda v: setattr(self, 'val_gamma', v))
        
        wbl_layout.addWidget(lbl_gamma)
        wbl_layout.addWidget(self.spin_gamma)
        layout.addWidget(self.container_wbl)
        
        # --- B. 1D Chain 參數區 ---
        self.container_chain = QWidget()
        chain_layout = QVBoxLayout(self.container_chain)
        chain_layout.setContentsMargins(0, 0, 0, 0)
        
        # Row 1: t_coupling
        row1 = QHBoxLayout()
        lbl_tc = QLabel("t_coupling:")
        self.spin_tc = QDoubleSpinBox()
        self.spin_tc.setRange(-20.0, 20.0)
        self.spin_tc.setSingleStep(0.1)
        self.spin_tc.setValue(self.val_t_coupling)
        self.spin_tc.valueChanged.connect(lambda v: setattr(self, 'val_t_coupling', v))
        row1.addWidget(lbl_tc); row1.addWidget(self.spin_tc)
        
        # Row 2: t_chain
        row2 = QHBoxLayout()
        lbl_tl = QLabel("t_chain:")
        self.spin_tl = QDoubleSpinBox()
        self.spin_tl.setRange(-20.0, 20.0)
        self.spin_tl.setSingleStep(0.1)
        self.spin_tl.setValue(self.val_t_chain)
        self.spin_tl.valueChanged.connect(lambda v: setattr(self, 'val_t_chain', v))
        row2.addWidget(lbl_tl); row2.addWidget(self.spin_tl)
        
        chain_layout.addLayout(row1)
        chain_layout.addLayout(row2)
        
        layout.addWidget(self.container_chain)
        
        # 初始狀態設定 (預設 WBL)
        self.container_chain.hide()

    def _on_lead_mode_changed(self, index):
        """
        當 Lead Model 改變時觸發
        index 0: WBL
        index 1: 1D Chain
        """
        if index == 0:
            self.lead_model_type = "WBL"
            self.container_wbl.show()
            self.container_chain.hide()
        else:
            self.lead_model_type = "1D-Chain"
            self.container_wbl.hide()
            self.container_chain.show()

    # ==========================================================================
    # 功能函數
    # ==========================================================================
    
    def init_lattice_display(self, coords):
        self.site_positions = coords
        self.site_scatter.setData(
            pos=coords, data=np.arange(len(coords))
        )
        self.plot_lattice.autoRange()

    def refresh_model_status(self):
        """
        從 Physics Tab 讀取幾何結構 (geo) 與 Hamiltonian。
        使用 geo.edges 繪製連接邊，使用 geo.sites 繪製原子。
        """
        if not self.physics_tab:
            return

        # 1. 取得 Hamiltonian Model
        model = getattr(self.physics_tab, 'hamiltonian_model', None)
        
        if not model:
            self.lbl_status_msg.setText("No Hamiltonian model found.")
            return

        # 2. 取得幾何物件 (geo)
        geo = getattr(model, 'geo', None)
        if not geo:
            self.lbl_status_msg.setText("No Geometry (geo) found in model.")
            return
        # 3. 檢查Hamiltonian矩陣是否已經建立
        if model.H_sparse is None:
            self.lbl_status_msg.setText("Hamiltonian matrix not constructed.")
            return

        # 3. 處理 Sites (座標)
        coords = None
        # 假設 geo.sites 是一個 list，每個 site 有 .pos 屬性
        if hasattr(geo, 'positions'): 
            try:
                # 提取 (x, y) 座標
                coords = geo.positions
            except Exception as e:
                print(f"Error parsing sites: {e}")

        self.site_positions = coords
        self.lbl_stat_sites.setText(f"Sites: {len(coords)}")

        # 繪製點
        self._refresh_site_visuals()
        
        # 4. 處理 Edges (邊) - 修正：直接使用 geo.edges
        edges = getattr(geo, 'edges', None)
        
        if edges is not None and len(edges) > 0:
            try:
                # 轉換為 numpy array (M, 2)，確保是整數索引
                # 假設 edges 是一個 list of tuples/lists: [(0, 1), (1, 2), ...]
                adj = edges
                
                # 繪製邊：淺灰色，細線
                self.bond_graph_item.setData(
                    pos=coords,
                    adj=adj,
                    pen=pg.mkPen(color='#808080', width=1), # 淺灰，細線
                    size=0, # 不畫節點 (由 scatter 負責)
                )
            except Exception as e:
                print(f"Error parsing edges: {e}")

        # 5. 更新狀態顯示
        self.lbl_status_msg.setText("Geometry Loaded.")
        self.lbl_status_msg.setStyleSheet("color: green;")

        # 6. 繪製 Sites (視覺刷新)
        self.kill_ants()
        self._refresh_site_visuals()
        
        # 7. 更新列表 (保留之前的選擇，但重新驗證索引有效性)
        self._update_lead_list_ui()

    def _set_status_invalid(self, msg):
        """輔助函數：設定為無效狀態"""
        self.lbl_stat_sites.setText("Sites (N): -")
        self.lbl_stat_nnz.setText("Non-zeros: -")
        self.lbl_stat_sparsity.setText("Sparsity: -")
        self.lbl_status_msg.setText(msg)
        self.lbl_status_msg.setStyleSheet("color: red;")

    def _refresh_site_visuals(self):
        """
        根據 self.lead_candidates 的狀態，重新繪製所有 Sites。
        """

        recommand_dot_size = 0.5 * np.linalg.norm(
            self.physics_tab.current_geometry.positions[1] - self.physics_tab.current_geometry.positions[0])
        recommand_dot_size = max(0.2, recommand_dot_size)
        if self.site_positions is None: return
        
        N = len(self.site_positions)
        
        # --- 樣式設定 ---
        # 一般點：藍色圓形，大小適中，容易點擊
        base_brush = pg.mkBrush("#51b2df")
        base_symbol = 'o'
        base_size = recommand_dot_size  # 稍微加大一點，讓滑鼠容易點中
        base_pen = pg.mkPen('k', width=0) # 無邊框

        # Lead 候選點：方形，顯眼
        lead_brush = pg.mkBrush("#0011FF") 
        lead_symbol = 's' # Square
        lead_size = 1.3 * recommand_dot_size    # 比一般點更大
        lead_pen = pg.mkPen("#628188", width=1) # 邊框
        
        # 建立屬性陣列
        brushes = [base_brush] * N
        symbols = [base_symbol] * N
        sizes = [base_size] * N
        pens = [base_pen] * N
        
        # 覆蓋 Lead Sites 的樣式
        for idx in self.lead_candidates.keys():
            if 0 <= idx < N:
                brushes[idx] = lead_brush
                symbols[idx] = lead_symbol
                sizes[idx] = lead_size
                pens[idx] = lead_pen

        # 更新 ScatterPlotItem
        self.site_scatter.setData(
            pos=self.site_positions,
            data=np.arange(N),
            brush=brushes,
            symbol=symbols,
            size=sizes,
            pen=pens,
            hoverSize=1.1 * recommand_dot_size
        )

        self._refresh_lead_labels()

    def _refresh_lead_labels(self):
        """
        清除舊的文字標籤，並根據 lead_candidates 在繪圖區顯示名稱。
        """
        # 1. 清除舊標籤
        for item in self.lead_label_items:
            self.plot_lattice.removeItem(item)
        self.lead_label_items.clear()

        if self.site_positions is None: return

        # 2. 重新繪製新標籤
        for idx, name in self.lead_candidates.items():
            if 0 <= idx < len(self.site_positions):
                pos = self.site_positions[idx]
                
                # 建立文字標籤
                # html 格式可以設定顏色與粗體
                text_item = pg.TextItem(
                    html=f'<div style="text-align: center"><span style="color: #00ff9d; font-weight: bold;">{name}</span></div>',
                    anchor=(0.5, 1) # (0.5, 1) 代表文字框的「底部中間」對齊座標點 (即文字會在點的上方)
                )
                
                # 設定位置
                text_item.setPos(pos[0], pos[1])
                text_item.setZValue(30) # 確保文字在最上層 (比點和線都高)
                
                self.plot_lattice.addItem(text_item)
                self.lead_label_items.append(text_item)

    def _on_site_clicked(self, plot_item, points):
        """
        當繪圖區的原子被點擊時觸發。
        """
        if not points: return
        
        # 獲取點擊的點 index
        click_spot = points[0]
        idx = click_spot.data()
        
        # 邏輯：Toggle (存在則刪除，不存在則加入)
        if idx in self.lead_candidates:
            # 移除
            del self.lead_candidates[idx]
        else:
            # 加入，給予預設名稱
            default_name = f"Lead_Site_{idx}"
            self.lead_candidates[idx] = default_name
            
        # 實時更新
        self._refresh_site_visuals() # 更新圖形顏色
        self._update_lead_list_ui()  # 更新左側列表

    def _update_lead_list_ui(self):
        """
        更新左側 QListWidget 內容
        """
        self.list_leads.clear()
        
        # 排序 index 以保持列表整齊
        sorted_indices = sorted(self.lead_candidates.keys())
        
        for idx in sorted_indices:
            name = self.lead_candidates[idx]
            display_text = f"[{idx}] {name}"
            
            item = QListWidgetItem(display_text)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable) # 允許編輯
            item.setData(Qt.ItemDataRole.UserRole, idx) # 將真實 index 藏在 Item 中
            
            self.list_leads.addItem(item)

    def _on_lead_name_edited(self, item):
        """
        當使用者在左側列表中編輯名稱後，回寫到字典
        """
        idx = item.data(Qt.ItemDataRole.UserRole)
        text = item.text()
        
        # 解析文字，去掉前面的 "[idx] " 格式，保留使用者輸入
        # 只存名稱部分 (格式假設是 "[10] Name"，則只存 Name )

        prefix = f"[{idx}] "
        if text.startswith(prefix):
            clean_name = text[len(prefix):]
        else:
            clean_name = text #前綴已經刪掉
            
        self.lead_candidates[idx] = clean_name
        self._refresh_lead_labels()

    def _clear_all_leads(self):
        self.lead_candidates.clear()
        self._refresh_site_visuals()
        self._update_lead_list_ui()
        
    def _on_lead_mode_changed(self, index):
        """
        當 Lead Model 改變時觸發
        index 0: WBL
        index 1: 1D Chain
        """
        if index == 0:
            self.lead_model_type = "WBL"
            self.container_wbl.show()
            self.container_chain.hide()
        else:
            self.lead_model_type = "1D-Chain"
            self.container_wbl.hide()
            self.container_chain.show()

    def _on_run_gf(self, *args, **kwargs):
        """
        [觸發] 當按下計算按鈕時：
        1. 驗證資料
        2. 打包 lead_config 與 energy_range
        3. 實例化 GFCalculator (QThread) 並啟動
        """
        # 1. 基礎驗證
        if self.physics_tab is None or not hasattr(self.physics_tab, 'hamiltonian_model'):
            self.lbl_cal_gf_msg.setText("Error: No Hamiltonian model found.")
            self.lbl_cal_gf_msg.setStyleSheet("color: red;")
            return
        
        if not self.lead_candidates:
            self.lbl_cal_gf_msg.setText("Error: No leads defined. Please select sites.")
            self.lbl_cal_gf_msg.setStyleSheet("color: red;")
            return

        # 2. 準備能量網格 (numpy array)
        try:
            energies = np.linspace(
                self.spin_emin.value(),
                self.spin_emax.value(),
                self.spin_step.value()
            )
        except Exception as e:
            self.lbl_status_msg.setText(f"Error creating energy grid: {e}")
            return

        lead_config = {
            'mode': self.lead_model_type,        # "WBL" or "1D-Chain"
            'wbl_gamma': self.val_gamma,         # WBL Gamma
            'coupling': self.val_t_coupling,   # Chain coupling
            'chain_hopping': self.val_t_chain,         # Chain hopping
            'leads': self.lead_candidates             # 打包好的 leads 資訊
        }

        # 4. 鎖定 UI 避免重複點擊
        self.btn_run_gf.setEnabled(False)
        self.prog_bar.setValue(0)
        self.lbl_cal_gf_msg.setText("Calculating Green's Function...")
        self.lbl_cal_gf_msg.setStyleSheet("color: green;")

        # 5. 實例化並啟動 GFCalculator (QThread)
        model = self.physics_tab.hamiltonian_model
        
        # 假設您已導入 GFCalculator
        self.gf_worker = GFCalculator(model, lead_config, energies)
        
        # 連接訊號 (Signals)
        self.gf_worker.progress_sig.connect(self._on_worker_progress)
        self.gf_worker.finished_sig.connect(self._on_worker_finished)
        self.gf_worker.error_sig.connect(self._on_worker_error)
        
        # 啟動執行緒
        self.gf_worker.start()

    def _on_worker_progress(self, value):
        """接收進度訊號，更新進度條"""
        self.prog_bar.setValue(value)

    def _on_worker_finished(self, bank):
        """
        計算完成：
        bank 為 GreenFunctionBank 物件 (包含所有能量點的 G^R)
        """
        self.lbl_cal_gf_msg.setText("Calculation Finished.")
        self.lbl_cal_gf_msg.setStyleSheet("color: green;")
        self.btn_run_gf.setEnabled(True)
        self.prog_bar.setValue(100)
        
        # 儲存結果 (如果之後需要存檔或進一步分析)
        self.last_gf_bank = bank

        self.btn_open_analyzer.setEnabled(True)
        
        # 如果視窗已經開著，自動更新它的資料
        if self.analyzer_window is not None or getattr(self.analyzer_window, 'isVisible', False):
            self.analyzer_window.update_data(bank, self.lead_candidates)
        
    def _on_worker_error(self, error_msg):
        """計算發生錯誤"""
        self.lbl_cal_gf_msg.setText(f"Error: {error_msg}")
        self.lbl_cal_gf_msg.setStyleSheet("color: red;")
        self.btn_run_gf.setEnabled(True)
        print(f"GF Calculation Error: {error_msg}")

    def _open_analyzer_window(self):
        """
        開啟或喚醒 Transmission Analyzer 視窗 (Non-Modal)
        """
        if self.last_gf_bank is None:
            return

        # 如果視窗不存在，或者已經被關閉 (PyQt 有時會刪除 C++ 物件)
        if self.analyzer_window is None:
            # 實例化新視窗
            self.analyzer_window = TransmissionAnalyzerWindow(
                gf_bank=self.last_gf_bank,
                lead_candidates=self.lead_candidates
            )
            self.analyzer_window.sig_arrived.connect(self.handle_analyzer_request)
            self.analyzer_window.show() # 使用 .show() 而非 .exec() 讓它變成 Non-Modal
        
        else:
            # 如果視窗已經存在
            # 1. 更新資料 (確保是用最新的)
            self.analyzer_window.update_data(self.last_gf_bank, self.lead_candidates)
            
            # 2. 帶到最上層 (Focus)
            self.analyzer_window.show()
            self.analyzer_window.raise_()
            self.analyzer_window.activateWindow()

    def _on_plot_trans_clicked(self, event):
        """
        處理傳輸圖表的點擊事件
        """
        # 0. 確保有數據且是左鍵點擊
        if self.current_energies is None:
            return
            
        # 檢查是否為滑鼠左鍵
        if event.button() == pg.QtCore.Qt.MouseButton.LeftButton:
            
            # 1. 取得滑鼠在 Scene 中的座標
            mouse_point = event.scenePos()
            
            # 2. 判斷點擊是否在繪圖區域內 (ViewBox)
            plot_item = self.plot_trans.plotItem
            if plot_item.sceneBoundingRect().contains(mouse_point):
                
                # 3. 將 Scene 座標轉換為 Plot (數據) 座標
                mouse_point_view = plot_item.vb.mapSceneToView(mouse_point)
                clicked_energy = mouse_point_view.x()
                
                # 4. 尋找最接近的 Energy Index (核心邏輯)
                # 使用 numpy 的絕對值差最小化
                idx = (np.abs(self.current_energies - clicked_energy)).argmin()
                nearest_energy = self.current_energies[idx]
                
                # 5. 更新狀態
                self.selected_energy_index = idx
                print(f"Selected Energy: {nearest_energy:.4f} (Index: {idx})")
                
                # 6. 移動垂直線到該位置
                self.vline.setPos(nearest_energy)
            
                # 7. 更新繪圖
                self.kill_ants()
                self.draw_ant_map(idx)
    
    def handle_analyzer_request(self, data):
        """
        [主視窗] 處理來自 TransmissionAnalyzerWindow 的計算請求。
        負責調用後端 Analyzer 並更新主視窗繪圖。
        
        Args:
            data (dict): 包含 'mode' 和對應參數 ('source_indices' 或 'wavefunction')
        """
        self.kill_ants()
        # 1. 基礎檢查
        if self.last_gf_bank is None:
            self.lbl_status_msg.setText("Error: No Green's Function data available.")
            return
        
        mode = data.get('mode')
        drain_indices = data.get('drain_indices')
        print(f"Main: Handling Analyzer Request - Mode: {mode}")
        
        # 準備通用參數
        # 注意：計算 T(E) 通常需要 Lead 的參數來計算 Gamma 矩陣
        # 這裡假設您的 Analyzer 需要 lead_config

        lead_config = {
            'mode': self.lead_model_type,
            'wbl_gamma': self.val_gamma,
            'coupling': self.val_t_coupling,
            'chain_hopping': self.val_t_chain,
            'leads': self.lead_candidates 
        }
        
        energies = self.last_gf_bank.energies
        T_values = None
        
        self.lbl_cal_gf_msg.setText(f"Calculating {mode} transmission...")
        self.lbl_cal_gf_msg.setStyleSheet("color: blue;")

        try:
            # =========================================================
            # [待確認] 呼叫後端核心
            # 請根據您提供的函數簽章，確認下方參數傳遞是否正確
            # =========================================================
            
            if mode == 'multichannel':
                # 取得使用者勾選的 Source Leads (Indices list)
                source_indices = data.get('source_indices', [])
                print(f"[DEBUG]Saving Source Indices: {source_indices}")

                # 儲存輸入電極(電流繪製用)
                self.deviceInput = source_indices
                if not source_indices:
                    self.lbl_status_msg.setText("Error: No source channels selected.")
                    return

                # 呼叫後端: 計算 Multi-channel Transmission
                # 假設簽章: (gf_bank, lead_config, source_indices)
                T_values = Analyzer.calculate_transmission(
                    gf_bank=self.last_gf_bank,
                    lead_config=lead_config,
                    source_indices=source_indices,
                    drain_indices=drain_indices
                    )

            elif mode == 'coherent':
                # 取得使用者定義的波函數 {site_idx: complex_val}
                wavefunction_dict = data.get('wavefunction', {})

                # 儲存輸入電極(電流繪製用)
                self.deviceInput = wavefunction_dict
                print('[DEBUG] Saveing source wavefun:', self.deviceInput)
                # 呼叫後端: 計算 Rank-1 Coherent Transmission
                # 假設簽章: (gf_bank, lead_config, wavefunction_dict)
                T_values = Analyzer.calculate_coherence_transmission(
                    gf_bank=self.last_gf_bank,
                    lead_config=lead_config,
                    wavefunction_dict=wavefunction_dict,
                    drain_indices=drain_indices
                )

            # =========================================================
            # 3. 繪圖與收尾
            # =========================================================
            self.current_energies = energies
            
            self._tranmission_plotter(energies, T_values, mode)
            self.lbl_cal_gf_msg.setText(f"{mode} transmission calculated.")
            self.lbl_cal_gf_msg.setStyleSheet("color: green;")

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.lbl_cal_gf_msg.setText(f"Analysis Error: {str(e)}")
            self.lbl_cal_gf_msg.setStyleSheet("color: red;")

    def _tranmission_plotter(self, energies, T_values, mode):
        if T_values is not None:
                self.plot_trans.clear()
                
                self.plot_trans.addItem(self.vline)
                self.selected_energy_index = None
                self.vline.setPos(energies[0]) # 或隱藏
                # 繪製 T(E) 曲線
                self.plot_trans.plot(
                    energies, T_values, 
                    pen=pg.mkPen("#00ffffaa", width=2), # 青色線條
                    fillLevel=0, 
                    brush=pg.mkBrush("#00ffff4c"),
                    name=f"T(E) - {mode}",
                    symbol='o',
                    symbolSize=8,
                    symbolBrush=pg.mkBrush("#b9b9b9")
                )
                
                # 標示圖例或標題
                title = "Multi-channel T(E)" if mode == 'multichannel' else "Coherent T(E)"
                self.plot_trans.setTitle(title)
                self.plot_trans.setLabel('left', "Transmission")
                
                # 更新狀態訊息
                self.lbl_cal_gf_msg.setText(f"Transmission calculated successfully({mode}).")
                self.lbl_cal_gf_msg.setStyleSheet("color: green;")
                self.btn_open_analyzer.setEnabled(True)

    def draw_ant_map(self, energy_index):
        """
        準備螞蟻圖數據並啟動動畫。
        """
        import numpy as np

        # 1. 停止舊的動畫 (如果有的話)
        self.ant_timer.stop()
        self.ant_scatter.clear() 

        # 檢查數據
        if self.last_gf_bank is None:
            return

        # 2. 後端計算 J_matrix (與原本相同)
        try:
            H = self.physics_tab.hamiltonian_model.H_sparse
            lead_config = {
                'mode': self.lead_model_type,
                'wbl_gamma': self.val_gamma,
                'coupling': self.val_t_coupling,
                'chain_hopping': self.val_t_chain,
                'leads': self.lead_candidates 
            }
            source_config = self.deviceInput # 或者是您的 source indices
            
            print(f'[Debug] Calling bond current backend, source config: {source_config}')
            J_matrix = Analyzer.calculate_bond_currents_universal(
                self.last_gf_bank, H, lead_config, source_config, energy_index
            )
        except Exception as e:
            print(f"Error calculating currents: {e}")
            return

        # 3. 提取路徑數據
        coords = np.array(self.physics_tab.current_geometry.positions)
        
        # 設定閾值
        max_current = np.max(np.abs(J_matrix))
        if max_current < 1e-20: 
            return
        threshold = 0.05 * max_current 
        # 找出符合條件的連接 (Start -> End)
        rows, cols = np.where(J_matrix > threshold)
        
        if len(rows) == 0:
            return

        # 4. 向量化準備數據 (Numpy 運算比 Python list 快)
        # 提取起點與終點座標
        points_per_edge = 3
        start_coords = np.repeat(coords[rows], points_per_edge, axis=0) # Shape: (N, 2)
        end_coords = np.repeat(coords[cols], points_per_edge, axis=0)   # Shape: (N, 2)
        
        # 計算向量差 (End - Start)
        diff_vecs = end_coords - start_coords
        
        # 計算每個點的大小 (根據電流大小)
        # 設定點的大小範圍，例如 5px ~ 15px
        current_vals = np.kron(J_matrix[rows, cols], np.ones(points_per_edge))
        normalized_J = current_vals / max_current
        sizes = 5 + 5 * normalized_J 
        
        # 初始化相位 (均勻分佈)
        phases = np.kron(np.ones(len(rows)), np.linspace(0, 1, points_per_edge, endpoint=False))
        # (可選) 讓電流大的跑比較快
        speeds = 0.05 * (0.5 + 0.5 * normalized_J) 

        # 5. 存入 self.ant_data
        self.ant_data['starts'] = start_coords
        self.ant_data['diffs'] = diff_vecs
        self.ant_data['sizes'] = sizes
        self.ant_data['phases'] = phases
        self.ant_data['speeds'] = speeds
        
        # 設定顏色 (小綠 大紅)
        cmap = pg.ColorMap(pos=[0.0, 1.0],color=[(0, 255, 0, 255),(255, 0, 0, 255)])
        colors = cmap.map(normalized_J, mode='byte')  # Nx4 uint8
        self.ant_brush = [pg.mkBrush(*c) for c in colors]

        # 6. 啟動計時器 (每 30ms 更新一次畫面)
        self.ant_timer.start(30)
        print(f"Ant animation started. {len(rows)} paths active.")

    def update_ant_animation(self):
        """
        定時更新螞蟻的位置
        """
        data = self.ant_data
        if data['starts'] is None:
            self.ant_timer.stop()
            return

        # 1. 更新相位 (Phase)
        # t = t + speed
        data['phases'] += data['speeds']
        
        # 循環邏輯：如果 t > 1，則減去 1 回到起點
        data['phases'][data['phases'] > 1.0] -= 1.0
        
        # 2. 計算當前座標
        # Pos = Start + Diff * t
        # 利用 numpy 的廣播機制 (Broadcasting) 進行快速運算
        # phases 需要 reshape 成 (N, 1) 才能乘上 (N, 2) 的座標
        t = data['phases'].reshape(-1, 1)
        current_positions = data['starts'] + data['diffs'] * t
        
        # 3. 更新 ScatterPlotItem
        # setData 是 PyQtGraph 中最高效的更新方式
        self.ant_scatter.setData(
            x=current_positions[:, 0],
            y=current_positions[:, 1],
            size=data['sizes'],
            brush=self.ant_brush
        )

    def kill_ants(self):
        self.ant_timer.stop()
        self.ant_scatter.clear()
        self.ant_data['starts'] = None