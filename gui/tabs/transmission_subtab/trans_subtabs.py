from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QScrollArea, QLabel, QComboBox, 
    QGroupBox, QCheckBox, QTableWidget, QTableWidgetItem, 
    QDoubleSpinBox, QPushButton, QHeaderView, QFrame
)
from PyQt6.QtCore import pyqtSignal, Qt
from numpy import abs, exp  # 移到最上方

class TransmissionAnalyzerWindow(QWidget):
    # 定義訊號：回傳設定參數字典
    sig_arrived = pyqtSignal(dict)

    def __init__(self, gf_bank, lead_candidates, parent=None):
        super().__init__()
        self.setWindowTitle("Transmission Analyzer")
        self.resize(500, 700) # 稍微加高一點
        
        # 1. 初始化變數
        self.gf_bank = gf_bank
        self.lead_candidates = lead_candidates 
        self.sorted_leads = []
        
        # 儲存 UI 參照
        self.chk_inputs = {}     
        self.chk_outputs = {} 
        self.spin_amps = {}      
        self.spin_phases = {}    

        # 2. 建立靜態 UI 骨架
        self._init_static_ui()
        
        # 3. 填入動態內容
        self._refresh_content()

    def update_data(self, gf_bank, lead_candidates):
        """
        更新數據並重繪 UI
        """
        print("[Analyzer]: Updating data and refreshing UI...")
        self.gf_bank = gf_bank
        self.lead_candidates = lead_candidates
        self._refresh_content()

    def _init_static_ui(self):
        """建立靜態 UI 框架"""
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(10)

        # --- A. Mode Selection ---
        main_layout.addWidget(QLabel("<b>1. Select Input Mode (Source):</b>"))
        self.combo_mode = QComboBox()
        self.combo_mode.addItems(["Multi-Channel (Incoherent)", "Coherent Injection (Rank-1)"])
        self.combo_mode.currentIndexChanged.connect(self._on_mode_changed)
        main_layout.addWidget(self.combo_mode)

        # --- B. Input Container (包含 Multi-channel 和 Coherent) ---
        self.input_container = QWidget()
        input_layout = QVBoxLayout(self.input_container)
        input_layout.setContentsMargins(0, 5, 0, 5)
        main_layout.addWidget(self.input_container)

        # B-1. Multi-Channel Input (Scrollable GroupBox)
        self.group_mcI = QGroupBox("Select Input Channels (Source Leads)")
        self.group_mcI_layout = QVBoxLayout(self.group_mcI)
        
        self.Scroll_mcI = QScrollArea()
        self.Scroll_mcI.setWidgetResizable(True)
        self.Scroll_mcI.setFixedHeight(150) # 限制高度，避免佔滿畫面
        
        self.mcI_layout_widget = QWidget()
        self.mcI_layout = QVBoxLayout(self.mcI_layout_widget)
        self.Scroll_mcI.setWidget(self.mcI_layout_widget)

        self.group_mcI_layout.addWidget(self.Scroll_mcI)
        input_layout.addWidget(self.group_mcI)

        # B-2. Coherent Input (Table)
        self.container_coherent = QWidget()
        coh_layout = QVBoxLayout(self.container_coherent)
        coh_layout.setContentsMargins(0, 0, 0, 0)
        coh_layout.addWidget(QLabel("Define Input Wavefunction (Phasor):"))
        
        self.table_coherent = QTableWidget()
        self.table_coherent.setColumnCount(3)
        self.table_coherent.setHorizontalHeaderLabels(["Lead Name", "|A|", "Phase (rad)"])
        self.table_coherent.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table_coherent.verticalHeader().setVisible(False)
        self.table_coherent.setFixedHeight(200) 
        
        coh_layout.addWidget(self.table_coherent)
        input_layout.addWidget(self.container_coherent)

        # --- C. Output Container (Scrollable GroupBox) ---
        main_layout.addWidget(QLabel("<b>2. Select Output Channels (Drain):</b>"))
        self.group_mcO = QGroupBox("Select Output Channels (Drain Leads)")
        self.group_mcO_layout = QVBoxLayout(self.group_mcO)

        self.Scroll_mcO = QScrollArea() # 修正：獨立的變數
        self.Scroll_mcO.setWidgetResizable(True)
        self.Scroll_mcO.setFixedHeight(150)

        self.mcO_layout_widget = QWidget()
        self.mcO_layout = QVBoxLayout(self.mcO_layout_widget)
        self.Scroll_mcO.setWidget(self.mcO_layout_widget) # 修正：設定給 Scroll_mcO

        self.group_mcO_layout.addWidget(self.Scroll_mcO)
        main_layout.addWidget(self.group_mcO)

        # --- D. Action Button ---
        main_layout.addStretch()
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        main_layout.addWidget(line)

        self.btn_calc = QPushButton("Calculate Transmission T(E)")
        self.btn_calc.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold; padding: 10px;")
        self.btn_calc.clicked.connect(self.on_btn_calc_clicked)
        main_layout.addWidget(self.btn_calc)
        
    def _refresh_content(self):
        """清除舊元件 -> 根據新 candidates 生成新元件"""
        self.sorted_leads = sorted(self.lead_candidates.items())
        
        # 1. 清除舊內容 (Helper function logic inline)
        for layout in [self.mcI_layout, self.mcO_layout]:
            while layout.count():
                item = layout.takeAt(0)
                if item.widget(): item.widget().deleteLater()
        
        self.chk_inputs.clear()
        self.chk_outputs.clear()
        self.spin_amps.clear()
        self.spin_phases.clear()
        self.table_coherent.setRowCount(0)

        # 2. 生成內容
        if not self.sorted_leads:
            self.mcI_layout.addWidget(QLabel("No leads defined."))
            self.mcO_layout.addWidget(QLabel("No leads defined."))
        else:
            # Loop for Inputs (Multi-channel)
            for idx, name in self.sorted_leads:
                chk = QCheckBox(f"{name} (Site {idx})")
                if idx == self.sorted_leads[0][0]: chk.setChecked(True)
                self.mcI_layout.addWidget(chk)
                self.chk_inputs[idx] = chk
            self.mcI_layout.addStretch()

            # Loop for Coherent Table
            self.table_coherent.setRowCount(len(self.sorted_leads))
            for row, (idx, name) in enumerate(self.sorted_leads):
                # Name
                item_name = QTableWidgetItem(f"{name} (Site {idx})")
                item_name.setFlags(item_name.flags() ^ Qt.ItemFlag.ItemIsEditable)
                self.table_coherent.setItem(row, 0, item_name)
                # Amp
                spin_amp = QDoubleSpinBox()
                spin_amp.setRange(0.0, 10.0); spin_amp.setSingleStep(0.1)
                spin_amp.setValue(1.0 if row == 0 else 0.0)
                self.table_coherent.setCellWidget(row, 1, spin_amp)
                self.spin_amps[idx] = spin_amp
                # Phase
                spin_phase = QDoubleSpinBox()
                spin_phase.setRange(-6.28, 6.28); spin_phase.setSingleStep(0.1)
                spin_phase.setValue(0.0)
                self.table_coherent.setCellWidget(row, 2, spin_phase)
                self.spin_phases[idx] = spin_phase

            # Loop for Outputs
            for idx, name in self.sorted_leads:
                chk = QCheckBox(f"{name} (Site {idx})")
                if idx != self.sorted_leads[0][0]: chk.setChecked(True)
                self.mcO_layout.addWidget(chk)
                self.chk_outputs[idx] = chk
            self.mcO_layout.addStretch()

        # 3. 刷新顯示狀態
        self._on_mode_changed(self.combo_mode.currentIndex())

    def _on_mode_changed(self, index):
        """切換介面顯示"""
        if index == 0: 
            # Multi-Channel
            self.group_mcI.show()      # 修正：顯示 Input Group
            self.container_coherent.hide()
        else:
            # Coherent
            self.group_mcI.hide()
            self.container_coherent.show()

    def on_btn_calc_clicked(self):
        """收集當前設定，發送訊號"""
        mode_idx = self.combo_mode.currentIndex()
        config = {}

        # 1. 收集 Output (Drain) Indices
        drain_indices = [idx for idx, chk in self.chk_outputs.items() if chk.isChecked()]
        config['drain_indices'] = drain_indices
        
        if not drain_indices:
            print("Warning: No output leads selected.")

        if mode_idx == 0:
            # --- Multi-Channel ---
            config['mode'] = 'multichannel'
            selected_leads = [idx for idx, chk in self.chk_inputs.items() if chk.isChecked()]
            config['source_indices'] = selected_leads
    
        else:
            # --- Coherent ---
            config['mode'] = 'coherent'
            wavefunction = {}
            
            # 修正：使用 spin_amps 的 keys 來遍歷，語意更精確
            for idx in self.spin_amps.keys():
                amp = self.spin_amps[idx].value()
                phase = self.spin_phases[idx].value()
                # 轉換為複數 c = r * e^(i*theta)
                c_val = abs(amp) * exp(1j * phase)
                wavefunction[idx] = c_val
                
            config['wavefunction'] = wavefunction

        # 發送訊號
        self.sig_arrived.emit(config)