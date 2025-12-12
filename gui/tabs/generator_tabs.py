# gui/tabs/generator_tab.py

import numpy as np
import traceback
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QComboBox, 
                             QPushButton, QLabel, QSpinBox, QDoubleSpinBox, 
                             QLineEdit, QSplitter, QScrollArea, QGroupBox, 
                             QMessageBox, QFileDialog)
from PyQt6.QtCore import Qt 
from PyQt6.QtGui import QFont
import pyqtgraph as pg

from gui.utils import *
from typing import Dict
# 從 core 引入後端邏輯
from core.LatticeGenerators import *
from config import GENERATOR_CONFIG

class LatticeGeneratorTab(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Lattice Physics Studio (PyQt6 + PyQtGraph)")
        self.resize(1400, 900)

        # --- Data Generators ---
        self.haldane_gen = HaldaneHoneycombGenerator()
        self.penrose_gen = MultigridGenerator()
        self.rule_gen = RuleBasedTranslationGenerator()
        self.current_complex = None

        # --- State ---
        self.widgets_ref: Dict[str, QWidget] = {} # 儲存輸入框引用 {'param_name': widget}
        self.edge_items: Dict[int, pg.PlotDataItem] = {}
        self.current_n_basis = 1
        self.current_n_rules = 1

        # --- UI Setup ---
        self.init_ui()
        
        # 初始化預設模式
        self._rebuild_dynamic_form('Haldane Hexagon')

        # 邊配色
        self.edge_color_palette = [
            (150, 150, 150), # Type 0: Gray 
            (255, 80, 80),   # Type 1: Red   
            (80, 80, 255),   # Type 2: Blue  
            (80, 255, 80),   # Type 3: Green
            (255, 255, 80),  # Type 4: Yellow
            (255, 80, 255)   # Type 5: Magenta
        ]
    def init_ui(self):
        """主介面佈局：左側控制面板 | 右側繪圖區"""
        """此 Tab 的佈局：左側控制面板 | 右側繪圖區"""
        layout = QHBoxLayout(self) # 使用 layout 而不是 setCentralWidget

        # 1. 左側控制區 (Scrollable)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFixedWidth(400)
        
        self.controls_container = QWidget()
        self.controls_layout = QVBoxLayout(self.controls_container)
        self.controls_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.scroll.setWidget(self.controls_container)

        # --- 靜態控制項 ---
        title = QLabel("<h2>Lattice Generator</h2>")
        self.controls_layout.addWidget(title)
        title.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        self.controls_layout.addWidget(QLabel("<b>Generator Mode:</b>"))
        self.combo_mode = QComboBox()
        self.combo_mode.addItems(list(GENERATOR_CONFIG.keys()))
        self.combo_mode.currentTextChanged.connect(self._rebuild_dynamic_form)
        self.controls_layout.addWidget(self.combo_mode)
        
        self.controls_layout.addWidget(HOLine())

        # --- 動態表單區 ---
        self.dynamic_form_layout = QVBoxLayout()
        self.controls_layout.addLayout(self.dynamic_form_layout)

        self.controls_layout.addStretch()

        # --- 底部按鈕 ---
        self.btn_generate = QPushButton("Generate Lattice")
        self.btn_generate.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 10px;")
        self.btn_generate.clicked.connect(self.run_generation)
        self.controls_layout.addWidget(self.btn_generate)

        self.btn_save = QPushButton("Save to .npz")
        self.btn_save.clicked.connect(self.save_data)
        self.controls_layout.addWidget(self.btn_save)

        # 2. 右側繪圖區 (PyQtGraph)
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('#1e1e1e')
        self.plot_widget.setAspectLocked(True)
        self.plot_widget.showGrid(x=True, y=True)
        
        self.scatter_item = pg.ScatterPlotItem()
        self.plot_widget.addItem(self.scatter_item)

        # 3. Splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.scroll)
        splitter.addWidget(self.plot_widget)
        splitter.setStretchFactor(1, 4)
        
        layout.addWidget(splitter)

    # =========================================================
    #  Dynamic UI Logic
    # =========================================================

    def _clear_layout(self, layout):
        """遞迴清除 Layout 中的所有元件"""
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
            elif item.layout():
                self._clear_layout(item.layout())

    def _rebuild_dynamic_form(self, mode_name):
        """根據 Config 重建輸入表單"""
        # 1. 清除舊表單
        self._clear_layout(self.dynamic_form_layout)
        self.widgets_ref.clear()

        # 2. 讀取配置
        if isinstance(mode_name, tuple): mode_name = mode_name[0]
        config = GENERATOR_CONFIG.get(mode_name, [])

        # 3. 生成元件
        for param in config:
            p_type = param.get('type')
            p_name = param.get('name')
            p_label = param.get('label')
            p_default = param.get('default')

            # (A) 標題
            if p_type == 'section_header':
                lbl = QLabel(f"<b>{p_label}</b>")
                lbl.setStyleSheet("margin-top: 10px; color: #aaa;")
                self.dynamic_form_layout.addWidget(lbl)
                continue

            # (B) 動作按鈕 (Update Fields)
            if p_type == 'action':
                btn = QPushButton(p_label)
                btn.clicked.connect(lambda checked, n=p_name: self.handle_action(n))
                self.dynamic_form_layout.addWidget(btn)
                self.widgets_ref[p_name] = btn
                continue

            # (C) 特殊排版：矩陣與陣列
            if p_name == 'lattice_vectors':
                self._add_matrix_2x2_ui(p_label, p_name, p_default)
                continue
            if p_name == 'repeat':
                self._add_repeat_ui(p_label, p_name, p_default)
                continue

            # (D) 標準欄位 (Label + Input)
            # 使用 GroupBox 或 Frame 稍微包一下看起來比較整齊
            row_layout = QHBoxLayout()
            lbl = QLabel(p_label)
            lbl.setFixedWidth(120)
            row_layout.addWidget(lbl)

            widget = None
            if p_type == int:
                widget = QSpinBox()
                widget.setRange(-9999, 9999)
                widget.setValue(int(p_default))
                # 特殊處理：如果是 n_basis/n_rules，綁定當前狀態
                if p_name == 'n_basis': widget.setValue(self.current_n_basis)
                if p_name == 'n_rules': widget.setValue(self.current_n_rules)

            elif p_type == float:
                widget = QDoubleSpinBox()
                widget.setRange(-9999.0, 9999.0)
                widget.setSingleStep(0.1)
                widget.setDecimals(4)
                widget.setValue(float(p_default))

            elif p_type in [str, 'float_array']:
                widget = QLineEdit(str(p_default))

            if widget:
                row_layout.addWidget(widget)
                self.dynamic_form_layout.addLayout(row_layout)
                self.widgets_ref[p_name] = widget

        # 4. 特殊處理 Rule Based 的動態欄位
        if mode_name == 'Rule Based Crystal':
            self._add_rule_based_dynamic_section()

    # --- 排版輔助函數 ---

    def _add_matrix_2x2_ui(self, label, prefix, default_str):
        """生成 2x2 矩陣輸入 (a1, a2) - 水平對齊"""
        group = QGroupBox(label)
        layout = QVBoxLayout()
        
        # Row 1: a1
        r1 = QHBoxLayout()
        r1.addWidget(QLabel("a1 (x,y):"))
        w_a1x = QDoubleSpinBox(); w_a1x.setRange(-99,99); w_a1x.setValue(1.0)
        w_a1y = QDoubleSpinBox(); w_a1y.setRange(-99,99); w_a1y.setValue(0.0)
        r1.addWidget(w_a1x); r1.addWidget(w_a1y)
        layout.addLayout(r1)
        
        # Row 2: a2
        r2 = QHBoxLayout()
        r2.addWidget(QLabel("a2 (x,y):"))
        w_a2x = QDoubleSpinBox(); w_a2x.setRange(-99,99); w_a2x.setValue(0.0)
        w_a2y = QDoubleSpinBox(); w_a2y.setRange(-99,99); w_a2y.setValue(1.0)
        r2.addWidget(w_a2x); r2.addWidget(w_a2y)
        layout.addLayout(r2)
        
        group.setLayout(layout)
        self.dynamic_form_layout.addWidget(group)
        
        # 註冊引用
        self.widgets_ref['lat_a1_x'] = w_a1x; self.widgets_ref['lat_a1_y'] = w_a1y
        self.widgets_ref['lat_a2_x'] = w_a2x; self.widgets_ref['lat_a2_y'] = w_a2y

    def _add_repeat_ui(self, label, prefix, default_str):
        row = QHBoxLayout()
        row.addWidget(QLabel(label))
        wm = QSpinBox(); wm.setValue(5); wm.setRange(1, 100)
        wn = QSpinBox(); wn.setValue(5); wn.setRange(1, 100)
        row.addWidget(wm); row.addWidget(wn)
        self.dynamic_form_layout.addLayout(row)
        self.widgets_ref['repeat_m'] = wm; self.widgets_ref['repeat_n'] = wn

    def _add_rule_based_dynamic_section(self):
        """生成 Basis 和 Rules 的動態輸入網格"""
        
        # --- Basis Coords ---
        basis_group = QGroupBox(f"Basis Coordinates ({self.current_n_basis})")
        b_layout = QVBoxLayout()
        
        for i in range(self.current_n_basis):
            row = QHBoxLayout()
            row.addWidget(QLabel(f"{i}:"))
            bx = QDoubleSpinBox(); bx.setRange(-10,10); bx.setSingleStep(0.1); bx.setPrefix("x: ") ; bx.setDecimals(4)
            by = QDoubleSpinBox(); by.setRange(-10,10); by.setSingleStep(0.1); by.setPrefix("y: ") ; by.setDecimals(4)
            row.addWidget(bx)
            row.addWidget(by)
            b_layout.addLayout(row)
            
            self.widgets_ref[f'basis_{i}_x'] = bx
            self.widgets_ref[f'basis_{i}_y'] = by
            
        basis_group.setLayout(b_layout)
        self.dynamic_form_layout.addWidget(basis_group)

        # --- Rules ---
        rules_group = QGroupBox(f"Connection Rules ({self.current_n_rules})")
        r_layout = QVBoxLayout()
        
        # Header
        header = QHBoxLayout()
        header.addWidget(QLabel("Src"), 1)
        header.addWidget(QLabel("Dst"), 1)
        header.addWidget(QLabel("dx"), 1)
        header.addWidget(QLabel("dy"), 1)
        header.addWidget(QLabel("Desc"), 2)
        r_layout.addLayout(header)

        for i in range(self.current_n_rules):
            row = QHBoxLayout()
            
            w_b1 = QSpinBox(); w_b1.setRange(0, self.current_n_basis-1)
            w_b2 = QSpinBox(); w_b2.setRange(0, self.current_n_basis-1)
            w_dx = QSpinBox(); w_dx.setRange(-10, 10)
            w_dy = QSpinBox(); w_dy.setRange(-10, 10)
            w_desc = QLineEdit("NN")
            
            row.addWidget(w_b1, 1)
            row.addWidget(w_b2, 1)
            row.addWidget(w_dx, 1)
            row.addWidget(w_dy, 1)
            row.addWidget(w_desc, 2)
            
            r_layout.addLayout(row)
            
            self.widgets_ref[f'rule_{i}_b1'] = w_b1
            self.widgets_ref[f'rule_{i}_b2'] = w_b2
            self.widgets_ref[f'rule_{i}_dx'] = w_dx
            self.widgets_ref[f'rule_{i}_dy'] = w_dy
            self.widgets_ref[f'rule_{i}_desc'] = w_desc

        rules_group.setLayout(r_layout)
        self.dynamic_form_layout.addWidget(rules_group)

    # =========================================================
    #  Logic & Events
    # =========================================================

    def handle_action(self, action_name):
        """處理 Update Fields 按鈕"""
        if action_name == 'btn_update_fields':
            try:
                self.current_n_basis = self.widgets_ref['n_basis'].value()
                self.current_n_rules = self.widgets_ref['n_rules'].value()
                self._rebuild_dynamic_form('Rule Based Crystal')
            except Exception as e:
                print(f"Error updating fields: {e}")

    def _parse_inputs(self, mode_name):
        """讀取 Widget 數值並轉換格式"""
        if isinstance(mode_name, tuple): mode_name = mode_name[0]
        schema = GENERATOR_CONFIG.get(mode_name, [])
        kwargs = {}

        try:
            # 1. 讀取標準 Config 中的參數
            for param in schema:
                key = param.get('name')
                p_type = param.get('type')
                
                # 跳過非輸入型元件
                if p_type in ['section_header', 'action', 'matrix_2x2', 'int_array']:
                    continue
                
                # 從 widgets_ref 獲取
                widget = self.widgets_ref.get(key)
                if not widget: continue

                if isinstance(widget, QSpinBox):
                    kwargs[key] = widget.value()
                elif isinstance(widget, QDoubleSpinBox):
                    kwargs[key] = widget.value()
                elif isinstance(widget, QLineEdit):
                    val_str = widget.text()
                    if p_type == 'float_array':
                        kwargs[key] = np.array([float(x) for x in val_str.split(',')])
                    else:
                        kwargs[key] = val_str

            # 2. 特殊處理 Rule Based 的複雜物件
            if mode_name == 'Rule Based Crystal':
                # Repeat
                kwargs['repeat'] = (
                    self.widgets_ref['repeat_m'].value(),
                    self.widgets_ref['repeat_n'].value()
                )
                
                # Lattice Vectors
                kwargs['settings'] = self._parse_unit_cell_settings()

            return kwargs
        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, "Input Error", str(e))
            return None

    def _parse_unit_cell_settings(self):
        # 1. Basis
        basis_list = []
        for i in range(self.current_n_basis):
            bx = self.widgets_ref[f'basis_{i}_x'].value()
            by = self.widgets_ref[f'basis_{i}_y'].value()
            basis_list.append([bx, by])
            
        # 2. Lattice Vectors
        a1 = [self.widgets_ref['lat_a1_x'].value(), self.widgets_ref['lat_a1_y'].value()]
        a2 = [self.widgets_ref['lat_a2_x'].value(), self.widgets_ref['lat_a2_y'].value()]
        
        # 3. Rules
        rules_list = []
        name_to_id_map = {} # 用來記錄 { '名稱': type_id }
        next_id = 0

        for i in range(self.current_n_rules):
            # 讀取使用者輸入
            b1 = self.widgets_ref[f'rule_{i}_b1'].value()
            b2 = self.widgets_ref[f'rule_{i}_b2'].value()
            dx = self.widgets_ref[f'rule_{i}_dx'].value()
            dy = self.widgets_ref[f'rule_{i}_dy'].value()
            raw_desc = self.widgets_ref[f'rule_{i}_desc'].text().strip()
            
            # 若使用者沒填名稱，給預設值
            if not raw_desc: raw_desc = f"Rule_{i}"
            
            # --- 自動整合 ID ---
            if raw_desc in name_to_id_map:
                # 如果名稱已存在，使用既有的 ID
                assigned_id = name_to_id_map[raw_desc]
            else:
                # 如果是新名稱，分配新 ID
                assigned_id = next_id
                name_to_id_map[raw_desc] = assigned_id
                next_id += 1
            
            rules_list.append(ConnectionRule(
                source_basis=b1,
                target_basis=b2,
                delta=(dx, dy),
                type_id=assigned_id,   # 使用歸一化後的 ID
                type_name=raw_desc     # 保持原始名稱
            ))
            
        return UnitCellSettings(
            basis_coords=np.array(basis_list),
            lattice_vectors=np.array([a1, a2]),
            rules=rules_list
        )

    def run_generation(self):
        mode = self.combo_mode.currentText()
        kwargs = self._parse_inputs(mode)
        if kwargs is None: return

        try:
            if mode == 'Haldane Hexagon':
                self.current_complex = self.haldane_gen.generate(shape='hexagon', **kwargs)
            elif mode == 'Haldane Rectangular':
                self.current_complex = self.haldane_gen.generate(shape='rectangular', **kwargs)
            elif mode == 'Haldane Parallelogram':
                self.current_complex = self.haldane_gen.generate(shape='parallelogram', **kwargs)

            elif mode == 'Penrose Multigrid':
                # 補充一些固定參數
                kwargs['truncation_mode'] = 'polygon' 
                if 'Cn_symmetry' not in kwargs: kwargs['Cn_symmetry'] = 5
                self.current_complex = self.penrose_gen.generate(**kwargs)
                
            elif mode == 'Rule Based Crystal':
                self.current_complex = self.rule_gen.generate(**kwargs) # kwargs 包含 settings 和 repeat

            self.update_plot()
            print(f"Generated {len(self.current_complex.positions)} sites.")

        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, "Generation Error", f"Logic Error:\n{str(e)}")

    def update_plot(self):
        if self.current_complex is None: return
        
        # 1. 調用調色盤 (Palette)
        palette = self.edge_color_palette

        # -----------------------------
        # 2. 清除舊的邊 (Clean up)
        # -----------------------------
        for item in self.edge_items.values():
            self.plot_widget.removeItem(item)
        self.edge_items.clear()

        # -----------------------------
        # 3. 繪製點 (Vertices)
        # -----------------------------
        pos = self.current_complex.positions
        self.scatter_item.setData(
            x=pos[:, 0], y=pos[:, 1], 
            size=6, # 點稍微小一點，讓線看得更清楚
            brush=pg.mkBrush(200, 200, 200, 255), 
            pen=None
        )

        # -----------------------------
        # 4. 根據類型繪製邊 (Edges by Type)
        # -----------------------------
        all_edges = self.current_complex.edges
        all_types = self.current_complex.edge_types
        
        if len(all_edges) == 0: return

        # 找出目前有哪些類型 (例如 [0, 1, 2])
        unique_types = np.unique(all_types)

        for t_id in unique_types:
            # A. 篩選出該類型的邊
            mask = (all_types == t_id)
            sub_edges = all_edges[mask]
            
            if len(sub_edges) == 0: continue

            # B. 準備 connect='pairs' 所需的座標陣列
            # 格式: [x_start_1, x_end_1, x_start_2, x_end_2, ...]
            start_pts = pos[sub_edges[:, 0]]
            end_pts = pos[sub_edges[:, 1]]
            
            x_pairs = np.column_stack((start_pts[:,0], end_pts[:,0])).flatten()
            y_pairs = np.column_stack((start_pts[:,1], end_pts[:,1])).flatten()

            # C. 決定樣式
            # 如果類型 ID 超出調色盤範圍，就循環使用
            color = palette[t_id % len(palette)]
            
            # 類型 0 (通常是 NN) 畫粗一點，其他 (NNN) 畫細一點
            width = 2 if t_id == 0 else 1.5
            style = Qt.PenStyle.SolidLine
            
            # 如果是 NNN (Type > 0)，也可以考慮用虛線區分 (Optional)
            # if t_id > 0: style = Qt.PenStyle.DashLine 

            pen = pg.mkPen(color=color, width=width, style=style)

            # D. 建立繪圖物件並加入場景
            edge_item = pg.PlotDataItem(
                x=x_pairs, y=y_pairs, 
                connect='pairs', # 關鍵優化參數
                pen=pen
            )
            self.plot_widget.addItem(edge_item)
            
            # 存入字典管理
            self.edge_items[t_id] = edge_item

        # (Optional) 可以在 Console 印出圖例資訊
        if hasattr(self.current_complex, 'edge_type_map'):
            print("Plot Legend:", self.current_complex.edge_type_map)

    def save_data(self):
        # 防呆：如果沒有數據，先警告
        if self.current_complex is None:
            QMessageBox.warning(self, "No Data", "Please generate a lattice first.")
            return

        # 開啟檔案選擇視窗
        # getSaveFileName(parent, caption, directory, filter)
        # 回傳值是一個 tuple: (完整路徑字串, 選中的濾鏡類型)
        file_path, _ = QFileDialog.getSaveFileName(
            self,                  # 父視窗
            "Save Lattice Data",   # 視窗標題
            "",                    # 預設路徑 (空字串代表當前目錄)
            "NumPy Compressed (*.npz);;All Files (*)" # 檔案類型過濾器
        )

        # 如果使用者按了「取消」，file_path 會是空字串
        if file_path:
            try:
                # 呼叫 geometry 的 save 方法
                self.current_complex.save(file_path)
                
                # 顯示成功訊息 (包含儲存路徑)
                QMessageBox.information(self, "Success", f"Data saved to:\n{file_path}")
            except Exception as e:
                traceback.print_exc()
                QMessageBox.critical(self, "Save Error", f"Failed to save file:\n{str(e)}")
