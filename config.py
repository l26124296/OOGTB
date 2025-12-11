# 定義每個生成器需要的參數規格
GENERATOR_CONFIG = {
    'Haldane Hexagon': [
        {'name': 'radius',  'label': 'Radius (int)',      'default': '5',   'type': int},
        {'name': 'a',       'label': 'Lattice Const (float)', 'default': '1.0', 'type': float},
        # 您可以擴充更多，例如 shape (dropdown)
    ],
    'Haldane Rectangular': [
        {'name': 'width',   'label': 'Width (int)',      'default': '5',   'type': int},
        {'name': 'height',  'label': 'Height (int)',     'default': '5',   'type': int},
        {'name': 'a',       'label': 'Lattice Const (float)', 'default': '1.0', 'type': float},
    ],
    'Haldane Parallelogram': [
        {'name': 'width',   'label': 'Width (int)',      'default': '5',   'type': int},
        {'name': 'height',  'label': 'Height (int)',     'default': '5',   'type': int},
        {'name': 'a',       'label': 'Lattice Const (float)', 'default': '1.0', 'type': float},
    ],
    'Penrose Multigrid': [
        {'name': 'EdgeLen', 'label': 'Edge Length of rhombus',      'default': '1', 'type': float},
        {'name': 'Cn_symmetry', 'label': 'Cn Symmetry (int)', 'default': '5',    'type': int},
        {'name': 'num_of_lines', 'label': 'Grid Lines (int)', 'default': '5',    'type': int},
        {'name': 'offsets', 'label': 'Offset (float)', 'default': '0.08', 'type': float},
        {'name': 'spacing', 'label': 'Grid Spacing',      'default': '1.0', 'type': float},
    ],
    'Rule Based Crystal': [
        # --- 1. 基礎參數 ---
        {'name': 'lattice_vectors', 'label': 'Lattice Vecs (a1, a2)', 'default': '1,0, 0,1', 'type': 'matrix_2x2'},
        {'name': 'repeat',          'label': 'Repeat (Nx, Ny)',       'default': '5, 5',     'type': 'int_array'},
        
        # --- 2. 數量控制 (Control Fields) ---
        {'type': 'section_header', 'label': '--- Structure Definitions ---'},
        {'name': 'n_basis', 'label': 'Num Basis Atoms', 'default': '1', 'type': int},
        {'name': 'n_rules', 'label': 'Num Rules',       'default': '1', 'type': int},
        
        # --- 3. 觸發按鈕 (用來刷新下方的輸入框) ---
        {'name': 'btn_update_fields', 'label': 'Update Input Fields', 'type': 'action'},
        
        # --- 4. 動態欄位 (會在程式中動態插入) ---
        # basis_0, basis_1...
        # rule_0, rule_1...
    ]

}