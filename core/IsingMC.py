import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal, QMutex

class IsingWorker(QThread):
    # 訊號: (自旋狀態, 磁化強度, 步數)
    data_updated = pyqtSignal(np.ndarray, float, int)

    def __init__(self, simplicial_complex, J_config: dict = None, parent=None):
        """
        Args:
            simplicial_complex: SimplicialComplex 資料物件
            J_config: 字典 {int: float}，定義不同 edge_type 的 J 值
                      例如: {0: -1.0, 1: 0.5} (0是鐵磁, 1是反鐵磁)
                      若為 None，預設所有類型 J = -1.0
        """
        super().__init__(parent)
        
        # --- 1. 幾何與拓撲解析 ---
        self.sites = simplicial_complex.positions.shape[0]
        
        # 構建 "帶類型" 的鄰接表
        # 結構: self.graph[i] =List of (neighbor_index, edge_type_id)
        # 這樣我們在迴圈裡就能知道 i 和 j 之間是用哪種 J 連接的
        self.graph = [[] for _ in range(self.sites)]
        
        # 確保 edge_types 存在，若無則預設全為 0
        if hasattr(simplicial_complex, 'edge_types') and simplicial_complex.edge_types is not None:
            edge_types = simplicial_complex.edge_types
        else:
            edge_types = np.zeros(len(simplicial_complex.edges), dtype=int)

        # 遍歷 edge 和 type
        for edge, type_id in zip(simplicial_complex.edges, edge_types):
            u, v = int(edge[0]), int(edge[1])
            t_id = int(type_id)
            # 存入 tuple: (鄰居索引, 邊類型)
            self.graph[u].append((v, t_id))
            self.graph[v].append((u, t_id))
            
        # --- 2. 物理參數初始化 ---
        self.mutex = QMutex() 
        self.T = 1.0          
        self.B_field = 0.0
        self.kb = 1.0
        
        # J 的查找表 (J Cache)
        # 我們將 J_config 轉換為一個 list 或 array，以便用 index 快速存取
        # 假設 type_id 是從 0 開始的整數
        max_type_id = int(np.max(edge_types)) if len(edge_types) > 0 else 0
        self.J_map = [ -1.0 ] * (max_type_id + 1) # 預設全為 -1
        
        # 載入使用者設定
        if J_config:
            self.update_J_map_internal(J_config)
            
        # 初始化自旋
        self.state = np.random.choice([-1, 1], size=self.sites)
        
        # 控制旗標
        self.is_running = True  
        self.is_paused = False  
        self.step_count = 0

    def update_J_map_internal(self, J_config):
        """內部函數：更新 J 查找表"""
        for type_id, value in J_config.items():
            if 0 <= type_id < len(self.J_map):
                self.J_map[type_id] = float(value)
            else:
                # 若 type_id 超出範圍，這裡可以選擇忽略或擴展 array
                # 這裡簡單處理：若超出範圍則忽略 (通常 type_id 是固定的)
                print(f"[Warning] Edge type {type_id} out of range, ignored.")

    # =========================================================
    # 外部控制介面
    # =========================================================
    def update_params(self, T, B, J_config=None):
        """
        支援同時更新 T, B 以及 J 的設定
        J_config: Optional dict {type: value}
        """
        self.mutex.lock()
        self.T = float(T)
        self.B_field = float(B)
        if J_config is not None:
            self.update_J_map_internal(J_config)
        self.mutex.unlock()

    # (set_paused, stop 方法同前，省略以節省篇幅)
    def set_paused(self, paused):
        self.mutex.lock()
        self.is_paused = paused
        self.mutex.unlock()

    def stop(self):
        self.mutex.lock()
        self.is_running = False
        self.mutex.unlock()
        self.wait()

    # =========================================================
    # 核心運算迴圈
    # =========================================================
    def run(self):
        import random
        
        while True:
            # --- A. 狀態檢查 ---
            self.mutex.lock()
            if not self.is_running:
                self.mutex.unlock()
                break
            if self.is_paused:
                self.mutex.unlock()
                self.msleep(100)
                continue
            
            # 取出參數
            curr_T = self.T
            curr_B = self.B_field
            # 複製一份 J_map，避免計算途中被外部修改導致 index error
            curr_J_map = list(self.J_map) 
            self.mutex.unlock()

            loops_per_frame = self.sites 
            
            # --- B. Monte Carlo Loop ---
            for _ in range(loops_per_frame):
                idx = random.randint(0, self.sites - 1)
                s_i = self.state[idx]
                
                # [關鍵修改] 計算有效場 (Effective Field)
                # 之前是: sum_neighbor * J
                # 現在是: sum( J_of_edge * s_neighbor )
                
                effective_interaction_field = 0.0
                
                # 遍歷該點的所有鄰居及其連接類型
                # neighbor_info 是 (neighbor_idx, type_id)
                for neighbor_idx, type_id in self.graph[idx]:
                    # 從 J_map 查出對應的 J 值
                    j_val = curr_J_map[type_id]
                    effective_interaction_field += j_val * self.state[neighbor_idx]
                
                # 計算能量差 Delta E
                # E_int = - sum (J_ij * S_i * S_j)
                # 當 S_i 翻轉為 -S_i 時:
                # dE_int = (- sum(J * (-S_i) * S_j)) - (- sum(J * S_i * S_j))
                #        = S_i * sum(J S_j) + S_i * sum(J S_j)
                #        = 2 * S_i * sum(J_ij * S_j)
                # (注意符號: 我們這裡 effective_interaction_field 就是 sum(J_ij * S_j))
                
                dE_int = 2 * s_i * effective_interaction_field
                dE_field = 2 * s_i * curr_B 
                
                dE = dE_int + dE_field
                
                # 接受判斷
                temp = curr_T if curr_T > 1e-5 else 1e-5
                accept_prob = 1.0 / (1.0 + np.exp(dE / (self.kb * temp)))
                
                if accept_prob > random.random():
                    self.state[idx] *= -1
            
            # --- C. 發送訊號 ---
            self.step_count += 1
            magnetization = np.mean(self.state)
            self.data_updated.emit(self.state.copy(), magnetization, self.step_count)
            self.msleep(10) # 若需要更快速度可調小此值