import numpy as np
import scipy.sparse as sp
from typing import Optional, Tuple, Union, List

# 引入 Model
from core.PhysicsModel import HamiltonianModel

class ElectronicState:
    """
    電子態分析器 (Electronic State Analyzer).
    
    負責將 Hamiltonian 的解 (E, Psi) 轉換為可供繪圖或統計的原始數據。
    不進行過度的平滑化處理，保持數據真實度。
    """
    def __init__(self, model: HamiltonianModel):
        self.model = model
        
        # --- System Properties (Cached) ---
        # 這些是 Hamiltonian 的屬性，與填充無關
        self.eigenvalues: Optional[np.ndarray] = None
        self.eigenvectors: Optional[np.ndarray] = None
        
        # --- State Properties (User Defined) ---
        # 這些取決於 Fermi Level
        self.energy_range: Optional[Tuple[float, float]] = None
        self.occupancy_mask: Optional[np.ndarray] = None 
        self.Projector: Optional[np.ndarray] = None      # P = sum |psi><psi|
        
    def diagonalize(self):
        """
        執行全對角化。
        """
        if self.model.H_sparse is None:
            raise RuntimeError("Hamiltonian is not constructed. Call model.construct() first.")
            
        print("[Analyzer] Diagonalizing Hamiltonian...")
        # 轉為 Dense 矩陣進行 eigh (Hermitian)
        H_dense = self.model.H_sparse.todense()
        self.eigenvalues, self.eigenvectors = np.linalg.eigh(H_dense)
        print(f"[Analyzer] Done. E range: [{self.eigenvalues[0]:.4f}, {self.eigenvalues[-1]:.4f}]")

    def set_filled_states(self, E_min: float = 0.0, E_max: float = 0.0):
        """
        設定填充態 (Zero Temperature)。
        """
        if self.eigenvalues is None:
            self.diagonalize()
            
        self.energy_range = (E_min, E_max)
        self.occupancy_mask = (self.eigenvalues <= E_max) & (self.eigenvalues >= E_min)
        
        n_occ = np.sum(self.occupancy_mask)
        n_total = len(self.eigenvalues)
        print(f"[Analyzer] Occupied energy window: {self.energy_range}. Occupied: {n_occ}/{n_total}")
        
        # 構建投影算符 P (Density Matrix)
        # P = V_occ @ V_occ.H
        # 雖然可以不存 P (因為很大 N*N)，但為了算 Chern Marker 還是需要
        V_occ = self.eigenvectors[:, self.occupancy_mask]
        self.Projector = V_occ @ V_occ.T.conj()

    # ==========================================
    #  物理量計算 (Raw Data Providers)
    # ==========================================

    def get_ldos_raw_data(self, site_indices: Union[int, List[int], np.ndarray, None] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        獲取 LDOS 原始數據，供前端繪製 Histogram。
        
        Args:
            site_indices: 
                - None: 回傳所有位置的總和 (即 DOS)
                - int: 指定單個原子
                - List/Array: 指定一群原子 (例如某個區域)
        
        Returns:
            energies: (N_states,) 所有能階
            weights: (N_states,) 該能階在指定位置的權重 sum(|psi|^2)
        """
        if self.eigenvalues is None: self.diagonalize()
        
        energies = self.eigenvalues
        all_weights = np.abs(self.eigenvectors)**2 # |Psi_n(r)|^2 shape: (N_sites, N_states)
        
        if site_indices is None:
            # 如果沒有指定位置，就是總 DOS，每個態的權重都是 1 (歸一化條件 sum_r |psi|^2 = 1)
            # 但為了與 LDOS 格式統一，我們回傳全 1
            weights = np.ones_like(energies)
        elif isinstance(site_indices, int):
            # 單點 LDOS
            weights = all_weights[site_indices, :]
        else:
            # 多點 LDOS (區域總和)
            weights = np.sum(all_weights[site_indices, :], axis=0)
            
        return energies, weights

    def get_am_spectrum(self, Cn: int) -> np.ndarray:
        """
        計算佔據態的角動量譜 (Spectrum of Angular Momentum for Occupied States).
        這相當於計算算符 P*L*P 的非零特徵值。
        
        數學上：
        令 V_occ 為 (N x N_occ) 的佔據態特徵向量矩陣。
        我們計算 N_occ x N_occ 的小矩陣 L_sub = V_occ^dagger @ L @ V_occ。
        對 L_sub 對角化得到的特徵值，即為此系統佔據態的角動量分佈。
        
        Args:
            Cn: 旋轉對稱階數 (e.g. 5 for Penrose).
        
        Returns:
            l_values: (N_occ,) array of angular momentum eigenvalues.
        """
        if self.eigenvalues is None: self.diagonalize()
        if self.occupancy_mask is None: raise RuntimeError("Set filled states first.")
        
        print(f"[Analyzer] Calculating AM Spectrum (C{Cn})...")
        
        # 1. 獲取稀疏角動量算符 L (N x N)
        L_sparse = self.model.get_L_operator(Cn, verbose=False)
        
        # 2. 取出佔據態波函數 V_occ (N x N_occ)
        V_occ = self.eigenvectors[:, self.occupancy_mask]
        
        if V_occ.shape[1] == 0:
            print("[Analyzer] No occupied states.")
            return np.array([])

        # 3. 投影算符 L 到佔據子空間 (Project L onto occupied subspace)
        # L_sub = V_occ^H * L * V_occ
        # 計算順序優化：先算 L * V_occ (sparse @ dense -> dense)
        LV = L_sparse @ V_occ
        L_sub = V_occ.T.conj() @ LV
        
        # 4. 對角化這個小矩陣 (N_occ x N_occ)
        # L 應該是 Hermitian，所以用 eigh
        l_vals = np.linalg.eigvalsh(L_sub)
        
        # 回傳實部 (理論上虛部為 0)
        return np.real(l_vals)

    def calculate_charge_density(self) -> np.ndarray:
        """電荷密度: rho_i = P_ii"""
        if self.Projector is None: raise RuntimeError("Set filled states first.")
        return np.real(np.diag(self.Projector))

    def calculate_chern_marker(self) -> np.ndarray:
        """局域陳數 C(r)"""
        if self.Projector is None: raise RuntimeError("Set filled states first.")
        
        # ... (這裡沿用之前的 Chern Marker 邏輯，需要 Model 提供 X, Y) ...
        X_sp, Y_sp = self.model.get_position_operators()
        X = X_sp.todense() # 或使用 sparse 乘法優化
        Y = Y_sp.todense()
        P = self.Projector
        Operator = P @ (X @ P @ Y - Y @ P @ X) @ P
        return 4 * np.pi * np.imag(np.diag(Operator))