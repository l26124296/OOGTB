import numpy as np
import time
from typing import Optional, Tuple, Union, List

# 引入 Model
from core.PhysicsModel import HamiltonianModel

# 引入 torch (可選)
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
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
    
    def calculate_structure_factor(self, q_range: float = 20.0, resolution: int = 400, use_gpu: bool = True) -> Tuple[np.ndarray, list]:
        """
        計算結構因子 S(q)。自動選擇 PyTorch (GPU) 或 NumPy (CPU)。
        
        Args:
            q_range: q 空間掃描範圍 ±q_range
            resolution: Grid size (res x res)
            use_gpu: 是否優先嘗試 GPU 加速
            
        Returns:
            (magnitude, extent)
        """
        if self.Projector is None: raise RuntimeError("Set filled states first.")
        
        extent = [-q_range, q_range, -q_range, q_range]
        
        # 1. 優先嘗試 Torch
        if HAS_TORCH and use_gpu:
            try:
                print("[Analyzer] Attempting PyTorch acceleration for Structure Factor...")
                return self._calculate_sf_torch(q_range, resolution), extent
            except Exception as e:
                print(f"[Analyzer] PyTorch failed ({e}). Falling back to NumPy.")
        
        # 2. Fallback to NumPy
        print("[Analyzer] Using NumPy for Structure Factor...")
        return self._calculate_sf_numpy(q_range, resolution), extent

    def _calculate_sf_torch(self, q_range, resolution) -> np.ndarray:
        """PyTorch 實作 (Tensor Contraction)"""
        # 裝置選擇
        device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        print(f"[Analyzer] Torch Device: {device}")

        # 準備數據
        P_tensor = torch.tensor(self.Projector, dtype=torch.complex128, device=device)
        W_tensor = torch.diagonal(P_tensor) # Column Sum
        del P_tensor
        
        pos = self.model.geo.positions
        x_vec = torch.tensor(pos[:, 0], dtype=torch.float64, device=device)
        y_vec = torch.tensor(pos[:, 1], dtype=torch.float64, device=device)
        
        qs = torch.linspace(-q_range, q_range, resolution, dtype=torch.float64, device=device)
        
        # Phase Calculation (Outer Product)
        # qs: (Res,), x_vec: (N,) -> (Res, N)
        phase_x = torch.exp(1j * torch.outer(qs, x_vec))
        phase_y = torch.exp(1j * torch.outer(qs, y_vec))
        
        # Contraction: sum_m (Ph_x[i,m] * Ph_y[j,m] * W[m])
        # 使用 einsum 處理
        sf_tensor = torch.einsum('im,jm,m->ij', phase_x, phase_y, W_tensor)
        
        res = sf_tensor.cpu().numpy()
        N_sites = self.model.geo.positions.shape[0]
        # 清理
        del x_vec, y_vec, W_tensor, phase_x, phase_y, sf_tensor
        if device.type != 'cpu': torch.cuda.empty_cache() if torch.cuda.is_available() else torch.mps.empty_cache()
            
        return np.abs(res) / N_sites

    def _calculate_sf_numpy(self, q_range, resolution) -> np.ndarray:
        """NumPy 實作 (Matrix Multiplication)"""
        start_t = time.time()
        
        # 1. 預計算權重 W (Column sum of P)
        # P is complex128
        W = np.diagonal(self.Projector) # Shape: (N,)
        
        # 2. 座標與動量向量
        pos = self.model.geo.positions
        x_vec = pos[:, 0]
        y_vec = pos[:, 1]
        qs = np.linspace(-q_range, q_range, resolution)
        
        # 3. 計算相位矩陣 (Broadcasting / Outer Product)
        # Phase_X shape: (Resolution, N_sites)
        # 使用 np.multiply.outer 是最快的
        phase_x = np.exp(1j * np.multiply.outer(qs, x_vec))
        phase_y = np.exp(1j * np.multiply.outer(qs, y_vec))
        
        # 4. 張量縮併 (Tensor Contraction) 轉為 矩陣乘法
        # 公式: S[i, j] = sum_m ( Ph_x[i, m] * Ph_y[j, m] * W[m] )
        # 令 Weighted_Ph_y[j, m] = Ph_y[j, m] * W[m]
        # 則 S[i, j] = sum_m ( Ph_x[i, m] * Weighted_Ph_y[j, m] )
        # 這就是矩陣乘法: S = Ph_x @ (Weighted_Ph_y)^T
        
        # 先將權重 W 廣播到 phase_y
        weighted_phase_y = phase_y * W[None, :] 
        
        # 矩陣相乘 (Res, N) @ (N, Res) -> (Res, Res)
        sf_matrix = phase_x @ weighted_phase_y.T
        
        print(f"[Analyzer] NumPy calculation time: {time.time() - start_t:.4f}s")
        
        N_sites = pos.shape[0]

        return np.abs(sf_matrix) / N_sites
