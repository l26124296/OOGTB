import numpy as np
import scipy.linalg as sla
from PyQt6.QtCore import QThread, pyqtSignal
import os
import tempfile
import uuid

# ==============================================================================
# 0. SelfEnergyUtils (統一數學邏輯庫)
#    負責底層物理計算，確保 Calculator 和 Analyzer 使用完全相同的公式
# ==============================================================================
class SelfEnergyUtils:
    @staticmethod
    def compute_1d_surface_g(energy, t_chain):
        """
        計算半無限長 1D Chain 的表面 Green's Function (Retarded)。
        """
        E = energy + 1e-10j # 加微小虛部避免奇異點
        t = float(t_chain)
        
        delta = E**2 - 4 * t**2
        sqrt_delta = np.sqrt(delta)
        
        # 選擇正確的物理分支 (Retarded solution)
        sgn = np.sign(E.real)
        if sgn == 0: 
            sgn = 1.0 
            
        numerator = E - sgn * sqrt_delta
        denominator = 2 * t**2
        
        g = numerator / denominator
        return g

    @staticmethod
    def get_sigma_value(energy, mode, wbl_gamma, t_chain, t_coupling):
        """
        取得單個 Site 的 Self-Energy 值 (Sigma)。
        
        Args:
            energy (complex): 能量點
            mode (str): 'WBL' or '1D-Chain'
            wbl_gamma (float): WBL 模式下的 Gamma 值
            t_chain (float): 1D Chain 內部的 hopping
            t_coupling (float): Lead 與 System 的耦合強度
        """
        if mode == 'WBL':
            # Wide Band Limit: Sigma = -i * Gamma/2
            return -0.5j * wbl_gamma
        elif mode == '1D-Chain':
            g_surf = SelfEnergyUtils.compute_1d_surface_g(energy, t_chain)
            return (np.abs(t_coupling)**2) * g_surf
        else:
            return 0j

    @staticmethod
    def get_gamma_value(energy, mode, wbl_gamma, t_chain, t_coupling):
        """
        取得單個 Site 的 Gamma 值 (Gamma = i(Sigma - Sigma^dagger) = -2 Im(Sigma))。
        """
        if mode == 'WBL':
            return wbl_gamma
        elif mode == '1D-Chain':
            sigma = SelfEnergyUtils.get_sigma_value(energy, mode, wbl_gamma, t_chain, t_coupling)
            val = -2 * sigma.imag
            # 數值穩定性修正：Gamma 必須非負
            return max(0.0, val)
        return 0.0

# ==============================================================================
# 1. GreenFunctionBank (資料儲存結構)
# ==============================================================================
class GreenFunctionBank:
    """
    負責儲存所有能量點的 Green's Function 矩陣。
    包含自動切換記憶體/硬碟儲存 (Memmap) 的邏輯。
    """
    def __init__(self, energy_points, system_size, temp_dir='./'):
        self.energies = energy_points
        self.n_steps = len(energy_points)
        self.N = system_size
        self.use_disk = False
        self._file_path = None
        
        # 估算記憶體 (Complex128 = 16 bytes)
        total_bytes = self.n_steps * self.N * self.N * 16
        gb_usage = total_bytes / (1024**3)
        
        if gb_usage > 1.0: # 超過 1GB 則使用硬碟映射
            self.use_disk = True
            if temp_dir is None:
                temp_dir = tempfile.gettempdir()
            unique_name = f"gf_storage_{uuid.uuid4().hex}.npy"
            self._file_path = os.path.join(temp_dir, unique_name)
            print(f"[GFBank] Data too large ({gb_usage:.2f} GB). Using Disk Mapping: {self._file_path}")
            self.G_R_array = np.memmap(
                self._file_path, dtype='complex128', mode='w+', 
                shape=(self.n_steps, self.N, self.N)
            )
        else:
            self.G_R_array = np.zeros((self.n_steps, self.N, self.N), dtype='complex128')

    def set_green_function(self, index, G_matrix):
        self.G_R_array[index] = G_matrix
        if self.use_disk:
            self.G_R_array.flush()

    def get_green_function(self, index):
        return self.G_R_array[index]

    def cleanup(self):
        if self.use_disk and self._file_path and os.path.exists(self._file_path):
            try:
                # 確保釋放 memmap 資源
                if hasattr(self.G_R_array, '_mmap'):
                    self.G_R_array._mmap.close()
                del self.G_R_array 
                os.remove(self._file_path)
            except Exception as e:
                print(f"[GFBank] Cleanup failed: {e}")

# ==============================================================================
# 2. GFCalculator (計算與生成 GFBank)
# ==============================================================================
class GFCalculator(QThread):
    progress_sig = pyqtSignal(int)
    finished_sig = pyqtSignal(object) # 回傳 GreenFunctionBank
    error_sig = pyqtSignal(str)

    def __init__(self, hamiltonian_model, lead_config, energy_range):
        super().__init__()
        self.model = hamiltonian_model
        self.lead_config = lead_config
        self.energies = energy_range
        self._is_running = True
        self.bank = None

    def _get_sigma_matrix_total(self, energy, N):
        """計算所有 Leads 的總 Self-Energy 矩陣"""
        Sigma_total = np.zeros((N, N), dtype=complex)
        
        # 提取參數
        mode = self.lead_config.get('mode', 'WBL')
        wbl_gamma = self.lead_config.get('wbl_gamma', 0.1)
        t_chain = self.lead_config.get('chain_hopping', 1.0) # 注意 key 對應
        t_coupling = self.lead_config.get('coupling', 1.0)
        leads = self.lead_config.get('leads', {})
        
        for idx in leads.keys():
            # 調用統一工具
            val = SelfEnergyUtils.get_sigma_value(energy, mode, wbl_gamma, t_chain, t_coupling)
            Sigma_total[idx, idx] += val
            
        return Sigma_total

    def run(self):
        try:
            H = self.model.H_sparse
            H_dense = H.toarray() if hasattr(H, 'toarray') else H
            N = H_dense.shape[0]
            
            self.bank = GreenFunctionBank(self.energies, N)
            total_steps = len(self.energies)
            
            for i, E in enumerate(self.energies):
                if not self._is_running: break
                
                # 1. 計算 Sigma
                Sigma = self._get_sigma_matrix_total(E, N)
                
                # 2. 求解 G^R = inv(E - H - Sigma)
                A = (E * np.eye(N, dtype=complex)) - H_dense - Sigma
                
                try:
                    G_R = sla.solve(A, np.eye(N, dtype=complex))
                except np.linalg.LinAlgError:
                    G_R = np.zeros((N, N), dtype=complex)
                
                self.bank.set_green_function(i, G_R)
                
                if i % 10 == 0:
                    self.progress_sig.emit(int((i+1)/total_steps * 100))
            
            if self._is_running:
                self.finished_sig.emit(self.bank)
            else:
                self.bank.cleanup()
                self.error_sig.emit("Aborted")

        except Exception as e:
            if self.bank: self.bank.cleanup()
            import traceback
            traceback.print_exc()
            self.error_sig.emit(str(e))

    def stop(self):
        self._is_running = False

# ==============================================================================
# 3. Analyzer (提供靜態方法供主程式調用)
#    這就是您主視窗要調用的後端入口
# ==============================================================================
class Analyzer:
    """
    靜態工具類，負責處理來自 UI 的傳輸分析請求。
    """

    @staticmethod
    def calculate_transmission(gf_bank, lead_config, source_indices, drain_indices=None):
        """
        Fisher-Lee 公式計算 Multi-channel Transmission T(E)。
        
        Args:
            gf_bank (GreenFunctionBank): 已計算好的格林函數庫
            lead_config (dict): Lead 參數配置
            source_indices (list[int]): 輸入端的 Site indices
            drain_indices (list[int]): 輸出端的 Site indices (若為 None 則計算 Total Transmission，但通常需指定)
        """
        if not source_indices:
            return np.zeros(len(gf_bank.energies))
        
        # 若未指定輸出，預設為所有 Leads 扣除 Source (較少用，通常由 UI 指定)
        if drain_indices is None:
            all_leads = list(lead_config['leads'].keys())
            drain_indices = [idx for idx in all_leads if idx not in source_indices]

        # 提取參數
        mode = lead_config.get('mode', 'WBL')
        wbl_gamma = lead_config.get('wbl_gamma', 0.1)
        t_chain = lead_config.get('chain_hopping', 1.0)
        t_coupling = lead_config.get('coupling', 1.0)

        T_values = []
        N = gf_bank.N
        energies = gf_bank.energies

        for i, E in enumerate(energies):
            G_R = gf_bank.get_green_function(i)
            # G_A = G_R.conj().T  (在 Trace 計算中隱式處理以節省記憶體)

            # 1. 構造 Gamma_in (Source)
            Gamma_in = np.zeros((N, N), dtype=complex)
            val_in = SelfEnergyUtils.get_gamma_value(E, mode, wbl_gamma, t_chain, t_coupling)
            for idx in source_indices:
                Gamma_in[idx, idx] = val_in

            # 2. 構造 Gamma_out (Drain)
            Gamma_out = np.zeros((N, N), dtype=complex)
            val_out = SelfEnergyUtils.get_gamma_value(E, mode, wbl_gamma, t_chain, t_coupling)
            for idx in drain_indices:
                Gamma_out[idx, idx] = val_out

            # 3. Fisher-Lee Formula: T = Tr[ Gamma_out @ G_R @ Gamma_in @ G_A ]
            # 優化寫法: T = sum( abs( (Gamma_out^1/2 @ G_R @ Gamma_in^1/2) )^2 ) 
            # 但標準寫法較易懂: M = Gamma_out @ G_R @ Gamma_in @ G_A
            
            # 使用 Trace 優化技巧: Tr(A @ B) = sum(A * B.T)
            # A = Gamma_out @ G_R
            # B = Gamma_in @ G_A -> B.T = G_A.T @ Gamma_in.T = G_R.conj() @ Gamma_in
            
            M1 = Gamma_out @ G_R
            M2_T = G_R.conj() @ Gamma_in 
            
            # Element-wise multiply then sum = Trace(M1 @ M2_T.T) = Trace(M1 @ B)
            # 這裡數學有點繞，簡單來說就是 Tr(Gamma_out G_R Gamma_in G_A)
            
            # 直接法 (最穩健，雖然多一次矩陣乘法)
            G_A = G_R.conj().T
            Mat = Gamma_out @ G_R @ Gamma_in @ G_A
            T_val = np.trace(Mat).real
            
            T_values.append(max(0.0, T_val)) # 物理上 T >= 0

        return np.array(T_values)

    @staticmethod
    def calculate_coherence_transmission(gf_bank, lead_config, wavefunction_dict, drain_indices):
        """
        計算 Rank-1 Coherent State 的傳輸機率。
        T = Sum_drain |<drain| G^R |source_psi>|^2 * Gamma_drain
        
        Args:
            wavefunction_dict (dict): {site_index: complex_amplitude}
            drain_indices (list[int]): 輸出端的 indices
        """
        # 提取參數
        mode = lead_config.get('mode', 'WBL')
        wbl_gamma = lead_config.get('wbl_gamma', 0.1)
        t_chain = lead_config.get('chain_hopping', 1.0)
        t_coupling = lead_config.get('coupling', 1.0)

        T_values = []
        N = gf_bank.N
        energies = gf_bank.energies

        for i, E in enumerate(energies):
            # 1. 取得 Gamma 值
            gamma_val = SelfEnergyUtils.get_gamma_value(E, mode, wbl_gamma, t_chain, t_coupling)

            # 2. 構造 Source Vector |S>
            # 物理上 Source term S = sum( amplitude_i * sqrt(gamma_i) * |i> )
            S_vec = np.zeros(N, dtype=complex)
            for idx, amp in wavefunction_dict.items():
                # 注意：這裡假設注入強度與 Gamma 有關，符合 Landauer-Büttiker formalism
                # S = A * sqrt(Gamma)
                weight = amp * np.sqrt(gamma_val)
                S_vec[idx] += weight

            # 3. 傳播: |Psi_out> = G^R |S>
            G_R = gf_bank.get_green_function(i)
            psi_out = G_R @ S_vec

            # 4. 計算 Drain 電流: I_out = sum( |psi_out_i|^2 * Gamma_i ) for i in drains
            current_sum = 0.0
            for d_idx in drain_indices:
                amplitude_sq = np.abs(psi_out[d_idx])**2
                current_sum += amplitude_sq * gamma_val
            
            T_values.append(current_sum)

        return np.array(T_values)

    @staticmethod
    def calculate_bond_currents_universal(gf_bank, hamiltonian_matrix, lead_config, source_config, energy_index):
        """
        通用的鍵電流計算函數 (Unified Bond Current Calculator)。
        
        Args:
            gf_bank: GreenFunctionBank 對象
            hamiltonian_matrix: 系統 H (dense or sparse)
            lead_config: Lead 參數 (包含 chain_hopping, coupling 等)
            source_config: 
                - 若為 list [int, int...]: 視為 Incoherent Multi-channel (indices)
                - 若為 dict {idx: complex}: 視為 Coherent Wavefunction
            energy_index: 能量點 index
            
        Returns:
            J_matrix (NxN): J[i,j] 為 i -> j 的電流
        """
        
        # 1. 基礎數據
        N = gf_bank.N
        E = gf_bank.energies[energy_index]
        G_R = gf_bank.get_green_function(energy_index)
        G_A = G_R.conj().T
        
        # 取得物理參數
        mode = lead_config.get('mode', 'WBL')
        wbl_gamma = lead_config.get('wbl_gamma', 0.1)
        t_chain = lead_config.get('chain_hopping', 1.0)
        t_coupling = lead_config.get('coupling', 1.0)
        
        # 2. 構造通用源矩陣 P (Source Matrix)
        P = np.zeros((N, N), dtype=complex)
        
        # 判斷輸入類型
        if isinstance(source_config, list):
            # --- Mode A: Incoherent Multi-channel ---
            # P = i * sum(Gamma_k)
            # 這是統計混合，機率流直接相加
            
            val_gamma = SelfEnergyUtils.get_gamma_value(E, mode, wbl_gamma, t_chain, t_coupling)
            
            for idx in source_config:
                P[idx, idx] += val_gamma 
                
        elif isinstance(source_config, dict):
            # --- Mode B: Coherent (Rank-1) ---
            # P = |W><W|
            # 這是相干疊加，需先構造向量再做外積
            
            W_vec = np.zeros(N, dtype=complex)
            val_gamma = SelfEnergyUtils.get_gamma_value(E, mode, wbl_gamma, t_chain, t_coupling)
            
            for idx, amp in source_config.items():
                # Source vector weight = Amplitude * sqrt(Gamma)
                weight = amp * np.sqrt(val_gamma)
                W_vec[idx] += weight
                
            # 構造密度矩陣 (Outer product)
            P = np.outer(W_vec, W_vec.conj())
            
        else:
            raise ValueError("Invalid source_config format.")

        # 3. 計算相關函數 G_lesser (或 Density Matrix Rho)
        # 公式: Rho = G^R @ P @ G^A
        # 這一步將源頭的電子分佈 "傳播" 到整個系統
        Rho = G_R @ P @ G_A
        
        # 4. 計算鍵電流 J_ij
        # 公式: J_ij = 4 * Im[ H_ji * Rho_ij ]
        # 使用 Element-wise multiplication (*)
        
        if hasattr(hamiltonian_matrix, "toarray"):
            H = hamiltonian_matrix.toarray()
        else:
            H = hamiltonian_matrix
            
        # J_ij = 4 Im(H_ij * Rho_ij)
        M = H * Rho
        J_matrix = 2 * M.imag
        
        print(f'[Debug] J-matrix Hermitian check: {np.allclose(J_matrix, -J_matrix.T)}')
        return J_matrix