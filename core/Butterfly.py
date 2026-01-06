import numpy as np
import traceback
import pickle
from scipy import linalg as sla  # Dense solver for full spectrum
from PyQt6.QtCore import QThread, pyqtSignal

# 引用 HamiltonianModel
from core.PhysicsModel import HamiltonianModel

# 嘗試引用 PyTorch
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# ==============================================================================
# 1. ButterflyCage (Data Container)
# ==============================================================================
class ButterflyCage:
    def __init__(self, eigenvalues=None, params=None, mode='1D', config=None, path_ticks=None):
        self.eigenvalues = eigenvalues if eigenvalues is not None else np.array([])
        self.params = params if params is not None else np.array([])
        self.mode = mode
        self.config = config if config else {}
        self.path_ticks = path_ticks if path_ticks else {}
    @classmethod
    def from_entomologist(cls, entomologist, results):
        return cls(
            eigenvalues=results['eigenvalues'],
            params=results['params'],
            path_ticks=results['path_ticks'],
            mode=entomologist.mode,
            config={
                'control_vars': entomologist.control_vars,
                'resolution': entomologist.resolution,
                'sweep_config': entomologist.sweep_config
            }
        )

    def save(self, filename):
        try:
            if filename.endswith('.npz'):
                np.savez(filename, 
                         eigenvalues=self.eigenvalues, 
                         params=self.params, 
                         mode=self.mode,
                         config=self.config)
            else:
                with open(filename, 'wb') as f:
                    pickle.dump(self, f)
            return True
        except Exception as e:
            print(f"Save failed: {e}")
            return False

    @classmethod
    def load(cls, filename):
        try:
            if filename.endswith('.npz'):
                data = np.load(filename, allow_pickle=True)
                cfg = data['config'].item() if data['config'].shape == () else data['config']
                return cls(data['eigenvalues'], data['params'], str(data['mode']), cfg)
            else:
                with open(filename, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            print(f"Load failed: {e}")
            return None


# ==============================================================================
# 2. Entomologist (Worker Logic)
# ==============================================================================
class Entomologist(QThread):
    progress_sig = pyqtSignal(int)
    finished_sig = pyqtSignal(object) 
    error_sig = pyqtSignal(str)

    def __init__(self, hamiltonian_model, control_vars, sweep_config):
        """
        :param geometry: SimplicialComplex
        :param control_vars: Dict {'onsite_val', 'hopping_map', ...}
        :param sweep_config: Dict {'mode', 'min', 'max', 'path_points', 'areas'...}
        """
        super().__init__()
        
        if HamiltonianModel is None:
            raise ImportError("HamiltonianModel class is missing.")
            
        self.model = hamiltonian_model
        self.control_vars = control_vars
        self.sweep_config = sweep_config
        
        self.mode = sweep_config.get('mode', '1D')
        self.resolution = sweep_config.get('resolution', 100)
        
        # 預先計算物理磁場序列 (Flux Steps)
        # 注意：計算這個可能需要一點時間 (特別是 2D Torch 搜索)，但只執行一次
        self.path_matrix , self.path_ticks = self._calculate_path_matrix()
        
        self._is_running = True

    def _calculate_path_matrix(self):
        """
        產生磁場序列與路徑標記。
        Returns: 
            (np.array, dict) -> (b_field_values, {index: "Label"})
        """
        
        # --- CASE 1: 1D Sweep ---
        if self.mode == '1D':
            f_min = self.sweep_config.get('min', 0.0)
            f_max = self.sweep_config.get('max', 1.0)
            
            # 1D 簡單標記頭尾
            vals = np.linspace(f_min, f_max, self.resolution)
            ticks = {0: f"{f_min:.2f}", len(vals)-1: f"{f_max:.2f}"}
            return vals, ticks
            
        # --- CASE 2: 2D Diophantine Path ---
        elif self.mode == '2D':
            # 讀取 path_points (tuple list)
            path_points = self.sweep_config.get('path_points', [])
            areas = self.sweep_config.get('areas', (1.0, 1.0)) 
            
            if len(path_points) < 2:
                return np.array([]), {}
            
            # A. 產生路徑點 (xi, yi) 與 Ticks
            raw_path = []
            ticks = {} # 用來存 {index: "(x, y)"}
            
            path_arr = np.array(path_points)
            n_segments = len(path_arr) - 1
            
            # 防止除以零或是過少點數
            if n_segments < 1: n_segments = 1
            cuts_per_seg = max(2, self.resolution // n_segments)
            
            current_idx = 0

            for i in range(n_segments):
                p_start = path_arr[i]
                p_end = path_arr[i+1]
                
                # 1. 記錄這一段起點的 Tick
                # 格式化 Label 為 "(0.12, 0.34)"
                label_str = f"({p_start[0]:.2g}, {p_start[1]:.2g})"
                ticks[current_idx] = label_str
                
                # 2. 插值生成路徑
                # 如果不是最後一段，endpoint=False 以避免下一段起點重複
                is_last_segment = (i == n_segments - 1)
                ts = np.linspace(0, 1, cuts_per_seg, endpoint=is_last_segment)
                
                for t in ts:
                    pt = p_start + (p_end - p_start) * t
                    raw_path.append(pt)
                
                # 3. 更新 Index 計數
                current_idx += len(ts)

            # 4. 補上最後一個點的 Tick
            p_last = path_arr[-1]
            last_label = f"({p_last[0]:.2g}, {p_last[1]:.2g})"
            ticks[len(raw_path) - 1] = last_label

            # B. 將 (xi, yi) 轉換為 B-field (保留您的 Diophantine 邏輯)
            A_thin, A_thick = areas
            if A_thick == 0 or A_thin == 0:
                raise ValueError("2D areas must be both non-zero.")
            r = A_thin / A_thick 
            
            b_field_sequence = []

            search_func = self._diophantine_search_torch if HAS_TORCH else self._diophantine_search_numpy
            
            print(f"[Entomologist] Starting 2D Path Search using {'PyTorch GPU' if HAS_TORCH else 'NumPy CPU'}...")
            for pt in raw_path:
                xi, yi = pt[0], pt[1] # raw_path 裡的元素是 numpy array
                u = xi - r * yi
                # 執行搜索
                n, m, error = search_func(r, u)
                
                # 反推磁場
                b1 = 2 * np.pi * (m + xi) / A_thin
                b2 = 2 * np.pi * (n + yi) / A_thick
                
                b_avg = ( b1 + b2 ) / 2
                b_field_sequence.append(b_avg)
            return np.array(b_field_sequence), ticks
        
        return np.array([]), {}

    # --------------------------------------------------------------------------
    # Diophantine Search Implementations
    # --------------------------------------------------------------------------

    @staticmethod
    def _diophantine_search_numpy(r, u, n_search_range=10000):
        """
        NumPy 向量化版本 (CPU)。
        當沒有 PyTorch 或 GPU 時的備案。
        求解: min |r*n - m - u|
        """
        # 1. 建立 n 的範圍陣列
        n_vals = np.arange(0, n_search_range + 1, dtype=np.int64)
        
        # 2. 計算理想的 m (浮點數)
        #    由 r*n - m = u => m = r*n - u
        m_ideal = r * n_vals - u
        
        # 3. 四捨五入得到整數 m
        m_vals = np.round(m_ideal).astype(np.int64)
        
        # 4. 計算誤差
        errors = np.abs(m_ideal - m_vals)
        
        # 5. 找出最小誤差的索引
        min_idx = np.argmin(errors)
        
        best_n = n_vals[min_idx]
        best_m = m_vals[min_idx]
        min_error = errors[min_idx]
        
        return best_n, best_m, min_error

    @staticmethod
    def _diophantine_search_torch(r, u, n_search_range=10000, batch_size=10_000_000):
        """
        PyTorch (GPU) 版本。
        求解: rn - m = u
        """
        if not torch.cuda.is_available():
            return Entomologist._diophantine_search_numpy(r, u, n_search_range)

        device = torch.device('cuda')
        
        r_gpu = torch.tensor(r, dtype=torch.float64, device=device)
        u_gpu = torch.tensor(u, dtype=torch.float64, device=device)
        
        min_error_global = torch.tensor(float('inf'), dtype=torch.float64, device=device)
        best_n_global = torch.tensor(0, dtype=torch.long, device=device)
        best_m_global = torch.tensor(0, dtype=torch.long, device=device)

        total_n_count = 2 * n_search_range + 1
        num_batches = (total_n_count + batch_size - 1) // batch_size
        
        for i in range(num_batches):
            n_start =  i * batch_size
            n_end = min([-n_search_range + (i + 1) * batch_size, n_search_range + 1])
            
            if n_start >= n_end: continue

            n_batch_gpu = torch.arange(n_start, n_end, dtype=torch.long, device=device)

            # m_ideal = r*n - u
            m_ideal_gpu = r_gpu * n_batch_gpu - u_gpu
            m_int_gpu = torch.round(m_ideal_gpu)
            errors_gpu = torch.abs(m_ideal_gpu - m_int_gpu)

            min_error_batch, min_error_idx_batch = torch.min(errors_gpu, dim=0)

            if min_error_batch < min_error_global:
                min_error_global = min_error_batch
                best_n_global = n_batch_gpu[min_error_idx_batch]
                best_m_global = m_int_gpu[min_error_idx_batch]
            
            del n_batch_gpu, m_ideal_gpu, m_int_gpu, errors_gpu, min_error_batch, min_error_idx_batch

        torch.cuda.synchronize()
        return best_n_global.item(), best_m_global.item(), min_error_global.item()

    # --------------------------------------------------------------------------
    # Main Execution Logic
    # --------------------------------------------------------------------------

    def construct(self, flux_param):
        """
        將控制變數注入 Model 並建構。
        flux_param 在此處已經是物理磁場 (Scalar B-field)。
        """
        self.model.construct(b_field=flux_param,silent=True, **self.control_vars)

    def run(self):
        try:
            total_steps = len(self.path_matrix)
            if total_steps == 0:
                self.error_sig.emit("Path calculation failed or empty.")
                return

            all_eigenvalues = []
            valid_params = [] # 這裡存的是計算後的物理磁場 B

            # 開始主迴圈
            for i, b_field in enumerate(self.path_matrix):
                if not self._is_running:
                    break

                # 1. 建構 Hamiltonian (更新 Peierls Phase)
                self.construct(b_field)

                # 2. 求解特徵值 (Dense Solver for Full Spectrum)
                H = self.model.H_sparse
                if hasattr(H, 'toarray'):
                    H_dense = H.toarray()
                else:
                    H_dense = H
                
                vals = sla.eigh(H_dense, eigvals_only=True)
                
                all_eigenvalues.append(vals)
                valid_params.append(b_field)

                # 進度回報 (降低頻率)
                if total_steps < 100 or i % (total_steps // 100) == 0:
                    self.progress_sig.emit(int((i + 1) / total_steps * 100))

            if self._is_running:
                results = {
                    'eigenvalues': np.array(all_eigenvalues),
                    'params': np.array(valid_params),
                    'path_ticks': self.path_ticks
                }
                cage = ButterflyCage.from_entomologist(self, results)
                self.finished_sig.emit(cage)
            else:
                self.error_sig.emit("Calculation aborted.")

        except Exception as e:
            traceback.print_exc()
            self.error_sig.emit(str(e))

    def stop(self):
        self._is_running = False