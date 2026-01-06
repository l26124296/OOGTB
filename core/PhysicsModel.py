import numpy as np
import scipy.sparse as sp
import warnings
from scipy.spatial import cKDTree
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

# 引入您定義的幾何結構
from core.GeometryData import SimplicialComplex

class HamiltonianModel:
    """
    負責從幾何結構構建哈密頓量，並包含物理場效應 (磁場、電位)。
    """
    def __init__(self, geometry: SimplicialComplex):
        self.geo = geometry
        self.H_sparse: Optional[sp.csc_matrix] = None # 儲存稀疏矩陣
        self.Op_X: Optional[sp.csc_matrix] = None    # 儲存位置算符 X (對角矩陣)
        self.Op_Y: Optional[sp.csc_matrix] = None    # 儲存位置算符 Y (對角矩陣)
        # 快取子晶格資訊 (Bipartite coloring)，避免每次重複計算
        self._sublattice_tags: Optional[np.ndarray] = None 
        self._is_bipartite: Optional[bool] = None


    def construct(self, 
                  ttdict: Dict[str, complex], 
                  b_field: float = 0.0, 
                  vec_pot_mode: str = 'Landau',
                  onsite_config: Dict = None
                  , silent: bool = False) -> sp.csc_matrix:
        """
        構建哈密頓量 (向量化實作)
        
        Args:
            ttdict: Hopping 參數字典, e.g. {'NN': 1.0, 'NNN_A': 0.1}
            b_field: 磁場強度 (Flux per area unit)
            vec_pot_mode: 'Landau' or 'Circular'
            onsite_config: 額外位能設定{'type':('bipartite, 'laplace, 'random'), 'scale': float}
        """
        if not silent:
            print(f"[Physics] Constructing Hamiltonian (B={b_field}, Gauge={vec_pot_mode})...")
        
        N = self.geo.positions.shape[0]
        edges = self.geo.edges
        edge_types = self.geo.edge_types
        
        if len(edges) == 0:
            self.H_sparse = sp.csc_matrix((N, N), dtype=complex)
            return self.H_sparse

        # ---------------------------------------------------------
        # 1. 準備 Hopping Amplitudes (根據 type 映射)
        # ---------------------------------------------------------
        # 建立一個 lookup array，讓 edge_type (int) 直接映射到 hopping (float)
        # 假設 edge_type_map = {0: 'NN', 1: 'NNN'}
        # ttdict = {'NN': 1.0, 'NNN': 0.1}
        
        max_type_id = max(self.geo.edge_type_map.keys()) if self.geo.edge_type_map else 0
        hopping_lookup = np.zeros(max_type_id + 1, dtype=complex)
        
        for type_id, name in self.geo.edge_type_map.items():
            if name in ttdict:
                hopping_lookup[type_id] = ttdict[name]
            else:
                # 若 ttdict 沒定義該邊，預設為 0
                print(f"Warning: Edge type '{name}' not found in ttdict. Setting t=0.")
                hopping_lookup[type_id] = 0.0
                
        # 向量化查找：所有邊的基礎 Hopping 值
        # base_hoppings shape: (M,)
        base_hoppings = hopping_lookup[edge_types]

        # ---------------------------------------------------------
        # 2. 計算 Peierls Phase (向量化)
        # ---------------------------------------------------------
        phase = np.zeros(len(edges))
        
        if b_field != 0 and vec_pot_mode != 'off':
            # 提取座標: r1 (起點), r2 (終點)
            r1 = self.geo.positions[edges[:, 0]]
            r2 = self.geo.positions[edges[:, 1]]
            
            x1, y1 = r1[:, 0], r1[:, 1]
            x2, y2 = r2[:, 0], r2[:, 1]
            
            if vec_pot_mode == 'Landau':
                # A = (0, Bx, 0) -> Integral ~ B * x_avg * dy
                # Analytical Integration result :
                #  b*(dy*x0 + dx*dy/2) = b * dy * (x0 + dx/2) = b * dy * x_mid
                dx = x2 - x1
                dy = y2 - y1
                phase = b_field * (dy * (x1 + dx/2))
                
            elif vec_pot_mode == 'Circular':
                # A = (-By/2, Bx/2, 0) -> Integral = B/2 * (x1*y2 - x2*y1)
                # Analytical Integration result :
                #  b*(dy*x0 + dx*dy/2) = b * dy * (x0 + dx/2) = b * dy * x_mid
                phase = (b_field / 2) * (x1 * y2 - x2 * y1)
                
            else:
                raise ValueError(f"Unknown gauge: {vec_pot_mode}")

        # 結合 Hopping 和 Phase
        # H_ij = t * exp(i * phase)
        final_hoppings = base_hoppings * np.exp(1j * phase)

        # ---------------------------------------------------------
        # 3. 構建稀疏矩陣 (COO -> CSR/CSC)
        # ---------------------------------------------------------
        row = edges[:, 0]
        col = edges[:, 1]
        
        # 構建上三角 + 下三角 (Hermitian)
        # 注意：我們需要處理 H[i,j] 和 H[j,i]
        # data = [h_12, h_21, h_23, h_32 ...]
        
        all_rows = np.concatenate([row, col])
        all_cols = np.concatenate([col, row])
        all_data = np.concatenate([final_hoppings, np.conj(final_hoppings)])
        
        self.H_sparse = sp.csc_matrix((all_data, (all_rows, all_cols)), shape=(N, N))

        # ---------------------------------------------------------
        # 4. 處理 On-site Potential (對角線項)
        # ---------------------------------------------------------
        if onsite_config:
            mode = onsite_config.get('type')
            
            if mode == 'bipartite':
                s_val = onsite_config.get('scale', 0.0)
                if s_val != 0:
                    # 預設使用 type 0 (NN) 來判斷
                    tags, success = self._get_bipartite_tags(target_type_id=0)
                    
                    if success:
                        potential = s_val * (tags - 0.5) 
                        self.H_sparse += sp.diags(potential)
                    else:
                        # 這裡拋出異常，讓外層 GUI 捕捉並印出 Error Log
                        raise RuntimeError("Bipartite check failed! Lattice is frustrated or NN definition incorrect.")
                    
            elif mode == 'random':
                # 隨機 On-site Potential
                s_val = onsite_config.get('scale', 0.0) 
                add_on = np.random.rand(N) * s_val  
                self.H_sparse += sp.diags(add_on-np.mean(add_on))

            elif mode == 'laplace':
                # Laplace term: H = Lap_scale * D - H_hopping
                # D 是度數矩陣 (Degree Matrix) 或者 onsite energy sum
                scale = onsite_config.get('scale', 1.0)
                
                # 計算每一列的 hopping 總和 (不含相位，通常只看強度或連線數)
                # 您原本的代碼: D = diag(sum(H)) -> 這是 row sum
                # 注意：這會受到相位影響變成複數，物理上 phonon matrix 應該用 abs(t)
                
                # 簡單做法：直接對目前的 H 取 row sum
                row_sums = np.array(np.sum(self.H_sparse.astype(bool), axis=1)).flatten()
                col_sums = np.array(np.sum(self.H_sparse.astype(bool), axis=0)).flatten()

                D = sp.diags((row_sums+col_sums)/2)
                
                self.H_sparse = scale * D + self.H_sparse

        return self.H_sparse

    def _get_bipartite_tags(self, target_type_id=0) -> tuple[np.ndarray, bool]:
        """
        只使用特定的 edge_type (通常是 NN, id=0) 進行二分圖檢查。
        
        Returns:
            tags (np.ndarray): 0/1 array.
            is_bipartite (bool): 是否成功二分。
        """
        # 已經算過且成功，直接回傳
        if self._is_bipartite is True:
            return self._sublattice_tags, True
        # 已經算過但失敗，直接回傳
        elif self._is_bipartite is False:
            return self._sublattice_tags, False

        N = self.geo.positions.shape[0]
        tags = np.full(N, -1, dtype=int)
        
        # 1. 篩選邊：只保留 type == target_type_id 的邊
        mask = (self.geo.edge_types == target_type_id)
        relevant_edges = self.geo.edges[mask]
        
        # 2. 建立鄰接表
        adj = [[] for _ in range(N)]
        for u, v in relevant_edges:
            adj[u].append(v)
            adj[v].append(u)
            
        self._is_bipartite = True
        
        # 3. BFS 著色
        for i in range(N):
            if tags[i] != -1: continue
            
            queue = [i]
            tags[i] = 0
            
            while queue:
                u = queue.pop(0)
                next_tag = 1 - tags[u]
                
                for v in adj[u]:
                    if tags[v] == -1:
                        tags[v] = next_tag
                        queue.append(v)
                    elif tags[v] == tags[u]:
                        # 找到奇環
                        self._is_bipartite = False
                        self._sublattice_tags = None
                        return None, self._is_bipartite
                        
        # 4. 儲存與回傳
        # 只有成功時才 cache，失敗時不 cache (或視需求決定)
        if self._is_bipartite:
            self._sublattice_tags = tags
            
        return tags, self._is_bipartite

    # --- 未來擴充物理量计算 ---
    def _clear_cache(self):
        self.eigenvalues = None
        self.eigenvectors = None
        # 算符 X, Y, Lz 通常只跟幾何有關，不用清，除非幾何變了

    def diagonalize(self):
        """
        全對角化 (Full Diagonalization)
        注意：對於超大系統 (>10000 sites) 可能會很慢，建議用 eigsh 算部分。
        但為了 Chern Marker 通常需要全頻譜。
        """
        if self.H_sparse is None: raise RuntimeError("Hamiltonian not constructed.")
        
        print("[Physics] Diagonalizing Hamiltonian...")
        # 轉換為 dense matrix 進行全對角化 (eigh for Hermitian)
        H_dense = self.H_sparse.todense()
        self.eigenvalues, self.eigenvectors = np.linalg.eigh(H_dense)
        print(f"[Physics] Done. E range: [{self.eigenvalues[0]:.4f}, {self.eigenvalues[-1]:.4f}]")
        
    def get_position_operators(self) -> Tuple[sp.csc_matrix, sp.csc_matrix]:
        """建構位置算符 X, Y (對角矩陣)"""
        if self.Op_X is not None: return self.Op_X, self.Op_Y
        
        N = self.geo.positions.shape[0]
        x = self.geo.positions[:, 0]
        y = self.geo.positions[:, 1]
        
        self.Op_X = sp.diags(x, format='csc')
        self.Op_Y = sp.diags(y, format='csc')
        return self.Op_X, self.Op_Y

    def build_rotation_operator(self, n, tol=1e-5, center_tol=1e-9, verbose: bool = False):
        """
        build_rotation_operator(package, n, tol=1e-6, center_tol=1e-9, verbose=True)

        Inputs:
        - package: dict with keys at least 'd0' (ndarray shape (N,2) of vertex positions).
                    (You said your existing package dict = {'d0': positions, 'd1': adjacency_list})
        - n: int, C_n discrete rotational symmetry order
        - tol: float, maximum distance to accept a rotated coordinate as matching an existing site
        - center_tol: float, tolerance to identify the center site at (0,0)
        - verbose: bool, print diagnostics

        Returns (dict):
        {
        'R': scipy.sparse.csr_matrix (N x N complex) rotation operator,
        'mapping': np.ndarray shape (N,) where mapping[i] = j such that R[j,i]=1,
        'center_index': int or None,
        'normality_err': float (Frobenius norm of R^H R - R R^H),
        'rpower_err': float (max |R^n - I| element absolute),
        'rpower_mat_err': float (Frobenius norm of R^n - I),
        'is_normal': bool,
        'is_nth_identity': bool
        }
        
        Notes:
        - The function expects each non-center site to be mapped one-to-one by rotation.
        - If some rotated coordinates cannot be matched within tol, function will raise an error (you can relax tol).
        - If multiple sites map to same target (collision), that is reported.
        """
        positions = self.geo.positions
        if positions.ndim != 2 or positions.shape[1] != 2:
            raise ValueError("package['d0'] must be shape (N,2).")
        N = positions.shape[0]
        # find center site index (close to (0,0))
        dists_to_origin = np.linalg.norm(positions, axis=1)
        center_candidates = np.where(dists_to_origin <= center_tol)[0]
        if center_candidates.size == 0:
            center_index = None
            if verbose:
                print("Warning: no center site found within center_tol =", center_tol)
        else:
            # take the one closest to origin
            center_index = int(center_candidates[np.argmin(dists_to_origin[center_candidates])])
            if verbose:
                print(f"Center site found: index {center_index}, pos {positions[center_index]}")
        # build KDTree for nearest-neighbor lookup
        kdtree = cKDTree(positions)
        # rotation matrix (2x2) by angle +2pi/n (rotate CCW)
        theta = 2.0 * np.pi / float(n)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        R2 = np.array([[cos_t, -sin_t],
                    [sin_t,  cos_t]])
        # prepare sparse matrix in LIL for assignment
        R_lil = sp.lil_matrix((N, N), dtype=np.complex128)
        mapping = -np.ones(N, dtype=int)
        # handle center: map center -> center (if exists)
        if center_index is not None:
            R_lil[center_index, center_index] = 1.0
            mapping[center_index] = center_index
        # for each site i (except center) compute rotated coordinate and find nearest site j
        for i in range(N):
            if center_index is not None and i == center_index:
                continue
            pi = positions[i]
            rotated = R2.dot(pi)
            # query nearest neighbor within tol
            dist, j = kdtree.query(rotated, k=1, distance_upper_bound=tol)
            # cKDTree returns j == kdtree.n if no neighbor found within bound
            if np.isinf(dist) or j >= N:
                # No match: maybe numerical shift or your tiling is not perfectly symmetric
                raise RuntimeError(f"Rotated coordinate of site {i} ({rotated}) has no match within tol={tol}. "
                                "Consider increasing tol or check sites_by_slice mapping.")
            # assign mapping (j index)
            mapping[i] = int(j)
            R_lil[int(j), i] = 1.0
        # check for collisions / surjectivity: ideally mapping is a permutation on non-center sites
        unique_targets, counts = np.unique(mapping[mapping>=0], return_counts=True)
        collisions = unique_targets[counts > 1]
        if collisions.size > 0:
            warnings.warn(f"Rotation mapping has collisions (multiple sources map to same target): {collisions.tolist()}")
        # optionally check that all non-center sites are covered as targets (permutation)
        # compute set of expected non-center indices
        expected_targets = set(range(N))
        if center_index is not None:
            expected_targets.remove(center_index)
        mapped_targets = set(mapping[mapping>=0].tolist())
        missing_targets = expected_targets - mapped_targets
        if len(missing_targets) > 0:
            warnings.warn(f"Some targets not mapped: {sorted(list(missing_targets))[:10]} (count {len(missing_targets)})")
        # finalize sparse matrix
        R = sp.csr_matrix(R_lil)
        # Diagnostics: normality check (R^H R - R R^H)
        # compute A = R^H * R - R * R^H
        R_H = R.conj().T
        A = (R_H.dot(R) - R.dot(R_H)).tocoo()
        normality_err = np.sqrt(np.sum(np.abs(A.data)**2))  # Frobenius norm of commutator
        is_normal = normality_err <= 1e-8  # threshold, may adjust
        # Check R^n == I (use repeated multiplication)
        I = sp.identity(N, dtype=np.complex128, format='csr')
        Rpow = sp.identity(N, dtype=np.complex128, format='csr')
        for _ in range(n):
            Rpow = R.dot(Rpow)
        D = (Rpow - I).tocoo()
        rpower_mat_err = np.sqrt(np.sum(np.abs(D.data)**2))
        # max absolute entry dev
        max_entry_dev = 0.0
        if D.nnz > 0:
            max_entry_dev = np.max(np.abs(D.data))
        is_nth_identity = max_entry_dev <= 1e-6  # threshold
        if verbose:
            print(f"Rotation operator constructed: N={N}, order n={n}")
            print(f"normality (Frobenius of [R^H,R]) = {normality_err:.3e}  -> is_normal={is_normal}")
            print(f"R^{n} vs I: Frobenius err = {rpower_mat_err:.3e}, max entry err = {max_entry_dev:.3e} -> is_nth_identity={is_nth_identity}")
        # return dictionary
        
        # return {
        #     'R': R,
        #     'mapping': mapping,
        #     'center_index': center_index,
        #     'normality_err': float(normality_err),
        #     'is_normal': bool(is_normal),
        #     'rpower_mat_err': float(rpower_mat_err),
        #     'rpower_max_entry_err': float(max_entry_dev),
        #     'is_nth_identity': bool(is_nth_identity)
        # }
        return R
    
    def get_L_operator(self , Cn , verbose: bool = False) -> sp.csc_matrix:
            """
            給定旋轉算符 R (scipy.sparse.csr_matrix)，構造角動量算符 L。
            定義: L = sum_k k * P_k
            其中 P_k = (1/n) * sum_m ω^{-km} R^m, ω = exp(2πi/n).
            
            Parameters
            ----------
            R : csr_matrix (N x N), rotation operator
            n : int, rotational symmetry order (R^n = I)
            dtype : type, matrix dtype (default complex128)
            verbose : bool, 是否輸出檢查資訊
            
            Returns
            -------
            L : csr_matrix (N x N), angular momentum operator
            """
            dtype = np.complex128
            R = self.build_rotation_operator(Cn)
            N = R.shape[0]
            omega = np.exp(2j * np.pi / Cn)
            I = sp.identity(N, dtype=dtype, format='csr')
            
            # powers of R
            R_powers = [I]
            for m in range(1, Cn):
                R_powers.append(R_powers[-1].dot(R))  # R^m
            
            # construct projectors P_k
            P_list = []
            for k in range(Cn):
                coeffs = [omega**(-k*m) for m in range(Cn)]
                Pk = sum(c * Rp for c, Rp in zip(coeffs, R_powers)) * (1.0/Cn)
                P_list.append(Pk)
            
            # angular momentum operator
            L = sum(k * P_list[k] for k in range(Cn))
            L = sp.csr_matrix(L, dtype=dtype)
            
            if verbose:
                # check hermiticity
                diff = (L - L.getH()).power(2).sum()
                print(f"[check] L Hermitian diff^2 sum = {diff:.3e}")
                # check commutation with R
                comm = (R.dot(L) - L.dot(R)).power(2).sum()
                print(f"[check] [R,L] Frobenius^2 = {comm:.3e}")
            
            return L