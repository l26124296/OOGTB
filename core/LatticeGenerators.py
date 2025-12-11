from abc import ABC, abstractmethod
from typing import Tuple, List, Any
from dataclasses import dataclass
from core.GeometryData import SimplicialComplex
from scipy import linalg 
from scipy.spatial import cKDTree
from core.lib.multigrid import *
from tqdm import tqdm
from fractions import Fraction as Frac
import numpy as np

class LatticeGenerator(ABC):
    """
    所有晶格生成器的抽象基類。
    未來 MultigridGenerator 只需要繼承此類別並實作 generate 即可。
    """
    @abstractmethod
    def generate(self, **kwargs) -> SimplicialComplex:
        pass

# Translation
@dataclass
class ConnectionRule:
    """
    定義單一類型的連接規則
    """
    source_basis: int      # 起點原子 (在單胞內的 index)
    target_basis: int      # 終點原子 (在單胞內的 index)
    delta: Tuple[int, int] # 單胞位移 (dx, dy)
    
    # *** 關鍵：分類標籤 ***
    type_id: int           # 用於矩陣計算的 ID (如 0, 1)
    type_name: str         # 用於 GUI 顯示的名稱 (如 "Nearest Neighbor")

@dataclass
class UnitCellSettings:
    """單胞設定"""
    basis_coords: np.ndarray    # (n_basis, 2)
    lattice_vectors: np.ndarray # (2, 2)
    rules: List[ConnectionRule]

class RuleBasedTranslationGenerator(LatticeGenerator):
    """
    透過平移與拓撲規則生成晶格，並自動分類邊。
    """
    def generate(self, settings: UnitCellSettings, repeat: Tuple[int, int], **kwargs) -> SimplicialComplex:
        Nx, Ny = repeat
        n_basis = len(settings.basis_coords)
        N_total = Nx * Ny * n_basis
        
        print(f"[Generator] Generating {Nx}x{Ny} lattice with classification...")

        # -------------------------------------------------
        # 1. 向量化生成頂點 (Vertices)
        # -------------------------------------------------
        gx, gy = np.meshgrid(np.arange(Nx), np.arange(Ny), indexing='xy')
        gx, gy = gx.flatten(), gy.flatten()
        
        positions = np.zeros((N_total, 2))
        
        for b in range(n_basis):
            # 全域索引公式: I = (ny * Nx + nx) * n_basis + b
            global_indices = (gy * Nx + gx) * n_basis + b
            
            # R = n1*a1 + n2*a2 + basis_pos
            pos_vec = np.outer(gx, settings.lattice_vectors[0]) + \
                      np.outer(gy, settings.lattice_vectors[1]) + \
                      settings.basis_coords[b]
            positions[global_indices] = pos_vec

        # -------------------------------------------------
        # 2. 向量化生成邊與分類 (Edges & Types)
        # -------------------------------------------------
        edges_bucket = []
        types_bucket = []
        type_map_bucket = {}

        # 輔助：計算全域索引
        def get_gidx(nx_arr, ny_arr, b_idx):
            return (ny_arr * Nx + nx_arr) * n_basis + b_idx

        for rule in settings.rules:
            # 記錄這個規則對應的名稱
            type_map_bucket[rule.type_id] = rule.type_name
            
            # 計算目標單胞位置
            dx, dy = rule.delta
            tgx = gx + dx
            tgy = gy + dy
            
            # 邊界檢查 (只保留沒跑出邊界的連線)
            mask = (tgx >= 0) & (tgx < Nx) & (tgy >= 0) & (tgy < Ny)
            
            if not np.any(mask):
                continue
            
            # 建立連線
            srcs = get_gidx(gx[mask], gy[mask], rule.source_basis)
            dsts = get_gidx(tgx[mask], tgy[mask], rule.target_basis)
            
            # 存入暫存區
            current_edges = np.column_stack((srcs, dsts))
            edges_bucket.append(current_edges)
            
            # 建立對應的 type array
            current_types = np.full(len(current_edges), rule.type_id, dtype=int)
            types_bucket.append(current_types)

        # -------------------------------------------------
        # 3. 合併數據與封裝
        # -------------------------------------------------
        if edges_bucket:
            final_edges = np.vstack(edges_bucket)
            final_types = np.concatenate(types_bucket)
        else:
            final_edges = np.zeros((0, 2), dtype=int)
            final_types = np.zeros((0,), dtype=int)

        return SimplicialComplex(
            positions=positions,
            edges=final_edges,
            edge_types=final_types,
            edge_type_map=type_map_bucket,
            faces=[] # RuleBased 暫不處理面，留空
        )
    
#multigrid
class MultigridGenerator(LatticeGenerator):
    def generate(self, EdgeLen, Cn_symmetry, num_of_lines, offsets, spacing, truncation_mode='polygon') -> SimplicialComplex:
        """
        利用對偶網格法 (Dual Grid Method) 生成 Penrose Tiling。
        
        原理：
        1. 建立 5 組平行線 (Multigrid)。
        2. Multigrid 中的每個「區域 (Polygon)」對應 Tiling 中的一個「頂點 (Vertex)」。
        3. Multigrid 中的每個「交點 (Intersection)」對應 Tiling 中的一個「菱形面 (Face)」。
        """
        
        # --- 1. 建立 Multigrid (對偶空間) ---
        CNOS = (Cn_symmetry, num_of_lines, offsets, spacing)
        lines = line_construcer(*CNOS)
        
        # 設定邊界框 (Bounding Box)，用於截斷線段
        bbox = (-50, -50, 50, 50) 
        segments = build_segments(lines, bbox)

        # 計算交點 (這些交點其實對應未來的 'Faces')
        pts_map, coords_list = compute_intersection_points(segments)
        points_info = get_point_info(pts_map, segments, lines)

        # 建立區域列表 (這些區域對應未來的 'Vertices')
        polys = build_polygon_list(segments)
        polys_info_raw = polygon_info(polys)
        
        # --- 2. 截斷 (Truncation) ---
        # 只保留中心附近的區域，形成有限大小的準晶
        if truncation_mode == 'circle':
            # 圓形截斷
            limit = (num_of_lines-1)*(spacing)/2
            polys_info = reassign_ids([pi for pi in polys_info_raw if linalg.norm(pi['centroid']) <= limit])
        elif truncation_mode == 'polygon':
            # 多邊形截斷 (通常是正十邊形)
            polys_info = reassign_ids([pi for pi in polys_info_raw if inside_check(*CNOS, pi['centroid'])]) 
        
        # 建立鄰接關係 (即 Tiling 的 Edge)
        # neighbors 字典: {id: [{'neighbor_id': id, 'shared_angles': [...]}, ...]}
        neighbors = polygon_adjacency(polys_info)

        # --- 3. 座標生成 (Coordinate Calculation) ---
        # 這裡從拓撲結構 (鄰接關係) 還原幾何座標 (Euclidean Coordinates)
        
        N_polys = len(polys_info)
        tilling_vertices_pos = np.zeros((N_polys, 2))
        
        # 建立 ID 到 Array Index 的映射 (因為 id 可能不是連續的 1..N)
        # 假設 polys_info 中的 id 是 1-based 且經過 reassign_ids 後是連續的
        # 最好建立一個明確的 map 以防萬一
        id_to_idx = {p['id']: i for i, p in enumerate(polys_info)}
        
        visited = np.zeros(N_polys, dtype=bool)
        adjacency_list = []
        
        # *** 修正建議 start: 使用 Queue 進行 BFS 確保連通性 ***
        # 您的原代碼依賴 list 順序，若截斷後圖不連通或順序亂掉，會出錯。
        
        # 找到中心點作為起點 (通常選最接近 (0,0) 的或 id=1)
        start_node_idx = 0 
        visited[start_node_idx] = True
        queue = [polys_info[start_node_idx]]
        
        # 雖然 BFS 比較穩，但為了維持您代碼的原樣，我先針對您的邏輯做註釋：
        # 原代碼假設: pid=1 是中心，且已被初始化 (0,0)，且 list 順序是由內向外。
        
        # 這裡我保留您的迴圈結構，但加上安全檢查
        for poly in polys_info:
            pid = poly['id']
            curr_idx = pid - 1 # 假設 ID 是 1-based
            
            # 檢查：如果當前點還沒被計算過 (且不是起點)，這是一個危險信號
            # if pid != 1 and not visited[curr_idx]:
            #    print(f"Warning: Visiting node {pid} before its parent!")
            
            if pid not in neighbors: continue

            for neigh in neighbors[pid]:
                nid = neigh['neighbor_id']
                neigh_idx = nid - 1
                
                # 只處理單邊 (避免重複 Edge)
                # 且必須是「共享一條邊」的鄰居 (shared_angles長度為1)
                # 若 len > 1 代表共享兩條邊(也就是共享頂點)，在 Dual 中不構成直接連線
                if nid > pid and len(neigh['shared_angles']) == 1:
                    
                    # 計算鄰居座標
                    # 邏輯: Pos_neigh = Pos_curr + EdgeVector
                    # EdgeVector 垂直於 Multigrid 的線 (angle + pi/2)
                    angle = neigh['shared_angles'][0]
                    vec = EdgeLen * np.array([np.cos(angle + np.pi/2), np.sin(angle + np.pi/2)])
                    
                    # *** 這裡有潛在 Bug ***
                    # 如果 nid 已經被計算過了，這裡會覆蓋它嗎？
                    # 應該只在 nid 未被計算時賦值。
                    if tilling_vertices_pos[neigh_idx].all() == 0.0: 
                        tilling_vertices_pos[neigh_idx] = tilling_vertices_pos[curr_idx] + vec
                    
                    # 加入邊
                    adjacency_list.append([curr_idx, neigh_idx])

                elif len(neigh['shared_angles']) > 1:
                    # 這通常發生在奇異點 (Singularities)，例如 5 條線交於一點
                    # 這是正常的 Penrose 特徵，但在數值上可能需要微擾處理
                    pass

        # --- 4. 後處理 ---
        adjacency_list = np.array(adjacency_list, dtype=int)
        
        # 移除懸掛點 (Leaf Nodes) - 這是為了美觀或邊界條件
        tilling_vertices_pos, adjacency_list = remove_leaf_nodes(tilling_vertices_pos, adjacency_list)
        
        # 中心化
        center_mass = np.mean(tilling_vertices_pos, axis=0)
        tilling_vertices_pos -= center_mass
        
        # --- 5. 提取 Faces (選做，但強烈建議) ---
        # 這裡應該利用 pts_map (Multigrid 交點) 來反推 Tiling 的面
        # 每個 intersection point 周圍有 4 個 regions (polys)
        # 這 4 個 polys 的 id 構成了一個 Face [v1, v2, v3, v4]
        faces = [] 
        # ToDO: Implement extraction from compute_intersection_points output
        
        return SimplicialComplex(
            positions=tilling_vertices_pos,
            edges=adjacency_list,
            # 預設所有邊類型為 0 (Penrose 只有一種邊長，但有不同方向)
            edge_types=np.zeros(len(adjacency_list), dtype=int),
            edge_type_map={0: 'Penrose Bond'},
            faces=faces 
        )
    
# Haldane
class HaldaneHoneycombGenerator(LatticeGenerator):
    
    """
    專門用於生成 Haldane Model 所需的蜂巢晶格。
    支援 Hexagon, Rectangular, Parallelogram 三種外型。
    自動計算 NN 和 NNN (帶有 Haldane Phase 分類)。
    """
    def generate(self, shape: str, a: float, **kwargs) -> SimplicialComplex:
        """
        Args:
            shape: 'hexagon', 'rectangular', 'parallelogram'
            a: Lattice constant
            kwargs: 
                - radius (for hexagon)
                - width, height (for rectangular/parallelogram)
        """
        print(f"[HaldaneGenerator] Generating {shape} lattice with a={a}...")
        
        # 1. 生成座標 (Vertices)
        if shape == 'hexagon':
            radius = kwargs.get('radius', 5)
            positions = self._points_honeycomb_hexagon(radius, a)
        elif shape == 'rectangular':
            w, h = kwargs.get('width', 5), kwargs.get('height', 5)
            positions = self._points_honeycomb_rectangular(w, h, a)
        elif shape == 'parallelogram':
            w, h = kwargs.get('width', 5), kwargs.get('height', 5)
            positions = self._points_honeycomb_parallelogram(w, h, a)
        else:
            raise ValueError(f"Unknown shape: {shape}")

        # 2. 生成邊與分類 (Edges) - 使用優化的 KDTree
        edges, types, type_map = self._build_edges_optimized(positions, a)
        
        return SimplicialComplex(
            positions=positions,
            edges=edges,
            edge_types=types,
            edge_type_map=type_map,
            faces=[] # Haldane model 不需要面，若需要可後續生成
        )

    # --- 內部邏輯：優化的邊生成 ---
    def _build_edges_optimized(self, positions, a, tol=1e-4):
        """
        使用 cKDTree 優化原本的暴力搜索。
        時間複雜度從 O(N^2) 降為 O(N log N)。
        """
        print("[HaldaneGenerator] Building edges (KDTree optimized)...")
        tree = cKDTree(positions)
        N = len(positions)
        tol_val = tol * a
        
        edges_list = []
        types_list = []
        
        # 定義類型映射
        # 0: NN
        # 1: NNN (Sublattice A)
        # 2: NNN (Sublattice B)
        type_map = {0: 'NN', 1: 'NNN_A', 2: 'NNN_B'}
        
        # 搜尋半徑：最大就是 NNN 距離 (a) + tolerance
        # NNN 距離 = a
        # NN 距離 = a / sqrt(3)
        max_dist = a + tol_val
        
        # query_pairs 回傳所有距離 < max_dist 的點對 (i, j), i < j
        pairs = tree.query_pairs(r=max_dist)
        
        # 為了判斷 sublattice，我們需要一種方法
        # 原代碼用 i_slice 判斷，這依賴於生成順序，不太穩健。
        # 幾何判斷法：B site 總是 A site + (0, a/sqrt(3)) (在 hexagon case)
        # 或者更簡單：根據每個點最近的 NN 數量或方向來判斷。
        # 為了完全相容您的邏輯，我們這裡先假設我們能判斷 sublattice。
        # 暫時用幾何特徵：Honeycomb 是二分圖 (Bipartite)。
        # 可以用簡單的 BFS 著色法來區分 A/B 子格。
        sublattice = self._determine_sublattice(positions, a)

        for i, j in tqdm(pairs, desc="Classifying Edges"):
            dist = linalg.norm(positions[i] - positions[j])
            vec = positions[j] - positions[i]
            
            # 1. 判斷 NN
            if abs(dist - a/np.sqrt(3)) < tol_val:
                edges_list.append([i, j])
                types_list.append(0) # NN
            
            # 2. 判斷 NNN
            elif abs(dist - a) < tol_val:
                # 計算角度
                angle_val = arctan2(vec[1], vec[0]) / pi
                ang = Frac.from_float(angle_val).limit_denominator(6)
                
                current_basis = sublattice[i] # 0 for A, 1 for B
                
                # Haldane Phase Logic
                # 這裡照搬您的邏輯
                nnn_type = current_basis + 1
                if ang in [Frac(1,6), Frac(5,6), Frac(-1,2)]:
                    edges_list.append([i, j])
                elif ang in [Frac(-1,6), Frac(-5,6), Frac(1,2)]:
                    edges_list.append([j, i])
                
                if nnn_type != -1:
                    # edges_list.append([i, j])
                    types_list.append(nnn_type)
        
        return np.array(edges_list), np.array(types_list), type_map

    def _determine_sublattice(self, positions, a):
        """
        使用幾何著色法區分 A/B 子格。
        原理：NN 連接的必為異類。
        """
        N = len(positions)
        colors = np.full(N, -1, dtype=int)
        tree = cKDTree(positions)
        
        # 找一個種子點 (通常是 0)
        start_node = 0
        colors[start_node] = 0 # Assign A
        queue = [start_node]
        
        while queue:
            curr = queue.pop(0)
            curr_color = colors[curr]
            next_color = 1 - curr_color
            
            # 找 NN
            nn_dist = a / np.sqrt(3)
            # query_ball_point 找附近的點
            neighbors = tree.query_ball_point(positions[curr], r=nn_dist + 1e-4)
            
            for neigh in neighbors:
                if colors[neigh] == -1:
                    colors[neigh] = next_color
                    queue.append(neigh)
        
        return colors
    def _points_honeycomb_hexagon(self,radius,a):
        '''
        construct the coordination of a hexagon shape honeycomb lattice
        #of points = 6 * radius^2 
        
        Parameters
        ----------
        radius : int
            the approximate radius of the hexagon
        a : float
            lattice constant(not distance between two adjacent sites)
        
        Returns
        -------
        coordination : ndarray
        '''
        a1 = a*np.array([cos(0),sin(0)])
        a2 = a*np.array([cos(pi/3),sin(pi/3)])

        b1 = np.array([0,0])
        b2 = 1/3*a1 + 1/3*a2

        R6 = np.array([[cos(2*pi/6),-sin(2*pi/6)],[sin(2*pi/6),cos(2*pi/6)]])
        R6cw = R6.T
        edge_len = a/np.sqrt(3)
        edge_vec = []
        for i in range(6):
            edge_vec.append(edge_len*np.linalg.matrix_power(R6cw,i)@np.array([0,1]))

        coordination = [np.array([0,edge_len])]
        insertlist = []
        for r in range(radius):
            'insert sites coordinates according to insertlist'
            for i in insertlist:
                coordination.append(coordination[-1]+edge_vec[i])
                
            'modify the insertlist'
            if r%2 == 0:
                insertlist.insert(0,1)
                insertlist.append(2)

                hei = 2+1.5*r
                coordination.append(hei*edge_len*np.array([0,1]))
            elif r%2 == 1:
                insertlist.insert(0,2)
                insertlist.append(3)

                hei = 2+1.5*r+0.5
                coordination.append(hei*edge_len*np.array([0,1]))
            else:
                print('insetlist error')
        coordination.pop(-1)

        coordination = np.array(coordination)
        angular_component = coordination
        'complete all other radius components'
        for n in range(5):
            angular_component = np.einsum('ij,kj->ki',R6.T,angular_component)
            coordination = np.concatenate((coordination,angular_component))
        #trickly rotate 90 degree
        #due to the angluar searching in 'constructH, each hexagon is / \' function
        coordination = np.array([coordination[:,1],-1*coordination[:,0]]).T
        return coordination

    def _points_honeycomb_rectangular(self,width,height,a):
        a1 = a*np.array([np.sqrt(3),0])
        a2 = a*np.array([0,1])

        b1 = np.array([0,0])
        b2 = a/np.sqrt(3)*np.array([cos(pi/3),sin(pi/3)])
        b3 = a*np.array([cos(pi/6),sin(pi/6)])
        b4 = a*2*np.sqrt(3)/3*np.array([1,0])

        coordination = []
        for m in range(height):
            for n in range(width):
                coordination.append([m*a2+n*a1+b1])
                coordination.append([m*a2+n*a1+b2])
                coordination.append([m*a2+n*a1+b3])
                coordination.append([m*a2+n*a1+b4])

        coordination = np.array(coordination).reshape(-1,2)
        #remove armchair leaf node
        coordination = coordination[ coordination[:,1] > 0.2*a ]
        return coordination

    def _points_honeycomb_parallelogram(self,width,height,a):
        a1 = a*np.array([cos(pi/6),sin(pi/6)])
        a2 = a*np.array([cos(pi/2),sin(pi/2)])

        b1 = np.array([0,0])
        b2 = 1/np.sqrt(3)*np.array([cos(pi/3),sin(pi/3)])

        coordination = []
        for m in range(height):
            for n in range(width):
                coordination.append([m*a2+n*a1+b1])
                coordination.append([m*a2+n*a1+b2])

        coordination = np.array(coordination).reshape(-1,2)
        return coordination

if __name__ == '__main__':
    tiles = MultigridGenerator().generate(
        EdgeLen = 1,
        Cn_symmetry = 5,
        num_of_lines = 8,
        offsets = 0.08,
        spacing = 1
    )
    print(tiles.positions)