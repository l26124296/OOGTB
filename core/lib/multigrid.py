from shapely.geometry import LineString, Point, box
from shapely.ops import unary_union, polygonize
from collections import defaultdict
from numpy import arctan2 , sin , cos , pi , arange
import time
import torch
import numpy as np
EPS = 1e-9

def angle_from_two_points(p1, p2):
    x1,y1 = p1
    x2,y2 = p2
    return arctan2(y2-y1, x2-x1)

def angle_from_line_coeff(a, b, c):
    # line: a x + b y + c = 0  => y = (-a/b) x + (-c/b)  if b != 0
    if abs(b) < EPS:
        return float('inf')
    return arctan2(-a, b)

def clip_line_to_bbox(line_coeff, bbox):
    # line_coeff: (a,b,c) for ax+by+c=0
    # bbox: (xmin,ymin,xmax,ymax)
    # return a shapely LineString clipped to bbox (possibly empty)
    xmin,ymin,xmax,ymax = bbox
    rect = box(xmin,ymin,xmax,ymax)
    a,b,c = line_coeff
    # parameterize line: choose two far points on the line
    if abs(b) > EPS:
        # pick x far
        x1, x2 = xmin - 10*(xmax-xmin+1), xmax + 10*(xmax-xmin+1)
        y1 = (-a*x1 - c)/b
        y2 = (-a*x2 - c)/b
    else:
        # vertical line x = -c/a
        if abs(a) < EPS:
            return None
        x = -c/a
        y1, y2 = ymin - 10*(ymax-ymin+1), ymax + 10*(ymax-ymin+1)
        x1 = x2 = x
    big_line = LineString([(x1,y1),(x2,y2)])
    inter = big_line.intersection(rect)
    if inter.is_empty:
        return None
    # intersection could be LineString or MultiLineString or Point
    if inter.geom_type == 'Point':
        # single point -> treat as degenerate line (zero-length)
        return LineString([inter.coords[0], inter.coords[0]])
    if inter.geom_type == 'LineString':
        return inter
    # if MultiLineString, merge into one by unary_union and return as multiline's pieces
    # but for polygonize we want all segments; here return union (could be MultiLineString)
    return inter  # may be MultiLineString

def build_segments(lines_coeffs, bbox):
    """
    lines_coeffs: list of (a,b,c) for ax+by+c=0
    bbox: (xmin,ymin,xmax,ymax)
    return: list of shapely LineString (segments clipped to bbox)
    """
    segments = []
    for coeff in lines_coeffs:
        seg = clip_line_to_bbox(coeff, bbox)
        if seg is None:
            continue
        # If seg is MultiLineString, break into components
        if hasattr(seg, 'geoms'):
            for g in seg.geoms:
                if g.length > EPS:
                    segments.append(g)
        else:
            if seg.length > EPS:
                segments.append(seg)
    return segments

def compute_intersection_points(segments):
    """
    segments: list of LineString
    return:
      points_map: dict mapping (x,y) -> set of segment indices that pass through it
      coords_list: list of (x,y)
    """
    n = len(segments)
    pts_map = defaultdict(set)
    for i in range(n):
        s1 = segments[i]
        for j in range(i+1, n):
            s2 = segments[j]
            inter = s1.intersection(s2)
            if inter.is_empty:
                continue
            # intersection might be Point or LineString (collinear overlap)
            if inter.geom_type == 'Point':
                x,y = round(inter.x, 12), round(inter.y, 12)
                pts_map[(x,y)].update([i,j])
            else:
                # overlap segment: we should treat overlap endpoints as intersection points
                # add endpoints of the overlapping geometry
                try:
                    for g in (inter.geoms if hasattr(inter, 'geoms') else [inter]):
                        coords = list(g.coords)
                        for c in coords:
                            x,y = round(c[0],12), round(c[1],12)
                            pts_map[(x,y)].update([i,j])
                except Exception:
                    pass
    # also add segment endpoints as graph vertices (to ensure polygonize works)
    for idx, s in enumerate(segments):
        for c in s.coords:
            x,y = round(c[0],12), round(c[1],12)
            pts_map[(x,y)].add(idx)
    # produce coords list
    coords_list = list(pts_map.keys())
    return pts_map, coords_list

def point_to_shapely(pt):
    return Point(pt[0], pt[1])

def get_point_info(pts_map, segments, lines_coeffs):
    """
    Produce list of dictionaries for each intersection point with:
      - id (serial)
      - coord
      - lines_idx (indices of segments that pass through it)
      - slope_of_through_lines (list of slopes corresponding to original line coeffs)
    """
    info = []
    for i, (coord, seg_idxs) in enumerate(sorted(pts_map.items())):
        # deduce original lines by checking lines_coeffs whose segment index is in seg_idxs
        angles = []
        line_ids = set()
        for seg_idx in seg_idxs:
            # attempt to map segment -> original line coeff (we assume 1-to-1 mapping order)
            # In our build, segments correspond to lines in order, but if you split multi segments
            # this may vary. For simplicity assume segments index maps to lines_coeffs index.
            # If your input lines map differently, maintain a mapping.
            if seg_idx < len(lines_coeffs):
                s = lines_coeffs[seg_idx]
                angles.append(angle_from_line_coeff(*s))
                line_ids.add(seg_idx)
            else:
                # fallback: compute slope from segment geometry
                sgeom = segments[seg_idx]
                angles.append(angle_from_two_points(sgeom.coords[0], sgeom.coords[-1]))
                line_ids.add(seg_idx)
        info.append({
            'id': i+1,
            'coord': coord,
            'num_lines': len(line_ids),
            'line_segment_indices': sorted(list(line_ids)),
            'angles': angles
        })
    return info

def build_polygon_list(segments):
    """
    Use shapely.ops.polygonize to get polygons (faces) from the set of segments.
    Return list of shapely Polygons, sorted by centroid position in polar coordinates.
    """
    merged = unary_union(segments)
    polys = list(polygonize(merged))

    # 依照 r^2 與 arctan2(y,x) 排序
    def polar_key(p):
        cx, cy = p.centroid.x, p.centroid.y
        r2 = round(cx*cx + cy*cy, 8)
        theta = round(arctan2(cy, cx), 8)
        return (r2, theta)

    polys_sorted = sorted(polys, key=polar_key)
    return polys_sorted

def polygon_info(polys):
    """
    For each polygon: id, area, centroid, number of edges, vertices coords
    """
    out = []
    for i, p in enumerate(polys):
        coords = list(p.exterior.coords)[:-1]  # omit repeated last
        out.append({
            'id': i+1,
            'area': p.area,
            'centroid': (p.centroid.x, p.centroid.y),
            'num_edges': len(coords),
            'vertices': coords,
            'polygon': p
        })
    return out

def polygon_adjacency(polygons_info):
    """
    Compute adjacency: for each polygon find neighbors by shared boundary (touches & intersection with dimension 1)
    For each shared edge, compute slope.
    """
    n = len(polygons_info)
    neighbors = {p['id']: [] for p in polygons_info}
    for i in range(n):
        pid = polygons_info[i]['id']
        pgeom = polygons_info[i]['polygon']
        for j in range(i+1, n):
            qid = polygons_info[j]['id']
            qgeom = polygons_info[j]['polygon']
            if pgeom.touches(qgeom):
                # shared boundary:
                shared = pgeom.boundary.intersection(qgeom.boundary)
                # shared could be LineString or MultiLineString; extract representative segment(s)
                angles = []
                if shared.is_empty:
                    continue
                if shared.geom_type == 'LineString':
                    coords = list(shared.coords)
                    angles.append(angle_from_two_points(coords[0], coords[-1]))
                else:
                    # MultiLineString or collection
                    for g in (shared.geoms if hasattr(shared, 'geoms') else [shared]):
                        if g.geom_type == 'LineString':
                            c = list(g.coords)
                            angles.append(angle_from_two_points(c[0], c[-1]))
                neighbors[pid].append({
                    'neighbor_id': qid,
                    'shared_angles': angles
                })
                neighbors[qid].append({
                    'neighbor_id': pid,
                    'shared_angles': angles
                })
    return neighbors

def line_construcer(Cn,num_of_lines,offsets,spacing):
    #input check: Cn in Z+ ,and >= 3
    if Cn < 3 or not isinstance(Cn,int):
        raise ValueError('Cn must be an integer >= 3')
    
    lines = []
    for rot in range(Cn):
        angle = rot * 2 * pi / Cn
        if Cn%2 == 0 and angle >= pi-EPS:
            break
        c = -spacing*(num_of_lines-1)/2
        while c <= spacing*(num_of_lines-1)/2:
            lines.append((sin(angle),cos(angle),c-offsets))
            c += spacing
    return lines

def inside_check(Cn,num_of_lines,offsets,spacing,points):
    """
    Note: this function is hughly dependent on the 'line_construcer' function
    ------
    """
    #(0,0) is alwways inside
    if Cn in [2,3]:
        cl = -spacing*(num_of_lines-1)/2 
        cr = spacing*(num_of_lines-1)/2 
        for rot in range(Cn):
            angle = rot * pi / Cn
            sl = points[0] * sin(angle) + points[1] * cos(angle) + cl
            sr = points[0] * sin(angle) + points[1] * cos(angle) + cr
            if sl * cl < 0 or sr * cr < 0:
                return False
    else:
        c = -spacing*(num_of_lines-1)/2 - offsets
        for rot in range(Cn):
            angle = rot * 2 * pi / Cn
            s = points[0] * sin(angle) + points[1] * cos(angle) + c
            if s * c < 0:
                return False
    return True

def reassign_ids(data, start=1, key="id"):
    for i, item in enumerate(data, start=start):
        item[key] = i
    return data

def remove_leaf_nodes(points: np.ndarray, edges: np.ndarray, recursive: bool = False):
    """
    移除deg=1的點，並更新對應的 edges 編號。
    
    Parameters
    ----------
    points : (n,2) float ndarray
        每一列為 (x, y) 點座標，點的 id 為其列索引。
    edges : (m,2) int ndarray
        每一列為 (pid1, pid2)，代表一條邊。
    recursive : bool, default=False
        - False: 只移除一次度=1的點
        - True : 重複移除直到沒有度=1的點
    
    Returns
    -------
    filtered_points : ndarray
    filtered_edges : ndarray
    """
    # ---- 輸入檢查 ----
    if not isinstance(points, np.ndarray) or not isinstance(edges, np.ndarray):
        raise TypeError("points 和 edges 都必須是 numpy ndarray")
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points 必須是 (n,2) 的 float ndarray")
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("edges 必須是 (m,2) 的 int ndarray")
    if len(points) == 0 or len(edges) == 0:
        return points.copy(), edges.copy()

    if np.any(edges < 0) or np.any(edges >= len(points)):
        raise ValueError("edges 含有無效的點 index")

    def _remove_once(points, edges):
        flat = edges.flatten()
        unique, counts = np.unique(flat, return_counts=True)
        degree = dict(zip(unique, counts))

        # 找出度=1的點
        to_remove = {pid for pid, deg in degree.items() if deg == 1}
        if not to_remove:
            return points, edges, False

        # Step1: 保留點
        mask = np.ones(len(points), dtype=bool)
        mask[list(to_remove)] = False
        points = points[mask]

        # Step2: 舊id→新id 對照表
        old_to_new = {old: new for new, old in enumerate(np.where(mask)[0])}

        # Step3: 過濾並更新 edges
        new_edges = []
        for a, b in edges:
            if a in old_to_new and b in old_to_new:
                new_edges.append([old_to_new[a], old_to_new[b]])
        edges = np.array(new_edges, dtype=int)

        return points, edges, True

    # ---- 主迴圈 ----
    changed = True
    while changed:
        points, edges, changed = _remove_once(points, edges)
        if not recursive:  # 如果只刪一次，就結束
            break
        if edges.size == 0:
            break

    return points, edges

def is_bipartite_with_cycle(adj_matrix):
    n = len(adj_matrix)
    color = [-1] * n      # -1: 未染色, 0/1: 兩種顏色
    parent = [-1] * n     # 父節點
    cycle = None          # 用來存奇環

    for start in range(n):
        if color[start] != -1:
            continue  # 已處理過的連通分量
        stack = [(start, 0)]
        color[start] = 0

        while stack:
            node, c = stack.pop()
            for nei in range(n):
                if adj_matrix[node][nei] != 0:  # 有邊
                    if color[nei] == -1:
                        color[nei] = 1 - c
                        parent[nei] = node
                        stack.append((nei, color[nei]))
                    elif color[nei] == c:
                        # 找到奇環
                        x, y = node, nei
                        path_x, path_y = [], []
                        while x != -1:
                            path_x.append(x)
                            x = parent[x]
                        while y != -1:
                            path_y.append(y)
                            y = parent[y]
                        # 找最近公共祖先 LCA
                        set_y = set(path_y)
                        lca = None
                        for v in path_x:
                            if v in set_y:
                                lca = v
                                break
                        # 拼出 cycle
                        cycle = []
                        x = node
                        while x != lca:
                            cycle.append(x)
                            x = parent[x]
                        cycle.append(lca)
                        tmp = []
                        y = nei
                        while y != lca:
                            tmp.append(y)
                            y = parent[y]
                        cycle += tmp[::-1]
                        cycle.append(cycle[0])
                        return False, cycle
    return True, color

def select_file_gui(initialdir):
    import tkinter , tkinter.filedialog
    root = tkinter.Tk()
    root.withdraw()
    filename = tkinter.filedialog.askopenfilename(initialdir=initialdir)
    root.destroy()
    return filename

def generate_path_points(waypoints,nk_per_segment):
    """
    waypoints: 
        an array of k points with shape (n,3)
    nk_per_segment: 
        number of k points per segment,
    total k points = nk_per_segment* n_waypoints
    
    """
    waypoints = np.array(waypoints)
    #check input
    if waypoints.shape[1] not in [2,3] or waypoints.shape[0] < 2:
        raise ValueError("waypoints must be a matrix with shape (n,3) and n >= 2")
    print("waypoints type: ",f'{waypoints.shape[1]}D')
    kpoints = np.linspace(waypoints[0],waypoints[1],nk_per_segment)
    if waypoints.shape[0] >= 3:
        for sags in range(waypoints.shape[0])[2:]:
            sag_tmp = np.linspace(waypoints[sags-1],waypoints[sags],nk_per_segment)
            kpoints = np.vstack((kpoints,sag_tmp))
    return kpoints

def DiophantusSearch(r, u, n_search_range=10000, batch_size=10_000_000):
    """
    使用 PyTorch (GPU) 加速窮舉搜索，分批次處理。
    求解: rn - m = u
    """
    
    # 1. 設置設備 (GPU or CPU)
    if torch.cuda.is_available():
        device = torch.device('cuda')
        # print(f"--- PyTorch GPU 搜索啟動 (設備: {torch.cuda.get_device_name(0)}) ---")
    else:
        device = torch.device('cpu')
        # print("--- PyTorch CPU 搜索啟動 (未檢測到 CUDA) ---")

    # print(f"搜索範圍 n: [{-n_search_range}, {n_search_range}]")
    # print(f"批次大小: {batch_size:,}")

    # 2. 將標量 r 和 u 傳輸到 GPU
    #    為保證精度，必須使用 64-bit 浮點數 (torch.double)
    r_gpu = torch.tensor(r, dtype=torch.float64, device=device)
    u_gpu = torch.tensor(u, dtype=torch.float64, device=device)
    
    # 3. 初始化全局最小值 (儲存在 GPU 上)
    min_error_global = torch.tensor(torch.inf, dtype=torch.float64, device=device)
    # n 和 m 需要是 64-bit 整數 (torch.long)
    best_n_global = torch.tensor(0, dtype=torch.long, device=device)
    best_m_global = torch.tensor(0, dtype=torch.long, device=device)

    # 總共要處理的 n 的數量
    total_n_count = 2 * n_search_range + 1
    # 向上取整計算批次數
    num_batches = (total_n_count + batch_size - 1) // batch_size
    
    start_time = time.time()

    for i in range(num_batches):
        # 4. 計算當前批次的 n 範圍
        n_start = -n_search_range + i * batch_size
        n_end = min([-n_search_range + (i + 1) * batch_size, n_search_range + 1])
        
        if n_start >= n_end:
            continue

        # 5. 在 GPU 上創建 n 值的批次
        n_batch_gpu = torch.arange(n_start, n_end, dtype=torch.long, device=device)

        # 6. === GPU 核心計算 (並行執行) ===
        #    m_ideal = r*n - u
        m_ideal_gpu = r_gpu * n_batch_gpu - u_gpu
        
        #    m_int = round(m_ideal)
        m_int_gpu = torch.round(m_ideal_gpu)
        
        #    error = |m_ideal - m_int| = |r*n - m - u|
        errors_gpu = torch.abs(m_ideal_gpu - m_int_gpu)
        # ------------------------------------

        # 7. 在 GPU 上執行批次內的「歸約」(Reduction)
        #    torch.min 會同時返回最小值和其索引
        min_error_batch, min_error_idx_batch = torch.min(errors_gpu, dim=0)

        # 8. 與全局最小值比較 (這是一個 GPU 上的 if 判斷)
        if min_error_batch < min_error_global:
            # 如果批次最小 < 全局最小，則更新全局最小
            min_error_global = min_error_batch
            best_n_global = n_batch_gpu[min_error_idx_batch]
            best_m_global = m_int_gpu[min_error_idx_batch]
            
        # 釋放 VRAM
        del n_batch_gpu, m_ideal_gpu, m_int_gpu, errors_gpu, min_error_batch, min_error_idx_batch

    # 9. 同步以確保所有 GPU 操作完成，以便準確計時
    if device.type == 'cuda':
        torch.cuda.synchronize()
        
    end_time = time.time()
    # print(f"PyTorch 搜索完成。總耗時: {end_time - start_time:.4f} 秒")

    # 10. 將最終結果從 GPU 傳回 CPU
    #     .item() 方法將 GPU 上的單元素張量轉換為 Python 標量
    return best_n_global.item(), best_m_global.item(), min_error_global.item()    
# -----------------------
# Example usage
# -----------------------

if __name__ == "__main__":
    # 定義一些線 (ax + by + c = 0)
    # 例如： x = 0  => 1*x + 0*y + 0 = 0  -> (1,0,0)
    lines = line_construcer(2,4,0,1)
    # bounding box (xmin,ymin,xmax,ymax)
    bbox = ( -10, -10, 10, 10 )

    # 給定線集合 與 bounding box, 透過線集合相交建立線段 
    segments = build_segments(lines, bbox)
    
    # 計算線段相交點
    pts_map, coords_list = compute_intersection_points(segments)
    points_info = get_point_info(pts_map, segments, lines)

    polys = build_polygon_list(segments)
    polys_info = polygon_info(polys)
    neighbors = polygon_adjacency(polys_info)

    '''
    Data Structure:
    
    points_info = [ 'id', 'coord', 'num_lines', 'angles' ]
    polys_info = [ 'id', 'area', 'centroid', 'num_edges', 'vertices' ]
    '''
    # 印出結果（簡短）
    print("=== Intersection points ===")
    for p in points_info:
        print(f"ID {p['id']:2d} coord={p['coord']}  num_lines={p['num_lines']} angles={p['angles']}")

    print("\n=== Polygons ===")
    for p in polys_info:
        pid = p['id']
        print(f"Poly {pid:2d}: area={p['area']:.4f}, edges={p['num_edges']}, centroid={p['centroid']}")
        # neighbors
        print("  Neighbors:")
        for neigh in neighbors.get(pid, []):
            print(f"    -> poly {neigh['neighbor_id']}, shared edge angles={neigh['shared_angles']}")
