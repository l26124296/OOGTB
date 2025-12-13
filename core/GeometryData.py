import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Union

@dataclass
class SimplicialComplex:
    """
    幾何數據容器 (Passive Data Object)
    只負責儲存，不負責生成邏輯。
    """
    # --- 0-Simplex ---
    positions: np.ndarray  # (N, 2) float
    
    # --- 1-Simplex ---
    edges: np.ndarray      # (M, 2) int
    edge_types: np.ndarray # (M,) int (邊的分類 ID)
    
    # --- Metadata ---
    # 物理意義映射: {0: 'NN', 1: 'NNN', 2: 'Inter-layer'}
    edge_type_map: Dict[int, str] = field(default_factory=dict)
    
    # --- 2-Simplex (預留) ---
    faces: List[List[int]] = field(default_factory=list)

    def save(self, filename: str):
        """儲存為壓縮格式"""
        np.savez_compressed(filename, 
                            positions=self.positions, 
                            edges=self.edges, 
                            edge_types=self.edge_types,
                            edge_type_map=np.array(self.edge_type_map, dtype=object),
                            faces=np.array(self.faces, dtype=object))
        print(f"[Data] Saved {len(self.positions)} sites, {len(self.edges)} edges to {filename}")

    @classmethod
    def load(cls, filename: str) :
        """
        從 .npz 檔案讀取並重建 SimplicialComplex 物件
        (存的東西不是SimplicialComplex)
        """
        try:
            # allow_pickle=True 是必須的，因為我們存了 dict 和 list
            with np.load(filename, allow_pickle=True) as data:
                
                # 1. 讀取基礎陣列 (直接讀取)
                positions = data['positions']
                edges = data['edges']
                edge_types = data['edge_types']
                
                # 2. 還原 edge_type_map (從 0-d object array 提取字典)
                # 當我們用 np.save 保存 dict 時，它被包裝成了一個 scalar object
                # 使用 .item() 將其取出
                raw_map = data['edge_type_map']
                if raw_map.shape == ():
                    edge_type_map = raw_map.item()
                else:
                    # 相容性處理：如果存的時候不是 0-d array
                    edge_type_map = raw_map.tolist()
                
                # 3. 還原 faces (從 object array 轉回 list)
                raw_faces = data['faces']
                faces = raw_faces.tolist() if isinstance(raw_faces, np.ndarray) else list(raw_faces)

                # 4. 回傳新的實例
                return cls(
                    positions=positions,
                    edges=edges,
                    edge_types=edge_types,
                    edge_type_map=edge_type_map,
                    faces=faces
                )
                
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {filename}")
        except Exception as e:
            raise RuntimeError(f"Failed to load SimplicialComplex from {filename}: {str(e)}")