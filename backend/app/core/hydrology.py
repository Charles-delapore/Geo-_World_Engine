"""
水文处理模块 - 使用pyflwdir进行洼地填充、流向计算和流量累积
"""
import numpy as np
from typing import Tuple


def fill_depressions(elev: np.ndarray) -> np.ndarray:
    """
    使用pyflwdir进行洼地填充
    
    Args:
        elev: 高程数组
        
    Returns:
        填充后的高程数组
    """
    try:
        import pyflwdir
        
        # 确保无NaN，设置no_data值
        elev_clean = np.where(np.isnan(elev), -9999, elev).astype(np.float64)
        
        # 使用pyflwdir进行洼地填充
        # from_dem会自动处理并返回FlwdirRaster对象
        flw = pyflwdir.from_dem(
            elev_clean,
            nodata=-9999,
            logger=None
        )
        
        # 获取填充后的DEM
        filled = flw.fill_depressions()
        
        return filled.astype(np.float32)
    
    except ImportError:
        print("Warning: pyflwdir not available, using simple fill")
        return _simple_fill_depressions(elev)
    except Exception as e:
        print(f"Warning: pyflwdir failed ({e}), using simple fill")
        return _simple_fill_depressions(elev)


def _simple_fill_depressions(elev: np.ndarray) -> np.ndarray:
    """简化的洼地填充（备选方案）"""
    # 简单实现：将低于周围平均值的点提升到平均值
    from scipy.ndimage import uniform_filter
    
    filled = elev.copy()
    for _ in range(5):  # 迭代几次
        neighborhood_avg = uniform_filter(filled, size=3)
        mask = filled < neighborhood_avg
        filled[mask] = neighborhood_avg[mask]
    
    return filled


def compute_flow_direction(elev: np.ndarray) -> np.ndarray:
    """
    计算D8流向
    
    Args:
        elev: 已填充洼地的高程数组
        
    Returns:
        流向数组 (0-7表示方向，255表示无流向/汇点)
    """
    try:
        import pyflwdir
        
        elev_clean = np.where(np.isnan(elev), -9999, elev).astype(np.float64)
        
        # 使用pyflwdir计算流向
        flw = pyflwdir.from_dem(elev_clean, nodata=-9999)
        
        # 获取D8流向数组(位编码: 1,2,4,8,16,32,64,128)
        dirs_bit = flw.to_array()
        
        # 将位编码转换为方向索引 (0-7)
        # pyflwdir的D8位编码: 1=E, 2=SE, 4=S, 8=SW, 16=W, 32=NW, 64=N, 128=NE
        bit_to_dir = {
            1: 0,    # E
            2: 1,    # SE  
            4: 2,    # S
            8: 3,    # SW
            16: 4,   # W
            32: 5,   # NW
            64: 6,   # N
            128: 7,  # NE
        }
        
        # 创建方向索引数组，默认为 255（无数据）
        dirs = np.full_like(dirs_bit, 255, dtype=np.uint8)
        
        # 转换每个位编码为方向索引
        for bit_val, dir_idx in bit_to_dir.items():
            dirs[dirs_bit == bit_val] = dir_idx
        
        return dirs
    
    except ImportError:
        print("Warning: pyflwdir not available, using simple flow direction")
        return _simple_flow_direction(elev)
    except Exception as e:
        print(f"Warning: pyflwdir flow direction failed ({e}), using simple method")
        return _simple_flow_direction(elev)


def _simple_flow_direction(elev: np.ndarray) -> np.ndarray:
    """简化的D8流向计算（备选方案）"""
    height, width = elev.shape
    flow_dir = np.full((height, width), 8, dtype=np.uint8)
    
    # D8方向偏移
    dy = [-1, -1, -1, 0, 0, 1, 1, 1]
    dx = [-1, 0, 1, -1, 1, -1, 0, 1]
    dist = [1.414, 1.0, 1.414, 1.0, 1.0, 1.414, 1.0, 1.414]
    
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            max_slope = 0.0
            best_dir = 8
            
            for k in range(8):
                ny, nx = y + dy[k], x + dx[k]
                delta = elev[y, x] - elev[ny, nx]
                
                if delta > 0:
                    slope = delta / dist[k]
                    if slope > max_slope:
                        max_slope = slope
                        best_dir = k
            
            flow_dir[y, x] = best_dir
    
    return flow_dir


def accumulate_flow(flow_dir: np.ndarray, rain: np.ndarray) -> np.ndarray:
    """
    基于拓扑排序的流量累积
    
    Args:
        flow_dir: 流向数组 (0-7表示方向，255表示无数据)
        rain: 降雨量数组
        
    Returns:
        流量累积数组
    """
    try:
        import pyflwdir
        
        # 将方向索引 (0-7) 转换回 D8 位编码
        dir_to_bit = np.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=np.uint8)
        
        # 创建位编码数组，无流向的点(值为255)设为 0
        dirs_bit = np.zeros_like(flow_dir, dtype=np.uint8)
        valid_mask = flow_dir < 255  # 有效流向的掩码
        dirs_bit[valid_mask] = dir_to_bit[flow_dir[valid_mask]]
        
        # 使用 from_array 创建 FlwdirRaster 对象
        flw = pyflwdir.from_array(dirs_bit, ftype='d8')
        
        # 计算上游面积（像元数）
        acc = flw.upstream_area()
        
        # 流量 = 上游面积 * 单位雨量
        flux = acc.astype(np.float32) * rain
        
        return flux
    
    except ImportError:
        print("Warning: pyflwdir not available, using simple accumulation")
        return _simple_accumulate_flow(flow_dir, rain)
    except Exception as e:
        print(f"Warning: pyflwdir accumulation failed ({e}), using simple method")
        return _simple_accumulate_flow(flow_dir, rain)


def _simple_accumulate_flow(flow_dir: np.ndarray, rain: np.ndarray) -> np.ndarray:
    """简化的流量累积（备选方案）"""
    height, width = flow_dir.shape
    flux = rain.copy().astype(np.float64)
    
    # D8方向偏移
    dy = [-1, -1, -1, 0, 0, 1, 1, 1]
    dx = [-1, 0, 1, -1, 1, -1, 0, 1]
    
    # 简单迭代累积（性能较差，仅作备选）
    for _ in range(max(height, width)):
        updated = False
        for y in range(height):
            for x in range(width):
                d = flow_dir[y, x]
                if d < 255:  # 修改：使用255作为无效值
                    ny = y + dy[d]
                    nx = x + dx[d]
                    if 0 <= ny < height and 0 <= nx < width:
                        flux[ny, nx] += flux[y, x]
                        updated = True
        
        if not updated:
            break
    
    return flux.astype(np.float32)


def hydraulic_erosion(elev: np.ndarray, rain: np.ndarray, iterations: int = 4, K: float = 0.1) -> np.ndarray:
    """
    简化的水力侵蚀模型
    
    Args:
        elev: 高程数组
        rain: 降雨量数组
        iterations: 迭代次数（建议≤4）
        K: 侵蚀系数
        
    Returns:
        侵蚀后的高程数组
    """
    elev_out = elev.copy()
    
    for i in range(iterations):
        # 填充洼地
        filled = fill_depressions(elev_out)
        
        # 计算流向
        flow_dir = compute_flow_direction(filled)
        
        # 累积流量
        flux = accumulate_flow(flow_dir, rain)
        
        # 侵蚀量与流量平方根成正比
        erosion = K * np.sqrt(flux)
        
        # 限制最大侵蚀深度
        erosion = np.clip(erosion, 0, 5.0)
        
        # 更新高程
        elev_out = elev_out - erosion
    
    return elev_out.astype(np.float32)
