"""
气候模拟模块 - 计算温度、降水和生物群系
"""
from typing import Dict

import numpy as np

try:
    from numba import jit
except ImportError:  # pragma: no cover - optional acceleration
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


@jit(nopython=True)
def calculate_temperature(elev: np.ndarray, latitude: np.ndarray, lapse_rate: float = 0.0065) -> np.ndarray:
    """
    根据海拔和纬度计算温度（摄氏度）
    
    Args:
        elev: 高程数组（米）
        latitude: 纬度数组（度，-90到90）
        lapse_rate: 温度递减率（°C/m）
        
    Returns:
        温度数组
    """
    # 基础温度：赤道约30°C，两极度低
    # 使用余弦函数更平滑地模拟纬度影响
    lat_rad = np.abs(latitude) * np.pi / 180.0  # 转换为弧度
    base_temp = 30.0 * np.cos(lat_rad)
    
    # 海拔递减（每升高1000米降低6.5°C）
    temp = base_temp - lapse_rate * elev
    
    return temp


@jit(nopython=True)
def calculate_precipitation(
    elev: np.ndarray, 
    wind_direction: str = 'westerly',
    moisture_factor: float = 1.0
) -> np.ndarray:
    """
    基于地形和风向计算降水（简化版）
    
    Args:
        elev: 高程数组
        wind_direction: 主导风向 ('westerly', 'easterly', 'northerly', 'southerly')
        moisture_factor: 湿度因子
        
    Returns:
        降水量数组（mm/年）
    """
    height, width = elev.shape
    precip = np.ones((height, width), dtype=np.float32) * 1000.0  # 基础降水1000mm
    
    # 计算坡度（用于地形雨）
    grad_y = np.zeros_like(elev)
    grad_x = np.zeros_like(elev)
    
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            grad_y[y, x] = elev[y + 1, x] - elev[y - 1, x]
            grad_x[y, x] = elev[y, x + 1] - elev[y, x - 1]
    
    # 根据风向调整降水
    if wind_direction == 'westerly':
        # 西风带：西侧迎风坡多雨
        for y in range(height):
            for x in range(width):
                if grad_x[y, x] > 0:  # 西坡
                    precip[y, x] *= 1.5
                elif grad_x[y, x] < 0:  # 东坡（雨影）
                    precip[y, x] *= 0.7
    
    # 应用湿度因子
    precip *= moisture_factor
    
    return precip


@jit(nopython=True)
def classify_biome(temp: np.ndarray, precip: np.ndarray) -> np.ndarray:
    """
    基于温度和降水分类生物群系
    
    Args:
        temp: 温度数组（°C）
        precip: 降水量数组（mm/年）
        
    Returns:
        生物群系ID数组
        0: 海洋, 1: 沙漠, 2: 草原, 3: 森林, 4: 苔原, 5: 冰原
    """
    height, width = temp.shape
    biome = np.zeros((height, width), dtype=np.int32)
    
    for y in range(height):
        for x in range(width):
            t = temp[y, x]
            p = precip[y, x]
            
            if t < -10:
                biome[y, x] = 5  # 冰原
            elif t < 0:
                biome[y, x] = 4  # 苔原
            elif p < 250:
                biome[y, x] = 1  # 沙漠
            elif p < 500:
                biome[y, x] = 2  # 草原
            else:
                biome[y, x] = 3  # 森林
    
    return biome


class ClimateSimulator:
    """气候模拟器"""
    
    def __init__(self, elev: np.ndarray, lat_grid: np.ndarray):
        """
        初始化气候模拟器
        
        Args:
            elev: 高程数组
            lat_grid: 纬度网格数组
        """
        self.elev = elev
        self.lat_grid = lat_grid
    
    def run(self, wind_direction: str = 'westerly') -> Dict[str, np.ndarray]:
        """
        运行气候模拟
        
        Args:
            wind_direction: 主导风向
            
        Returns:
            包含温度、降水、生物群系的字典
        """
        # 计算温度
        temp = calculate_temperature(self.elev, self.lat_grid)
        
        # 计算降水
        precip = calculate_precipitation(self.elev, wind_direction)
        
        # 分类生物群系
        biome = classify_biome(temp, precip)
        
        return {
            'temperature': temp,
            'precipitation': precip,
            'biome': biome
        }
