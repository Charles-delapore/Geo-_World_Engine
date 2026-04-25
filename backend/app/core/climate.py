"""
气候模拟模块 - 计算温度、降水和生物群系
"""
from typing import Dict

import numpy as np


def calculate_temperature(
    elev: np.ndarray,
    latitude: np.ndarray,
    lapse_rate: float = 0.0065,
    temperature_bias: float = 0.0,
) -> np.ndarray:
    lat_rad = np.abs(latitude) * np.pi / 180.0
    base_temp = 30.0 * np.cos(lat_rad)
    temp = base_temp - lapse_rate * elev + temperature_bias
    return temp


def calculate_precipitation(
    elev: np.ndarray,
    wind_direction: str = 'westerly',
    moisture_factor: float = 1.0
) -> np.ndarray:
    precip = np.full(elev.shape, 1000.0, dtype=np.float32)

    grad_y = np.zeros_like(elev)
    grad_x = np.zeros_like(elev)
    grad_y[1:-1, :] = elev[2:, :] - elev[:-2, :]
    grad_x[:, 1:-1] = elev[:, 2:] - elev[:, :-2]

    if wind_direction == 'westerly':
        precip = np.where(grad_x > 0, precip * 1.5, np.where(grad_x < 0, precip * 0.7, precip))
    elif wind_direction == 'easterly':
        precip = np.where(grad_x < 0, precip * 1.45, np.where(grad_x > 0, precip * 0.72, precip))
    elif wind_direction == 'northerly':
        precip = np.where(grad_y > 0, precip * 1.35, np.where(grad_y < 0, precip * 0.76, precip))
    elif wind_direction == 'southerly':
        precip = np.where(grad_y < 0, precip * 1.35, np.where(grad_y > 0, precip * 0.76, precip))

    precip *= moisture_factor
    return precip


def classify_biome(temp: np.ndarray, precip: np.ndarray) -> np.ndarray:
    biome = np.zeros(temp.shape, dtype=np.int32)
    biome = np.where(temp < -10, 5, biome)
    biome = np.where((temp >= -10) & (temp < 0), 4, biome)
    biome = np.where((temp >= 0) & (precip < 250), 1, biome)
    biome = np.where((temp >= 0) & (temp >= 0) & (precip >= 250) & (precip < 500), 2, biome)
    biome = np.where((temp >= 0) & (precip >= 500), 3, biome)
    return biome


class ClimateSimulator:

    def __init__(self, elev: np.ndarray, lat_grid: np.ndarray):
        self.elev = elev
        self.lat_grid = lat_grid

    def run(
        self,
        wind_direction: str = 'westerly',
        moisture_factor: float = 1.0,
        temperature_bias: float = 0.0,
    ) -> Dict[str, np.ndarray]:
        temp = calculate_temperature(self.elev, self.lat_grid, temperature_bias=temperature_bias)
        precip = calculate_precipitation(self.elev, wind_direction, moisture_factor)
        biome = classify_biome(temp, precip)
        return {
            'temperature': temp,
            'precipitation': precip,
            'biome': biome
        }
