from typing import Dict, Optional

import numpy as np
from scipy.ndimage import gaussian_filter


BIOME_NAMES = [
    "Marine",
    "Hot desert",
    "Cold desert",
    "Savanna",
    "Grassland",
    "Tropical seasonal forest",
    "Temperate deciduous forest",
    "Tropical rainforest",
    "Temperate rainforest",
    "Taiga",
    "Tundra",
    "Glacier",
    "Wetland",
]

BIOME_COLORS = [
    "#466eab",
    "#fbe79f",
    "#b5b887",
    "#d2d082",
    "#c8d68f",
    "#b6d95d",
    "#29bc56",
    "#7dcb35",
    "#409c43",
    "#4b6b32",
    "#96784b",
    "#d5e7eb",
    "#0b9131",
]

BIOME_HABITABILITY = [0, 4, 10, 22, 30, 50, 100, 80, 90, 12, 4, 0, 12]

BIOME_COST = [10, 200, 150, 60, 50, 70, 70, 80, 90, 200, 1000, 5000, 150]

_BIOMES_MATRIX = np.array(
    [
        [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 10],
        [3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 9, 9, 9, 9, 10, 10, 10],
        [5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 9, 9, 9, 9, 9, 10, 10, 10],
        [5, 6, 6, 6, 6, 6, 6, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10],
        [7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 10, 10],
    ],
    dtype=np.int32,
)


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


def _is_wetland(moisture: np.ndarray, temperature: np.ndarray, height: np.ndarray) -> np.ndarray:
    result = np.zeros_like(temperature, dtype=bool)
    result &= temperature > -2
    near_coast = (moisture > 40) & (height < 25)
    off_coast = (moisture > 24) & (height >= 25) & (height < 60)
    result &= near_coast | off_coast
    return result


def classify_biome(
    temp: np.ndarray,
    precip: np.ndarray,
    elevation: np.ndarray | None = None,
    river_flux: np.ndarray | None = None,
) -> np.ndarray:
    biome = np.zeros(temp.shape, dtype=np.int32)

    if elevation is None:
        elevation = np.zeros_like(temp)

    height = np.clip(elevation * 100, 0, 100)

    moisture = precip.copy()
    if river_flux is not None:
        moisture = moisture + np.maximum(river_flux / 10, 2)

    biome = np.where(height < 20, 0, biome)
    biome = np.where((height >= 20) & (temp < -5), 11, biome)

    hot_dry = (temp >= 25) & (moisture < 8)
    if river_flux is not None:
        hot_dry = hot_dry & (river_flux <= 0)
    biome = np.where((height >= 20) & (temp >= -5) & hot_dry, 1, biome)

    wetland = _is_wetland(moisture, temp, height) & (height >= 20) & (temp >= -5) & ~hot_dry
    biome = np.where(wetland, 12, biome)

    land_unclassified = (height >= 20) & (biome == 0)
    if np.any(land_unclassified):
        moisture_band = np.clip((moisture / 5).astype(np.int32), 0, 4)
        temperature_band = np.clip(20 - temp.astype(np.int32), 0, 25)

        matrix_biome = _BIOMES_MATRIX[moisture_band, temperature_band]
        biome = np.where(land_unclassified, matrix_biome, biome)

    return biome


def classify_biome_simple(temp: np.ndarray, precip: np.ndarray) -> np.ndarray:
    biome = np.zeros(temp.shape, dtype=np.int32)
    biome = np.where(temp < -10, 5, biome)
    biome = np.where((temp >= -10) & (temp < 0), 4, biome)
    biome = np.where((temp >= 0) & (precip < 250), 1, biome)
    biome = np.where((temp >= 0) & (precip >= 250) & (precip < 500), 2, biome)
    biome = np.where((temp >= 0) & (precip >= 500), 3, biome)
    return biome


def calculate_moisture(
    precip: np.ndarray,
    elev: np.ndarray,
    river_flux: Optional[np.ndarray] = None,
    sea_level: float = 0.2,
) -> np.ndarray:
    is_land = elev > sea_level
    base_moisture = precip.astype(np.float64).copy()

    if river_flux is not None:
        flux_norm = river_flux.astype(np.float64)
        if flux_norm.max() > 0:
            flux_norm = flux_norm / flux_norm.max() * 100.0
        river_contribution = np.maximum(flux_norm / 10.0, 2.0)
        has_river = (river_flux > 0) & is_land
        base_moisture[has_river] += river_contribution[has_river]

    neighbor_sum = np.zeros_like(base_moisture)
    neighbor_count = np.zeros_like(base_moisture)
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dy == 0 and dx == 0:
                continue
            shifted_m = np.roll(np.roll(base_moisture, -dy, axis=0), -dx, axis=1)
            shifted_l = np.roll(np.roll(is_land.astype(np.float64), -dy, axis=0), -dx, axis=1)
            neighbor_sum += shifted_m * shifted_l
            neighbor_count += shifted_l
    neighbor_mean = np.where(neighbor_count > 0, neighbor_sum / np.maximum(neighbor_count, 1), 0.0)
    all_moisture = np.where(is_land, (base_moisture + neighbor_mean) / 2.0, 0.0)
    moisture = 4.0 + all_moisture
    moisture[~is_land] = 0.0
    return moisture.astype(np.float32)


def compute_resources(
    biome: np.ndarray,
    elev: np.ndarray,
    precip: np.ndarray,
    temp: np.ndarray,
    sea_level: float = 0.2,
) -> Dict[str, np.ndarray]:
    h, w = biome.shape
    is_land = elev > sea_level

    habitability_map = np.zeros((h, w), dtype=np.float32)
    for bid, hab in enumerate(BIOME_HABITABILITY):
        habitability_map[biome == bid] = hab / 100.0

    agriculture = np.zeros((h, w), dtype=np.float32)
    agriculture[biome == 3] = 0.4
    agriculture[biome == 4] = 0.6
    agriculture[biome == 5] = 0.5
    agriculture[biome == 6] = 0.9
    agriculture[biome == 7] = 0.3
    agriculture[biome == 8] = 0.7
    agriculture[biome == 12] = 0.6
    agriculture[~is_land] = 0.0

    rng = np.random.RandomState(42)
    mineral_noise = rng.rand(h, w).astype(np.float32)
    mineral_noise = gaussian_filter(mineral_noise, sigma=5.0)
    mineral_noise = (mineral_noise - mineral_noise.min()) / (mineral_noise.max() - mineral_noise.min() + 1e-10)
    minerals = np.zeros((h, w), dtype=np.float32)
    minerals[is_land] = 0.2 + 0.6 * mineral_noise[is_land]
    minerals[(biome == 9) | (biome == 10)] = np.maximum(minerals[(biome == 9) | (biome == 10)], 0.7)
    minerals[biome == 1] = np.maximum(minerals[biome == 1], 0.5)
    minerals[~is_land] = 0.0

    coast_dist = np.abs(elev - sea_level)
    coast_band = np.exp(-(coast_dist ** 2) / 0.01)
    fishery = coast_band * 0.8
    fishery[biome == 0] = 0.6 + 0.3 * mineral_noise[biome == 0]

    timber = np.zeros((h, w), dtype=np.float32)
    timber[biome == 5] = 0.7
    timber[biome == 6] = 0.8
    timber[biome == 7] = 0.9
    timber[biome == 8] = 0.85
    timber[biome == 9] = 0.5
    timber[biome == 12] = 0.6
    timber[~is_land] = 0.0

    water = np.zeros((h, w), dtype=np.float32)
    water[is_land] = np.clip(precip[is_land] / 2000.0, 0.1, 1.0)
    water[biome == 0] = 1.0
    water[biome == 12] = 1.0

    for res in [agriculture, minerals, fishery, timber, water]:
        res[:] = gaussian_filter(res, sigma=1.5)

    return {
        "agriculture": agriculture,
        "minerals": minerals,
        "fishery": fishery,
        "timber": timber,
        "water": water,
        "habitability": habitability_map,
    }


class ClimateSimulator:

    def __init__(self, elev: np.ndarray, lat_grid: np.ndarray):
        self.elev = elev
        self.lat_grid = lat_grid

    def run(
        self,
        wind_direction: str = 'westerly',
        moisture_factor: float = 1.0,
        temperature_bias: float = 0.0,
        river_flux: Optional[np.ndarray] = None,
        include_resources: bool = False,
    ) -> Dict[str, np.ndarray]:
        temp = calculate_temperature(self.elev, self.lat_grid, temperature_bias=temperature_bias)
        precip = calculate_precipitation(self.elev, wind_direction, moisture_factor)
        moisture = calculate_moisture(precip, self.elev, river_flux)
        biome = classify_biome(temp, precip, self.elev, river_flux)
        result = {
            'temperature': temp,
            'precipitation': precip,
            'biome': biome,
            'moisture': moisture,
        }
        if include_resources:
            result.update(compute_resources(biome, self.elev, precip, temp))
        return result
