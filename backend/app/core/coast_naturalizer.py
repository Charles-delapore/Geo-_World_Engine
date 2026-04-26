from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter

from app.core.terrain import TerrainGenerator


def naturalize_coastline(
    terrain: TerrainGenerator,
    elevation: np.ndarray,
    boundary_irregularity: float = 0.5,
    coast_complexity: float = 0.5,
) -> np.ndarray:
    land_mask = elevation > 0.0
    water_mask = elevation <= 0.0
    if not np.any(land_mask) or not np.any(water_mask):
        return elevation

    dist_land = distance_transform_edt(land_mask)
    dist_water = distance_transform_edt(water_mask)
    boundary_width = max(3, int(6 + coast_complexity * 8))
    boundary_band = (dist_land < boundary_width) & (dist_water < boundary_width)

    if not np.any(boundary_band):
        return elevation

    low_freq = terrain._fbm(scale=120.0, octaves=3, persistence=0.56, lacunarity=2.0, offset=1201.0)
    mid_freq = terrain._fbm(scale=52.0, octaves=4, persistence=0.55, lacunarity=2.1, offset=1207.0)
    high_freq = terrain._fbm(scale=24.0, octaves=3, persistence=0.5, lacunarity=2.2, offset=1213.0)

    perturbation = (
        (low_freq * 2.0 - 1.0) * 0.12 * boundary_irregularity
        + (mid_freq * 2.0 - 1.0) * 0.08 * boundary_irregularity * coast_complexity
        + (high_freq * 2.0 - 1.0) * 0.03 * coast_complexity
    )

    falloff = np.clip(1.0 - np.minimum(dist_land, dist_water) / max(boundary_width, 1), 0.0, 1.0)
    shaped = elevation.astype(np.float32).copy()
    shaped[boundary_band] += (perturbation * falloff)[boundary_band].astype(np.float32)

    land_core = dist_land >= boundary_width
    water_core = dist_water >= boundary_width
    shaped[land_core] = np.maximum(shaped[land_core], elevation[land_core])
    shaped[water_core] = np.minimum(shaped[water_core], elevation[water_core])

    return gaussian_filter(shaped, sigma=0.8).astype(np.float32)


def naturalize_basin_boundary(
    terrain: TerrainGenerator,
    basin_mask: np.ndarray,
    irregularity: float = 0.5,
) -> np.ndarray:
    if irregularity < 0.01:
        return basin_mask

    low_freq = terrain._fbm(scale=80.0, octaves=3, persistence=0.56, lacunarity=2.0, offset=1221.0)
    mid_freq = terrain._fbm(scale=38.0, octaves=4, persistence=0.55, lacunarity=2.1, offset=1229.0)
    high_freq = terrain._fbm(scale=18.0, octaves=3, persistence=0.48, lacunarity=2.3, offset=1237.0)

    dist_inside = distance_transform_edt(basin_mask > 0.5)
    dist_outside = distance_transform_edt(basin_mask <= 0.5)
    boundary_width = max(5, int(6 + irregularity * 10))
    boundary = (dist_inside < boundary_width) | (dist_outside < boundary_width)

    perturbation = (
        (low_freq * 2.0 - 1.0) * 0.35 * irregularity
        + (mid_freq * 2.0 - 1.0) * 0.2 * irregularity
        + (high_freq * 2.0 - 1.0) * 0.1 * irregularity
    )

    disturbed = basin_mask.astype(np.float32).copy()
    falloff = np.clip(1.0 - np.minimum(dist_inside, dist_outside) / max(boundary_width, 1), 0.0, 1.0)
    disturbed[boundary] += (perturbation * falloff)[boundary].astype(np.float32)

    core = dist_inside >= boundary_width
    disturbed[core] = np.maximum(disturbed[core], basin_mask[core].astype(np.float32))

    return np.clip(gaussian_filter(disturbed, sigma=0.9), 0.0, 1.0).astype(np.float32)


def enforce_fractal_dimension(
    terrain: TerrainGenerator,
    elevation: np.ndarray,
    target_fd: float = 1.15,
    tolerance: float = 0.08,
    max_iterations: int = 3,
    coast_complexity: float = 0.5,
) -> np.ndarray:
    from app.core.terrain_analysis import compute_coastline_fractal_dimension

    land_mask = elevation > 0.0
    if np.sum(land_mask) < 100:
        return elevation

    current_fd = compute_coastline_fractal_dimension(land_mask)

    result = elevation.astype(np.float32).copy()
    for _ in range(max_iterations):
        if abs(current_fd - target_fd) <= tolerance:
            break

        land = result > 0.0
        water = result <= 0.0
        if not np.any(land) or not np.any(water):
            break

        dist_land = distance_transform_edt(land)
        dist_water = distance_transform_edt(water)
        boundary_width = max(3, int(4 + coast_complexity * 6))
        boundary_band = (dist_land < boundary_width) & (dist_water < boundary_width)
        if not np.any(boundary_band):
            break

        falloff = np.clip(1.0 - np.minimum(dist_land, dist_water) / max(boundary_width, 1), 0.0, 1.0)

        if current_fd < target_fd:
            noise = terrain._fbm(scale=28.0, octaves=4, persistence=0.52, lacunarity=2.2, offset=1311.0) * 2.0 - 1.0
            strength = min((target_fd - current_fd) * 2.0, 0.3)
            perturbation = noise * strength * falloff
            result[boundary_band] += perturbation[boundary_band].astype(np.float32)
        else:
            smoothed = gaussian_filter(result, sigma=1.2)
            blend = min((current_fd - target_fd) * 1.5, 0.5)
            result[boundary_band] = (
                result[boundary_band] * (1.0 - blend * falloff[boundary_band])
                + smoothed[boundary_band] * blend * falloff[boundary_band]
            )

        land_mask_new = result > 0.0
        if np.sum(land_mask_new) < 100:
            break
        current_fd = compute_coastline_fractal_dimension(land_mask_new)

    return result
