from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter, label

from app.core.terrain import TerrainGenerator


def build_target_land_mask(
    terrain: TerrainGenerator,
    topology_intent: dict,
    width: int | None = None,
    height: int | None = None,
) -> np.ndarray:
    kind = str(topology_intent.get("kind", "")).strip().lower()
    modifiers = topology_intent.get("modifiers") or {}

    if kind == "single_island":
        return _build_single_island_target_mask(terrain, modifiers)
    if kind == "two_continents_with_rift_sea":
        return _build_two_continents_target_mask(terrain, modifiers)
    if kind == "central_enclosed_inland_sea":
        return _build_inland_sea_target_mask(terrain, modifiers)
    if kind == "archipelago_chain":
        return _build_archipelago_target_mask(terrain, modifiers)
    if kind == "peninsula_coast":
        return _build_peninsula_target_mask(terrain, modifiers)
    return np.zeros((terrain.height, terrain.width), dtype=np.float32)


def build_target_water_mask(
    terrain: TerrainGenerator,
    topology_intent: dict,
) -> np.ndarray:
    kind = str(topology_intent.get("kind", "")).strip().lower()
    modifiers = topology_intent.get("modifiers") or {}

    if kind == "two_continents_with_rift_sea":
        return _build_rift_sea_target_mask(terrain, modifiers)
    if kind == "central_enclosed_inland_sea":
        return _build_inland_basin_target_mask(terrain, modifiers)
    return np.ones((terrain.height, terrain.width), dtype=np.float32)


def fit_elevation_to_mask(
    target_land_mask: np.ndarray,
    target_water_mask: np.ndarray | None = None,
    ocean_depth: float = 0.6,
    noise_amplitude: float = 0.15,
    seed: int = 42,
) -> np.ndarray:
    dist_inside = distance_transform_edt(target_land_mask > 0.5)
    dist_outside = distance_transform_edt(target_land_mask <= 0.5)

    dist_inside_norm = dist_inside / max(float(dist_inside.max()), 1e-6)
    dist_outside_norm = dist_outside / max(float(dist_outside.max()), 1e-6)

    elevation = dist_inside_norm * 0.8 - dist_outside_norm * ocean_depth

    rng = np.random.RandomState(seed)
    noise = rng.randn(*elevation.shape).astype(np.float32)
    noise = gaussian_filter(noise, sigma=8.0)
    noise = noise / max(float(np.abs(noise).max()), 1e-6) * noise_amplitude
    elevation += noise

    return np.clip(elevation, -1.0, 1.0).astype(np.float32)


def _build_single_island_target_mask(terrain: TerrainGenerator, modifiers: dict) -> np.ndarray:
    shape_bias = str(modifiers.get("shape_bias", "balanced")).lower()
    shape_axis = str(modifiers.get("shape_axis", "east_west")).lower()

    if shape_bias == "elongated":
        if shape_axis == "north_south":
            skeleton = np.maximum.reduce([
                terrain._elliptic_gaussian(0.5, 0.5, 0.38, 0.05, 0.0),
                terrain._elliptic_gaussian(0.25, 0.49, 0.16, 0.05, -0.03) * 0.85,
                terrain._elliptic_gaussian(0.76, 0.52, 0.17, 0.05, 0.04) * 0.88,
            ])
        else:
            skeleton = np.maximum.reduce([
                terrain._elliptic_gaussian(0.5, 0.5, 0.05, 0.38, 0.0),
                terrain._elliptic_gaussian(0.49, 0.25, 0.05, 0.16, -0.03) * 0.85,
                terrain._elliptic_gaussian(0.52, 0.76, 0.05, 0.17, 0.04) * 0.88,
            ])
    elif shape_bias == "round":
        skeleton = terrain._elliptic_gaussian(0.5, 0.5, 0.18, 0.18, 0.0)
    else:
        skeleton = np.maximum.reduce([
            terrain._elliptic_gaussian(0.52, 0.48, 0.18, 0.15, -0.15),
            terrain._elliptic_gaussian(0.47, 0.55, 0.14, 0.18, 0.2),
        ])

    coastal_noise = terrain._fbm(scale=52.0, octaves=4, persistence=0.56, lacunarity=2.05, offset=805.0)
    irregularity = float(modifiers.get("boundary_irregularity", 0.5))
    mask = np.clip(skeleton * (0.86 + coastal_noise * 0.26 * irregularity), 0.0, 1.0)
    mask = keep_largest_component(mask > 0.4).astype(np.float32)
    return gaussian_filter(mask, sigma=1.2).astype(np.float32)


def _build_two_continents_target_mask(terrain: TerrainGenerator, modifiers: dict) -> np.ndarray:
    west = np.maximum.reduce([
        terrain._elliptic_gaussian(0.52, 0.24, 0.24, 0.16, -0.16),
        terrain._elliptic_gaussian(0.36, 0.3, 0.14, 0.11, 0.3) * 0.72,
    ])
    east = np.maximum.reduce([
        terrain._elliptic_gaussian(0.48, 0.76, 0.22, 0.17, 0.2),
        terrain._elliptic_gaussian(0.33, 0.69, 0.12, 0.1, -0.28) * 0.7,
    ])
    coastal_noise_w = terrain._fbm(scale=58.0, octaves=3, persistence=0.55, lacunarity=2.0, offset=847.0)
    coastal_noise_e = terrain._fbm(scale=61.0, octaves=3, persistence=0.57, lacunarity=2.02, offset=883.0)
    irregularity = float(modifiers.get("boundary_irregularity", 0.5))
    west = np.clip(west * (0.88 + coastal_noise_w * 0.18 * irregularity), 0.0, 1.0)
    east = np.clip(east * (0.9 + coastal_noise_e * 0.18 * irregularity), 0.0, 1.0)
    return np.maximum(west, east)


def _build_inland_sea_target_mask(terrain: TerrainGenerator, modifiers: dict) -> np.ndarray:
    north = np.maximum.reduce([
        terrain._elliptic_gaussian(0.24, 0.46, 0.16, 0.28, -0.1),
        terrain._elliptic_gaussian(0.31, 0.68, 0.11, 0.15, 0.3) * 0.74,
    ])
    south = np.maximum.reduce([
        terrain._elliptic_gaussian(0.76, 0.54, 0.17, 0.26, 0.14),
        terrain._elliptic_gaussian(0.67, 0.31, 0.1, 0.14, -0.26) * 0.7,
    ])
    west_shoulder = terrain._elliptic_gaussian(0.53, 0.25, 0.09, 0.12, -0.5) * 0.6
    east_shoulder = terrain._elliptic_gaussian(0.47, 0.76, 0.07, 0.11, 0.4) * 0.46
    return np.maximum.reduce([north, south, west_shoulder, east_shoulder])


def _build_archipelago_target_mask(terrain: TerrainGenerator, modifiers: dict) -> np.ndarray:
    density = str(modifiers.get("island_density", "balanced")).lower()
    specs = [
        ("west", 0.082, 0.0, 0.74),
        ("east", 0.08, 0.18, 0.7),
        ("northwest", 0.072, -0.32, 0.64),
        ("northeast", 0.07, 0.28, 0.6),
        ("southwest", 0.074, 0.14, 0.66),
        ("southeast", 0.068, -0.22, 0.58),
    ]
    if density == "dense":
        specs.append(("center", 0.062, 0.08, 0.54))
    archipelago = np.zeros((terrain.height, terrain.width), dtype=np.float32)
    for position, size, rotation, weight in specs:
        cy, cx = terrain._resolve_position(position)
        island = terrain._elliptic_gaussian(cy, cx, size * 0.8, size * 0.55, rotation)
        archipelago = np.maximum(archipelago, island.astype(np.float32) * weight)
    return archipelago


def _build_peninsula_target_mask(terrain: TerrainGenerator, modifiers: dict) -> np.ndarray:
    mainland = np.maximum.reduce([
        terrain._elliptic_gaussian(0.5, 0.23, 0.28, 0.2, -0.06),
        terrain._elliptic_gaussian(0.34, 0.29, 0.14, 0.1, 0.2) * 0.7,
    ])
    peninsula = terrain._elliptic_gaussian(0.5, 0.68, 0.06, 0.18, 0.12) * 1.1
    return np.maximum(mainland, peninsula)


def _build_rift_sea_target_mask(terrain: TerrainGenerator, modifiers: dict) -> np.ndarray:
    rift_width = str(modifiers.get("rift_width", "balanced")).lower()
    width_scale = {"narrow": 0.78, "balanced": 1.0, "broad": 1.3}.get(rift_width, 1.0)
    x = terrain._x_norm
    centerline = 0.5 + np.sin((terrain._y_norm - 0.5) * np.pi * 2.3) * 0.025
    width = 0.05 * width_scale
    channel = np.exp(-((x - centerline) ** 2) / max(width ** 2, 1e-4))
    path_noise = terrain._fbm(scale=92.0, octaves=3, persistence=0.56, lacunarity=2.0, offset=417.0)
    centerline_perturbed = centerline + (np.mean(path_noise, axis=1, keepdims=True) * 2.0 - 1.0) * 0.04
    channel_perturbed = np.exp(-((x - centerline_perturbed) ** 2) / max(width ** 2, 1e-4))
    return gaussian_filter(np.maximum(channel, channel_perturbed).astype(np.float32), sigma=1.2)


def _build_inland_basin_target_mask(terrain: TerrainGenerator, modifiers: dict) -> np.ndarray:
    basin_shape = str(modifiers.get("basin_shape", "balanced")).lower()
    base = np.maximum.reduce([
        terrain._elliptic_gaussian(0.47, 0.43, 0.1, 0.13, -0.2),
        terrain._elliptic_gaussian(0.54, 0.58, 0.095, 0.17, 0.18),
        terrain._elliptic_gaussian(0.5, 0.51, 0.07, 0.22, 0.08) * 0.88,
    ])
    if basin_shape == "broad":
        base = np.maximum(base, terrain._elliptic_gaussian(0.5, 0.5, 0.12, 0.26, 0.0) * 0.92)
    elif basin_shape == "branched":
        coves = np.maximum.reduce([
            terrain._elliptic_gaussian(0.38, 0.34, 0.04, 0.12, -0.72) * 0.82,
            terrain._elliptic_gaussian(0.62, 0.68, 0.035, 0.1, 0.44) * 0.66,
        ])
        base = np.maximum(base, coves)
    return gaussian_filter(base, sigma=1.2)


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    labels_arr, num = label(mask.astype(np.int8))
    if num <= 1:
        return mask
    sizes = {cid: int(np.sum(labels_arr == cid)) for cid in range(1, num + 1)}
    dominant = max(sizes, key=sizes.get)
    return (labels_arr == dominant).astype(np.float32)
