from __future__ import annotations

import logging

import numpy as np
from scipy.ndimage import (
    label,
    binary_dilation,
    generate_binary_structure,
    uniform_filter,
    sobel,
    gaussian_filter,
)

logger = logging.getLogger(__name__)


def compute_d8_flow_direction(dem: np.ndarray) -> np.ndarray:
    h, w = dem.shape
    dem_f = dem.astype(np.float64)
    dirs = np.zeros((h, w), dtype=np.uint8)
    dy_offsets = [-1, -1, 0, 1, 1, 1, 0, -1]
    dx_offsets = [0, 1, 1, 1, 0, -1, -1, -1]
    dir_codes = [1, 2, 4, 8, 16, 32, 64, 128]

    for i in range(8):
        shifted = np.roll(np.roll(dem_f, -dy_offsets[i], axis=0), -dx_offsets[i], axis=1)
        drop = dem_f - shifted
        is_steepest = np.zeros((h, w), dtype=bool)
        for j in range(8):
            shifted_j = np.roll(np.roll(dem_f, -dy_offsets[j], axis=0), -dx_offsets[j], axis=1)
            drop_j = dem_f - shifted_j
            is_steepest |= (drop > drop_j)
        dirs[is_steepest & (drop > 0)] = dir_codes[i]

    return dirs


def compute_accumulation(d8_dirs: np.ndarray, dem: np.ndarray) -> np.ndarray:
    h, w = dem.shape
    acc = np.ones((h, w), dtype=np.float64)
    dem_f = dem.astype(np.float64)

    dy_map = {1: -1, 2: -1, 4: 0, 8: 1, 16: 1, 32: 1, 64: 0, 128: -1}
    dx_map = {1: 0, 2: 1, 4: 1, 8: 1, 16: 0, 32: -1, 64: -1, 128: -1}

    order_indices = np.argsort(dem_f.ravel())[::-1]
    for idx in order_indices:
        r, c = divmod(int(idx), w)
        d = int(d8_dirs[r, c])
        if d == 0:
            continue
        dr = dy_map.get(d, 0)
        dc = dx_map.get(d, 0)
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            acc[nr, nc] += acc[r, c]

    return acc


def detect_ridges(
    dem: np.ndarray,
    accumulation_threshold: float = 0.15,
    min_elevation_percentile: float = 50.0,
) -> np.ndarray:
    d8 = compute_d8_flow_direction(dem)
    acc = compute_accumulation(d8, dem)
    max_acc = np.max(acc)
    if max_acc < 1:
        return np.zeros_like(dem, dtype=bool)
    norm_acc = acc / max_acc
    ridge_mask = norm_acc < accumulation_threshold
    elev_threshold = np.percentile(dem[dem > 0], min_elevation_percentile) if np.any(dem > 0) else 0
    ridge_mask &= dem > elev_threshold
    struct = generate_binary_structure(2, 2)
    ridge_mask = binary_dilation(ridge_mask, structure=struct, iterations=1)
    return ridge_mask


def detect_rivers(
    dem: np.ndarray,
    accumulation_threshold: float = 0.6,
) -> np.ndarray:
    d8 = compute_d8_flow_direction(dem)
    acc = compute_accumulation(d8, dem)
    max_acc = np.max(acc)
    if max_acc < 1:
        return np.zeros_like(dem, dtype=bool)
    norm_acc = acc / max_acc
    river_mask = norm_acc > accumulation_threshold
    river_mask &= dem > 0
    return river_mask


def protect_features_during_repair(
    original: np.ndarray,
    modified: np.ndarray,
    ridge_mask: np.ndarray,
    river_mask: np.ndarray,
    ridge_strength: float = 0.7,
    river_strength: float = 0.5,
) -> np.ndarray:
    result = modified.astype(np.float64).copy()
    if np.any(ridge_mask):
        ridge_deficit = original.astype(np.float64) - result
        ridge_boost = np.where(ridge_mask & (ridge_deficit > 0), ridge_deficit * ridge_strength, 0.0)
        result += ridge_boost
    if np.any(river_mask):
        river_excess = result - original.astype(np.float64)
        river_cut = np.where(river_mask & (river_excess > 0), river_excess * river_strength, 0.0)
        result -= river_cut
    return result.astype(np.float32)


def fill_depressions(dem: np.ndarray) -> np.ndarray:
    try:
        import pyflwdir
        dem_f = dem.astype(np.float32)
        flw = pyflwdir.from_dem(dem_f)
        filled = pyflwdir.fill_depressions(flw, dem_f)
        return filled
    except ImportError:
        return _fill_depressions_simple(dem)


def _fill_depressions_simple(dem: np.ndarray, iterations: int = 5) -> np.ndarray:
    result = dem.astype(np.float64).copy()
    struct = generate_binary_structure(2, 2)
    for _ in range(iterations):
        neighbor_max = np.zeros_like(result)
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                shifted = np.roll(np.roll(result, -dy, axis=0), -dx, axis=1)
                neighbor_max = np.maximum(neighbor_max, shifted)
        is_pit = (result < neighbor_max - 0.01) & (result <= 0)
        if not np.any(is_pit):
            break
        result[is_pit] = neighbor_max[is_pit] * 0.5
    return result.astype(np.float32)


def detect_closed_basins(dem: np.ndarray, min_area: int = 50) -> list[dict]:
    land_mask = dem > 0
    filled = fill_depressions(dem)
    diff = filled - dem.astype(np.float64)
    pit_mask = (diff > 0.01) & land_mask
    labeled, num_features = label(pit_mask, structure=generate_binary_structure(2, 2))
    basins = []
    for i in range(1, num_features + 1):
        component = labeled == i
        area = int(np.sum(component))
        if area < min_area:
            continue
        ys, xs = np.where(component)
        center_y = float(np.mean(ys))
        center_x = float(np.mean(xs))
        max_depth = float(np.max(diff[component]))
        basins.append({
            "id": i,
            "area": area,
            "center": (center_y, center_x),
            "max_depth": max_depth,
            "mean_depth": float(np.mean(diff[component])),
            "mask": component,
        })
    return basins


def simulate_lakes(
    dem: np.ndarray,
    basins: list[dict] | None = None,
    fill_fraction: float = 0.7,
    min_area: int = 50,
) -> tuple[np.ndarray, list[dict]]:
    if basins is None:
        basins = detect_closed_basins(dem, min_area=min_area)
    result = dem.astype(np.float64).copy()
    lake_info = []
    for basin in basins:
        mask = basin["mask"]
        pit_elev = dem[mask].astype(np.float64)
        fill_level = float(np.min(pit_elev) + basin["max_depth"] * fill_fraction)
        lake_surface = np.full_like(pit_elev, fill_level)
        result[mask] = np.minimum(pit_elev, lake_surface)
        lake_mask = result[mask] < pit_elev
        lake_area = int(np.sum(lake_mask))
        if lake_area > 0:
            lake_info.append({
                "basin_id": basin["id"],
                "lake_area": lake_area,
                "fill_level": fill_level,
                "center": basin["center"],
            })
    return result.astype(np.float32), lake_info


def fill_spill_merge(dem: np.ndarray, min_area: int = 30) -> tuple[np.ndarray, list[dict]]:
    filled = fill_depressions(dem)
    basins = detect_closed_basins(dem, min_area=min_area)
    if not basins:
        return dem.copy(), []

    basins.sort(key=lambda b: b["mean_depth"], reverse=True)
    result = dem.astype(np.float64).copy()
    merged_lakes = []

    for i, basin in enumerate(basins):
        mask = basin["mask"]
        pit_elev = dem[mask].astype(np.float64)
        fill_level = float(np.min(pit_elev) + basin["max_depth"] * 0.85)
        result[mask] = np.minimum(pit_elev, fill_level)

    for i in range(len(basins)):
        for j in range(i + 1, len(basins)):
            mask_i = basins[i]["mask"]
            mask_j = basins[j]["mask"]
            dilated_i = binary_dilation(mask_i, structure=generate_binary_structure(2, 2), iterations=2)
            if np.any(dilated_i & mask_j):
                fill_level = max(
                    float(np.min(result[mask_i])),
                    float(np.min(result[mask_j])),
                )
                combined = mask_i | mask_j
                result[combined] = np.minimum(result[combined], fill_level)
                merged_lakes.append({
                    "merged_from": [basins[i]["id"], basins[j]["id"]],
                    "fill_level": fill_level,
                })

    return result.astype(np.float32), merged_lakes


def curvature_guided_erosion(
    dem: np.ndarray,
    iterations: int = 3,
    erosion_rate: float = 0.02,
    ridge_protection: float = 0.8,
) -> np.ndarray:
    from app.core.terrain_analysis import compute_profile_curvature, compute_plan_curvature
    result = dem.astype(np.float64).copy()
    for _ in range(iterations):
        k_prof = compute_profile_curvature(result)
        k_plan = compute_plan_curvature(result)
        is_convex = (k_prof > 0) | (k_plan > 0)
        is_concave = (k_prof < 0) | (k_plan < 0)
        erosion = np.zeros_like(result)
        erosion[is_convex] = erosion_rate * (1.0 - ridge_protection)
        erosion[is_concave] = erosion_rate * 1.5
        erosion[~is_convex & ~is_concave] = erosion_rate
        land_mask = result > 0
        erosion *= land_mask.astype(np.float64)
        result -= erosion
    return result.astype(np.float32)


def multi_scale_erosion(
    dem: np.ndarray,
    scales: list[int] | None = None,
    base_rate: float = 0.01,
) -> np.ndarray:
    if scales is None:
        scales = [1, 3, 9]
    result = dem.astype(np.float64).copy()
    land_mask = result > 0
    for s in scales:
        smoothed = uniform_filter(result, size=s * 2 + 1, mode="nearest")
        local_var = (result - smoothed) ** 2
        var_norm = local_var / (np.max(local_var) + 1e-10)
        rate = base_rate * (1.0 + var_norm * 2.0)
        erosion = rate * land_mask.astype(np.float64)
        result -= erosion
    return result.astype(np.float32)


def simulate_meandering_river(
    dem: np.ndarray,
    start_y: int,
    start_x: int,
    end_y: int,
    end_x: int,
    width: int = 2,
    sinuosity: float = 0.3,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    h, w = dem.shape
    river_mask = np.zeros((h, w), dtype=bool)
    oxbow_mask = np.zeros((h, w), dtype=bool)

    n_points = max(h, w)
    t = np.linspace(0, 1, n_points)
    base_y = start_y + (end_y - start_y) * t
    base_x = start_x + (end_x - start_x) * t

    freq1 = rng.uniform(0.5, 2.0)
    freq2 = rng.uniform(2.0, 5.0)
    phase1 = rng.uniform(0, 2 * np.pi)
    phase2 = rng.uniform(0, 2 * np.pi)
    amp1 = sinuosity * h * 0.1
    amp2 = sinuosity * h * 0.03

    river_y = base_y + amp1 * np.sin(freq1 * 2 * np.pi * t + phase1) + amp2 * np.sin(freq2 * 2 * np.pi * t + phase2)
    river_x = base_x + amp1 * 0.3 * np.cos(freq1 * 2 * np.pi * t + phase1)

    river_y = np.clip(river_y, 0, h - 1).astype(int)
    river_x = np.clip(river_x, 0, w - 1).astype(int)

    for i in range(len(river_y)):
        cy, cx = river_y[i], river_x[i]
        for dy in range(-width, width + 1):
            for dx in range(-width, width + 1):
                if dy * dy + dx * dx <= width * width:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        river_mask[ny, nx] = True

    if rng.random() < 0.5 and n_points > 100:
        cut_start = rng.randint(n_points // 4, n_points // 2)
        cut_length = rng.randint(10, 30)
        for i in range(cut_start, min(cut_start + cut_length, n_points)):
            cy, cx = river_y[i], river_x[i]
            for dy in range(-width, width + 1):
                for dx in range(-width, width + 1):
                    if dy * dy + dx * dx <= width * width:
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < h and 0 <= nx < w:
                            oxbow_mask[ny, nx] = True

    result = dem.astype(np.float64).copy()
    result[river_mask] = np.minimum(result[river_mask], -0.05)
    result[oxbow_mask] = np.minimum(result[oxbow_mask], -0.02)

    return result.astype(np.float32), oxbow_mask
