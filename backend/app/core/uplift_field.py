from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


def generate_uplift_field(
    erp_height: int,
    erp_width: int,
    seed: int = 0,
    tectonic_plates: list[dict] | None = None,
    mountain_ridges: list[dict] | None = None,
    rift_zones: list[dict] | None = None,
    base_uplift: float = 0.0,
    continent_mask: np.ndarray | None = None,
    ruggedness: float = 0.55,
) -> np.ndarray:
    uplift = np.full((erp_height, erp_width), base_uplift, dtype=np.float32)

    lat_grid = np.linspace(np.pi / 2, -np.pi / 2, erp_height, dtype=np.float32)
    lon_grid = np.linspace(0, 2 * np.pi, erp_width, endpoint=False, dtype=np.float32)
    lon_2d, lat_2d = np.meshgrid(lon_grid, lat_grid)

    sx = np.cos(lat_2d) * np.cos(lon_2d)
    sy = np.cos(lat_2d) * np.sin(lon_2d)
    sz = np.sin(lat_2d)

    if tectonic_plates:
        uplift = _apply_plate_uplift(uplift, sx, sy, sz, tectonic_plates, ruggedness)

    if mountain_ridges:
        uplift = _apply_ridge_uplift(uplift, sx, sy, sz, mountain_ridges, seed)

    if rift_zones:
        uplift = _apply_rift_subsidence(uplift, sx, sy, sz, rift_zones)

    if continent_mask is not None:
        land_uplift = 0.0005 * ruggedness
        ocean_uplift = 0.0
        uplift[continent_mask > 0] += land_uplift
        uplift[continent_mask <= 0] = np.minimum(uplift[continent_mask <= 0], ocean_uplift)

    try:
        from app.core.sphere_terrain import sphere_fbm
        noise = sphere_fbm(
            sx, sy, sz,
            scale=60.0,
            octaves=3,
            persistence=0.5,
            lacunarity=2.0,
            seed=seed + 4281,
        )
        uplift *= (0.6 + noise * 0.8)
    except Exception:
        rng = np.random.RandomState(seed + 4281)
        noise = rng.randn(erp_height, erp_width).astype(np.float32)
        noise = gaussian_filter(noise, sigma=8.0)
        noise = noise / (np.std(noise) + 1e-10)
        uplift *= (0.6 + noise * 0.4)

    uplift = gaussian_filter(uplift, sigma=2.0).astype(np.float32)

    return uplift


def _apply_plate_uplift(
    uplift: np.ndarray,
    sx: np.ndarray, sy: np.ndarray, sz: np.ndarray,
    plates: list[dict],
    ruggedness: float,
) -> np.ndarray:
    result = uplift.copy()

    for plate in plates:
        center_lat_deg = float(plate.get("lat", 0.0))
        center_lon_deg = float(plate.get("lon", 0.0))
        rate = float(plate.get("rate", 0.001)) * ruggedness
        radius_deg = float(plate.get("radius", 30.0))
        falloff = float(plate.get("falloff", 0.5))

        center_lat = np.radians(center_lat_deg)
        center_lon = np.radians(center_lon_deg)
        cx = np.cos(center_lat) * np.cos(center_lon)
        cy = np.cos(center_lat) * np.sin(center_lon)
        cz = np.sin(center_lat)

        dot = sx * cx + sy * cy + sz * cz
        angular_dist = np.arccos(np.clip(dot, -1.0, 1.0))
        radius_rad = np.radians(radius_deg)

        sigma = radius_rad * falloff
        plate_uplift = rate * np.exp(-(angular_dist ** 2) / (2.0 * max(sigma, 0.01) ** 2))
        result += plate_uplift.astype(np.float32)

    return result


def _apply_ridge_uplift(
    uplift: np.ndarray,
    sx: np.ndarray, sy: np.ndarray, sz: np.ndarray,
    ridges: list[dict],
    seed: int,
) -> np.ndarray:
    result = uplift.copy()

    for ridge in ridges:
        start_lat = np.radians(float(ridge.get("start_lat", 0.0)))
        start_lon = np.radians(float(ridge.get("start_lon", 0.0)))
        end_lat = np.radians(float(ridge.get("end_lat", 0.0)))
        end_lon = np.radians(float(ridge.get("end_lon", 0.0)))
        rate = float(ridge.get("rate", 0.002))
        width_deg = float(ridge.get("width", 5.0))
        asymmetry = float(ridge.get("asymmetry", 0.0))

        angular_dist = np.arccos(np.clip(
            np.sin(start_lat) * np.sin(end_lat)
            + np.cos(start_lat) * np.cos(end_lat) * np.cos(end_lon - start_lon),
            -1.0, 1.0,
        ))
        n_segments = max(10, int(np.degrees(angular_dist)) * 2)

        min_dist = np.full_like(result, np.pi, dtype=np.float32)
        for i in range(n_segments + 1):
            t = i / max(n_segments, 1)
            lat = start_lat + t * (end_lat - start_lat)
            lon = start_lon + t * (end_lon - start_lon)

            px = np.cos(lat) * np.cos(lon)
            py = np.cos(lat) * np.sin(lon)
            pz = np.sin(lat)

            dot = sx * px + sy * py + sz * pz
            dist = np.arccos(np.clip(dot, -1.0, 1.0))
            min_dist = np.minimum(min_dist, dist)

        width_rad = np.radians(width_deg)
        ridge_field = rate * np.exp(-(min_dist ** 2) / (2.0 * max(width_rad * 0.5, 0.001) ** 2))

        if abs(asymmetry) > 0.01:
            start_x = np.cos(start_lat) * np.cos(start_lon)
            start_y = np.cos(start_lat) * np.sin(start_lon)
            end_x = np.cos(end_lat) * np.cos(end_lon)
            end_y = np.cos(end_lat) * np.sin(end_lon)
            ridge_dx = end_x - start_x
            ridge_dy = end_y - start_y
            perp_x = -ridge_dy
            perp_y = ridge_dx
            perp_len = np.sqrt(perp_x ** 2 + perp_y ** 2 + 1e-10)
            perp_x /= perp_len
            perp_y /= perp_len

            mid_x = (start_x + end_x) / 2
            mid_y = (start_y + end_y) / 2
            side = (sx - mid_x) * perp_x + (sy - mid_y) * perp_y
            asym_factor = 1.0 + np.tanh(side * asymmetry * 5.0) * 0.3
            ridge_field *= asym_factor.astype(np.float32)

        result += ridge_field.astype(np.float32)

    return result


def _apply_rift_subsidence(
    uplift: np.ndarray,
    sx: np.ndarray, sy: np.ndarray, sz: np.ndarray,
    rifts: list[dict],
) -> np.ndarray:
    result = uplift.copy()

    for rift in rifts:
        center_lat = np.radians(float(rift.get("lat", 0.0)))
        center_lon = np.radians(float(rift.get("lon", 0.0)))
        subsidence_rate = float(rift.get("rate", -0.001))
        width_deg = float(rift.get("width", 10.0))
        length_deg = float(rift.get("length", 60.0))
        orientation_deg = float(rift.get("orientation", 0.0))

        cx = np.cos(center_lat) * np.cos(center_lon)
        cy = np.cos(center_lat) * np.sin(center_lon)
        cz = np.sin(center_lat)

        dot = sx * cx + sy * cy + sz * cz
        angular_dist = np.arccos(np.clip(dot, -1.0, 1.0))

        width_rad = np.radians(width_deg)
        length_rad = np.radians(length_deg)

        orientation = np.radians(orientation_deg)
        local_x = sx * np.cos(orientation) + sy * np.sin(orientation)
        local_y = -sx * np.sin(orientation) + sy * np.cos(orientation)

        along_dist = np.abs(local_x - (cx * np.cos(orientation) + cy * np.sin(orientation)))
        across_dist = np.abs(local_y - (-cx * np.sin(orientation) + cy * np.cos(orientation)))

        width_falloff = np.exp(-(across_dist ** 2) / (2.0 * max(width_rad * 0.5, 0.001) ** 2))
        length_gate = np.where(along_dist < length_rad, 1.0, np.exp(-((along_dist - length_rad) ** 2) / (2.0 * max(width_rad, 0.01) ** 2)))

        rift_field = subsidence_rate * width_falloff * length_gate
        result += rift_field.astype(np.float32)

    return result


def generate_hardness_map(
    elevation: np.ndarray,
    ruggedness: float = 0.55,
    mountain_hardness: float = 0.6,
    coastal_hardness: float = 1.4,
    sedimentary_hardness: float = 0.85,
    seed: int = 0,
) -> np.ndarray:
    h, w = elevation.shape
    hardness = np.ones((h, w), dtype=np.float32)

    gy, gx = np.gradient(elevation)
    slope = np.sqrt(gx ** 2 + gy ** 2 + 1e-10)

    land = elevation > 0
    if np.any(land):
        slope_threshold = np.percentile(slope[land], 70)
        steep = (slope > slope_threshold) & land
        hardness[steep] = mountain_hardness

    from scipy.ndimage import distance_transform_edt
    land_mask = elevation > 0
    water_mask = elevation <= 0
    if np.any(land_mask) and np.any(water_mask):
        dist_to_water = distance_transform_edt(land_mask)
        coastal_band = (dist_to_water < 5) & land_mask
        hardness[coastal_band] = coastal_hardness

    low_elevation = (elevation > 0) & (elevation < 0.15) & (slope < np.percentile(slope[land], 40) if np.any(land) else 0.1)
    hardness[low_elevation] = sedimentary_hardness

    try:
        from app.core.sphere_terrain import erp_grid_to_sphere, sphere_fbm
        sx, sy, sz = erp_grid_to_sphere(w, h)
        noise = sphere_fbm(sx, sy, sz, scale=40.0, octaves=3, persistence=0.45, lacunarity=2.0, seed=seed + 4282)
        hardness *= (0.85 + noise * 0.3)
    except Exception:
        rng = np.random.RandomState(seed + 4282)
        noise = rng.randn(h, w).astype(np.float32)
        noise = gaussian_filter(noise, sigma=6.0)
        noise = noise / (np.std(noise) + 1e-10)
        hardness *= (0.85 + noise * 0.15)

    hardness = np.clip(hardness, 0.1, 3.0).astype(np.float32)
    hardness = gaussian_filter(hardness, sigma=1.5).astype(np.float32)
    return hardness


def extract_ridge_tree_from_dem(
    elevation: np.ndarray,
    min_ridge_height: float = 0.3,
    ridge_persistence: float = 0.6,
) -> list[dict]:
    h, w = elevation.shape
    gy, gx = np.gradient(elevation)
    curvature = np.gradient(gx, axis=1) + np.gradient(gy, axis=0)

    ridge_mask = (curvature > 0) & (elevation > min_ridge_height)

    from scipy.ndimage import label
    labeled, n_features = label(ridge_mask.astype(np.int8))

    ridges = []
    for i in range(1, min(n_features + 1, 20)):
        component = labeled == i
        if np.sum(component) < 10:
            continue

        ys, xs = np.where(component)
        weights = elevation[component]
        total_weight = weights.sum() + 1e-10

        mean_lat_idx = np.average(ys, weights=weights)
        mean_lon_idx = np.average(xs, weights=weights)

        lat_deg = 90.0 - (mean_lat_idx / h) * 180.0
        lon_deg = (mean_lon_idx / w) * 360.0 - 180.0

        mean_elev = np.average(elevation[component], weights=weights)
        extent = max(np.max(ys) - np.min(ys), np.max(xs) - np.min(xs))
        rate = 0.001 * (mean_elev / max(min_ridge_height, 0.01)) * ridge_persistence

        ridges.append({
            "lat": lat_deg,
            "lon": lon_deg,
            "rate": rate,
            "width": max(3.0, extent * 0.3),
            "start_lat": 90.0 - (np.min(ys) / h) * 180.0,
            "start_lon": (np.min(xs) / w) * 360.0 - 180.0,
            "end_lat": 90.0 - (np.max(ys) / h) * 180.0,
            "end_lon": (np.max(xs) / w) * 360.0 - 180.0,
        })

    return ridges
