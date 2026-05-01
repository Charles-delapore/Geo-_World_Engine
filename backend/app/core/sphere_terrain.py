from __future__ import annotations

import numpy as np
from scipy.interpolate import RegularGridInterpolator

_GRID_CACHE_MAX_BYTES = 256 * 1024 * 1024


def _noise3_fast(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    import opensimplex

    flat_x = x.ravel().astype(np.float64)
    flat_y = y.ravel().astype(np.float64)
    flat_z = z.ravel().astype(np.float64)

    pad = 1.0
    x_min, x_max = flat_x.min() - pad, flat_x.max() + pad
    y_min, y_max = flat_y.min() - pad, flat_y.max() + pad
    z_min, z_max = flat_z.min() - pad, flat_z.max() + pad

    range_x = max(x_max - x_min, 1.0)
    range_y = max(y_max - y_min, 1.0)
    range_z = max(z_max - z_min, 1.0)
    max_range = max(range_x, range_y, range_z)

    n_pts = max(16, min(int(max_range * 8), 128))

    est_bytes = n_pts ** 3 * 8
    if est_bytes > _GRID_CACHE_MAX_BYTES:
        _ufunc = np.frompyfunc(opensimplex.noise3, 3, 1)
        result = _ufunc(flat_x, flat_y, flat_z)
        return result.astype(np.float32).reshape(x.shape)

    xs = np.linspace(x_min, x_max, n_pts)
    ys = np.linspace(y_min, y_max, n_pts)
    zs = np.linspace(z_min, z_max, n_pts)

    grid_3d = opensimplex.noise3array(xs, ys, zs)

    interp = RegularGridInterpolator(
        (zs, ys, xs),
        grid_3d,
        method="linear",
        bounds_error=False,
        fill_value=0.0,
    )

    points = np.stack([flat_z, flat_y, flat_x], axis=-1)
    result = interp(points)

    return result.reshape(x.shape).astype(np.float32)


def erp_grid_to_sphere(width: int, height: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lon = np.linspace(0, 2 * np.pi, width, endpoint=False, dtype=np.float64)
    lat = np.linspace(np.pi / 2, -np.pi / 2, height, dtype=np.float64)
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    sx = np.cos(lat_grid) * np.cos(lon_grid)
    sy = np.cos(lat_grid) * np.sin(lon_grid)
    sz = np.sin(lat_grid)

    return sx.astype(np.float32), sy.astype(np.float32), sz.astype(np.float32)


def sphere_fbm(
    sx: np.ndarray,
    sy: np.ndarray,
    sz: np.ndarray,
    scale: float = 4.0,
    octaves: int = 6,
    persistence: float = 0.5,
    lacunarity: float = 2.0,
    seed: int = 0,
) -> np.ndarray:
    import opensimplex

    rng = np.random.RandomState(seed)
    opensimplex.seed(seed)

    result = np.zeros_like(sx, dtype=np.float32)
    amplitude = 1.0
    frequency = scale
    max_amp = 0.0

    for octave in range(octaves):
        offset_x = rng.uniform(-1000, 1000)
        offset_y = rng.uniform(-1000, 1000)
        offset_z = rng.uniform(-1000, 1000)

        sample_x = sx * frequency + offset_x
        sample_y = sy * frequency + offset_y
        sample_z = sz * frequency + offset_z

        noise_vals = _noise3_fast(sample_x, sample_y, sample_z)

        result += noise_vals * amplitude
        max_amp += amplitude
        amplitude *= persistence
        frequency *= lacunarity

    if max_amp > 0:
        result /= max_amp

    return result


def sphere_ridged_noise(
    sx: np.ndarray,
    sy: np.ndarray,
    sz: np.ndarray,
    scale: float = 4.0,
    octaves: int = 5,
    persistence: float = 0.5,
    lacunarity: float = 2.0,
    seed: int = 0,
) -> np.ndarray:
    fbm = sphere_fbm(sx, sy, sz, scale, octaves, persistence, lacunarity, seed)
    ridged = 1.0 - np.abs(2.0 * fbm - 1.0)
    return ridged * ridged


def sphere_warp(
    sx: np.ndarray,
    sy: np.ndarray,
    sz: np.ndarray,
    strength: float = 0.15,
    scale: float = 3.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    wx = sphere_fbm(sx, sy, sz, scale=scale, octaves=3, seed=seed)
    wy = sphere_fbm(sx, sy, sz, scale=scale, octaves=3, seed=seed + 100)
    wz = sphere_fbm(sx, sy, sz, scale=scale, octaves=3, seed=seed + 200)

    warped_sx = sx + wx * strength
    warped_sy = sy + wy * strength
    warped_sz = sz + wz * strength

    mag = np.sqrt(warped_sx**2 + warped_sy**2 + warped_sz**2)
    mag = np.maximum(mag, 1e-8)
    warped_sx /= mag
    warped_sy /= mag
    warped_sz /= mag

    return warped_sx.astype(np.float32), warped_sy.astype(np.float32), warped_sz.astype(np.float32)


def sphere_distance(
    sx1: np.ndarray, sy1: np.ndarray, sz1: np.ndarray,
    sx2: float, sy2: float, sz2: float,
) -> np.ndarray:
    dot = sx1 * sx2 + sy1 * sy2 + sz1 * sz2
    dot = np.clip(dot, -1.0, 1.0)
    return np.arccos(dot)


def sphere_gaussian_basin(
    sx: np.ndarray,
    sy: np.ndarray,
    sz: np.ndarray,
    center_lon_deg: float,
    center_lat_deg: float,
    radius_deg: float,
    amplitude: float = 1.0,
    irregularity: float = 0.3,
    seed: int = 0,
) -> np.ndarray:
    center_lon = np.radians(center_lon_deg)
    center_lat = np.radians(center_lat_deg)
    cx = np.cos(center_lat) * np.cos(center_lon)
    cy = np.cos(center_lat) * np.sin(center_lon)
    cz = np.sin(center_lat)

    dist = sphere_distance(sx, sy, sz, cx, cy, cz)

    if irregularity > 0.01:
        rng = np.random.RandomState(seed)
        warp_sx, warp_sy, warp_sz = sphere_warp(
            sx, sy, sz,
            strength=irregularity * 0.2,
            scale=3.0,
            seed=seed,
        )
        dist = sphere_distance(warp_sx, warp_sy, warp_sz, cx, cy, cz)

    radius_rad = np.radians(radius_deg)
    falloff = np.exp(-0.5 * (dist / max(radius_rad, 1e-6)) ** 2)

    return falloff * amplitude


def latitudinal_bias(sy_norm: np.ndarray, land_bias: float = 0.1) -> np.ndarray:
    lat_factor = np.cos((sy_norm - 0.5) * np.pi) ** 2
    return lat_factor * land_bias


def polar_ocean_bias(
    sx: np.ndarray,
    sy: np.ndarray,
    sz: np.ndarray,
    threshold_deg: float = 70.0,
    strength: float = 0.3,
) -> np.ndarray:
    lat = np.arcsin(np.clip(sz, -1, 1))
    lat_deg = np.degrees(lat)
    pole_mask = np.abs(lat_deg) > threshold_deg
    falloff = np.clip((np.abs(lat_deg) - threshold_deg) / (90.0 - threshold_deg), 0, 1)
    return -falloff * strength * pole_mask


def scale_aware_frequency(
    sx: np.ndarray,
    sy: np.ndarray,
    sz: np.ndarray,
    base_frequency: float = 4.0,
) -> np.ndarray:
    lat = np.arcsin(np.clip(sz, -1, 1))
    cos_lat = np.cos(lat)
    cos_lat = np.maximum(cos_lat, 0.1)
    return base_frequency / cos_lat
