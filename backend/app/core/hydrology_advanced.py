from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, label, uniform_filter


def detect_ridges(
    elevation: np.ndarray,
    threshold_percentile: float = 85.0,
    min_length: int = 5,
    smooth_sigma: float = 2.0,
) -> np.ndarray:
    smoothed = gaussian_filter(elevation, sigma=smooth_sigma)
    gy, gx = np.gradient(smoothed)
    gxx = np.gradient(gx, axis=1)
    gyy = np.gradient(gy, axis=0)
    gxy = np.gradient(gx, axis=0)

    denom = (gx ** 2 + gy ** 2) ** 1.5 + 1e-10
    profile_curvature = -(gxx * gx ** 2 + 2 * gxy * gx * gy + gyy * gy ** 2) / denom

    plan_curvature = -(gxx * gy ** 2 - 2 * gxy * gx * gy + gyy * gx ** 2) / (gx ** 2 + gy ** 2 + 1e-10) ** 1.5

    ridge_score = np.maximum(-profile_curvature, 0) + np.maximum(-plan_curvature, 0) * 0.5

    threshold = np.percentile(ridge_score[elevation > 0], threshold_percentile) if np.any(elevation > 0) else 1.0
    ridge_mask = ridge_score > threshold

    if min_length > 0:
        labeled, n_features = label(ridge_mask.astype(np.int8))
        for i in range(1, n_features + 1):
            component = labeled == i
            if np.sum(component) < min_length:
                ridge_mask[component] = False

    return ridge_mask.astype(np.float32)


def detect_rivers(
    elevation: np.ndarray,
    threshold_percentile: float = 15.0,
    min_length: int = 10,
    smooth_sigma: float = 2.0,
) -> np.ndarray:
    smoothed = gaussian_filter(elevation, sigma=smooth_sigma)
    gy, gx = np.gradient(smoothed)
    gxx = np.gradient(gx, axis=1)
    gyy = np.gradient(gy, axis=0)
    gxy = np.gradient(gx, axis=0)

    denom = (gx ** 2 + gy ** 2) ** 1.5 + 1e-10
    profile_curvature = -(gxx * gx ** 2 + 2 * gxy * gx * gy + gyy * gy ** 2) / denom

    valley_score = np.maximum(profile_curvature, 0)

    try:
        from app.core.sphere_stream_power import _compute_d8_flow, _accumulate_drainage_area_fast
        flow_dir = _compute_d8_flow(smoothed)
        drainage = _accumulate_drainage_area_fast(smoothed, flow_dir)
        log_drainage = np.log1p(drainage)
        if np.max(log_drainage) > 0:
            valley_score = valley_score * 0.5 + (log_drainage / np.max(log_drainage)) * 0.5
    except Exception:
        pass

    land = elevation > 0
    if np.any(land):
        threshold = np.percentile(valley_score[land], 100 - threshold_percentile)
    else:
        threshold = 1.0
    river_mask = valley_score > threshold

    if min_length > 0:
        labeled, n_features = label(river_mask.astype(np.int8))
        for i in range(1, n_features + 1):
            component = labeled == i
            if np.sum(component) < min_length:
                river_mask[component] = False

    return river_mask.astype(np.float32)


def protect_features_during_repair(
    elevation: np.ndarray,
    original_elevation: np.ndarray,
    protection_strength: float = 0.7,
) -> np.ndarray:
    ridges = detect_ridges(original_elevation)
    rivers = detect_rivers(original_elevation)

    feature_mask = np.maximum(ridges, rivers)

    if np.max(feature_mask) < 1e-6:
        return elevation

    blend = feature_mask * protection_strength
    result = elevation * (1 - blend) + original_elevation * blend

    return result.astype(np.float32)


def compute_drainage_density(
    elevation: np.ndarray,
    cell_size: float = 1.0,
) -> float:
    rivers = detect_rivers(elevation, threshold_percentile=10.0, min_length=3)
    river_length = np.sum(rivers > 0) * cell_size
    total_area = elevation.size * cell_size ** 2
    return river_length / max(total_area, 1e-10)


def compute_hypsometric_curve(
    elevation: np.ndarray,
    n_bins: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    land = elevation[elevation > 0]
    if len(land) == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])

    h_min, h_max = land.min(), land.max()
    if h_max - h_min < 1e-10:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])

    bins = np.linspace(h_min, h_max, n_bins + 1)
    hist, _ = np.histogram(land, bins=bins)
    cumulative = np.cumsum(hist)
    cumulative = cumulative / cumulative[-1]

    normalized_height = (bins[1:] - h_min) / (h_max - h_min)

    return normalized_height.astype(np.float32), cumulative.astype(np.float32)


def compute_hypsometric_integral(elevation: np.ndarray) -> float:
    norm_h, cum_area = compute_hypsometric_curve(elevation)
    if len(norm_h) < 2:
        return 0.5
    hi = np.trapezoid(cum_area, norm_h)
    return float(hi)


def curvature_guided_erosion(
    elevation: np.ndarray,
    iterations: int = 2,
    erosion_rate: float = 0.012,
    ridge_protection: float = 0.75,
) -> np.ndarray:
    result = elevation.astype(np.float32).copy()

    for _ in range(iterations):
        gy, gx = np.gradient(result)
        grad_mag = np.sqrt(gx ** 2 + gy ** 2 + 1e-10)

        gyy, gyx = np.gradient(gy)
        gxy, gxx = np.gradient(gx)
        curvature = gxx + gyy

        is_valley = curvature < 0
        is_ridge = curvature > 0

        erosion = np.zeros_like(result)
        erosion[is_valley] = erosion_rate * (1.0 - ridge_protection) * grad_mag[is_valley]
        erosion[is_ridge] = -erosion_rate * 0.1 * grad_mag[is_ridge]

        land = result > 0
        result[land] -= erosion[land]

    return np.clip(result, -1.0, 1.0).astype(np.float32)


def multi_scale_erosion(
    elevation: np.ndarray,
    scales: list[int] | None = None,
    base_rate: float = 0.006,
) -> np.ndarray:
    if scales is None:
        scales = [1, 3]

    result = elevation.astype(np.float32).copy()

    for scale in scales:
        smoothed = uniform_filter(result, size=scale * 2 + 1)
        diff = result - smoothed
        erosion = base_rate * scale * np.clip(diff, 0, None)
        land = result > 0
        result[land] -= erosion[land]

    return np.clip(result, -1.0, 1.0).astype(np.float32)
