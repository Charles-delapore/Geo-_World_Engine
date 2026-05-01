from __future__ import annotations

import numpy as np
from scipy.ndimage import (
    distance_transform_edt,
    gaussian_filter,
    uniform_filter,
    label,
)


def _compute_gradient(elevation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gy, gx = np.gradient(elevation)
    return gx.astype(np.float32), gy.astype(np.float32)


def _compute_slope(elevation: np.ndarray) -> np.ndarray:
    gx, gy = _compute_gradient(elevation)
    return np.sqrt(gx ** 2 + gy ** 2 + 1e-10).astype(np.float32)


def _compute_d8_flow(elevation: np.ndarray) -> np.ndarray:
    h, w = elevation.shape
    dx_diag = 1.4142135623730951
    dx_card = 1.0

    padded = np.pad(elevation, 1, mode="edge")
    flow_dir = np.zeros((h, w), dtype=np.int8)

    neighbors = [
        (-1, -1, dx_diag), (-1, 0, dx_card), (-1, 1, dx_diag),
        (0, -1, dx_card),                     (0, 1, dx_card),
        (1, -1, dx_diag),  (1, 0, dx_card),  (1, 1, dx_diag),
    ]

    max_slope = np.full((h, w), -np.inf, dtype=np.float32)
    for idx, (dy, ddx, dist) in enumerate(neighbors):
        neighbor_elev = padded[1 + dy : h + 1 + dy, 1 + ddx : w + 1 + ddx]
        slope = (elevation - neighbor_elev) / dist
        better = slope > max_slope
        max_slope[better] = slope[better]
        flow_dir[better] = idx + 1

    flow_dir[elevation <= np.min(elevation) + 1e-6] = 0
    return flow_dir


FLOW_DX = np.array([0, -1, -1, -1, 0, 0, 1, 1, 1], dtype=np.int32)
FLOW_DY = np.array([0, -1, 0, 1, 1, -1, 1, 0, -1], dtype=np.int32)


def _accumulate_drainage_area(
    elevation: np.ndarray,
    flow_dir: np.ndarray,
    cell_area: float = 1.0,
) -> np.ndarray:
    h, w = elevation.shape
    area = np.full((h, w), cell_area, dtype=np.float32)

    sorted_idx = np.argsort(elevation.ravel())[::-1]
    sorted_idx = sorted_idx[elevation.ravel()[sorted_idx] > -np.inf]

    for idx in sorted_idx:
        y, x = divmod(int(idx), w)
        d = flow_dir[y, x]
        if d == 0:
            continue
        ny = y + FLOW_DY[d]
        nx = x + FLOW_DX[d]
        if 0 <= ny < h and 0 <= nx < w:
            area[ny, nx] += area[y, x]

    return area


def _accumulate_drainage_area_fast(
    elevation: np.ndarray,
    flow_dir: np.ndarray,
    cell_area: float = 1.0,
) -> np.ndarray:
    h, w = elevation.shape
    area = np.full((h, w), cell_area, dtype=np.float32)

    flat_elev = elevation.ravel()
    n = len(flat_elev)

    n_bins = min(n, 65536)
    elev_min = flat_elev.min()
    elev_max = flat_elev.max() + 1e-10
    bin_idx = ((flat_elev - elev_min) / (elev_max - elev_min) * (n_bins - 1)).astype(np.int32)
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    sorted_idx = np.argsort(-bin_idx, kind="mergesort").astype(np.int32)

    flat_flow = flow_dir.ravel()
    flat_area = area.ravel()

    for i in range(n):
        idx = sorted_idx[i]
        if flat_elev[idx] < -1e6:
            break
        d = flat_flow[idx]
        if d == 0:
            continue
        y = idx // w
        x = idx % w
        ny = y + FLOW_DY[d]
        nx = x + FLOW_DX[d]
        if 0 <= ny < h and 0 <= nx < w:
            nidx = ny * w + nx
            flat_area[nidx] += flat_area[idx]

    return flat_area.reshape(h, w)


def stream_power_erosion(
    elevation: np.ndarray,
    K: float = 2e-4,
    m: float = 0.5,
    n: float = 1.0,
    dt: float = 2000.0,
    iterations: int = 50,
    uplift: np.ndarray | None = None,
    hardness: np.ndarray | None = None,
    max_erosion_depth: float | None = None,
    wrap_longitude: bool = True,
) -> np.ndarray:
    result = elevation.astype(np.float32).copy()
    h, w = result.shape

    if uplift is None:
        uplift = np.zeros_like(result)
    else:
        uplift = uplift.astype(np.float32)

    if hardness is None:
        hardness = np.ones_like(result)
    else:
        hardness = np.clip(hardness.astype(np.float32), 0.01, 10.0)

    original_elev = result.copy()

    for _ in range(iterations):
        if wrap_longitude:
            padded = np.concatenate([result[:, -1:], result, result[:, :1]], axis=1)
            flow_input = padded
        else:
            flow_input = result

        flow_dir = _compute_d8_flow(flow_input)
        if wrap_longitude:
            flow_dir = flow_dir[:, 1:-1]

        area = _accumulate_drainage_area_fast(result, flow_dir)
        slope = _compute_slope(result)

        erosion = K * (area ** m) * (slope ** n) * hardness

        if max_erosion_depth is not None:
            erosion = np.minimum(erosion, max_erosion_depth / max(dt, 1.0))

        land_mask = result > 0
        result[land_mask] += uplift[land_mask] * dt
        result[land_mask] -= erosion[land_mask] * dt

        result = np.clip(result, -1.0, 1.0).astype(np.float32)

    return result


def thermal_erosion(
    elevation: np.ndarray,
    talus_angle: float = 0.8,
    erosion_rate: float = 0.01,
    iterations: int = 10,
    wrap_longitude: bool = True,
) -> np.ndarray:
    result = elevation.astype(np.float32).copy()
    h, w = result.shape

    for _ in range(iterations):
        if wrap_longitude:
            padded = np.concatenate([result[:, -1:], result, result[:, :1]], axis=1)
        else:
            padded = np.pad(result, ((1, 1), (1, 1)), mode="edge")

        deposition = np.zeros_like(result)

        for dy, ddx in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
            if wrap_longitude:
                src_y_start = max(0, 1 + dy)
                src_y_end = min(src_y_start + h, padded.shape[0])
                src_x_start = max(0, 1 + ddx)
                src_x_end = min(src_x_start + w, padded.shape[1])
                neighbor = padded[src_y_start:src_y_end, src_x_start:src_x_end]

                dst_y_start = src_y_start - (1 + dy)
                dst_y_end = dst_y_start + neighbor.shape[0]
                dst_x_start = src_x_start - (1 + ddx)
                dst_x_end = dst_x_start + neighbor.shape[1]

                ry = slice(max(0, dst_y_start), min(h, dst_y_end))
                rx = slice(max(0, dst_x_start), min(w, dst_x_end))
            else:
                neighbor = padded[1 + dy:1 + dy + h, 1 + ddx:1 + ddx + w]
                ry = slice(0, h)
                rx = slice(0, w)

            diff = result[ry, rx] - neighbor
            excess = np.maximum(diff - talus_angle, 0.0)
            dist = np.sqrt(dy * dy + ddx * ddx)
            talus_threshold = talus_angle * dist
            adjusted_excess = np.maximum(diff - talus_threshold, 0.0)
            transfer = erosion_rate * adjusted_excess * 0.5

            land = result[ry, rx] > 0
            transfer = transfer * land

            result[ry, rx] -= transfer

            dep_ry = slice(max(0, ry.start + dy), min(h, ry.stop + dy))
            dep_rx = slice(max(0, rx.start + ddx), min(w, rx.stop + ddx))

            src_ry = slice(dep_ry.start - (ry.start + dy), dep_ry.stop - (ry.start + dy))
            src_rx = slice(dep_rx.start - (rx.start + ddx), dep_rx.stop - (rx.start + ddx))

            if dep_ry.stop > dep_ry.start and dep_rx.stop > dep_rx.start:
                deposition[dep_ry, dep_rx] += transfer[src_ry, src_rx]

        result += deposition

    return np.clip(result, -1.0, 1.0).astype(np.float32)


def sediment_deposition(
    elevation: np.ndarray,
    flow_dir: np.ndarray,
    drainage_area: np.ndarray,
    deposition_rate: float = 0.003,
    wrap_longitude: bool = True,
) -> np.ndarray:
    result = elevation.astype(np.float32).copy()
    h, w = result.shape

    slope = _compute_slope(result)
    capacity = drainage_area ** 0.5 * slope

    low_capacity = capacity < np.percentile(capacity[result > 0], 20) if np.any(result > 0) else np.zeros_like(capacity)
    deposit_mask = (low_capacity > 0) & (result > 0)

    deposit = deposition_rate * (1.0 - slope / (np.max(slope) + 1e-10))
    result[deposit_mask] += deposit[deposit_mask]

    return np.clip(result, -1.0, 1.0).astype(np.float32)


def diffusion_retargeting(
    amplified: np.ndarray,
    original: np.ndarray,
    feature_mask: np.ndarray | None = None,
    strength: float = 0.5,
    iterations: int = 20,
) -> np.ndarray:
    if feature_mask is None:
        gx, gy = _compute_gradient(original)
        grad_mag = np.sqrt(gx ** 2 + gy ** 2)
        threshold = np.percentile(grad_mag, 85)
        feature_mask = grad_mag > threshold

    result = amplified.astype(np.float32).copy()
    error = np.where(feature_mask, original - result, 0.0).astype(np.float32)

    diffused = error.copy()
    for _ in range(iterations):
        diffused = gaussian_filter(diffused, sigma=2.0)

    result += diffused * strength
    return np.clip(result, -1.0, 1.0).astype(np.float32)


def multi_scale_breaching(
    elevation: np.ndarray,
    wrap_longitude: bool = True,
) -> np.ndarray:
    result = elevation.astype(np.float32).copy()
    h, w = result.shape

    land = result > 0
    if not np.any(land):
        return result

    filled = _fill_depressions(result, wrap_longitude=wrap_longitude)
    diff = filled - result

    breach_mask = (diff > 0) & land
    if not np.any(breach_mask):
        return result

    breach_depth = np.minimum(diff, 0.02)
    result[breach_mask] -= breach_depth[breach_mask]

    return np.clip(result, -1.0, 1.0).astype(np.float32)


def _fill_depressions(
    elevation: np.ndarray,
    wrap_longitude: bool = True,
) -> np.ndarray:
    result = elevation.copy()
    h, w = result.shape

    for _ in range(3):
        if wrap_longitude:
            padded = np.concatenate([result[:, -1:], result, result[:, :1]], axis=1)
        else:
            padded = np.pad(result, ((0, 0), (1, 1)), mode="edge")

        for dy, ddx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            src_y_start = max(0, 1 + dy)
            src_y_end = min(h + 1 + dy, padded.shape[0])
            dst_y_start = src_y_start - (1 + dy)
            dst_y_end = dst_y_start + (src_y_end - src_y_start)

            src_x_start = max(0, 1 + ddx)
            src_x_end = min(w + 1 + ddx, padded.shape[1])
            dst_x_start = src_x_start - (1 + ddx)
            dst_x_end = dst_x_start + (src_x_end - src_x_start)

            neighbor = padded[src_y_start:src_y_end, src_x_start:src_x_end]
            result_slice = result[dst_y_start:dst_y_end, dst_x_start:dst_x_end]
            lower = neighbor < result_slice
            result_slice[lower] = neighbor[lower]

    return result


def multiscale_erosion_amplification(
    elevation: np.ndarray,
    target_resolution_factor: int = 2,
    n_scales: int = 2,
    K: float = 2e-4,
    m: float = 0.5,
    n_exp: float = 1.0,
    dt: float = 2000.0,
    iterations_per_scale: int = 30,
    hardness: np.ndarray | None = None,
    uplift: np.ndarray | None = None,
    wrap_longitude: bool = True,
) -> np.ndarray:
    result = elevation.astype(np.float32).copy()

    for scale_idx in range(n_scales):
        scale_factor = target_resolution_factor ** scale_idx
        scale_K = K / (1.0 + scale_idx * 0.3)
        scale_dt = dt / (1.0 + scale_idx * 0.2)
        scale_iter = max(5, iterations_per_scale // (1 + scale_idx))

        if scale_factor > 1:
            from PIL import Image as PILImage

            h, w = result.shape
            img = PILImage.fromarray(
                ((result + 1.0) * 127.5).clip(0, 255).astype(np.uint8), mode="L"
            )
            img_up = img.resize((w * scale_factor, h * scale_factor), PILImage.BICUBIC)
            result = np.asarray(img_up).astype(np.float32) / 127.5 - 1.0

        result = stream_power_erosion(
            result,
            K=scale_K, m=m, n=n_exp, dt=scale_dt,
            iterations=scale_iter,
            uplift=uplift,
            hardness=hardness,
            wrap_longitude=wrap_longitude,
        )

        result = thermal_erosion(
            result,
            talus_angle=0.6 + scale_idx * 0.1,
            erosion_rate=0.008,
            iterations=5,
            wrap_longitude=wrap_longitude,
        )

        result = sediment_deposition(
            result,
            _compute_d8_flow(result if not wrap_longitude else np.concatenate([result[:, -1:], result, result[:, :1]], axis=1)),
            _accumulate_drainage_area_fast(result, _compute_d8_flow(result)),
            deposition_rate=0.002,
            wrap_longitude=wrap_longitude,
        )

    original = elevation
    if result.shape == original.shape:
        result = diffusion_retargeting(result, original, strength=0.3, iterations=10)

    result = multi_scale_breaching(result, wrap_longitude=wrap_longitude)

    return np.clip(result, -1.0, 1.0).astype(np.float32)
