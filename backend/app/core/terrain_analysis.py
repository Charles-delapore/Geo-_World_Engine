from __future__ import annotations

import logging

import numpy as np
from scipy.ndimage import uniform_filter, sobel, distance_transform_edt

logger = logging.getLogger(__name__)

GEOMORPHON_NAMES = [
    "flat", "summit", "ridge", "shoulder",
    "spur", "slope", "hollow", "floodplain",
    "valley", "depression",
]

GEOMORPHON_LABELS = {
    0: "flat", 1: "summit", 2: "ridge", 3: "shoulder",
    4: "spur", 5: "slope", 6: "hollow", 7: "floodplain",
    8: "valley", 9: "depression",
}

PTRM_WEIGHTS = np.array([
    0.042, 0.128, 0.095, 0.067,
    0.053, 0.198, 0.072, 0.088,
    0.112, 0.145,
], dtype=np.float64)

PTRM_INTERCEPT = -0.134


def compute_tpi(dem: np.ndarray, radius: int = 3) -> np.ndarray:
    h, w = dem.shape
    mean_elev = uniform_filter(dem.astype(np.float64), size=2 * radius + 1, mode="nearest")
    return dem.astype(np.float64) - mean_elev


def compute_multi_scale_tpi(
    dem: np.ndarray,
    scales: list[int] | None = None,
) -> dict[str, np.ndarray]:
    if scales is None:
        scales = [3, 9, 27]
    result = {}
    for s in scales:
        result[f"tpi_{s}"] = compute_tpi(dem, radius=s)
    return result


def compute_geomorphons(
    dem: np.ndarray,
    lookup_distance: int = 3,
    flat_threshold: float = 1.0,
    use_zenith_angle: bool = True,
    cell_size: float = 1.0,
) -> np.ndarray:
    h, w = dem.shape
    dem_f = dem.astype(np.float64)
    angles = np.zeros((h, w, 8), dtype=np.int16)

    offsets = [
        (0, -lookup_distance),
        (lookup_distance, -lookup_distance),
        (lookup_distance, 0),
        (lookup_distance, lookup_distance),
        (0, lookup_distance),
        (-lookup_distance, lookup_distance),
        (-lookup_distance, 0),
        (-lookup_distance, -lookup_distance),
    ]

    if use_zenith_angle:
        for i, (dy, dx) in enumerate(offsets):
            dist = np.sqrt(dy * dy + dx * dx) * cell_size

            shifted_y_start = max(0, dy)
            shifted_y_end = min(h, h + dy)
            shifted_x_start = max(0, dx)
            shifted_x_end = min(w, w + dx)

            src_y_start = max(0, -dy)
            src_y_end = src_y_start + (shifted_y_end - shifted_y_start)
            src_x_start = max(0, -dx)
            src_x_end = src_x_start + (shifted_x_end - shifted_x_start)

            if shifted_y_end <= shifted_y_start or shifted_x_end <= shifted_x_start:
                continue

            diff = dem_f[src_y_start:src_y_end, src_x_start:src_x_end] - dem_f[shifted_y_start:shifted_y_end, shifted_x_start:shifted_x_end]

            zenith = np.arctan2(diff, dist)
            nadir = np.arctan2(-diff, dist)

            angle_threshold = np.radians(flat_threshold)

            patch = np.where(
                zenith > angle_threshold, 1,
                np.where(nadir > angle_threshold, -1, 0)
            ).astype(np.int16)

            angles[src_y_start:src_y_end, src_x_start:src_x_end, i] = patch

        pad = lookup_distance
        for i in range(8):
            angles[:pad, :, i] = 0
            angles[-pad:, :, i] = 0
            angles[:, :pad, i] = 0
            angles[:, -pad:, i] = 0
    else:
        for i, (dy, dx) in enumerate(offsets):
            shifted = np.roll(np.roll(dem_f, -dy, axis=0), -dx, axis=1)
            diff = dem_f - shifted
            angles[:, :, i] = np.where(
                diff > flat_threshold, 1,
                np.where(diff < -flat_threshold, -1, 0)
            ).astype(np.int16)

        pad = lookup_distance
        angles[:pad, :, :] = 0
        angles[-pad:, :, :] = 0
        angles[:, :pad, :] = 0
        angles[:, -pad:, :] = 0

    codes = np.zeros((h, w), dtype=np.int64)
    for i in range(8):
        codes += (angles[:, :, i] + 1) * (3 ** i)

    _LUT = _build_geomorphon_lut()
    result = _LUT[codes]

    if pad > 0:
        result[:pad, :] = 5
        result[-pad:, :] = 5
        result[:, :pad] = 5
        result[:, -pad:] = 5

    return result


def _build_geomorphon_lut() -> np.ndarray:
    lut = np.zeros(3 ** 8, dtype=np.int32)
    for code in range(3 ** 8):
        ternary = []
        tmp = code
        for _ in range(8):
            ternary.append(tmp % 3 - 1)
            tmp //= 3
        pos_count = sum(1 for t in ternary if t > 0)
        neg_count = sum(1 for t in ternary if t < 0)
        if pos_count == 0 and neg_count == 0:
            lut[code] = 0
        elif pos_count >= 6 and neg_count == 0:
            lut[code] = 1
        elif pos_count >= 4 and neg_count <= 1:
            lut[code] = 2
        elif pos_count >= 2 and neg_count <= 2 and pos_count > neg_count:
            lut[code] = 3
        elif pos_count >= 1 and neg_count >= 1 and abs(pos_count - neg_count) <= 1:
            lut[code] = 4
        elif pos_count == neg_count and pos_count > 0:
            lut[code] = 5
        elif neg_count >= 1 and pos_count >= 1 and neg_count > pos_count:
            lut[code] = 6
        elif neg_count >= 2 and pos_count <= 2 and neg_count > pos_count:
            lut[code] = 7
        elif neg_count >= 4 and pos_count <= 1:
            lut[code] = 8
        elif neg_count >= 6 and pos_count == 0:
            lut[code] = 9
        else:
            lut[code] = 5
    return lut


def geomorphon_histogram(geom: np.ndarray) -> np.ndarray:
    hist = np.zeros(10, dtype=np.float64)
    total = geom.size
    if total == 0:
        return hist
    for i in range(10):
        hist[i] = np.sum(geom == i) / total
    return hist


def compute_ptrm(geom: np.ndarray) -> float:
    hist = geomorphon_histogram(geom)
    score = float(np.dot(hist, PTRM_WEIGHTS) + PTRM_INTERCEPT)
    return max(0.0, min(1.0, score))


def compute_profile_curvature(dem: np.ndarray) -> np.ndarray:
    dzdx = sobel(dem.astype(np.float64), axis=1) / 8.0
    dzdy = sobel(dem.astype(np.float64), axis=0) / 8.0
    d2zdx2 = sobel(dzdx, axis=1) / 8.0
    d2zdy2 = sobel(dzdy, axis=0) / 8.0
    d2zdxdy = sobel(dzdx, axis=0) / 8.0
    slope_mag = np.sqrt(dzdx ** 2 + dzdy ** 2) + 1e-10
    k_prof = -(d2zdx2 * dzdx ** 2 + 2 * d2zdxdy * dzdx * dzdy + d2zdy2 * dzdy ** 2) / (slope_mag ** 3)
    return k_prof


def compute_plan_curvature(dem: np.ndarray) -> np.ndarray:
    dzdx = sobel(dem.astype(np.float64), axis=1) / 8.0
    dzdy = sobel(dem.astype(np.float64), axis=0) / 8.0
    d2zdx2 = sobel(dzdx, axis=1) / 8.0
    d2zdy2 = sobel(dzdy, axis=0) / 8.0
    d2zdxdy = sobel(dzdx, axis=0) / 8.0
    slope_mag = np.sqrt(dzdx ** 2 + dzdy ** 2) + 1e-10
    k_plan = -(d2zdx2 * dzdy ** 2 - 2 * d2zdxdy * dzdx * dzdy + d2zdy2 * dzdx ** 2) / (slope_mag ** 3)
    return k_plan


def compute_gradient_magnitude(dem: np.ndarray) -> np.ndarray:
    dzdx = sobel(dem.astype(np.float64), axis=1) / 8.0
    dzdy = sobel(dem.astype(np.float64), axis=0) / 8.0
    return np.sqrt(dzdx ** 2 + dzdy ** 2)


def compute_gradient_loss(
    original: np.ndarray,
    modified: np.ndarray,
    land_mask: np.ndarray | None = None,
) -> float:
    grad_orig = compute_gradient_magnitude(original)
    grad_mod = compute_gradient_magnitude(modified)
    diff = (grad_orig - grad_mod) ** 2
    if land_mask is not None:
        diff = diff * land_mask.astype(np.float64)
        total = np.sum(land_mask)
        if total < 1:
            return 0.0
        return float(np.sum(diff) / total)
    return float(np.mean(diff))


def restore_gradient(
    original: np.ndarray,
    modified: np.ndarray,
    strength: float = 0.5,
    land_mask: np.ndarray | None = None,
) -> np.ndarray:
    grad_orig = compute_gradient_magnitude(original)
    grad_mod = compute_gradient_magnitude(modified)
    deficit = grad_orig - grad_mod
    correction = np.where(deficit > 0, deficit * strength, 0.0)
    dzdx_orig = sobel(original.astype(np.float64), axis=1) / 8.0
    dzdy_orig = sobel(original.astype(np.float64), axis=0) / 8.0
    slope_mag = np.sqrt(dzdx_orig ** 2 + dzdy_orig ** 2) + 1e-10
    direction_x = dzdx_orig / slope_mag
    direction_y = dzdy_orig / slope_mag
    from scipy.ndimage import uniform_filter
    boost_x = uniform_filter(correction * direction_x, size=3, mode="nearest")
    boost_y = uniform_filter(correction * direction_y, size=3, mode="nearest")
    boost = (boost_x + boost_y)
    result = modified.astype(np.float64) + boost
    if land_mask is not None:
        result = np.where(land_mask, result, modified.astype(np.float64))
    return result.astype(np.float32)


def compute_coastline_fractal_dimension(land_mask: np.ndarray, max_box_size: int | None = None) -> float:
    h, w = land_mask.shape
    boundary = _extract_coastline(land_mask)
    if len(boundary) < 10:
        return 1.0
    if max_box_size is None:
        max_box_size = min(h, w) // 4
    sizes = []
    counts = []
    for s in [2, 4, 8, 16, 32, 64]:
        if s > max_box_size:
            break
        count = _box_count(boundary, s)
        if count > 0:
            sizes.append(s)
            counts.append(count)
    if len(sizes) < 2:
        return 1.0
    log_sizes = np.log(np.array(sizes, dtype=np.float64))
    log_counts = np.log(np.array(counts, dtype=np.float64))
    slope, _ = np.polyfit(log_sizes, log_counts, 1)
    return float(-slope)


def compute_coastline_roughness_exponent(land_mask: np.ndarray) -> float:
    boundary = _extract_coastline(land_mask)
    if len(boundary) < 20:
        return 1.0
    n = len(boundary)
    max_window = min(n // 4, 200)
    if max_window < 4:
        return 1.0
    windows = [4, 8, 16, 32, 64, 128]
    log_w = []
    log_v = []
    for w_size in windows:
        if w_size > max_window:
            break
        variances = []
        for start in range(0, n - w_size, max(w_size // 2, 1)):
            segment = boundary[start:start + w_size]
            if len(segment) < w_size:
                continue
            dists = _cumulative_distance(segment)
            if len(dists) < 2:
                continue
            local_var = np.var(np.diff(dists))
            variances.append(local_var)
        if variances:
            mean_var = np.mean(variances)
            if mean_var > 0:
                log_w.append(np.log(w_size))
                log_v.append(np.log(mean_var))
    if len(log_w) < 2:
        return 1.0
    slope, _ = np.polyfit(log_w, log_v, 1)
    alpha = slope / 2.0
    return float(np.clip(alpha, 0.5, 1.5))


def _extract_coastline(land_mask: np.ndarray) -> np.ndarray:
    from scipy.ndimage import binary_erosion, generate_binary_structure
    struct = generate_binary_structure(2, 2)
    eroded = binary_erosion(land_mask.astype(bool), structure=struct)
    boundary = land_mask.astype(bool) & ~eroded
    ys, xs = np.where(boundary)
    if len(ys) == 0:
        return np.array([]).reshape(0, 2)
    points = np.column_stack([xs, ys]).astype(np.float64)
    return points


def _box_count(points: np.ndarray, box_size: int) -> int:
    if len(points) == 0:
        return 0
    grid_x = (points[:, 0] / box_size).astype(int)
    grid_y = (points[:, 1] / box_size).astype(int)
    occupied = set(zip(grid_x.tolist(), grid_y.tolist()))
    return len(occupied)


def _cumulative_distance(points: np.ndarray) -> np.ndarray:
    if len(points) < 2:
        return np.array([0.0])
    diffs = np.diff(points, axis=0)
    dists = np.sqrt(np.sum(diffs ** 2, axis=1))
    return np.concatenate([[0.0], np.cumsum(dists)])
