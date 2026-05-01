from __future__ import annotations

from typing import List, Tuple

import numpy as np
from scipy.ndimage import label, find_objects


def extract_contours(
    elevation: np.ndarray,
    levels: List[float] | None = None,
    wrap_longitude: bool = False,
) -> List[List[Tuple[float, float]]]:
    if levels is None:
        elev_min, elev_max = elevation.min(), elevation.max()
        if elev_max - elev_min < 0.01:
            return []
        n_levels = 10
        levels = list(np.linspace(elev_min + 0.05, elev_max - 0.05, n_levels))

    all_contours: List[List[Tuple[float, float]]] = []

    for level in levels:
        binary = elevation >= level
        if wrap_longitude:
            padded = np.concatenate([binary[:, -1:], binary, binary[:, :1]], axis=1)
            labeled, n_features = label(padded)
            labeled = labeled[:, 1:-1]
        else:
            labeled, n_features = label(binary)

        for feat_id in range(1, n_features + 1):
            mask = labeled == feat_id
            if not np.any(mask):
                continue

            rows, cols = np.where(mask)
            if len(rows) < 4:
                continue

            boundary = _extract_boundary(mask, wrap_longitude)
            if len(boundary) >= 4:
                all_contours.append(boundary)

    return all_contours


def _extract_boundary(
    mask: np.ndarray,
    wrap_longitude: bool = False,
) -> List[Tuple[float, float]]:
    h, w = mask.shape
    boundary_points: List[Tuple[float, float]] = []

    eroded = np.zeros_like(mask)
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dy == 0 and dx == 0:
                continue
            shifted = np.roll(np.roll(mask, -dy, axis=0), -dx, axis=1)
            eroded |= shifted
    eroded = mask & eroded
    boundary = mask & ~eroded

    if not np.any(boundary):
        return boundary_points

    rows, cols = np.where(boundary)
    if len(rows) == 0:
        return boundary_points

    start_idx = 0
    min_col = cols.min()
    candidates = np.where((cols == min_col) & boundary)[0]
    if len(candidates) > 0:
        start_idx = candidates[np.argmin(rows[candidates])]

    visited = set()
    current = start_idx
    order = [current]
    visited.add(current)

    for _ in range(len(rows) - 1):
        cr, cc = rows[current], cols[current]
        best_next = -1
        best_dist = float("inf")

        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                nr, nc = cr + dy, cc + dx
                if wrap_longitude:
                    nc = nc % w
                if nr < 0 or nr >= h:
                    continue
                if not boundary[nr, nc]:
                    continue

                matches = np.where((rows == nr) & (cols == nc))[0]
                for m in matches:
                    if m not in visited:
                        dist = dy * dy + dx * dx
                        if dist < best_dist:
                            best_dist = dist
                            best_next = m

        if best_next == -1:
            break
        visited.add(best_next)
        order.append(best_next)
        current = best_next

    for idx in order:
        boundary_points.append((float(cols[idx]), float(rows[idx])))

    return boundary_points


def extract_bathymetry_contours(
    elevation: np.ndarray,
    sea_level: float = 0.0,
    n_depth_levels: int = 5,
    wrap_longitude: bool = False,
) -> List[List[Tuple[float, float]]]:
    water = elevation < sea_level
    if not np.any(water):
        return []

    water_min = elevation[water].min()
    depth_range = sea_level - water_min
    if depth_range < 0.01:
        return []

    levels = [sea_level - depth_range * (i + 1) / (n_depth_levels + 1) for i in range(n_depth_levels)]

    return extract_contours(elevation, levels=levels, wrap_longitude=wrap_longitude)
