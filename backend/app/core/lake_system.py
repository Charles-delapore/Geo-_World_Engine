from __future__ import annotations

from collections import deque
from typing import List, Optional, Tuple

import numpy as np
from scipy.ndimage import label


def penman_evaporation(temperature: float, elevation: float) -> float:
    if temperature < -40:
        return 0.0
    height_mm = elevation * 1000.0
    evap = ((700.0 * (temperature + 0.006 * height_mm)) / 50.0 + 75.0) / (80.0 - temperature)
    return max(0.0, evap)


def penman_evaporation_array(
    temperature: np.ndarray,
    elevation: np.ndarray,
) -> np.ndarray:
    height_mm = elevation * 1000.0
    with np.errstate(divide="ignore", invalid="ignore"):
        evap = ((700.0 * (temperature + 0.006 * height_mm)) / 50.0 + 75.0) / (80.0 - temperature)
    evap = np.where(temperature < -40, 0.0, evap)
    evap = np.clip(evap, 0.0, None)
    return evap.astype(np.float32)


def detect_closed_lakes(
    elevation: np.ndarray,
    sea_level: float = 0.0,
    min_lake_area: int = 10,
) -> List[dict]:
    h, w = elevation.shape
    water_mask = elevation < sea_level
    land_mask = elevation >= sea_level

    if not np.any(water_mask) or not np.any(land_mask):
        return []

    labeled, n_features = label(water_mask)
    slices = find_objects(labeled)

    closed_lakes = []

    for feat_id in range(1, n_features + 1):
        if slices[feat_id - 1] is None:
            continue

        lake_mask = labeled == feat_id
        lake_area = int(np.sum(lake_mask))

        if lake_area < min_lake_area:
            continue

        is_border = _is_lake_on_border(lake_mask)
        if is_border:
            continue

        is_closed = _check_closed_basin(lake_mask, land_mask, elevation, sea_level)

        if is_closed:
            lake_cells = np.argwhere(lake_mask)
            mean_elev = float(elevation[lake_mask].mean())
            min_elev = float(elevation[lake_mask].min())

            closed_lakes.append({
                "feature_id": feat_id,
                "area": lake_area,
                "mean_elevation": mean_elev,
                "min_elevation": min_elev,
                "cells": lake_cells.tolist(),
                "closed": True,
            })

    return closed_lakes


def _is_lake_on_border(lake_mask: np.ndarray) -> bool:
    return bool(
        np.any(lake_mask[0, :])
        or np.any(lake_mask[-1, :])
        or np.any(lake_mask[:, 0])
        or np.any(lake_mask[:, -1])
    )


def _check_closed_basin(
    lake_mask: np.ndarray,
    land_mask: np.ndarray,
    elevation: np.ndarray,
    sea_level: float,
) -> bool:
    h, w = elevation.shape

    dilated = np.zeros_like(lake_mask)
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dy == 0 and dx == 0:
                continue
            shifted = np.roll(np.roll(lake_mask, -dy, axis=0), -dx, axis=1)
            dilated |= shifted
    shoreline = dilated & land_mask

    if not np.any(shoreline):
        return False

    shore_elevations = elevation[shoreline]
    min_shore_elev = float(shore_elevations.min())

    visited = np.zeros((h, w), dtype=bool)
    queue = deque()

    shore_rows, shore_cols = np.where(shoreline)
    for i in range(len(shore_rows)):
        r, c = shore_rows[i], shore_cols[i]
        if elevation[r, c] <= min_shore_elev + 0.001:
            visited[r, c] = True
            queue.append((r, c))

    while queue:
        r, c = queue.popleft()

        if r == 0 or r == h - 1 or c == 0 or c == w - 1:
            return False

        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                nr, nc = r + dy, c + dx
                if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc]:
                    if elevation[nr, nc] >= sea_level:
                        visited[nr, nc] = True
                        queue.append((nr, nc))

    return True


def compute_lake_properties(
    elevation: np.ndarray,
    temperature: Optional[np.ndarray] = None,
    precipitation: Optional[np.ndarray] = None,
    sea_level: float = 0.0,
    min_lake_area: int = 10,
) -> List[dict]:
    h, w = elevation.shape
    water_mask = elevation < sea_level

    if not np.any(water_mask):
        return []

    labeled, n_features = label(water_mask)
    slices = find_objects(labeled)

    lakes = []

    for feat_id in range(1, n_features + 1):
        if slices[feat_id - 1] is None:
            continue

        lake_mask = labeled == feat_id
        lake_area = int(np.sum(lake_mask))

        if lake_area < min_lake_area:
            continue

        is_border = _is_lake_on_border(lake_mask)

        mean_elev = float(elevation[lake_mask].mean())

        lake_temp = 0.0
        lake_precip = 0.0
        evaporation = 0.0

        if temperature is not None:
            lake_temp = float(temperature[lake_mask].mean())
            evaporation = penman_evaporation(lake_temp, mean_elev)

        if precipitation is not None:
            lake_precip = float(precipitation[lake_mask].mean())

        is_closed = False
        if not is_border:
            is_closed = _check_closed_basin(lake_mask, elevation >= sea_level, elevation, sea_level)

        lake_type = "ocean" if is_border else ("closed_lake" if is_closed else "open_lake")

        lakes.append({
            "feature_id": feat_id,
            "area": lake_area,
            "mean_elevation": mean_elev,
            "temperature": lake_temp,
            "precipitation": lake_precip,
            "evaporation": evaporation,
            "type": lake_type,
            "closed": is_closed,
        })

    return lakes
