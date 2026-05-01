from __future__ import annotations

from collections import deque
from typing import List, Optional, Tuple

import numpy as np

from app.core.voronoi_grid import VoronoiDiagram


_BLOB_POWER_MAP = {
    1000: 0.93, 2000: 0.95, 5000: 0.97, 10000: 0.98,
    20000: 0.99, 30000: 0.991, 40000: 0.993, 50000: 0.994,
    60000: 0.995, 70000: 0.9955, 80000: 0.996, 90000: 0.9964,
    100000: 0.9973,
}

_LINE_POWER_MAP = {
    1000: 0.75, 2000: 0.77, 5000: 0.79, 10000: 0.81,
    20000: 0.82, 30000: 0.83, 40000: 0.84, 50000: 0.86,
    60000: 0.87, 70000: 0.88, 80000: 0.91, 90000: 0.92,
    100000: 0.93,
}


def _get_blob_power(n_cells: int) -> float:
    if n_cells in _BLOB_POWER_MAP:
        return _BLOB_POWER_MAP[n_cells]
    keys = sorted(_BLOB_POWER_MAP.keys())
    for k in keys:
        if n_cells <= k:
            return _BLOB_POWER_MAP[k]
    return _BLOB_POWER_MAP[keys[-1]]


def _get_line_power(n_cells: int) -> float:
    if n_cells in _LINE_POWER_MAP:
        return _LINE_POWER_MAP[n_cells]
    keys = sorted(_LINE_POWER_MAP.keys())
    for k in keys:
        if n_cells <= k:
            return _LINE_POWER_MAP[k]
    return _LINE_POWER_MAP[keys[-1]]


def _find_nearest_cell(x: float, y: float, voronoi: VoronoiDiagram) -> int:
    dists = (voronoi.points[:, 0] - x) ** 2 + (voronoi.points[:, 1] - y) ** 2
    return int(np.argmin(dists))


class HeightmapTools:
    def __init__(self, voronoi: VoronoiDiagram):
        self.voronoi = voronoi
        self.heights: np.ndarray = np.zeros(voronoi.points_n, dtype=np.float64)
        self.blob_power = _get_blob_power(voronoi.points_n)
        self.line_power = _get_line_power(voronoi.points_n)

    def set_heights(self, heights: np.ndarray) -> None:
        self.heights = heights.copy()

    def get_heights(self) -> np.ndarray:
        return self.heights.copy()

    def add_hill(
        self,
        count: int = 1,
        height: float = 50.0,
        cx: Optional[float] = None,
        cy: Optional[float] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        rng = rng or np.random.default_rng()
        w, h = self.voronoi.width, self.voronoi.height

        for _ in range(count):
            start_x = cx if cx is not None else rng.uniform(0, w)
            start_y = cy if cy is not None else rng.uniform(0, h)
            start = _find_nearest_cell(start_x, start_y, self.voronoi)

            limit = 0
            while self.heights[start] + height > 90 and limit < 50:
                start_x = rng.uniform(0, w)
                start_y = rng.uniform(0, h)
                start = _find_nearest_cell(start_x, start_y, self.voronoi)
                limit += 1

            change = np.zeros(self.voronoi.points_n, dtype=np.float64)
            change[start] = height
            queue = deque([start])

            while queue:
                q = queue.popleft()
                for c in self.voronoi.get_neighbor_cells(q):
                    if c < 0 or c >= self.voronoi.points_n or change[c] > 0:
                        continue
                    change[c] = change[q] ** self.blob_power * rng.uniform(0.9, 1.1)
                    if change[c] > 1:
                        queue.append(c)

            self.heights = np.clip(self.heights + change, 0, 100)

    def add_pit(
        self,
        count: int = 1,
        depth: float = 30.0,
        cx: Optional[float] = None,
        cy: Optional[float] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        rng = rng or np.random.default_rng()
        w, h = self.voronoi.width, self.voronoi.height

        for _ in range(count):
            start_x = cx if cx is not None else rng.uniform(0, w)
            start_y = cy if cy is not None else rng.uniform(0, h)
            start = _find_nearest_cell(start_x, start_y, self.voronoi)

            limit = 0
            while self.heights[start] < 20 and limit < 50:
                start_x = rng.uniform(0, w)
                start_y = rng.uniform(0, h)
                start = _find_nearest_cell(start_x, start_y, self.voronoi)
                limit += 1

            used = np.zeros(self.voronoi.points_n, dtype=bool)
            used[start] = True
            queue = deque([start])
            current_depth = depth

            while queue:
                q = queue.popleft()
                current_depth = current_depth ** self.blob_power * rng.uniform(0.9, 1.1)
                if current_depth < 1:
                    break

                for c in self.voronoi.get_neighbor_cells(q):
                    if c < 0 or c >= self.voronoi.points_n or used[c]:
                        continue
                    self.heights[c] = max(0, self.heights[c] - current_depth * rng.uniform(0.9, 1.1))
                    used[c] = True
                    queue.append(c)

    def add_range(
        self,
        count: int = 1,
        height: float = 60.0,
        start_pos: Optional[Tuple[float, float]] = None,
        end_pos: Optional[Tuple[float, float]] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        rng = rng or np.random.default_rng()
        w, h = self.voronoi.width, self.voronoi.height

        for _ in range(count):
            used = np.zeros(self.voronoi.points_n, dtype=bool)

            if start_pos is not None:
                sx, sy = start_pos
            else:
                sx = rng.uniform(0, w)
                sy = rng.uniform(0, h)

            start_cell = _find_nearest_cell(sx, sy, self.voronoi)

            if end_pos is not None:
                ex, ey = end_pos
            else:
                dist = 0
                limit = 0
                while (dist < w / 8 or dist > w / 3) and limit < 50:
                    ex = rng.uniform(w * 0.1, w * 0.9)
                    ey = rng.uniform(h * 0.15, h * 0.85)
                    dist = abs(ey - sy) + abs(ex - sx)
                    limit += 1

            end_cell = _find_nearest_cell(ex, ey, self.voronoi)

            ridge = self._find_ridge_path(start_cell, end_cell, used, rng)

            current_h = height
            queue = list(ridge)
            iteration = 0

            while queue and current_h >= 2:
                frontier = queue[:]
                queue = []
                iteration += 1

                for cell_id in frontier:
                    self.heights[cell_id] = min(100, self.heights[cell_id] + current_h * rng.uniform(0.85, 1.15))

                current_h = current_h ** self.line_power - 1

                for f in frontier:
                    for c in self.voronoi.get_neighbor_cells(f):
                        if 0 <= c < self.voronoi.points_n and not used[c]:
                            queue.append(c)
                            used[c] = True

            for idx, cur in enumerate(ridge):
                if idx % 6 != 0:
                    continue
                for _ in range(iteration):
                    neighbors = self.voronoi.get_neighbor_cells(cur)
                    valid = [n for n in neighbors if 0 <= n < self.voronoi.points_n]
                    if not valid:
                        break
                    min_cell = min(valid, key=lambda n: self.heights[n])
                    self.heights[min_cell] = (self.heights[cur] * 2 + self.heights[min_cell]) / 3
                    cur = min_cell

    def add_trough(
        self,
        count: int = 1,
        depth: float = 40.0,
        start_pos: Optional[Tuple[float, float]] = None,
        end_pos: Optional[Tuple[float, float]] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        rng = rng or np.random.default_rng()
        w, h = self.voronoi.width, self.voronoi.height

        for _ in range(count):
            used = np.zeros(self.voronoi.points_n, dtype=bool)

            if start_pos is not None:
                sx, sy = start_pos
            else:
                sx = rng.uniform(0, w)
                sy = rng.uniform(0, h)

            start_cell = _find_nearest_cell(sx, sy, self.voronoi)

            if end_pos is not None:
                ex, ey = end_pos
            else:
                dist = 0
                limit = 0
                while (dist < w / 8 or dist > w / 2) and limit < 50:
                    ex = rng.uniform(w * 0.1, w * 0.9)
                    ey = rng.uniform(h * 0.15, h * 0.85)
                    dist = abs(ey - sy) + abs(ex - sx)
                    limit += 1

            end_cell = _find_nearest_cell(ex, ey, self.voronoi)

            ridge = self._find_ridge_path(start_cell, end_cell, used, rng)

            current_h = depth
            queue = list(ridge)
            iteration = 0

            while queue and current_h >= 2:
                frontier = queue[:]
                queue = []
                iteration += 1

                for cell_id in frontier:
                    self.heights[cell_id] = max(0, self.heights[cell_id] - current_h * rng.uniform(0.85, 1.15))

                current_h = current_h ** self.line_power - 1

                for f in frontier:
                    for c in self.voronoi.get_neighbor_cells(f):
                        if 0 <= c < self.voronoi.points_n and not used[c]:
                            queue.append(c)
                            used[c] = True

            for idx, cur in enumerate(ridge):
                if idx % 6 != 0:
                    continue
                for _ in range(iteration):
                    neighbors = self.voronoi.get_neighbor_cells(cur)
                    valid = [n for n in neighbors if 0 <= n < self.voronoi.points_n]
                    if not valid:
                        break
                    max_cell = max(valid, key=lambda n: self.heights[n])
                    self.heights[max_cell] = (self.heights[cur] * 2 + self.heights[max_cell]) / 3
                    cur = max_cell

    def add_strait(
        self,
        width: int = 3,
        direction: str = "vertical",
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        rng = rng or np.random.default_rng()
        w, h = self.voronoi.width, self.voronoi.height
        used = np.zeros(self.voronoi.points_n, dtype=bool)

        vert = direction == "vertical"
        if vert:
            sx = int(rng.uniform(w * 0.3, w * 0.7))
            sy = 5
            ex = int(rng.uniform(w * 0.3, w * 0.9))
            ey = h - 5
        else:
            sx = 5
            sy = int(rng.uniform(h * 0.3, h * 0.7))
            ex = w - 5
            ey = int(rng.uniform(h * 0.3, h * 0.9))

        start = _find_nearest_cell(sx, sy, self.voronoi)
        end = _find_nearest_cell(ex, ey, self.voronoi)

        path = self._find_ridge_path(start, end, used, rng, mark_used=False)

        step = 0.1 / max(width, 1)
        for i in range(width):
            exp = 0.9 - step * width
            next_layer: List[int] = []
            for r in path:
                for c in self.voronoi.get_neighbor_cells(r):
                    if 0 <= c < self.voronoi.points_n and not used[c]:
                        used[c] = True
                        next_layer.append(c)
                        self.heights[c] = self.heights[c] ** exp
                        if self.heights[c] > 100:
                            self.heights[c] = 5
            path = next_layer

    def smooth(self, fr: float = 2.0) -> None:
        result = np.zeros_like(self.heights)
        for i in range(self.voronoi.points_n):
            neighbors = self.voronoi.get_neighbor_cells(i)
            valid = [n for n in neighbors if 0 <= n < self.voronoi.points_n]
            if valid:
                neighbor_mean = np.mean(self.heights[valid])
            else:
                neighbor_mean = self.heights[i]
            result[i] = (self.heights[i] * (fr - 1) + neighbor_mean) / fr
        self.heights = np.clip(result, 0, 100)

    def mask(self, power: float = 1.0) -> None:
        w, h = self.voronoi.width, self.voronoi.height
        fr = abs(power) if power else 1.0

        for i in range(self.voronoi.points_n):
            x, y = self.voronoi.points[i]
            nx = 2.0 * x / w - 1.0
            ny = 2.0 * y / h - 1.0
            distance = (1 - nx ** 2) * (1 - ny ** 2)
            if power < 0:
                distance = 1 - distance
            masked = self.heights[i] * distance
            self.heights[i] = (self.heights[i] * (fr - 1) + masked) / fr

        self.heights = np.clip(self.heights, 0, 100)

    def modify(
        self,
        add: float = 0,
        mult: float = 1.0,
        power: Optional[float] = None,
        land_only: bool = True,
    ) -> None:
        sea_level = 20.0
        for i in range(self.voronoi.points_n):
            h = self.heights[i]
            if land_only and h < sea_level:
                continue

            if add != 0:
                h = max(h + add, sea_level) if land_only else h + add
            if mult != 1.0:
                h = (h - sea_level) * mult + sea_level if land_only else h * mult
            if power is not None:
                h = (h - sea_level) ** power + sea_level if land_only else h ** power

            self.heights[i] = np.clip(h, 0, 100)

    def _find_ridge_path(
        self,
        start: int,
        end: int,
        used: np.ndarray,
        rng: np.random.Generator,
        mark_used: bool = True,
    ) -> List[int]:
        path = [start]
        cur = start
        if mark_used:
            used[cur] = True
        points = self.voronoi.points

        limit = 0
        while cur != end and limit < self.voronoi.points_n:
            min_dist = float("inf")
            next_cell = cur
            for e in self.voronoi.get_neighbor_cells(cur):
                if e < 0 or e >= self.voronoi.points_n:
                    continue
                if mark_used and used[e]:
                    continue
                diff = (points[end, 0] - points[e, 0]) ** 2 + (points[end, 1] - points[e, 1]) ** 2
                if rng.random() > 0.85:
                    diff /= 2
                if diff < min_dist:
                    min_dist = diff
                    next_cell = e

            if next_cell == cur:
                break
            path.append(next_cell)
            if mark_used:
                used[next_cell] = True
            cur = next_cell
            limit += 1

        return path
