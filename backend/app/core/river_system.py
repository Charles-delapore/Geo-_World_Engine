from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from app.core.voronoi_grid import VoronoiDiagram


@dataclass
class River:
    river_id: int
    source: int
    mouth: int
    parent: int = 0
    basin: int = 0
    length: float = 0.0
    discharge: float = 0.0
    width: float = 0.0
    width_factor: float = 1.0
    source_width: float = 0.0
    cells: List[int] = field(default_factory=list)
    points: List[Tuple[float, float]] = field(default_factory=list)


@dataclass
class Lake:
    lake_id: int
    cells: List[int] = field(default_factory=list)
    out_cell: int = -1
    flux: float = 0.0
    evaporation: float = 0.0
    river: int = 0
    inlets: List[int] = field(default_factory=list)


class RiverSystem:
    FLUX_FACTOR = 500.0
    LENGTH_FACTOR = 200.0
    MAX_DOWNCUT = 5
    MIN_FLUX_TO_FORM_RIVER = 30

    def __init__(self, voronoi: VoronoiDiagram):
        self.voronoi = voronoi
        self.rivers: List[River] = []
        self.lakes: List[Lake] = []
        self.flux: np.ndarray = np.zeros(voronoi.points_n, dtype=np.float64)
        self.river_ids: np.ndarray = np.zeros(voronoi.points_n, dtype=np.int32)
        self.confluence: np.ndarray = np.zeros(voronoi.points_n, dtype=np.float64)

    def generate(
        self,
        elevation: np.ndarray,
        precipitation: np.ndarray,
        temperature: Optional[np.ndarray] = None,
        allow_erosion: bool = True,
    ) -> None:
        h = elevation.astype(np.float64).copy()
        n = len(h)

        land_mask = h >= 0.2
        land_indices = np.where(land_mask)[0]
        land_sorted = land_indices[np.argsort(-h[land_indices])]

        self.flux[:] = 0
        self.river_ids[:] = 0
        self.confluence[:] = 0
        self.rivers.clear()
        self.lakes.clear()

        precip_norm = precipitation / (precipitation.max() + 1e-10)
        cells_number_modifier = (self.voronoi.points_n / 10000) ** 0.25

        for i in range(n):
            if land_mask[i]:
                self.flux[i] += precip_norm[i] * 100 / cells_number_modifier

        river_next = 1
        rivers_data: dict[int, list[int]] = {}
        river_parents: dict[int, int] = {}

        def add_cell_to_river(cell_id: int, rid: int) -> None:
            if rid not in rivers_data:
                rivers_data[rid] = [cell_id]
            else:
                rivers_data[rid].append(cell_id)

        def flow_down(to_cell: int, from_flux: float, river: int) -> None:
            to_flux = self.flux[to_cell] - self.confluence[to_cell]
            to_river = self.river_ids[to_cell]

            if to_river:
                if from_flux > to_flux:
                    self.confluence[to_cell] += self.flux[to_cell]
                    if h[to_cell] >= 0.2:
                        river_parents[to_river] = river
                    self.river_ids[to_cell] = river
                else:
                    self.confluence[to_cell] += from_flux
                    if h[to_cell] >= 0.2:
                        river_parents[river] = to_river
            else:
                self.river_ids[to_cell] = river

            if h[to_cell] < 0.2:
                pass
            else:
                self.flux[to_cell] += from_flux

            add_cell_to_river(to_cell, river)

        for i in land_sorted:
            if self.voronoi.is_border(i) and self.river_ids[i]:
                add_cell_to_river(-1, self.river_ids[i])
                continue

            neighbors = self.voronoi.get_neighbor_cells(i)
            if not neighbors:
                continue

            valid_neighbors = [c for c in neighbors if 0 <= c < n]
            if not valid_neighbors:
                continue

            min_neighbor = min(valid_neighbors, key=lambda c: h[c])

            if h[i] <= h[min_neighbor]:
                continue

            if self.flux[i] < self.MIN_FLUX_TO_FORM_RIVER:
                if h[min_neighbor] >= 0.2:
                    self.flux[min_neighbor] += self.flux[i]
                continue

            if not self.river_ids[i]:
                self.river_ids[i] = river_next
                add_cell_to_river(i, river_next)
                river_next += 1

            flow_down(min_neighbor, self.flux[i], self.river_ids[i])

        default_width_factor = 1.0 / ((self.voronoi.points_n / 10000) ** 0.25)
        main_stem_width_factor = default_width_factor * 1.2

        for key, river_cells in rivers_data.items():
            if len(river_cells) < 3:
                continue

            river_id = int(key)
            for cell in river_cells:
                if cell < 0 or cell >= n or h[cell] < 0.2:
                    continue
                if self.river_ids[cell]:
                    self.confluence[cell] = 1
                else:
                    self.river_ids[cell] = river_id

            source = river_cells[0]
            mouth = river_cells[-2] if len(river_cells) >= 2 else river_cells[-1]
            parent = river_parents.get(key, 0)

            width_factor = (
                main_stem_width_factor
                if not parent or parent == river_id
                else default_width_factor
            )
            discharge = self.flux[mouth] if 0 <= mouth < n else 0
            source_flux = self.flux[source] if 0 <= source < n else 0
            source_width = self._get_source_width(source_flux)
            river_length = self._get_approximate_length(river_cells)
            width = self._get_width(
                self._get_offset(
                    discharge,
                    len(river_cells),
                    width_factor,
                    source_width,
                )
            )

            self.rivers.append(
                River(
                    river_id=river_id,
                    source=source,
                    mouth=mouth,
                    parent=parent,
                    basin=parent if parent else river_id,
                    length=river_length,
                    discharge=discharge,
                    width=width,
                    width_factor=width_factor,
                    source_width=source_width,
                    cells=river_cells,
                )
            )

        if allow_erosion:
            self._downcut_rivers(h)

    def _downcut_rivers(self, h: np.ndarray) -> None:
        for i in range(len(h)):
            if h[i] < 0.35 or not self.flux[i]:
                continue
            neighbors = self.voronoi.get_neighbor_cells(i)
            higher_cells = [c for c in neighbors if 0 <= c < len(h) and h[c] > h[i]]
            if not higher_cells:
                continue
            higher_flux = sum(self.flux[c] for c in higher_cells) / len(higher_cells)
            if not higher_flux:
                continue
            downcut = int(self.flux[i] / higher_flux)
            if downcut:
                h[i] -= min(downcut, self.MAX_DOWNCUT) / 100.0

    def _get_source_width(self, source_flux: float) -> float:
        return max(0.1, source_flux / self.FLUX_FACTOR * 0.3)

    def _get_offset(
        self,
        flux: float,
        point_index: int,
        width_factor: float,
        starting_width: float,
    ) -> float:
        base = flux / self.FLUX_FACTOR * width_factor
        progression = [1, 1, 2, 3, 5, 8, 13, 21, 34]
        step = min(point_index, len(progression) - 1)
        return starting_width + (base - starting_width) * (progression[step] / 34.0)

    def _get_width(self, offset: float) -> float:
        return max(0.1, offset)

    def _get_approximate_length(self, cells: List[int]) -> float:
        if len(cells) < 2:
            return 0.0
        total = 0.0
        for i in range(len(cells) - 1):
            c1, c2 = cells[i], cells[i + 1]
            if c1 < 0 or c2 < 0 or c1 >= len(self.voronoi.points) or c2 >= len(self.voronoi.points):
                continue
            p1 = self.voronoi.points[c1]
            p2 = self.voronoi.points[c2]
            total += np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
        return total / self.LENGTH_FACTOR

    def add_meandering(
        self,
        river_cells: List[int],
        meandering: float = 0.3,
    ) -> List[Tuple[float, float]]:
        if len(river_cells) < 2:
            return []

        points = []
        for cell_id in river_cells:
            if 0 <= cell_id < len(self.voronoi.points):
                points.append(tuple(self.voronoi.points[cell_id]))

        if len(points) < 3:
            return points

        result = [points[0]]
        for i in range(1, len(points) - 1):
            px, py = points[i]
            prev_x, prev_y = points[i - 1]
            next_x, next_y = points[i + 1]

            dx = next_x - prev_x
            dy = next_y - prev_y
            length = np.sqrt(dx * dx + dy * dy) + 1e-10

            perp_x = -dy / length
            perp_y = dx / length

            offset = (np.random.random() - 0.5) * 2 * meandering * length * 0.3
            new_x = px + perp_x * offset
            new_y = py + perp_y * offset

            result.append((new_x, new_y))

        result.append(points[-1])
        return result

    def get_river_mask(self, shape: Tuple[int, int]) -> np.ndarray:
        mask = np.zeros(shape, dtype=np.float32)
        h, w = shape

        for river in self.rivers:
            for cell_id in river.cells:
                if 0 <= cell_id < len(self.voronoi.points):
                    px, py = self.voronoi.points[cell_id]
                    ix = int(px / self.voronoi.width * w)
                    iy = int(py / self.voronoi.height * h)
                    if 0 <= iy < h and 0 <= ix < w:
                        river_width = max(1, int(river.width * 2))
                        for dy in range(-river_width, river_width + 1):
                            for dx in range(-river_width, river_width + 1):
                                ny, nx = iy + dy, ix + dx
                                if 0 <= ny < h and 0 <= nx < w:
                                    dist = np.sqrt(dx * dx + dy * dy)
                                    if dist <= river_width:
                                        mask[ny, nx] = max(mask[ny, nx], 1.0 - dist / (river_width + 1))

        return mask

    def get_strahler_order(self, river_id: int) -> int:
        children = [r for r in self.rivers if r.parent == river_id]
        if not children:
            return 1
        child_orders = sorted(
            [self.get_strahler_order(r.river_id) for r in children], reverse=True
        )
        if len(child_orders) >= 2 and child_orders[0] == child_orders[1]:
            return child_orders[0] + 1
        return child_orders[0]
