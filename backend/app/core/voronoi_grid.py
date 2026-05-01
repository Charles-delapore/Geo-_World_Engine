from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

try:
    from scipy.spatial import Delaunay

    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


Point = Tuple[float, float]


@dataclass
class VoronoiCells:
    v: List[List[int]] = field(default_factory=list)
    c: List[List[int]] = field(default_factory=list)
    b: List[int] = field(default_factory=list)
    i: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int32))


@dataclass
class VoronoiVertices:
    p: List[Point] = field(default_factory=list)
    v: List[List[int]] = field(default_factory=list)
    c: List[List[int]] = field(default_factory=list)


class VoronoiDiagram:
    def __init__(
        self,
        points: np.ndarray,
        width: int,
        height: int,
    ):
        self.points = points
        self.points_n = len(points)
        self.width = width
        self.height = height
        self.cells = VoronoiCells()
        self.vertices = VoronoiVertices()

        if _HAS_SCIPY:
            self._build_from_scipy()
        else:
            self._build_simple()

    def _build_from_scipy(self) -> None:
        tri = Delaunay(self.points)
        neighbor_dict: dict[int, set[int]] = {
            i: set() for i in range(self.points_n)
        }
        vertex_points: dict[int, Point] = {}
        vertex_neighbors: dict[int, list[int]] = {}
        vertex_cells: dict[int, list[int]] = {}

        for simplex in tri.simplices:
            cx, cy = self._circumcenter(
                self.points[simplex[0]],
                self.points[simplex[1]],
                self.points[simplex[2]],
            )
            t_id = int(
                simplex[0] * self.points_n * self.points_n
                + simplex[1] * self.points_n
                + simplex[2]
            ) % (10**7)

            vertex_points[t_id] = (cx, cy)
            vertex_neighbors[t_id] = []
            vertex_cells[t_id] = [int(s) for s in simplex]

            for i in range(3):
                for j in range(3):
                    if i != j:
                        neighbor_dict[simplex[i]].add(simplex[j])

        for i in range(self.points_n):
            self.cells.v.append([])
            self.cells.c.append(sorted(neighbor_dict[i]))
            px, py = self.points[i]
            is_border = px < 2 or px > self.width - 2 or py < 2 or py > self.height - 2
            self.cells.b.append(1 if is_border else 0)

        self.vertices.p = list(vertex_points.values())
        self.vertices.v = list(vertex_neighbors.values())
        self.vertices.c = list(vertex_cells.values())
        self.cells.i = np.arange(self.points_n, dtype=np.int32)

    def _build_simple(self) -> None:
        grid_size = int(np.sqrt(self.points_n))
        cell_w = self.width / grid_size
        cell_h = self.height / grid_size

        for i in range(self.points_n):
            px, py = self.points[i]
            gx = int(px / cell_w)
            gy = int(py / cell_h)
            neighbors = []
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = gx + dx, gy + dy
                    if 0 <= nx < grid_size and 0 <= ny < grid_size:
                        ni = ny * grid_size + nx
                        if ni < self.points_n:
                            neighbors.append(ni)
            self.cells.v.append([])
            self.cells.c.append(neighbors)
            is_border = gx == 0 or gx == grid_size - 1 or gy == 0 or gy == grid_size - 1
            self.cells.b.append(1 if is_border else 0)

        self.cells.i = np.arange(self.points_n, dtype=np.int32)

    @staticmethod
    def _circumcenter(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> Point:
        ax, ay = a
        bx, by = b
        cx, cy = c
        ad = ax * ax + ay * ay
        bd = bx * bx + by * by
        cd = cx * cx + cy * cy
        D = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
        if abs(D) < 1e-10:
            return ((ax + bx + cx) / 3.0, (ay + by + cy) / 3.0)
        ux = (ad * (by - cy) + bd * (cy - ay) + cd * (ay - by)) / D
        uy = (ad * (cx - bx) + bd * (ax - cx) + cd * (bx - ax)) / D
        return (ux, uy)

    def get_neighbor_cells(self, cell_id: int) -> List[int]:
        if cell_id < len(self.cells.c):
            return self.cells.c[cell_id]
        return []

    def is_border(self, cell_id: int) -> bool:
        if cell_id < len(self.cells.b):
            return bool(self.cells.b[cell_id])
        return False


def generate_poisson_points(
    width: int,
    height: int,
    n_points: int,
    seed: int | None = None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    points = np.column_stack(
        [rng.uniform(0, width, n_points), rng.uniform(0, height, n_points)]
    )
    return points


def generate_relaxed_points(
    width: int,
    height: int,
    n_points: int,
    seed: int | None = None,
    iterations: int = 2,
) -> np.ndarray:
    points = generate_poisson_points(width, height, n_points, seed)
    if not _HAS_SCIPY:
        return points

    for _ in range(iterations):
        tri = Delaunay(points)
        centroids = np.zeros_like(points)
        counts = np.zeros(len(points))
        for simplex in tri.simplices:
            cx = points[simplex, 0].mean()
            cy = points[simplex, 1].mean()
            for idx in simplex:
                centroids[idx, 0] += cx
                centroids[idx, 1] += cy
                counts[idx] += 1
        mask = counts > 0
        points[mask, 0] = centroids[mask, 0] / counts[mask]
        points[mask, 1] = centroids[mask, 1] / counts[mask]
        points[:, 0] = np.clip(points[:, 0], 1, width - 1)
        points[:, 1] = np.clip(points[:, 1], 1, height - 1)

    return points
