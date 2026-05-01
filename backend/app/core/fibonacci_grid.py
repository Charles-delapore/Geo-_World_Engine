from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree, ConvexHull, Delaunay


GOLDEN_RATIO = (1.0 + np.sqrt(5.0)) / 2.0


def generate_fibonacci_grid(
    n_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    k = np.arange(n_points, dtype=np.float64)
    cos_theta = 1.0 - 2.0 * k / (2.0 * n_points - 1.0)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    phi = (2.0 * np.pi * k / GOLDEN_RATIO) % (2.0 * np.pi)

    sx = np.sin(theta) * np.cos(phi)
    sy = np.sin(theta) * np.sin(phi)
    sz = np.cos(theta)

    return sx.astype(np.float32), sy.astype(np.float32), sz.astype(np.float32)


def fibonacci_to_latlon(
    sx: np.ndarray, sy: np.ndarray, sz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    lat = np.arcsin(np.clip(sz, -1.0, 1.0))
    lon = np.arctan2(sy, sx) % (2.0 * np.pi)
    return lat.astype(np.float32), lon.astype(np.float32)


def resolution_to_n_points(resolution_km: float, radius_km: float = 6371.0) -> int:
    surface_area = 4.0 * np.pi * radius_km ** 2
    n = int(np.ceil(surface_area / resolution_km ** 2))
    return max(n, 100)


class FibonacciSphericalGrid:
    def __init__(self, n_points: int):
        self.n_points = n_points
        self.sx, self.sy, self.sz = generate_fibonacci_grid(n_points)
        self.lat, self.lon = fibonacci_to_latlon(self.sx, self.sy, self.sz)
        self._kdtree: cKDTree | None = None
        self._triangles: np.ndarray | None = None
        self._neighbors: list[list[int]] | None = None
        self.height: np.ndarray = np.zeros(n_points, dtype=np.float32)

    def build_kdtree(self) -> cKDTree:
        if self._kdtree is None:
            coords = np.stack([self.sx, self.sy, self.sz], axis=-1)
            self._kdtree = cKDTree(coords)
        return self._kdtree

    def query_nearest(self, points: np.ndarray, k: int = 1) -> tuple[np.ndarray, np.ndarray]:
        tree = self.build_kdtree()
        dist, idx = tree.query(points, k=k)
        return dist.astype(np.float32), idx.astype(np.int32)

    def build_delaunay_triangulation(self) -> np.ndarray:
        if self._triangles is not None:
            return self._triangles

        coords = np.stack([self.sx, self.sy, self.sz], axis=-1)
        hull = ConvexHull(coords)
        self._triangles = hull.simplices.astype(np.int32)
        return self._triangles

    def build_neighbors(self) -> list[list[int]]:
        if self._neighbors is not None:
            return self._neighbors

        triangles = self.build_delaunay_triangulation()
        adj: dict[int, set[int]] = {i: set() for i in range(self.n_points)}
        for tri in triangles:
            for a in range(3):
                for b in range(3):
                    if a != b:
                        adj[int(tri[a])].add(int(tri[b]))

        self._neighbors = [sorted(adj[i]) for i in range(self.n_points)]
        return self._neighbors

    def to_erp(
        self,
        values: np.ndarray,
        erp_width: int = 1024,
        erp_height: int = 512,
        interpolation: str = "nearest",
    ) -> np.ndarray:
        lon_grid = np.linspace(0, 2 * np.pi, erp_width, endpoint=False, dtype=np.float32)
        lat_grid = np.linspace(np.pi / 2, -np.pi / 2, erp_height, dtype=np.float32)
        lon_2d, lat_2d = np.meshgrid(lon_grid, lat_grid)

        query_sx = np.cos(lat_2d) * np.cos(lon_2d)
        query_sy = np.cos(lat_2d) * np.sin(lon_2d)
        query_sz = np.sin(lat_2d)

        query_points = np.stack([query_sx.ravel(), query_sy.ravel(), query_sz.ravel()], axis=-1)

        if interpolation == "nearest":
            _, idx = self.query_nearest(query_points, k=1)
            erp = values[idx.ravel()].reshape(erp_height, erp_width)
        else:
            _, idx = self.query_nearest(query_points, k=4)
            idx_flat = idx.ravel()
            weights = 1.0 / (np.maximum(_spherical_distance_from_xyz(
                query_points[:, 0], query_points[:, 1], query_points[:, 2],
                self.sx[idx_flat], self.sy[idx_flat], self.sz[idx_flat],
            ), 1e-8))
            weights = weights.reshape(-1, 4)
            weights /= weights.sum(axis=1, keepdims=True)

            vals = values[idx.ravel()].reshape(-1, 4)
            erp = (vals * weights).sum(axis=1).reshape(erp_height, erp_width)

        return erp.astype(np.float32)

    def from_erp(self, erp_data: np.ndarray) -> np.ndarray:
        erp_h, erp_w = erp_data.shape[:2]
        lon = self.lon
        lat = self.lat

        col = (lon / (2 * np.pi) * erp_w).astype(np.float64) % erp_w
        row = ((np.pi / 2 - lat) / np.pi * erp_h).astype(np.float64)
        row = np.clip(row, 0, erp_h - 1)

        col0 = np.floor(col).astype(np.int32) % erp_w
        col1 = (col0 + 1) % erp_w
        row0 = np.clip(np.floor(row).astype(np.int32), 0, erp_h - 1)
        row1 = np.clip(row0 + 1, 0, erp_h - 1)

        fc = col - np.floor(col)
        fr = row - np.floor(row)

        v00 = erp_data[row0, col0]
        v01 = erp_data[row0, col1]
        v10 = erp_data[row1, col0]
        v11 = erp_data[row1, col1]

        result = (
            v00 * (1 - fc) * (1 - fr)
            + v01 * fc * (1 - fr)
            + v10 * (1 - fc) * fr
            + v11 * fc * fr
        )
        return result.astype(np.float32)

    def from_cubemap(self, cube_faces: list[np.ndarray]) -> np.ndarray:
        from app.core.cubemap_to_erp import cubemap_to_erp
        erp = cubemap_to_erp(cube_faces, erp_width=max(f.shape[1] for f in cube_faces) * 2)
        return self.from_erp(erp)

    def to_cubemap(self, face_size: int = 512) -> list[np.ndarray]:
        erp = self.to_erp(self.height, erp_width=face_size * 4, erp_height=face_size * 2)
        from app.core.cubemap_to_erp import erp_to_cubemap
        return erp_to_cubemap(erp, face_size=face_size)

    def compute_uplift_from_plates(
        self,
        plate_centers: list[tuple[float, float]],
        plate_uplift_rates: list[float],
        plate_radii: list[float],
        ridge_noise_scale: float = 80.0,
        seed: int = 0,
    ) -> np.ndarray:
        uplift = np.zeros(self.n_points, dtype=np.float32)

        for center_lat_deg, center_lon_deg in plate_centers:
            center_lat = np.radians(center_lat_deg)
            center_lon = np.radians(center_lon_deg)
            cx = np.cos(center_lat) * np.cos(center_lon)
            cy = np.cos(center_lat) * np.sin(center_lon)
            cz = np.sin(center_lat)

            dot = self.sx * cx + self.sy * cy + self.sz * cz
            angular_dist = np.arccos(np.clip(dot, -1.0, 1.0))

            for rate, radius in zip(plate_uplift_rates, plate_radii):
                mask = angular_dist < radius
                falloff = np.exp(-((angular_dist / max(radius * 0.5, 0.01)) ** 2))
                uplift += rate * falloff

        try:
            from app.core.sphere_terrain import sphere_fbm
            noise = sphere_fbm(
                self.sx, self.sy, self.sz,
                scale=ridge_noise_scale,
                octaves=3,
                persistence=0.5,
                lacunarity=2.0,
                seed=seed + 428,
            )
            uplift *= (0.7 + noise * 0.6)
        except Exception:
            pass

        self.height = uplift
        return uplift

    def compute_hardness_map(
        self,
        elevation: np.ndarray | None = None,
        base_hardness: float = 1.0,
        mountain_hardness: float = 0.6,
        coastal_hardness: float = 1.4,
    ) -> np.ndarray:
        hardness = np.full(self.n_points, base_hardness, dtype=np.float32)

        if elevation is not None:
            slope = np.zeros(self.n_points, dtype=np.float32)
            neighbors = self.build_neighbors()
            for i in range(self.n_points):
                if not neighbors[i]:
                    continue
                nbr_idx = np.array(neighbors[i], dtype=np.int32)
                if elevation[i] <= 0:
                    continue
                dists = _spherical_distance_from_xyz(
                    self.sx[i], self.sy[i], self.sz[i],
                    self.sx[nbr_idx], self.sy[nbr_idx], self.sz[nbr_idx],
                )
                valid = dists > 1e-10
                if not np.any(valid):
                    continue
                dh = np.abs(elevation[i] - elevation[nbr_idx[valid]])
                slope[i] = np.mean(dh / dists[valid])

            steep = slope > np.percentile(slope[elevation > 0], 70) if np.any(elevation > 0) else np.zeros(self.n_points, dtype=bool)
            hardness[steep] = mountain_hardness

            near_coast = (elevation > -0.05) & (elevation < 0.05)
            hardness[near_coast] = coastal_hardness

        return hardness


def _spherical_distance_from_xyz(
    ax: np.ndarray, ay: np.ndarray, az: np.ndarray,
    bx: np.ndarray, by: np.ndarray, bz: np.ndarray,
) -> np.ndarray:
    dot = ax * bx + ay * by + az * bz
    dot = np.clip(dot, -1.0, 1.0)
    return np.arccos(dot).astype(np.float32)


def fibonacci_sht_weights(n_points: int, max_degree: int) -> np.ndarray:
    sx, sy, sz = generate_fibonacci_grid(n_points)
    theta = np.arccos(np.clip(sz, -1.0, 1.0))
    phi = np.arctan2(sy, sx) % (2 * np.pi)

    sin_theta = np.sin(theta)
    area_element = sin_theta / (sin_theta.sum() + 1e-10)
    weights = area_element * (4.0 * np.pi / n_points)

    return weights.astype(np.float32)
