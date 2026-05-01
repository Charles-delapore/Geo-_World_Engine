from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


class CSGSphere:
    def __init__(
        self,
        center_lon_deg: float,
        center_lat_deg: float,
        radius_deg: float,
        depth: float = -0.3,
        operation: str = "subtract",
        irregularity: float = 0.2,
        seed: int = 0,
    ):
        self.center_lon_deg = center_lon_deg
        self.center_lat_deg = center_lat_deg
        self.radius_deg = radius_deg
        self.depth = depth
        self.operation = operation
        self.irregularity = irregularity
        self.seed = seed


class CSGTree:
    def __init__(self, base_elevation: np.ndarray):
        self.base_elevation = base_elevation.copy()
        self.spheres: list[CSGSphere] = []

    def add_sphere(self, sphere: CSGSphere) -> None:
        self.spheres.append(sphere)

    def remove_sphere(self, index: int) -> None:
        if 0 <= index < len(self.spheres):
            self.spheres.pop(index)

    def evaluate(self) -> np.ndarray:
        result = self.base_elevation.astype(np.float32).copy()
        h, w = result.shape

        lon = np.linspace(-180, 180, w, endpoint=False, dtype=np.float32)
        lat = np.linspace(90, -90, h, dtype=np.float32)
        lon_grid, lat_grid = np.meshgrid(lon, lat)

        for sphere in self.spheres:
            dlon = lon_grid - sphere.center_lon_deg
            dlon = np.where(dlon > 180, dlon - 360, np.where(dlon < -180, dlon + 360, dlon))
            dlat = lat_grid - sphere.center_lat_deg

            dist_deg = np.sqrt(dlon ** 2 + dlat ** 2)

            if sphere.irregularity > 0.01:
                from app.core.sphere_terrain import sphere_fbm, erp_grid_to_sphere

                sx, sy, sz = erp_grid_to_sphere(w, h)
                perturbation = sphere_fbm(
                    sx, sy, sz,
                    scale=8.0, octaves=2, seed=sphere.seed,
                )
                dist_deg = dist_deg + perturbation * sphere.irregularity * sphere.radius_deg

            radius_rad = sphere.radius_deg
            mask = dist_deg < radius_rad

            falloff = np.clip(1.0 - (dist_deg / max(radius_rad, 1e-6)) ** 2, 0, 1)
            influence = falloff ** 2

            if sphere.operation == "subtract":
                carved = result + sphere.depth * influence
                result = np.where(mask, np.minimum(result, carved), result)
            elif sphere.operation == "add":
                added = result + abs(sphere.depth) * influence
                result = np.where(mask, np.maximum(result, added), result)
            elif sphere.operation == "flatten":
                target = sphere.depth
                blend = influence * 0.8
                result = np.where(mask, result * (1 - blend) + target * blend, result)

        return np.clip(result, -1.0, 1.0).astype(np.float32)

    def evaluate_preview(self, resolution: int = 256) -> np.ndarray:
        h, w = self.base_elevation.shape
        if max(h, w) <= resolution:
            return self.evaluate()

        scale_h = resolution / h
        scale_w = resolution / w
        from PIL import Image as PILImage

        small = np.asarray(
            PILImage.fromarray(
                ((self.base_elevation + 1.0) * 127.5).clip(0, 255).astype(np.uint8), mode="L"
            ).resize((int(w * scale_w), int(h * scale_h)), PILImage.BILINEAR)
        ).astype(np.float32) / 127.5 - 1.0

        temp_tree = CSGTree(small)
        temp_tree.spheres = self.spheres
        return temp_tree.evaluate()


def apply_csg_operation(
    elevation: np.ndarray,
    operation: str,
    center_lon_deg: float,
    center_lat_deg: float,
    radius_deg: float,
    depth: float = -0.3,
    irregularity: float = 0.2,
    seed: int = 0,
) -> np.ndarray:
    tree = CSGTree(elevation)
    sphere = CSGSphere(
        center_lon_deg=center_lon_deg,
        center_lat_deg=center_lat_deg,
        radius_deg=radius_deg,
        depth=depth,
        operation=operation,
        irregularity=irregularity,
        seed=seed,
    )
    tree.add_sphere(sphere)
    return tree.evaluate()


def csg_carve_river(
    elevation: np.ndarray,
    start_lon: float,
    start_lat: float,
    end_lon: float,
    end_lat: float,
    width_deg: float = 3.0,
    depth: float = -0.15,
    n_spheres: int = 8,
    irregularity: float = 0.3,
    seed: int = 0,
) -> np.ndarray:
    tree = CSGTree(elevation)

    lons = np.linspace(start_lon, end_lon, n_spheres)
    lats = np.linspace(start_lat, end_lat, n_spheres)

    rng = np.random.RandomState(seed)
    for i in range(n_spheres):
        jitter_lon = rng.uniform(-1, 1) * width_deg * 0.3
        jitter_lat = rng.uniform(-1, 1) * width_deg * 0.3
        sphere = CSGSphere(
            center_lon_deg=float(lons[i]) + jitter_lon,
            center_lat_deg=float(lats[i]) + jitter_lat,
            radius_deg=width_deg * (0.8 + rng.uniform(-0.2, 0.2)),
            depth=depth * (0.7 + rng.uniform(0, 0.3)),
            operation="subtract",
            irregularity=irregularity,
            seed=seed + i * 7,
        )
        tree.add_sphere(sphere)

    return tree.evaluate()


def csg_create_crater(
    elevation: np.ndarray,
    center_lon_deg: float,
    center_lat_deg: float,
    radius_deg: float = 5.0,
    rim_height: float = 0.1,
    depth: float = -0.3,
    irregularity: float = 0.15,
    seed: int = 0,
) -> np.ndarray:
    tree = CSGTree(elevation)

    tree.add_sphere(CSGSphere(
        center_lon_deg=center_lon_deg,
        center_lat_deg=center_lat_deg,
        radius_deg=radius_deg,
        depth=depth,
        operation="subtract",
        irregularity=irregularity,
        seed=seed,
    ))

    tree.add_sphere(CSGSphere(
        center_lon_deg=center_lon_deg,
        center_lat_deg=center_lat_deg,
        radius_deg=radius_deg * 1.3,
        depth=rim_height,
        operation="add",
        irregularity=irregularity * 0.5,
        seed=seed + 100,
    ))

    return tree.evaluate()
