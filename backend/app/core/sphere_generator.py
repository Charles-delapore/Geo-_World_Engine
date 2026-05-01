from __future__ import annotations

import numpy as np

from app.core.sphere_terrain import (
    erp_grid_to_sphere,
    sphere_fbm,
    sphere_ridged_noise,
    sphere_warp,
    sphere_gaussian_basin,
    sphere_distance,
    latitudinal_bias,
    polar_ocean_bias,
)


class SphereTerrainGenerator:
    def __init__(self, width: int = 1024, height: int = 512, seed: int = 42):
        self.width = width
        self.height = height
        self.seed = seed
        self.sx, self.sy, self.sz = erp_grid_to_sphere(width, height)
        self._y_norm = np.linspace(0, 1, height, dtype=np.float32).reshape(-1, 1)
        self._x_norm = np.linspace(0, 1, width, endpoint=False, dtype=np.float32).reshape(1, -1)

    def generate_base(self) -> np.ndarray:
        warped_sx, warped_sy, warped_sz = sphere_warp(
            self.sx, self.sy, self.sz,
            strength=0.12, scale=2.5, seed=self.seed,
        )
        base = sphere_fbm(
            warped_sx, warped_sy, warped_sz,
            scale=3.0, octaves=8, persistence=0.5, lacunarity=2.0, seed=self.seed,
        )
        return base

    def generate_tectonic_plates(self, n_plates: int = 6) -> np.ndarray:
        rng = np.random.RandomState(self.seed + 100)
        field = np.zeros((self.height, self.width), dtype=np.float32)

        for i in range(n_plates):
            lon = rng.uniform(-180, 180)
            lat = rng.uniform(-60, 60)
            radius = rng.uniform(25, 55)
            is_land = rng.random() > 0.4
            amplitude = rng.uniform(0.3, 0.8) if is_land else rng.uniform(-0.6, -0.2)

            basin = sphere_gaussian_basin(
                self.sx, self.sy, self.sz,
                center_lon_deg=lon, center_lat_deg=lat,
                radius_deg=radius, amplitude=amplitude,
                irregularity=rng.uniform(0.1, 0.4),
                seed=self.seed + i * 17,
            )
            field += basin

        return field

    def apply_continent_constraint(
        self,
        elev: np.ndarray,
        center_lon_deg: float,
        center_lat_deg: float,
        radius_deg: float,
        amplitude: float = 0.8,
        irregularity: float = 0.35,
        seed_offset: int = 0,
    ) -> np.ndarray:
        basin = sphere_gaussian_basin(
            self.sx, self.sy, self.sz,
            center_lon_deg=center_lon_deg,
            center_lat_deg=center_lat_deg,
            radius_deg=radius_deg,
            amplitude=amplitude,
            irregularity=irregularity,
            seed=self.seed + seed_offset,
        )
        land_mask = basin > 0.15
        elev = np.where(land_mask, np.maximum(elev, basin * 0.6), elev)
        return elev

    def apply_mountain_chain(
        self,
        elev: np.ndarray,
        center_lon_deg: float,
        center_lat_deg: float,
        length_deg: float = 30.0,
        width_deg: float = 8.0,
        orientation_deg: float = 0.0,
        amplitude: float = 0.6,
        seed_offset: int = 0,
    ) -> np.ndarray:
        center_lon = np.radians(center_lon_deg)
        center_lat = np.radians(center_lat_deg)
        cx = np.cos(center_lat) * np.cos(center_lon)
        cy = np.cos(center_lat) * np.sin(center_lon)
        cz = np.sin(center_lat)

        dist = sphere_distance(self.sx, self.sy, self.sz, cx, cy, cz)

        ridge = sphere_ridged_noise(
            self.sx, self.sy, self.sz,
            scale=6.0, octaves=5, seed=self.seed + seed_offset,
        )

        length_rad = np.radians(length_deg)
        width_rad = np.radians(width_deg)
        mask = np.exp(-0.5 * (dist / max(length_rad, 1e-6)) ** 2)
        ridge_mask = np.exp(-0.5 * (dist / max(width_rad, 1e-6)) ** 2)

        mountain = ridge * ridge_mask * amplitude
        elev = elev + mountain * mask
        return elev

    def apply_sea_zone(
        self,
        elev: np.ndarray,
        center_lon_deg: float,
        center_lat_deg: float,
        radius_deg: float,
        depth: float = -0.5,
        seed_offset: int = 0,
    ) -> np.ndarray:
        basin = sphere_gaussian_basin(
            self.sx, self.sy, self.sz,
            center_lon_deg=center_lon_deg,
            center_lat_deg=center_lat_deg,
            radius_deg=radius_deg,
            amplitude=1.0,
            irregularity=0.3,
            seed=self.seed + seed_offset,
        )
        sea_mask = basin > 0.2
        elev = np.where(sea_mask, np.minimum(elev, depth * basin), elev)
        return elev

    def rebalance_sea_level(self, elev: np.ndarray, target_ocean: float = 0.56) -> np.ndarray:
        sorted_elev = np.sort(elev.ravel())
        idx = int(target_ocean * len(sorted_elev))
        idx = min(max(idx, 0), len(sorted_elev) - 1)
        sea_level = sorted_elev[idx]
        return elev - sea_level

    def apply_latitudinal_features(self, elev: np.ndarray) -> np.ndarray:
        lat_bias = latitudinal_bias(self._y_norm, land_bias=0.08)
        polar = polar_ocean_bias(self.sx, self.sy, self.sz, threshold_deg=72.0, strength=0.4)
        elev = elev + lat_bias + polar
        return elev

    def add_detail_noise(self, elev: np.ndarray, scale: float = 12.0, amplitude: float = 0.08) -> np.ndarray:
        detail = sphere_fbm(
            self.sx, self.sy, self.sz,
            scale=scale, octaves=4, persistence=0.4, seed=self.seed + 500,
        )
        return elev + detail * amplitude

    def generate(self, constraints: dict | None = None) -> np.ndarray:
        elev = self.generate_base()
        tectonic = self.generate_tectonic_plates(n_plates=6)
        elev = elev * 0.4 + tectonic * 0.6

        elev = self.apply_latitudinal_features(elev)

        if constraints:
            elev = self._apply_constraints(elev, constraints)

        elev = self.rebalance_sea_level(elev, target_ocean=0.56)
        elev = self.add_detail_noise(elev, scale=12.0, amplitude=0.06)

        elev = np.clip(elev, -1.0, 1.0).astype(np.float32)
        return elev

    def _apply_constraints(self, elev: np.ndarray, constraints: dict) -> np.ndarray:
        position_map = {
            "northwest": (-90, 40), "north": (0, 40), "northeast": (90, 40),
            "west": (-90, 0), "center": (0, 0), "east": (90, 0),
            "southwest": (-90, -30), "south": (0, -30), "southeast": (90, -30),
        }

        for cont in constraints.get("continents", []):
            pos = cont.get("position", "center")
            lon, lat = position_map.get(pos, (0, 0))
            size = cont.get("size", "medium")
            radius_map = {"small": 20, "medium": 35, "large": 50}
            radius = radius_map.get(size, 35)
            elev = self.apply_continent_constraint(
                elev, lon, lat, radius,
                amplitude=0.7, irregularity=0.35,
                seed_offset=hash(pos) % 1000,
            )

        for mt in constraints.get("mountains", []):
            pos = mt.get("position", "center")
            lon, lat = position_map.get(pos, (0, 0))
            elev = self.apply_mountain_chain(
                elev, lon, lat,
                length_deg=25, width_deg=6,
                amplitude=0.5, seed_offset=hash(pos) % 1000 + 200,
            )

        for sea in constraints.get("sea_zones", []):
            pos = sea.get("position", "center")
            lon, lat = position_map.get(pos, (0, 0))
            elev = self.apply_sea_zone(
                elev, lon, lat, radius_deg=20,
                depth=-0.5, seed_offset=hash(pos) % 1000 + 400,
            )

        return elev
