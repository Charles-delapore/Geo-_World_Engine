from __future__ import annotations

import numpy as np
from scipy.interpolate import RegularGridInterpolator

FACE_NAMES = ("PX", "NX", "PY", "NY", "PZ", "NZ")
NUM_FACES = 6
_GRID_CACHE_MAX_BYTES = 256 * 1024 * 1024


def _noise3_fast(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    import opensimplex

    flat_x = x.ravel().astype(np.float64)
    flat_y = y.ravel().astype(np.float64)
    flat_z = z.ravel().astype(np.float64)

    pad = 1.0
    x_min, x_max = flat_x.min() - pad, flat_x.max() + pad
    y_min, y_max = flat_y.min() - pad, flat_y.max() + pad
    z_min, z_max = flat_z.min() - pad, flat_z.max() + pad

    range_x = max(x_max - x_min, 1.0)
    range_y = max(y_max - y_min, 1.0)
    range_z = max(z_max - z_min, 1.0)
    max_range = max(range_x, range_y, range_z)

    n_pts = max(16, min(int(max_range * 8), 128))

    est_bytes = n_pts ** 3 * 8
    if est_bytes > _GRID_CACHE_MAX_BYTES:
        _ufunc = np.frompyfunc(opensimplex.noise3, 3, 1)
        result = _ufunc(flat_x, flat_y, flat_z)
        return result.astype(np.float32).reshape(x.shape)

    xs = np.linspace(x_min, x_max, n_pts)
    ys = np.linspace(y_min, y_max, n_pts)
    zs = np.linspace(z_min, z_max, n_pts)

    grid_3d = opensimplex.noise3array(xs, ys, zs)

    interp = RegularGridInterpolator(
        (zs, ys, xs),
        grid_3d,
        method="linear",
        bounds_error=False,
        fill_value=0.0,
    )

    points = np.stack([flat_z, flat_y, flat_x], axis=-1)
    result = interp(points)

    return result.reshape(x.shape).astype(np.float32)


def _face_to_sphere(face_id: int, u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if face_id == 0:
        sx, sy, sz = np.ones_like(u), v, -u
    elif face_id == 1:
        sx, sy, sz = -np.ones_like(u), v, u
    elif face_id == 2:
        sx, sy, sz = u, np.ones_like(u), -v
    elif face_id == 3:
        sx, sy, sz = u, -np.ones_like(u), v
    elif face_id == 4:
        sx, sy, sz = u, v, np.ones_like(u)
    elif face_id == 5:
        sx, sy, sz = -u, v, -np.ones_like(u)
    else:
        raise ValueError(f"Invalid face_id {face_id}")

    mag = np.sqrt(sx ** 2 + sy ** 2 + sz ** 2)
    mag = np.maximum(mag, 1e-12)
    return (sx / mag).astype(np.float32), (sy / mag).astype(np.float32), (sz / mag).astype(np.float32)


def build_face_grids(resolution: int) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    coords = np.linspace(-1.0, 1.0, resolution, dtype=np.float32)
    u_grid, v_grid = np.meshgrid(coords, coords)
    result = []
    for fid in range(NUM_FACES):
        sx, sy, sz = _face_to_sphere(fid, u_grid, v_grid)
        result.append((sx, sy, sz))
    return result


def _scale_aware_fbm(
    sx: np.ndarray,
    sy: np.ndarray,
    sz: np.ndarray,
    base_scale: float = 4.0,
    octaves: int = 8,
    persistence: float = 0.5,
    lacunarity: float = 2.0,
    seed: int = 0,
    compensate_latitude: bool = True,
) -> np.ndarray:
    import opensimplex

    rng = np.random.RandomState(seed)
    opensimplex.seed(seed)

    lat = np.arcsin(np.clip(sz, -1.0, 1.0))
    cos_lat = np.cos(lat)
    cos_lat = np.maximum(cos_lat, 0.15)

    result = np.zeros_like(sx, dtype=np.float32)
    amplitude = 1.0
    frequency = base_scale
    max_amp = 0.0

    for _ in range(octaves):
        offset_x = rng.uniform(-1000, 1000)
        offset_y = rng.uniform(-1000, 1000)
        offset_z = rng.uniform(-1000, 1000)

        if compensate_latitude:
            eff_freq = frequency / cos_lat
            sample_x = sx * eff_freq + offset_x
            sample_y = sy * eff_freq + offset_y
            sample_z = sz * eff_freq + offset_z
        else:
            sample_x = sx * frequency + offset_x
            sample_y = sy * frequency + offset_y
            sample_z = sz * frequency + offset_z

        noise_vals = _noise3_fast(sample_x, sample_y, sample_z)

        result += noise_vals * amplitude
        max_amp += amplitude
        amplitude *= persistence
        frequency *= lacunarity

    if max_amp > 0:
        result /= max_amp

    return result


def _scale_aware_ridged(
    sx: np.ndarray,
    sy: np.ndarray,
    sz: np.ndarray,
    base_scale: float = 4.0,
    octaves: int = 6,
    persistence: float = 0.5,
    lacunarity: float = 2.0,
    seed: int = 0,
) -> np.ndarray:
    fbm = _scale_aware_fbm(sx, sy, sz, base_scale, octaves, persistence, lacunarity, seed)
    ridged = 1.0 - np.abs(2.0 * fbm - 1.0)
    return ridged * ridged


def _sphere_warp(
    sx: np.ndarray,
    sy: np.ndarray,
    sz: np.ndarray,
    strength: float = 0.15,
    scale: float = 3.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    wx = _scale_aware_fbm(sx, sy, sz, base_scale=scale, octaves=2, seed=seed, compensate_latitude=False)
    wy = _scale_aware_fbm(sx, sy, sz, base_scale=scale, octaves=2, seed=seed + 100, compensate_latitude=False)
    wz = _scale_aware_fbm(sx, sy, sz, base_scale=scale, octaves=2, seed=seed + 200, compensate_latitude=False)

    warped_sx = sx + wx * strength
    warped_sy = sy + wy * strength
    warped_sz = sz + wz * strength

    mag = np.sqrt(warped_sx ** 2 + warped_sy ** 2 + warped_sz ** 2)
    mag = np.maximum(mag, 1e-8)
    return (warped_sx / mag).astype(np.float32), (warped_sy / mag).astype(np.float32), (warped_sz / mag).astype(np.float32)


def _sphere_distance(
    sx1: np.ndarray, sy1: np.ndarray, sz1: np.ndarray,
    sx2: float, sy2: float, sz2: float,
) -> np.ndarray:
    dot = sx1 * sx2 + sy1 * sy2 + sz1 * sz2
    dot = np.clip(dot, -1.0, 1.0)
    return np.arccos(dot)


def _sphere_gaussian_basin(
    sx: np.ndarray,
    sy: np.ndarray,
    sz: np.ndarray,
    center_lon_deg: float,
    center_lat_deg: float,
    radius_deg: float,
    amplitude: float = 1.0,
    irregularity: float = 0.3,
    seed: int = 0,
) -> np.ndarray:
    center_lon = np.radians(center_lon_deg)
    center_lat = np.radians(center_lat_deg)
    cx = np.cos(center_lat) * np.cos(center_lon)
    cy = np.cos(center_lat) * np.sin(center_lon)
    cz = np.sin(center_lat)

    dist = _sphere_distance(sx, sy, sz, cx, cy, cz)

    if irregularity > 0.01:
        wsx, wsy, wsz = _sphere_warp(sx, sy, sz, strength=irregularity * 0.25, scale=3.0, seed=seed)
        dist = _sphere_distance(wsx, wsy, wsz, cx, cy, cz)

    radius_rad = np.radians(radius_deg)
    falloff = np.exp(-0.5 * (dist / max(radius_rad, 1e-6)) ** 2)
    return falloff * amplitude


def _latitudinal_bias(sz: np.ndarray, land_bias: float = 0.08) -> np.ndarray:
    lat = np.arcsin(np.clip(sz, -1.0, 1.0))
    lat_norm = lat / (np.pi / 2)
    return (np.cos(lat_norm * np.pi) ** 2 * land_bias).astype(np.float32)


def _polar_ocean_bias(sz: np.ndarray, threshold_deg: float = 72.0, strength: float = 0.4) -> np.ndarray:
    lat = np.arcsin(np.clip(sz, -1.0, 1.0))
    lat_deg = np.degrees(lat)
    pole_mask = np.abs(lat_deg) > threshold_deg
    falloff = np.clip((np.abs(lat_deg) - threshold_deg) / (90.0 - threshold_deg), 0, 1)
    return (-falloff * strength * pole_mask).astype(np.float32)


class CubeMapTerrainGenerator:
    def __init__(self, face_resolution: int = 1024, seed: int = 42):
        self.face_resolution = face_resolution
        self.seed = seed
        self.face_coords = build_face_grids(face_resolution)

    def generate_base(self) -> list[np.ndarray]:
        try:
            from concurrent.futures import ThreadPoolExecutor
            from functools import partial

            def _generate_face(fid: int) -> np.ndarray:
                sx, sy, sz = self.face_coords[fid]
                wsx, wsy, wsz = _sphere_warp(sx, sy, sz, strength=0.12, scale=2.5, seed=self.seed + fid)
                return _scale_aware_fbm(
                    wsx, wsy, wsz,
                    base_scale=3.0, octaves=6, persistence=0.5,
                    lacunarity=2.0, seed=self.seed,
                    compensate_latitude=True,
                )

            with ThreadPoolExecutor(max_workers=min(6, NUM_FACES)) as executor:
                faces = list(executor.map(_generate_face, range(NUM_FACES)))
        except Exception:
            faces = []
            for fid, (sx, sy, sz) in enumerate(self.face_coords):
                wsx, wsy, wsz = _sphere_warp(sx, sy, sz, strength=0.12, scale=2.5, seed=self.seed + fid)
                base = _scale_aware_fbm(
                    wsx, wsy, wsz,
                    base_scale=3.0, octaves=6, persistence=0.5,
                    lacunarity=2.0, seed=self.seed,
                    compensate_latitude=True,
                )
                faces.append(base)
        return faces

    def generate_tectonic_plates(self, n_plates: int = 7) -> list[np.ndarray]:
        rng = np.random.RandomState(self.seed + 100)
        face_fields = [np.zeros_like(sx, dtype=np.float32) for sx, _, _ in self.face_coords]

        for i in range(n_plates):
            lon = rng.uniform(-180, 180)
            lat = rng.uniform(-70, 70)
            radius = rng.uniform(20, 55)
            is_land = rng.random() > 0.4
            amplitude = rng.uniform(0.3, 0.8) if is_land else rng.uniform(-0.6, -0.2)

            for fid, (sx, sy, sz) in enumerate(self.face_coords):
                basin = _sphere_gaussian_basin(
                    sx, sy, sz,
                    center_lon_deg=lon, center_lat_deg=lat,
                    radius_deg=radius, amplitude=amplitude,
                    irregularity=rng.uniform(0.1, 0.4),
                    seed=self.seed + i * 17,
                )
                face_fields[fid] += basin

        return face_fields

    def apply_continent_constraint(
        self,
        faces: list[np.ndarray],
        center_lon_deg: float,
        center_lat_deg: float,
        radius_deg: float,
        amplitude: float = 0.8,
        irregularity: float = 0.35,
        seed_offset: int = 0,
    ) -> list[np.ndarray]:
        result = []
        for fid, (sx, sy, sz) in enumerate(self.face_coords):
            basin = _sphere_gaussian_basin(
                sx, sy, sz,
                center_lon_deg=center_lon_deg,
                center_lat_deg=center_lat_deg,
                radius_deg=radius_deg,
                amplitude=amplitude,
                irregularity=irregularity,
                seed=self.seed + seed_offset,
            )
            land_mask = basin > 0.15
            new_elev = np.where(land_mask, np.maximum(faces[fid], basin * 0.6), faces[fid])
            result.append(new_elev.astype(np.float32))
        return result

    def apply_mountain_chain(
        self,
        faces: list[np.ndarray],
        center_lon_deg: float,
        center_lat_deg: float,
        length_deg: float = 30.0,
        width_deg: float = 8.0,
        amplitude: float = 0.6,
        seed_offset: int = 0,
    ) -> list[np.ndarray]:
        center_lon = np.radians(center_lon_deg)
        center_lat = np.radians(center_lat_deg)
        cx = np.cos(center_lat) * np.cos(center_lon)
        cy = np.cos(center_lat) * np.sin(center_lon)
        cz = np.sin(center_lat)

        result = []
        for fid, (sx, sy, sz) in enumerate(self.face_coords):
            dist = _sphere_distance(sx, sy, sz, cx, cy, cz)
            ridge = _scale_aware_ridged(
                sx, sy, sz,
                base_scale=6.0, octaves=5, seed=self.seed + seed_offset,
            )
            length_rad = np.radians(length_deg)
            width_rad = np.radians(width_deg)
            mask = np.exp(-0.5 * (dist / max(length_rad, 1e-6)) ** 2)
            ridge_mask = np.exp(-0.5 * (dist / max(width_rad, 1e-6)) ** 2)
            mountain = ridge * ridge_mask * amplitude
            result.append((faces[fid] + mountain * mask).astype(np.float32))
        return result

    def apply_sea_zone(
        self,
        faces: list[np.ndarray],
        center_lon_deg: float,
        center_lat_deg: float,
        radius_deg: float,
        depth: float = -0.5,
        seed_offset: int = 0,
    ) -> list[np.ndarray]:
        result = []
        for fid, (sx, sy, sz) in enumerate(self.face_coords):
            basin = _sphere_gaussian_basin(
                sx, sy, sz,
                center_lon_deg=center_lon_deg,
                center_lat_deg=center_lat_deg,
                radius_deg=radius_deg,
                amplitude=1.0,
                irregularity=0.3,
                seed=self.seed + seed_offset,
            )
            sea_mask = basin > 0.2
            new_elev = np.where(sea_mask, np.minimum(faces[fid], depth * basin), faces[fid])
            result.append(new_elev.astype(np.float32))
        return result

    def apply_latitudinal_features(self, faces: list[np.ndarray]) -> list[np.ndarray]:
        result = []
        for fid, (sx, sy, sz) in enumerate(self.face_coords):
            lat_bias = _latitudinal_bias(sz, land_bias=0.08)
            polar = _polar_ocean_bias(sz, threshold_deg=72.0, strength=0.4)
            result.append((faces[fid] + lat_bias + polar).astype(np.float32))
        return result

    def add_detail_noise(
        self,
        faces: list[np.ndarray],
        base_scale: float = 12.0,
        amplitude: float = 0.06,
    ) -> list[np.ndarray]:
        result = []
        for fid, (sx, sy, sz) in enumerate(self.face_coords):
            detail = _scale_aware_fbm(
                sx, sy, sz,
                base_scale=base_scale, octaves=3, persistence=0.4,
                seed=self.seed + 500, compensate_latitude=True,
            )
            result.append((faces[fid] + detail * amplitude).astype(np.float32))
        return result

    def rebalance_sea_level(self, faces: list[np.ndarray], target_ocean: float = 0.56) -> list[np.ndarray]:
        all_elev = np.concatenate([f.ravel() for f in faces])
        sorted_elev = np.sort(all_elev)
        idx = int(target_ocean * len(sorted_elev))
        idx = min(max(idx, 0), len(sorted_elev) - 1)
        sea_level = sorted_elev[idx]
        return [(f - sea_level).astype(np.float32) for f in faces]

    def generate(self, constraints: dict | None = None) -> list[np.ndarray]:
        base_faces = self.generate_base()
        tectonic_faces = self.generate_tectonic_plates(n_plates=7)

        faces = []
        for fid in range(NUM_FACES):
            blended = base_faces[fid] * 0.35 + tectonic_faces[fid] * 0.65
            faces.append(blended.astype(np.float32))

        faces = self.apply_latitudinal_features(faces)

        if constraints:
            faces = self._apply_constraints(faces, constraints)

        faces = self.rebalance_sea_level(faces, target_ocean=0.56)
        faces = self.add_detail_noise(faces, base_scale=12.0, amplitude=0.06)

        return [np.clip(f, -1.0, 1.0).astype(np.float32) for f in faces]

    def _apply_constraints(self, faces: list[np.ndarray], constraints: dict) -> list[np.ndarray]:
        position_map = {
            "northwest": (-90, 40), "north": (0, 40), "northeast": (90, 40),
            "west": (-90, 0), "center": (0, 0), "east": (90, 0),
            "southwest": (-90, -30), "south": (0, -30), "southeast": (90, -30),
        }

        for cont in constraints.get("continents", []):
            pos = cont.get("position", "center") if isinstance(cont, dict) else str(cont)
            lon, lat = position_map.get(pos, (0, 0))
            size = cont.get("size", "medium") if isinstance(cont, dict) else "medium"
            radius_map = {"small": 20, "medium": 35, "large": 50}
            radius = float(size) * 80.0 if isinstance(size, (int, float)) else radius_map.get(str(size), 35)
            faces = self.apply_continent_constraint(
                faces, lon, lat, radius,
                amplitude=0.7, irregularity=0.35,
                seed_offset=hash(pos) % 1000,
            )

        for mt in constraints.get("mountains", []):
            pos = (mt.get("position") or mt.get("location", "center")) if isinstance(mt, dict) else str(mt)
            lon, lat = position_map.get(pos, (0, 0))
            faces = self.apply_mountain_chain(
                faces, lon, lat,
                length_deg=25, width_deg=6,
                amplitude=0.5, seed_offset=hash(pos) % 1000 + 200,
            )

        for sea in constraints.get("sea_zones", []):
            pos = sea.get("position", "center") if isinstance(sea, dict) else str(sea)
            lon, lat = position_map.get(pos, (0, 0))
            faces = self.apply_sea_zone(
                faces, lon, lat, radius_deg=20,
                depth=-0.5, seed_offset=hash(pos) % 1000 + 400,
            )

        return faces

    def generate_multi_resolution(
        self,
        constraints: dict | None = None,
        levels: list[int] | None = None,
    ) -> dict[int, list[np.ndarray]]:
        if levels is None:
            levels = [256, 512, 1024]

        result = {}
        for res in levels:
            gen = CubeMapTerrainGenerator(face_resolution=res, seed=self.seed)
            faces = gen.generate(constraints=constraints)
            result[res] = faces

        return result
