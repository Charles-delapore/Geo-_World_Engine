from __future__ import annotations

import numpy as np
from PIL import Image

from app.core.cubemap_terrain import CubeMapTerrainGenerator, NUM_FACES
from app.core.cubemap_to_erp import cubemap_to_erp


class PlanetLODLevel:
    def __init__(
        self,
        level: int,
        face_resolution: int,
        erp_width: int,
        erp_height: int,
        max_zoom: int,
        noise_base_scale: float,
        noise_octaves: int,
        detail_amplitude: float,
    ):
        self.level = level
        self.face_resolution = face_resolution
        self.erp_width = erp_width
        self.erp_height = erp_height
        self.max_zoom = max_zoom
        self.noise_base_scale = noise_base_scale
        self.noise_octaves = noise_octaves
        self.detail_amplitude = detail_amplitude


PLANET_LOD_PRESETS: dict[str, list[PlanetLODLevel]] = {
    "standard": [
        PlanetLODLevel(0, 256, 1024, 512, 2, 3.0, 6, 0.06),
        PlanetLODLevel(1, 512, 2048, 1024, 4, 3.0, 8, 0.06),
        PlanetLODLevel(2, 1024, 4096, 2048, 6, 3.0, 8, 0.06),
    ],
    "high": [
        PlanetLODLevel(0, 512, 2048, 1024, 3, 3.0, 8, 0.06),
        PlanetLODLevel(1, 1024, 4096, 2048, 5, 3.0, 8, 0.06),
        PlanetLODLevel(2, 2048, 8192, 4096, 7, 3.0, 10, 0.05),
    ],
    "performance": [
        PlanetLODLevel(0, 256, 1024, 512, 2, 3.0, 6, 0.06),
        PlanetLODLevel(1, 512, 2048, 1024, 4, 3.0, 8, 0.06),
    ],
}


class PlanetLODManager:
    def __init__(self, preset: str = "standard", seed: int = 42):
        self.preset = preset
        self.seed = seed
        self.levels = PLANET_LOD_PRESETS.get(preset, PLANET_LOD_PRESETS["standard"])

    def generate_base_terrain(self, constraints: dict | None = None) -> np.ndarray:
        base_level = self.levels[0]
        gen = CubeMapTerrainGenerator(
            face_resolution=base_level.face_resolution,
            seed=self.seed,
        )
        cube_faces = gen.generate(constraints=constraints)
        erp = cubemap_to_erp(cube_faces, erp_width=base_level.erp_width, erp_height=base_level.erp_height)
        return erp

    def generate_detail_terrain(self, constraints: dict | None = None, target_level: int = 1) -> np.ndarray:
        if target_level >= len(self.levels):
            target_level = len(self.levels) - 1

        lod = self.levels[target_level]
        gen = CubeMapTerrainGenerator(
            face_resolution=lod.face_resolution,
            seed=self.seed,
        )
        cube_faces = gen.generate(constraints=constraints)
        erp = cubemap_to_erp(cube_faces, erp_width=lod.erp_width, erp_height=lod.erp_height)
        return erp

    def get_max_zoom(self, level: int = 0) -> int:
        if level < len(self.levels):
            return self.levels[level].max_zoom
        return self.levels[-1].max_zoom

    def get_erp_dimensions(self, level: int = 0) -> tuple[int, int]:
        if level < len(self.levels):
            lod = self.levels[level]
            return lod.erp_width, lod.erp_height
        lod = self.levels[-1]
        return lod.erp_width, lod.erp_height

    def select_level_for_zoom(self, zoom: int) -> int:
        for i, lod in enumerate(self.levels):
            if zoom <= lod.max_zoom:
                return i
        return len(self.levels) - 1


def compute_geographic_tile_coordinates(
    width: int,
    height: int,
    zoom: int,
    tx: int,
    ty: int,
    tile_size: int,
    wrap_x: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    scale = 2 ** zoom
    global_x = (tx * tile_size) + np.arange(tile_size, dtype=np.float32) + 0.5
    longitudes = global_x / (tile_size * scale) * 360.0 - 180.0

    global_y = (ty * tile_size) + np.arange(tile_size, dtype=np.float32) + 0.5
    latitudes = 90.0 - global_y / (tile_size * scale) * 180.0
    latitudes = np.clip(latitudes, -90.0, 90.0)

    if wrap_x:
        src_x = ((longitudes + 180.0) / 360.0 * width) - 0.5
        src_x = np.mod(src_x, max(width, 1)).astype(np.float32)
    else:
        src_x = np.clip(
            ((longitudes + 180.0) / 360.0 * max(width, 1)) - 0.5,
            0,
            max(width - 1, 0),
        ).astype(np.float32)

    src_y = np.clip(
        ((90.0 - latitudes) / 180.0 * max(height, 1)) - 0.5,
        0,
        max(height - 1, 0),
    ).astype(np.float32)

    return src_x, src_y


def render_geographic_tile(
    source_rgb: np.ndarray,
    zoom: int,
    tx: int,
    ty: int,
    tile_size: int,
    wrap_x: bool = True,
) -> Image.Image:
    height, width = source_rgb.shape[:2]
    src_x, src_y = compute_geographic_tile_coordinates(width, height, zoom, tx, ty, tile_size, wrap_x=wrap_x)

    from app.core.tile_projection import _sample_bilinear
    tile_rgb = _sample_bilinear(source_rgb.astype(np.float32), src_x, src_y, wrap_x=wrap_x)
    return Image.fromarray(tile_rgb.astype(np.uint8), mode="RGB")
