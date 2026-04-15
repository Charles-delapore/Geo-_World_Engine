from __future__ import annotations

import numpy as np
from PIL import Image

from app.core.climate import ClimateSimulator
from app.core.terrain import TerrainGenerator


def render_world(plan: dict, width: int, height: int, seed: int) -> tuple[dict[str, np.ndarray], Image.Image]:
    terrain = TerrainGenerator(width=width, height=height, seed=seed)
    elevation = terrain.generate()

    constraints = (plan or {}).get("constraints") or {}
    if constraints:
        elevation = terrain.apply_constraints(elevation, constraints)

    filled = elevation
    latitude = np.linspace(90.0, -90.0, height, dtype=np.float32).reshape(height, 1)
    lat_grid = np.repeat(latitude, width, axis=1)
    climate = ClimateSimulator(elev=((filled + 1.0) * 2500.0).astype(np.float32), lat_grid=lat_grid).run()
    preview = _render_preview_image(filled, climate["biome"])
    arrays = {
        "elevation": filled.astype(np.float32),
        "temperature": climate["temperature"].astype(np.float32),
        "precipitation": climate["precipitation"].astype(np.float32),
        "biome": climate["biome"].astype(np.int16),
    }
    return arrays, preview


def _render_preview_image(elevation: np.ndarray, biome: np.ndarray) -> Image.Image:
    ocean = elevation < 0
    palette = np.zeros((*elevation.shape, 3), dtype=np.uint8)
    palette[ocean] = np.array([32, 79, 140], dtype=np.uint8)

    land = ~ocean
    palette[land & (biome == 1)] = np.array([203, 177, 120], dtype=np.uint8)
    palette[land & (biome == 2)] = np.array([145, 168, 94], dtype=np.uint8)
    palette[land & (biome == 3)] = np.array([66, 122, 72], dtype=np.uint8)
    palette[land & (biome == 4)] = np.array([143, 152, 133], dtype=np.uint8)
    palette[land & (biome == 5)] = np.array([236, 240, 245], dtype=np.uint8)

    normalized = np.clip((elevation + 1.0) / 2.0, 0.0, 1.0)
    shade = (normalized * 55).astype(np.uint8)
    palette = np.clip(palette + shade[..., None], 0, 255).astype(np.uint8)
    return Image.fromarray(palette, mode="RGB")
