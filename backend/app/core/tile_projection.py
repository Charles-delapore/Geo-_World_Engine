from __future__ import annotations

import numpy as np
from PIL import Image


WEB_MERCATOR_MAX_LAT = 85.05112878


def mercator_tile_source_indices(
    width: int,
    height: int,
    zoom: int,
    tx: int,
    ty: int,
    tile_size: int,
):
    """Map one XYZ tile to source-image pixel indices from an equirectangular world map."""
    scale = 2 ** zoom
    global_x = (tx * tile_size) + np.arange(tile_size, dtype=np.float32) + 0.5
    world_x = global_x / (tile_size * scale)
    longitudes = world_x * 360.0 - 180.0
    src_x = np.clip(
        np.rint((longitudes + 180.0) / 360.0 * max(width - 1, 1)),
        0,
        max(width - 1, 0),
    ).astype(np.int32)

    global_y = (ty * tile_size) + np.arange(tile_size, dtype=np.float32) + 0.5
    world_y = global_y / (tile_size * scale)
    mercator_n = np.pi * (1.0 - 2.0 * world_y)
    latitudes = np.degrees(np.arctan(np.sinh(mercator_n)))
    latitudes = np.clip(latitudes, -WEB_MERCATOR_MAX_LAT, WEB_MERCATOR_MAX_LAT)
    src_y = np.clip(
        np.rint((90.0 - latitudes) / 180.0 * max(height - 1, 1)),
        0,
        max(height - 1, 0),
    ).astype(np.int32)

    return src_x, src_y


def render_xyz_tile(source_rgb: np.ndarray, zoom: int, tx: int, ty: int, tile_size: int):
    """Render a Web Mercator raster tile from an equirectangular RGB source image."""
    height, width = source_rgb.shape[:2]
    src_x, src_y = mercator_tile_source_indices(width, height, zoom, tx, ty, tile_size)
    tile_rgb = source_rgb[src_y[:, None], src_x[None, :]]
    return Image.fromarray(tile_rgb.astype(np.uint8), mode='RGB')
