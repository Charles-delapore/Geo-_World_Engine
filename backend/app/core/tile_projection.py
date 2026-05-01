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
    wrap_x: bool = True,
):
    src_x, src_y = mercator_tile_source_coordinates(width, height, zoom, tx, ty, tile_size, wrap_x=wrap_x)
    if wrap_x:
        src_x = np.floor(src_x + 0.5).astype(np.int32) % max(width, 1)
    else:
        src_x = np.clip(np.floor(src_x + 0.5), 0, max(width - 1, 0)).astype(np.int32)
    src_y = np.clip(np.floor(src_y + 0.5), 0, max(height - 1, 0)).astype(np.int32)
    return src_x, src_y


def mercator_tile_source_coordinates(
    width: int,
    height: int,
    zoom: int,
    tx: int,
    ty: int,
    tile_size: int,
    wrap_x: bool = True,
):
    scale = 2 ** zoom
    global_x = (tx * tile_size) + np.arange(tile_size, dtype=np.float32) + 0.5
    world_x = global_x / (tile_size * scale)
    longitudes = world_x * 360.0 - 180.0

    if wrap_x:
        src_x = ((longitudes + 180.0) / 360.0 * width) - 0.5
        src_x = np.mod(src_x, max(width, 1)).astype(np.float32)
    else:
        src_x = np.clip(
            ((longitudes + 180.0) / 360.0 * max(width, 1)) - 0.5,
            0,
            max(width - 1, 0),
        ).astype(np.float32)

    global_y = (ty * tile_size) + np.arange(tile_size, dtype=np.float32) + 0.5
    world_y = global_y / (tile_size * scale)
    mercator_n = np.pi * (1.0 - 2.0 * world_y)
    latitudes = np.degrees(np.arctan(np.sinh(mercator_n)))
    latitudes = np.clip(latitudes, -WEB_MERCATOR_MAX_LAT, WEB_MERCATOR_MAX_LAT)
    src_y = np.clip(
        ((90.0 - latitudes) / 180.0 * max(height, 1)) - 0.5,
        0,
        max(height - 1, 0),
    ).astype(np.float32)

    return src_x, src_y


def _sample_bilinear(source_rgb: np.ndarray, src_x: np.ndarray, src_y: np.ndarray, wrap_x: bool) -> np.ndarray:
    height, width = source_rgb.shape[:2]
    x0 = np.floor(src_x).astype(np.int32)
    y0 = np.floor(src_y).astype(np.int32)
    wx = (src_x - x0).astype(np.float32)
    wy = (src_y - y0).astype(np.float32)

    if wrap_x:
        x0 = x0 % max(width, 1)
        x1 = (x0 + 1) % max(width, 1)
    else:
        x0 = np.clip(x0, 0, max(width - 1, 0))
        x1 = np.clip(x0 + 1, 0, max(width - 1, 0))
    y0 = np.clip(y0, 0, max(height - 1, 0))
    y1 = np.clip(y0 + 1, 0, max(height - 1, 0))

    top = source_rgb[y0[:, None], x0[None, :]] * (1.0 - wx[None, :, None]) + source_rgb[
        y0[:, None], x1[None, :]
    ] * wx[None, :, None]
    bottom = source_rgb[y1[:, None], x0[None, :]] * (1.0 - wx[None, :, None]) + source_rgb[
        y1[:, None], x1[None, :]
    ] * wx[None, :, None]
    return top * (1.0 - wy[:, None, None]) + bottom * wy[:, None, None]


def render_xyz_tile(source_rgb: np.ndarray, zoom: int, tx: int, ty: int, tile_size: int, wrap_x: bool = True):
    height, width = source_rgb.shape[:2]
    src_x, src_y = mercator_tile_source_coordinates(width, height, zoom, tx, ty, tile_size, wrap_x=wrap_x)
    tile_rgb = _sample_bilinear(source_rgb.astype(np.float32), src_x, src_y, wrap_x=wrap_x)
    return Image.fromarray(tile_rgb.astype(np.uint8), mode='RGB')
