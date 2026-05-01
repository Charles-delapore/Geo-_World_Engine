from __future__ import annotations

import numpy as np
from PIL import Image


def render_preview_image(
    elevation: np.ndarray,
    biome: np.ndarray,
    temperature: np.ndarray,
    precipitation: np.ndarray,
    profile: dict,
) -> Image.Image:
    ocean = elevation < 0
    land = ~ocean

    palette = np.zeros((*elevation.shape, 3), dtype=np.float32)
    normalized = np.clip((elevation + 1.0) / 2.0, 0.0, 1.0)
    depth = np.clip((-elevation) / max(float(np.max(-elevation[ocean])) if np.any(ocean) else 1.0, 1e-6), 0.0, 1.0)
    moisture = np.clip(precipitation / max(float(np.max(precipitation)), 1.0), 0.0, 1.0)
    warmth = np.clip((temperature + 20.0) / 55.0, 0.0, 1.0)

    ocean_shallow = np.array([44, 127, 184], dtype=np.float32)
    ocean_deep = np.array([8, 46, 97], dtype=np.float32)
    palette[ocean] = ocean_shallow * (1.0 - depth[ocean, None]) + ocean_deep * depth[ocean, None]

    if profile.get("palette_hint") == "frozen":
        palette[ocean] = palette[ocean] * 0.72 + np.array([168, 200, 220], dtype=np.float32) * 0.28
    elif profile.get("palette_hint") == "tropical":
        palette[ocean] = palette[ocean] * 0.78 + np.array([42, 168, 198], dtype=np.float32) * 0.22

    beach = land & (elevation < 0.06)
    lowland = land & (elevation >= 0.06) & (elevation < 0.28)
    upland = land & (elevation >= 0.28) & (elevation < 0.58)
    alpine = land & (elevation >= 0.58)

    palette[beach] = np.array([220, 202, 155], dtype=np.float32)
    palette[lowland & (biome == 1)] = np.array([204, 170, 102], dtype=np.float32)
    palette[lowland & (biome == 2)] = np.array([153, 176, 96], dtype=np.float32)
    palette[lowland & (biome == 3)] = np.array([84, 140, 78], dtype=np.float32)
    palette[lowland & (biome == 4)] = np.array([126, 147, 112], dtype=np.float32)
    palette[lowland & (biome == 5)] = np.array([232, 238, 242], dtype=np.float32)
    palette[lowland & (biome == 6)] = np.array([41, 188, 86], dtype=np.float32)
    palette[lowland & (biome == 7)] = np.array([125, 203, 53], dtype=np.float32)
    palette[lowland & (biome == 8)] = np.array([64, 156, 67], dtype=np.float32)
    palette[lowland & (biome == 9)] = np.array([75, 107, 50], dtype=np.float32)
    palette[lowland & (biome == 10)] = np.array([150, 120, 75], dtype=np.float32)
    palette[lowland & (biome == 11)] = np.array([213, 231, 235], dtype=np.float32)
    palette[lowland & (biome == 12)] = np.array([11, 145, 49], dtype=np.float32)
    uncolored_lowland = lowland & (palette.sum(axis=2) == 0)
    palette[uncolored_lowland] = np.array([126, 147, 112], dtype=np.float32)

    palette[upland & (biome == 1)] = np.array([168, 128, 86], dtype=np.float32)
    palette[upland & (biome == 2)] = np.array([126, 151, 88], dtype=np.float32)
    palette[upland & (biome == 3)] = np.array([58, 108, 70], dtype=np.float32)
    palette[upland & (biome == 4)] = np.array([124, 134, 126], dtype=np.float32)
    palette[upland & (biome == 5)] = np.array([238, 242, 247], dtype=np.float32)
    palette[upland & (biome == 6)] = np.array([34, 152, 72], dtype=np.float32)
    palette[upland & (biome == 7)] = np.array([100, 172, 48], dtype=np.float32)
    palette[upland & (biome == 8)] = np.array([52, 130, 56], dtype=np.float32)
    palette[upland & (biome == 9)] = np.array([60, 88, 42], dtype=np.float32)
    palette[upland & (biome == 10)] = np.array([120, 96, 62], dtype=np.float32)
    palette[upland & (biome == 11)] = np.array([200, 218, 224], dtype=np.float32)
    palette[upland & (biome == 12)] = np.array([10, 120, 42], dtype=np.float32)
    uncolored_upland = upland & (palette.sum(axis=2) == 0)
    palette[uncolored_upland] = np.array([124, 134, 126], dtype=np.float32)

    palette[alpine] = np.array([118, 112, 106], dtype=np.float32)
    snowcaps = land & ((warmth < 0.22) | (elevation > 0.74))
    palette[snowcaps] = palette[snowcaps] * 0.28 + np.array([244, 246, 250], dtype=np.float32) * 0.72

    dry_bias = np.clip((0.52 - moisture) * 1.6, 0.0, 1.0)
    lush_bias = np.clip((moisture - 0.5) * 1.4, 0.0, 1.0)
    palette[land] = palette[land] * (1.0 - dry_bias[land, None] * 0.18) + np.array([205, 176, 113], dtype=np.float32) * (
        dry_bias[land, None] * 0.18
    )
    palette[land] = palette[land] * (1.0 - lush_bias[land, None] * 0.18) + np.array([73, 133, 82], dtype=np.float32) * (
        lush_bias[land, None] * 0.18
    )

    grad_y, grad_x = np.gradient(elevation.astype(np.float32))
    light = np.array([-0.7, -0.45, 0.55], dtype=np.float32)
    nx = -grad_x * 2.4
    ny = -grad_y * 2.4
    nz = np.ones_like(elevation, dtype=np.float32)
    norm = np.sqrt(nx * nx + ny * ny + nz * nz)
    nx /= norm
    ny /= norm
    nz /= norm
    hillshade = np.clip(nx * light[0] + ny * light[1] + nz * light[2], 0.0, 1.0)
    ambient = 0.58 + hillshade * 0.52
    palette *= ambient[..., None]

    coast_band = np.abs(elevation) < 0.03
    palette[coast_band] = palette[coast_band] * 0.45 + np.array([240, 229, 192], dtype=np.float32) * 0.55

    image = np.clip(palette, 0, 255).astype(np.uint8)
    return Image.fromarray(image, mode="RGB")


def render_preview_from_arrays(arrays: dict[str, np.ndarray], profile: dict | None = None) -> Image.Image:
    return render_preview_image(
        arrays["elevation"],
        arrays.get("biome", np.zeros_like(arrays["elevation"], dtype=np.int16)),
        arrays.get("temperature", np.zeros_like(arrays["elevation"])),
        arrays.get("precipitation", np.zeros_like(arrays["elevation"])),
        profile or {},
    )
