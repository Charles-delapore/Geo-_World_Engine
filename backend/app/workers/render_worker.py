from __future__ import annotations

import logging

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def make_preview_longitude_seamless(image: Image.Image, band_width: int | None = None) -> Image.Image:
    from scipy.ndimage import gaussian_filter

    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    height, width = rgb.shape[:2]
    band = min(max(2, int(band_width or width // 96)), max(2, width // 8))
    if band * 2 >= width:
        return image.convert("RGB")

    padded = np.concatenate([rgb[:, -band:], rgb, rgb[:, :band]], axis=1)
    periodic = gaussian_filter(padded, sigma=(0.0, 1.0, 0.0)).astype(np.float32)[:, band:-band]

    weights = np.zeros((height, width, 1), dtype=np.float32)
    ramp = np.linspace(1.0, 0.0, band, dtype=np.float32)
    weights[:, :band, 0] = ramp
    weights[:, -band:, 0] = ramp[::-1]

    relaxed = rgb * (1.0 - weights * 0.42) + periodic * (weights * 0.42)
    seam = (relaxed[:, 0, :] + relaxed[:, -1, :]) * 0.5
    relaxed[:, 0, :] = seam
    relaxed[:, -1, :] = seam
    return Image.fromarray(np.clip(relaxed, 0, 255).astype(np.uint8), mode="RGB")


def render_world(
    plan: dict,
    width: int,
    height: int,
    seed: int,
    emit_debug_artifacts: bool = False,
    task_id: str | None = None,
    projection: str = "planet",
) -> tuple[dict[str, np.ndarray], Image.Image]:
    from app.orchestrator.orchestrator import render_world as _render_world
    return _render_world(
        plan, width=width, height=height, seed=seed,
        emit_debug_artifacts=emit_debug_artifacts,
        task_id=task_id, projection=projection,
    )
