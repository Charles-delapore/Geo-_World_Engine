from __future__ import annotations

import logging

import numpy as np
from PIL import Image

from app.core.tile_projection import render_xyz_tile
from app.core.planet_lod import render_geographic_tile
from app.core.preview_renderer import render_preview_from_arrays
from app.storage.artifact_repo import ArtifactRepository

logger = logging.getLogger(__name__)


def generate_tiles(
    task_id: str,
    repo: ArtifactRepository,
    preview: Image.Image,
    max_zoom: int = 0,
    projection: str = "planet",
    profile: dict | None = None,
) -> None:
    world = repo.load_world(task_id)
    wrap_x = projection == "planet"
    source_image = render_preview_from_arrays(world, profile or {}).convert("RGB")
    source_rgb = np.asarray(source_image)

    if projection == "planet":
        _generate_planet_tiles(task_id, repo, source_rgb, max_zoom)
    else:
        _generate_flat_tiles(task_id, repo, source_rgb, max_zoom, wrap_x)


def regenerate_dirty_tiles(
    task_id: str,
    repo: ArtifactRepository,
    dirty_bounds: list[dict],
    projection: str = "planet",
    profile: dict | None = None,
) -> int:
    if not dirty_bounds:
        return 0

    import time
    world = repo.load_world(task_id)
    elevation = world.get("elevation")
    if elevation is not None:
        repo.save_cog(task_id, elevation.astype(np.float32))
    time.sleep(0.05)

    world = repo.load_world(task_id)
    source_image = render_preview_from_arrays(world, profile or {}).convert("RGB")
    source_rgb = np.asarray(source_image)
    h, w = source_rgb.shape[:2]

    manifest_data = {}
    try:
        manifest_bytes = repo.read_manifest_bytes(task_id)
        manifest_data = __import__("json").loads(manifest_bytes.decode("utf-8"))
    except Exception:
        pass

    max_zoom = manifest_data.get("max_zoom", 6)
    regenerated = 0
    tiles_regenerated: set[tuple[int, int, int]] = set()

    for bounds in dirty_bounds:
        min_y = bounds.get("min_y", 0)
        min_x = bounds.get("min_x", 0)
        max_y = bounds.get("max_y", h - 1)
        max_x = bounds.get("max_x", w - 1)

        dirty_regions = _split_antimeridian_bounds(min_x, max_x, w)

        for (region_min_x, region_max_x) in dirty_regions:
            for zoom in range(max_zoom + 1):
                scale = 2 ** zoom
                tile_w = w / scale
                tile_h = h / scale

                if region_min_x > region_max_x:
                    continue

                tx_min = max(0, int(region_min_x / tile_w))
                tx_max = min(scale - 1, int(region_max_x / tile_w))
                ty_min = max(0, int(min_y / tile_h))
                ty_max = min(scale - 1, int(max_y / tile_h))

                for tx in range(tx_min, tx_max + 1):
                    for ty in range(ty_min, ty_max + 1):
                        tile_key = (zoom, tx, ty)
                        if tile_key in tiles_regenerated:
                            continue
                        tiles_regenerated.add(tile_key)
                        if projection == "planet":
                            tile = render_geographic_tile(source_rgb, zoom, tx, ty, 256, wrap_x=True)
                        else:
                            tile = render_xyz_tile(source_rgb, zoom, tx, ty, 256, wrap_x=False)
                        repo.save_tile_image(task_id, zoom, tx, ty, tile)
                        regenerated += 1

    try:
        manifest_data["_edit_timestamp"] = time.time()
        manifest_data["_regenerated_tiles"] = regenerated
        repo.save_manifest(task_id, manifest_data)
    except Exception as exc:
        logger.warning("Failed to update manifest timestamp: %s", exc)

    logger.info("regenerate_dirty_tiles: regenerated=%d tiles for task=%s", regenerated, task_id)
    return regenerated


def _split_antimeridian_bounds(min_x: int, max_x: int, img_width: int) -> list[tuple[int, int]]:
    if min_x <= max_x:
        return [(min_x, max_x)]
    left_region = (min_x, img_width - 1)
    right_region = (0, max_x)
    return [left_region, right_region]


def _generate_flat_tiles(
    task_id: str,
    repo: ArtifactRepository,
    source_rgb: np.ndarray,
    max_zoom: int,
    wrap_x: bool,
) -> None:
    for zoom in range(max_zoom + 1):
        scale = 2 ** zoom
        for tx in range(scale):
            for ty in range(scale):
                tile = render_xyz_tile(source_rgb, zoom, tx, ty, 256, wrap_x=wrap_x)
                repo.save_tile_image(task_id, zoom, tx, ty, tile)

    manifest = {
        "tile_url_template": f"/api/maps/{task_id}/tiles/{{z}}/{{x}}/{{y}}.png",
        "bounds": [-180, -85, 180, 85],
        "center": [0, 0],
        "min_zoom": 0,
        "max_zoom": max_zoom,
        "projection": "flat",
        "wrap_x": False,
        "tiling_scheme": "web_mercator",
        "attribution": "Geo-WorldEngine Beta",
    }
    repo.save_manifest(task_id, manifest)


def _generate_planet_tiles(
    task_id: str,
    repo: ArtifactRepository,
    source_rgb: np.ndarray,
    max_zoom: int,
) -> None:
    for zoom in range(max_zoom + 1):
        scale_x = 2 ** zoom
        scale_y = 2 ** zoom
        for tx in range(scale_x):
            for ty in range(scale_y):
                tile = render_geographic_tile(source_rgb, zoom, tx, ty, 256, wrap_x=True)
                repo.save_tile_image(task_id, zoom, tx, ty, tile)

    manifest = {
        "tile_url_template": f"/api/maps/{task_id}/tiles/{{z}}/{{x}}/{{y}}.png",
        "bounds": [-180, -90, 180, 90],
        "center": [0, 0],
        "min_zoom": 0,
        "max_zoom": max_zoom,
        "projection": "planet",
        "wrap_x": True,
        "tiling_scheme": "geographic",
        "level_zero_tiles_x": 1,
        "level_zero_tiles_y": 1,
        "attribution": "Geo-WorldEngine Beta",
    }
    repo.save_manifest(task_id, manifest)
