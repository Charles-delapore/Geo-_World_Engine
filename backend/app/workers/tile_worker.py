from __future__ import annotations

import numpy as np
from PIL import Image

from app.core.tile_projection import render_xyz_tile
from app.storage.artifact_repo import ArtifactRepository
from app.workers.render_worker import render_preview_from_arrays


def generate_tiles(task_id: str, repo: ArtifactRepository, preview: Image.Image, max_zoom: int = 0) -> None:
    world = repo.load_world(task_id)
    source_rgb = np.asarray(render_preview_from_arrays(world, {}).convert("RGB"))

    for zoom in range(max_zoom + 1):
        scale = 2**zoom
        for tx in range(scale):
            for ty in range(scale):
                tile = render_xyz_tile(source_rgb, zoom, tx, ty, 256)
                repo.save_tile_image(task_id, zoom, tx, ty, tile)

    manifest = {
        "tile_url_template": f"/api/maps/{task_id}/tiles/{{z}}/{{x}}/{{y}}.png",
        "bounds": [-180, -85, 180, 85],
        "center": [0, 0],
        "min_zoom": 0,
        "max_zoom": max_zoom,
        "attribution": "Geo-WorldEngine Beta",
    }
    repo.save_manifest(task_id, manifest)
