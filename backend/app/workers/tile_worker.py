from __future__ import annotations

from PIL import Image

from app.storage.artifact_repo import ArtifactRepository


def generate_tiles(task_id: str, repo: ArtifactRepository, preview: Image.Image, max_zoom: int = 0) -> None:
    tile = preview.resize((256, 256))
    repo.save_tile_image(task_id, 0, 0, 0, tile)
    manifest = {
        "tile_url_template": f"/api/maps/{task_id}/tiles/{{z}}/{{x}}/{{y}}.png",
        "bounds": [-180, -85, 180, 85],
        "center": [0, 0],
        "min_zoom": 0,
        "max_zoom": max_zoom,
        "attribution": "Geo-WorldEngine Beta",
    }
    repo.save_manifest(task_id, manifest)
