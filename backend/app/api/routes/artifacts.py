from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response

from app.storage.artifact_repo import ArtifactRepository

router = APIRouter(prefix="/maps", tags=["artifacts"])
repo = ArtifactRepository()


@router.get("/{task_id}/preview.png")
def get_preview(task_id: str):
    if not repo.has_preview(task_id):
        raise HTTPException(status_code=404, detail="Preview not found")
    return Response(content=repo.read_preview_bytes(task_id), media_type="image/png")


@router.get("/{task_id}/tiles/manifest.json")
def get_manifest(task_id: str, request: Request):
    if not repo.has_manifest(task_id):
        raise HTTPException(status_code=404, detail="Manifest not found")
    manifest = json.loads(repo.read_manifest_bytes(task_id).decode("utf-8"))
    forwarded_host = request.headers.get("x-forwarded-host") or request.headers.get("host")
    forwarded_proto = request.headers.get("x-forwarded-proto", "http")
    if forwarded_host:
        base = f"{forwarded_proto}://{forwarded_host}"
    else:
        base = str(request.base_url).rstrip("/")
    if manifest.get("tile_url_template", "").startswith("/"):
        manifest["tile_url_template"] = f"{base}{manifest['tile_url_template']}"
    version = request.query_params.get("v")
    if version and manifest.get("tile_url_template"):
        separator = "&" if "?" in manifest["tile_url_template"] else "?"
        manifest["tile_url_template"] = f"{manifest['tile_url_template']}{separator}v={version}"
    return Response(content=json.dumps(manifest, ensure_ascii=False), media_type="application/json")


@router.get("/{task_id}/tiles/{z}/{x}/{y}.png")
def get_tile(task_id: str, z: int, x: int, y: int):
    if repo.has_tile(task_id, z, x, y):
        return Response(content=repo.read_tile_bytes(task_id, z, x, y), media_type="image/png")
    return Response(content=repo.transparent_tile_bytes(), media_type="image/png")
