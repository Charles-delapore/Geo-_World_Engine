from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from app.config import settings
from app.storage.artifact_repo import ArtifactRepository

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/assets", tags=["assets"])
repo = ArtifactRepository()


class AssetInfo(BaseModel):
    asset_id: str
    asset_type: str
    filename: str
    size_bytes: int
    created_at: datetime


def _asset_dir(asset_id: str) -> Path:
    path = repo.root / "assets" / asset_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def _detect_asset_type(filename: str, content_type: str | None) -> str:
    ext = Path(filename).suffix.lower()
    type_map = {
        ".png": "image",
        ".jpg": "image",
        ".jpeg": "image",
        ".webp": "image",
        ".tif": "elevation",
        ".tiff": "elevation",
        ".geojson": "sketch",
        ".json": "sketch",
        ".npz": "mask",
    }
    return type_map.get(ext, "image")


@router.post("", response_model=AssetInfo, status_code=201)
async def upload_asset(file: UploadFile = File(...)):
    asset_id = str(uuid.uuid4())
    filename = file.filename or "upload"
    content_type = file.content_type
    asset_type = _detect_asset_type(filename, content_type)

    data = await file.read()
    size_bytes = len(data)

    asset_path = _asset_dir(asset_id) / filename
    asset_path.write_bytes(data)

    meta = {
        "asset_id": asset_id,
        "asset_type": asset_type,
        "filename": filename,
        "content_type": content_type,
        "size_bytes": size_bytes,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    meta_path = _asset_dir(asset_id) / "meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    return AssetInfo(
        asset_id=asset_id,
        asset_type=asset_type,
        filename=filename,
        size_bytes=size_bytes,
        created_at=datetime.now(timezone.utc),
    )


@router.get("/{asset_id}", response_model=AssetInfo)
def get_asset(asset_id: str):
    meta_path = _asset_dir(asset_id) / "meta.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="Asset not found")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return AssetInfo(
        asset_id=meta["asset_id"],
        asset_type=meta["asset_type"],
        filename=meta["filename"],
        size_bytes=meta["size_bytes"],
        created_at=datetime.fromisoformat(meta["created_at"]),
    )
