from __future__ import annotations

import io
import json
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
import zarr
from zarr.storage import DirectoryStore, ZipStore

from app.config import settings
from app.storage.s3_client import s3_client


class ArtifactRepository:
    def __init__(self, root: Path | None = None):
        self.root = Path(root or settings.ARTIFACT_ROOT)
        self.root.mkdir(parents=True, exist_ok=True)

    def task_dir(self, task_id: str) -> Path:
        path = self.root / "maps" / task_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def preview_path(self, task_id: str) -> Path:
        return self.task_dir(task_id) / "preview.png"

    def manifest_path(self, task_id: str) -> Path:
        path = self.task_dir(task_id) / "tiles"
        path.mkdir(parents=True, exist_ok=True)
        return path / "manifest.json"

    def tile_path(self, task_id: str, z: int, x: int, y: int) -> Path:
        path = self.task_dir(task_id) / "tiles" / str(z) / str(x)
        path.mkdir(parents=True, exist_ok=True)
        return path / f"{y}.png"

    def world_path(self, task_id: str) -> Path:
        return self.task_dir(task_id) / "world.zarr"

    def legacy_world_path(self, task_id: str) -> Path:
        return self.task_dir(task_id) / "world.npz"

    def _uses_s3(self) -> bool:
        return settings.ARTIFACT_BACKEND.lower() == "s3"

    def _object_key(self, *parts: str) -> str:
        return "/".join(part.strip("/\\") for part in parts)

    def save_preview(self, task_id: str, image: Image.Image) -> Path:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        data = buffer.getvalue()
        if self._uses_s3():
            s3_client.put_bytes(self._object_key("maps", task_id, "preview.png"), data, "image/png")
            return self.preview_path(task_id)
        path = self.preview_path(task_id)
        path.write_bytes(data)
        return path

    def save_world(self, task_id: str, **arrays: np.ndarray) -> Path:
        if self._uses_s3():
            with tempfile.NamedTemporaryFile(suffix=".zarr.zip", delete=False) as temp_file:
                temp_path = Path(temp_file.name)
            try:
                with ZipStore(str(temp_path), mode="w") as store:
                    group = zarr.group(store=store, overwrite=True)
                    for name, array in arrays.items():
                        group.array(name, data=np.asarray(array), chunks=np.asarray(array).shape, overwrite=True)
                s3_client.put_bytes(
                    self._object_key("maps", task_id, "world.zarr.zip"),
                    temp_path.read_bytes(),
                    "application/octet-stream",
                )
            finally:
                temp_path.unlink(missing_ok=True)
            return self.world_path(task_id)

        path = self.world_path(task_id)
        if path.exists():
            import shutil

            shutil.rmtree(path)
        store = DirectoryStore(str(path))
        group = zarr.group(store=store, overwrite=True)
        for name, array in arrays.items():
            np_array = np.asarray(array)
            chunks = tuple(min(512, dim) for dim in np_array.shape)
            group.array(name, data=np_array, chunks=chunks, overwrite=True)
        return path

    def save_manifest(self, task_id: str, manifest: dict) -> Path:
        data = json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8")
        if self._uses_s3():
            s3_client.put_bytes(self._object_key("maps", task_id, "tiles", "manifest.json"), data, "application/json")
            return self.manifest_path(task_id)
        path = self.manifest_path(task_id)
        path.write_bytes(data)
        return path

    def save_tile_image(self, task_id: str, z: int, x: int, y: int, image: Image.Image) -> Path:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        data = buffer.getvalue()
        if self._uses_s3():
            s3_client.put_bytes(self._object_key("maps", task_id, "tiles", str(z), str(x), f"{y}.png"), data, "image/png")
            return self.tile_path(task_id, z, x, y)
        path = self.tile_path(task_id, z, x, y)
        path.write_bytes(data)
        return path

    def load_world(self, task_id: str) -> dict[str, np.ndarray]:
        if self._uses_s3():
            with tempfile.NamedTemporaryFile(suffix=".zarr.zip", delete=False) as temp_file:
                temp_path = Path(temp_file.name)
                temp_file.write(s3_client.get_bytes(self._object_key("maps", task_id, "world.zarr.zip")))
            try:
                with ZipStore(str(temp_path), mode="r") as store:
                    group = zarr.open_group(store=store, mode="r")
                    return {key: np.asarray(group[key]) for key in group.array_keys()}
            finally:
                temp_path.unlink(missing_ok=True)

        path = self.world_path(task_id)
        if path.exists():
            group = zarr.open_group(store=DirectoryStore(str(path)), mode="r")
            return {key: np.asarray(group[key]) for key in group.array_keys()}

        legacy_path = self.legacy_world_path(task_id)
        with np.load(legacy_path) as data:
            return {key: data[key] for key in data.files}

    def has_preview(self, task_id: str) -> bool:
        if self._uses_s3():
            return s3_client.exists(self._object_key("maps", task_id, "preview.png"))
        return self.preview_path(task_id).exists()

    def has_manifest(self, task_id: str) -> bool:
        if self._uses_s3():
            return s3_client.exists(self._object_key("maps", task_id, "tiles", "manifest.json"))
        return self.manifest_path(task_id).exists()

    def has_tile(self, task_id: str, z: int, x: int, y: int) -> bool:
        if self._uses_s3():
            return s3_client.exists(self._object_key("maps", task_id, "tiles", str(z), str(x), f"{y}.png"))
        return self.tile_path(task_id, z, x, y).exists()

    def read_preview_bytes(self, task_id: str) -> bytes:
        if self._uses_s3():
            return s3_client.get_bytes(self._object_key("maps", task_id, "preview.png"))
        return self.preview_path(task_id).read_bytes()

    def read_manifest_bytes(self, task_id: str) -> bytes:
        if self._uses_s3():
            return s3_client.get_bytes(self._object_key("maps", task_id, "tiles", "manifest.json"))
        return self.manifest_path(task_id).read_bytes()

    def read_tile_bytes(self, task_id: str, z: int, x: int, y: int) -> bytes:
        if self._uses_s3():
            return s3_client.get_bytes(self._object_key("maps", task_id, "tiles", str(z), str(x), f"{y}.png"))
        return self.tile_path(task_id, z, x, y).read_bytes()

    def transparent_tile_bytes(self, size: int | None = None) -> bytes:
        image = Image.new("RGBA", (size or settings.DEFAULT_TILE_SIZE, size or settings.DEFAULT_TILE_SIZE), (0, 0, 0, 0))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

    def debug_dir(self, task_id: str) -> Path:
        path = self.task_dir(task_id) / "debug"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save_debug_image(self, task_id: str, name: str, image: Image.Image) -> Path:
        path = self.debug_dir(task_id) / f"{name}.png"
        if self._uses_s3():
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            s3_client.put_bytes(
                self._object_key("maps", task_id, "debug", f"{name}.png"),
                buffer.getvalue(),
                "image/png",
            )
            return path
        image.save(path, format="PNG")
        return path

    def save_debug_json(self, task_id: str, name: str, data: dict) -> Path:
        path = self.debug_dir(task_id) / f"{name}.json"
        encoded = json.dumps(data, ensure_ascii=False, indent=2, default=str).encode("utf-8")
        if self._uses_s3():
            s3_client.put_bytes(
                self._object_key("maps", task_id, "debug", f"{name}.json"),
                encoded,
                "application/json",
            )
            return path
        path.write_bytes(encoded)
        return path

    def save_debug_array(self, task_id: str, name: str, array: np.ndarray) -> Path:
        path = self.debug_dir(task_id) / f"{name}.npz"
        if self._uses_s3():
            buffer = io.BytesIO()
            np.savez_compressed(buffer, **{name: array})
            s3_client.put_bytes(
                self._object_key("maps", task_id, "debug", f"{name}.npz"),
                buffer.getvalue(),
                "application/octet-stream",
            )
            return path
        np.savez_compressed(path, **{name: array})
        return path
