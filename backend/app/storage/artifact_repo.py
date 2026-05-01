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

    def _save_zarr(self, path: Path, **arrays: np.ndarray) -> Path:
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

    def _load_zarr(self, path: Path) -> dict[str, np.ndarray]:
        group = zarr.open_group(store=DirectoryStore(str(path)), mode="r")
        return {key: np.asarray(group[key]) for key in group.array_keys()}

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

        return self._save_zarr(self.world_path(task_id), **arrays)

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
            return self._load_zarr(path)

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

    def cog_path(self, task_id: str) -> Path:
        return self.task_dir(task_id) / "terrain.tif"

    def save_cog(self, task_id: str, elevation: np.ndarray) -> Path:
        from app.core.terrain_io import elevation_to_cog
        path = self.cog_path(task_id)
        return elevation_to_cog(elevation, path)

    def load_cog(self, task_id: str) -> np.ndarray:
        from app.core.terrain_io import geotiff_to_elevation
        path = self.cog_path(task_id)
        if path.exists():
            return geotiff_to_elevation(path)
        world = self.load_world(task_id)
        return world.get("elevation", np.zeros((256, 512), dtype=np.float32))

    def has_cog(self, task_id: str) -> bool:
        return self.cog_path(task_id).exists()

    def version_dir(self, task_id: str) -> Path:
        path = self.task_dir(task_id) / "versions"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def immutable_version_dir(self, task_id: str, version_num: int) -> Path:
        vdir = self.task_dir(task_id) / "versions" / f"v{version_num:04d}"
        vdir.mkdir(parents=True, exist_ok=True)
        return vdir

    def save_version_full(
        self,
        task_id: str,
        version_num: int,
        elevation: np.ndarray,
        preview: Image.Image | None = None,
        manifest: dict | None = None,
        edit_summary: str = "",
        world_arrays: dict[str, np.ndarray] | None = None,
    ) -> Path:
        vdir = self.immutable_version_dir(task_id, version_num)
        np.savez_compressed(vdir / "elevation.npz", elevation=elevation, edit_summary=edit_summary)
        if world_arrays:
            self._save_zarr(vdir / "world.zarr", **world_arrays)
        if preview is not None:
            preview.save(vdir / "preview.png", format="PNG")
        if manifest is not None:
            (vdir / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        return vdir

    def load_version_full(self, task_id: str, version_num: int) -> dict:
        vdir = self.immutable_version_dir(task_id, version_num)
        if not vdir.exists():
            raise FileNotFoundError(f"Version {version_num} not found for task {task_id}")
        result: dict = {"version_num": version_num}
        elev_path = vdir / "elevation.npz"
        if elev_path.exists():
            with np.load(elev_path, allow_pickle=False) as data:
                result["elevation"] = data["elevation"]
                result["edit_summary"] = str(data.get("edit_summary", ""))
        world_path = vdir / "world.zarr"
        if world_path.exists():
            result["world"] = self._load_zarr(world_path)
        preview_path = vdir / "preview.png"
        if preview_path.exists():
            result["preview"] = Image.open(preview_path).convert("RGB")
        manifest_path = vdir / "manifest.json"
        if manifest_path.exists():
            result["manifest"] = json.loads(manifest_path.read_text(encoding="utf-8"))
        return result

    def save_version(self, task_id: str, version_num: int, elevation: np.ndarray, edit_summary: str = "") -> Path:
        vdir = self.version_dir(task_id)
        vdir.mkdir(parents=True, exist_ok=True)
        path = vdir / f"v{version_num:04d}.npz"
        np.savez_compressed(path, elevation=elevation, edit_summary=edit_summary)
        return path

    def load_version(self, task_id: str, version_num: int) -> tuple[np.ndarray, str]:
        path = self.version_dir(task_id) / f"v{version_num:04d}.npz"
        if not path.exists():
            raise FileNotFoundError(f"Version {version_num} not found for task {task_id}")
        with np.load(path, allow_pickle=False) as data:
            return data["elevation"], str(data.get("edit_summary", ""))

    def list_versions(self, task_id: str) -> list[int]:
        vdir = self.version_dir(task_id)
        if not vdir.exists():
            return []
        versions = []
        for f in vdir.iterdir():
            if f.name.startswith("v") and f.suffix == ".npz":
                try:
                    num = int(f.stem[1:])
                    versions.append(num)
                except ValueError:
                    pass
        return sorted(versions)

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
