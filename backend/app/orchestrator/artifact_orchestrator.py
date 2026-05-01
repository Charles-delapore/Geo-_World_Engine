from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from PIL import Image

from app.config import settings
from app.core.preview_renderer import render_preview_from_arrays
from app.pipelines.base import ArtifactManifest, TerrainBundle
from app.storage.artifact_repo import ArtifactRepository
from app.workers.tile_worker import generate_tiles, regenerate_dirty_tiles

logger = logging.getLogger(__name__)
repo = ArtifactRepository()


class ArtifactOrchestrator:
    def __init__(self, task_id: str, projection: str = "flat"):
        self.task_id = task_id
        self.projection = projection

    def save_terrain_bundle(self, bundle: TerrainBundle) -> None:
        repo.save_world(self.task_id, **bundle.to_arrays())
        try:
            repo.save_cog(self.task_id, bundle.elevation)
        except Exception as exc:
            logger.warning("COG save skipped: %s", exc)

    def save_version_full(
        self,
        version_num: int,
        bundle: TerrainBundle,
        preview: Image.Image,
        plan: dict,
        edit_summary: str = "",
    ) -> None:
        manifest = self._build_manifest_for_version(version_num, plan)
        repo.save_version_full(
            self.task_id,
            version_num=version_num,
            elevation=bundle.elevation,
            preview=preview,
            manifest=manifest,
            edit_summary=edit_summary,
            world_arrays=bundle.to_arrays(),
        )

    def save_version(self, version_num: int, elevation: np.ndarray, edit_summary: str = "") -> None:
        from app.pipelines.base import TerrainBundle
        world = repo.load_world(self.task_id)
        world["elevation"] = elevation.astype(np.float32)
        profile = {}
        try:
            import json as _json
            manifest_bytes = repo.read_manifest_bytes(self.task_id)
            manifest_data = _json.loads(manifest_bytes.decode("utf-8"))
            profile = manifest_data.get("profile") or {}
        except Exception:
            pass
        bundle = TerrainBundle.from_arrays(world, self.projection)
        preview = self.build_and_save_preview(bundle, profile)
        self.save_version_full(version_num, bundle, preview, {"profile": profile}, edit_summary)

    def build_and_save_preview(self, bundle: TerrainBundle, profile: dict) -> Image.Image:
        preview = render_preview_from_arrays(bundle.to_arrays(), profile)
        repo.save_preview(self.task_id, preview)
        return preview

    def build_and_save_tiles(self, profile: dict, max_zoom: int | None = None) -> None:
        zoom = max_zoom or settings.MAX_TILE_ZOOM
        preview = Image.open(Path(repo.preview_path(self.task_id))).convert("RGB")
        generate_tiles(
            self.task_id, repo, preview,
            max_zoom=zoom,
            projection=self.projection,
            profile=profile,
        )

    def regenerate_dirty_tiles(self, dirty_bounds: list[dict], profile: dict) -> int:
        if not dirty_bounds:
            return 0
        return regenerate_dirty_tiles(
            self.task_id, repo, dirty_bounds,
            projection=self.projection,
            profile=profile,
        )

    def refresh_tiles(self, dirty_bounds: list[dict], profile: dict) -> int:
        if not dirty_bounds:
            logger.info("refresh_tiles: no dirty_bounds, scheduling full tile generation for task=%s", self.task_id)
            self._schedule_full_tile_refresh("no_dirty_bounds")
            return 0

        try:
            regenerated = self.regenerate_dirty_tiles(dirty_bounds, profile)
        except Exception as exc:
            logger.warning(
                "refresh_tiles: regenerate_dirty_tiles failed for task=%s, falling back to full: %s",
                self.task_id, exc,
            )
            self._schedule_full_tile_refresh(f"dirty_tile_error:{exc}")
            return 0

        if regenerated == 0:
            logger.info(
                "refresh_tiles: dirty tile regeneration returned 0 tiles for task=%s with %d bounds, falling back to full",
                self.task_id, len(dirty_bounds),
            )
            self._schedule_full_tile_refresh("zero_tiles_regenerated")
            return 0

        logger.info(
            "refresh_tiles: regenerated %d dirty tiles for task=%s from %d bounds",
            regenerated, self.task_id, len(dirty_bounds),
        )
        return regenerated

    def _schedule_full_tile_refresh(self, reason: str) -> None:
        logger.info("Scheduling full tile refresh for task=%s reason=%s", self.task_id, reason)
        try:
            from app.orchestrator.orchestrator import _run_local_async, generate_tiles_for_task
            _run_local_async(generate_tiles_for_task, self.task_id)
        except Exception as exc:
            logger.error(
                "Full tile refresh scheduling also failed for task=%s reason=%s: %s",
                self.task_id, reason, exc,
            )

    def build_artifact_manifest(self, version: int) -> ArtifactManifest:
        manifest_data = {}
        try:
            manifest_bytes = repo.read_manifest_bytes(self.task_id)
            manifest_data = json.loads(manifest_bytes.decode("utf-8"))
        except Exception:
            pass

        return ArtifactManifest(
            version=version,
            projection=self.projection,
            bounds=manifest_data.get("bounds", [-180, -85, 180, 85] if self.projection == "flat" else [-180, -90, 180, 90]),
            min_zoom=manifest_data.get("min_zoom", 0),
            max_zoom=manifest_data.get("max_zoom", settings.MAX_TILE_ZOOM),
            tiling_scheme=manifest_data.get("tiling_scheme", "web_mercator" if self.projection == "flat" else "geographic"),
            wrap_x=manifest_data.get("wrap_x", self.projection == "planet"),
        )

    def save_metric_report(self, metric_report: dict) -> None:
        from app.storage.models import TaskRecord, session_scope
        metric_str = json.dumps(metric_report, ensure_ascii=False)
        with session_scope() as db:
            task = db.get(TaskRecord, self.task_id)
            if task is not None:
                task.metric_report = metric_str

    def _build_manifest_for_version(self, version: int, plan: dict) -> dict:
        profile = plan.get("profile") or {}
        topology_intent = plan.get("topology_intent") or {}
        manifest = self.build_artifact_manifest(version)
        manifest_dict = manifest.to_dict()
        manifest_dict["edit_summary"] = ""
        manifest_dict["topology_kind"] = topology_intent.get("kind", "")
        manifest_dict["land_ratio"] = float(profile.get("land_ratio", 0.44))
        return manifest_dict
