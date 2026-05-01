from __future__ import annotations

import logging
from datetime import datetime, timezone

import numpy as np
from PIL import Image

from app.core.geometry_metrics import compute_metric_report
from app.core.preview_renderer import render_preview_from_arrays
from app.orchestrator.artifact_orchestrator import ArtifactOrchestrator
from app.pipelines.base import TerrainBundle
from app.storage.artifact_repo import ArtifactRepository
from app.storage.models import MapVersion, TaskRecord, session_scope
from app.utils.helpers import safe_dict

logger = logging.getLogger(__name__)
repo = ArtifactRepository()


class EditorOrchestrator:
    def __init__(self, task_id: str, plan: dict, projection: str = "flat"):
        self.task_id = task_id
        self.plan = plan
        self.projection = projection
        self._artifact_orch = ArtifactOrchestrator(task_id=task_id, projection=projection)

    def load_elevation(self) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        world = repo.load_world(self.task_id)
        elevation = world.get("elevation")
        if elevation is None:
            elevation = repo.load_cog(self.task_id)
        return elevation.astype(np.float32), world

    def get_version_history(self) -> list[dict]:
        with session_scope() as db:
            task = db.get(TaskRecord, self.task_id)
            if task is None:
                return []
            versions = db.query(MapVersion).filter(MapVersion.task_id == self.task_id).order_by(MapVersion.version_num).all()
            if not versions:
                return [{
                    "versionNum": 0,
                    "editSummary": "Initial generation",
                    "editType": "generation",
                    "parentVersion": None,
                    "createdAt": task.created_at,
                }]
            return [
                {
                    "versionNum": v.version_num,
                    "editSummary": v.edit_summary,
                    "editType": v.edit_type,
                    "parentVersion": v.parent_version,
                    "createdAt": v.created_at,
                }
                for v in versions
            ]

    def get_next_version(self) -> int:
        with session_scope() as db:
            max_version = (
                db.query(MapVersion)
                .filter(MapVersion.task_id == self.task_id)
                .order_by(MapVersion.version_num.desc())
                .first()
            )
            return (max_version.version_num + 1) if max_version else 1

    def apply_edit(
        self,
        elevation: np.ndarray,
        world: dict[str, np.ndarray],
        instructions: list[dict] | None = None,
        text_instruction: str | None = None,
        sketch_geojson: dict | None = None,
        edit_summary: str = "",
        edit_type: str = "constraints",
    ) -> tuple[np.ndarray, int, list[dict]]:
        from app.core.incremental_editor import EditInstruction, IncrementalEditor

        h, w = elevation.shape
        editor = IncrementalEditor(elevation, self.plan)

        if instructions:
            instruction = EditInstruction(instructions, (h, w))
            editor.add_constraints_from_instruction(instruction)

        if text_instruction:
            text_instruction_obj = EditInstruction.from_text(text_instruction, (h, w), elevation)
            editor.add_constraints_from_instruction(text_instruction_obj)

        if sketch_geojson:
            from app.core.incremental_editor import _geojson_to_mask
            sketch_mask = _geojson_to_mask(sketch_geojson, (h, w))
            editor.add_constraint(__import__("app.core.incremental_editor", fromlist=["EditConstraint"]).EditConstraint(
                constraint_type=__import__("app.core.incremental_editor", fromlist=["EditConstraintType"]).EditConstraintType.ELEVATION_OFFSET,
                mask=sketch_mask,
                value=0.1,
            ))

        old_elevation = elevation.copy()
        new_elevation = editor.apply_constraints()
        dirty_bounds = self._compute_dirty_bounds(old_elevation, new_elevation)

        profile = safe_dict(self.plan.get("profile"))
        updated_world = dict(world)
        updated_world["elevation"] = new_elevation.astype(np.float32)

        bundle = TerrainBundle.from_arrays(updated_world, self.projection)
        preview = render_preview_from_arrays(updated_world, profile)

        next_version = self._create_version_record(edit_summary or "Edit", edit_type)
        self._artifact_orch.save_version_full(
            version_num=next_version,
            bundle=bundle,
            preview=preview,
            plan=self.plan,
            edit_summary=edit_summary or "Edit",
        )
        self._artifact_orch.save_terrain_bundle(bundle)
        self._artifact_orch.build_and_save_preview(bundle, profile)

        return new_elevation, next_version, dirty_bounds

    def apply_sketch(
        self,
        elevation: np.ndarray,
        world: dict[str, np.ndarray],
        geojson: dict,
        instruction_type: str = "elevation_offset",
        brush_size: int = 5,
    ) -> tuple[np.ndarray, int, list[dict]]:
        from app.core.incremental_editor import EditInstruction, IncrementalEditor, _geojson_to_mask

        h, w = elevation.shape
        sketch_mask = _geojson_to_mask(geojson, (h, w), brush_radius=brush_size)

        points = self._extract_points_from_geojson(geojson)
        pixel_points = [self._lonlat_to_pixel(p, (h, w)) for p in points]

        instructions = []
        if instruction_type == "mountain":
            instructions = [{"type": "mountain_ridge", "height": 0.15, "sharpness": 0.5, "points": pixel_points}]
        elif instruction_type == "river":
            instructions = [{"type": "river_vector", "depth": 0.12, "width": max(1, brush_size // 2), "points": pixel_points}]
        elif instruction_type == "lake":
            instructions = [{"type": "lake_mask", "mask": sketch_mask}]
        elif instruction_type == "roughen":
            instructions = [{"type": "roughness_adjust", "target_roughness": 0.5, "mask": sketch_mask}]
        elif instruction_type == "lower":
            instructions = [{"type": "elevation_offset", "value": -0.1, "mask": sketch_mask}]
        elif instruction_type == "flatten":
            new_elevation = self._flatten_region(elevation, sketch_mask)
            dirty_bounds = self._compute_dirty_bounds(elevation, new_elevation)
            profile = safe_dict(self.plan.get("profile"))
            updated_world = dict(world)
            updated_world["elevation"] = new_elevation.astype(np.float32)
            bundle = TerrainBundle.from_arrays(updated_world, self.projection)
            preview = render_preview_from_arrays(updated_world, profile)
            next_version = self._create_version_record(f"Brush {instruction_type}", "sketch")
            self._artifact_orch.save_version_full(next_version, bundle, preview, self.plan, f"Brush {instruction_type}")
            self._artifact_orch.save_terrain_bundle(bundle)
            self._artifact_orch.build_and_save_preview(bundle, profile)
            return new_elevation, next_version, dirty_bounds
        else:
            instructions = [{"type": "elevation_offset", "value": 0.1, "mask": sketch_mask}]

        old_elevation = elevation.copy()
        editor = IncrementalEditor(elevation, self.plan)
        editor.add_constraints_from_instruction(EditInstruction(instructions, (h, w)))
        new_elevation = editor.apply_constraints()
        dirty_bounds = self._compute_dirty_bounds(old_elevation, new_elevation)

        profile = safe_dict(self.plan.get("profile"))
        updated_world = dict(world)
        updated_world["elevation"] = new_elevation.astype(np.float32)
        bundle = TerrainBundle.from_arrays(updated_world, self.projection)
        preview = render_preview_from_arrays(updated_world, profile)
        next_version = self._create_version_record(f"Brush {instruction_type}", "sketch")
        self._artifact_orch.save_version_full(next_version, bundle, preview, self.plan, f"Brush {instruction_type}")
        self._artifact_orch.save_terrain_bundle(bundle)
        self._artifact_orch.build_and_save_preview(bundle, profile)

        return new_elevation, next_version, dirty_bounds

    def revert_to_version(self, version_num: int) -> tuple[np.ndarray, list[dict]]:
        if version_num == 0:
            world = repo.load_world(self.task_id)
            elevation = world.get("elevation")
            if elevation is None:
                raise FileNotFoundError("Original elevation not found")
            return elevation.astype(np.float32), []

        version_data = repo.load_version_full(self.task_id, version_num)
        elevation = version_data.get("elevation")
        if elevation is None:
            elevation, _ = repo.load_version(self.task_id, version_num)

        current_world = repo.load_world(self.task_id)
        current_elev = current_world.get("elevation", np.zeros_like(elevation))
        dirty_bounds = self._compute_dirty_bounds(current_elev, elevation)

        profile = safe_dict(self.plan.get("profile"))
        updated_world = dict(current_world)
        updated_world["elevation"] = elevation.astype(np.float32)
        bundle = TerrainBundle.from_arrays(updated_world, self.projection)
        preview = render_preview_from_arrays(updated_world, profile)

        next_version = self._create_version_record(f"Reverted to version {version_num}", "revert")
        self._artifact_orch.save_version_full(next_version, bundle, preview, self.plan, f"Reverted to version {version_num}")
        self._artifact_orch.save_terrain_bundle(bundle)
        self._artifact_orch.build_and_save_preview(bundle, profile)

        return elevation.astype(np.float32), dirty_bounds

    def _create_version_record(self, edit_summary: str, edit_type: str) -> int:
        with session_scope() as db:
            task = db.get(TaskRecord, self.task_id)
            if task is None:
                raise ValueError(f"Task {self.task_id} not found")
            max_version = (
                db.query(MapVersion)
                .filter(MapVersion.task_id == self.task_id)
                .order_by(MapVersion.version_num.desc())
                .first()
            )
            next_version = (max_version.version_num + 1) if max_version else 1
            db.add(MapVersion(
                task_id=self.task_id,
                version_num=next_version,
                edit_summary=edit_summary,
                edit_type=edit_type,
                parent_version=next_version - 1 if next_version > 1 else None,
            ))
            db.commit()
            return next_version

    @staticmethod
    def _compute_dirty_bounds(old_elevation: np.ndarray, new_elevation: np.ndarray, threshold: float = 0.005) -> list[dict]:
        diff = np.abs(new_elevation.astype(np.float32) - old_elevation.astype(np.float32))
        changed = diff > threshold
        if not np.any(changed):
            return []

        from scipy.ndimage import label as ndimage_label
        h, w = old_elevation.shape
        labeled, num_features = ndimage_label(changed.astype(np.int8))
        bounds = []
        for i in range(1, num_features + 1):
            comp = labeled == i
            rows = np.any(comp, axis=1)
            cols = np.any(comp, axis=0)
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            pad = max(16, min(w, h) // 32)
            bounds.append({
                "min_y": max(0, int(rmin) - pad),
                "min_x": max(0, int(cmin) - pad),
                "max_y": min(h - 1, int(rmax) + pad),
                "max_x": min(w - 1, int(cmax) + pad),
                "_crosses_antimeridian": False,
                "_img_width": w,
                "_img_height": h,
            })

        if len(bounds) > 1:
            merged = bounds[0]
            for b in bounds[1:]:
                merged["min_y"] = min(merged["min_y"], b["min_y"])
                merged["min_x"] = min(merged["min_x"], b["min_x"])
                merged["max_y"] = max(merged["max_y"], b["max_y"])
                merged["max_x"] = max(merged["max_x"], b["max_x"])
                merged["_img_width"] = w
                merged["_img_height"] = h
            return [merged]

        for b in bounds:
            b.pop("_crosses_antimeridian", None)
            b.pop("_img_width", None)
            b.pop("_img_height", None)
        return bounds

    @staticmethod
    def _flatten_region(elevation: np.ndarray, mask: np.ndarray) -> np.ndarray:
        new_elevation = elevation.copy()
        region = mask > 0.05
        if np.any(region):
            target = float(np.median(elevation[region]))
            blend = np.clip(mask, 0.0, 1.0)
            new_elevation = (new_elevation * (1.0 - blend) + target * blend).astype(np.float32)
        return new_elevation

    @staticmethod
    def _lonlat_to_pixel(point: list[float] | tuple[float, float], shape: tuple[int, int]) -> list[int]:
        h, w = shape
        lon, lat = float(point[0]), float(point[1])
        x = int(np.clip((lon + 180.0) / 360.0 * (w - 1), 0, w - 1))
        y = int(np.clip((90.0 - lat) / 180.0 * (h - 1), 0, h - 1))
        return [y, x]

    @staticmethod
    def _extract_points_from_geojson(geojson: dict) -> list[list[float]]:
        geometry = geojson.get("geometry", geojson)
        geom_type = geometry.get("type", "")
        coords = []
        if geom_type == "LineString":
            coords = geometry.get("coordinates", [])
        elif geom_type == "MultiLineString":
            for line in geometry.get("coordinates", []):
                coords.extend(line)
        elif geom_type == "Polygon":
            coords = geometry.get("coordinates", [[]])[0]
        elif geom_type == "Feature":
            return EditorOrchestrator._extract_points_from_geojson(geometry)
        elif geom_type == "FeatureCollection":
            for feat in geometry.get("features", []):
                coords.extend(EditorOrchestrator._extract_points_from_geojson(feat))
        return coords
