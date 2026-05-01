from __future__ import annotations

import logging
from enum import StrEnum
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter, distance_transform_edt, binary_dilation, generate_binary_structure

logger = logging.getLogger(__name__)


class EditConstraintType(StrEnum):
    ELEVATION_OFFSET = "elevation_offset"
    RIVER_VECTOR = "river_vector"
    MOUNTAIN_RIDGE = "mountain_ridge"
    LAKE_MASK = "lake_mask"
    ROUGHNESS_ADJUST = "roughness_adjust"


class EditConstraint:
    def __init__(
        self,
        constraint_type: EditConstraintType,
        mask: np.ndarray | None = None,
        value: float = 0.0,
        points: list[tuple[int, int]] | None = None,
        width: int = 3,
        depth: float = 0.1,
        height: float = 0.15,
        sharpness: float = 0.5,
        target_roughness: float = 0.5,
    ):
        self.constraint_type = constraint_type
        self.mask = mask
        self.value = value
        self.points = points or []
        self.width = width
        self.depth = depth
        self.height = height
        self.sharpness = sharpness
        self.target_roughness = target_roughness

    @classmethod
    def from_dict(cls, data: dict, shape: tuple[int, int]) -> EditConstraint:
        ctype = EditConstraintType(data.get("type", "elevation_offset"))
        mask = None
        if isinstance(data.get("mask"), np.ndarray):
            mask = data["mask"].astype(np.float32)
            if mask.shape != shape:
                mask = None
        elif "mask_base64" in data:
            import base64
            raw = base64.b64decode(data["mask_base64"])
            mask = np.frombuffer(raw, dtype=np.uint8).reshape(shape).astype(np.float32) / 255.0
        elif "mask_geojson" in data:
            mask = _geojson_to_mask(data["mask_geojson"], shape)

        return cls(
            constraint_type=ctype,
            mask=mask,
            value=float(data.get("value", 0.0)),
            points=data.get("points"),
            width=int(data.get("width", 3)),
            depth=float(data.get("depth", 0.1)),
            height=float(data.get("height", 0.15)),
            sharpness=float(data.get("sharpness", 0.5)),
            target_roughness=float(data.get("target_roughness", 0.5)),
        )


class EditInstruction:
    def __init__(self, instructions: list[dict], shape: tuple[int, int]):
        self.constraints: list[EditConstraint] = []
        for inst in instructions:
            self.constraints.append(EditConstraint.from_dict(inst, shape))

    @classmethod
    def from_text(cls, text: str, shape: tuple[int, int], current_elevation: np.ndarray) -> EditInstruction:
        constraints = _parse_edit_text(text, shape, current_elevation)
        obj = cls.__new__(cls)
        obj.constraints = constraints
        return obj


class IncrementalEditor:
    def __init__(self, original_elevation: np.ndarray, original_plan: dict | None = None):
        self.base = original_elevation.astype(np.float32).copy()
        self.plan = original_plan or {}
        self.constraints: list[EditConstraint] = []
        self.h, self.w = original_elevation.shape

    def add_constraint(self, constraint: EditConstraint) -> None:
        self.constraints.append(constraint)

    def add_constraints_from_instruction(self, instruction: EditInstruction) -> None:
        self.constraints.extend(instruction.constraints)

    def apply_constraints(self) -> np.ndarray:
        result = self.base.copy()

        global_constraints = [c for c in self.constraints if c.constraint_type == EditConstraintType.ELEVATION_OFFSET and c.mask is None]
        for c in global_constraints:
            result += c.value
            result = np.clip(result, -1.0, 1.0)

        mask_constraints = [c for c in self.constraints if c.constraint_type == EditConstraintType.ELEVATION_OFFSET and c.mask is not None]
        for c in mask_constraints:
            if c.mask is not None and c.mask.shape == result.shape:
                result += c.mask * c.value
                result = np.clip(result, -1.0, 1.0)

        for c in self.constraints:
            if c.constraint_type == EditConstraintType.RIVER_VECTOR and c.points:
                result = self._apply_river(result, c)
            elif c.constraint_type == EditConstraintType.MOUNTAIN_RIDGE and c.points:
                result = self._apply_ridge(result, c)
            elif c.constraint_type == EditConstraintType.LAKE_MASK and c.mask is not None:
                result = self._apply_lake(result, c)
            elif c.constraint_type == EditConstraintType.ROUGHNESS_ADJUST and c.mask is not None:
                result = self._apply_roughness(result, c)

        result = self._local_postprocess(result)
        return result.astype(np.float32)

    def _apply_river(self, elevation: np.ndarray, constraint: EditConstraint) -> np.ndarray:
        result = elevation.copy()
        points = constraint.points
        if len(points) < 2:
            return result

        river_mask = np.zeros((self.h, self.w), dtype=np.float32)
        for i in range(len(points) - 1):
            y0, x0 = points[i]
            y1, x1 = points[i + 1]
            n_steps = max(abs(y1 - y0), abs(x1 - x0), 1)
            for step in range(n_steps + 1):
                t = step / n_steps
                cy = int(y0 + (y1 - y0) * t)
                cx = int(x0 + (x1 - x0) * t)
                half_w = constraint.width // 2
                for dy in range(-half_w, half_w + 1):
                    for dx in range(-half_w, half_w + 1):
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < self.h and 0 <= nx < self.w:
                            dist = np.sqrt(dy * dy + dx * dx)
                            river_mask[ny, nx] = max(river_mask[ny, nx], max(0, 1.0 - dist / max(half_w, 1)))

        result -= river_mask * constraint.depth
        return np.clip(result, -1.0, 1.0)

    def _apply_ridge(self, elevation: np.ndarray, constraint: EditConstraint) -> np.ndarray:
        result = elevation.copy()
        points = constraint.points
        if len(points) < 2:
            return result

        ridge_mask = np.zeros((self.h, self.w), dtype=np.float32)
        for i in range(len(points) - 1):
            y0, x0 = points[i]
            y1, x1 = points[i + 1]
            n_steps = max(abs(y1 - y0), abs(x1 - x0), 1)
            for step in range(n_steps + 1):
                t = step / n_steps
                cy = int(y0 + (y1 - y0) * t)
                cx = int(x0 + (x1 - x0) * t)
                spread = int(constraint.width * (1.0 + constraint.sharpness))
                for dy in range(-spread, spread + 1):
                    for dx in range(-spread, spread + 1):
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < self.h and 0 <= nx < self.w:
                            dist = np.sqrt(dy * dy + dx * dx)
                            falloff = np.exp(-(dist / max(constraint.width, 1)) ** 2 * constraint.sharpness)
                            ridge_mask[ny, nx] = max(ridge_mask[ny, nx], falloff)

        result += ridge_mask * constraint.height
        return np.clip(result, -1.0, 1.0)

    def _apply_lake(self, elevation: np.ndarray, constraint: EditConstraint) -> np.ndarray:
        result = elevation.copy()
        if constraint.mask is not None and constraint.mask.shape == result.shape:
            lake_area = constraint.mask > 0.5
            result[lake_area] = np.minimum(result[lake_area], -0.05)
        return result

    def _apply_roughness(self, elevation: np.ndarray, constraint: EditConstraint) -> np.ndarray:
        result = elevation.copy()
        if constraint.mask is None or constraint.mask.shape != result.shape:
            return result

        target = constraint.target_roughness
        current_std = float(np.std(result[constraint.mask > 0.5])) if np.any(constraint.mask > 0.5) else 0.1
        if current_std < 1e-6:
            return result

        scale = target / current_std
        region = constraint.mask > 0.5
        mean_val = float(np.mean(result[region]))
        result[region] = mean_val + (result[region] - mean_val) * scale

        dist_in = distance_transform_edt(region)
        dist_out = distance_transform_edt(~region)
        blend_width = 5
        blend = np.clip(dist_in / blend_width, 0, 1) * np.clip(dist_out / blend_width, 0, 1)
        blend = np.where(region, 1.0 - blend, blend)
        result = result * blend + elevation * (1.0 - blend)
        return result

    def _local_postprocess(self, elevation: np.ndarray) -> np.ndarray:
        diff = elevation - self.base
        changed = np.abs(diff) > 0.001
        if not np.any(changed):
            return elevation

        dilated = binary_dilation(changed, structure=generate_binary_structure(2, 1), iterations=3)
        boundary = dilated & ~changed
        if np.any(boundary):
            smoothed = gaussian_filter(elevation, sigma=1.0)
            blend = np.zeros_like(elevation)
            dist = distance_transform_edt(boundary)
            blend_width = 4.0
            blend[boundary] = np.clip(1.0 - dist[boundary] / blend_width, 0, 1) if np.any(boundary) else 0
            elevation[boundary] = elevation[boundary] * (1 - blend[boundary]) + smoothed[boundary] * blend[boundary]

        return elevation


def _geojson_to_mask(geojson: dict, shape: tuple[int, int] | None, brush_radius: int = 5) -> np.ndarray:
    if shape is not None:
        h, w = shape
    else:
        h, w = 512, 1024
    mask = np.zeros((h, w), dtype=np.float32)

    def to_pixel(pt) -> tuple[int, int]:
        x, y = float(pt[0]), float(pt[1])
        if -180.0 <= x <= 180.0 and -90.0 <= y <= 90.0:
            px = int(np.clip((x + 180.0) / 360.0 * (w - 1), 0, w - 1))
            py = int(np.clip((90.0 - y) / 180.0 * (h - 1), 0, h - 1))
            return px, py
        if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
            return int(np.clip(x * (w - 1), 0, w - 1)), int(np.clip(y * (h - 1), 0, h - 1))
        return int(np.clip(x, 0, w - 1)), int(np.clip(y, 0, h - 1))

    features = geojson.get("features", [geojson] if "geometry" in geojson else [])
    for feat in features:
        geom = feat.get("geometry", feat)
        geom_type = geom.get("type", "")
        coords = geom.get("coordinates", [])
        if geom_type == "Polygon" and coords:
            for ring in coords:
                for pt in ring:
                    px, py = to_pixel(pt)
                    _fill_circle(mask, py, px, brush_radius, h, w)
        elif geom_type == "LineString" and coords:
            for i in range(len(coords) - 1):
                x0, y0 = to_pixel(coords[i])
                x1, y1 = to_pixel(coords[i + 1])
                n_steps = max(abs(x1 - x0), abs(y1 - y0), 1)
                for step in range(n_steps + 1):
                    t = step / n_steps
                    cx = int(x0 + (x1 - x0) * t)
                    cy = int(y0 + (y1 - y0) * t)
                    _fill_circle(mask, cy, cx, brush_radius, h, w)
    return gaussian_filter(mask, sigma=max(1.0, brush_radius * 0.3))


def _fill_circle(mask: np.ndarray, cy: int, cx: int, radius: int, h: int, w: int) -> None:
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < h and 0 <= nx < w:
                dist = (dx * dx + dy * dy) ** 0.5
                if dist <= radius:
                    val = 1.0 - dist / (radius + 1)
                    mask[ny, nx] = max(mask[ny, nx], val)


def _parse_edit_text(text: str, shape: tuple[int, int], current_elevation: np.ndarray) -> list[EditConstraint]:
    constraints = []
    text_lower = text.lower()

    if any(kw in text_lower for kw in ["降低", "lower", "减少", "decrease"]):
        offset = -0.1
        if "山" in text_lower or "mountain" in text_lower or "ridge" in text_lower:
            offset = -0.15
        constraints.append(EditConstraint(
            constraint_type=EditConstraintType.ELEVATION_OFFSET,
            value=offset,
        ))

    if any(kw in text_lower for kw in ["升高", "raise", "增加", "increase", "抬升"]):
        offset = 0.1
        if "山" in text_lower or "mountain" in text_lower or "ridge" in text_lower:
            offset = 0.15
        constraints.append(EditConstraint(
            constraint_type=EditConstraintType.ELEVATION_OFFSET,
            value=offset,
        ))

    if any(kw in text_lower for kw in ["河", "river", "水道", "stream"]):
        h, w = shape
        mid_y = h // 2
        constraints.append(EditConstraint(
            constraint_type=EditConstraintType.RIVER_VECTOR,
            points=[(mid_y, w // 4), (mid_y + h // 8, w // 2), (mid_y + h // 6, 3 * w // 4)],
            width=3,
            depth=0.12,
        ))

    if any(kw in text_lower for kw in ["湖", "lake", "内海", "inland"]):
        h, w = shape
        cy, cx = h // 2, w // 2
        r = min(h, w) // 8
        yy, xx = np.ogrid[:h, :w]
        lake_mask = ((yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2).astype(np.float32)
        constraints.append(EditConstraint(
            constraint_type=EditConstraintType.LAKE_MASK,
            mask=lake_mask,
        ))

    if not constraints:
        constraints.append(EditConstraint(
            constraint_type=EditConstraintType.ELEVATION_OFFSET,
            value=0.0,
        ))

    return constraints
