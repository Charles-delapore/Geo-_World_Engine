from __future__ import annotations

import logging

import numpy as np
from scipy.ndimage import label, binary_dilation, generate_binary_structure, binary_erosion, distance_transform_edt, gaussian_filter

from app.core.geometry_metrics import (
    count_components,
    cross_cut_score,
    principal_axis_angle,
    bbox_aspect_ratio,
    compute_metric_report,
    component_labels,
)

logger = logging.getLogger(__name__)

HARD_CONSTRAINT_KINDS = {
    "single_island": ["land_component_count", "forbid_cross_cut", "forbid_internal_ocean"],
    "two_continents_with_rift_sea": ["land_component_count", "forbid_cross_cut", "must_disconnect_west_east"],
    "central_enclosed_inland_sea": ["water_enclosure", "forbid_open_ocean_center"],
    "archipelago_chain": ["min_land_component_count"],
    "peninsula_coast": ["peninsula_attached"],
}

SOFT_CONSTRAINT_KINDS = {
    "single_island": ["coast_roughness", "principal_axis_angle", "elongation_ratio"],
    "two_continents_with_rift_sea": ["coast_roughness", "rift_width_ratio"],
    "central_enclosed_inland_sea": ["basin_naturalness", "coast_roughness"],
    "archipelago_chain": ["island_density", "coast_roughness"],
    "peninsula_coast": ["coast_roughness", "peninsula_shape"],
}


def classify_constraints(topology_intent: dict) -> dict[str, list[str]]:
    kind = str((topology_intent or {}).get("kind", "")).strip().lower()
    return {
        "hard": HARD_CONSTRAINT_KINDS.get(kind, []),
        "soft": SOFT_CONSTRAINT_KINDS.get(kind, []),
    }


class TopologyGuardResult:
    def __init__(self):
        self.passed: bool = True
        self.issues: list[str] = []
        self.repairs: list[str] = []
        self.metric_report: dict = {}
        self.hard_violations: list[str] = []
        self.soft_violations: list[str] = []

    def add_issue(self, issue: str, severity: str = "hard") -> None:
        self.issues.append(issue)
        self.passed = False
        if severity == "hard":
            self.hard_violations.append(issue)
        else:
            self.soft_violations.append(issue)

    def add_repair(self, repair: str) -> None:
        self.repairs.append(repair)

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "issues": self.issues,
            "repairs": self.repairs,
            "metrics": self.metric_report,
            "hard_violations": self.hard_violations,
            "soft_violations": self.soft_violations,
        }


class TopologyGuard:
    def __init__(self, max_repair_strength: float = 0.3):
        self.max_repair_strength = max_repair_strength

    def validate(self, elevation: np.ndarray, topology_intent: dict | None = None) -> TopologyGuardResult:
        result = TopologyGuardResult()
        result.metric_report = compute_metric_report(elevation, topology_intent)
        if topology_intent is None:
            return result

        kind = str(topology_intent.get("kind", "")).strip().lower()
        target_land = topology_intent.get("target_land_component_count")
        target_water = topology_intent.get("target_water_component_count")
        forbid_cross = topology_intent.get("forbid_cross_cut", False)
        main_axis = str(topology_intent.get("main_axis", "none")).strip().lower()
        elongation = topology_intent.get("elongation_target")

        land_mask = elevation > 0.0
        water_mask = elevation <= 0.0
        land_count = count_components(land_mask, min_cells=20)
        water_count = count_components(water_mask, min_cells=20)

        if target_land is not None and land_count != target_land:
            result.add_issue(f"land_component_count={land_count}, expected={target_land}", severity="hard")

        if target_water is not None and water_count != target_water:
            result.add_issue(f"water_component_count={water_count}, expected={target_water}", severity="hard")

        if forbid_cross:
            ccs = cross_cut_score(water_mask)
            if ccs > 0.15:
                result.add_issue(f"cross_cut_score={ccs:.3f} exceeds threshold 0.15", severity="hard")

        if main_axis in ("north_south", "east_west") and np.any(land_mask):
            angle = principal_axis_angle(land_mask)
            if main_axis == "north_south" and abs(abs(angle) - 90.0) > 30.0:
                result.add_issue(f"principal_axis_angle={angle:.1f}, expected near 90 for north_south", severity="soft")
            if main_axis == "east_west" and abs(angle) > 30.0 and abs(abs(angle) - 180.0) > 30.0:
                result.add_issue(f"principal_axis_angle={angle:.1f}, expected near 0/180 for east_west", severity="soft")

        if elongation is not None and np.any(land_mask):
            aspect = bbox_aspect_ratio(land_mask)
            if main_axis == "north_south" and aspect < elongation:
                result.add_issue(f"aspect_ratio={aspect:.2f}, expected >={elongation} for north_south elongation", severity="soft")
            if main_axis == "east_west" and (1.0 / max(aspect, 0.01)) < elongation:
                result.add_issue(f"aspect_ratio={aspect:.2f}, expected width/height >={elongation} for east_west elongation", severity="soft")

        if kind == "single_island" and land_count > 1:
            result.add_issue(f"single_island has {land_count} land components, expected 1", severity="hard")

        if kind == "two_continents_with_rift_sea" and land_count != 2:
            result.add_issue(f"two_continents has {land_count} land components, expected 2", severity="hard")

        if kind == "single_island" and main_axis in ("north_south", "east_west") and np.any(land_mask):
            angle = principal_axis_angle(land_mask)
            aspect = bbox_aspect_ratio(land_mask)
            if main_axis == "north_south":
                angle_dev = abs(abs(angle) - 90.0)
                if angle_dev > 25.0 or aspect < 1.3:
                    result.add_issue(f"north_south island axis deviation: angle={angle:.1f} aspect={aspect:.2f}", severity="soft")
            elif main_axis == "east_west":
                angle_dev = min(abs(angle), abs(abs(angle) - 180.0))
                if angle_dev > 25.0 or aspect > 0.77:
                    result.add_issue(f"east_west island axis deviation: angle={angle:.1f} aspect={aspect:.2f}", severity="soft")

        return result

    def repair(self, elevation: np.ndarray, topology_intent: dict | None = None) -> tuple[np.ndarray, TopologyGuardResult]:
        result = self.validate(elevation, topology_intent)
        if result.passed:
            return elevation, result

        shaped = elevation.astype(np.float32).copy()
        kind = str((topology_intent or {}).get("kind", "")).strip().lower()
        land_mask = shaped > 0.0
        water_mask = shaped <= 0.0

        if kind == "single_island":
            shaped, repair_desc = self._repair_single_island(shaped, topology_intent or {})
            if repair_desc:
                result.add_repair(repair_desc)

        if kind == "two_continents_with_rift_sea":
            shaped, repair_desc = self._repair_two_continents(shaped, topology_intent or {})
            if repair_desc:
                result.add_repair(repair_desc)

        if kind == "central_enclosed_inland_sea":
            shaped, repair_desc = self._repair_inland_sea(shaped, topology_intent or {})
            if repair_desc:
                result.add_repair(repair_desc)

        if (topology_intent or {}).get("forbid_cross_cut"):
            ccs = cross_cut_score(shaped <= 0.0)
            if ccs > 0.15:
                shaped, repair_desc = self._repair_cross_cut(shaped)
                if repair_desc:
                    result.add_repair(repair_desc)

        main_axis = str((topology_intent or {}).get("main_axis", "none")).strip().lower()
        if kind == "single_island" and main_axis in ("north_south", "east_west"):
            shaped, repair_desc = self._repair_axis_deviation(shaped, main_axis, topology_intent or {})
            if repair_desc:
                result.add_repair(repair_desc)

        shaped = self._restore_gradient_features(elevation, shaped)

        recheck = self.validate(shaped, topology_intent)
        result.metric_report = recheck.metric_report
        if recheck.passed:
            result.passed = True
            result.issues = []

        return np.clip(shaped, -1.0, 1.0).astype(np.float32), result

    def _repair_single_island(self, elevation: np.ndarray, topology_intent: dict) -> tuple[np.ndarray, str]:
        land_mask = elevation > 0.0
        labels_arr, num = label(land_mask.astype(np.int8))
        if num <= 1:
            return elevation, ""

        sizes = {cid: int(np.sum(labels_arr == cid)) for cid in range(1, num + 1)}
        dominant = max(sizes, key=sizes.get)
        stray = land_mask & (labels_arr != dominant)
        shaped = elevation.copy()
        shaped[stray] = np.minimum(shaped[stray], -0.08)
        return shaped, f"removed {int(np.sum(stray))} stray land cells from {num - 1} minor components"

    def _repair_two_continents(self, elevation: np.ndarray, topology_intent: dict) -> tuple[np.ndarray, str]:
        land_mask = elevation > 0.0
        labels_arr, num = label(land_mask.astype(np.int8))
        if num == 2:
            return elevation, ""
        if num < 2:
            return elevation, ""
        sizes = {cid: int(np.sum(labels_arr == cid)) for cid in range(1, num + 1)}
        sorted_ids = sorted(sizes, key=sizes.get, reverse=True)
        keep = set(sorted_ids[:2])
        stray = land_mask & ~np.isin(labels_arr, list(keep))
        shaped = elevation.copy()
        shaped[stray] = np.minimum(shaped[stray], -0.08)
        return shaped, f"removed {int(np.sum(stray))} stray land cells, kept top 2 components"

    def _repair_inland_sea(self, elevation: np.ndarray, topology_intent: dict) -> tuple[np.ndarray, str]:
        water_mask = elevation <= 0.0
        labels_arr, num = label(water_mask.astype(np.int8))
        if num <= 1:
            return elevation, ""
        h, w = elevation.shape
        central = np.zeros_like(elevation, dtype=bool)
        central[int(h * 0.25):int(h * 0.75), int(w * 0.25):int(w * 0.75)] = True
        keep_ids = set()
        for cid in range(1, num + 1):
            component = labels_arr == cid
            if np.any(component & central):
                keep_ids.add(cid)
        stray = water_mask & central & ~np.isin(labels_arr, list(keep_ids))
        shaped = elevation.copy()
        shaped[stray] = np.maximum(shaped[stray], 0.04)
        return shaped, f"filled {int(np.sum(stray))} stray water cells in central region"

    def _repair_cross_cut(self, elevation: np.ndarray) -> tuple[np.ndarray, str]:
        h, w = elevation.shape
        water_mask = elevation <= 0.0
        center_h_start = int(h * 0.35)
        center_h_end = int(h * 0.65)
        horizontal_band = water_mask[center_h_start:center_h_end, :]
        col_water_ratio = np.mean(horizontal_band, axis=0)
        cross_cols = np.where(col_water_ratio > 0.6)[0]
        if len(cross_cols) == 0:
            return elevation, ""

        shaped = elevation.copy()
        for col in cross_cols:
            land_rows = np.where(shaped[:, col] > 0.0)[0]
            if len(land_rows) > 0:
                center_rows = land_rows[(land_rows >= center_h_start) & (land_rows < center_h_end)]
                if len(center_rows) > 0:
                    boost = 0.06 * self.max_repair_strength
                    shaped[center_rows, col] = np.maximum(shaped[center_rows, col], boost)
        return shaped, f"boosted land in {len(cross_cols)} cross-cut columns"

    def _restore_gradient_features(self, original: np.ndarray, modified: np.ndarray) -> np.ndarray:
        try:
            from app.core.terrain_analysis import compute_gradient_loss, restore_gradient
            from app.core.hydrology_advanced import detect_ridges, detect_rivers, protect_features_during_repair

            land_mask = original > 0.0
            grad_loss = compute_gradient_loss(original, modified, land_mask=land_mask)
            if grad_loss < 0.01:
                return modified

            ridge_mask = detect_ridges(original, accumulation_threshold=0.12, min_elevation_percentile=60.0)
            river_mask = detect_rivers(original, accumulation_threshold=0.65)
            result = protect_features_during_repair(
                original, modified, ridge_mask, river_mask,
                ridge_strength=0.4, river_strength=0.3,
            )
            result = restore_gradient(original, result, strength=0.2, land_mask=land_mask)
            return result
        except Exception as exc:
            logger.warning("gradient restoration failed: %s", exc)
            return modified

    def _repair_axis_deviation(self, elevation: np.ndarray, main_axis: str, topology_intent: dict) -> tuple[np.ndarray, str]:
        land_mask = elevation > 0.0
        if not np.any(land_mask):
            return elevation, ""

        angle = principal_axis_angle(land_mask)
        aspect = bbox_aspect_ratio(land_mask)
        shaped = elevation.copy()
        h, w = shaped.shape
        repairs_made = []

        if main_axis == "north_south":
            angle_dev = abs(abs(angle) - 90.0)
            needs_fix = angle_dev > 20.0 or aspect < 1.2
            if not needs_fix:
                return elevation, ""

            lateral_band_start = int(w * 0.3)
            lateral_band_end = int(w * 0.7)
            lateral_land = land_mask[:, lateral_band_start:lateral_band_end]
            lateral_water = ~lateral_land

            if np.any(lateral_water):
                erosion_strength = min(0.12, self.max_repair_strength * 0.4)
                dist_from_center_w = np.abs(np.arange(w) - w * 0.5).reshape(1, -1) / (w * 0.5)
                lateral_weight = np.broadcast_to(np.clip(dist_from_center_w * 1.5, 0.0, 1.0), (h, w)).copy()
                lateral_mask = (land_mask & (lateral_weight > 0.4))
                shaped[lateral_mask] -= erosion_strength * lateral_weight[lateral_mask].astype(np.float32)
                repairs_made.append("compressed lateral land")

            vertical_band_start = int(h * 0.15)
            vertical_band_end = int(h * 0.85)
            vertical_land = land_mask[vertical_band_start:vertical_band_end, :]
            vertical_water = ~vertical_land
            if np.any(vertical_water) and aspect < 1.4:
                boost_strength = min(0.08, self.max_repair_strength * 0.3)
                dist_from_center_h = np.abs(np.arange(h) - h * 0.5).reshape(-1, 1) / (h * 0.5)
                vertical_weight = np.broadcast_to(np.clip(1.0 - dist_from_center_h, 0.0, 1.0), (h, w)).copy()
                vertical_mask = (~land_mask & (vertical_weight > 0.5))
                shaped[vertical_mask] = np.maximum(
                    shaped[vertical_mask],
                    boost_strength * vertical_weight[vertical_mask].astype(np.float32),
                )
                repairs_made.append("boosted vertical land")

            transverse_water = np.zeros_like(land_mask)
            for row_idx in range(h):
                row_land = np.where(land_mask[row_idx])[0]
                if len(row_land) > 0:
                    gaps = np.diff(row_land)
                    big_gaps = np.where(gaps > w * 0.08)[0]
                    if len(big_gaps) > 0:
                        for gap_idx in big_gaps:
                            gap_start = row_land[gap_idx]
                            gap_end = row_land[gap_idx + 1]
                            transverse_water[row_idx, gap_start:gap_end] = True

            if np.any(transverse_water):
                fill_strength = min(0.06, self.max_repair_strength * 0.2)
                shaped[transverse_water] = np.maximum(shaped[transverse_water], fill_strength)
                repairs_made.append("filled transverse gaps")

        elif main_axis == "east_west":
            angle_dev = min(abs(angle), abs(abs(angle) - 180.0))
            needs_fix = angle_dev > 20.0 or aspect > 0.83
            if not needs_fix:
                return elevation, ""

            vertical_band_start = int(h * 0.3)
            vertical_band_end = int(h * 0.7)
            vertical_land = land_mask[vertical_band_start:vertical_band_end, :]
            vertical_water = ~vertical_land

            if np.any(vertical_water):
                erosion_strength = min(0.12, self.max_repair_strength * 0.4)
                dist_from_center_h = np.abs(np.arange(h) - h * 0.5).reshape(-1, 1) / (h * 0.5)
                vertical_weight = np.broadcast_to(np.clip(dist_from_center_h * 1.5, 0.0, 1.0), (h, w)).copy()
                vertical_mask = (land_mask & (vertical_weight > 0.4))
                shaped[vertical_mask] -= erosion_strength * vertical_weight[vertical_mask].astype(np.float32)
                repairs_made.append("compressed vertical land")

            lateral_band_start = int(w * 0.15)
            lateral_band_end = int(w * 0.85)
            lateral_land = land_mask[:, lateral_band_start:lateral_band_end]
            lateral_water = ~lateral_land
            if np.any(lateral_water) and aspect > 0.6:
                boost_strength = min(0.08, self.max_repair_strength * 0.3)
                dist_from_center_w = np.abs(np.arange(w) - w * 0.5).reshape(1, -1) / (w * 0.5)
                lateral_weight = np.broadcast_to(np.clip(1.0 - dist_from_center_w, 0.0, 1.0), (h, w)).copy()
                lateral_mask = (~land_mask & (lateral_weight > 0.5))
                shaped[lateral_mask] = np.maximum(
                    shaped[lateral_mask],
                    boost_strength * lateral_weight[lateral_mask].astype(np.float32),
                )
                repairs_made.append("boosted lateral land")

        if repairs_made:
            labels_arr = component_labels(shaped > 0.0, min_cells=20)
            dominant_id = 0
            max_size = 0
            for cid in range(1, labels_arr.max() + 1):
                size = int(np.sum(labels_arr == cid))
                if size > max_size:
                    max_size = size
                    dominant_id = cid
            stray = (shaped > 0.0) & (labels_arr != dominant_id) & (labels_arr > 0)
            shaped[stray] = np.minimum(shaped[stray], -0.06)
            shaped = gaussian_filter(shaped, sigma=0.6).astype(np.float32)

        return shaped, "; ".join(repairs_made) if repairs_made else ""
