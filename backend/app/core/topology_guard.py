from __future__ import annotations

import logging

import numpy as np
from scipy.ndimage import label, binary_dilation, generate_binary_structure

from app.core.geometry_metrics import (
    count_components,
    cross_cut_score,
    principal_axis_angle,
    bbox_aspect_ratio,
    compute_metric_report,
)

logger = logging.getLogger(__name__)


class TopologyGuardResult:
    def __init__(self):
        self.passed: bool = True
        self.issues: list[str] = []
        self.repairs: list[str] = []
        self.metric_report: dict = {}

    def add_issue(self, issue: str) -> None:
        self.issues.append(issue)
        self.passed = False

    def add_repair(self, repair: str) -> None:
        self.repairs.append(repair)

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "issues": self.issues,
            "repairs": self.repairs,
            "metrics": self.metric_report,
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
            result.add_issue(f"land_component_count={land_count}, expected={target_land}")

        if target_water is not None and water_count != target_water:
            result.add_issue(f"water_component_count={water_count}, expected={target_water}")

        if forbid_cross:
            ccs = cross_cut_score(water_mask)
            if ccs > 0.15:
                result.add_issue(f"cross_cut_score={ccs:.3f} exceeds threshold 0.15")

        if main_axis in ("north_south", "east_west") and np.any(land_mask):
            angle = principal_axis_angle(land_mask)
            if main_axis == "north_south" and abs(abs(angle) - 90.0) > 30.0:
                result.add_issue(f"principal_axis_angle={angle:.1f}, expected near 90 for north_south")
            if main_axis == "east_west" and abs(angle) > 30.0 and abs(abs(angle) - 180.0) > 30.0:
                result.add_issue(f"principal_axis_angle={angle:.1f}, expected near 0/180 for east_west")

        if elongation is not None and np.any(land_mask):
            aspect = bbox_aspect_ratio(land_mask)
            if main_axis == "north_south" and aspect < elongation:
                result.add_issue(f"aspect_ratio={aspect:.2f}, expected >={elongation} for north_south elongation")
            if main_axis == "east_west" and (1.0 / max(aspect, 0.01)) < elongation:
                result.add_issue(f"aspect_ratio={aspect:.2f}, expected width/height >={elongation} for east_west elongation")

        if kind == "single_island" and land_count > 1:
            result.add_issue(f"single_island has {land_count} land components, expected 1")

        if kind == "two_continents_with_rift_sea" and land_count != 2:
            result.add_issue(f"two_continents has {land_count} land components, expected 2")

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
