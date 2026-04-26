from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from app.config import settings

logger = logging.getLogger(__name__)

FAILURE_CASES: list[dict] = [
    {
        "id": "fail-single-island-archipelago",
        "prompt": "一座四面环海的岛屿",
        "failure_type": "single_island_becomes_archipelago",
        "wrong_intent": "archipelago_chain",
        "correct_intent": "single_island",
        "wrong_metrics": {"land_components": 3},
        "fix": {
            "topology_intent": {
                "kind": "single_island",
                "target_land_component_count": 1,
                "forbid_cross_cut": True,
                "forbid_internal_ocean": True,
                "forbid_fragmented_islands": True,
            },
        },
    },
    {
        "id": "fail-dual-continent-cross-cut",
        "prompt": "东西大陆中间被海隔开",
        "failure_type": "two_continents_cross_cut",
        "wrong_intent": "two_continents_with_rift_sea",
        "correct_intent": "two_continents_with_rift_sea",
        "wrong_metrics": {"land_components": 4, "cross_cut_score": 0.45},
        "fix": {
            "topology_intent": {
                "kind": "two_continents_with_rift_sea",
                "target_land_component_count": 2,
                "forbid_cross_cut": True,
                "must_disconnect_pairs": [["west", "east"]],
            },
            "modifiers": {
                "rift_width": "balanced",
                "rift_profile": "natural",
            },
        },
    },
    {
        "id": "fail-inland-sea-ellipse",
        "prompt": "中间有内海",
        "failure_type": "inland_sea_regular_ellipse",
        "wrong_intent": "central_enclosed_inland_sea",
        "correct_intent": "central_enclosed_inland_sea",
        "wrong_metrics": {"enclosure_score": 0.9, "coast_roughness": 1.05},
        "fix": {
            "topology_intent": {
                "kind": "central_enclosed_inland_sea",
                "boundary_irregularity": 0.65,
                "symmetry_break": 0.4,
            },
            "modifiers": {
                "basin_shape": "branched",
                "basin_style": "mediterranean",
            },
        },
    },
    {
        "id": "fail-north-south-round",
        "prompt": "一座南北向狭长的四面环海岛屿",
        "failure_type": "north_south_island_becomes_round",
        "wrong_intent": "single_island",
        "correct_intent": "single_island",
        "wrong_metrics": {"principal_axis_angle": 15.0, "bbox_aspect_ratio": 1.1},
        "fix": {
            "topology_intent": {
                "kind": "single_island",
                "main_axis": "north_south",
                "elongation_target": 1.8,
                "target_land_component_count": 1,
            },
            "modifiers": {
                "shape_bias": "elongated",
                "shape_axis": "north_south",
            },
        },
    },
    {
        "id": "fail-dual-continent-four-lands",
        "prompt": "东西两块大陆中间被海隔开",
        "failure_type": "two_continents_four_landmasses",
        "wrong_intent": "two_continents_with_rift_sea",
        "correct_intent": "two_continents_with_rift_sea",
        "wrong_metrics": {"land_components": 4},
        "fix": {
            "topology_intent": {
                "kind": "two_continents_with_rift_sea",
                "target_land_component_count": 2,
                "forbid_cross_cut": True,
                "must_disconnect_pairs": [["west", "east"]],
            },
        },
    },
]


class FailureCaseDB:
    def __init__(self, data_dir: str | Path | None = None):
        self.data_dir = Path(data_dir or getattr(settings, "ARTIFACT_ROOT", "./data"))
        self._cases: list[dict] = []
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._cases = list(FAILURE_CASES)
        custom_path = self.data_dir / "failure_cases.json"
        if custom_path.exists():
            try:
                custom = json.loads(custom_path.read_text(encoding="utf-8"))
                if isinstance(custom, list):
                    self._cases.extend(custom)
            except Exception as exc:
                logger.warning("Failed to load custom failure cases: %s", exc)
        self._loaded = True

    def find_by_prompt(self, prompt: str) -> list[dict]:
        self._ensure_loaded()
        results = []
        prompt_lower = prompt.lower()
        for case in self._cases:
            if case.get("prompt", "").lower() in prompt_lower or prompt_lower in case.get("prompt", "").lower():
                results.append(case)
        return results

    def find_by_failure_type(self, failure_type: str) -> list[dict]:
        self._ensure_loaded()
        return [c for c in self._cases if c.get("failure_type") == failure_type]

    def list_all(self) -> list[dict]:
        self._ensure_loaded()
        return list(self._cases)

    def count(self) -> int:
        self._ensure_loaded()
        return len(self._cases)
