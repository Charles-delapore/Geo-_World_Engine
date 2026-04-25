from __future__ import annotations

import numpy as np
import pytest

from app.core.geometry_metrics import (
    count_components,
    cross_cut_score,
    principal_axis_angle,
    bbox_aspect_ratio,
    compute_metric_report,
    enclosure_score,
    coast_roughness,
)
from app.core.topology_guard import TopologyGuard
from app.core.topology_validator import validate_world_plan, auto_fix_world_plan
from app.core.llm_parser import parse_with_rag
from app.workers.render_worker import render_world


GOLDEN_PROMPTS = [
    {
        "prompt": "一座四面环海的岛屿。",
        "topology_kind": "single_island",
        "expected_land_components": 1,
        "min_land_ratio": 0.15,
        "max_land_ratio": 0.65,
    },
    {
        "prompt": "东西大陆中间被海隔开。",
        "topology_kind": "two_continents_with_rift_sea",
        "expected_land_components": 2,
        "min_land_ratio": 0.2,
        "max_land_ratio": 0.7,
    },
    {
        "prompt": "中央是一片群岛。",
        "topology_kind": "archipelago_chain",
        "min_land_components": 3,
        "min_land_ratio": 0.05,
        "max_land_ratio": 0.6,
    },
    {
        "prompt": "东侧伸出一条半岛。",
        "topology_kind": "peninsula_coast",
        "min_land_ratio": 0.15,
        "max_land_ratio": 0.7,
    },
    {
        "prompt": "中间是一片宽阔的内海。",
        "topology_kind": "central_enclosed_inland_sea",
        "min_land_ratio": 0.2,
        "max_land_ratio": 0.75,
    },
    {
        "prompt": "一座南北向狭长的四面环海岛屿。",
        "topology_kind": "single_island",
        "expected_land_components": 1,
        "min_land_ratio": 0.1,
        "max_land_ratio": 0.55,
    },
    {
        "prompt": "东西大陆中间被狭窄的海隔开。",
        "topology_kind": "two_continents_with_rift_sea",
        "expected_land_components": 2,
        "min_land_ratio": 0.25,
        "max_land_ratio": 0.7,
    },
    {
        "prompt": "中央是一片密集的群岛。",
        "topology_kind": "archipelago_chain",
        "min_land_components": 3,
        "min_land_ratio": 0.05,
        "max_land_ratio": 0.55,
    },
]


@pytest.mark.parametrize("golden", GOLDEN_PROMPTS, ids=lambda g: g["prompt"][:20])
def test_golden_prompt_topology(golden: dict) -> None:
    plan = parse_with_rag(golden["prompt"])
    topology_intent = plan.get("topology_intent") or {}
    assert topology_intent.get("kind") == golden["topology_kind"], (
        f"Expected topology kind {golden['topology_kind']}, got {topology_intent.get('kind')}"
    )

    arrays, _ = render_world(plan, width=160, height=96, seed=42)
    elevation = arrays["elevation"]
    land_mask = elevation > 0.0
    land_ratio = float(np.mean(land_mask))

    assert land_ratio >= golden["min_land_ratio"], f"Land ratio {land_ratio} below minimum {golden['min_land_ratio']}"
    assert land_ratio <= golden["max_land_ratio"], f"Land ratio {land_ratio} above maximum {golden['max_land_ratio']}"

    if "expected_land_components" in golden:
        land_count = count_components(land_mask, min_cells=20)
        assert land_count == golden["expected_land_components"], (
            f"Expected {golden['expected_land_components']} land components, got {land_count}"
        )

    if "min_land_components" in golden:
        land_count = count_components(land_mask, min_cells=20)
        assert land_count >= golden["min_land_components"], (
            f"Expected at least {golden['min_land_components']} land components, got {land_count}"
        )

    if topology_intent.get("forbid_cross_cut") and topology_intent.get("kind") == "two_continents_with_rift_sea":
        water_mask = elevation <= 0.0
        ccs = cross_cut_score(water_mask)
        assert ccs <= 0.25, f"Cross-cut score {ccs} exceeds threshold 0.25 for forbid_cross_cut topology"


def test_topology_guard_repairs_single_island() -> None:
    rng = np.random.RandomState(99)
    elevation = rng.randn(64, 128).astype(np.float32) * 0.3
    elevation[20:44, 30:60] = 0.5
    elevation[10:20, 80:100] = 0.4

    topology_intent = {
        "kind": "single_island",
        "target_land_component_count": 1,
        "forbid_cross_cut": True,
    }
    guard = TopologyGuard()
    result = guard.validate(elevation, topology_intent)
    assert not result.passed, "Should detect multiple land components"

    repaired, repair_result = guard.repair(elevation, topology_intent)
    land_count = count_components(repaired > 0.0, min_cells=20)
    assert land_count == 1, f"After repair, expected 1 land component, got {land_count}"


def test_topology_guard_repairs_two_continents() -> None:
    rng = np.random.RandomState(77)
    elevation = rng.randn(64, 128).astype(np.float32) * 0.2
    elevation[15:50, 10:50] = 0.5
    elevation[15:50, 78:118] = 0.5
    elevation[10:15, 55:75] = 0.3

    topology_intent = {
        "kind": "two_continents_with_rift_sea",
        "target_land_component_count": 2,
        "forbid_cross_cut": True,
    }
    guard = TopologyGuard()
    repaired, repair_result = guard.repair(elevation, topology_intent)
    land_count = count_components(repaired > 0.0, min_cells=20)
    assert land_count == 2, f"After repair, expected 2 land components, got {land_count}"


def test_validate_world_plan_detects_conflicts() -> None:
    conflicting_plan = {
        "topology_intent": {"kind": "single_island"},
        "continents": [{"position": "west", "size": 0.4}, {"position": "east", "size": 0.4}],
        "island_chains": [{"position": "center", "density": 0.7}],
        "peninsulas": [{"location": "east", "size": 0.2}],
    }
    issues = validate_world_plan(conflicting_plan)
    assert len(issues) > 0, "Should detect conflicts in single_island with multiple continents and island_chains"


def test_auto_fix_world_plan_resolves_conflicts() -> None:
    conflicting_plan = {
        "topology_intent": {"kind": "single_island"},
        "continents": [{"position": "west", "size": 0.4}, {"position": "east", "size": 0.4}],
        "island_chains": [{"position": "center", "density": 0.7}],
    }
    fixed = auto_fix_world_plan(conflicting_plan)
    assert len(fixed["continents"]) == 1, "Should reduce to 1 continent for single_island"
    assert fixed["island_chains"] == [], "Should remove island_chains for single_island"
    assert fixed["topology_intent"]["forbid_cross_cut"] is True


def test_compute_metric_report() -> None:
    rng = np.random.RandomState(42)
    elevation = rng.randn(64, 128).astype(np.float32) * 0.3
    elevation[20:44, 30:98] = 0.5

    report = compute_metric_report(elevation)
    assert "land_components" in report
    assert "water_components" in report
    assert "cross_cut_score" in report
    assert "enclosure_score" in report
    assert "coast_roughness" in report
    assert report["land_components"] >= 1


def test_geometry_metrics_functions() -> None:
    mask = np.zeros((50, 50), dtype=bool)
    mask[10:40, 10:40] = True

    assert count_components(mask) == 1
    assert bbox_aspect_ratio(mask) == pytest.approx(1.0, abs=0.1)
    assert enclosure_score(~mask, mask) >= 0.0
    assert coast_roughness(mask) >= 1.0

    angle = principal_axis_angle(mask)
    assert -180.0 <= angle <= 180.0
