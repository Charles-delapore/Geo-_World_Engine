from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from app.core.semantic_mapper import compute_consistency_score, resolve_position_continuous
from app.core.spatial_critic import evaluate_terrain_consistency
from app.core.spatial_relation_graph import (
    SpatialRelationGraph,
    SRGEntity,
    SRGEdge,
    EntityType,
    TopologicalPredicate,
    FuzzyQuantifier,
)

logger = logging.getLogger(__name__)

CONSISTENCY_THRESHOLD = 0.55
REGEN_MAX_ITERATIONS = 3


@dataclass
class NlCritique:
    passed: bool
    overall_score: float = 0.0
    contribution: float = 0.0
    feasibility: float = 0.0
    consistency: float = 0.0
    land_ratio_match: float = 0.0
    ruggedness_match: float = 0.0
    flatness_match: float = 0.0
    island_count_match: float = 0.0
    srg_overall: float = 0.0
    landmass_count_match: float = 0.0
    grade: str = "Neutral"
    issues: list[str] = field(default_factory=list)
    param_adjustments: dict[str, float] = field(default_factory=dict)
    plan_patch: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "overall_score": round(self.overall_score, 3),
            "contribution": round(self.contribution, 3),
            "feasibility": round(self.feasibility, 3),
            "consistency": round(self.consistency, 3),
            "land_ratio_match": round(self.land_ratio_match, 3),
            "ruggedness_match": round(self.ruggedness_match, 3),
            "flatness_match": round(self.flatness_match, 3),
            "island_count_match": round(self.island_count_match, 3),
            "srg_overall": round(self.srg_overall, 3),
            "landmass_count_match": round(self.landmass_count_match, 3),
            "grade": self.grade,
            "issues": self.issues,
            "param_adjustments": self.param_adjustments,
        }


def critique_terrain(
    prompt: str,
    plan: dict,
    elevation: np.ndarray,
    srg: Optional[SpatialRelationGraph] = None,
) -> NlCritique:
    issues: list[str] = []
    adjustments: dict[str, float] = {}
    plan_patch: dict = {}

    ti = plan.get("topology_intent") or {}
    profile = plan.get("profile") or {}

    metrics = _extract_terrain_metrics(elevation)
    plan_land_ratio = float(profile.get("land_ratio", 0.44))
    plan_ruggedness = float(profile.get("ruggedness", 0.55))

    land_ratio_match = _match_ratio(metrics["land_ratio"], plan_land_ratio, tolerance=0.10)

    ruggedness_match = 1.0
    if _contains_keyword(prompt, ["崎岖", "rugged", "steep", "mountain", "山", "山脉"]):
        ruggedness_match = _match_min(metrics["mean_slope"], 0.015)
    elif _contains_keyword(prompt, ["平坦", "flat", "plain", "平原"]):
        ruggedness_match = _match_max(metrics["mean_slope"], 0.008)

    flatness_match = 1.0
    if _contains_keyword(prompt, ["平坦", "flat", "plain", "平原"]):
        flatness_match = _match_max(metrics["mean_slope"], 0.008)

    island_count_match = 1.0
    if _contains_keyword(prompt, ["岛", "island", "archipelago", "群岛"]):
        target = max(int(ti.get("target_land_component_count") or 1), 1)
        actual = metrics["n_land_components"]
        if target > 0:
            island_count_match = min(target, actual) / max(target, actual)
        if island_count_match < 0.5:
            issues.append(f"岛屿数不匹配: 期望~{target}, 实际{actual}")

    srg_data = plan.get("srg")
    srg_obj = srg
    if srg_data and srg_obj is None:
        srg_obj = _reconstruct_srg(srg_data)
    srg_consistency = {}
    if srg_obj:
        srg_consistency = evaluate_terrain_consistency(elevation, srg_obj, ti)
    srg_overall = float(srg_consistency.get("overall", 0.5)) if srg_consistency else 0.5
    landmass_count_match_srg = float(srg_consistency.get("landmass_count_match", island_count_match))
    separation_verified = float(srg_consistency.get("separation_verified", 1.0))
    rift_sea_presence = float(srg_consistency.get("rift_sea_presence", 1.0))
    inland_sea_presence = float(srg_consistency.get("inland_sea_presence", 1.0))

    if land_ratio_match < 0.55:
        issues.append(f"陆地占比偏差大: 实际{metrics['land_ratio']:.2f}, 期望~{plan_land_ratio:.2f}")
        delta = plan_land_ratio - metrics["land_ratio"]
        adjustments["land_ratio_shift"] = round(float(delta), 4)

    if ruggedness_match < 0.5:
        issues.append(f"地形粗糙度不匹配: 实际坡度{metrics['mean_slope']:.4f}")
        if metrics["mean_slope"] < 0.005:
            adjustments["ruggedness_boost"] = 0.15
        else:
            adjustments["ruggedness_reduce"] = 0.15

    if flatness_match < 0.5:
        issues.append(f"期望平坦地形但实际坡度偏高: {metrics['mean_slope']:.4f}")
        adjustments["ruggedness_reduce"] = 0.15

    if landmass_count_match_srg < 0.6:
        issues.append(f"大陆/岛屿数量不匹配 (SRG)")

    if separation_verified < 0.5:
        issues.append("声明被海隔开的区域实际连通")
        adjustments["sea_enforce_center"] = 0.3

    if rift_sea_presence < 0.5 and ti.get("kind") == "two_continents_with_rift_sea":
        issues.append("rift_sea模式下中心区域无海域")
        adjustments["sea_enforce_center"] = 0.4

    if inland_sea_presence < 0.4 and ti.get("kind") == "central_enclosed_inland_sea":
        issues.append("内海模式下中心无水")
        adjustments["sea_enforce_center"] = 0.45

    mountain_region_check = _check_mountain_regions(prompt, elevation, plan)
    if mountain_region_check:
        issues.append(mountain_region_check)
        adjustments["mountain_region_boost"] = 0.2

    contribution = _avg(land_ratio_match, ruggedness_match, landmass_count_match_srg)
    feasibility = 0.9 if len(issues) <= 1 else max(0.3, 0.9 - len(issues) * 0.15)
    consistency_score = _avg(srg_overall, landmass_count_match_srg, separation_verified)
    overall = _avg(contribution, feasibility, consistency_score)

    if not issues:
        grade = "Excellent"
    elif len(issues) == 1:
        grade = "Good"
    elif len(issues) <= 2:
        grade = "Neutral"
    elif len(issues) <= 3:
        grade = "Poor"
    else:
        grade = "Very Poor"

    passed = overall >= CONSISTENCY_THRESHOLD

    if adjustments:
        plan_patch = _build_plan_patch(plan, adjustments)

    return NlCritique(
        passed=passed,
        overall_score=overall,
        contribution=contribution,
        feasibility=feasibility,
        consistency=consistency_score,
        land_ratio_match=land_ratio_match,
        ruggedness_match=ruggedness_match,
        flatness_match=flatness_match,
        island_count_match=island_count_match,
        srg_overall=srg_overall,
        landmass_count_match=landmass_count_match_srg,
        grade=grade,
        issues=issues,
        param_adjustments=adjustments,
        plan_patch=plan_patch,
    )


def iterative_regenerate(
    prompt: str,
    plan: dict,
    generate_fn,
    max_iterations: int = REGEN_MAX_ITERATIONS,
) -> tuple[np.ndarray, list[NlCritique], dict]:
    plan = deepcopy(dict(plan))
    srg_data = plan.get("srg")
    srg = _reconstruct_srg(srg_data) if srg_data else None

    history: list[NlCritique] = []
    best_elevation: np.ndarray | None = None
    best_critique: NlCritique | None = None

    for iteration in range(max_iterations):
        elevation = generate_fn(plan)
        critique = critique_terrain(prompt, plan, elevation, srg)
        history.append(critique)

        logger.info(
            "NL verify iteration %d/%d: passed=%s grade=%s overall=%.3f issues=%d",
            iteration + 1, max_iterations, critique.passed, critique.grade,
            critique.overall_score, len(critique.issues),
        )

        if critique.passed:
            plan["nl_critique_history"] = [h.to_dict() for h in history]
            return elevation, history, plan

        if best_critique is None or critique.overall_score > best_critique.overall_score:
            best_critique = critique
            best_elevation = elevation.copy()

        if critique.plan_patch:
            plan = _apply_plan_patch(plan, critique.plan_patch)
        else:
            logger.warning("NL critic found issues but no patch available: %s", critique.issues)
            break

    plan["nl_critique_history"] = [h.to_dict() for h in history]
    if best_elevation is not None:
        logger.info("NL verification returned best result (score=%.3f)", best_critique.overall_score if best_critique else 0)
        return best_elevation, history, plan
    return generate_fn(plan), history, plan


def _extract_terrain_metrics(elevation: np.ndarray) -> dict:
    from scipy.ndimage import label as ndimage_label

    land = elevation > 0.0
    land_ratio = float(np.mean(land))

    grad_y, grad_x = np.gradient(elevation.astype(np.float64))
    slope = np.sqrt(grad_y ** 2 + grad_x ** 2)
    mean_slope = float(np.mean(slope[land])) if np.any(land) else 0.0

    labeled, n_components = ndimage_label(land.astype(np.int8))

    return {
        "land_ratio": land_ratio,
        "mean_slope": mean_slope,
        "n_land_components": n_components,
    }


def _match_ratio(actual: float, target: float, tolerance: float = 0.10) -> float:
    diff = abs(actual - target)
    if diff <= tolerance:
        return 1.0
    return max(0.0, 1.0 - (diff - tolerance) / (0.3))


def _match_min(actual: float, expected_min: float) -> float:
    if actual >= expected_min:
        return 1.0
    return max(0.0, actual / expected_min)


def _match_max(actual: float, expected_max: float) -> float:
    if actual <= expected_max:
        return 1.0
    return max(0.0, expected_max / actual)


def _contains_keyword(text: str, keywords: list[str]) -> bool:
    lower = text.lower()
    return any(kw in lower for kw in keywords)


def _avg(*values: float) -> float:
    return sum(values) / max(len(values), 1)


def _check_mountain_regions(prompt: str, elevation: np.ndarray, plan: dict) -> str | None:
    constraints = plan.get("constraints") or {}
    mountains = constraints.get("mountains") or plan.get("mountains") or []
    if not mountains:
        return None

    h, w = elevation.shape
    for mtn in mountains:
        location = mtn.get("location") or mtn.get("position", "center")
        y, x = resolve_position_continuous(location)
        py, px = int(y * h), int(x * w)
        region_size = int(min(h, w) * 0.12)
        y0 = max(0, py - region_size)
        y1 = min(h, py + region_size)
        x0 = max(0, px - region_size)
        x1 = min(w, px + region_size)
        region = elevation[y0:y1, x0:x1]
        region_land = region[region > 0]
        if len(region_land) < region.size * 0.2:
            return f"山脉区域({location})几乎无水或陆地占比极低"
        expected_height = float(mtn.get("height", 0.7))
        if region_land.size > 0:
            region_max = float(np.max(region_land))
            if region_max < expected_height * 0.3:
                return f"山脉区域({location})高度不足: max={region_max:.2f}, 期望>={expected_height*0.3:.2f}"
    return None


def _build_plan_patch(plan: dict, adjustments: dict[str, float]) -> dict:
    patch: dict = {"profile": {}, "constraints": {}, "topology_intent": {}}
    profile = plan.get("profile") or {}

    if "land_ratio_shift" in adjustments:
        current = float(profile.get("land_ratio", 0.44))
        patch["profile"]["land_ratio"] = round(np.clip(current + adjustments["land_ratio_shift"] * 0.8, 0.2, 0.8), 3)

    if "ruggedness_boost" in adjustments:
        current = float(profile.get("ruggedness", 0.55))
        patch["profile"]["ruggedness"] = round(min(1.0, current + adjustments["ruggedness_boost"]), 3)

    if "ruggedness_reduce" in adjustments:
        current = float(profile.get("ruggedness", 0.55))
        patch["profile"]["ruggedness"] = round(max(0.0, current - adjustments["ruggedness_reduce"]), 3)

    if "sea_enforce_center" in adjustments:
        constraints = plan.get("constraints") or {}
        sea_zones = list(constraints.get("sea_zones") or [])
        if "center" not in sea_zones:
            sea_zones.append("center")
        patch["constraints"]["sea_zones"] = sea_zones
        if "topology_intent" not in patch:
            patch["topology_intent"] = {}
        patch["topology_intent"]["kind"] = "two_continents_with_rift_sea"
        patch["topology_intent"]["forbid_cross_cut"] = True
        if "must_disconnect_pairs" not in patch["topology_intent"]:
            patch["topology_intent"]["must_disconnect_pairs"] = [["west", "east"]]

    if "mountain_region_boost" in adjustments:
        mountains = list(plan.get("mountains") or [])
        if mountains:
            patch["mountains"] = [
                {**m, "height": min(1.0, float(m.get("height", 0.7)) * 1.2)}
                for m in mountains
            ]

    return patch


def _apply_plan_patch(plan: dict, patch: dict) -> dict:
    merged = deepcopy(dict(plan))

    for section_key in ("profile", "constraints", "topology_intent"):
        if section_key in patch:
            existing = dict(merged.get(section_key) or {})
            existing.update(patch[section_key])
            merged[section_key] = existing

    if "mountains" in patch:
        merged["mountains"] = patch["mountains"]

    return merged


def _reconstruct_srg(srg_data: dict) -> SpatialRelationGraph | None:
    if not srg_data:
        return None
    try:
        graph = SpatialRelationGraph(
            raw_text=srg_data.get("raw_text", ""),
            cost_steps=srg_data.get("cost_steps", []),
        )
        for entity_data in srg_data.get("entities", []):
            entity_type_str = entity_data.get("entity_type", "continent")
            try:
                entity_type = EntityType(entity_type_str)
            except ValueError:
                entity_type = EntityType.CONTINENT
            graph.add_entity(SRGEntity(
                name=entity_data.get("name", ""),
                entity_type=entity_type,
                position=entity_data.get("position"),
                attributes=entity_data.get("attributes", {}),
                text_span=entity_data.get("text_span", ""),
            ))
        for edge_data in srg_data.get("edges", []):
            predicate_str = edge_data.get("predicate", "disjoint")
            try:
                predicate = TopologicalPredicate(predicate_str)
            except ValueError:
                predicate = TopologicalPredicate.DISJOINT
            quantifier = None
            if edge_data.get("quantifier"):
                try:
                    quantifier = FuzzyQuantifier(edge_data["quantifier"])
                except ValueError:
                    pass
            graph.add_edge(SRGEdge(
                subject=edge_data.get("subject", ""),
                predicate=predicate,
                object=edge_data.get("object", ""),
                quantifier=quantifier,
                confidence=edge_data.get("confidence", 1.0),
                text_evidence=edge_data.get("text_evidence", ""),
            ))
        return graph
    except Exception:
        return None
