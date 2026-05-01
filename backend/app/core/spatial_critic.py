from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from app.core.spatial_relation_graph import (
    SpatialRelationGraph,
    TopologicalPredicate,
    EntityType,
    SRGEdge,
    extract_srg,
)

logger = logging.getLogger(__name__)


@dataclass
class CritiqueResult:
    passed: bool
    contribution_score: float = 0.0
    feasibility_score: float = 0.0
    consistency_score: float = 0.0
    grade: str = "Neutral"
    issues: list[str] = field(default_factory=list)
    suggested_revisions: list[str] = field(default_factory=list)
    corrected_plan: Optional[dict] = None

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "contribution_score": round(self.contribution_score, 3),
            "feasibility_score": round(self.feasibility_score, 3),
            "consistency_score": round(self.consistency_score, 3),
            "grade": self.grade,
            "issues": self.issues,
            "suggested_revisions": self.suggested_revisions,
        }


SPATIAL_RULES: list[dict] = [
    {
        "name": "separated_implies_water_between",
        "condition": lambda srg, plan: any(
            e.predicate == TopologicalPredicate.SEPARATED_BY for e in srg.edges
        ),
        "check": lambda srg, plan: _check_separated_has_water(srg, plan),
        "message": "若A与B被C隔开，则C必须是连通的且其主轴应横跨A与B的质心连线",
    },
    {
        "name": "single_island_no_fragmentation",
        "condition": lambda srg, plan: plan.get("kind") == "single_island",
        "check": lambda srg, plan: _check_single_island_integrity(srg, plan),
        "message": "single_island模式下不应存在群岛或多大陆",
    },
    {
        "name": "two_continents_must_disconnect",
        "condition": lambda srg, plan: plan.get("kind") == "two_continents_with_rift_sea",
        "check": lambda srg, plan: _check_two_continents_disconnect(srg, plan),
        "message": "two_continents模式下必须确保两大陆不连通",
    },
    {
        "name": "position_consistency",
        "condition": lambda srg, plan: len(srg.entities) >= 2,
        "check": lambda srg, plan: _check_position_consistency(srg, plan),
        "message": "实体的空间位置应与拓扑谓词一致",
    },
    {
        "name": "inland_sea_enclosure",
        "condition": lambda srg, plan: any(
            e.entity_type == EntityType.INLAND_SEA for e in srg.entities
        ),
        "check": lambda srg, plan: _check_inland_sea_enclosure(srg, plan),
        "message": "内海必须被陆地包围或半包围",
    },
    {
        "name": "mountain_on_land",
        "condition": lambda srg, plan: any(
            e.entity_type == EntityType.MOUNTAIN for e in srg.entities
        ),
        "check": lambda srg, plan: _check_mountains_on_land(srg, plan),
        "message": "山脉应位于大陆之上",
    },
    {
        "name": "sea_zone_between_continents",
        "condition": lambda srg, plan: len([
            e for e in srg.entities if e.entity_type == EntityType.CONTINENT
        ]) >= 2,
        "check": lambda srg, plan: _check_sea_between_continents(srg, plan),
        "message": "两块大陆之间应有海域分隔",
    },
]


def critique_plan(srg: SpatialRelationGraph, plan: dict) -> CritiqueResult:
    issues: list[str] = []
    revisions: list[str] = []
    contribution_scores: list[float] = []
    feasibility_scores: list[float] = []
    consistency_scores: list[float] = []

    for rule in SPATIAL_RULES:
        if not rule["condition"](srg, plan):
            continue
        result = rule["check"](srg, plan)
        if result["passed"]:
            contribution_scores.append(result.get("contribution", 0.8))
            feasibility_scores.append(result.get("feasibility", 0.9))
            consistency_scores.append(result.get("consistency", 0.9))
        else:
            issues.append(f"[{rule['name']}] {result.get('message', rule['message'])}")
            if result.get("revision"):
                revisions.append(result["revision"])
            contribution_scores.append(result.get("contribution", 0.4))
            feasibility_scores.append(result.get("feasibility", 0.3))
            consistency_scores.append(result.get("consistency", 0.3))

    contribution = sum(contribution_scores) / max(len(contribution_scores), 1)
    feasibility = sum(feasibility_scores) / max(len(feasibility_scores), 1)
    consistency = sum(consistency_scores) / max(len(consistency_scores), 1)

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

    passed = len(issues) == 0 or (len(issues) <= 1 and grade in ("Good", "Neutral"))

    corrected = None
    if issues:
        corrected = _apply_corrections(plan, issues, revisions, srg)

    return CritiqueResult(
        passed=passed,
        contribution_score=contribution,
        feasibility_score=feasibility,
        consistency_score=consistency,
        grade=grade,
        issues=issues,
        suggested_revisions=revisions,
        corrected_plan=corrected,
    )


def critique_with_iteration(
    text: str,
    initial_plan: dict,
    max_iterations: int = 3,
) -> dict:
    srg = extract_srg(text)
    plan = dict(initial_plan)

    for iteration in range(max_iterations):
        result = critique_plan(srg, plan)
        logger.info(
            "Critic iteration %d: passed=%s grade=%s issues=%d",
            iteration + 1,
            result.passed,
            result.grade,
            len(result.issues),
        )

        if result.passed:
            plan["critic_result"] = result.to_dict()
            plan["srg"] = srg.to_dict()
            return plan

        if result.corrected_plan:
            plan = result.corrected_plan
            logger.info("Critic applied corrections, retrying...")
        else:
            plan["critic_result"] = result.to_dict()
            plan["srg"] = srg.to_dict()
            logger.warning("Critic found issues but no auto-correction available: %s", result.issues)
            return plan

    plan["critic_result"] = result.to_dict()
    plan["srg"] = srg.to_dict()
    logger.warning("Critic max iterations reached, issues remain: %s", result.issues)
    return plan


def _check_separated_has_water(srg: SpatialRelationGraph, plan: dict) -> dict:
    for edge in srg.edges:
        if edge.predicate != TopologicalPredicate.SEPARATED_BY:
            continue
        sea_zones = plan.get("sea_zones") or []
        inland_seas = plan.get("inland_seas") or []
        continents = plan.get("continents") or []
        if not sea_zones and not inland_seas:
            return {
                "passed": False,
                "message": f"{edge.subject}与{edge.object}被声明为隔海相望，但plan中无海域",
                "revision": "添加center到sea_zones以在两大陆间创建分隔海域",
                "contribution": 0.3,
                "feasibility": 0.2,
                "consistency": 0.2,
            }
    return {"passed": True, "contribution": 0.9, "feasibility": 0.9, "consistency": 0.9}


def _check_single_island_integrity(srg: SpatialRelationGraph, plan: dict) -> dict:
    continents = plan.get("continents") or []
    island_chains = plan.get("island_chains") or []
    if len(continents) > 1:
        return {
            "passed": False,
            "message": f"single_island模式但有{len(continents)}个大陆",
            "revision": "将continents缩减为1个position=center的大陆",
            "contribution": 0.2,
            "feasibility": 0.2,
            "consistency": 0.1,
        }
    if island_chains:
        return {
            "passed": False,
            "message": "single_island模式不应有island_chains",
            "revision": "移除island_chains",
            "contribution": 0.3,
            "feasibility": 0.4,
            "consistency": 0.3,
        }
    return {"passed": True, "contribution": 0.9, "feasibility": 0.9, "consistency": 0.9}


def _check_two_continents_disconnect(srg: SpatialRelationGraph, plan: dict) -> dict:
    continents = plan.get("continents") or []
    sea_zones = plan.get("sea_zones") or []
    must_disconnect = plan.get("must_disconnect_pairs") or []
    positions = {str(c.get("position", "")) for c in continents}

    if len(continents) < 2:
        return {
            "passed": False,
            "message": "two_continents模式需要至少2个大陆",
            "revision": "添加west和east两个大陆",
            "contribution": 0.2,
            "feasibility": 0.1,
            "consistency": 0.1,
        }

    if not sea_zones and not must_disconnect:
        return {
            "passed": False,
            "message": "两大陆之间无分隔海域",
            "revision": "在sea_zones中添加center以创建分隔海域",
            "contribution": 0.3,
            "feasibility": 0.3,
            "consistency": 0.2,
        }

    if {"west", "east"} <= positions and "center" not in sea_zones:
        return {
            "passed": False,
            "message": "东西大陆之间缺少center海域",
            "revision": "在sea_zones中添加center",
            "contribution": 0.4,
            "feasibility": 0.5,
            "consistency": 0.4,
        }

    return {"passed": True, "contribution": 0.9, "feasibility": 0.9, "consistency": 0.9}


def _check_position_consistency(srg: SpatialRelationGraph, plan: dict) -> dict:
    for edge in srg.edges:
        subj = srg.get_entity(edge.subject)
        obj = srg.get_entity(edge.object)
        if not subj or not obj or not subj.position or not obj.position:
            continue

        if edge.predicate == TopologicalPredicate.NORTH_OF:
            if subj.position in ("south", "southwest", "southeast") and obj.position in ("north", "northwest", "northeast"):
                return {
                    "passed": False,
                    "message": f"{subj.name}声明在{obj.name}北侧，但位置{subj.position}与{obj.position}矛盾",
                    "revision": f"调整{subj.name}位置为north或{obj.name}位置为south",
                    "contribution": 0.3,
                    "feasibility": 0.4,
                    "consistency": 0.2,
                }

        if edge.predicate == TopologicalPredicate.EAST_OF:
            if subj.position in ("west", "southwest", "northwest") and obj.position in ("east", "southeast", "northeast"):
                return {
                    "passed": False,
                    "message": f"{subj.name}声明在{obj.name}东侧，但位置{subj.position}与{obj.position}矛盾",
                    "revision": f"调整{subj.name}位置为east或{obj.name}位置为west",
                    "contribution": 0.3,
                    "feasibility": 0.4,
                    "consistency": 0.2,
                }

    return {"passed": True, "contribution": 0.9, "feasibility": 0.9, "consistency": 0.9}


def _check_inland_sea_enclosure(srg: SpatialRelationGraph, plan: dict) -> dict:
    inland_seas = [e for e in srg.entities if e.entity_type == EntityType.INLAND_SEA]
    continents = [e for e in srg.entities if e.entity_type == EntityType.CONTINENT]

    if not continents:
        return {
            "passed": False,
            "message": "内海存在但无大陆包围",
            "revision": "添加南北或东西大陆以包围内海",
            "contribution": 0.3,
            "feasibility": 0.2,
            "consistency": 0.2,
        }

    has_enclosure_edge = any(
        e.predicate in (TopologicalPredicate.ENCLOSED_BY, TopologicalPredicate.ENCLOSES)
        for e in srg.edges
    )
    if not has_enclosure_edge and len(continents) < 2:
        return {
            "passed": False,
            "message": "内海需要至少2块大陆形成包围",
            "revision": "添加north和south大陆以包围内海",
            "contribution": 0.4,
            "feasibility": 0.5,
            "consistency": 0.4,
        }

    return {"passed": True, "contribution": 0.8, "feasibility": 0.9, "consistency": 0.8}


def _check_mountains_on_land(srg: SpatialRelationGraph, plan: dict) -> dict:
    mountains = [e for e in srg.entities if e.entity_type == EntityType.MOUNTAIN]
    continents = [e for e in srg.entities if e.entity_type == EntityType.CONTINENT]

    for mtn in mountains:
        if mtn.position == "center" and not continents:
            sea_zones = plan.get("sea_zones") or []
            if "center" in sea_zones:
                return {
                    "passed": False,
                    "message": f"山脉{mtn.name}位于center但center是海域",
                    "revision": "将山脉位置调整到大陆所在方向",
                    "contribution": 0.3,
                    "feasibility": 0.3,
                    "consistency": 0.2,
                }

    return {"passed": True, "contribution": 0.8, "feasibility": 0.9, "consistency": 0.8}


def _check_sea_between_continents(srg: SpatialRelationGraph, plan: dict) -> dict:
    continents = [e for e in srg.entities if e.entity_type == EntityType.CONTINENT]
    if len(continents) < 2:
        return {"passed": True, "contribution": 0.8, "feasibility": 0.9, "consistency": 0.8}

    has_separation = any(
        e.predicate == TopologicalPredicate.SEPARATED_BY
        for e in srg.edges
    )
    sea_zones = plan.get("sea_zones") or []

    if has_separation and not sea_zones:
        return {
            "passed": False,
            "message": "大陆间声明隔海但plan中无sea_zones",
            "revision": "添加center到sea_zones",
            "contribution": 0.3,
            "feasibility": 0.3,
            "consistency": 0.2,
        }

    return {"passed": True, "contribution": 0.9, "feasibility": 0.9, "consistency": 0.9}


def _apply_corrections(
    plan: dict, issues: list[str], revisions: list[str], srg: SpatialRelationGraph
) -> dict:
    corrected = dict(plan)

    for revision in revisions:
        if "添加center到sea_zones" in revision:
            sea_zones = list(corrected.get("sea_zones") or [])
            if "center" not in sea_zones:
                sea_zones.append("center")
            corrected["sea_zones"] = sea_zones

        if "continents缩减为1个" in revision:
            continents = list(corrected.get("continents") or [])
            if len(continents) > 1:
                corrected["continents"] = [continents[0]]
            elif not continents:
                corrected["continents"] = [{"position": "center", "size": 0.38}]

        if "移除island_chains" in revision:
            corrected["island_chains"] = []

        if "添加west和east两个大陆" in revision:
            corrected["continents"] = [
                {"position": "west", "size": 0.38},
                {"position": "east", "size": 0.38},
            ]

        if "添加north和south大陆" in revision:
            existing = list(corrected.get("continents") or [])
            positions = {c.get("position") for c in existing}
            if "north" not in positions:
                existing.append({"position": "north", "size": 0.28})
            if "south" not in positions:
                existing.append({"position": "south", "size": 0.28})
            corrected["continents"] = existing

        if "将山脉位置调整" in revision:
            continents = list(corrected.get("continents") or [])
            if continents:
                land_positions = [c.get("position", "center") for c in continents]
                mountains = list(corrected.get("mountains") or [])
                for mtn in mountains:
                    if mtn.get("location") == "center" and "center" not in land_positions:
                        mtn["location"] = land_positions[0]
                corrected["mountains"] = mountains

    if "must_disconnect_pairs" not in corrected:
        corrected["must_disconnect_pairs"] = []
    for edge in srg.edges:
        if edge.predicate == TopologicalPredicate.SEPARATED_BY:
            pair = sorted([edge.subject, edge.object])
            if pair not in corrected["must_disconnect_pairs"]:
                corrected["must_disconnect_pairs"].append(pair)

    return corrected


def evaluate_terrain_consistency(
    elevation,
    srg: SpatialRelationGraph,
    plan: dict,
) -> dict:
    import numpy as np
    from scipy.ndimage import label as ndimage_label

    scores: dict[str, float] = {}
    land_mask = elevation > 0.0
    water_mask = elevation <= 0.0

    labeled_land, n_land = ndimage_label(land_mask.astype(np.int8))
    labeled_water, n_water = ndimage_label(water_mask.astype(np.int8))

    kind = plan.get("kind", "")

    if kind == "single_island":
        target = 1
        ratio = min(n_land, target) / max(n_land, target) if target > 0 else 0.0
        scores["landmass_count_match"] = round(ratio, 3)

    elif kind == "two_continents_with_rift_sea":
        target = 2
        ratio = min(n_land, target) / max(n_land, target) if target > 0 else 0.0
        scores["landmass_count_match"] = round(ratio, 3)
        if n_water > 0:
            center_mask = np.zeros_like(water_mask)
            h, w = water_mask.shape
            center_mask[int(h * 0.3):int(h * 0.7), int(w * 0.35):int(w * 0.65)] = True
            center_water = water_mask & center_mask
            if np.any(center_water):
                scores["rift_sea_presence"] = 0.9
            else:
                scores["rift_sea_presence"] = 0.2

    elif kind == "central_enclosed_inland_sea":
        if n_water > 0:
            center_mask = np.zeros_like(water_mask)
            h, w = water_mask.shape
            center_mask[int(h * 0.25):int(h * 0.75), int(w * 0.28):int(w * 0.72)] = True
            center_water = water_mask & center_mask
            if np.any(center_water):
                scores["inland_sea_presence"] = 0.85
            else:
                scores["inland_sea_presence"] = 0.15

    for edge in srg.edges:
        if edge.predicate == TopologicalPredicate.SEPARATED_BY:
            subj = srg.get_entity(edge.subject)
            obj = srg.get_entity(edge.object)
            if subj and obj and subj.position and obj.position:
                from app.core.semantic_mapper import resolve_position_continuous
                sy, sx = resolve_position_continuous(subj.position)
                oy, ox = resolve_position_continuous(obj.position)
                h, w = elevation.shape
                subj_land = land_mask[int(sy * h), int(sx * w)]
                obj_land = land_mask[int(oy * h), int(ox * w)]
                mid_y = int((sy + oy) / 2 * h)
                mid_x = int((sx + ox) / 2 * w)
                mid_water = water_mask[mid_y, mid_x]
                if subj_land and obj_land and mid_water:
                    scores["separation_verified"] = 0.9
                elif subj_land and obj_land:
                    scores["separation_verified"] = 0.4
                else:
                    scores["separation_verified"] = 0.1

    if scores:
        scores["overall"] = round(sum(scores.values()) / len(scores), 3)

    return scores
