from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

MUTUALLY_EXCLUSIVE_KINDS = {
    "single_island": {"archipelago_chain", "peninsula_coast", "two_continents_with_rift_sea"},
    "archipelago_chain": {"single_island", "peninsula_coast", "central_enclosed_inland_sea"},
    "peninsula_coast": {"single_island", "archipelago_chain"},
    "two_continents_with_rift_sea": {"single_island", "archipelago_chain"},
    "central_enclosed_inland_sea": {"archipelago_chain"},
}


def validate_world_plan(plan: dict) -> list[str]:
    issues: list[str] = []
    topology_intent = plan.get("topology_intent") or {}
    kind = str(topology_intent.get("kind", "")).strip().lower()
    continents = list(plan.get("continents") or [])
    inland_seas = list(plan.get("inland_seas") or [])
    island_chains = list(plan.get("island_chains") or [])
    peninsulas = list(plan.get("peninsulas") or [])
    profile = plan.get("profile") or {}

    if kind == "single_island":
        if len(continents) > 1:
            issues.append("single_island should have exactly 1 continent, found {}".format(len(continents)))
        if island_chains:
            issues.append("single_island must not have island_chains")
        if peninsulas:
            issues.append("single_island must not have peninsulas")
        if not topology_intent.get("forbid_cross_cut"):
            issues.append("single_island should have forbid_cross_cut=True")

    if kind == "archipelago_chain":
        if not island_chains and len(continents) < 2:
            issues.append("archipelago_chain should have island_chains or multiple continents")

    if kind == "two_continents_with_rift_sea":
        positions = {str(c.get("position", "")) for c in continents}
        if not ({"west", "east"} <= positions or len(continents) >= 2):
            issues.append("two_continents_with_rift_sea should have west+east continents")
        if not topology_intent.get("forbid_cross_cut"):
            issues.append("two_continents_with_rift_sea should have forbid_cross_cut=True")

    if kind == "central_enclosed_inland_sea":
        if not inland_seas and profile.get("sea_style") != "inland":
            issues.append("central_enclosed_inland_sea should have inland_seas or sea_style=inland")

    if kind == "peninsula_coast":
        if not peninsulas:
            issues.append("peninsula_coast should have peninsulas")

    if kind and island_chains and kind == "single_island":
        issues.append("single_island and archipelago are mutually exclusive")

    if kind == "two_continents_with_rift_sea" and topology_intent.get("modifiers", {}).get("rift_width") == "broad":
        if profile.get("land_ratio", 0.44) > 0.55:
            issues.append("broad rift with high land_ratio may cause cross-cut issues")

    return issues


def auto_fix_world_plan(plan: dict) -> dict:
    fixed = dict(plan)
    topology_intent = dict(fixed.get("topology_intent") or {})
    kind = str(topology_intent.get("kind", "")).strip().lower()
    continents = list(fixed.get("continents") or [])
    island_chains = list(fixed.get("island_chains") or [])
    peninsulas = list(fixed.get("peninsulas") or [])
    inland_seas = list(fixed.get("inland_seas") or [])

    if kind == "single_island":
        if len(continents) > 1:
            fixed["continents"] = [continents[0]]
        if island_chains:
            fixed["island_chains"] = []
        if peninsulas:
            fixed["peninsulas"] = []
        topology_intent["forbid_cross_cut"] = True
        topology_intent.setdefault("target_land_component_count", 1)
        topology_intent.setdefault("target_water_component_count", 1)

    if kind == "two_continents_with_rift_sea":
        topology_intent["forbid_cross_cut"] = True
        topology_intent.setdefault("target_land_component_count", 2)
        positions = {str(c.get("position", "")) for c in fixed.get("continents", [])}
        if not ({"west", "east"} <= positions):
            fixed["continents"] = [
                {"position": "west", "size": 0.38},
                {"position": "east", "size": 0.38},
            ]

    if kind == "central_enclosed_inland_sea":
        topology_intent.setdefault("target_water_component_count", 1)
        if not inland_seas:
            fixed["inland_seas"] = [{"position": "center", "connection": "enclosed"}]

    if kind == "archipelago_chain":
        topology_intent.setdefault("min_land_component_count", 3)

    if kind == "peninsula_coast":
        if not peninsulas:
            fixed["peninsulas"] = [{"location": "east", "size": 0.2}]

    fixed["topology_intent"] = topology_intent
    return fixed
