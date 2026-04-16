from __future__ import annotations

import logging
import re
from typing import Iterable, List, Optional

import numpy as np
from pydantic import BaseModel, Field

from app.core.terrain import TerrainGenerator

logger = logging.getLogger(__name__)

POSITION_ALIASES = {
    "northwest": ["northwest", "north west", "nw", "西北"],
    "north": ["north", "northern", "north side", "北", "北部", "北方"],
    "northeast": ["northeast", "north east", "ne", "东北"],
    "west": ["west", "western", "west side", "西", "西部", "西侧"],
    "center": ["center", "central", "middle", "mid", "中央", "中心", "中部", "中间"],
    "east": ["east", "eastern", "east side", "东", "东部", "东侧"],
    "southwest": ["southwest", "south west", "sw", "西南"],
    "south": ["south", "southern", "south side", "南", "南部", "南方"],
    "southeast": ["southeast", "south east", "se", "东南"],
}

COMPOSITE_POSITION_HINTS = {
    "northwest": ["northwestern", "north west side", "西北部", "西北侧"],
    "northeast": ["northeastern", "north east side", "东北部", "东北侧"],
    "southwest": ["southwestern", "south west side", "西南部", "西南侧"],
    "southeast": ["southeastern", "south east side", "东南部", "东南侧"],
    "west": ["central west", "west of center", "midwest", "中部偏西", "中西部"],
    "east": ["central east", "east of center", "中部偏东", "中东部"],
    "north": ["north central", "central north", "中北部"],
    "south": ["south central", "central south", "中南部"],
    "center": ["central inland", "inner center", "腹地", "内陆中心"],
}

CONTINENT_TERMS = ["continent", "continents", "landmass", "mainland", "大陆", "陆地"]
MOUNTAIN_TERMS = ["mountain", "mountains", "range", "ridge", "peaks", "mount", "山", "山脉", "高山"]
SEA_TERMS = ["sea", "ocean", "gulf", "channel", "strait", "海", "海洋", "内海", "海峡"]
RIVER_TERMS = ["river", "rivers", "stream", "watershed", "河", "河流", "水系"]
SEPARATOR_TERMS = ["between", "separate", "divid", "split", "隔开", "分隔", "之间", "分开"]


class ContinentConstraint(BaseModel):
    position: str = Field(..., description="west/east/north/south/center/northwest/northeast/southwest/southeast")
    size: float = Field(0.4, ge=0.1, le=0.85)


class MountainConstraint(BaseModel):
    location: str = Field(..., description="mountain location")
    height: float = Field(0.8, ge=0.2, le=1.0)


class MapConstraints(BaseModel):
    continents: List[ContinentConstraint] = Field(default_factory=list)
    mountains: List[MountainConstraint] = Field(default_factory=list)
    sea_zones: List[str] = Field(default_factory=list)
    river_sources: List[str] = Field(default_factory=list)


class WorldProfile(BaseModel):
    land_ratio: float = Field(0.44, ge=0.2, le=0.8)
    ruggedness: float = Field(0.55, ge=0.0, le=1.0)
    coast_complexity: float = Field(0.5, ge=0.0, le=1.0)
    island_factor: float = Field(0.25, ge=0.0, le=1.0)
    moisture: float = Field(1.0, ge=0.4, le=1.8)
    temperature_bias: float = Field(0.0, ge=-14.0, le=14.0)
    wind_direction: str = Field("westerly")
    palette_hint: str = Field("temperate")
    layout_template: str = Field("default")
    sea_style: str = Field("open")


class ConstraintMapper:
    """Compatibility wrapper that applies constraints through TerrainGenerator."""

    def __init__(self, width: int, height: int, seed: int = 42):
        self.width = width
        self.height = height
        self.seed = seed
        self._terrain = TerrainGenerator(width=width, height=height, seed=seed)

    def location_to_mask(self, location_desc: str, radius: float = 0.2, sigma: float = 0.4) -> np.ndarray:
        return self._terrain._create_location_mask(location_desc, radius=radius, sigma=sigma)

    def apply_constraints(self, constraints: MapConstraints, base_elev: np.ndarray) -> np.ndarray:
        normalized = normalize_constraints(constraints)
        data = normalized.model_dump() if hasattr(normalized, "model_dump") else normalized.dict()
        return self._terrain.apply_constraints(base_elev, data)


def parse_constraints(
    user_prompt: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
) -> MapConstraints:
    if api_key and api_key.strip():
        try:
            return normalize_constraints(_parse_with_llm(user_prompt, api_key, base_url, model))
        except Exception as exc:
            logger.warning("LLM parsing failed, fallback to rule-based parser: %s", exc)

    return normalize_constraints(_rule_based_parse(user_prompt))


def _parse_with_llm(
    user_prompt: str,
    api_key: str,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
) -> MapConstraints:
    try:
        import instructor
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError("Missing LLM parsing dependencies: instructor, openai") from exc

    client_kwargs = {"api_key": api_key.strip()}
    if base_url and base_url.strip():
        client_kwargs["base_url"] = base_url.strip()

    openai_client = OpenAI(**client_kwargs)
    try:
        client = instructor.from_openai(openai_client, mode=instructor.Mode.JSON)
    except AttributeError:
        client = instructor.patch(openai_client)

    return client.chat.completions.create(
        model=model or "gpt-4o-mini",
        response_model=MapConstraints,
        max_retries=2,
        messages=[
            {
                "role": "system",
                "content": (
                    "Extract map-generation constraints from the user's prompt. "
                    "Return only JSON matching MapConstraints. "
                    "Support continent placement, mountain placement, sea zones, and river sources. "
                    "Valid positions: northwest, north, northeast, west, center, east, southwest, south, southeast."
                ),
            },
            {"role": "user", "content": user_prompt},
        ],
    )


def _rule_based_parse(user_prompt: str) -> MapConstraints:
    prompt = (user_prompt or "").strip()
    lower = prompt.lower()
    constraints = MapConstraints()
    if not prompt:
        return constraints

    positions = _extract_positions(prompt)

    for position in _extract_continent_positions(prompt, positions):
        constraints.continents.append(
            ContinentConstraint(position=position, size=_extract_size(lower, position))
        )

    if _contains_any(lower, MOUNTAIN_TERMS):
        mountain_positions = _extract_feature_positions(prompt, positions, MOUNTAIN_TERMS)
        for position in mountain_positions[:3]:
            constraints.mountains.append(
                MountainConstraint(location=position, height=_extract_mountain_height(lower))
            )

    if _contains_any(lower, SEA_TERMS):
        sea_positions = _extract_feature_positions(prompt, positions, SEA_TERMS)
        if sea_positions:
            constraints.sea_zones.extend(sea_positions[:3])
    if _contains_any(lower, SEPARATOR_TERMS):
        if len(constraints.continents) >= 2:
            constraints.sea_zones.append(_infer_separator_zone(constraints.continents))
        elif not constraints.sea_zones:
            constraints.sea_zones.append("center")

    if _contains_any(lower, RIVER_TERMS):
        if constraints.mountains:
            constraints.river_sources = [item.location for item in constraints.mountains[:2]]
        elif positions:
            constraints.river_sources = positions[:2]
        else:
            constraints.river_sources = ["center"]

    constraints.sea_zones = _dedupe(constraints.sea_zones)
    constraints.river_sources = _dedupe(constraints.river_sources)
    return constraints


def normalize_constraints(constraints: MapConstraints) -> MapConstraints:
    normalized = MapConstraints()

    for continent in constraints.continents:
        normalized_position = _canonicalize_position(continent.position)
        normalized.continents.append(
            ContinentConstraint(
                position=normalized_position,
                size=float(np.clip(continent.size, 0.1, 0.85)),
            )
        )

    for mountain in constraints.mountains:
        normalized_location = _canonicalize_position(mountain.location)
        normalized.mountains.append(
            MountainConstraint(
                location=normalized_location,
                height=float(np.clip(mountain.height, 0.2, 1.0)),
            )
        )

    normalized.sea_zones = _normalize_position_list(constraints.sea_zones)
    normalized.river_sources = _normalize_position_list(constraints.river_sources)
    return normalized


def get_constraints_summary(constraints: MapConstraints) -> str:
    parts: List[str] = []
    if constraints.continents:
        items = ", ".join(f"{item.position}({item.size:.2f})" for item in constraints.continents)
        parts.append(f"Continents: {items}")
    if constraints.mountains:
        items = ", ".join(f"{item.location}({item.height:.2f})" for item in constraints.mountains)
        parts.append(f"Mountains: {items}")
    if constraints.sea_zones:
        parts.append(f"Sea zones: {', '.join(constraints.sea_zones)}")
    if constraints.river_sources:
        parts.append(f"River sources: {', '.join(constraints.river_sources)}")
    return "\n".join(parts) if parts else "No special constraints detected."


def infer_world_profile(prompt: str, constraints: MapConstraints) -> WorldProfile:
    lower = _normalize_prompt(prompt)
    profile = WorldProfile()
    has_inland_sea = _contains_any(lower, ["inland sea", "inner sea", "enclosed sea", "内海", "内陆海"])
    has_open_separator = _contains_any(lower, ["隔着海", "被海隔开", "ocean between", "sea between", "separated by sea"])
    has_strait_or_bay = _contains_any(lower, ["海峡", "strait", "gulf", "bay", "海湾"])

    if _contains_any(lower, ["surrounded by sea", "island continent", "四面环海", "环海大陆", "海中大陆"]):
        profile.layout_template = "single_island"
        profile.land_ratio = 0.38
        profile.coast_complexity = 0.88
        profile.island_factor = 0.18
        profile.sea_style = "open"
    elif _contains_any(lower, ["supercontinent", "single continent", "盘古大陆", "超大陆", "单一大陆"]):
        profile.layout_template = "supercontinent"
        profile.land_ratio = 0.62
        profile.island_factor = 0.08
    elif len(constraints.continents) >= 3 or _contains_any(lower, ["archipelago", "islands", "群岛", "列岛"]):
        profile.layout_template = "archipelago"
        profile.land_ratio = 0.32
        profile.island_factor = 0.75
        profile.coast_complexity = 0.82
    elif len(constraints.continents) == 2:
        positions = {item.position for item in constraints.continents}
        if {"west", "east"} <= positions:
            profile.layout_template = "split_east_west"
        elif {"north", "south"} <= positions:
            profile.layout_template = "split_north_south"
        profile.land_ratio = 0.46
        profile.coast_complexity = 0.62

    if has_inland_sea:
        profile.sea_style = "inland"
    elif has_open_separator:
        profile.sea_style = "open"
    elif has_strait_or_bay:
        profile.sea_style = "strait"

    if constraints.sea_zones or has_inland_sea or has_strait_or_bay:
        profile.coast_complexity = max(profile.coast_complexity, 0.72)
        profile.land_ratio = min(profile.land_ratio, 0.5)
        if has_inland_sea and len(constraints.continents) >= 2:
            profile.layout_template = "mediterranean"
        elif profile.layout_template == "default" and len(constraints.continents) >= 2:
            profile.layout_template = "mediterranean"

    if has_open_separator and profile.layout_template == "default" and len(constraints.continents) >= 2:
        positions = {item.position for item in constraints.continents}
        if {"west", "east"} <= positions:
            profile.layout_template = "split_east_west"
        elif {"north", "south"} <= positions:
            profile.layout_template = "split_north_south"

    if constraints.mountains or _contains_any(lower, ["mountainous", "rugged", "山脉", "高山", "峡谷", "崇山"]):
        profile.ruggedness = 0.85
    if _contains_any(lower, ["flat", "plain", "gentle", "平原", "平坦", "缓丘"]):
        profile.ruggedness = 0.28

    if _contains_any(lower, ["jagged", "fractured", "broken coast", "崎岖海岸", "破碎海岸"]):
        profile.coast_complexity = 0.9
    if _contains_any(lower, ["smooth coast", "round coast", "平滑海岸", "圆润海岸"]):
        profile.coast_complexity = 0.24

    if _contains_any(lower, ["desert", "arid", "dry", "沙漠", "干旱"]):
        profile.moisture = 0.55
        profile.temperature_bias = max(profile.temperature_bias, 4.0)
        profile.palette_hint = "arid"
    elif _contains_any(lower, ["lush", "wet", "rainforest", "swamp", "湿润", "雨林", "沼泽"]):
        profile.moisture = 1.45
        profile.palette_hint = "lush"

    if _contains_any(lower, ["frozen", "glacial", "icy", "tundra", "冰雪", "冻土", "寒冷"]):
        profile.temperature_bias = -9.0
        profile.palette_hint = "frozen"
    elif _contains_any(lower, ["tropical", "equatorial", "warm", "热带", "温暖", "赤道"]):
        profile.temperature_bias = 6.0
        profile.moisture = max(profile.moisture, 1.15)
        profile.palette_hint = "tropical"

    if _contains_any(lower, ["easterly", "eastern winds", "东风"]):
        profile.wind_direction = "easterly"
    elif _contains_any(lower, ["northerly", "north wind", "北风"]):
        profile.wind_direction = "northerly"
    elif _contains_any(lower, ["southerly", "south wind", "南风"]):
        profile.wind_direction = "southerly"

    if _contains_any(lower, ["volcanic", "lava", "火山", "熔岩"]):
        profile.ruggedness = max(profile.ruggedness, 0.92)
        profile.palette_hint = "volcanic"
        profile.temperature_bias = max(profile.temperature_bias, 3.0)

    return profile


def parse_with_rag(
    user_prompt: str,
    examples: list[dict] | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
) -> dict:
    constraints = parse_constraints(user_prompt, api_key=api_key, base_url=base_url, model=model)
    constraints = _enrich_constraints_from_sea_language(user_prompt, constraints)
    profile = infer_world_profile(user_prompt, constraints)
    generation_backend = _select_generation_backend(user_prompt, profile)

    plan = {
        "constraints": constraints.model_dump(mode="json"),
        "profile": profile.model_dump(mode="json"),
        "generation_backend": generation_backend,
        "continents": [{"position": item.position, "size": item.size} for item in constraints.continents],
        "mountains": [
            {"location": item.location, "height": item.height, "orientation": _infer_mountain_orientation(item.location)}
            for item in constraints.mountains
        ],
        "island_chains": _extract_island_chains(user_prompt),
        "peninsulas": _extract_peninsulas(user_prompt),
        "inland_seas": _extract_inland_seas(user_prompt, constraints, profile),
        "river_hints": [{"region": region, "length": "long"} for region in constraints.river_sources],
        "water_bodies": _extract_water_bodies(constraints, profile),
        "regional_relations": _extract_regional_relations(user_prompt, constraints),
        "module_sequence": _build_module_sequence(user_prompt, constraints, profile, generation_backend),
        "climate_hints": _build_climate_hints(profile),
    }
    if examples:
        plan = _merge_rag_examples(user_prompt, plan, examples)
    plan["constraints"] = _sync_constraints_from_plan(plan, constraints).model_dump(mode="json")
    plan["profile"] = _sync_profile_from_plan(user_prompt, plan, profile).model_dump(mode="json")
    return plan


def _enrich_constraints_from_sea_language(user_prompt: str, constraints: MapConstraints) -> MapConstraints:
    lower = _normalize_prompt(user_prompt)
    enriched = normalize_constraints(constraints)
    continents = list(enriched.continents)
    positions = {item.position for item in continents}

    has_inland = _contains_any(lower, ["inland sea", "inner sea", "enclosed sea", "内海", "内陆海"])
    has_middle_separator = _contains_any(
        lower,
        [
            "separated by sea",
            "sea between",
            "ocean between",
            "through the middle",
            "middle sea",
            "海隔开",
            "被海隔开",
            "中间被海隔开",
        ],
    )
    has_north_south_enclosure = _contains_any(
        lower,
        [
            "north and south",
            "to the north and south",
            "land to the north and south",
            "north and south sides",
            "南北两侧",
            "北部和南部",
        ],
    )
    has_east_west_split = _contains_any(
        lower,
        [
            "east and west",
            "west and east",
            "eastern and western",
            "东西两侧",
            "东部和西部",
        ],
    )

    if has_middle_separator or (has_east_west_split and _contains_any(lower, SEA_TERMS)):
        if len(continents) < 2 or positions == {"center"}:
            enriched.continents = [
                ContinentConstraint(position="west", size=0.28),
                ContinentConstraint(position="east", size=0.28),
            ]
        if "center" not in enriched.sea_zones:
            enriched.sea_zones.append("center")

    if has_inland:
        if has_north_south_enclosure or not enriched.continents or {item.position for item in enriched.continents} == {"center"}:
            enriched.continents = [
                ContinentConstraint(position="north", size=0.28),
                ContinentConstraint(position="south", size=0.28),
            ]
        if "center" not in enriched.sea_zones:
            enriched.sea_zones.append("center")

    enriched.sea_zones = _dedupe(enriched.sea_zones)
    return normalize_constraints(enriched)


def _extract_positions(prompt: str) -> List[str]:
    normalized = _normalize_prompt(prompt)
    results: List[str] = []
    for canonical, phrases in COMPOSITE_POSITION_HINTS.items():
        if any(phrase in normalized for phrase in phrases):
            results.append(canonical)
    for canonical, aliases in POSITION_ALIASES.items():
        if any(_alias_matches(normalized, alias) for alias in aliases):
            results.append(canonical)
    return _dedupe(results)


def _infer_mountain_orientation(location: str) -> str | None:
    normalized = _canonicalize_position(location)
    if normalized in {"north", "south"}:
        return "east-west"
    if normalized in {"east", "west"}:
        return "north-south"
    if normalized in {"northwest", "northeast", "southwest", "southeast"}:
        return "arc"
    return None


def _extract_island_chains(prompt: str) -> list[dict]:
    normalized = _normalize_prompt(prompt)
    if _contains_any(normalized, ["archipelago", "island chain", "群岛", "列岛", "岛链"]):
        positions = _extract_positions(prompt)
        return [{"position": position, "density": 0.66} for position in (positions[:2] or ["center"])]
    return []


def _extract_peninsulas(prompt: str) -> list[dict]:
    normalized = _normalize_prompt(prompt)
    if not _contains_any(normalized, ["peninsula", "半岛"]):
        return []
    positions = _extract_positions(prompt)
    return [{"location": position, "size": 0.18} for position in (positions[:1] or ["west"])]


def _extract_inland_seas(prompt: str, constraints: MapConstraints, profile: WorldProfile) -> list[dict]:
    normalized = _normalize_prompt(prompt)
    if profile.sea_style != "inland" and not _contains_any(normalized, ["内海", "inner sea", "inland sea", "内陆海"]):
        return []
    position = (constraints.sea_zones[:1] or ["center"])[0]
    connection = "strait" if _contains_any(normalized, ["海峡", "strait"]) else "east ocean"
    return [{"position": position, "connection": connection}]


def _build_climate_hints(profile: WorldProfile) -> list[str]:
    hints = [profile.palette_hint, profile.layout_template, profile.sea_style]
    if profile.temperature_bias <= -4:
        hints.append("cold")
    elif profile.temperature_bias >= 4:
        hints.append("warm")
    if profile.moisture <= 0.7:
        hints.append("dry")
    elif profile.moisture >= 1.25:
        hints.append("wet")
    return _dedupe([hint for hint in hints if hint and hint != "default"])


def _select_generation_backend(user_prompt: str, profile: WorldProfile) -> str:
    normalized = _normalize_prompt(user_prompt)
    modular_terms = [
        "模块化",
        "程序化",
        "modular",
        "procedural",
        "terrace",
        "terraced",
        "plateau",
        "mesa",
        "台地",
        "阶梯",
        "阶地",
    ]
    if _contains_any(normalized, modular_terms):
        return "modular"
    if profile.layout_template in {"single_island", "archipelago"} and _contains_any(normalized, ["ridge", "山脊", "脊线", "plateau", "台地"]):
        return "modular"
    return "gaussian_voronoi"


def _build_module_sequence(
    user_prompt: str,
    constraints: MapConstraints,
    profile: WorldProfile,
    generation_backend: str,
) -> list[dict]:
    if generation_backend != "modular":
        return []

    normalized = _normalize_prompt(user_prompt)
    sequence: list[dict] = [
        {"module": "noise", "params": {"scale": 175.0, "octaves": 4, "amplitude": 0.16, "operation": "add"}},
        {"module": "ridged_noise", "params": {"scale": 68.0, "octaves": 4, "amplitude": 0.14, "operation": "add"}},
    ]

    for continent in constraints.continents:
        sequence.append(
            {
                "module": "continent",
                "params": {
                    "position": continent.position,
                    "size": continent.size,
                    "height": 0.88,
                    "operation": "add",
                },
            }
        )

    for mountain in constraints.mountains:
        sequence.append(
            {
                "module": "gaussian_mountain",
                "params": {
                    "location": mountain.location,
                    "height": mountain.height,
                    "sigma": 0.12,
                    "operation": "add",
                },
            }
        )
        sequence.append(
            {
                "module": "ridge",
                "params": {
                    "location": mountain.location,
                    "height": mountain.height,
                    "operation": "add",
                },
            }
        )

    if _contains_any(normalized, ["台地", "高原", "plateau", "mesa", "terrace", "阶梯"]):
        plateau_position = constraints.mountains[0].location if constraints.mountains else (constraints.continents[0].position if constraints.continents else "center")
        sequence.append(
            {
                "module": "plateau",
                "params": {
                    "position": plateau_position,
                    "height": 0.2,
                    "radius_y": 0.14,
                    "radius_x": 0.24,
                    "operation": "add",
                },
            }
        )

    for water_body in _extract_water_bodies(constraints, profile):
        body_type = str(water_body.get("type", "ocean")).lower()
        sequence.append(
            {
                "module": "strait" if "strait" in body_type else "water_body",
                "params": {
                    "position": water_body.get("position", "center"),
                    "coverage": water_body.get("coverage", 0.2),
                    "depth": 0.8,
                    "connection": water_body.get("connection"),
                    "operation": "subtract",
                },
            }
        )

    sequence.append({"module": "smooth", "params": {"sigma": 1.4, "operation": "replace"}})
    return sequence


def _extract_water_bodies(constraints: MapConstraints, profile: WorldProfile) -> list[dict]:
    water_bodies: list[dict] = []
    if profile.sea_style == "open" and constraints.sea_zones:
        for zone in constraints.sea_zones:
            water_bodies.append({"type": "ocean", "position": zone, "coverage": 0.4})
    elif profile.sea_style == "inland":
        for zone in constraints.sea_zones or ["center"]:
            water_bodies.append({"type": "inland_sea", "position": zone, "coverage": 0.2, "connection": "strait"})
    elif profile.sea_style == "strait":
        for zone in constraints.sea_zones or ["center"]:
            water_bodies.append({"type": "strait", "position": zone, "coverage": 0.12, "connection": "open"})
    return water_bodies


def _extract_regional_relations(user_prompt: str, constraints: MapConstraints) -> list[dict]:
    normalized = _normalize_prompt(user_prompt)
    relations: list[dict] = []
    if len(constraints.continents) >= 2 and _contains_any(normalized, ["隔开", "分隔", "between", "separate", "split"]):
        ordered = constraints.continents[:2]
        relations.append(
            {
                "relation": "separated_by_water",
                "subject": ordered[0].position,
                "object": ordered[1].position,
                "strength": 0.92,
            }
        )
    for mountain in constraints.mountains:
        relations.append(
            {
                "relation": "elevated_region",
                "subject": mountain.location,
                "object": "mountain_chain",
                "strength": float(np.clip(mountain.height, 0.2, 1.0)),
            }
        )
    return relations


def _merge_rag_examples(user_prompt: str, plan: dict, examples: list[dict]) -> dict:
    normalized = _normalize_prompt(user_prompt)
    merged = {**plan}
    for example in examples:
        world_plan = example.get("world_plan") or {}
        example_profile = world_plan.get("profile") or {}
        example_layout = example_profile.get("layout_template")
        example_sea_style = example_profile.get("sea_style")

        wants_inland = _contains_any(normalized, ["内海", "inner sea", "inland sea", "内陆海"])
        wants_open = _contains_any(normalized, ["隔着海", "被海隔开", "sea between", "ocean between", "开阔海洋", "外海"])
        wants_peninsula = _contains_any(normalized, ["半岛", "peninsula"])
        wants_archipelago = _contains_any(normalized, ["群岛", "archipelago", "岛链"])
        wants_island = _contains_any(normalized, ["四面环海", "surrounded by sea", "island continent", "环海大陆"])
        wants_mountains = _contains_any(normalized, MOUNTAIN_TERMS)

        if wants_inland and world_plan.get("inland_seas") and example_sea_style == "inland":
            merged["inland_seas"] = world_plan.get("inland_seas") or merged["inland_seas"]
            merged["continents"] = world_plan.get("continents") or merged["continents"]
            merged["profile"]["layout_template"] = example_layout or "mediterranean"
            merged["profile"]["sea_style"] = "inland"
        if wants_open and example_sea_style == "open":
            merged["inland_seas"] = []
            if example_layout in {"split_east_west", "split_north_south"}:
                merged["continents"] = world_plan.get("continents") or merged["continents"]
                merged["profile"]["layout_template"] = example_layout
            merged["profile"]["sea_style"] = "open"
        if wants_peninsula and world_plan.get("peninsulas"):
            merged["peninsulas"] = world_plan.get("peninsulas") or merged["peninsulas"]
        if wants_archipelago and world_plan.get("island_chains"):
            merged["island_chains"] = world_plan.get("island_chains") or merged["island_chains"]
            if example_layout:
                merged["profile"]["layout_template"] = example_layout
        if wants_island and example_layout == "single_island":
            merged["continents"] = world_plan.get("continents") or merged["continents"]
            merged["peninsulas"] = world_plan.get("peninsulas") or merged["peninsulas"]
            merged["profile"]["layout_template"] = "single_island"
            merged["profile"]["sea_style"] = "open"
        if wants_mountains and world_plan.get("mountains"):
            requested_positions = {item["location"] for item in merged["mountains"] if item.get("location")}
            candidate_positions = {item["location"] for item in world_plan.get("mountains") or [] if item.get("location")}
            if not requested_positions or requested_positions & candidate_positions:
                merged["mountains"] = world_plan.get("mountains") or merged["mountains"]

        if world_plan.get("river_hints") and not merged["river_hints"]:
            merged["river_hints"] = world_plan.get("river_hints") or merged["river_hints"]
    merged["climate_hints"] = _dedupe((merged.get("climate_hints") or []) + _build_climate_hints(WorldProfile(**merged["profile"])))
    return merged


def _sync_constraints_from_plan(plan: dict, fallback: MapConstraints) -> MapConstraints:
    constraints = MapConstraints(
        continents=[
            ContinentConstraint(position=item["position"], size=float(item.get("size", 0.3)))
            for item in plan.get("continents") or []
        ],
        mountains=[
            MountainConstraint(location=item["location"], height=float(item.get("height", 0.7)))
            for item in plan.get("mountains") or []
        ],
        sea_zones=[item["position"] for item in plan.get("inland_seas") or []],
        river_sources=[item["region"] for item in plan.get("river_hints") or []],
    )
    normalized = normalize_constraints(constraints)
    if not normalized.continents and fallback.continents:
        normalized.continents = fallback.continents
    if not normalized.mountains and fallback.mountains:
        normalized.mountains = fallback.mountains
    if not normalized.sea_zones and fallback.sea_zones:
        normalized.sea_zones = fallback.sea_zones
    if not normalized.river_sources and fallback.river_sources:
        normalized.river_sources = fallback.river_sources
    return normalized


def _sync_profile_from_plan(user_prompt: str, plan: dict, fallback: WorldProfile) -> WorldProfile:
    constraints = _sync_constraints_from_plan(plan, normalize_constraints(MapConstraints()))
    profile = infer_world_profile(user_prompt, constraints)
    merged = fallback.model_dump(mode="json")
    merged.update(plan.get("profile") or {})
    merged.update(
        {
            "layout_template": merged.get("layout_template") or profile.layout_template,
            "sea_style": merged.get("sea_style") or profile.sea_style,
        }
    )
    return WorldProfile(**merged)


def _extract_continent_positions(prompt: str, positions: List[str]) -> List[str]:
    normalized = _normalize_prompt(prompt)
    if any(token in normalized for token in ["east west", "western eastern", "一东一西", "东西"]):
        return ["west", "east"]
    if any(token in normalized for token in ["north south", "northern southern", "一南一北", "南北"]):
        return ["north", "south"]
    if any(token in normalized for token in ["northwest southeast", "西北 东南", "西北和东南"]):
        return ["northwest", "southeast"]
    if any(token in normalized for token in ["northeast southwest", "东北 西南", "东北和西南"]):
        return ["northeast", "southwest"]
    if _contains_any(normalized, ["archipelago", "islands", "群岛", "岛链"]):
        return positions[:3] or ["east", "west"]
    if _contains_any(normalized, CONTINENT_TERMS):
        if len(positions) >= 2:
            return positions[:2]
        return positions or ["center"]
    return []


def _extract_feature_positions(prompt: str, positions: List[str], terms: List[str]) -> List[str]:
    normalized = _normalize_prompt(prompt)
    terms_pattern = "|".join(map(re.escape, terms))
    paired_feature_positions = _extract_feature_position_pairs(normalized, terms_pattern)
    if paired_feature_positions:
        return paired_feature_positions
    ranked: List[str] = []
    for canonical, phrases in COMPOSITE_POSITION_HINTS.items():
        if any(
            re.search(rf"(?:{terms_pattern}).{{0,36}}{re.escape(phrase)}|{re.escape(phrase)}.{{0,36}}(?:{terms_pattern})", normalized)
            for phrase in phrases
        ):
            ranked.append(canonical)
    for canonical, aliases in POSITION_ALIASES.items():
        for alias in aliases:
            alias_pattern = _alias_pattern(alias)
            pattern = rf"(?:{terms_pattern}).{{0,36}}{alias_pattern}|{alias_pattern}.{{0,36}}(?:{terms_pattern})"
            if re.search(pattern, normalized):
                ranked.append(canonical)
                break
    return _dedupe(ranked or positions or ["center"])


def _extract_feature_position_pairs(normalized: str, terms_pattern: str) -> List[str]:
    results: List[str] = []
    for canonical, aliases in POSITION_ALIASES.items():
        for alias in aliases:
            alias_pattern = _alias_pattern(alias)
            if re.search(rf"{alias_pattern}.{{0,18}}(?:的)?(?:大陆|continent|landmass).{{0,18}}(?:有|with)?(?:{terms_pattern})", normalized):
                results.append(canonical)
                break
            if re.search(rf"(?:{terms_pattern}).{{0,18}}位于.{{0,8}}{alias_pattern}(?:的)?(?:大陆|continent|landmass)?", normalized):
                results.append(canonical)
                break
    return _dedupe(results)


def _extract_size(lower: str, position: str) -> float:
    local_window = _extract_position_window(lower, position)
    if _contains_any(local_window, ["huge", "giant", "massive", "large", "辽阔", "巨大"]):
        return 0.62
    if _contains_any(local_window, ["small", "tiny", "narrow", "slim", "狭长", "小型", "较小"]):
        return 0.28
    if position in {"west", "east", "north", "south"}:
        return 0.44
    return 0.52


def _extract_mountain_height(lower: str) -> float:
    if _contains_any(lower, ["towering", "very high", "lofty", "极高", "高耸"]):
        return 1.0
    if _contains_any(lower, ["high", "tall", "rugged", "高", "山脉", "高山"]):
        return 0.9
    if _contains_any(lower, ["low", "gentle", "small", "低矮", "丘陵"]):
        return 0.55
    return 0.78


def _normalize_prompt(prompt: str) -> str:
    return re.sub(r"\s+", " ", (prompt or "").lower().replace("-", " ")).strip()


def _canonicalize_position(value: str) -> str:
    normalized = _normalize_prompt(value)
    if not normalized:
        return "center"

    for canonical, phrases in COMPOSITE_POSITION_HINTS.items():
        if any(phrase in normalized for phrase in phrases):
            return canonical

    for canonical, aliases in POSITION_ALIASES.items():
        if any(_alias_matches(normalized, alias) for alias in aliases):
            return canonical

    squeezed = normalized.replace(" ", "")
    direct_mapping = {
        "northwest": "northwest",
        "northeast": "northeast",
        "southwest": "southwest",
        "southeast": "southeast",
        "north": "north",
        "south": "south",
        "west": "west",
        "east": "east",
        "center": "center",
        "central": "center",
        "middle": "center",
        "中央": "center",
        "中心": "center",
        "中部": "center",
        "西北": "northwest",
        "东北": "northeast",
        "西南": "southwest",
        "东南": "southeast",
        "北部": "north",
        "南部": "south",
        "西部": "west",
        "东部": "east",
    }
    return direct_mapping.get(squeezed, "center")


def _normalize_position_list(items: Iterable[str]) -> List[str]:
    return _dedupe([_canonicalize_position(item) for item in items if str(item).strip()])


def _extract_position_window(text: str, position: str, radius: int = 24) -> str:
    aliases = POSITION_ALIASES.get(position, []) + COMPOSITE_POSITION_HINTS.get(position, [])
    for alias in aliases:
        match = re.search(_alias_pattern(alias), text)
        if match:
            start = max(0, match.start() - radius)
            end = min(len(text), match.end() + radius)
            return text[start:end]
    return text


def _contains_any(text: str, keywords: List[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def _dedupe(items: List[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _alias_matches(text: str, alias: str) -> bool:
    return re.search(_alias_pattern(alias), text) is not None


def _alias_pattern(alias: str) -> str:
    escaped = re.escape(alias.strip().lower())
    if re.fullmatch(r"[a-z ]+", alias.strip().lower()):
        return rf"(?<![a-z]){escaped}(?![a-z])"
    return escaped


def _infer_separator_zone(continents: List[ContinentConstraint]) -> str:
    positions = {item.position for item in continents}
    if {"west", "east"} <= positions:
        return "center"
    if {"north", "south"} <= positions:
        return "center"
    if any(pos in positions for pos in {"northwest", "northeast"}) and any(
        pos in positions for pos in {"southwest", "southeast"}
    ):
        return "center"
    if "west" in positions:
        return "east"
    if "east" in positions:
        return "west"
    return "center"
