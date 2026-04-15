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
