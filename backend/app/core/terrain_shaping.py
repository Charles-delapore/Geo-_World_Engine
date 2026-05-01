from __future__ import annotations

import logging

import numpy as np

from app.core.terrain import TerrainGenerator
from app.utils.helpers import safe_dict

logger = logging.getLogger(__name__)


def _as_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _coerce_continent(item) -> dict | None:
    if isinstance(item, dict):
        return {
            "position": str(item.get("position") or item.get("location") or "center"),
            "size": float(item.get("size", 0.38)),
        }
    if isinstance(item, str) and item.strip():
        return {"position": item.strip(), "size": 0.38}
    return None


def _coerce_mountain(item) -> dict | None:
    if isinstance(item, dict):
        location = str(item.get("location") or item.get("position") or "center")
        return {
            "location": location,
            "position": location,
            "height": float(item.get("height", 0.8)),
        }
    if isinstance(item, str) and item.strip():
        return {"location": item.strip(), "position": item.strip(), "height": 0.8}
    return None


def _coerce_position_item(item, key: str = "position", **defaults) -> dict | None:
    if isinstance(item, dict):
        result = dict(defaults)
        result.update(item)
        result[key] = str(result.get(key) or result.get("location") or "center")
        return result
    if isinstance(item, str) and item.strip():
        result = dict(defaults)
        result[key] = item.strip()
        return result
    return None


def _coerce_position(value) -> str | None:
    if isinstance(value, dict):
        raw = value.get("position") or value.get("location") or value.get("region")
        return str(raw) if raw else None
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def normalize_generation_plan(plan: dict) -> dict:
    normalized = dict(safe_dict(plan))
    normalized["profile"] = safe_dict(normalized.get("profile"))
    normalized["topology_intent"] = safe_dict(normalized.get("topology_intent"))

    constraints = safe_dict(normalized.get("constraints"))
    continents = [_coerce_continent(item) for item in _as_list(normalized.get("continents"))]
    mountains = [_coerce_mountain(item) for item in _as_list(normalized.get("mountains"))]
    constraints_continents = [_coerce_continent(item) for item in _as_list(constraints.get("continents"))]
    constraints_mountains = [_coerce_mountain(item) for item in _as_list(constraints.get("mountains"))]

    normalized["continents"] = [item for item in continents if item]
    normalized["mountains"] = [item for item in mountains if item]
    normalized["peninsulas"] = [
        item for item in (
            _coerce_position_item(item, key="location", size=0.18)
            for item in _as_list(normalized.get("peninsulas"))
        ) if item
    ]
    normalized["island_chains"] = [
        item for item in (
            _coerce_position_item(item, key="position", density=0.66)
            for item in _as_list(normalized.get("island_chains"))
        ) if item
    ]
    normalized["inland_seas"] = [
        item for item in (
            _coerce_position_item(item, key="position", connection="enclosed")
            for item in _as_list(normalized.get("inland_seas"))
        ) if item
    ]
    normalized["water_bodies"] = [
        item for item in (
            _coerce_position_item(item, key="position", type="ocean", coverage=0.25, connection="")
            for item in _as_list(normalized.get("water_bodies"))
        ) if item
    ]
    normalized["regional_relations"] = [
        item for item in _as_list(normalized.get("regional_relations")) if isinstance(item, dict)
    ]

    sea_zones = [_coerce_position(item) for item in _as_list(constraints.get("sea_zones"))]
    river_sources = [_coerce_position(item) for item in _as_list(constraints.get("river_sources"))]
    constraints["continents"] = [item for item in constraints_continents if item]
    constraints["mountains"] = [item for item in constraints_mountains if item]
    constraints["sea_zones"] = [item for item in sea_zones if item]
    constraints["river_sources"] = [item for item in river_sources if item]
    normalized["constraints"] = constraints
    return normalized


def _generate_region_elevation(
    width: int, height: int, seed: int,
    region_slice: tuple[slice, slice],
    blend_width: int = 12,
) -> np.ndarray:
    h, w = height, width
    region_terrain = TerrainGenerator(width=w, height=h, seed=seed)
    region_elev = region_terrain.generate()
    full = np.zeros((h, w), dtype=np.float32)
    rs, cs = region_slice
    full[rs, cs] = region_elev[rs, cs]
    if blend_width > 0:
        row_start = rs.start or 0
        row_end = rs.stop or h
        col_start = cs.start or 0
        col_end = cs.stop or w
        for i in range(blend_width):
            alpha = i / blend_width
            if row_start > 0:
                r = row_start + i
                if r < h:
                    full[r, :] *= alpha
            if row_end < h:
                r = row_end - 1 - i
                if r >= 0:
                    full[r, :] *= alpha
            if col_start > 0:
                c = col_start + i
                if c < w:
                    full[:, c] *= alpha
            if col_end < w:
                c = col_end - 1 - i
                if c >= 0:
                    full[:, c] *= alpha
    return full


def _relax_periodic_longitude(values: np.ndarray, band_width: int, strength: float = 0.35) -> np.ndarray:
    from scipy.ndimage import gaussian_filter

    source = values.astype(np.float32)
    width = source.shape[1]
    band = min(max(2, int(band_width)), max(2, width // 8))
    if band * 2 >= width:
        return source

    padded = np.concatenate([source[:, -band:], source, source[:, :band]], axis=1)
    periodic = gaussian_filter(padded, sigma=(0.0, 1.0)).astype(np.float32)[:, band:-band]

    weights = np.zeros_like(source, dtype=np.float32)
    ramp = np.linspace(1.0, 0.0, band, dtype=np.float32)
    weights[:, :band] = ramp
    weights[:, -band:] = ramp[::-1]

    relaxed = source * (1.0 - weights * strength) + periodic * (weights * strength)
    seam = (relaxed[:, 0] + relaxed[:, -1]) * 0.5
    relaxed[:, 0] = seam
    relaxed[:, -1] = seam
    return relaxed.astype(np.float32)


def rebalance_and_scale_elevation(elevation: np.ndarray, land_ratio: float) -> np.ndarray:
    target_land = float(np.clip(land_ratio, 0.24, 0.72))
    sea_level = float(np.quantile(elevation, 1.0 - target_land))
    shifted = elevation.astype(np.float32) - sea_level

    result = np.zeros_like(shifted, dtype=np.float32)
    land = shifted > 0
    water = ~land
    if np.any(land):
        land_scale = max(float(np.percentile(shifted[land], 98)), 1e-6)
        result[land] = np.clip(shifted[land] / land_scale, 0.0, 1.0) * 0.78
    if np.any(water):
        water_scale = max(float(np.percentile(-shifted[water], 98)), 1e-6)
        result[water] = -np.clip((-shifted[water]) / water_scale, 0.0, 1.0) * 0.9

    result += np.tanh(shifted * 2.0).astype(np.float32) * 0.08
    return np.clip(result, -1.0, 1.0).astype(np.float32)


def planet_constraints_from_plan(plan: dict) -> dict:
    constraints = dict(plan.get("constraints") or {})
    constraints["continents"] = list(constraints.get("continents") or plan.get("continents") or [])
    constraints["mountains"] = list(constraints.get("mountains") or plan.get("mountains") or [])
    sea_zones = list(constraints.get("sea_zones") or [])
    for item in plan.get("inland_seas") or []:
        if isinstance(item, dict) and item.get("position"):
            sea_zones.append(str(item["position"]))
    for item in plan.get("water_bodies") or []:
        if isinstance(item, dict) and item.get("position"):
            sea_zones.append(str(item["position"]))
    constraints["sea_zones"] = list(dict.fromkeys(sea_zones))
    return constraints


def apply_planet_semantic_shape(elevation: np.ndarray, plan: dict, seed: int, blend: float = 0.88) -> np.ndarray:
    profile = plan.get("profile") or {}
    topology_intent = plan.get("topology_intent") or {}
    layout_template = str(profile.get("layout_template", "default"))
    constraints = plan.get("constraints") or {}
    has_semantic_shape = bool(
        topology_intent
        or constraints.get("continents")
        or constraints.get("sea_zones")
        or constraints.get("mountains")
        or plan.get("continents")
        or plan.get("water_bodies")
        or layout_template != "default"
    )
    if not has_semantic_shape:
        return elevation.astype(np.float32)

    h, w = elevation.shape
    terrain = TerrainGenerator(width=w, height=h, seed=seed)
    semantic_plan = dict(plan)
    semantic_plan["constraints"] = planet_constraints_from_plan(plan)
    semantic, _ = shape_world_profile(terrain, elevation, semantic_plan)
    semantic = _relax_periodic_longitude(semantic, band_width=max(4, w // 96), strength=0.22)
    mixed = elevation.astype(np.float32) * (1.0 - blend) + semantic.astype(np.float32) * blend
    return np.clip(mixed, -1.0, 1.0).astype(np.float32)


def gaussian_smooth(values: np.ndarray) -> np.ndarray:
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(values.astype(np.float32), sigma=1.2).astype(np.float32)


def _normalize_topology(field: np.ndarray) -> np.ndarray:
    normalized = np.tanh(field * 1.35).astype(np.float32)
    return gaussian_smooth(normalized)


def _intent_modifier(topology_intent: dict, key: str, default: str) -> str:
    modifiers = topology_intent.get("modifiers") or {}
    value = str(modifiers.get(key, default)).strip().lower()
    return value or default


def has_explicit_plan_geometry(plan: dict) -> bool:
    return any(
        plan.get(key)
        for key in ("continents", "mountains", "peninsulas", "island_chains", "inland_seas", "water_bodies", "regional_relations", "topology_intent")
    )


def _apply_mountain_topology(terrain: TerrainGenerator, field: np.ndarray, constraints: dict, ruggedness: float) -> np.ndarray:
    result = field.astype(np.float32)
    for mountain in constraints.get("mountains") or []:
        location = str(mountain.get("location", "center"))
        height = float(mountain.get("height", 0.8))
        chain = terrain._create_mountain_chain_mask(location)
        result += gaussian_smooth(chain * (0.18 + height * 0.24 + ruggedness * 0.08))
    return result


def _continent_component(
    terrain: TerrainGenerator,
    anchor: str,
    size: float,
    satellites: list[tuple[str, float, float]],
) -> np.ndarray:
    base = terrain._create_continent_mask(anchor, size)
    pieces = [base]
    for position, satellite_size, weight in satellites:
        pieces.append(terrain._create_continent_mask(position, satellite_size) * weight)
    return np.maximum.reduce(pieces)


def _longitudinal_gate(terrain: TerrainGenerator, side: str) -> np.ndarray:
    x = terrain._x_norm
    if side == "west":
        return np.clip(1.0 - ((x - 0.34) / 0.32) ** 2, 0.0, 1.0).astype(np.float32)
    return np.clip(1.0 - ((x - 0.66) / 0.32) ** 2, 0.0, 1.0).astype(np.float32)


def _axis_aligned_ocean_barrier(terrain: TerrainGenerator, axis: str, center: float, width: float) -> np.ndarray:
    if axis == "vertical":
        x = terrain._x_norm
        barrier = np.exp(-((x - center) ** 2) / max(width**2, 1e-4))
        return barrier.astype(np.float32)
    y = terrain._y_norm
    barrier = np.exp(-((y - center) ** 2) / max(width**2, 1e-4))
    return barrier.astype(np.float32)


def _separation_channel(terrain: TerrainGenerator, subject: str, object_name: str, strength: float) -> np.ndarray:
    sy, sx = terrain._resolve_position(subject)
    oy, ox = terrain._resolve_position(object_name)
    cy = float(np.clip((sy + oy) * 0.5, 0.08, 0.92))
    cx = float(np.clip((sx + ox) * 0.5, 0.04, 0.96))
    dy = oy - sy
    dx = ox - sx
    rotation = float(np.arctan2(dy, dx)) + np.pi / 2.0
    distance = min(np.hypot(dx, dy), 0.75)
    length = 0.18 + distance * 0.78
    width = 0.06 + strength * 0.05
    trench = terrain._elliptic_gaussian(cy, cx, width, length, rotation)
    if abs(dx) >= abs(dy) * 1.2:
        trench = np.maximum(trench, _axis_aligned_ocean_barrier(terrain, "vertical", center=cx, width=width * 1.45))
    elif abs(dy) >= abs(dx) * 1.2:
        trench = np.maximum(trench, _axis_aligned_ocean_barrier(terrain, "horizontal", center=cy, width=width * 1.45))
    center_cut = terrain._create_location_mask("center", radius=0.12 + strength * 0.04, sigma=0.28 + strength * 0.2)
    return gaussian_smooth(np.maximum(trench, center_cut * 0.82) * (0.58 + strength * 0.34))


def _apply_regional_relations(terrain: TerrainGenerator, field: np.ndarray, relations: list[dict]) -> np.ndarray:
    if not relations:
        return field

    shaped = field.astype(np.float32).copy()
    for relation in relations:
        relation_type = str(relation.get("relation", "")).lower()
        subject = str(relation.get("subject", "center"))
        object_name = str(relation.get("object", "center"))
        strength = float(np.clip(relation.get("strength", 1.0), 0.1, 1.2))

        if relation_type == "separated_by_water":
            shaped -= _separation_channel(terrain, subject, object_name, strength)
        elif relation_type == "elevated_region":
            ridge = terrain._create_mountain_chain_mask(subject)
            shaped += gaussian_smooth(ridge * (0.12 + strength * 0.16))
    return shaped


def _plan_continent_component(terrain: TerrainGenerator, position: str, size: float) -> np.ndarray:
    size = float(np.clip(size, 0.18, 0.72))
    base = terrain._create_continent_mask(position, size)
    satellites = [_plan_satellite_mask(terrain, sat_position, size, weight) for sat_position, weight in _satellite_positions(position)]
    coastline_noise = terrain._fbm(scale=58.0, octaves=3, persistence=0.56, lacunarity=2.05, offset=len(position) * 29.0)
    combined = np.maximum.reduce([base, *satellites]) if satellites else base
    combined = np.clip(combined * (0.86 + coastline_noise * 0.28), 0.0, 1.0)
    return gaussian_smooth(combined)


def _plan_satellite_mask(terrain: TerrainGenerator, position: str, size: float, weight: float) -> np.ndarray:
    return terrain._create_continent_mask(position, max(0.12, size * weight)) * (0.55 + weight * 0.35)


def _satellite_positions(position: str) -> list[tuple[str, float]]:
    mapping = {
        "west": [("northwest", 0.46), ("southwest", 0.5), ("center", 0.24)],
        "east": [("northeast", 0.46), ("southeast", 0.5), ("center", 0.24)],
        "north": [("northwest", 0.42), ("northeast", 0.42), ("center", 0.22)],
        "south": [("southwest", 0.42), ("southeast", 0.42), ("center", 0.22)],
        "northwest": [("west", 0.24), ("north", 0.24)],
        "northeast": [("east", 0.24), ("north", 0.24)],
        "southwest": [("west", 0.24), ("south", 0.24)],
        "southeast": [("east", 0.24), ("south", 0.24)],
        "center": [("west", 0.22), ("east", 0.22), ("south", 0.18)],
    }
    return mapping.get(position, [])


def _plan_peninsula_component(terrain: TerrainGenerator, location: str, size: float) -> np.ndarray:
    cy, cx = terrain._resolve_position(location)
    rotation = terrain._position_rotation(location)
    stem = terrain._elliptic_gaussian(cy, cx, max(0.12, size * 0.9), max(0.05, size * 0.42), rotation)
    head = terrain._elliptic_gaussian(
        np.clip(cy + np.sin(rotation) * size * 0.38, 0.06, 0.94),
        np.clip(cx + np.cos(rotation) * size * 0.38, 0.04, 0.96),
        max(0.08, size * 0.66),
        max(0.04, size * 0.34),
        rotation + 0.18,
    )
    return gaussian_smooth(np.maximum(stem, head * 0.92))


def _plan_island_chain_component(terrain: TerrainGenerator, position: str, density: float) -> np.ndarray:
    cy, cx = terrain._resolve_position(position)
    orientation = terrain._position_rotation(position)
    chain = np.zeros((terrain.height, terrain.width), dtype=np.float32)
    count = int(np.clip(round(3 + density * 4), 3, 7))
    for index in range(count):
        offset = (index - (count - 1) / 2.0) * 0.075
        iy = np.clip(cy + np.sin(orientation) * offset, 0.08, 0.92)
        ix = np.clip(cx + np.cos(orientation) * offset, 0.04, 0.96)
        island = terrain._elliptic_gaussian(iy, ix, 0.06 + density * 0.025, 0.038 + density * 0.016, orientation + (index % 2) * 0.35)
        chain = np.maximum(chain, island.astype(np.float32))
    return gaussian_smooth(chain)


def _plan_inland_sea_component(terrain: TerrainGenerator, position: str, connection: str) -> np.ndarray:
    cy, cx = terrain._resolve_position(position)
    basin = np.maximum(
        terrain._elliptic_gaussian(cy, cx, 0.16, 0.12, 0.0),
        terrain._elliptic_gaussian(cy, cx, 0.11, 0.2, 0.0) * 0.84,
    )
    basin = np.maximum(basin, terrain._elliptic_gaussian(cy, cx, 0.22, 0.3, 0.0) * 0.9)
    connection = (connection or "").lower()
    if "strait" in connection:
        basin = np.maximum(
            basin,
            terrain._elliptic_gaussian(cy, np.clip(cx + 0.18, 0.04, 0.96), 0.06, 0.045, 0.0) * 0.88,
        )
    elif "east" in connection:
        basin = np.maximum(
            basin,
            terrain._elliptic_gaussian(cy, np.clip(cx + 0.22, 0.04, 0.96), 0.08, 0.06, 0.0) * 0.84,
        )
    elif "west" in connection:
        basin = np.maximum(
            basin,
            terrain._elliptic_gaussian(cy, np.clip(cx - 0.22, 0.04, 0.96), 0.08, 0.06, 0.0) * 0.84,
        )
    return gaussian_smooth(basin)


def _plan_water_body_component(
    terrain: TerrainGenerator,
    body_type: str,
    position: str,
    coverage: float,
    connection: str,
) -> np.ndarray:
    normalized_type = (body_type or "ocean").lower()
    coverage = float(np.clip(coverage, 0.08, 0.68))
    radius = 0.12 + coverage * 0.22
    sigma = 0.28 + coverage * 0.62
    mask = terrain._create_location_mask(position, radius=radius, sigma=sigma)

    if "strait" in normalized_type:
        rotation = terrain._position_rotation(position)
        cy, cx = terrain._resolve_position(position)
        channel = terrain._elliptic_gaussian(cy, cx, 0.08 + coverage * 0.12, 0.24 + coverage * 0.18, rotation)
        return gaussian_smooth(np.maximum(mask * 0.8, channel) * (0.75 + coverage))

    if "inland" in normalized_type:
        return _plan_inland_sea_component(terrain, position, connection or "strait") * (0.72 + coverage)

    if "ocean" in normalized_type or "sea" in normalized_type:
        side_masks = [mask]
        if position in {"west", "east", "north", "south"}:
            side_masks.append(terrain._create_continent_mask(position, min(0.32 + coverage * 0.2, 0.58)))
        return gaussian_smooth(np.maximum.reduce(side_masks) * (0.72 + coverage))

    return gaussian_smooth(mask * (0.68 + coverage))


def _force_water_mask(
    elevation: np.ndarray,
    mask: np.ndarray,
    base_depth: float,
    mask_threshold: float,
) -> np.ndarray:
    if not np.any(mask > 0.02):
        return elevation

    shaped = elevation.astype(np.float32).copy()
    blend = np.clip(mask * 0.94, 0.0, 0.97)
    target = (base_depth - mask * 0.5).astype(np.float32)
    shaped = shaped * (1.0 - blend) + target * blend
    forced = mask >= mask_threshold
    shaped[forced] = np.minimum(shaped[forced], target[forced])
    return shaped


def _requires_opposite_side_split(relations: list[dict], a: str, b: str) -> bool:
    for relation in relations:
        if str(relation.get("relation", "")).lower() != "separated_by_water":
            continue
        subject = str(relation.get("subject", "")).lower()
        object_name = str(relation.get("object", "")).lower()
        if {subject, object_name} == {a, b}:
            return True
    return False


def _label_land_components(mask: np.ndarray) -> tuple[np.ndarray, dict[int, int]]:
    from scipy.ndimage import label
    labels, count = label(mask.astype(np.int8))
    component_sizes = {component_id: int(np.sum(labels == component_id)) for component_id in range(1, count + 1)}
    return labels.astype(np.int32), component_sizes


def _label_water_components(mask: np.ndarray) -> tuple[np.ndarray, dict[int, int]]:
    from scipy.ndimage import label
    labels, count = label(mask.astype(np.int8))
    component_sizes = {component_id: int(np.sum(labels == component_id)) for component_id in range(1, count + 1)}
    return labels.astype(np.int32), component_sizes


def _count_components_from_mask(mask: np.ndarray, min_cells: int) -> int:
    from scipy.ndimage import label
    labels, count = label(mask.astype(np.int8))
    kept = 0
    for component_id in range(1, count + 1):
        if int(np.sum(labels == component_id)) >= min_cells:
            kept += 1
    return kept


def _meandering_channel_mask(
    terrain: TerrainGenerator,
    axis: str,
    center: float,
    width: float,
    waviness: float,
    skew: float = 0.0,
) -> np.ndarray:
    path_noise = terrain._fbm(scale=92.0, octaves=3, persistence=0.56, lacunarity=2.0, offset=417.0)
    width_noise = terrain._fbm(scale=74.0, octaves=3, persistence=0.52, lacunarity=2.15, offset=523.0)
    if axis == "vertical":
        centerline = center + (np.mean(path_noise, axis=1, keepdims=True) * 2.0 - 1.0) * waviness
        row_wave = np.sin((terrain._y_norm[:, :1] * np.pi * 2.3) + 0.45) * (waviness * 0.34)
        centerline = centerline + row_wave + (terrain._y_norm[:, :1] - 0.5) * skew
        width_map = width * (0.82 + np.mean(width_noise, axis=1, keepdims=True) * 0.52)
        distance = np.abs(terrain._x_norm - centerline)
    else:
        centerline = center + (np.mean(path_noise, axis=0, keepdims=True) * 2.0 - 1.0) * waviness
        col_wave = np.sin((terrain._x_norm[:1, :] * np.pi * 2.1) + 0.3) * (waviness * 0.34)
        centerline = centerline + col_wave + (terrain._x_norm[:1, :] - 0.5) * skew
        width_map = width * (0.82 + np.mean(width_noise, axis=0, keepdims=True) * 0.52)
        distance = np.abs(terrain._y_norm - centerline)
    channel = np.exp(-(distance**2) / np.maximum(width_map**2, 1e-4))
    return gaussian_smooth(channel.astype(np.float32))


def _natural_strait_connector(terrain: TerrainGenerator, axis: str) -> np.ndarray:
    if axis == "vertical":
        return _meandering_channel_mask(terrain, axis="vertical", center=0.49, width=0.042, waviness=0.024, skew=0.028)
    return _meandering_channel_mask(terrain, axis="horizontal", center=0.51, width=0.044, waviness=0.02, skew=-0.024)


def _asymmetry_field(terrain: TerrainGenerator, axis: str) -> np.ndarray:
    if axis == "vertical":
        field = (terrain._x_norm - 0.5) * 1.15 + (terrain._y_norm - 0.5) * 0.55
    else:
        field = (terrain._y_norm - 0.5) * 1.15 + (terrain._x_norm - 0.5) * 0.55
    low_freq = terrain._fbm(scale=128.0, octaves=2, persistence=0.62, lacunarity=1.95, offset=733.0) * 2.0 - 1.0
    return np.clip(field + low_freq * 0.35, -1.0, 1.0).astype(np.float32)


def _naturalize_water_mask(
    terrain: TerrainGenerator,
    mask: np.ndarray,
    amplitude: float,
    smooth_passes: int,
    asymmetry: float = 0.0,
    axis: str = "vertical",
) -> np.ndarray:
    from scipy.ndimage import map_coordinates

    h, w = mask.shape
    warp_strength = amplitude * 0.14
    warp_y = terrain._fbm(scale=62.0, octaves=4, persistence=0.52, lacunarity=2.1, offset=881.0) * 2.0 - 1.0
    warp_x = terrain._fbm(scale=58.0, octaves=4, persistence=0.52, lacunarity=2.1, offset=997.0) * 2.0 - 1.0

    y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)
    y_warped = y_coords + warp_y * warp_strength * h
    x_warped = x_coords + warp_x * warp_strength * w

    warped_mask = map_coordinates(mask, [y_warped, x_warped], order=1, mode="reflect")

    coast_noise = terrain._fbm(scale=46.0, octaves=4, persistence=0.55, lacunarity=2.15, offset=611.0) * 2.0 - 1.0
    macro_noise = terrain._fbm(scale=96.0, octaves=3, persistence=0.58, lacunarity=2.0, offset=677.0) * 2.0 - 1.0
    fine_noise = terrain._fbm(scale=22.0, octaves=3, persistence=0.48, lacunarity=2.3, offset=653.0) * 2.0 - 1.0

    edge_band = np.clip(warped_mask * (1.0 - warped_mask) * 4.6, 0.0, 1.0)

    disturbed = warped_mask + edge_band * (
        coast_noise * amplitude
        + macro_noise * (amplitude * 0.55)
        + fine_noise * (amplitude * 0.3)
    )

    directional = _asymmetry_field(terrain, axis=axis)
    disturbed += edge_band * directional * asymmetry

    embayments = np.clip(warped_mask - 0.52, 0.0, 1.0) * np.clip(-coast_noise, 0.0, 1.0) * (amplitude * 0.22)
    disturbed = np.clip(disturbed + embayments * (1.0 + directional * 0.45), 0.0, 1.0)

    for _ in range(max(1, smooth_passes)):
        disturbed = gaussian_smooth(disturbed)

    return np.clip(disturbed, 0.0, 1.0).astype(np.float32)


def _natural_split_barrier(terrain: TerrainGenerator, axis: str, width_scale: float = 1.0) -> np.ndarray:
    channel = _meandering_channel_mask(
        terrain,
        axis=axis,
        center=0.495,
        width=0.05 * width_scale,
        waviness=0.04,
        skew=0.05,
    )
    connector = _natural_strait_connector(terrain, axis=axis)
    mask = np.maximum(channel, connector * 0.9)
    if axis == "vertical":
        asym_lobes = np.maximum.reduce(
            [
                terrain._elliptic_gaussian(0.32, 0.47, 0.09, 0.045, -0.2) * 0.68,
                terrain._elliptic_gaussian(0.64, 0.53, 0.12, 0.05, 0.24) * 0.82,
                terrain._elliptic_gaussian(0.84, 0.49, 0.07, 0.038, -0.1) * 0.52,
            ]
        )
    else:
        asym_lobes = np.maximum.reduce(
            [
                terrain._elliptic_gaussian(0.47, 0.3, 0.045, 0.09, 0.18) * 0.68,
                terrain._elliptic_gaussian(0.54, 0.66, 0.05, 0.12, -0.22) * 0.82,
                terrain._elliptic_gaussian(0.49, 0.85, 0.038, 0.07, 0.08) * 0.5,
            ]
        )
    mask = np.maximum(mask, asym_lobes)
    return _naturalize_water_mask(terrain, mask, amplitude=0.23, smooth_passes=2, asymmetry=0.14, axis=axis)


def _noise_driven_basin(
    terrain: TerrainGenerator,
    center_y: float,
    center_x: float,
    base_radius: float,
    irregularity: float = 0.5,
    aspect_bias: float = 0.0,
    seed_offset: float = 0.0,
) -> np.ndarray:
    y = terrain._y_norm
    x = terrain._x_norm

    warp_x = terrain._fbm(scale=48.0, octaves=4, persistence=0.52, lacunarity=2.1, offset=1401.0 + seed_offset) * 2.0 - 1.0
    warp_y = terrain._fbm(scale=52.0, octaves=4, persistence=0.52, lacunarity=2.1, offset=1409.0 + seed_offset) * 2.0 - 1.0
    radius_noise = terrain._fbm(scale=36.0, octaves=5, persistence=0.54, lacunarity=2.15, offset=1417.0 + seed_offset) * 2.0 - 1.0

    warp_strength = irregularity * 0.12
    warped_x = x + warp_x * warp_strength
    warped_y = y + warp_y * warp_strength

    dx = warped_x - center_x
    dy = warped_y - center_y

    radius_mod = base_radius * (1.0 + radius_noise * irregularity * 0.45)
    radius_mod = np.maximum(radius_mod, 0.01)

    if abs(aspect_bias) > 0.01:
        cos_a = np.cos(aspect_bias)
        sin_a = np.sin(aspect_bias)
        dx_rot = dx * cos_a - dy * sin_a
        dy_rot = dx * sin_a + dy * cos_a
        stretch = 1.0 + 0.3 * irregularity
        distance = np.sqrt((dx_rot / radius_mod) ** 2 + (dy_rot / (radius_mod * stretch)) ** 2)
    else:
        distance = np.sqrt((dx / radius_mod) ** 2 + (dy / radius_mod) ** 2)

    field = np.exp(-(distance ** 2) / 2.0)
    return np.clip(field, 0.0, 1.0).astype(np.float32)


def _natural_inland_sea_basin(terrain: TerrainGenerator, basin_shape: str = "balanced", basin_style: str = "balanced") -> np.ndarray:
    rng = terrain._rng if hasattr(terrain, '_rng') else np.random.RandomState(42)
    jitter_y = lambda: (rng.random() - 0.5) * 0.06
    jitter_x = lambda: (rng.random() - 0.5) * 0.08
    jitter_angle = lambda: (rng.random() - 0.5) * 0.6

    if basin_shape == "compact":
        base_radius = 0.10
        aspect = jitter_angle()
        basin = _noise_driven_basin(terrain, 0.50 + jitter_y(), 0.50 + jitter_x(), base_radius, irregularity=0.55, aspect_bias=aspect, seed_offset=0.0)
        lobe1 = _noise_driven_basin(terrain, 0.46 + jitter_y(), 0.44 + jitter_x(), 0.06, irregularity=0.6, aspect_bias=aspect + 0.8, seed_offset=11.0) * 0.72
        lobe2 = _noise_driven_basin(terrain, 0.54 + jitter_y(), 0.58 + jitter_x(), 0.055, irregularity=0.6, aspect_bias=aspect - 0.6, seed_offset=23.0) * 0.64
        cove1 = _noise_driven_basin(terrain, 0.42 + jitter_y(), 0.34 + jitter_x(), 0.04, irregularity=0.7, aspect_bias=-1.1, seed_offset=37.0) * 0.42
        cove2 = _noise_driven_basin(terrain, 0.58 + jitter_y(), 0.66 + jitter_x(), 0.035, irregularity=0.7, aspect_bias=0.7, seed_offset=41.0) * 0.36
        mask = np.maximum.reduce([basin, lobe1, lobe2, cove1, cove2])
    elif basin_shape == "broad":
        base_radius = 0.18
        aspect = jitter_angle()
        basin = _noise_driven_basin(terrain, 0.50 + jitter_y(), 0.50 + jitter_x(), base_radius, irregularity=0.55, aspect_bias=aspect, seed_offset=0.0)
        lobe1 = _noise_driven_basin(terrain, 0.46 + jitter_y(), 0.38 + jitter_x(), 0.10, irregularity=0.6, aspect_bias=aspect + 0.5, seed_offset=13.0) * 0.82
        lobe2 = _noise_driven_basin(terrain, 0.55 + jitter_y(), 0.64 + jitter_x(), 0.09, irregularity=0.6, aspect_bias=aspect - 0.4, seed_offset=19.0) * 0.74
        arm1 = _noise_driven_basin(terrain, 0.38 + jitter_y(), 0.30 + jitter_x(), 0.06, irregularity=0.7, aspect_bias=-1.2, seed_offset=31.0) * 0.62
        arm2 = _noise_driven_basin(terrain, 0.62 + jitter_y(), 0.72 + jitter_x(), 0.055, irregularity=0.7, aspect_bias=0.9, seed_offset=37.0) * 0.56
        arm3 = _noise_driven_basin(terrain, 0.44 + jitter_y(), 0.76 + jitter_x(), 0.045, irregularity=0.7, aspect_bias=0.5, seed_offset=43.0) * 0.48
        mask = np.maximum.reduce([basin, lobe1, lobe2, arm1, arm2, arm3])
    elif basin_shape == "branched":
        base_radius = 0.12
        aspect = jitter_angle()
        basin = _noise_driven_basin(terrain, 0.50 + jitter_y(), 0.50 + jitter_x(), base_radius, irregularity=0.55, aspect_bias=aspect, seed_offset=0.0)
        branch1 = _noise_driven_basin(terrain, 0.36 + jitter_y(), 0.34 + jitter_x(), 0.07, irregularity=0.65, aspect_bias=-1.3, seed_offset=17.0) * 0.78
        branch2 = _noise_driven_basin(terrain, 0.63 + jitter_y(), 0.69 + jitter_x(), 0.06, irregularity=0.65, aspect_bias=0.9, seed_offset=29.0) * 0.68
        branch3 = _noise_driven_basin(terrain, 0.44 + jitter_y(), 0.77 + jitter_x(), 0.05, irregularity=0.7, aspect_bias=0.6, seed_offset=37.0) * 0.58
        branch4 = _noise_driven_basin(terrain, 0.57 + jitter_y(), 0.23 + jitter_x(), 0.055, irregularity=0.7, aspect_bias=-0.8, seed_offset=43.0) * 0.52
        branch5 = _noise_driven_basin(terrain, 0.52 + jitter_y(), 0.61 + jitter_x(), 0.04, irregularity=0.7, aspect_bias=0.3, seed_offset=53.0) * 0.48
        mask = np.maximum.reduce([basin, branch1, branch2, branch3, branch4, branch5])
    else:
        base_radius = 0.14
        aspect = jitter_angle()
        basin = _noise_driven_basin(terrain, 0.50 + jitter_y(), 0.50 + jitter_x(), base_radius, irregularity=0.55, aspect_bias=aspect, seed_offset=0.0)
        lobe1 = _noise_driven_basin(terrain, 0.46 + jitter_y(), 0.40 + jitter_x(), 0.08, irregularity=0.6, aspect_bias=aspect + 0.6, seed_offset=11.0) * 0.80
        lobe2 = _noise_driven_basin(terrain, 0.55 + jitter_y(), 0.62 + jitter_x(), 0.075, irregularity=0.6, aspect_bias=aspect - 0.5, seed_offset=19.0) * 0.70
        arm1 = _noise_driven_basin(terrain, 0.39 + jitter_y(), 0.32 + jitter_x(), 0.055, irregularity=0.7, aspect_bias=-1.2, seed_offset=31.0) * 0.58
        arm2 = _noise_driven_basin(terrain, 0.61 + jitter_y(), 0.68 + jitter_x(), 0.05, irregularity=0.7, aspect_bias=0.8, seed_offset=37.0) * 0.50
        arm3 = _noise_driven_basin(terrain, 0.47 + jitter_y(), 0.74 + jitter_x(), 0.04, irregularity=0.7, aspect_bias=0.6, seed_offset=43.0) * 0.44
        mask = np.maximum.reduce([basin, lobe1, lobe2, arm1, arm2, arm3])

    if basin_style == "rift":
        rift1 = _noise_driven_basin(terrain, 0.49 + jitter_y(), 0.47 + jitter_x(), 0.06, irregularity=0.6, aspect_bias=-0.15, seed_offset=61.0) * 0.86
        rift2 = _noise_driven_basin(terrain, 0.53 + jitter_y(), 0.56 + jitter_x(), 0.055, irregularity=0.6, aspect_bias=0.2, seed_offset=67.0) * 0.78
        rift3 = _noise_driven_basin(terrain, 0.45 + jitter_y(), 0.37 + jitter_x(), 0.045, irregularity=0.65, aspect_bias=-0.35, seed_offset=71.0) * 0.62
        mask = np.maximum(mask * 0.9, np.maximum.reduce([rift1, rift2, rift3]))
    elif basin_style == "mediterranean":
        west_bay = _noise_driven_basin(terrain, 0.50 + jitter_y(), 0.30 + jitter_x(), 0.065, irregularity=0.6, aspect_bias=-0.2, seed_offset=79.0) * 0.72
        east_bay = _noise_driven_basin(terrain, 0.50 + jitter_y(), 0.70 + jitter_x(), 0.06, irregularity=0.6, aspect_bias=0.25, seed_offset=83.0) * 0.68
        mask = np.maximum(mask, np.maximum(west_bay, east_bay))

    return _naturalize_water_mask(terrain, mask, amplitude=0.45, smooth_passes=1, asymmetry=0.25, axis="vertical")


def _build_water_enforcement_mask(terrain: TerrainGenerator, plan: dict) -> np.ndarray:
    mask = np.zeros((terrain.height, terrain.width), dtype=np.float32)
    for inland_sea in plan.get("inland_seas") or []:
        mask = np.maximum(
            mask,
            _plan_inland_sea_component(
                terrain,
                str(inland_sea.get("position", "center")),
                str(inland_sea.get("connection", "strait")),
            ),
        )

    for water_body in plan.get("water_bodies") or []:
        mask = np.maximum(
            mask,
            _plan_water_body_component(
                terrain,
                body_type=str(water_body.get("type", "ocean")),
                position=str(water_body.get("position", "center")),
                coverage=float(water_body.get("coverage", 0.25)),
                connection=str(water_body.get("connection", "")),
            ),
        )

    for relation in plan.get("regional_relations") or []:
        if str(relation.get("relation", "")).lower() == "separated_by_water":
            mask = np.maximum(
                mask,
                _separation_channel(
                    terrain,
                    str(relation.get("subject", "west")),
                    str(relation.get("object", "east")),
                    float(np.clip(relation.get("strength", 1.0), 0.1, 1.2)),
                ),
            )
    return gaussian_smooth(mask)


def apply_water_enforcement(terrain: TerrainGenerator, elevation: np.ndarray, plan: dict) -> np.ndarray:
    mask = _build_water_enforcement_mask(terrain, plan)
    if not np.any(mask > 0.02):
        return elevation

    shaped = elevation.astype(np.float32).copy()
    blend = np.clip(mask * 0.92, 0.0, 0.94)
    water_target = (-0.42 - mask * 0.72).astype(np.float32)
    shaped = shaped * (1.0 - blend) + water_target * blend
    forced = mask >= 0.42
    shaped[forced] = np.minimum(shaped[forced], (-0.22 - mask[forced] * 0.58).astype(np.float32))
    return gaussian_smooth(shaped)


def enforce_required_water_gaps(terrain: TerrainGenerator, elevation: np.ndarray, plan: dict) -> np.ndarray:
    profile = plan.get("profile") or {}
    constraints = plan.get("constraints") or {}
    continents = list(plan.get("continents") or constraints.get("continents") or [])
    sea_zones = list(constraints.get("sea_zones") or [])
    relations = list(plan.get("regional_relations") or [])
    shaped = elevation.astype(np.float32).copy()

    positions = {str(item.get("position", "")) for item in continents}
    layout_template = str(profile.get("layout_template", "default"))
    sea_style = str(profile.get("sea_style", "open"))

    split_barrier = np.zeros_like(shaped)
    if layout_template == "split_east_west" or _requires_opposite_side_split(relations, "west", "east"):
        split_barrier = np.maximum(split_barrier, _natural_split_barrier(terrain, axis="vertical"))
    elif layout_template == "split_north_south" or _requires_opposite_side_split(relations, "north", "south"):
        split_barrier = np.maximum(split_barrier, _natural_split_barrier(terrain, axis="horizontal"))

    if "center" in sea_zones and {"west", "east"} <= positions:
        split_barrier = np.maximum(
            split_barrier,
            _natural_split_barrier(terrain, axis="vertical"),
        )
    if "center" in sea_zones and {"north", "south"} <= positions and sea_style != "inland":
        split_barrier = np.maximum(
            split_barrier,
            _natural_split_barrier(terrain, axis="horizontal"),
        )

    inland_basin = np.zeros_like(shaped)
    if sea_style == "inland" or layout_template == "mediterranean":
        inland_basin = np.maximum(inland_basin, _natural_inland_sea_basin(terrain))
        if {"north", "south"} <= positions:
            inland_basin = np.maximum(
                inland_basin,
                terrain._elliptic_gaussian(0.5, 0.5, 0.11, 0.26, 0.04) * 0.72,
            )

    shaped = _force_water_mask(shaped, split_barrier, base_depth=-0.32, mask_threshold=0.62)
    shaped = _force_water_mask(shaped, inland_basin, base_depth=-0.28, mask_threshold=0.5)
    return shaped


def apply_constraint_topology(
    terrain: TerrainGenerator,
    elevation: np.ndarray,
    constraints: dict,
    coast_complexity: float,
    ruggedness: float,
) -> np.ndarray:
    continents = constraints.get("continents") or []
    sea_zones = constraints.get("sea_zones") or []
    mountains = constraints.get("mountains") or []
    if not continents and not sea_zones and not mountains:
        return elevation

    land_masks = []
    for continent in continents:
        position = str(continent.get("position", "center"))
        size = float(continent.get("size", 0.45))
        land_masks.append(terrain._create_continent_mask(position, size))

    result = elevation.astype(np.float32).copy()
    if land_masks:
        land = np.maximum.reduce(land_masks)
        coastal_noise = terrain._fbm(scale=56.0, octaves=3, persistence=0.55, lacunarity=2.1, offset=301.0)
        land_shape = np.clip(land * (0.92 + coastal_noise * (0.18 + coast_complexity * 0.12)), 0.0, 1.0)
        land_shape = terrain._shoreline_profile(land_shape, water_level=0.43, gain=1.7)
        result = result * 0.34 + (land_shape * 2.0 - 1.0) * 0.66

    for zone in sea_zones:
        sea_mask = terrain._create_location_mask(str(zone), radius=0.24, sigma=0.6)
        result -= gaussian_smooth(sea_mask * 0.46)

    for mountain in mountains:
        location = str(mountain.get("location", "center"))
        height = float(mountain.get("height", 0.8))
        chain = terrain._create_mountain_chain_mask(location)
        result += gaussian_smooth(chain * (0.12 + height * 0.18 + ruggedness * 0.06))

    return np.clip(result, -1.0, 1.0).astype(np.float32)


def apply_layout_template(
    terrain: TerrainGenerator,
    elevation: np.ndarray,
    layout_template: str,
    sea_style: str,
    constraints: dict,
    coast_complexity: float,
) -> np.ndarray:
    result = elevation.astype(np.float32).copy()

    if layout_template == "split_east_west":
        west = np.maximum(
            terrain._create_continent_mask("west", 0.43),
            terrain._create_continent_mask("southwest", 0.2) * 0.78,
        )
        east = np.maximum(
            terrain._create_continent_mask("east", 0.43),
            terrain._create_continent_mask("northeast", 0.22) * 0.8,
        )
        central_gap = np.maximum(
            terrain._create_location_mask("center", radius=0.15, sigma=0.34),
            terrain._create_location_mask("center", radius=0.22, sigma=0.52) * 0.84,
        )
        vertical_ocean = np.clip(
            terrain._create_location_mask("north", radius=0.18, sigma=0.86)
            + terrain._create_location_mask("center", radius=0.2, sigma=0.92)
            + terrain._create_location_mask("south", radius=0.18, sigma=0.86),
            0.0,
            1.4,
        )
        sea = central_gap * 0.78 + vertical_ocean * 0.62
        structure = np.maximum(west, east) * 1.18 - sea * 1.14
        result = result * 0.24 + (structure * 2.0 - 1.0) * 0.76
    elif layout_template == "split_north_south":
        north = np.maximum(
            terrain._create_continent_mask("north", 0.4),
            terrain._create_continent_mask("northwest", 0.22) * 0.76,
        )
        south = np.maximum(
            terrain._create_continent_mask("south", 0.4),
            terrain._create_continent_mask("southeast", 0.22) * 0.76,
        )
        sea = np.maximum(
            terrain._create_location_mask("center", radius=0.18, sigma=0.42),
            terrain._create_location_mask("center", radius=0.28, sigma=0.82) * 0.68,
        )
        structure = np.maximum(north, south) * 1.16 - sea * 0.98
        result = result * 0.24 + (structure * 2.0 - 1.0) * 0.76
    elif layout_template == "mediterranean":
        north_rim = np.maximum(
            terrain._create_continent_mask("northwest", 0.34),
            terrain._create_continent_mask("northeast", 0.34),
        )
        south_rim = np.maximum(
            terrain._create_continent_mask("southwest", 0.34),
            terrain._create_continent_mask("southeast", 0.34),
        )
        west_gate = terrain._create_continent_mask("west", 0.24)
        east_gate = terrain._create_continent_mask("east", 0.24)
        middle_sea = np.maximum(
            terrain._create_location_mask("center", radius=0.18, sigma=0.38),
            terrain._create_location_mask("center", radius=0.28, sigma=0.66) * 0.72,
        )
        west_strait = terrain._create_location_mask("west", radius=0.11, sigma=0.24)
        east_strait = terrain._create_location_mask("east", radius=0.11, sigma=0.24)
        rim = np.maximum.reduce([north_rim, south_rim, west_gate * 0.72, east_gate * 0.72])
        enclosed_sea = middle_sea * 0.96 - (west_strait + east_strait) * 0.22
        structure = rim * 1.16 - enclosed_sea * 0.94
        result = result * 0.22 + (structure * 2.0 - 1.0) * 0.78
    elif layout_template == "single_island":
        core = terrain._create_continent_mask("center", 0.42)
        west_lobe = terrain._create_continent_mask("west", 0.14) * 0.68
        east_lobe = terrain._create_continent_mask("east", 0.16) * 0.72
        north_lobe = terrain._create_continent_mask("north", 0.13) * 0.58
        south_lobe = terrain._create_continent_mask("south", 0.14) * 0.64
        west_bay = terrain._create_location_mask("west", radius=0.18, sigma=0.46)
        east_bay = terrain._create_location_mask("east", radius=0.16, sigma=0.44)
        north_bay = terrain._create_location_mask("north", radius=0.14, sigma=0.4)
        south_bay = terrain._create_location_mask("south", radius=0.15, sigma=0.42)
        bite_noise = terrain._fbm(scale=44.0, octaves=4, persistence=0.56, lacunarity=2.1, offset=211.0)
        shape = np.maximum.reduce([core, west_lobe, east_lobe, north_lobe, south_lobe])
        carved = (west_bay * 0.14) + (east_bay * 0.16) + (north_bay * 0.1) + (south_bay * 0.12)
        irregular = np.clip(shape * (0.9 + bite_noise * 0.34) - carved, 0.0, 1.0)
        structure = terrain._shoreline_profile(irregular, water_level=0.42, gain=1.8)
        result = result * 0.18 + (structure * 2.0 - 1.0) * 0.82
    elif layout_template == "supercontinent":
        core = terrain._create_continent_mask("center", 0.68)
        shoulder_w = terrain._create_continent_mask("west", 0.26)
        shoulder_e = terrain._create_continent_mask("east", 0.26)
        structure = np.maximum.reduce([core, shoulder_w * 0.72, shoulder_e * 0.72])
        result = result * 0.26 + (structure * 2.0 - 1.0) * 0.74
    elif layout_template == "archipelago":
        centers = ["west", "east", "northwest", "northeast", "southwest", "southeast"]
        islands = [terrain._create_continent_mask(position, 0.16 + (index % 2) * 0.05) for index, position in enumerate(centers)]
        structure = np.maximum.reduce(islands)
        result = result * 0.18 + (structure * 2.0 - 1.0) * (0.5 + coast_complexity * 0.18)

    if (constraints.get("sea_zones") or []) and layout_template in {"split_east_west", "split_north_south", "mediterranean"}:
        for zone in constraints.get("sea_zones", []):
            radius = 0.18 if sea_style == "inland" else 0.24
            sigma = 0.42 if sea_style == "inland" else 0.6
            intensity = 0.18 if sea_style == "inland" else 0.32
            trench = terrain._create_location_mask(str(zone), radius=radius, sigma=sigma)
            result -= gaussian_smooth(trench * intensity)

    return gaussian_smooth(result)


def constraints_for_refinement(constraints: dict, profile: dict, uses_hard_topology: bool) -> dict:
    if not constraints:
        return {}

    refined = {
        "continents": list(constraints.get("continents") or []),
        "mountains": list(constraints.get("mountains") or []),
        "sea_zones": list(constraints.get("sea_zones") or []),
        "river_sources": list(constraints.get("river_sources") or []),
    }
    if uses_hard_topology and str(profile.get("layout_template", "default")) in {"split_east_west", "mediterranean", "single_island"}:
        refined["sea_zones"] = []
    return refined


def _single_island_axis_skeleton(terrain: TerrainGenerator, axis: str, strong: bool = False) -> np.ndarray:
    if axis == "north_south":
        if strong:
            return np.maximum.reduce(
                [
                    terrain._elliptic_gaussian(0.5, 0.5, 0.4, 0.048, 0.0),
                    terrain._elliptic_gaussian(0.26, 0.49, 0.17, 0.045, -0.03) * 0.88,
                    terrain._elliptic_gaussian(0.75, 0.52, 0.18, 0.048, 0.05) * 0.9,
                ]
            )
        return np.maximum.reduce(
            [
                terrain._elliptic_gaussian(0.5, 0.5, 0.34, 0.07, 0.0),
                terrain._elliptic_gaussian(0.3, 0.49, 0.16, 0.06, -0.04) * 0.8,
                terrain._elliptic_gaussian(0.71, 0.52, 0.17, 0.065, 0.06) * 0.82,
            ]
        )
    if strong:
        return np.maximum.reduce(
            [
                terrain._elliptic_gaussian(0.5, 0.5, 0.048, 0.4, 0.0),
                terrain._elliptic_gaussian(0.49, 0.26, 0.045, 0.17, -0.03) * 0.88,
                terrain._elliptic_gaussian(0.52, 0.75, 0.048, 0.18, 0.05) * 0.9,
            ]
        )
    return np.maximum.reduce(
        [
            terrain._elliptic_gaussian(0.5, 0.5, 0.07, 0.34, 0.0),
            terrain._elliptic_gaussian(0.49, 0.3, 0.06, 0.16, -0.04) * 0.8,
            terrain._elliptic_gaussian(0.52, 0.71, 0.065, 0.17, 0.06) * 0.82,
        ]
    )


def _single_island_axis_width_envelope(terrain: TerrainGenerator, axis: str, strong: bool) -> np.ndarray:
    y = terrain._y_norm
    x = terrain._x_norm
    if axis == "north_south":
        centerline = 0.5 + (y - 0.5) * 0.035 + np.sin((y - 0.5) * np.pi * 2.1) * (0.018 if strong else 0.014)
        width = (0.068 if strong else 0.082) + np.exp(-((y - 0.5) ** 2) / 0.08) * (0.034 if strong else 0.042)
        mask = np.exp(-((x - centerline) ** 2) / np.maximum(width**2, 1e-4))
        length_gate = np.exp(-((y - 0.5) ** 2) / (0.18 if strong else 0.22))
        tip_caps = np.maximum(
            terrain._elliptic_gaussian(0.22, 0.49, 0.12, 0.05, -0.04) * 0.74,
            terrain._elliptic_gaussian(0.79, 0.53, 0.13, 0.052, 0.05) * 0.76,
        )
    else:
        centerline = 0.5 + (x - 0.5) * 0.03 + np.sin((x - 0.5) * np.pi * 2.1) * (0.018 if strong else 0.014)
        width = (0.068 if strong else 0.082) + np.exp(-((x - 0.5) ** 2) / 0.08) * (0.034 if strong else 0.042)
        mask = np.exp(-((y - centerline) ** 2) / np.maximum(width**2, 1e-4))
        length_gate = np.exp(-((x - 0.5) ** 2) / (0.18 if strong else 0.22))
        tip_caps = np.maximum(
            terrain._elliptic_gaussian(0.49, 0.22, 0.05, 0.12, -0.04) * 0.74,
            terrain._elliptic_gaussian(0.53, 0.79, 0.052, 0.13, 0.05) * 0.76,
        )
    envelope = gaussian_smooth(np.clip(mask * length_gate, 0.0, 1.0).astype(np.float32))
    return np.maximum(envelope, tip_caps.astype(np.float32))


def _single_island_axis_cross_lobes(terrain: TerrainGenerator, axis: str) -> np.ndarray:
    if axis == "north_south":
        return np.maximum.reduce(
            [
                terrain._elliptic_gaussian(0.42, 0.43, 0.09, 0.12, -0.3) * 0.76,
                terrain._elliptic_gaussian(0.58, 0.57, 0.1, 0.13, 0.22) * 0.8,
            ]
        )
    return np.maximum.reduce(
        [
            terrain._elliptic_gaussian(0.43, 0.42, 0.12, 0.09, -0.3) * 0.76,
            terrain._elliptic_gaussian(0.57, 0.58, 0.13, 0.1, 0.22) * 0.8,
        ]
    )


def _single_island_axis_landform_mask(terrain: TerrainGenerator, axis: str, strong: bool) -> np.ndarray:
    width_envelope = _single_island_axis_width_envelope(terrain, axis=axis, strong=strong)
    skeleton = _single_island_axis_skeleton(terrain, axis=axis, strong=strong)
    if axis == "north_south":
        north_cap = terrain._elliptic_gaussian(0.18, 0.49, 0.11 if strong else 0.1, 0.06 if strong else 0.07, -0.05) * 0.72
        south_cap = terrain._elliptic_gaussian(0.82, 0.53, 0.12 if strong else 0.11, 0.06 if strong else 0.07, 0.06) * 0.74
        center_fill = terrain._elliptic_gaussian(0.5, 0.5, 0.3 if strong else 0.27, 0.08 if strong else 0.095, 0.0) * 0.58
    else:
        north_cap = terrain._elliptic_gaussian(0.49, 0.18, 0.06 if strong else 0.07, 0.11 if strong else 0.1, -0.05) * 0.72
        south_cap = terrain._elliptic_gaussian(0.53, 0.82, 0.06 if strong else 0.07, 0.12 if strong else 0.11, 0.06) * 0.74
        center_fill = terrain._elliptic_gaussian(0.5, 0.5, 0.08 if strong else 0.095, 0.3 if strong else 0.27, 0.0) * 0.58
    return gaussian_smooth(np.maximum.reduce([width_envelope, skeleton * 0.92, center_fill, north_cap, south_cap]).astype(np.float32))


def _single_island_axis_outer_ocean(terrain: TerrainGenerator, axis: str, width_envelope: np.ndarray) -> np.ndarray:
    if axis == "north_south":
        along_axis = 0.5 + np.exp(-((terrain._y_norm - 0.5) ** 2) / 0.16) * 0.5
        outside = np.clip((1.0 - np.clip(width_envelope * 1.42, 0.0, 0.96)) * along_axis, 0.0, 1.0)
        flank_bias = np.maximum(
            terrain._create_location_mask("west", radius=0.24, sigma=0.5) * 0.34,
            terrain._create_location_mask("east", radius=0.24, sigma=0.5) * 0.34,
        )
        return gaussian_smooth(np.maximum(outside, flank_bias).astype(np.float32))
    along_axis = 0.46 + np.exp(-((terrain._x_norm - 0.5) ** 2) / 0.18) * 0.42
    outside = np.clip((1.0 - np.clip(width_envelope * 1.38, 0.0, 0.96)) * along_axis, 0.0, 1.0)
    flank_bias = np.maximum(
        terrain._create_location_mask("north", radius=0.22, sigma=0.48) * 0.24,
        terrain._create_location_mask("south", radius=0.22, sigma=0.48) * 0.24,
    )
    return gaussian_smooth(np.maximum(outside, flank_bias).astype(np.float32))


def _single_island_axis_carve(terrain: TerrainGenerator, axis: str, strong: bool) -> np.ndarray:
    if axis == "north_south":
        return np.maximum.reduce(
            [
                terrain._create_location_mask("west", radius=0.24 if strong else 0.2, sigma=0.52 if strong else 0.44) * (0.3 if strong else 0.24),
                terrain._create_location_mask("east", radius=0.24 if strong else 0.2, sigma=0.52 if strong else 0.44) * (0.28 if strong else 0.22),
                terrain._elliptic_gaussian(0.5, 0.24, 0.2 if strong else 0.18, 0.075 if strong else 0.08, 0.0) * (0.2 if strong else 0.16),
                terrain._elliptic_gaussian(0.5, 0.76, 0.2 if strong else 0.18, 0.075 if strong else 0.08, 0.0) * (0.2 if strong else 0.16),
            ]
        )
    return np.maximum.reduce(
        [
            terrain._create_location_mask("north", radius=0.18 if strong else 0.16, sigma=0.44 if strong else 0.4) * (0.14 if strong else 0.1),
            terrain._create_location_mask("south", radius=0.18 if strong else 0.16, sigma=0.44 if strong else 0.4) * (0.14 if strong else 0.1),
        ]
    )


def _apply_single_island_axis_final_shape(terrain: TerrainGenerator, elevation: np.ndarray, axis: str) -> np.ndarray:
    if axis != "north_south":
        return elevation
    width_envelope = _single_island_axis_width_envelope(terrain, axis=axis, strong=True)
    spine = _single_island_axis_skeleton(terrain, axis=axis, strong=True)
    landform_mask = _single_island_axis_landform_mask(terrain, axis=axis, strong=True)
    lateral_ocean = _single_island_axis_outer_ocean(terrain, axis=axis, width_envelope=width_envelope)
    shaped = elevation.astype(np.float32).copy()
    target_land = ((np.maximum(spine * 1.06, landform_mask) - 0.31) * 0.74).astype(np.float32)
    corridor_mask = landform_mask >= 0.3
    shaped[corridor_mask] = np.maximum(shaped[corridor_mask], target_land[corridor_mask])
    envelope_cap = ((np.maximum.reduce([spine, width_envelope, landform_mask * 0.92]) - 0.42) * 0.54).astype(np.float32)
    shaped = np.minimum(shaped, envelope_cap)
    outside_mask = landform_mask < 0.3
    shaped[outside_mask] = np.minimum(
        shaped[outside_mask],
        (-0.2 - (0.3 - landform_mask[outside_mask]) * 0.62).astype(np.float32),
    )
    hard_ocean = lateral_ocean >= 0.28
    shaped[hard_ocean] = np.minimum(shaped[hard_ocean], (-0.22 - lateral_ocean[hard_ocean] * 0.34).astype(np.float32))
    spine_mask = spine >= 0.2
    spine_target = ((spine - 0.26) * 0.56).astype(np.float32)
    shaped[spine_mask] = np.maximum(shaped[spine_mask], spine_target[spine_mask])
    return gaussian_smooth(shaped)


def _enforce_single_dominant_landmass(terrain: TerrainGenerator, elevation: np.ndarray, topology_intent: dict | None = None) -> np.ndarray:
    land_mask = elevation > 0.0
    labels, counts = _label_land_components(land_mask)
    if len(counts) <= 1:
        shaped = elevation.astype(np.float32).copy()
    else:
        dominant_label = max(counts, key=counts.get)
        dominant_mask = labels == dominant_label
        component_noise = terrain._fbm(scale=42.0, octaves=3, persistence=0.58, lacunarity=2.02, offset=941.0) * 2.0 - 1.0
        edge_band = gaussian_smooth(dominant_mask.astype(np.float32))

        shaped = elevation.astype(np.float32).copy()
        stray_mask = land_mask & ~dominant_mask
        shaped[stray_mask] = np.minimum(shaped[stray_mask], (-0.12 + component_noise[stray_mask] * 0.04).astype(np.float32))
        shaped = np.maximum(shaped, ((edge_band - 0.42) * 0.26).astype(np.float32))

    ring_ocean = np.clip(
        terrain._create_location_mask("west", radius=0.18, sigma=0.48) * 0.08
        + terrain._create_location_mask("east", radius=0.18, sigma=0.48) * 0.08
        + terrain._create_location_mask("north", radius=0.16, sigma=0.42) * 0.06
        + terrain._create_location_mask("south", radius=0.16, sigma=0.42) * 0.06,
        0.0,
        0.2,
    )
    shaped -= ring_ocean.astype(np.float32)
    intent = topology_intent or {}
    if _intent_modifier(intent, "shape_bias", "balanced") == "elongated":
        shape_axis = _intent_modifier(intent, "shape_axis", "east_west")
        spine = _single_island_axis_skeleton(terrain, axis=shape_axis, strong=True)
        width_envelope = _single_island_axis_width_envelope(terrain, axis=shape_axis, strong=True)
        landform_mask = _single_island_axis_landform_mask(terrain, axis=shape_axis, strong=True)
        shoulders = _single_island_axis_cross_lobes(terrain, axis=shape_axis)
        spine = np.maximum.reduce([spine, width_envelope, landform_mask * 0.84, shoulders * 0.36])
        carve = _single_island_axis_carve(terrain, axis=shape_axis, strong=True)
        spine_target = ((spine - 0.28) * (0.54 if shape_axis == "north_south" else 0.42)).astype(np.float32)
        spine_mask = spine >= (0.22 if shape_axis == "north_south" else 0.27)
        shaped[spine_mask] = np.maximum(shaped[spine_mask], spine_target[spine_mask])
        if shape_axis == "north_south":
            envelope_cap = ((np.maximum(spine, width_envelope * 0.96) - 0.34) * 0.62).astype(np.float32)
            shaped = np.minimum(shaped, envelope_cap)
            outer_corridor = landform_mask < 0.24
            shaped[outer_corridor] = np.minimum(
                shaped[outer_corridor],
                (-0.14 - (0.24 - landform_mask[outer_corridor]) * 0.52).astype(np.float32),
            )
            shaped = _force_water_mask(shaped, carve, base_depth=-0.18, mask_threshold=0.32)
            lateral_ocean = _single_island_axis_outer_ocean(terrain, axis=shape_axis, width_envelope=width_envelope)
            shaped = _force_water_mask(shaped, lateral_ocean, base_depth=-0.16, mask_threshold=0.44)
            hard_ocean = lateral_ocean >= 0.34
            shaped[hard_ocean] = np.minimum(shaped[hard_ocean], (-0.18 - lateral_ocean[hard_ocean] * 0.36).astype(np.float32))
            shaped[spine_mask] = np.maximum(shaped[spine_mask], spine_target[spine_mask])
        else:
            shaped -= carve.astype(np.float32)
    shaped = gaussian_smooth(shaped)
    if _intent_modifier(intent, "shape_bias", "balanced") == "elongated":
        shaped = _apply_single_island_axis_final_shape(
            terrain,
            shaped,
            axis=_intent_modifier(intent, "shape_axis", "east_west"),
        )
    return shaped


def _enforce_archipelago_islands(terrain: TerrainGenerator, elevation: np.ndarray, topology_intent: dict) -> np.ndarray:
    density = _intent_modifier(topology_intent, "island_density", "balanced")
    shaped = elevation.astype(np.float32).copy()
    target_components = 4 if density == "dense" else 3
    if _count_components_from_mask(shaped > 0.0, min_cells=90 if density != "sparse" else 70) >= target_components:
        return shaped

    core_specs = [
        (0.38, 0.33, 0.08, 0.06, -0.24, 0.92),
        (0.62, 0.54, 0.075, 0.06, 0.2, 0.88),
        (0.45, 0.72, 0.07, 0.055, -0.18, 0.84),
        (0.66, 0.24, 0.065, 0.05, 0.28, 0.78),
    ]
    if density == "dense":
        core_specs.extend(
            [
                (0.32, 0.57, 0.055, 0.045, 0.14, 0.68),
                (0.57, 0.36, 0.05, 0.04, -0.22, 0.64),
            ]
        )
    island_cores = np.maximum.reduce(
        [terrain._elliptic_gaussian(cy, cx, ry, rx, rotation) * weight for cy, cx, ry, rx, rotation, weight in core_specs]
    )
    core_target = ((island_cores - 0.36) * 0.4).astype(np.float32)
    core_mask = island_cores >= 0.34
    shaped[core_mask] = np.maximum(shaped[core_mask], core_target[core_mask])

    channels = np.maximum.reduce(
        [
            terrain._elliptic_gaussian(0.5, 0.5, 0.08, 0.28, 0.0),
            terrain._elliptic_gaussian(0.5, 0.5, 0.22, 0.05, 0.0) * 0.92,
            terrain._elliptic_gaussian(0.42, 0.38, 0.05, 0.16, -0.4),
            terrain._elliptic_gaussian(0.58, 0.63, 0.05, 0.16, 0.38),
            terrain._elliptic_gaussian(0.44, 0.64, 0.05, 0.15, 0.2),
            terrain._elliptic_gaussian(0.56, 0.44, 0.045, 0.14, -0.28),
        ]
    )
    if density == "sparse":
        channels = np.maximum(channels, terrain._elliptic_gaussian(0.51, 0.52, 0.12, 0.34, 0.06) * 0.78)
    channels = _naturalize_water_mask(
        terrain,
        channels,
        amplitude=0.2 if density == "sparse" else 0.16,
        smooth_passes=2,
        asymmetry=0.1,
        axis="vertical",
    )
    shaped = _force_water_mask(
        shaped,
        channels,
        base_depth=-0.27 if density == "sparse" else -0.22,
        mask_threshold=0.46 if density == "dense" else 0.5,
    )
    shaped[core_mask] = np.maximum(shaped[core_mask], core_target[core_mask])
    return np.clip(shaped, -1.0, 1.0).astype(np.float32)


def _enforce_two_dominant_landmasses(terrain: TerrainGenerator, elevation: np.ndarray, topology_intent: dict) -> np.ndarray:
    rift_width = _intent_modifier(topology_intent, "rift_width", "balanced")
    rift_profile = _intent_modifier(topology_intent, "rift_profile", "natural")
    land_mask = elevation > 0.0
    labels, counts = _label_land_components(land_mask)
    if len(counts) <= 2:
        shaped = elevation.astype(np.float32).copy()
    else:
        sorted_labels = sorted(counts, key=counts.get, reverse=True)
        keep_labels = set(sorted_labels[:2])
        keep_mask = np.isin(labels, list(keep_labels))
        stray_mask = land_mask & ~keep_mask
        shaped = elevation.astype(np.float32).copy()
        removal_noise = terrain._fbm(scale=39.0, octaves=3, persistence=0.56, lacunarity=2.08, offset=977.0) * 2.0 - 1.0
        shaped[stray_mask] = np.minimum(shaped[stray_mask], (-0.1 + removal_noise[stray_mask] * 0.05).astype(np.float32))

    west_spine = np.maximum.reduce(
        [
            terrain._elliptic_gaussian(0.5, 0.23, 0.34, 0.09, -0.08),
            terrain._elliptic_gaussian(0.34, 0.28, 0.15, 0.08, 0.22) * 0.74,
            terrain._elliptic_gaussian(0.69, 0.19, 0.16, 0.07, -0.16) * 0.68,
        ]
    )
    east_spine = np.maximum.reduce(
        [
            terrain._elliptic_gaussian(0.5, 0.77, 0.34, 0.09, 0.1),
            terrain._elliptic_gaussian(0.31, 0.72, 0.14, 0.08, -0.2) * 0.72,
            terrain._elliptic_gaussian(0.66, 0.82, 0.15, 0.07, 0.16) * 0.66,
        ]
    )
    continental_spines = np.maximum(west_spine, east_spine)
    spine_target = ((continental_spines - 0.38) * 0.44).astype(np.float32)
    spine_mask = continental_spines >= 0.32
    shaped[spine_mask] = np.maximum(shaped[spine_mask], spine_target[spine_mask])

    west_midland = terrain._elliptic_gaussian(0.5, 0.24, 0.1, 0.18, 0.02) * 0.72
    east_midland = terrain._elliptic_gaussian(0.5, 0.76, 0.1, 0.18, -0.03) * 0.72
    midland = np.maximum(west_midland, east_midland)
    midland_target = ((midland - 0.36) * 0.34).astype(np.float32)
    midland_mask = midland >= 0.34
    shaped[midland_mask] = np.maximum(shaped[midland_mask], midland_target[midland_mask])

    width_scale = {"narrow": 0.78, "balanced": 1.0, "broad": 1.3}.get(rift_width, 1.0)
    rift_mask = _natural_split_barrier(terrain, axis="vertical", width_scale=width_scale)
    if rift_profile == "broken":
        rift_mask = np.maximum(
            rift_mask,
            np.maximum.reduce(
                [
                    terrain._elliptic_gaussian(0.35, 0.48, 0.075, 0.07, -0.14) * 0.72,
                    terrain._elliptic_gaussian(0.62, 0.53, 0.095, 0.075, 0.18) * 0.84,
                ]
            ),
        )
    shaped = _force_water_mask(
        shaped,
        rift_mask,
        base_depth={"narrow": -0.26, "balanced": -0.3, "broad": -0.36}.get(rift_width, -0.3) - (0.03 if rift_profile == "broken" else 0.0),
        mask_threshold={"narrow": 0.6, "balanced": 0.56, "broad": 0.5}.get(rift_width, 0.56) - (0.03 if rift_profile == "broken" else 0.0),
    )
    return gaussian_smooth(shaped)


def _enforce_central_inland_sea_basin(terrain: TerrainGenerator, elevation: np.ndarray, plan: dict) -> np.ndarray:
    inland_seas = list(plan.get("inland_seas") or [])
    topology_intent = plan.get("topology_intent") or {}
    basin_shape = _intent_modifier(topology_intent, "basin_shape", "balanced")
    basin_style = _intent_modifier(topology_intent, "basin_style", "balanced")
    connection = str((inland_seas[0] if inland_seas else {}).get("connection", "enclosed")).lower()
    shaped = elevation.astype(np.float32).copy()

    basin_mask = _natural_inland_sea_basin(terrain, basin_shape=basin_shape, basin_style=basin_style)
    shaped = _force_water_mask(
        shaped,
        basin_mask,
        base_depth={"compact": -0.24, "balanced": -0.28, "broad": -0.32, "branched": -0.29}.get(basin_shape, -0.28),
        mask_threshold={"compact": 0.58, "balanced": 0.52, "broad": 0.46, "branched": 0.5}.get(basin_shape, 0.52),
    )

    outer_ring = np.maximum.reduce(
        [
            terrain._elliptic_gaussian(0.5, 0.5, 0.22, 0.34, 0.02),
            terrain._elliptic_gaussian(0.5, 0.5, 0.18, 0.3, -0.04) * 0.9,
        ]
    )
    rim_seed = np.clip(outer_ring - basin_mask * 0.92, 0.0, 1.0)
    rim_noise = terrain._fbm(scale=57.0, octaves=3, persistence=0.57, lacunarity=2.02, offset=1019.0) * 2.0 - 1.0
    rim_mask = np.clip(rim_seed * (0.9 + rim_noise * 0.18), 0.0, 1.0)

    if "strait" in connection:
        opening = np.maximum.reduce(
            [
                terrain._elliptic_gaussian(0.49, 0.76, 0.045, 0.08, 0.22),
                terrain._elliptic_gaussian(0.53, 0.7, 0.04, 0.06, -0.1) * 0.75,
            ]
        )
        rim_mask = np.clip(rim_mask - opening * 0.9, 0.0, 1.0)
        shaped = _force_water_mask(shaped, opening, base_depth=-0.22, mask_threshold=0.58)
    elif "east" in connection or "open" in connection:
        outlet = terrain._elliptic_gaussian(0.5, 0.82, 0.05, 0.1, 0.08)
        rim_mask = np.clip(rim_mask - outlet * 0.86, 0.0, 1.0)
        shaped = _force_water_mask(shaped, outlet, base_depth=-0.24, mask_threshold=0.56)

    land_target = ((rim_mask - 0.42) * 0.42).astype(np.float32)
    strong_rim = rim_mask >= 0.34
    shaped[strong_rim] = np.maximum(shaped[strong_rim], land_target[strong_rim])

    labels, counts = _label_water_components(shaped < 0.0)
    central_window = np.zeros_like(shaped, dtype=bool)
    h, w = shaped.shape
    central_window[int(h * 0.25) : int(h * 0.75), int(w * 0.28) : int(w * 0.72)] = True
    keep_water = np.zeros_like(shaped, dtype=bool)
    for component_id, size in counts.items():
        component = labels == component_id
        if np.any(component & central_window):
            keep_water |= component
    stray_water = (shaped < 0.0) & central_window & ~keep_water
    shaped[stray_water] = np.maximum(shaped[stray_water], 0.04)
    return gaussian_smooth(shaped)


def enforce_topology_components(terrain: TerrainGenerator, elevation: np.ndarray, plan: dict) -> np.ndarray:
    topology_intent = plan.get("topology_intent") or {}
    kind = str(topology_intent.get("kind", "")).strip().lower()
    if kind == "single_island":
        return _enforce_single_dominant_landmass(terrain, elevation, topology_intent)
    if kind == "archipelago_chain":
        return _enforce_archipelago_islands(terrain, elevation, topology_intent)
    if kind == "two_continents_with_rift_sea":
        return _enforce_two_dominant_landmasses(terrain, elevation, topology_intent)
    if kind == "central_enclosed_inland_sea":
        return _enforce_central_inland_sea_basin(terrain, elevation, plan)
    return elevation


def _build_topology_intent_field(terrain: TerrainGenerator, topology_intent: dict, ruggedness: float) -> np.ndarray | None:
    kind = str(topology_intent.get("kind", "")).strip().lower()
    if kind == "single_island":
        return _build_single_island_intent_topology(terrain, ruggedness, topology_intent)
    if kind == "archipelago_chain":
        return _build_archipelago_intent_topology(terrain, ruggedness, topology_intent)
    if kind == "peninsula_coast":
        return _build_peninsula_intent_topology(terrain, ruggedness, topology_intent)
    if kind == "two_continents_with_rift_sea":
        return _build_two_continents_rift_topology(terrain, ruggedness, topology_intent)
    if kind == "central_enclosed_inland_sea":
        return _build_central_inland_sea_intent_topology(terrain, ruggedness, topology_intent)
    return None


def _build_single_island_intent_topology(terrain: TerrainGenerator, ruggedness: float, topology_intent: dict) -> np.ndarray:
    shape_bias = _intent_modifier(topology_intent, "shape_bias", "balanced")
    shape_axis = _intent_modifier(topology_intent, "shape_axis", "east_west")
    if shape_bias == "elongated":
        skeleton = _single_island_axis_skeleton(terrain, axis=shape_axis)
        width_envelope = _single_island_axis_width_envelope(terrain, axis=shape_axis, strong=False)
        landform_mask = _single_island_axis_landform_mask(terrain, axis=shape_axis, strong=False)
        cross_lobes = _single_island_axis_cross_lobes(terrain, axis=shape_axis)
        core = np.maximum.reduce([skeleton, width_envelope * 0.92, landform_mask * 0.86, cross_lobes * 0.58])
        bite_masks = _single_island_axis_carve(terrain, axis=shape_axis, strong=False)
        bite_scale = 1.18
    elif shape_bias == "round":
        core = np.maximum.reduce(
            [
                terrain._elliptic_gaussian(0.5, 0.5, 0.19, 0.19, 0.0),
                terrain._elliptic_gaussian(0.46, 0.56, 0.1, 0.1, 0.0) * 0.66,
                terrain._elliptic_gaussian(0.56, 0.43, 0.095, 0.095, 0.0) * 0.64,
            ]
        )
        bite_scale = 0.84
        bite_masks = np.maximum.reduce(
            [
                terrain._create_location_mask("west", radius=0.18, sigma=0.42) * 0.2,
                terrain._create_location_mask("east", radius=0.16, sigma=0.38) * 0.16,
                terrain._create_location_mask("south", radius=0.14, sigma=0.36) * 0.12,
            ]
        )
    else:
        core = np.maximum.reduce(
            [
                terrain._elliptic_gaussian(0.52, 0.48, 0.19, 0.16, -0.2),
                terrain._elliptic_gaussian(0.47, 0.55, 0.15, 0.19, 0.28),
                terrain._elliptic_gaussian(0.57, 0.42, 0.11, 0.13, -0.44) * 0.74,
            ]
        )
        bite_scale = 1.0
        bite_masks = np.maximum.reduce(
            [
                terrain._create_location_mask("west", radius=0.18, sigma=0.42) * 0.2,
                terrain._create_location_mask("east", radius=0.16, sigma=0.38) * 0.16,
                terrain._create_location_mask("south", radius=0.14, sigma=0.36) * 0.12,
            ]
        )
    coastal_noise = terrain._fbm(scale=52.0, octaves=4, persistence=0.56, lacunarity=2.05, offset=805.0)
    irregular = np.clip(core * (0.86 + coastal_noise * 0.26), 0.0, 1.0)
    island = terrain._shoreline_profile(irregular, water_level=0.4, gain=1.86)
    ocean_bites = bite_masks * bite_scale
    field = island * 1.32 - ocean_bites
    if shape_bias == "elongated" and shape_axis == "north_south":
        outer_ocean = _single_island_axis_outer_ocean(terrain, axis=shape_axis, width_envelope=width_envelope)
        field -= outer_ocean * 1.18
        field -= np.clip(1.0 - landform_mask * 1.16, 0.0, 1.0) * 0.32
    return _normalize_topology(np.tanh((field * 2.0 - 1.0) * (1.06 + ruggedness * 0.1)))


def _build_archipelago_intent_topology(terrain: TerrainGenerator, ruggedness: float, topology_intent: dict) -> np.ndarray:
    density = _intent_modifier(topology_intent, "island_density", "balanced")
    archipelago = np.zeros((terrain.height, terrain.width), dtype=np.float32)
    if density == "dense":
        island_specs = [
            ("west", 0.082, 0.0, 0.76),
            ("east", 0.08, 0.18, 0.72),
            ("northwest", 0.072, -0.32, 0.66),
            ("northeast", 0.07, 0.28, 0.64),
            ("southwest", 0.074, 0.14, 0.68),
            ("southeast", 0.068, -0.22, 0.62),
            ("center", 0.062, 0.08, 0.58),
        ]
        ocean_scale = 0.72
    elif density == "sparse":
        island_specs = [
            ("west", 0.075, -0.08, 0.62),
            ("east", 0.07, 0.24, 0.58),
            ("northwest", 0.062, -0.34, 0.52),
            ("southeast", 0.06, -0.2, 0.48),
        ]
        ocean_scale = 1.16
    else:
        island_specs = [
            ("west", 0.085, 0.0, 0.7),
            ("east", 0.082, 0.18, 0.66),
            ("northwest", 0.074, -0.32, 0.6),
            ("northeast", 0.07, 0.28, 0.56),
            ("southwest", 0.076, 0.14, 0.62),
            ("southeast", 0.068, -0.22, 0.54),
            ("center", 0.058, 0.08, 0.42),
        ]
        ocean_scale = 1.0
    for position, size, rotation, weight in island_specs:
        cy, cx = terrain._resolve_position(position)
        island = terrain._elliptic_gaussian(cy, cx, size * 0.8, size * 0.55, rotation)
        archipelago = np.maximum(archipelago, island.astype(np.float32) * weight)
    scatter_noise = terrain._fbm(scale=41.0, octaves=4, persistence=0.57, lacunarity=2.1, offset=1091.0)
    archipelago = np.clip(archipelago * (0.78 + scatter_noise * 0.24), 0.0, 1.0)
    separated_ocean = np.maximum.reduce(
        [
            terrain._create_location_mask("center", radius=0.18, sigma=0.52) * 0.2 * ocean_scale,
            terrain._elliptic_gaussian(0.5, 0.5, 0.08, 0.34, 0.0) * 0.16 * ocean_scale,
        ]
    )
    field = archipelago * 1.1 - separated_ocean
    return _normalize_topology(np.tanh((field * 2.0 - 1.0) * (0.96 + ruggedness * 0.06)))


def _build_peninsula_intent_topology(terrain: TerrainGenerator, ruggedness: float, topology_intent: dict) -> np.ndarray:
    notes = " ".join(topology_intent.get("notes") or [])
    anchor = "east"
    for candidate in ("west", "east", "north", "south"):
        if candidate in notes:
            anchor = candidate
            break

    mainland = {
        "east": np.maximum(
            terrain._elliptic_gaussian(0.5, 0.23, 0.3, 0.22, -0.08),
            terrain._elliptic_gaussian(0.34, 0.29, 0.16, 0.12, 0.24) * 0.72,
        ),
        "west": np.maximum(
            terrain._elliptic_gaussian(0.5, 0.77, 0.3, 0.22, 0.08),
            terrain._elliptic_gaussian(0.66, 0.7, 0.16, 0.12, -0.24) * 0.72,
        ),
        "north": np.maximum(
            terrain._elliptic_gaussian(0.24, 0.5, 0.22, 0.3, 0.04),
            terrain._elliptic_gaussian(0.3, 0.66, 0.12, 0.16, 0.34) * 0.72,
        ),
        "south": np.maximum(
            terrain._elliptic_gaussian(0.76, 0.5, 0.22, 0.3, -0.04),
            terrain._elliptic_gaussian(0.7, 0.34, 0.12, 0.16, -0.34) * 0.72,
        ),
    }[anchor]
    peninsula = _plan_peninsula_component(terrain, anchor, 0.24) * 1.18
    coastal_bite = {
        "east": terrain._create_location_mask("east", radius=0.18, sigma=0.42) * 0.18,
        "west": terrain._create_location_mask("west", radius=0.18, sigma=0.42) * 0.18,
        "north": terrain._create_location_mask("north", radius=0.18, sigma=0.42) * 0.18,
        "south": terrain._create_location_mask("south", radius=0.18, sigma=0.42) * 0.18,
    }[anchor]
    shape_noise = terrain._fbm(scale=49.0, octaves=3, persistence=0.56, lacunarity=2.08, offset=1139.0)
    land = np.clip(np.maximum(mainland, peninsula) * (0.86 + shape_noise * 0.22), 0.0, 1.0)
    field = land * 1.28 - coastal_bite
    return _normalize_topology(np.tanh((field * 2.0 - 1.0) * (1.0 + ruggedness * 0.08)))


def _build_two_continents_rift_topology(terrain: TerrainGenerator, ruggedness: float, topology_intent: dict) -> np.ndarray:
    rift_width = _intent_modifier(topology_intent, "rift_width", "balanced")
    rift_profile = _intent_modifier(topology_intent, "rift_profile", "natural")
    h, w = terrain.height, terrain.width
    base_seed = terrain._rng.integers(0, 2**31) if hasattr(terrain, '_rng') else 42
    rng = terrain._rng if hasattr(terrain, '_rng') else np.random.RandomState(42)

    west_seed = (base_seed * 7 + 13) % (2**31)
    east_seed = (base_seed * 11 + 97) % (2**31)
    west_region = _generate_region_elevation(
        w, h, west_seed,
        (slice(0, h), slice(0, int(w * 0.52))),
        blend_width=8,
    )
    east_region = _generate_region_elevation(
        w, h, east_seed,
        (slice(0, h), slice(int(w * 0.48), w)),
        blend_width=8,
    )

    w_cy = 0.52 + (rng.random() - 0.5) * 0.04
    e_cy = 0.48 + (rng.random() - 0.5) * 0.04
    w_ry = 0.17 + (rng.random() - 0.5) * 0.03
    e_ry = 0.18 + (rng.random() - 0.5) * 0.03
    west_mask = np.clip(west_region * 3.0 + terrain._elliptic_gaussian(w_cy, 0.24, 0.26, w_ry, -0.18) * 0.5, 0.0, 1.0)
    east_mask = np.clip(east_region * 3.0 + terrain._elliptic_gaussian(e_cy, 0.76, 0.24, e_ry, 0.22) * 0.5, 0.0, 1.0)

    west_noise = terrain._fbm(scale=58.0, octaves=4, persistence=0.55, lacunarity=2.0, offset=847.0)
    east_noise = terrain._fbm(scale=61.0, octaves=4, persistence=0.57, lacunarity=2.02, offset=883.0)
    west = np.clip(west_mask * (0.82 + west_noise * 0.22), 0.0, 1.0)
    east = np.clip(east_mask * (0.84 + east_noise * 0.22), 0.0, 1.0)

    west *= _longitudinal_gate(terrain, "west")
    east *= _longitudinal_gate(terrain, "east")

    w_lobe1_y = 0.36 + (rng.random() - 0.5) * 0.04
    w_lobe2_y = 0.71 + (rng.random() - 0.5) * 0.04
    e_lobe1_y = 0.33 + (rng.random() - 0.5) * 0.04
    e_lobe2_y = 0.67 + (rng.random() - 0.5) * 0.04
    west += terrain._elliptic_gaussian(w_lobe1_y, 0.3, 0.15, 0.12, 0.34) * 0.74
    west += terrain._elliptic_gaussian(w_lobe2_y, 0.18, 0.14, 0.11, -0.26) * 0.68
    east += terrain._elliptic_gaussian(e_lobe1_y, 0.69, 0.13, 0.1, -0.32) * 0.72
    east += terrain._elliptic_gaussian(e_lobe2_y, 0.82, 0.13, 0.1, 0.28) * 0.64

    asym_noise = terrain._fbm(scale=84.0, octaves=3, persistence=0.54, lacunarity=2.05, offset=723.0) * 2.0 - 1.0
    west = np.clip(west * (0.92 + asym_noise * 0.10), 0.0, 1.5)
    east = np.clip(east * (0.92 - asym_noise * 0.07), 0.0, 1.5)

    width_scale = {"narrow": 0.78, "balanced": 1.0, "broad": 1.3}.get(rift_width, 1.0)
    strength = {"narrow": 0.96, "balanced": 1.1, "broad": 1.3}.get(rift_width, 1.1)
    rift = _natural_split_barrier(terrain, axis="vertical", width_scale=width_scale)
    if rift_profile == "broken":
        rift = np.maximum(
            rift,
            np.maximum.reduce(
                [
                    terrain._elliptic_gaussian(0.34, 0.49, 0.08, 0.07, -0.18) * 0.74,
                    terrain._elliptic_gaussian(0.58, 0.52, 0.1, 0.08, 0.24) * 0.88,
                    terrain._elliptic_gaussian(0.79, 0.47, 0.065, 0.06, -0.08) * 0.62,
                ]
            ),
        )
        strength += 0.08
    elif rift_profile == "smooth":
        strength -= 0.08
    field = np.maximum(west, east) * 1.34 - rift * strength
    return _normalize_topology(field)


def _build_central_inland_sea_intent_topology(terrain: TerrainGenerator, ruggedness: float, topology_intent: dict) -> np.ndarray:
    basin_shape = _intent_modifier(topology_intent, "basin_shape", "balanced")
    basin_style = _intent_modifier(topology_intent, "basin_style", "balanced")
    h, w = terrain.height, terrain.width
    base_seed = terrain._rng.integers(0, 2**31) if hasattr(terrain, '_rng') else 42

    north_seed = (base_seed * 23 + 41) % (2**31)
    south_seed = (base_seed * 31 + 59) % (2**31)
    north_region = _generate_region_elevation(
        w, h, north_seed,
        (slice(0, int(h * 0.55)), slice(0, w)),
        blend_width=10,
    )
    south_region = _generate_region_elevation(
        w, h, south_seed,
        (slice(int(h * 0.45), h), slice(0, w)),
        blend_width=10,
    )

    north = np.maximum.reduce(
        [
            terrain._elliptic_gaussian(0.24, 0.46, 0.17, 0.29, -0.12),
            terrain._elliptic_gaussian(0.31, 0.68, 0.12, 0.16, 0.34) * 0.76,
        ]
    )
    south = np.maximum.reduce(
        [
            terrain._elliptic_gaussian(0.76, 0.54, 0.18, 0.27, 0.16),
            terrain._elliptic_gaussian(0.67, 0.31, 0.11, 0.15, -0.28) * 0.72,
        ]
    )
    north = np.clip(north * 0.65 + north_region * 0.55, 0.0, 1.5)
    south = np.clip(south * 0.65 + south_region * 0.55, 0.0, 1.5)

    west_shoulder = terrain._elliptic_gaussian(0.53, 0.25, 0.1, 0.13, -0.5) * 0.62
    east_shoulder = terrain._elliptic_gaussian(0.47, 0.76, 0.08, 0.12, 0.44) * 0.48
    basin = _natural_inland_sea_basin(terrain, basin_shape=basin_shape, basin_style=basin_style)
    if basin_style == "rift":
        west_shoulder *= 0.44
        east_shoulder *= 0.4
    elif basin_style == "mediterranean":
        west_shoulder *= 0.8
        east_shoulder *= 0.8
    land = np.maximum.reduce([north, south, west_shoulder, east_shoulder])
    field = land * 1.28 - basin * {"compact": 0.92, "balanced": 1.05, "broad": 1.2, "branched": 1.14}.get(basin_shape, 1.05)
    return _normalize_topology(field)


def _build_split_east_west_topology(terrain: TerrainGenerator, constraints: dict, ruggedness: float) -> np.ndarray:
    h, w = terrain.height, terrain.width
    base_seed = terrain._rng.integers(0, 2**31) if hasattr(terrain, '_rng') else 42
    rng = terrain._rng if hasattr(terrain, '_rng') else np.random.RandomState(42)

    west_seed = (base_seed * 13 + 29) % (2**31)
    east_seed = (base_seed * 17 + 71) % (2**31)
    west_region = _generate_region_elevation(
        w, h, west_seed,
        (slice(0, h), slice(0, int(w * 0.52))),
        blend_width=10,
    )
    east_region = _generate_region_elevation(
        w, h, east_seed,
        (slice(0, h), slice(int(w * 0.48), w)),
        blend_width=10,
    )

    w_size = 0.46 + (rng.random() - 0.5) * 0.06
    e_size = 0.46 + (rng.random() - 0.5) * 0.06
    w_nw_w = 0.58 + (rng.random() - 0.5) * 0.10
    w_sw_w = 0.66 + (rng.random() - 0.5) * 0.10
    e_ne_w = 0.62 + (rng.random() - 0.5) * 0.10
    e_se_w = 0.54 + (rng.random() - 0.5) * 0.10

    west = _continent_component(terrain, "west", w_size, [("northwest", 0.2, w_nw_w), ("southwest", 0.22, w_sw_w)])
    east = _continent_component(terrain, "east", e_size, [("northeast", 0.2, e_ne_w), ("southeast", 0.18, e_se_w)])
    west *= _longitudinal_gate(terrain, "west")
    east *= _longitudinal_gate(terrain, "east")

    west = np.clip(west * 0.7 + west_region * 0.5, 0.0, 1.5)
    east = np.clip(east * 0.7 + east_region * 0.5, 0.0, 1.5)

    asym_noise = terrain._fbm(scale=78.0, octaves=3, persistence=0.54, lacunarity=2.05, offset=717.0) * 2.0 - 1.0
    west = np.clip(west * (0.92 + asym_noise * 0.12), 0.0, 1.5)
    east = np.clip(east * (0.92 - asym_noise * 0.08), 0.0, 1.5)

    open_channel = np.maximum.reduce(
        [
            terrain._create_location_mask("north", radius=0.16, sigma=0.9),
            terrain._create_location_mask("center", radius=0.18, sigma=1.0),
            terrain._create_location_mask("south", radius=0.16, sigma=0.9),
        ]
    )
    center_spine = terrain._elliptic_gaussian(0.5, 0.5, 0.1, 0.46, 0.0)
    full_channel = _axis_aligned_ocean_barrier(terrain, "vertical", center=0.5, width=0.11)
    open_channel = np.clip(open_channel * 0.84 + center_spine * 0.66 + full_channel * 0.92, 0.0, 1.75)

    field = np.maximum(west, east) * 1.26 - open_channel * 1.42
    field = _apply_mountain_topology(terrain, field, constraints, ruggedness)
    return _normalize_topology(field)


def _build_inland_sea_topology(terrain: TerrainGenerator, constraints: dict, ruggedness: float) -> np.ndarray:
    rng = terrain._rng if hasattr(terrain, '_rng') else np.random.RandomState(42)
    n_size = 0.34 + (rng.random() - 0.5) * 0.06
    s_size = 0.34 + (rng.random() - 0.5) * 0.06
    nw_weight = 0.88 + (rng.random() - 0.5) * 0.12
    ne_weight = 0.88 + (rng.random() - 0.5) * 0.12
    sw_weight = 0.88 + (rng.random() - 0.5) * 0.12
    se_weight = 0.88 + (rng.random() - 0.5) * 0.12
    w_size = 0.18 + (rng.random() - 0.5) * 0.04
    e_size = 0.18 + (rng.random() - 0.5) * 0.04

    north_arc = _continent_component(terrain, "north", n_size, [("northwest", 0.26, nw_weight), ("northeast", 0.26, ne_weight)])
    south_arc = _continent_component(terrain, "south", s_size, [("southwest", 0.26, sw_weight), ("southeast", 0.26, se_weight)])
    west_wall = _continent_component(terrain, "west", w_size, [("northwest", 0.18, 0.55), ("southwest", 0.18, 0.55)])
    east_wall = _continent_component(terrain, "east", e_size, [("northeast", 0.18, 0.55), ("southeast", 0.18, 0.55)])

    basin = _natural_inland_sea_basin(terrain)

    west_outlet_y = 0.5 + (rng.random() - 0.5) * 0.04
    east_outlet_y = 0.5 + (rng.random() - 0.5) * 0.04
    west_outlet = terrain._elliptic_gaussian(west_outlet_y, 0.18, 0.05, 0.08, 0.0)
    east_outlet = terrain._elliptic_gaussian(east_outlet_y, 0.82, 0.05, 0.08, 0.0)
    north_cut_y = 0.34 + (rng.random() - 0.5) * 0.04
    south_cut_y = 0.66 + (rng.random() - 0.5) * 0.04
    north_cut = terrain._elliptic_gaussian(north_cut_y, 0.5, 0.06, 0.18, 0.0)
    south_cut = terrain._elliptic_gaussian(south_cut_y, 0.5, 0.06, 0.18, 0.0)

    asym_noise = terrain._fbm(scale=72.0, octaves=3, persistence=0.54, lacunarity=2.05, offset=711.0) * 2.0 - 1.0
    enclosure = np.maximum.reduce([north_arc, south_arc, west_wall, east_wall]) * 1.2
    enclosure = np.clip(enclosure * (0.92 + asym_noise * 0.14), 0.0, 1.5)

    field = enclosure - basin * 1.34 - (west_outlet + east_outlet) * 0.22 - (north_cut + south_cut) * 0.24
    field = _apply_mountain_topology(terrain, field, constraints, ruggedness)
    return _normalize_topology(field)


def _build_single_island_topology(terrain: TerrainGenerator, constraints: dict, ruggedness: float) -> np.ndarray:
    core = _continent_component(terrain, "center", 0.34, [("west", 0.12, 0.52), ("east", 0.14, 0.58), ("south", 0.12, 0.46)])
    bays = (
        terrain._create_location_mask("west", radius=0.16, sigma=0.42) * 0.24
        + terrain._create_location_mask("east", radius=0.14, sigma=0.4) * 0.2
        + terrain._create_location_mask("north", radius=0.12, sigma=0.34) * 0.14
    )
    field = core * 1.22 - bays
    field = _apply_mountain_topology(terrain, field, constraints, ruggedness)
    return _normalize_topology(field)


def _build_plan_structural_topology(terrain: TerrainGenerator, plan: dict, ruggedness: float) -> np.ndarray | None:
    topology_intent = plan.get("topology_intent") or {}
    intent_field = _build_topology_intent_field(terrain, topology_intent, ruggedness)
    if intent_field is not None:
        return intent_field

    continents = list(plan.get("continents") or [])
    mountains = list(plan.get("mountains") or [])
    peninsulas = list(plan.get("peninsulas") or [])
    island_chains = list(plan.get("island_chains") or [])
    inland_seas = list(plan.get("inland_seas") or [])
    water_bodies = list(plan.get("water_bodies") or [])
    regional_relations = list(plan.get("regional_relations") or [])
    constraints = plan.get("constraints") or {}

    if not any([continents, mountains, peninsulas, island_chains, inland_seas, water_bodies, regional_relations]):
        return None

    field = np.full((terrain.height, terrain.width), -0.45, dtype=np.float32)

    if continents:
        continent_masks = []
        for continent in continents:
            position = str(continent.get("position", "center"))
            size = float(continent.get("size", 0.38))
            component = _plan_continent_component(terrain, position, size)
            continent_masks.append(component)
            field += component * 1.25
        field += np.maximum.reduce(continent_masks) * 0.28

    for chain in island_chains:
        field += _plan_island_chain_component(
            terrain,
            str(chain.get("position", "center")),
            float(chain.get("density", 0.66)),
        ) * 0.9

    for peninsula in peninsulas:
        field += _plan_peninsula_component(
            terrain,
            str(peninsula.get("location", "west")),
            float(peninsula.get("size", 0.18)),
        ) * 1.05

    for inland_sea in inland_seas:
        field -= _plan_inland_sea_component(
            terrain,
            str(inland_sea.get("position", "center")),
            str(inland_sea.get("connection", "strait")),
        ) * 1.35

    for water_body in water_bodies:
        field -= _plan_water_body_component(
            terrain,
            body_type=str(water_body.get("type", "ocean")),
            position=str(water_body.get("position", "center")),
            coverage=float(water_body.get("coverage", 0.25)),
            connection=str(water_body.get("connection", "")),
        )

    field = _apply_mountain_topology(terrain, field, {"mountains": mountains or constraints.get("mountains") or []}, ruggedness)
    field = _apply_regional_relations(terrain, field, regional_relations)
    return _normalize_topology(field)


def build_hard_topology(
    terrain: TerrainGenerator,
    layout_template: str,
    sea_style: str,
    plan: dict,
    ruggedness: float,
) -> np.ndarray | None:
    constraints = plan.get("constraints") or {}
    explicit = _build_plan_structural_topology(terrain, plan, ruggedness)
    if explicit is not None:
        return explicit

    layout_field: np.ndarray | None = None
    if layout_template == "split_east_west":
        layout_field = _build_split_east_west_topology(terrain, constraints, ruggedness)
    elif layout_template == "mediterranean" or sea_style == "inland":
        layout_field = _build_inland_sea_topology(terrain, constraints, ruggedness)
    elif layout_template == "single_island":
        layout_field = _build_single_island_topology(terrain, constraints, ruggedness)

    return layout_field


def shape_world_profile(terrain: TerrainGenerator, elevation: np.ndarray, plan: dict) -> tuple[np.ndarray, bool]:
    profile = plan.get("profile") or {}
    constraints = plan.get("constraints") or {}
    explicit_geometry = has_explicit_plan_geometry(plan)
    land_ratio = float(profile.get("land_ratio", 0.44))
    ruggedness = float(profile.get("ruggedness", 0.55))
    coast_complexity = float(profile.get("coast_complexity", 0.5))
    island_factor = float(profile.get("island_factor", 0.25))
    layout_template = str(profile.get("layout_template", "default"))
    sea_style = str(profile.get("sea_style", "open"))
    topology_intent = plan.get("topology_intent") or {}

    macro = terrain._fbm(scale=145.0, octaves=4, persistence=0.56, lacunarity=2.05, offset=77.0)
    ridges = terrain._ridged_noise(scale=54.0, octaves=4, offset=31.0)
    archipelago = terrain._fbm(scale=64.0, octaves=4, persistence=0.52, lacunarity=2.15, offset=91.0)
    coastline_noise = terrain._fbm(scale=88.0, octaves=3, persistence=0.58, lacunarity=2.05, offset=143.0)

    hard_topology = build_hard_topology(terrain, layout_template, sea_style, plan, ruggedness)
    uses_hard_topology = hard_topology is not None

    shaped = elevation.astype(np.float32)
    if hard_topology is not None:
        shaped = shaped * 0.14 + hard_topology * 0.86
    else:
        shaped = apply_constraint_topology(terrain, shaped, constraints, coast_complexity, ruggedness)

    if not explicit_geometry and not topology_intent:
        shaped = apply_layout_template(terrain, shaped, layout_template, sea_style, constraints, coast_complexity)
    shaped = apply_constraint_topology(terrain, shaped, constraints, coast_complexity, ruggedness)
    shaped += (macro * 2.0 - 1.0) * (0.14 + coast_complexity * 0.12)
    shaped += (ridges * 2.0 - 1.0) * (0.08 + ruggedness * 0.24)
    shaped += (archipelago * 2.0 - 1.0) * island_factor * 0.18
    shaped += (coastline_noise * 2.0 - 1.0) * coast_complexity * 0.1
    shaped = apply_water_enforcement(terrain, shaped, plan)
    shaped = terrain._rebalance_sea_level(shaped, target_ocean=float(np.clip(1.0 - land_ratio, 0.28, 0.78)))
    shaped = np.tanh(shaped * (0.9 + ruggedness * 0.55)).astype(np.float32)
    shaped = terrain.apply_coastal_smoothing(shaped, sea_level=0.0, smoothing_radius=max(3, int(8 - coast_complexity * 4)))
    shaped = enforce_required_water_gaps(terrain, shaped, plan)
    shaped = enforce_topology_components(terrain, shaped, plan)
    return np.clip(shaped, -1.0, 1.0).astype(np.float32), uses_hard_topology


__all__ = [
    "apply_constraint_topology",
    "apply_layout_template",
    "apply_planet_semantic_shape",
    "apply_water_enforcement",
    "build_hard_topology",
    "constraints_for_refinement",
    "enforce_required_water_gaps",
    "enforce_topology_components",
    "has_explicit_plan_geometry",
    "normalize_generation_plan",
    "planet_constraints_from_plan",
    "rebalance_and_scale_elevation",
    "shape_world_profile",
    "gaussian_smooth",
]
