from __future__ import annotations

import numpy as np
from PIL import Image

from app.core.climate import ClimateSimulator
from app.core.terrain import TerrainGenerator
from app.core.terrain_modular import ModularTerrainGenerator


def render_world(plan: dict, width: int, height: int, seed: int) -> tuple[dict[str, np.ndarray], Image.Image]:
    backend = str((plan or {}).get("generation_backend") or "gaussian_voronoi")
    terrain = TerrainGenerator(width=width, height=height, seed=seed)
    if backend == "modular":
        elevation = ModularTerrainGenerator(width=width, height=height, seed=seed).generate(plan or {})
    else:
        elevation = terrain.generate()

    constraints = (plan or {}).get("constraints") or {}
    profile = (plan or {}).get("profile") or {}
    elevation, uses_hard_topology = _shape_world_profile(terrain, elevation, plan or {})
    if constraints:
        refinement_constraints = _constraints_for_refinement(constraints, profile, uses_hard_topology)
        if refinement_constraints and not uses_hard_topology:
            elevation = terrain.apply_constraints(elevation, refinement_constraints)
            elevation = terrain.reinforce_constraints(elevation, refinement_constraints, blend=0.66)

    filled = elevation
    latitude = np.linspace(90.0, -90.0, height, dtype=np.float32).reshape(height, 1)
    lat_grid = np.repeat(latitude, width, axis=1)
    climate = ClimateSimulator(elev=((filled + 1.0) * 2500.0).astype(np.float32), lat_grid=lat_grid).run(
        wind_direction=profile.get("wind_direction", "westerly"),
        moisture_factor=float(profile.get("moisture", 1.0)),
        temperature_bias=float(profile.get("temperature_bias", 0.0)),
    )
    arrays = {
        "elevation": filled.astype(np.float32),
        "temperature": climate["temperature"].astype(np.float32),
        "precipitation": climate["precipitation"].astype(np.float32),
        "biome": climate["biome"].astype(np.int16),
    }
    preview = render_preview_from_arrays(arrays, profile)
    return arrays, preview


def render_preview_from_arrays(arrays: dict[str, np.ndarray], profile: dict | None = None) -> Image.Image:
    return _render_preview_image(
        arrays["elevation"],
        arrays["biome"],
        arrays["temperature"],
        arrays["precipitation"],
        profile or {},
    )


def _shape_world_profile(terrain: TerrainGenerator, elevation: np.ndarray, plan: dict) -> tuple[np.ndarray, bool]:
    profile = plan.get("profile") or {}
    constraints = plan.get("constraints") or {}
    explicit_geometry = _has_explicit_plan_geometry(plan)
    land_ratio = float(profile.get("land_ratio", 0.44))
    ruggedness = float(profile.get("ruggedness", 0.55))
    coast_complexity = float(profile.get("coast_complexity", 0.5))
    island_factor = float(profile.get("island_factor", 0.25))
    layout_template = str(profile.get("layout_template", "default"))
    sea_style = str(profile.get("sea_style", "open"))

    macro = terrain._fbm(scale=145.0, octaves=4, persistence=0.56, lacunarity=2.05, offset=77.0)
    ridges = terrain._ridged_noise(scale=54.0, octaves=4, offset=31.0)
    archipelago = terrain._fbm(scale=64.0, octaves=4, persistence=0.52, lacunarity=2.15, offset=91.0)
    coastline_noise = terrain._fbm(scale=88.0, octaves=3, persistence=0.58, lacunarity=2.05, offset=143.0)

    hard_topology = _build_hard_topology(terrain, layout_template, sea_style, plan, ruggedness)
    uses_hard_topology = hard_topology is not None

    shaped = elevation.astype(np.float32)
    if hard_topology is not None:
        shaped = shaped * 0.14 + hard_topology * 0.86
    else:
        shaped = _apply_constraint_topology(terrain, shaped, constraints, coast_complexity, ruggedness)

    if not explicit_geometry:
        shaped = _apply_layout_template(terrain, shaped, layout_template, sea_style, constraints, coast_complexity)
    shaped = _apply_constraint_topology(terrain, shaped, constraints, coast_complexity, ruggedness)
    shaped += (macro * 2.0 - 1.0) * (0.14 + coast_complexity * 0.12)
    shaped += (ridges * 2.0 - 1.0) * (0.08 + ruggedness * 0.24)
    shaped += (archipelago * 2.0 - 1.0) * island_factor * 0.18
    shaped += (coastline_noise * 2.0 - 1.0) * coast_complexity * 0.1
    shaped = _apply_water_enforcement(terrain, shaped, plan)
    shaped = terrain._rebalance_sea_level(shaped, target_ocean=float(np.clip(1.0 - land_ratio, 0.28, 0.78)))
    shaped = np.tanh(shaped * (0.9 + ruggedness * 0.55)).astype(np.float32)
    shaped = terrain.apply_coastal_smoothing(shaped, sea_level=0.0, smoothing_radius=max(3, int(8 - coast_complexity * 4)))
    shaped = _enforce_required_water_gaps(terrain, shaped, plan)
    return np.clip(shaped, -1.0, 1.0).astype(np.float32), uses_hard_topology


def _constraints_for_refinement(constraints: dict, profile: dict, uses_hard_topology: bool) -> dict:
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


def _apply_layout_template(
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


def _build_hard_topology(
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


def _has_explicit_plan_geometry(plan: dict) -> bool:
    return any(
        plan.get(key)
        for key in ("continents", "mountains", "peninsulas", "island_chains", "inland_seas", "water_bodies", "regional_relations")
    )


def _build_plan_structural_topology(terrain: TerrainGenerator, plan: dict, ruggedness: float) -> np.ndarray | None:
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


def _build_split_east_west_topology(terrain: TerrainGenerator, constraints: dict, ruggedness: float) -> np.ndarray:
    west = _continent_component(terrain, "west", 0.46, [("northwest", 0.2, 0.58), ("southwest", 0.22, 0.66)])
    east = _continent_component(terrain, "east", 0.46, [("northeast", 0.2, 0.62), ("southeast", 0.18, 0.54)])
    west *= _longitudinal_gate(terrain, "west")
    east *= _longitudinal_gate(terrain, "east")

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
    north_arc = _continent_component(terrain, "north", 0.34, [("northwest", 0.26, 0.88), ("northeast", 0.26, 0.88)])
    south_arc = _continent_component(terrain, "south", 0.34, [("southwest", 0.26, 0.88), ("southeast", 0.26, 0.88)])
    west_wall = _continent_component(terrain, "west", 0.18, [("northwest", 0.18, 0.55), ("southwest", 0.18, 0.55)])
    east_wall = _continent_component(terrain, "east", 0.18, [("northeast", 0.18, 0.55), ("southeast", 0.18, 0.55)])

    basin = np.maximum(
        terrain._elliptic_gaussian(0.5, 0.5, 0.14, 0.24, 0.0),
        terrain._create_location_mask("center", radius=0.16, sigma=0.38),
    )
    basin = np.maximum(basin, terrain._elliptic_gaussian(0.5, 0.5, 0.2, 0.3, 0.0) * 0.88)
    west_outlet = terrain._elliptic_gaussian(0.5, 0.18, 0.05, 0.08, 0.0)
    east_outlet = terrain._elliptic_gaussian(0.5, 0.82, 0.05, 0.08, 0.0)
    north_cut = terrain._elliptic_gaussian(0.34, 0.5, 0.06, 0.18, 0.0)
    south_cut = terrain._elliptic_gaussian(0.66, 0.5, 0.06, 0.18, 0.0)
    enclosure = np.maximum.reduce([north_arc, south_arc, west_wall, east_wall]) * 1.2
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


def _axis_aligned_ocean_barrier(terrain: TerrainGenerator, axis: str, center: float, width: float) -> np.ndarray:
    if axis == "vertical":
        x = terrain._x_norm
        barrier = np.exp(-((x - center) ** 2) / max(width**2, 1e-4))
        return barrier.astype(np.float32)
    y = terrain._y_norm
    barrier = np.exp(-((y - center) ** 2) / max(width**2, 1e-4))
    return barrier.astype(np.float32)


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


def _apply_water_enforcement(terrain: TerrainGenerator, elevation: np.ndarray, plan: dict) -> np.ndarray:
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


def _enforce_required_water_gaps(terrain: TerrainGenerator, elevation: np.ndarray, plan: dict) -> np.ndarray:
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
        split_barrier = np.maximum(
            split_barrier,
            _axis_aligned_ocean_barrier(terrain, "vertical", center=0.5, width=0.075),
        )
        split_barrier = np.maximum(
            split_barrier,
            terrain._elliptic_gaussian(0.5, 0.5, 0.5, 0.095, 0.0),
        )
    elif layout_template == "split_north_south" or _requires_opposite_side_split(relations, "north", "south"):
        split_barrier = np.maximum(
            split_barrier,
            _axis_aligned_ocean_barrier(terrain, "horizontal", center=0.5, width=0.075),
        )
        split_barrier = np.maximum(
            split_barrier,
            terrain._elliptic_gaussian(0.5, 0.5, 0.095, 0.5, 0.0),
        )

    if "center" in sea_zones and {"west", "east"} <= positions:
        split_barrier = np.maximum(
            split_barrier,
            _axis_aligned_ocean_barrier(terrain, "vertical", center=0.5, width=0.08),
        )
    if "center" in sea_zones and {"north", "south"} <= positions and sea_style != "inland":
        split_barrier = np.maximum(
            split_barrier,
            _axis_aligned_ocean_barrier(terrain, "horizontal", center=0.5, width=0.08),
        )

    inland_basin = np.zeros_like(shaped)
    if sea_style == "inland" or layout_template == "mediterranean":
        inland_basin = np.maximum(
            inland_basin,
            terrain._elliptic_gaussian(0.5, 0.5, 0.2, 0.32, 0.0),
        )
        inland_basin = np.maximum(
            inland_basin,
            terrain._elliptic_gaussian(0.5, 0.5, 0.13, 0.4, 0.0) * 0.9,
        )
        if {"north", "south"} <= positions:
            inland_basin = np.maximum(
                inland_basin,
                terrain._elliptic_gaussian(0.5, 0.5, 0.16, 0.36, 0.0),
            )

    shaped = _force_water_mask(shaped, split_barrier, base_depth=-0.34, mask_threshold=0.52)
    shaped = _force_water_mask(shaped, inland_basin, base_depth=-0.3, mask_threshold=0.44)
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


def _apply_mountain_topology(terrain: TerrainGenerator, field: np.ndarray, constraints: dict, ruggedness: float) -> np.ndarray:
    result = field.astype(np.float32)
    for mountain in constraints.get("mountains") or []:
        location = str(mountain.get("location", "center"))
        height = float(mountain.get("height", 0.8))
        chain = terrain._create_mountain_chain_mask(location)
        result += gaussian_smooth(chain * (0.18 + height * 0.24 + ruggedness * 0.08))
    return result


def _normalize_topology(field: np.ndarray) -> np.ndarray:
    normalized = np.tanh(field * 1.35).astype(np.float32)
    return gaussian_smooth(normalized)


def _apply_constraint_topology(
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


def gaussian_smooth(values: np.ndarray) -> np.ndarray:
    from scipy.ndimage import gaussian_filter

    return gaussian_filter(values.astype(np.float32), sigma=1.2).astype(np.float32)


def _render_preview_image(
    elevation: np.ndarray,
    biome: np.ndarray,
    temperature: np.ndarray,
    precipitation: np.ndarray,
    profile: dict,
) -> Image.Image:
    ocean = elevation < 0
    land = ~ocean

    palette = np.zeros((*elevation.shape, 3), dtype=np.float32)
    normalized = np.clip((elevation + 1.0) / 2.0, 0.0, 1.0)
    depth = np.clip((-elevation) / max(float(np.max(-elevation[ocean])) if np.any(ocean) else 1.0, 1e-6), 0.0, 1.0)
    moisture = np.clip(precipitation / max(float(np.max(precipitation)), 1.0), 0.0, 1.0)
    warmth = np.clip((temperature + 20.0) / 55.0, 0.0, 1.0)

    ocean_shallow = np.array([44, 127, 184], dtype=np.float32)
    ocean_deep = np.array([8, 46, 97], dtype=np.float32)
    palette[ocean] = ocean_shallow * (1.0 - depth[ocean, None]) + ocean_deep * depth[ocean, None]

    if profile.get("palette_hint") == "frozen":
        palette[ocean] = palette[ocean] * 0.72 + np.array([168, 200, 220], dtype=np.float32) * 0.28
    elif profile.get("palette_hint") == "tropical":
        palette[ocean] = palette[ocean] * 0.78 + np.array([42, 168, 198], dtype=np.float32) * 0.22

    beach = land & (elevation < 0.06)
    lowland = land & (elevation >= 0.06) & (elevation < 0.28)
    upland = land & (elevation >= 0.28) & (elevation < 0.58)
    alpine = land & (elevation >= 0.58)

    palette[beach] = np.array([220, 202, 155], dtype=np.float32)
    palette[lowland & (biome == 1)] = np.array([204, 170, 102], dtype=np.float32)
    palette[lowland & (biome == 2)] = np.array([153, 176, 96], dtype=np.float32)
    palette[lowland & (biome == 3)] = np.array([84, 140, 78], dtype=np.float32)
    palette[lowland & (biome == 4)] = np.array([126, 147, 112], dtype=np.float32)
    palette[lowland & (biome == 5)] = np.array([232, 238, 242], dtype=np.float32)

    palette[upland & (biome == 1)] = np.array([168, 128, 86], dtype=np.float32)
    palette[upland & (biome == 2)] = np.array([126, 151, 88], dtype=np.float32)
    palette[upland & (biome == 3)] = np.array([58, 108, 70], dtype=np.float32)
    palette[upland & (biome == 4)] = np.array([124, 134, 126], dtype=np.float32)
    palette[upland & (biome == 5)] = np.array([238, 242, 247], dtype=np.float32)

    palette[alpine] = np.array([118, 112, 106], dtype=np.float32)
    snowcaps = land & ((warmth < 0.22) | (elevation > 0.74))
    palette[snowcaps] = palette[snowcaps] * 0.28 + np.array([244, 246, 250], dtype=np.float32) * 0.72

    dry_bias = np.clip((0.52 - moisture) * 1.6, 0.0, 1.0)
    lush_bias = np.clip((moisture - 0.5) * 1.4, 0.0, 1.0)
    palette[land] = palette[land] * (1.0 - dry_bias[land, None] * 0.18) + np.array([205, 176, 113], dtype=np.float32) * (
        dry_bias[land, None] * 0.18
    )
    palette[land] = palette[land] * (1.0 - lush_bias[land, None] * 0.18) + np.array([73, 133, 82], dtype=np.float32) * (
        lush_bias[land, None] * 0.18
    )

    grad_y, grad_x = np.gradient(elevation.astype(np.float32))
    light = np.array([-0.7, -0.45, 0.55], dtype=np.float32)
    nx = -grad_x * 2.4
    ny = -grad_y * 2.4
    nz = np.ones_like(elevation, dtype=np.float32)
    norm = np.sqrt(nx * nx + ny * ny + nz * nz)
    nx /= norm
    ny /= norm
    nz /= norm
    hillshade = np.clip(nx * light[0] + ny * light[1] + nz * light[2], 0.0, 1.0)
    ambient = 0.58 + hillshade * 0.52
    palette *= ambient[..., None]

    coast_band = np.abs(elevation) < 0.03
    palette[coast_band] = palette[coast_band] * 0.45 + np.array([240, 229, 192], dtype=np.float32) * 0.55

    image = np.clip(palette, 0, 255).astype(np.uint8)
    return Image.fromarray(image, mode="RGB")
