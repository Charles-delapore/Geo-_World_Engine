from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from app.config import settings
from app.core.llm_parser import parse_with_rag
from app.storage.artifact_repo import ArtifactRepository
from app.storage import artifact_repo as artifact_repo_module
from app.api.routes.maps import CreateMapRequest
from app.workers.planner_worker import build_world_plan
from app.workers.render_worker import render_world


def _shoreline_span_std(mask: np.ndarray) -> float:
    spans: list[float] = []
    for row in mask:
        cols = np.flatnonzero(row)
        if cols.size >= 2:
            spans.append(float(cols[-1] - cols[0]))
    return float(np.std(spans)) if spans else 0.0


def _mirrored_half_difference(mask: np.ndarray) -> float:
    width = mask.shape[1]
    half = width // 2
    left = mask[:, :half]
    right = np.fliplr(mask[:, width - half :])
    return float(np.mean(left != right))


def _count_large_land_components(elevation: np.ndarray, min_cells: int = 160) -> int:
    from scipy.ndimage import label

    labels, count = label((elevation > 0.0).astype(np.int8))
    large = 0
    for component_id in range(1, count + 1):
        if int(np.sum(labels == component_id)) >= min_cells:
            large += 1
    return large


def _count_large_water_components(mask: np.ndarray, min_cells: int = 120) -> int:
    from scipy.ndimage import label

    labels, count = label(mask.astype(np.int8))
    large = 0
    for component_id in range(1, count + 1):
        if int(np.sum(labels == component_id)) >= min_cells:
            large += 1
    return large


def test_parse_with_rag_emits_beta_module_fields() -> None:
    plan = parse_with_rag("一东一西两个被海隔开的大陆，东北角有高山。")

    assert plan["generation_backend"] == "gaussian_voronoi"
    assert plan["water_bodies"]
    assert any(item["relation"] == "separated_by_water" for item in plan["regional_relations"])


def test_parse_with_rag_emits_single_island_topology_intent() -> None:
    plan = parse_with_rag("一座四面环海的岛屿。")

    assert plan["topology_intent"]["kind"] == "single_island"
    assert plan["topology_intent"]["exact_landmass_count"] == 1
    assert len(plan["continents"]) == 1
    assert plan["continents"][0]["position"] == "center"
    assert plan["island_chains"] == []


def test_parse_with_rag_distinguishes_archipelago_and_peninsula() -> None:
    archipelago = parse_with_rag("中央是一片群岛。")
    peninsula = parse_with_rag("东侧伸出一条半岛。")

    assert archipelago["profile"]["layout_template"] == "archipelago"
    assert archipelago["island_chains"]
    assert archipelago["topology_intent"]["kind"] == "archipelago_chain"

    assert peninsula["peninsulas"]
    assert peninsula["island_chains"] == []
    assert peninsula["topology_intent"]["kind"] == "peninsula_coast"


def test_parse_with_rag_extracts_topology_modifiers() -> None:
    elongated_island = parse_with_rag("一座狭长的四面环海岛屿。")
    north_south_island = parse_with_rag("一座南北向狭长的四面环海岛屿。")
    broad_rift = parse_with_rag("东西大陆中间被宽阔的大海隔开。")
    dense_archipelago = parse_with_rag("中央是一片密集的群岛。")
    branched_inland_sea = parse_with_rag("中间是一片多海湾的内海。")
    rift_inland_sea = parse_with_rag("中间是一片裂谷式内海。")

    assert elongated_island["topology_intent"]["modifiers"]["shape_bias"] == "elongated"
    assert north_south_island["topology_intent"]["modifiers"]["shape_axis"] == "north_south"
    assert broad_rift["topology_intent"]["modifiers"]["rift_width"] == "broad"
    assert dense_archipelago["topology_intent"]["modifiers"]["island_density"] == "dense"
    assert branched_inland_sea["topology_intent"]["modifiers"]["basin_shape"] == "branched"
    assert rift_inland_sea["topology_intent"]["modifiers"]["basin_style"] == "rift"


def test_parse_with_rag_can_select_modular_backend() -> None:
    plan = parse_with_rag("生成一个模块化程序化大陆，东北有山脊高原，中央有狭长海峡。")

    assert plan["generation_backend"] == "modular"
    assert plan["module_sequence"]
    assert any(item["module"] == "plateau" for item in plan["module_sequence"])


def test_parse_with_rag_enriches_middle_sea_split_prompt() -> None:
    plan = parse_with_rag("Generate two major continents separated by sea through the middle of the map.")
    positions = {item["position"] for item in plan["continents"]}

    assert {"west", "east"} <= positions
    assert "center" in plan["constraints"]["sea_zones"]
    assert plan["topology_intent"]["kind"] == "two_continents_with_rift_sea"


def test_parse_with_rag_enriches_inland_sea_enclosure_prompt() -> None:
    plan = parse_with_rag("Generate a world with a broad inland sea in the middle, enclosed by land to the north and south.")
    positions = {item["position"] for item in plan["continents"]}

    assert {"north", "south"} <= positions
    assert plan["profile"]["sea_style"] == "inland"
    assert plan["inland_seas"]
    assert plan["topology_intent"]["kind"] == "central_enclosed_inland_sea"


def test_render_world_accepts_beta_module_structures() -> None:
    plan = {
        "generation_backend": "gaussian_voronoi",
        "constraints": {
            "continents": [{"position": "west", "size": 0.4}, {"position": "east", "size": 0.4}],
            "mountains": [{"location": "northeast", "height": 0.9}],
            "sea_zones": ["center"],
            "river_sources": ["northeast"],
        },
        "profile": {
            "land_ratio": 0.46,
            "ruggedness": 0.8,
            "coast_complexity": 0.7,
            "island_factor": 0.2,
            "moisture": 1.0,
            "temperature_bias": 0.0,
            "wind_direction": "westerly",
            "palette_hint": "temperate",
            "layout_template": "split_east_west",
            "sea_style": "open",
        },
        "continents": [{"position": "west", "size": 0.4}, {"position": "east", "size": 0.4}],
        "mountains": [{"location": "northeast", "height": 0.9, "orientation": "arc"}],
        "water_bodies": [{"type": "ocean", "position": "center", "coverage": 0.4}],
        "regional_relations": [{"relation": "separated_by_water", "subject": "west", "object": "east", "strength": 0.92}],
        "module_sequence": [],
    }

    arrays, preview = render_world(plan, width=128, height=64, seed=42)

    assert preview.size == (128, 64)
    assert arrays["elevation"].shape == (64, 128)
    assert arrays["temperature"].shape == (64, 128)
    assert np.isfinite(arrays["elevation"]).all()


def test_render_world_keeps_split_continents_separated_by_water() -> None:
    plan = {
        "generation_backend": "gaussian_voronoi",
        "profile": {
            "land_ratio": 0.46,
            "ruggedness": 0.82,
            "coast_complexity": 0.72,
            "island_factor": 0.18,
            "moisture": 1.0,
            "temperature_bias": 0.0,
            "wind_direction": "westerly",
            "layout_template": "split_east_west",
            "sea_style": "open",
        },
        "constraints": {
            "continents": [{"position": "west", "size": 0.4}, {"position": "east", "size": 0.4}],
            "mountains": [],
            "sea_zones": ["center"],
            "river_sources": [],
        },
        "continents": [{"position": "west", "size": 0.4}, {"position": "east", "size": 0.4}],
        "water_bodies": [{"type": "ocean", "position": "center", "coverage": 0.42}],
        "regional_relations": [{"relation": "separated_by_water", "subject": "west", "object": "east", "strength": 0.96}],
    }

    arrays, _ = render_world(plan, width=160, height=96, seed=11)
    center_band = arrays["elevation"][:, 74:86]
    top_slice = arrays["elevation"][:28, 74:86]
    bottom_slice = arrays["elevation"][68:, 74:86]
    central_window = arrays["elevation"][:, 58:102] < 0.0

    assert float(np.mean(center_band < 0.0)) > 0.78
    assert float(np.mean(top_slice < 0.0)) > 0.72
    assert float(np.mean(bottom_slice < 0.0)) > 0.72
    assert _shoreline_span_std(central_window) > 1.2
    assert _mirrored_half_difference(central_window) > 0.025


def test_render_world_keeps_central_inland_sea_open() -> None:
    plan = {
        "generation_backend": "gaussian_voronoi",
        "profile": {
            "land_ratio": 0.48,
            "ruggedness": 0.75,
            "coast_complexity": 0.68,
            "island_factor": 0.12,
            "moisture": 1.0,
            "temperature_bias": 0.0,
            "wind_direction": "westerly",
            "layout_template": "mediterranean",
            "sea_style": "inland",
        },
        "constraints": {
            "continents": [{"position": "north", "size": 0.34}, {"position": "south", "size": 0.34}],
            "mountains": [],
            "sea_zones": ["center"],
            "river_sources": [],
        },
        "continents": [{"position": "north", "size": 0.34}, {"position": "south", "size": 0.34}],
        "inland_seas": [{"position": "center", "connection": "strait"}],
        "water_bodies": [{"type": "inland_sea", "position": "center", "coverage": 0.26, "connection": "strait"}],
    }

    arrays, _ = render_world(plan, width=160, height=96, seed=17)
    central_basin = arrays["elevation"][32:64, 52:108]
    centerline = arrays["elevation"][38:58, 60:100]
    basin_window = arrays["elevation"][24:72, 38:122].T < 0.0

    assert float(np.mean(central_basin < 0.0)) > 0.7
    assert float(np.mean(centerline < 0.0)) > 0.82
    assert _shoreline_span_std(basin_window) > 1.2
    assert _mirrored_half_difference(basin_window) > 0.06
    assert _count_large_water_components(central_basin < 0.0) == 1


def test_render_world_supports_modular_backend() -> None:
    plan = {
        "generation_backend": "modular",
        "profile": {
            "land_ratio": 0.5,
            "layout_template": "single_island",
            "sea_style": "open",
            "wind_direction": "westerly",
            "moisture": 1.0,
            "temperature_bias": 0.0,
        },
        "constraints": {
            "continents": [{"position": "center", "size": 0.42}],
            "mountains": [{"location": "northeast", "height": 0.88}],
            "sea_zones": ["center"],
            "river_sources": [],
        },
        "module_sequence": [
            {"module": "continent", "params": {"position": "center", "size": 0.42, "height": 0.92, "operation": "add"}},
            {"module": "ridge", "params": {"location": "northeast", "height": 0.88, "operation": "add"}},
            {"module": "plateau", "params": {"position": "north", "height": 0.2, "radius_y": 0.14, "radius_x": 0.24, "operation": "add"}},
            {"module": "water_body", "params": {"position": "south", "coverage": 0.24, "depth": 0.76, "operation": "subtract"}},
            {"module": "smooth", "params": {"sigma": 1.2, "operation": "replace"}},
        ],
    }

    arrays, preview = render_world(plan, width=96, height=64, seed=7)

    assert preview.size == (96, 64)
    assert arrays["elevation"].shape == (64, 96)
    assert np.nanmax(arrays["elevation"]) > 0.1
    assert np.nanmin(arrays["elevation"]) < -0.1


def test_render_world_single_island_topology_keeps_dominant_center_mass() -> None:
    plan = parse_with_rag("一座四面环海的岛屿。")

    arrays, _ = render_world(plan, width=160, height=96, seed=23)
    center_mass = arrays["elevation"][24:72, 46:114]
    edge_ocean = np.concatenate(
        [
            arrays["elevation"][:14, :].ravel(),
            arrays["elevation"][-14:, :].ravel(),
            arrays["elevation"][:, :18].ravel(),
            arrays["elevation"][:, -18:].ravel(),
        ]
    )

    assert float(np.mean(center_mass > 0.0)) > 0.6
    assert float(np.mean(edge_ocean < 0.0)) > 0.8
    assert _count_large_land_components(arrays["elevation"]) == 1


def test_render_world_rift_topology_keeps_two_dominant_landmasses() -> None:
    plan = parse_with_rag("东西大陆中间被海隔开。")

    arrays, _ = render_world(plan, width=160, height=96, seed=29)
    west_midland = arrays["elevation"][34:62, 14:56]
    east_midland = arrays["elevation"][34:62, 104:146]

    assert _count_large_land_components(arrays["elevation"]) == 2
    assert float(np.mean(west_midland > 0.0)) > 0.5
    assert float(np.mean(east_midland > 0.0)) > 0.5


def test_render_world_archipelago_topology_keeps_multiple_islands() -> None:
    plan = parse_with_rag("中央是一片群岛。")

    arrays, _ = render_world(plan, width=160, height=96, seed=31)

    assert _count_large_land_components(arrays["elevation"], min_cells=20) >= 3


def test_render_world_topology_modifiers_shift_geometry() -> None:
    east_west_island, _ = render_world(parse_with_rag("一座东西向狭长的四面环海岛屿。"), width=160, height=96, seed=41)
    north_south_island, _ = render_world(parse_with_rag("一座南北向狭长的四面环海岛屿。"), width=160, height=96, seed=41)
    narrow_rift, _ = render_world(parse_with_rag("东西大陆中间被狭窄的海隔开。"), width=160, height=96, seed=43)
    broad_rift, _ = render_world(parse_with_rag("东西大陆中间被宽阔的大海隔开。"), width=160, height=96, seed=43)
    sparse_archipelago, _ = render_world(parse_with_rag("中央是一片稀疏的群岛。"), width=160, height=96, seed=47)
    dense_archipelago, _ = render_world(parse_with_rag("中央是一片密集的群岛。"), width=160, height=96, seed=47)
    compact_inland, _ = render_world(parse_with_rag("中间是一片紧凑的内海。"), width=160, height=96, seed=53)
    broad_inland, _ = render_world(parse_with_rag("中间是一片宽阔的内海。"), width=160, height=96, seed=53)
    branched_inland, _ = render_world(parse_with_rag("中间是一片多海湾的内海。"), width=160, height=96, seed=53)
    rift_inland, _ = render_world(parse_with_rag("中间是一片裂谷式内海。"), width=160, height=96, seed=53)

    east_west_land = east_west_island["elevation"] > 0.0
    north_south_land = north_south_island["elevation"] > 0.0
    ew_center_row = np.flatnonzero(east_west_land[48])
    ew_center_col = np.flatnonzero(east_west_land[:, 80])
    ns_center_row = np.flatnonzero(north_south_land[48])
    ns_center_col = np.flatnonzero(north_south_land[:, 80])
    east_west_row_span = float(ew_center_row[-1] - ew_center_row[0] + 1)
    east_west_col_span = float(ew_center_col[-1] - ew_center_col[0] + 1)
    north_south_row_span = float(ns_center_row[-1] - ns_center_row[0] + 1)
    north_south_col_span = float(ns_center_col[-1] - ns_center_col[0] + 1)
    narrow_window = narrow_rift["elevation"][:, 64:96]
    broad_window = broad_rift["elevation"][:, 64:96]
    compact_basin = compact_inland["elevation"][28:68, 48:112] < 0.0
    broad_basin = broad_inland["elevation"][28:68, 48:112] < 0.0
    branched_basin = branched_inland["elevation"][24:72, 38:122].T < 0.0

    assert east_west_row_span > east_west_col_span + 4
    assert float(np.mean(broad_window < 0.0)) > float(np.mean(narrow_window < 0.0)) + 0.05
    assert _count_large_land_components(dense_archipelago["elevation"], min_cells=20) >= _count_large_land_components(
        sparse_archipelago["elevation"],
        min_cells=20,
    )
    assert float(np.mean(broad_basin)) > float(np.mean(compact_basin)) + 0.05
    assert _shoreline_span_std(branched_basin) > _shoreline_span_std(compact_basin.T) + 0.6


def test_render_world_peninsula_topology_keeps_connected_spur() -> None:
    plan = parse_with_rag("东侧伸出一条半岛。")

    arrays, _ = render_world(plan, width=160, height=96, seed=37)
    east_spur = arrays["elevation"][28:70, 98:150]

    assert float(np.mean(east_spur > 0.0)) > 0.35
    assert _count_large_land_components(arrays["elevation"], min_cells=140) >= 1


def test_build_world_plan_respects_explicit_backend_override() -> None:
    modular_plan = build_world_plan(
        "Generate a simple world with two continents and a central sea.",
        {"generation_backend": "modular"},
    )
    gaussian_plan = build_world_plan(
        "生成一个模块化程序化大陆，东北有山脊高原，中央有狭长海峡。",
        {"generation_backend": "gaussian_voronoi"},
    )

    assert modular_plan.generation_backend == "modular"
    assert modular_plan.module_sequence
    assert gaussian_plan.generation_backend == "gaussian_voronoi"
    assert gaussian_plan.module_sequence == []


def test_build_world_plan_accepts_explicit_module_sequence() -> None:
    requested_modules = [
        {"module": "continent", "params": {"position": "center", "size": 0.44, "height": 0.94, "operation": "add"}},
        {"module": "ridge", "params": {"location": "north", "height": 0.86, "operation": "add"}},
        {"module": "water_body", "params": {"position": "south", "coverage": 0.22, "depth": 0.8, "operation": "subtract"}},
    ]

    plan = build_world_plan(
        "A simple world.",
        {"module_sequence": requested_modules},
    )
    ignored = build_world_plan(
        "A modular world.",
        {"generation_backend": "gaussian_voronoi", "module_sequence": requested_modules},
    )

    assert plan.generation_backend == "modular"
    assert [item.module for item in plan.module_sequence] == ["continent", "ridge", "water_body"]
    assert plan.module_sequence[0].params["size"] == 0.44
    assert ignored.generation_backend == "gaussian_voronoi"
    assert ignored.module_sequence == []


def test_create_map_request_validates_module_sequence_schema() -> None:
    valid = CreateMapRequest(
        prompt="Schema validation world",
        module_sequence=[
            {"module": "continent", "params": {"position": "center", "size": 0.4, "operation": "add"}},
            {"module": "water_body", "params": {"position": "south", "coverage": 0.2, "operation": "subtract"}},
        ],
    )

    assert valid.module_sequence[0].module == "continent"
    assert valid.module_sequence[1].params["operation"] == "subtract"

    with pytest.raises(ValidationError):
        CreateMapRequest(
            prompt="Bad module world",
            module_sequence=[{"module": "volcano", "params": {"operation": "add"}}],
        )

    with pytest.raises(ValidationError):
        CreateMapRequest(
            prompt="Bad operation world",
            module_sequence=[{"module": "continent", "params": {"operation": "blend"}}],
        )

    with pytest.raises(ValidationError):
        CreateMapRequest(
            prompt="Missing continent params",
            module_sequence=[{"module": "continent", "params": {"operation": "add"}}],
        )

    with pytest.raises(ValidationError):
        CreateMapRequest(
            prompt="Bad water params",
            module_sequence=[{"module": "water_body", "params": {"position": "south", "coverage": 2.0, "operation": "subtract"}}],
        )

    with pytest.raises(ValidationError):
        CreateMapRequest(
            prompt="Bad smooth params",
            module_sequence=[{"module": "smooth", "params": {"sigma": 0.0, "operation": "replace"}}],
        )


def test_artifact_repository_roundtrips_world_arrays_with_s3_backend(tmp_path, monkeypatch) -> None:
    stored: dict[str, bytes] = {}

    class FakeS3Client:
        def put_bytes(self, key: str, data: bytes, content_type: str) -> None:
            stored[key] = data

        def get_bytes(self, key: str) -> bytes:
            return stored[key]

        def exists(self, key: str) -> bool:
            return key in stored

    monkeypatch.setattr(settings, "ARTIFACT_BACKEND", "s3")
    monkeypatch.setattr(artifact_repo_module, "s3_client", FakeS3Client())

    repo = ArtifactRepository(root=tmp_path)
    task_id = "artifact-s3-roundtrip"
    elevation = np.arange(16, dtype=np.float32).reshape(4, 4)
    moisture = np.full((4, 4), 0.5, dtype=np.float32)

    saved_path = repo.save_world(task_id, elevation=elevation, moisture=moisture)
    loaded = repo.load_world(task_id)

    assert saved_path.name == "world.zarr"
    assert repo._object_key("maps", task_id, "world.zarr.zip") in stored
    assert loaded["elevation"].shape == (4, 4)
    assert np.array_equal(loaded["elevation"], elevation)
    assert np.array_equal(loaded["moisture"], moisture)
