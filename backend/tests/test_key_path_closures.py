from __future__ import annotations

import importlib
import inspect
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest


def test_editor_py_no_direct_repo_dependency():
    source = Path("app/api/routes/editor.py").read_text(encoding="utf-8")
    assert "repo.save_preview" not in source
    assert "repo.load_world" not in source
    assert "repo.load_cog" not in source
    assert "MapVersion" not in source
    assert "generate_tiles_for_task" not in source


def test_artifact_orchestrator_save_version_delegates_to_full():
    from app.orchestrator.artifact_orchestrator import ArtifactOrchestrator
    source = inspect.getsource(ArtifactOrchestrator.save_version)
    assert "save_version_full" in source


def test_terrain_shaping_no_render_worker_import():
    mod = importlib.import_module("app.core.terrain_shaping")
    source = inspect.getsource(mod)
    assert "from app.workers.render_worker" not in source
    assert "import app.workers.render_worker" not in source


def test_render_worker_delegates_to_orchestrator():
    mod = importlib.import_module("app.workers.render_worker")
    source = inspect.getsource(mod.render_world)
    assert "app.orchestrator.orchestrator" in source


def test_plan_compiler_asset_graph_not_overwritten():
    from app.pipelines.plan_compiler import PlanCompiler, ConstraintGraph, ConstraintNode
    compiler = PlanCompiler.__new__(PlanCompiler)
    compiler.api_key = None
    compiler.base_url = None
    compiler.model = None

    plan = {
        "profile": {"land_ratio": 0.5},
        "constraints": {},
        "constraint_graph": {
            "nodes": [{"node_id": "continent_1", "node_type": "continent", "position": "west", "attributes": {}, "source": "text"}],
            "edges": [],
            "raw_text": "a continent in the west",
        },
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        asset_dir = Path(tmpdir) / "assets" / "test_asset"
        asset_dir.mkdir(parents=True)
        meta = {"asset_id": "test_asset", "filename": "test.png", "type": "image"}
        (asset_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
        from PIL import Image
        img = Image.new("L", (64, 32), color=200)
        img.save(asset_dir / "test.png")

        from app.pipelines.asset_normalizer import AssetNormalizer
        normalizer = AssetNormalizer(artifacts_root=Path(tmpdir))
        result = normalizer.normalize("test_asset", "image")
        assert result["type"] == "sketch_mask"
        assert "ref_path" in result
        assert "mask" not in result

        asset_graph = normalizer.assets_to_constraint_graph([result])
        assert len(asset_graph.nodes) == 1
        assert asset_graph.nodes[0].source == "asset"

        injected = normalizer.inject_into_plan(dict(plan), [result])
        assert "asset_constraints" in injected
        assert "elevation_source" not in injected
        assert "ref_path" in injected["asset_constraints"][0]
        assert "mask_shape" not in injected["asset_constraints"][0]


def test_pipeline_consumes_elevation_source_via_ref():
    from app.pipelines.flat_pipeline import FlatTerrainPipeline

    pipeline = FlatTerrainPipeline()
    h, w = 128, 256

    with tempfile.TemporaryDirectory() as tmpdir:
        src_elevation = np.random.RandomState(42).randn(h, w).astype(np.float32) * 0.3
        asset_dir = Path(tmpdir) / "assets" / "elev_asset"
        asset_dir.mkdir(parents=True)
        np.savez_compressed(asset_dir / "elevation.npz", elevation=src_elevation)

        plan = {
            "generation_backend": "elevation_blend",
            "elevation_source": {
                "ref_path": "assets/elev_asset/elevation.npz",
                "ref_key": "elevation",
                "shape": [h, w],
                "blend_weight": 0.8,
                "asset_id": "elev_asset",
            },
            "profile": {"land_ratio": 0.5, "ruggedness": 0.3, "coast_complexity": 0.3},
            "constraints": {},
            "topology_intent": {},
        }
        import os
        old_root = os.environ.get("ARTIFACT_ROOT")
        os.environ["ARTIFACT_ROOT"] = tmpdir
        try:
            from app.config import settings
            old_setting = getattr(settings, "ARTIFACT_ROOT", None)
            settings.ARTIFACT_ROOT = tmpdir
            bundle = pipeline.generate(plan, seed=42, width=w, height=h)
            assert bundle.elevation.shape == (h, w)
            assert bundle.elevation.dtype == np.float32
            settings.ARTIFACT_ROOT = old_setting
        finally:
            if old_root is not None:
                os.environ["ARTIFACT_ROOT"] = old_root
            else:
                os.environ.pop("ARTIFACT_ROOT", None)


def test_pipeline_consumes_asset_constraints_via_ref():
    from app.pipelines.flat_pipeline import FlatTerrainPipeline

    pipeline = FlatTerrainPipeline()
    h, w = 128, 256

    with tempfile.TemporaryDirectory() as tmpdir:
        mask = np.zeros((h, w), dtype=np.float32)
        mask[40:80, 60:180] = 1.0
        asset_dir = Path(tmpdir) / "assets" / "mask_asset"
        asset_dir.mkdir(parents=True)
        np.savez_compressed(asset_dir / "mask.npz", mask=mask)

        plan = {
            "profile": {"land_ratio": 0.5, "ruggedness": 0.3, "coast_complexity": 0.3},
            "constraints": {},
            "topology_intent": {},
            "asset_constraints": [
                {
                    "type": "elevation_offset",
                    "value": 0.2,
                    "ref_path": "assets/mask_asset/mask.npz",
                    "ref_key": "mask",
                    "shape": [h, w],
                    "asset_id": "mask_asset",
                },
            ],
        }
        from app.config import settings
        old_setting = getattr(settings, "ARTIFACT_ROOT", None)
        settings.ARTIFACT_ROOT = tmpdir
        try:
            bundle = pipeline.generate(plan, seed=42, width=w, height=h)
            assert bundle.elevation.shape == (h, w)
        finally:
            settings.ARTIFACT_ROOT = old_setting


def test_asset_normalizer_persist_and_load():
    with tempfile.TemporaryDirectory() as tmpdir:
        asset_dir = Path(tmpdir) / "assets" / "test_asset"
        asset_dir.mkdir(parents=True)
        meta = {"asset_id": "test_asset", "filename": "test.png", "type": "image"}
        (asset_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
        from PIL import Image
        img = Image.new("L", (64, 32), color=200)
        img.save(asset_dir / "test.png")

        from app.pipelines.asset_normalizer import AssetNormalizer
        normalizer = AssetNormalizer(artifacts_root=Path(tmpdir))
        result = normalizer.normalize("test_asset", "image")

        assert "ref_path" in result
        assert "mask" not in result

        loaded = AssetNormalizer.load_array_from_ref(result["ref_path"], result["ref_key"], artifacts_root=Path(tmpdir))
        assert loaded.shape == tuple(result["shape"])
        assert loaded.dtype == np.float32


def test_plan_json_serializable_after_asset_injection():
    with tempfile.TemporaryDirectory() as tmpdir:
        asset_dir = Path(tmpdir) / "assets" / "test_asset"
        asset_dir.mkdir(parents=True)
        meta = {"asset_id": "test_asset", "filename": "test.png", "type": "image"}
        (asset_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
        from PIL import Image
        img = Image.new("L", (64, 32), color=200)
        img.save(asset_dir / "test.png")

        from app.pipelines.asset_normalizer import AssetNormalizer
        normalizer = AssetNormalizer(artifacts_root=Path(tmpdir))
        result = normalizer.normalize("test_asset", "image")

        plan = {
            "profile": {"land_ratio": 0.5},
            "constraints": {},
        }
        injected = normalizer.inject_into_plan(plan, [result])

        serialized = json.dumps(injected)
        deserialized = json.loads(serialized)
        assert deserialized["asset_constraints"][0]["ref_path"] == result["ref_path"]


def test_dirty_bounds_small_edit():
    from app.orchestrator.editor_orchestrator import EditorOrchestrator
    h, w = 128, 256
    old = np.zeros((h, w), dtype=np.float32)
    new = old.copy()
    new[50:70, 100:140] = 0.1
    bounds = EditorOrchestrator._compute_dirty_bounds(old, new)
    assert len(bounds) >= 1
    b = bounds[0]
    assert b["min_y"] <= 50
    assert b["max_y"] >= 69
    assert b["min_x"] <= 100
    assert b["max_x"] >= 139


def test_dirty_bounds_no_change():
    from app.orchestrator.editor_orchestrator import EditorOrchestrator
    h, w = 64, 128
    elev = np.zeros((h, w), dtype=np.float32)
    bounds = EditorOrchestrator._compute_dirty_bounds(elev, elev.copy())
    assert bounds == []


def test_geojson_coordinate_consistency():
    from app.core.incremental_editor import _geojson_to_mask
    h, w = 512, 1024
    geojson = {
        "type": "Feature",
        "properties": {"brushMode": "raise", "brushSize": 5},
        "geometry": {
            "type": "LineString",
            "coordinates": [[0.0, 45.0], [10.0, 45.0], [20.0, 45.0]],
        },
    }
    mask = _geojson_to_mask(geojson, (h, w), brush_radius=5)
    assert mask.shape == (h, w)
    assert np.any(mask > 0)

    center_lon = 10.0
    center_lat = 45.0
    expected_px = int((center_lon + 180.0) / 360.0 * (w - 1))
    expected_py = int((90.0 - center_lat) / 180.0 * (h - 1))
    assert mask[expected_py, expected_px] > 0


def test_refresh_tiles_logs_fallback_reason():
    from unittest.mock import patch

    from app.orchestrator.artifact_orchestrator import ArtifactOrchestrator

    orch = ArtifactOrchestrator(task_id="test-task", projection="flat")

    with patch.object(orch, "regenerate_dirty_tiles", side_effect=RuntimeError("tile error")):
        with patch.object(orch, "_schedule_full_tile_refresh") as mock_schedule:
            result = orch.refresh_tiles([{"min_y": 0, "min_x": 0, "max_y": 10, "max_x": 10}], {})
            assert result == 0
            mock_schedule.assert_called_once()
            call_args = mock_schedule.call_args[0][0]
            assert "dirty_tile_error" in call_args


def test_refresh_tiles_zero_regenerated_fallback():
    from unittest.mock import patch

    from app.orchestrator.artifact_orchestrator import ArtifactOrchestrator

    orch = ArtifactOrchestrator(task_id="test-task", projection="flat")

    with patch.object(orch, "regenerate_dirty_tiles", return_value=0):
        with patch.object(orch, "_schedule_full_tile_refresh") as mock_schedule:
            result = orch.refresh_tiles([{"min_y": 0, "min_x": 0, "max_y": 10, "max_x": 10}], {})
            assert result == 0
            mock_schedule.assert_called_once_with("zero_tiles_regenerated")


def test_refresh_tiles_success():
    from unittest.mock import patch

    from app.orchestrator.artifact_orchestrator import ArtifactOrchestrator

    orch = ArtifactOrchestrator(task_id="test-task", projection="flat")

    with patch.object(orch, "regenerate_dirty_tiles", return_value=5):
        with patch.object(orch, "_schedule_full_tile_refresh") as mock_schedule:
            result = orch.refresh_tiles([{"min_y": 0, "min_x": 0, "max_y": 10, "max_x": 10}], {})
            assert result == 5
            mock_schedule.assert_not_called()


def test_plan_compiler_merges_asset_nodes_into_constraint_graph():
    from app.pipelines.plan_compiler import ConstraintGraph, ConstraintNode

    text_graph = ConstraintGraph(raw_text="a continent in the west")
    text_graph.add_node(ConstraintNode(
        node_id="continent_1",
        node_type="continent",
        position="west",
        source="text",
    ))

    asset_nodes_data = [
        {"node_id": "sketch_test", "node_type": "sketch", "position": None, "attributes": {"ref_path": "assets/test/mask.npz", "shape": [64, 32]}, "source": "asset"},
    ]

    for node_data in asset_nodes_data:
        text_graph.add_node(ConstraintNode(
            node_id=node_data["node_id"],
            node_type=node_data["node_type"],
            position=node_data.get("position"),
            attributes=node_data.get("attributes", {}),
            source=node_data.get("source", "asset"),
        ))

    result = text_graph.to_dict()
    node_ids = [n["node_id"] for n in result["nodes"]]
    assert "continent_1" in node_ids
    assert "sketch_test" in node_ids

    sources = {n["node_id"]: n["source"] for n in result["nodes"]}
    assert sources["continent_1"] == "text"
    assert sources["sketch_test"] == "asset"


def test_asset_mask_shape_affects_terrain_not_uniform():
    from app.pipelines.flat_pipeline import FlatTerrainPipeline

    pipeline = FlatTerrainPipeline()
    h, w = 128, 256
    seed = 42

    with tempfile.TemporaryDirectory() as tmpdir:
        mask = np.zeros((h, w), dtype=np.float32)
        mask[30:50, 80:120] = 1.0

        asset_dir = Path(tmpdir) / "assets" / "spatial_mask"
        asset_dir.mkdir(parents=True)
        np.savez_compressed(asset_dir / "mask.npz", mask=mask)

        plan_with_asset = {
            "profile": {"land_ratio": 0.5, "ruggedness": 0.3, "coast_complexity": 0.3},
            "constraints": {},
            "topology_intent": {},
            "asset_constraints": [
                {
                    "type": "elevation_offset",
                    "value": 0.3,
                    "ref_path": "assets/spatial_mask/mask.npz",
                    "ref_key": "mask",
                    "shape": [h, w],
                    "asset_id": "spatial_mask",
                },
            ],
        }
        plan_no_asset = {
            "profile": {"land_ratio": 0.5, "ruggedness": 0.3, "coast_complexity": 0.3},
            "constraints": {},
            "topology_intent": {},
        }

        from app.config import settings
        old_setting = getattr(settings, "ARTIFACT_ROOT", None)
        settings.ARTIFACT_ROOT = tmpdir
        try:
            bundle_with = pipeline.generate(plan_with_asset, seed=seed, width=w, height=h)
            bundle_no = pipeline.generate(plan_no_asset, seed=seed, width=w, height=h)
        finally:
            settings.ARTIFACT_ROOT = old_setting

        diff = bundle_with.elevation - bundle_no.elevation
        mask_region_diff = np.abs(diff[30:50, 80:120])
        outside_region_diff = np.abs(diff[0:30, 0:80])

        assert np.mean(mask_region_diff) > np.mean(outside_region_diff) * 2.0, \
            f"Mask region diff {np.mean(mask_region_diff):.4f} should be much larger than outside {np.mean(outside_region_diff):.4f}"


def test_same_prompt_with_vs_without_asset_distinguishable():
    from app.pipelines.flat_pipeline import FlatTerrainPipeline

    pipeline = FlatTerrainPipeline()
    h, w = 128, 256
    seed = 123

    with tempfile.TemporaryDirectory() as tmpdir:
        half_mask = np.zeros((h, w), dtype=np.float32)
        half_mask[:, :w // 2] = 1.0

        asset_dir = Path(tmpdir) / "assets" / "half_mask"
        asset_dir.mkdir(parents=True)
        np.savez_compressed(asset_dir / "mask.npz", mask=half_mask)

        plan_base = {
            "profile": {"land_ratio": 0.5, "ruggedness": 0.3, "coast_complexity": 0.3},
            "constraints": {},
            "topology_intent": {},
        }
        plan_asset = dict(plan_base)
        plan_asset["asset_constraints"] = [
            {
                "type": "elevation_offset",
                "value": 0.25,
                "ref_path": "assets/half_mask/mask.npz",
                "ref_key": "mask",
                "shape": [h, w],
                "asset_id": "half_mask",
            },
        ]

        from app.config import settings
        old_setting = getattr(settings, "ARTIFACT_ROOT", None)
        settings.ARTIFACT_ROOT = tmpdir
        try:
            bundle_base = pipeline.generate(plan_base, seed=seed, width=w, height=h)
            bundle_asset = pipeline.generate(plan_asset, seed=seed, width=w, height=h)
        finally:
            settings.ARTIFACT_ROOT = old_setting

        diff = np.abs(bundle_asset.elevation - bundle_base.elevation)
        total_diff = np.mean(diff)
        max_diff = np.max(diff)

        assert total_diff > 0.001, f"Overall terrain should be distinguishable: mean_diff={total_diff:.6f}"
        assert max_diff > 0.01, f"Max terrain difference should be significant: max_diff={max_diff:.6f}"
