from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from PIL import Image

from app.pipelines.plan_compiler import ConstraintGraph, ConstraintNode, ConstraintEdge

logger = logging.getLogger(__name__)


class AssetNormalizer:
    def __init__(self, artifacts_root: Path | None = None):
        from app.config import settings
        self.root = Path(artifacts_root or settings.ARTIFACT_ROOT)

    def normalize(self, asset_id: str, asset_type: str) -> dict:
        asset_dir = self.root / "assets" / asset_id
        meta_path = asset_dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Asset {asset_id} not found")

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        filename = meta.get("filename", "")
        file_path = asset_dir / filename

        if not file_path.exists():
            raise FileNotFoundError(f"Asset file {filename} not found for asset {asset_id}")

        if asset_type == "image" or asset_type == "sketch":
            return self._normalize_image(file_path, meta, asset_id)
        elif asset_type == "elevation":
            return self._normalize_elevation(file_path, meta, asset_id)
        elif asset_type == "mask":
            return self._normalize_mask(file_path, meta, asset_id)
        else:
            return self._normalize_image(file_path, meta, asset_id)

    def _persist_array(self, asset_id: str, key: str, array: np.ndarray) -> str:
        asset_dir = self.root / "assets" / asset_id
        asset_dir.mkdir(parents=True, exist_ok=True)
        npz_path = asset_dir / f"{key}.npz"
        np.savez_compressed(npz_path, **{key: array})
        rel_path = f"assets/{asset_id}/{key}.npz"
        logger.info("Persisted asset array %s/%s shape=%s to %s", asset_id, key, array.shape, rel_path)
        return rel_path

    def _normalize_image(self, file_path: Path, meta: dict, asset_id: str) -> dict:
        img = Image.open(file_path).convert("L")
        arr = np.array(img, dtype=np.float32) / 255.0
        mask = (arr > 0.5).astype(np.float32)
        h, w = mask.shape
        ref_path = self._persist_array(asset_id, "mask", mask)
        return {
            "type": "sketch_mask",
            "ref_path": ref_path,
            "ref_key": "mask",
            "shape": [h, w],
            "asset_id": meta.get("asset_id", asset_id),
            "source_filename": meta.get("filename", ""),
        }

    def _normalize_elevation(self, file_path: Path, meta: dict, asset_id: str) -> dict:
        elevation = None
        try:
            import rasterio
            with rasterio.open(file_path) as src:
                elevation = src.read(1).astype(np.float32)
                if src.nodata is not None:
                    elevation[elevation == src.nodata] = np.nan
                elevation = np.nan_to_num(elevation, nan=0.0)
                p2, p98 = np.percentile(elevation, [2, 98])
                if p98 - p2 > 1e-6:
                    elevation = (elevation - p2) / (p98 - p2) * 2.0 - 1.0
                elevation = np.clip(elevation, -1.0, 1.0).astype(np.float32)
        except ImportError:
            pass

        if elevation is None:
            img = Image.open(file_path).convert("L")
            arr = np.array(img, dtype=np.float32) / 255.0
            elevation = (arr * 2.0 - 1.0).astype(np.float32)

        h, w = elevation.shape
        ref_path = self._persist_array(asset_id, "elevation", elevation)
        return {
            "type": "elevation_source",
            "ref_path": ref_path,
            "ref_key": "elevation",
            "shape": [h, w],
            "blend_weight": 0.72,
            "asset_id": meta.get("asset_id", asset_id),
            "source_filename": meta.get("filename", ""),
        }

    def _normalize_mask(self, file_path: Path, meta: dict, asset_id: str) -> dict:
        data = np.load(file_path, allow_pickle=False)
        key = list(data.keys())[0]
        mask = data[key].astype(np.float32)
        h, w = mask.shape
        ref_path = self._persist_array(asset_id, "mask", mask)
        return {
            "type": "mask",
            "ref_path": ref_path,
            "ref_key": "mask",
            "shape": [h, w],
            "asset_id": meta.get("asset_id", asset_id),
            "source_filename": meta.get("filename", ""),
        }

    def assets_to_constraint_graph(self, asset_results: list[dict]) -> ConstraintGraph:
        graph = ConstraintGraph()
        for result in asset_results:
            asset_id = result.get("asset_id", "unknown")
            result_type = result.get("type", "sketch_mask")

            if result_type == "sketch_mask":
                node = ConstraintNode(
                    node_id=f"sketch_{asset_id}",
                    node_type="sketch",
                    attributes={
                        "ref_path": result.get("ref_path", ""),
                        "ref_key": result.get("ref_key", "mask"),
                        "shape": result.get("shape", []),
                    },
                    source="asset",
                )
                graph.add_node(node)
            elif result_type == "elevation_source":
                node = ConstraintNode(
                    node_id=f"elev_{asset_id}",
                    node_type="elevation_source",
                    attributes={
                        "ref_path": result.get("ref_path", ""),
                        "ref_key": result.get("ref_key", "elevation"),
                        "shape": result.get("shape", []),
                        "blend_weight": result.get("blend_weight", 0.72),
                    },
                    source="asset",
                )
                graph.add_node(node)
            elif result_type == "mask":
                node = ConstraintNode(
                    node_id=f"mask_{asset_id}",
                    node_type="mask",
                    attributes={
                        "ref_path": result.get("ref_path", ""),
                        "ref_key": result.get("ref_key", "mask"),
                        "shape": result.get("shape", []),
                    },
                    source="asset",
                )
                graph.add_node(node)

        return graph

    def inject_into_plan(self, plan: dict, asset_results: list[dict]) -> dict:
        plan = dict(plan)
        asset_constraints = []

        for result in asset_results:
            result_type = result.get("type", "sketch_mask")
            if result_type == "elevation_source":
                plan["elevation_source"] = {
                    "ref_path": result["ref_path"],
                    "ref_key": result.get("ref_key", "elevation"),
                    "shape": result.get("shape"),
                    "blend_weight": result.get("blend_weight", 0.72),
                    "asset_id": result.get("asset_id"),
                }
                plan["generation_backend"] = "elevation_blend"
            elif result_type in ("sketch_mask", "mask"):
                asset_constraints.append({
                    "type": "elevation_offset",
                    "value": 0.15,
                    "ref_path": result["ref_path"],
                    "ref_key": result.get("ref_key", "mask"),
                    "shape": result.get("shape"),
                    "asset_id": result.get("asset_id"),
                })

        if asset_constraints:
            existing = list(plan.get("asset_constraints") or [])
            existing.extend(asset_constraints)
            plan["asset_constraints"] = existing

        return plan

    @staticmethod
    def load_array_from_ref(ref_path: str, ref_key: str, artifacts_root: Path | None = None) -> np.ndarray:
        from app.config import settings
        root = Path(artifacts_root or settings.ARTIFACT_ROOT)
        full_path = root / ref_path
        if not full_path.exists():
            raise FileNotFoundError(f"Asset reference not found: {full_path}")
        data = np.load(full_path, allow_pickle=False)
        if ref_key not in data:
            available = list(data.keys())
            if len(available) == 1:
                ref_key = available[0]
            else:
                raise KeyError(f"Key {ref_key} not found in {ref_path}, available: {available}")
        return data[ref_key].astype(np.float32)
