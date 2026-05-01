from __future__ import annotations

import logging

import numpy as np
from PIL import Image

from app.core.climate import ClimateSimulator
from app.core.coast_naturalizer import naturalize_coastline
from app.core.geometry_metrics import compute_metric_report
from app.core.preview_renderer import render_preview_image
from app.core.terrain import TerrainGenerator
from app.core.terrain_modular import ModularTerrainGenerator
from app.core.terrain_shaping import (
    apply_constraint_topology,
    apply_layout_template,
    apply_water_enforcement,
    build_hard_topology,
    constraints_for_refinement,
    enforce_required_water_gaps,
    enforce_topology_components,
    has_explicit_plan_geometry,
    normalize_generation_plan,
    shape_world_profile,
)
from app.core.nl_verifier import iterative_regenerate
from app.core.topology_guard import TopologyGuard
from app.pipelines.base import ArtifactManifest, TerrainBundle, TerrainPipeline

logger = logging.getLogger(__name__)

NL_VERIFY_ENABLED = True
NL_VERIFY_MAX_ITERATIONS = 3


class FlatTerrainPipeline(TerrainPipeline):
    projection = "flat"

    def generate(self, plan: dict, seed: int, width: int, height: int) -> TerrainBundle:
        plan = normalize_generation_plan(plan)
        backend = str(plan.get("generation_backend") or "gaussian_voronoi")
        constraints = plan.get("constraints") or {}
        profile = plan.get("profile") or {}
        topology_intent = plan.get("topology_intent") or {}
        coast_complexity = float(profile.get("coast_complexity", 0.5))
        ruggedness = float(profile.get("ruggedness", 0.55))

        elevation_source = plan.get("elevation_source")
        if elevation_source and backend == "elevation_blend":
            elevation = self._blend_elevation_source(elevation_source, width, height, seed)
        else:
            terrain = TerrainGenerator(width=width, height=height, seed=seed)
            if backend == "modular":
                elevation = ModularTerrainGenerator(width=width, height=height, seed=seed).generate(plan or {})
            else:
                elevation = terrain.generate()

        asset_constraints = plan.get("asset_constraints")
        if asset_constraints:
            elevation = self._apply_asset_constraints(elevation, asset_constraints)

        terrain = TerrainGenerator(width=width, height=height, seed=seed)
        elevation, uses_hard_topology = shape_world_profile(terrain, elevation, plan or {})
        if constraints:
            refinement_constraints = constraints_for_refinement(constraints, profile, uses_hard_topology)
            if refinement_constraints and not uses_hard_topology:
                elevation = terrain.apply_constraints(elevation, refinement_constraints)
                elevation = terrain.reinforce_constraints(elevation, refinement_constraints, blend=0.66)

        if topology_intent:
            boundary_irregularity = float(topology_intent.get("boundary_irregularity", 0.5))
            elevation = naturalize_coastline(
                terrain, elevation,
                boundary_irregularity=boundary_irregularity,
                coast_complexity=coast_complexity,
            )
            try:
                from app.core.coast_naturalizer import enforce_fractal_dimension
                target_fd = 1.10 + coast_complexity * 0.12
                elevation = enforce_fractal_dimension(
                    terrain, elevation,
                    target_fd=target_fd,
                    tolerance=0.10,
                    max_iterations=1,
                    coast_complexity=coast_complexity,
                )
            except Exception as exc:
                logger.warning("Fractal dimension enforcement skipped: %s", exc)

        guard = TopologyGuard(max_repair_strength=0.3)
        elevation, guard_result = guard.repair(elevation, topology_intent if topology_intent else None)

        if ruggedness > 0.3:
            try:
                from app.core.hydrology_advanced import curvature_guided_erosion, multi_scale_erosion
                erosion_strength = min(ruggedness * 0.6, 0.35)
                elevation = curvature_guided_erosion(
                    elevation, iterations=2,
                    erosion_rate=0.015 * erosion_strength,
                    ridge_protection=0.75,
                )
                elevation = multi_scale_erosion(
                    elevation, scales=[1, 3],
                    base_rate=0.008 * erosion_strength,
                )
            except Exception as exc:
                logger.warning("Curvature-guided erosion skipped: %s", exc)

        try:
            from app.core.terrain_analysis import compute_multi_scale_tpi
            mstpi = compute_multi_scale_tpi(elevation, scales=[3, 9])
            tpi_fine = mstpi["tpi_3"]
            tpi_coarse = mstpi["tpi_9"]
            tpi_combined = tpi_fine * 0.6 + tpi_coarse * 0.4
            tpi_norm = tpi_combined / (np.std(tpi_combined) + 1e-10)
            ridge_mask = tpi_norm > 2.0
            valley_mask = tpi_norm < -2.0
            land = elevation > 0
            elevation[ridge_mask & land] *= 1.0 + 0.01 * ruggedness
            elevation[valley_mask & land] *= 1.0 - 0.005 * ruggedness
            elevation = np.clip(elevation, -1.0, 1.0).astype(np.float32)
        except Exception as exc:
            logger.warning("Multi-scale TPI feedback skipped: %s", exc)

        filled = elevation
        latitude = np.linspace(90.0, -90.0, height, dtype=np.float32).reshape(height, 1)
        lat_grid = np.repeat(latitude, width, axis=1)
        climate = ClimateSimulator(elev=((filled + 1.0) * 2500.0).astype(np.float32), lat_grid=lat_grid).run(
            wind_direction=profile.get("wind_direction", "westerly"),
            moisture_factor=float(profile.get("moisture", 1.0)),
            temperature_bias=float(profile.get("temperature_bias", 0.0)),
        )

        return TerrainBundle(
            elevation=filled.astype(np.float32),
            temperature=climate["temperature"].astype(np.float32),
            precipitation=climate["precipitation"].astype(np.float32),
            biome=climate["biome"].astype(np.int16),
            projection="flat",
            resolution=(height, width),
        )

    def edit(self, terrain: TerrainBundle, instruction: dict) -> TerrainBundle:
        from app.core.incremental_editor import EditInstruction, IncrementalEditor

        elevation = terrain.elevation.copy()
        h, w = elevation.shape
        editor = IncrementalEditor(elevation, instruction.get("plan", {}))

        if instruction.get("constraints"):
            edit_inst = EditInstruction(instruction["constraints"], (h, w))
            editor.add_constraints_from_instruction(edit_inst)

        if instruction.get("text"):
            text_inst = EditInstruction.from_text(instruction["text"], (h, w), elevation)
            editor.add_constraints_from_instruction(text_inst)

        new_elevation = editor.apply_constraints()
        return TerrainBundle(
            elevation=new_elevation.astype(np.float32),
            temperature=terrain.temperature.copy(),
            precipitation=terrain.precipitation.copy(),
            biome=terrain.biome.copy(),
            projection="flat",
            resolution=terrain.resolution,
        )

    def build_preview(self, terrain: TerrainBundle, profile: dict) -> Image.Image:
        return render_preview_image(
            terrain.elevation,
            terrain.biome,
            terrain.temperature,
            terrain.precipitation,
            profile or {},
        )

    def build_artifacts(self, terrain: TerrainBundle, version: int) -> ArtifactManifest:
        return ArtifactManifest(
            version=version,
            projection="flat",
            bounds=[-180, -85, 180, 85],
            tiling_scheme="web_mercator",
            wrap_x=False,
        )

    def _blend_elevation_source(self, source: dict, width: int, height: int, seed: int) -> np.ndarray:
        from app.pipelines.asset_normalizer import AssetNormalizer

        ref_path = source.get("ref_path")
        ref_key = source.get("ref_key", "elevation")
        src_elev = None

        if ref_path:
            try:
                src_elev = AssetNormalizer.load_array_from_ref(ref_path, ref_key)
                logger.info("Loaded elevation source from ref %s shape=%s", ref_path, src_elev.shape)
            except Exception as exc:
                logger.warning("Failed to load elevation source from ref %s: %s, falling back to procedural", ref_path, exc)

        if src_elev is None:
            logger.warning("elevation_source has no loadable data, falling back to procedural")
            return TerrainGenerator(width=width, height=height, seed=seed).generate()

        terrain = TerrainGenerator(width=width, height=height, seed=seed)
        procedural = terrain.generate()

        from PIL import Image as PILImage
        src_resized = np.array(
            PILImage.fromarray(((src_elev + 1.0) * 127.5).astype(np.uint8), mode="L").resize(
                (width, height), PILImage.BILINEAR
            ),
            dtype=np.float32,
        ) / 127.5 - 1.0

        blend_weight = float(source.get("blend_weight", 0.72))
        blended = src_resized * blend_weight + procedural * (1.0 - blend_weight)
        return np.clip(blended, -1.0, 1.0).astype(np.float32)

    def _apply_asset_constraints(self, elevation: np.ndarray, asset_constraints: list[dict]) -> np.ndarray:
        from app.pipelines.asset_normalizer import AssetNormalizer

        result = elevation.copy()
        h, w = result.shape
        for constraint in asset_constraints:
            ctype = constraint.get("type", "elevation_offset")
            value = float(constraint.get("value", 0.1))
            ref_path = constraint.get("ref_path")
            ref_key = constraint.get("ref_key", "mask")

            mask = None
            if ref_path:
                try:
                    mask = AssetNormalizer.load_array_from_ref(ref_path, ref_key)
                    logger.info("Loaded asset mask from ref %s shape=%s", ref_path, mask.shape)
                except Exception as exc:
                    logger.warning("Failed to load asset mask from ref %s: %s", ref_path, exc)

            if mask is not None:
                mask = self._resize_mask(mask, (h, w))
                if ctype == "elevation_offset":
                    result = result * (1.0 - mask * 0.8) + (result + value) * (mask * 0.8)
                elif ctype == "elevation_set":
                    result = result * (1.0 - mask) + value * mask
            else:
                logger.warning("No mask data for asset constraint %s, skipping", constraint.get("asset_id"))

        return np.clip(result, -1.0, 1.0).astype(np.float32)

    @staticmethod
    def _resize_mask(mask: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
        from PIL import Image as PILImage
        th, tw = target_shape
        mh, mw = mask.shape
        if mh == th and mw == tw:
            return mask.astype(np.float32)
        return np.array(
            PILImage.fromarray((mask * 255).astype(np.uint8), mode="L").resize(
                (tw, th), PILImage.BILINEAR
            ),
            dtype=np.float32,
        ) / 255.0
