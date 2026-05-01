from __future__ import annotations

import logging

import numpy as np
from PIL import Image

from app.core.climate import ClimateSimulator
from app.core.geometry_metrics import compute_metric_report
from app.core.preview_renderer import render_preview_image
from app.core.terrain_shaping import (
    apply_planet_semantic_shape,
    normalize_generation_plan,
    planet_constraints_from_plan,
    rebalance_and_scale_elevation,
)
from app.core.topology_guard import TopologyGuard
from app.pipelines.base import ArtifactManifest, TerrainBundle, TerrainPipeline

logger = logging.getLogger(__name__)


class PlanetTerrainPipeline(TerrainPipeline):
    projection = "planet"

    def generate(self, plan: dict, seed: int, width: int, height: int) -> TerrainBundle:
        plan = normalize_generation_plan(plan)
        profile = plan.get("profile") or {}
        topology_intent = plan.get("topology_intent") or {}
        ruggedness = float(profile.get("ruggedness", 0.55))

        elevation_source = plan.get("elevation_source")
        if elevation_source and plan.get("generation_backend") == "elevation_blend":
            elevation = self._blend_elevation_source(elevation_source, width, height, seed)
        else:
            from app.core.cubemap_terrain import CubeMapTerrainGenerator
            from app.core.cubemap_to_erp import cubemap_to_erp
            from app.core.sphere_postprocess import run_sphere_postprocess

            planet_quality = str(profile.get("planet_quality", "standard"))
            face_resolution = {"performance": 512, "standard": 1024, "high": 2048}.get(planet_quality, 1024)
            erp_width = max(width, 2048)
            erp_height = max(height, 1024)

            cube_gen = CubeMapTerrainGenerator(face_resolution=face_resolution, seed=seed)
            p_constraints = planet_constraints_from_plan(plan)
            cube_faces = cube_gen.generate(constraints=p_constraints if p_constraints else None)
            elevation = cubemap_to_erp(cube_faces, erp_width=erp_width, erp_height=erp_height)
            elevation = apply_planet_semantic_shape(elevation, plan, seed)

            uplift = None
            hardness = None
            try:
                from app.core.uplift_field import generate_uplift_field, generate_hardness_map
                continent_mask = (elevation > 0).astype(np.float32)
                uplift = generate_uplift_field(
                    erp_height=erp_height, erp_width=erp_width,
                    seed=seed, continent_mask=continent_mask, ruggedness=ruggedness,
                )
                hardness = generate_hardness_map(elevation, ruggedness=ruggedness, seed=seed)
            except Exception as exc:
                logger.warning("Uplift/hardness generation skipped: %s", exc)

            elevation, sphere_guard = run_sphere_postprocess(
                elevation, profile=profile, topology_intent=topology_intent,
                seed=seed, uplift=uplift, hardness=hardness,
            )
            elevation = rebalance_and_scale_elevation(
                apply_planet_semantic_shape(elevation, plan, seed, blend=0.35),
                land_ratio=float(profile.get("land_ratio", 0.44)),
            )

        asset_constraints = plan.get("asset_constraints")
        if asset_constraints:
            elevation = self._apply_asset_constraints(elevation, asset_constraints)

        h, w = elevation.shape
        latitude = np.linspace(90.0, -90.0, h, dtype=np.float32).reshape(h, 1)
        lat_grid = np.repeat(latitude, w, axis=1)
        climate = ClimateSimulator(elev=((elevation + 1.0) * 2500.0).astype(np.float32), lat_grid=lat_grid).run(
            wind_direction=profile.get("wind_direction", "westerly"),
            moisture_factor=float(profile.get("moisture", 1.0)),
            temperature_bias=float(profile.get("temperature_bias", 0.0)),
        )

        return TerrainBundle(
            elevation=elevation.astype(np.float32),
            temperature=climate["temperature"].astype(np.float32),
            precipitation=climate["precipitation"].astype(np.float32),
            biome=climate["biome"].astype(np.int16),
            projection="planet",
            resolution=(h, w),
        )

    def edit(self, terrain: TerrainBundle, instruction: dict) -> TerrainBundle:
        from app.core.incremental_editor import EditInstruction, IncrementalEditor

        elevation = terrain.elevation.copy()
        h, w = elevation.shape
        editor = IncrementalEditor(elevation, instruction.get("plan", {}))

        if instruction.get("constraints"):
            edit_inst = EditInstruction(instruction["constraints"], (h, w))
            editor.add_constraints_from_instruction(edit_inst)

        new_elevation = editor.apply_constraints()
        return TerrainBundle(
            elevation=new_elevation.astype(np.float32),
            temperature=terrain.temperature.copy(),
            precipitation=terrain.precipitation.copy(),
            biome=terrain.biome.copy(),
            projection="planet",
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
            projection="planet",
            bounds=[-180, -90, 180, 90],
            tiling_scheme="geographic",
            wrap_x=True,
            level_zero_tiles_x=1,
            level_zero_tiles_y=1,
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
            from app.core.terrain import TerrainGenerator
            return TerrainGenerator(width=width, height=height, seed=seed).generate()

        from app.core.terrain import TerrainGenerator
        procedural = TerrainGenerator(width=width, height=height, seed=seed).generate()

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
