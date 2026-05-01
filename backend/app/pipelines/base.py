from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from PIL import Image


@dataclass
class TerrainBundle:
    elevation: np.ndarray
    temperature: np.ndarray
    precipitation: np.ndarray
    biome: np.ndarray
    projection: str = "flat"
    resolution: tuple[int, int] = (512, 1024)

    def to_arrays(self) -> dict[str, np.ndarray]:
        return {
            "elevation": self.elevation.astype(np.float32),
            "temperature": self.temperature.astype(np.float32),
            "precipitation": self.precipitation.astype(np.float32),
            "biome": self.biome.astype(np.int16),
        }

    @classmethod
    def from_arrays(cls, arrays: dict[str, np.ndarray], projection: str = "flat") -> "TerrainBundle":
        h, w = arrays["elevation"].shape
        return cls(
            elevation=arrays["elevation"].astype(np.float32),
            temperature=arrays.get("temperature", np.zeros((h, w), dtype=np.float32)).astype(np.float32),
            precipitation=arrays.get("precipitation", np.zeros((h, w), dtype=np.float32)).astype(np.float32),
            biome=arrays.get("biome", np.zeros((h, w), dtype=np.int16)).astype(np.int16),
            projection=projection,
            resolution=(h, w),
        )


@dataclass
class ArtifactManifest:
    version: int = 0
    preview_url: str = ""
    manifest_url: str = ""
    cog_url: str = ""
    tile_url_template: str = ""
    projection: str = "flat"
    bounds: list[float] = field(default_factory=lambda: [-180, -85, 180, 85])
    min_zoom: int = 0
    max_zoom: int = 6
    tiling_scheme: str = "web_mercator"
    wrap_x: bool = False
    level_zero_tiles_x: int = 2
    level_zero_tiles_y: int = 1
    ready_tiles: list[str] = field(default_factory=list)
    dirty_bounds: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "preview_url": self.preview_url,
            "manifest_url": self.manifest_url,
            "cog_url": self.cog_url,
            "tile_url_template": self.tile_url_template,
            "projection": self.projection,
            "bounds": self.bounds,
            "min_zoom": self.min_zoom,
            "max_zoom": self.max_zoom,
            "tiling_scheme": self.tiling_scheme,
            "wrap_x": self.wrap_x,
            "level_zero_tiles_x": self.level_zero_tiles_x,
            "level_zero_tiles_y": self.level_zero_tiles_y,
            "ready_tiles": self.ready_tiles,
            "dirty_bounds": self.dirty_bounds,
        }


class TerrainPipeline(ABC):
    projection: str

    @abstractmethod
    def generate(self, plan: dict, seed: int, width: int, height: int) -> TerrainBundle:
        ...

    @abstractmethod
    def edit(self, terrain: TerrainBundle, instruction: dict) -> TerrainBundle:
        ...

    @abstractmethod
    def build_preview(self, terrain: TerrainBundle, profile: dict) -> Image.Image:
        ...

    @abstractmethod
    def build_artifacts(self, terrain: TerrainBundle, version: int) -> ArtifactManifest:
        ...
