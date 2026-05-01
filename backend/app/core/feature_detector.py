from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from app.core.voronoi_grid import VoronoiDiagram


DEEPER_LAND = 3
LANDLOCKED = 2
LAND_COAST = 1
UNMARKED = 0
WATER_COAST = -1
DEEP_WATER = -2


@dataclass
class GeoFeature:
    feature_id: int
    land: bool
    border: bool
    feature_type: str
    cells: List[int] = field(default_factory=list)
    first_cell: int = -1
    area: float = 0.0
    height: float = 0.0
    temp: float = 0.0
    flux: float = 0.0
    evaporation: float = 0.0
    group: str = ""


class FeatureDetector:
    def __init__(self, voronoi: VoronoiDiagram):
        self.voronoi = voronoi
        self.distance_field: np.ndarray = np.zeros(voronoi.points_n, dtype=np.int8)
        self.feature_ids: np.ndarray = np.zeros(voronoi.points_n, dtype=np.int32)
        self.features: List[GeoFeature] = []

    def markup(
        self,
        distance_field: np.ndarray,
        start: int,
        increment: int,
        limit: int = -10,
    ) -> None:
        current = start
        while True:
            marked = 0
            prev = current - increment
            for cell_id in range(self.voronoi.points_n):
                if distance_field[cell_id] != prev:
                    continue
                for neighbor_id in self.voronoi.get_neighbor_cells(cell_id):
                    if neighbor_id < 0 or neighbor_id >= self.voronoi.points_n:
                        continue
                    if distance_field[neighbor_id] != UNMARKED:
                        continue
                    distance_field[neighbor_id] = current
                    marked += 1
            if marked == 0 or current == limit:
                break
            current += increment

    def markup_grid(self, heights: np.ndarray, sea_level: float = 20.0) -> None:
        n = self.voronoi.points_n
        distance_field = np.zeros(n, dtype=np.int8)
        feature_ids = np.zeros(n, dtype=np.int32)
        features: List[GeoFeature] = [GeoFeature(feature_id=0, land=False, border=False, feature_type="void")]

        unvisited = set(range(n))
        feature_counter = 0

        while unvisited:
            start_cell = min(unvisited)
            feature_counter += 1
            is_land = heights[start_cell] >= sea_level
            is_border = False

            queue = deque([start_cell])
            feature_ids[start_cell] = feature_counter
            unvisited.discard(start_cell)
            feature_cells = [start_cell]

            while queue:
                cell_id = queue.popleft()
                if self.voronoi.is_border(cell_id):
                    is_border = True

                for neighbor_id in self.voronoi.get_neighbor_cells(cell_id):
                    if neighbor_id < 0 or neighbor_id >= n:
                        continue
                    neighbor_land = heights[neighbor_id] >= sea_level

                    if is_land == neighbor_land and feature_ids[neighbor_id] == 0:
                        feature_ids[neighbor_id] = feature_counter
                        queue.append(neighbor_id)
                        unvisited.discard(neighbor_id)
                        feature_cells.append(neighbor_id)
                    elif is_land and not neighbor_land:
                        distance_field[cell_id] = LAND_COAST
                        if distance_field[neighbor_id] == UNMARKED:
                            distance_field[neighbor_id] = WATER_COAST

            if is_land:
                ftype = "island"
            elif is_border:
                ftype = "ocean"
            else:
                ftype = "lake"

            feat = GeoFeature(
                feature_id=feature_counter,
                land=is_land,
                border=is_border,
                feature_type=ftype,
                cells=feature_cells,
                first_cell=start_cell,
            )
            features.append(feat)

        self.markup(
            distance_field,
            start=DEEP_WATER,
            increment=-1,
            limit=-10,
        )

        self.markup(
            distance_field,
            start=DEEPER_LAND,
            increment=1,
            limit=10,
        )

        self.distance_field = distance_field
        self.feature_ids = feature_ids
        self.features = features

    def define_groups(
        self,
        heights: np.ndarray,
        temperature: Optional[np.ndarray] = None,
    ) -> None:
        total_cells = self.voronoi.points_n

        for feat in self.features:
            if feat.feature_id == 0:
                continue

            if feat.feature_type == "ocean":
                feat.group = self._classify_ocean(feat, total_cells)
            elif feat.feature_type == "island":
                feat.group = self._classify_island(feat, total_cells)
            elif feat.feature_type == "lake":
                feat.group = self._classify_lake(feat, heights, temperature)

    def _classify_ocean(self, feat: GeoFeature, total_cells: int) -> str:
        ratio = len(feat.cells) / total_cells
        if ratio > 0.15:
            return "ocean"
        elif ratio > 0.03:
            return "sea"
        else:
            return "gulf"

    def _classify_island(self, feat: GeoFeature, total_cells: int) -> str:
        ratio = len(feat.cells) / total_cells
        if ratio > 0.15:
            return "continent"
        elif ratio > 0.01:
            return "island"
        else:
            return "islet"

    def _classify_lake(
        self,
        feat: GeoFeature,
        heights: np.ndarray,
        temperature: Optional[np.ndarray] = None,
    ) -> str:
        if temperature is not None and len(feat.cells) > 0:
            mean_temp = np.mean([temperature[c] for c in feat.cells if 0 <= c < len(temperature)])
            if mean_temp < -10:
                return "frozen_lake"

        if len(feat.cells) > 0:
            mean_height = np.mean([heights[c] for c in feat.cells if 0 <= c < len(heights)])
            if mean_height < 5:
                return "dry_lake"

        return "freshwater_lake"

    def get_coast_distance(self) -> np.ndarray:
        return self.distance_field.astype(np.float32)

    def get_land_mask(self, sea_level: float = 20.0) -> np.ndarray:
        return self.feature_ids > 0

    def get_ocean_features(self) -> List[GeoFeature]:
        return [f for f in self.features if f.feature_type == "ocean"]

    def get_lake_features(self) -> List[GeoFeature]:
        return [f for f in self.features if f.feature_type == "lake"]

    def get_island_features(self) -> List[GeoFeature]:
        return [f for f in self.features if f.feature_type == "island"]
