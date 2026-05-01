from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter


class TerrainGenerator:
    """Generate world-scale terrain with smooth coastlines and constraint hooks."""

    def __init__(self, width: int, height: int, seed: int = 42):
        self.width = int(width)
        self.height = int(height)
        self.seed = int(seed)
        self.fastnoise = None
        self.simplex = None
        self._simplex_noise2array = None
        self._rng = np.random.default_rng(self.seed)

        try:
            import opensimplex

            self.simplex = opensimplex.OpenSimplex(seed=self.seed)
            self._simplex_noise2array = getattr(self.simplex, "noise2array", None)
        except ImportError:
            self.simplex = None

        if self.simplex is None:
            try:
                import fastnoiselite

                self.fastnoise = fastnoiselite.FastNoiseLite(seed=self.seed)
                self.fastnoise.noise_type = fastnoiselite.NoiseType.OpenSimplex2
            except ImportError:
                self.fastnoise = None

        self._y, self._x = np.mgrid[0:self.height, 0:self.width]
        self._x_norm = self._x / max(1, self.width - 1)
        self._y_norm = self._y / max(1, self.height - 1)

    def generate(self) -> np.ndarray:
        """Generate normalized elevation in the range [-1, 1]."""
        continental = gaussian_filter(self._continental_base(), sigma=0.9)
        macro = self._fbm(scale=180.0, octaves=4, persistence=0.55, lacunarity=2.05)
        detail = self._fbm(scale=82.0, octaves=5, persistence=0.5, lacunarity=2.0, offset=19.7)
        ridges = self._ridged_noise(scale=58.0, octaves=4, offset=41.0)
        warp_x = self._fbm(scale=240.0, octaves=2, persistence=0.58, lacunarity=2.0, offset=13.0)
        warp_y = self._fbm(scale=210.0, octaves=2, persistence=0.58, lacunarity=2.0, offset=53.0)
        warped = self._sample_warped_noise(warp_x, warp_y)
        land_profile = self._signed_mask(continental, midpoint=0.5, gain=3.5)
        shelf = gaussian_filter(np.clip(continental - 0.36, 0.0, 1.0), sigma=4.8)
        continental_core = self._signed_mask(gaussian_filter(continental, sigma=2.4), midpoint=0.56, gain=4.5)
        shoreline = self._shoreline_profile(continental, water_level=0.48, gain=1.4)
        basin_noise = self._fbm(scale=320.0, octaves=3, persistence=0.57, lacunarity=2.0, offset=91.0)

        elevation = (
            land_profile * 0.4
            + continental_core * 0.12
            + macro * 0.15
            + detail * 0.08
            + ridges * 0.12
            + warped * 0.06
            + shelf * 0.1
            + shoreline * 0.03
        )
        elevation -= (1.0 - shelf) * (0.15 + basin_noise * 0.08)
        elevation = gaussian_filter(elevation, sigma=1.45)
        elevation = np.tanh(elevation * 1.42).astype(np.float32)
        elevation = self._rebalance_sea_level(elevation, target_ocean=0.56)
        elevation = self.apply_edge_smoothing(elevation, border_width=0.14)
        elevation = self.apply_coastal_smoothing(elevation, sea_level=0.0, smoothing_radius=6)
        elevation = self._naturalize_relief(elevation, sea_level=0.0, coast_sigma=5.5)
        elevation = self._relax_land_edges(elevation, sea_level=0.0, coastal_blend=0.42, distance_gain=1.12)
        elevation = np.tanh(elevation * 0.88).astype(np.float32)
        elevation = self._enforce_longitude_wrap(elevation, band_ratio=0.1)
        return np.clip(elevation, -1.0, 1.0).astype(np.float32)

    def apply_constraints(self, elev: np.ndarray, constraints: Dict[str, Any]) -> np.ndarray:
        """Apply continent, mountain, and sea-zone constraints with strong shape influence."""
        result = gaussian_filter(elev.astype(np.float32), sigma=1.0)
        land_masks = []
        continent_fields = []

        for continent in constraints.get("continents", []):
            position = continent.get("position", "center")
            size = float(continent.get("size", 0.45))
            mask = self._create_continent_mask(position, size)
            land_masks.append(mask)
            uplift = self._normalize(gaussian_filter(mask, sigma=max(5.0, min(self.width, self.height) * 0.018)))
            shelf = self._normalize(gaussian_filter(mask, sigma=max(7.0, min(self.width, self.height) * 0.028)))
            land_target = self._signed_mask(uplift, midpoint=0.31, gain=4.1) * 0.88
            land_target += self._signed_mask(shelf, midpoint=0.26, gain=2.4) * 0.24
            land_target += gaussian_filter(mask, sigma=3.6).astype(np.float32) * 0.1
            continent_fields.append(land_target.astype(np.float32))
            blend = np.clip((mask**0.88) * 0.96, 0.0, 0.96)
            result = result * (1.0 - blend) + land_target * blend

        if land_masks:
            combined_land = np.maximum.reduce(land_masks).astype(np.float32)
            coast = self._normalize(gaussian_filter(combined_land, sigma=4.2))
            open_ocean = np.clip(1.0 - gaussian_filter(combined_land, sigma=7.5), 0.0, 1.0)
            land_support = self._signed_mask(coast, midpoint=0.34, gain=3.6)
            continental_core = self._signed_mask(gaussian_filter(combined_land, sigma=2.6), midpoint=0.29, gain=4.1)
            result = result * (0.76 + coast * 0.18) + land_support * 0.3 + continental_core * 0.16 - open_ocean * 0.24

            if len(land_masks) >= 2:
                sea_style = constraints.get("sea_style", "open")
                width_hint = 0.055 if sea_style == "strait" else (0.12 if sea_style == "open" else 0.085)
                separators = np.zeros_like(result)
                for left in range(len(land_masks)):
                    for right in range(left + 1, len(land_masks)):
                        corridor = self._sea_corridor_mask(land_masks[left], land_masks[right], width_hint)
                        separators = np.maximum(separators, corridor)
                result -= gaussian_filter(separators, sigma=5.4) * 0.24
        else:
            combined_land = np.clip(self._normalize(result + 1.0), 0.0, 1.0)

        for zone in constraints.get("sea_zones", []):
            mask = self._create_location_mask(zone, radius=0.32, sigma=0.62)
            trench = self._normalize(gaussian_filter(mask, sigma=4.8))
            result -= trench * 0.56
            result -= self._signed_mask(trench, midpoint=0.18, gain=2.6) * 0.1

        for mountain in constraints.get("mountains", []):
            location = mountain.get("location", "center")
            height = float(mountain.get("height", 0.8))
            chain = self._create_mountain_chain_mask(location, height)
            ridge = self._ridged_noise(scale=38.0, octaves=3, offset=height * 17.0 + len(location))
            ridge = ridge * 2.0 - 1.0
            land_anchor = np.clip(gaussian_filter(np.maximum(combined_land, np.clip(result, 0.0, 1.0)), sigma=2.2), 0.0, 1.0)
            uplift = np.clip(chain * (0.62 + land_anchor * 0.38) * (0.82 + ridge * 0.18), 0.0, 1.0)
            foothills = gaussian_filter(uplift, sigma=3.8).astype(np.float32)
            result += uplift * (0.64 + height * 0.72)
            result += foothills * (0.14 + height * 0.12)
            result -= gaussian_filter(chain * (1.0 - land_anchor), sigma=2.8) * 0.12

        if continent_fields:
            target_land = self._normalize(np.maximum.reduce(continent_fields))
            result = result * 0.78 + self._signed_mask(target_land, midpoint=0.31, gain=3.2) * 0.22

        result = gaussian_filter(result, sigma=1.5)
        result = np.tanh(result * 1.28).astype(np.float32)
        result = self._rebalance_sea_level(result, target_ocean=0.57)
        result = self.apply_edge_smoothing(result, border_width=0.14)
        result = self.apply_coastal_smoothing(result, sea_level=0.0, smoothing_radius=7)
        result = self._naturalize_relief(result, sea_level=0.0, coast_sigma=6.5)
        result = self._relax_land_edges(result, sea_level=0.0, coastal_blend=0.5, distance_gain=1.2)
        result = np.tanh(result * 0.9).astype(np.float32)
        result = self._enforce_longitude_wrap(result, band_ratio=0.1)
        return np.clip(result, -1.0, 1.0).astype(np.float32)

    def reinforce_constraints(
        self,
        elev: np.ndarray,
        constraints: Dict[str, Any],
        blend: float = 0.34,
    ) -> np.ndarray:
        """Re-apply macro terrain intent after erosion so constraints remain visible in the final world."""
        constrained = self.apply_constraints(elev, constraints)
        blend = float(np.clip(blend, 0.0, 0.7))
        result = elev.astype(np.float32) * (1.0 - blend) + constrained.astype(np.float32) * blend
        result = gaussian_filter(result, sigma=0.9)
        result = self._rebalance_sea_level(result, target_ocean=0.57)
        result = self.apply_coastal_smoothing(result, sea_level=0.0, smoothing_radius=5)
        result = self._relax_land_edges(result, sea_level=0.0, coastal_blend=0.34, distance_gain=1.06)
        result = np.tanh(result * 0.92).astype(np.float32)
        result = self._enforce_longitude_wrap(result, band_ratio=0.1)
        return np.clip(result, -1.0, 1.0).astype(np.float32)

    def apply_edge_smoothing(self, elev: np.ndarray, border_width: float = 0.1) -> np.ndarray:
        border_width = max(border_width, 1e-4)
        vertical_distance = np.minimum(self._y_norm, 1.0 - self._y_norm)
        horizontal_distance = np.minimum(self._x_norm, 1.0 - self._x_norm)

        vertical_t = np.clip(vertical_distance / border_width, 0.0, 1.0)
        horizontal_t = np.clip(horizontal_distance / max(border_width * 0.45, 1e-4), 0.0, 1.0)
        vertical_mask = 0.5 - 0.5 * np.cos(np.pi * vertical_t)
        horizontal_mask = 0.5 - 0.5 * np.cos(np.pi * horizontal_t)

        # Treat the world as horizontally wrapping; only the polar bands should strongly taper toward ocean.
        mask = np.clip(vertical_mask * 0.88 + 0.12, 0.0, 1.0)
        ocean_bias = (-0.015 * (1.0 - horizontal_mask)) + (-0.18 * (1.0 - vertical_mask))
        softened = elev * (0.58 + mask * 0.42) + ocean_bias * (1.0 - mask)
        return gaussian_filter(softened, sigma=1.05).astype(np.float32)

    def apply_coastal_smoothing(self, elev: np.ndarray, sea_level: float = 0.0, smoothing_radius: int = 3) -> np.ndarray:
        sigma = max(0.8, smoothing_radius * 0.75)
        smoothed = gaussian_filter(elev, sigma=sigma)
        coast_band = np.exp(-((elev - sea_level) ** 2) / 0.04)
        return (elev * (1.0 - coast_band * 0.6) + smoothed * (coast_band * 0.6)).astype(np.float32)

    def _continental_base(self) -> np.ndarray:
        plates = self._tectonic_plate_field()
        supercontinent = self._tectonic_plate_field()
        low = self._fbm(scale=250.0, octaves=4, persistence=0.58, lacunarity=2.0)
        medium = self._fbm(scale=135.0, octaves=4, persistence=0.52, lacunarity=2.05, offset=27.5)
        ocean_basins = self._fbm(scale=300.0, octaves=2, persistence=0.6, lacunarity=2.0, offset=71.0)
        rifts = self._ridged_noise(scale=220.0, octaves=3, offset=17.0)
        edge_falloff = self._edge_falloff(power=1.7)
        latitudinal_bias = np.cos((self._y_norm - 0.5) * np.pi) ** 2
        equatorial_archipelago = self._fbm(scale=92.0, octaves=3, persistence=0.55, lacunarity=2.2, offset=9.0)
        warp = self._fbm(scale=210.0, octaves=2, persistence=0.6, lacunarity=2.0, offset=123.0)

        continental = plates * 0.34 + supercontinent * 0.23 + low * 0.15 + medium * 0.12 + edge_falloff * 0.04
        continental += gaussian_filter((plates * supercontinent).astype(np.float32), sigma=3.6) * 0.16
        continental += latitudinal_bias.astype(np.float32) * 0.05
        continental += equatorial_archipelago * 0.03
        continental -= (1.0 - ocean_basins) * 0.16
        continental -= rifts * 0.08
        continental -= np.abs(warp - 0.5) * 0.03
        continental = gaussian_filter(continental, sigma=3.4)
        continental = self._shoreline_profile(self._normalize(continental), water_level=0.46, gain=1.2)
        return self._normalize(continental)

    def _tectonic_plate_field(self) -> np.ndarray:
        field = np.zeros((self.height, self.width), dtype=np.float32)
        continent_count = 5 + (self.seed % 3)
        ocean_count = 2 + (self.seed % 2)

        for _ in range(continent_count):
            cy = float(self._rng.uniform(0.16, 0.84))
            cx = float(self._rng.uniform(0.12, 0.88))
            rx = float(self._rng.uniform(0.12, 0.26))
            ry = float(self._rng.uniform(0.1, 0.24))
            rotation = float(self._rng.uniform(-0.85, 0.85))
            weight = float(self._rng.uniform(0.75, 1.2))
            primary = self._elliptic_gaussian(cy, cx, rx, ry, rotation)
            shoulder = self._elliptic_gaussian(
                np.clip(cy + np.sin(rotation) * ry * 0.55, 0.06, 0.94),
                np.clip(cx + np.cos(rotation) * rx * 0.48, 0.04, 0.96),
                rx * 0.74,
                ry * 0.82,
                rotation + float(self._rng.uniform(-0.4, 0.4)),
            )
            tail = self._elliptic_gaussian(
                np.clip(cy - np.sin(rotation) * ry * 0.44, 0.06, 0.94),
                np.clip(cx - np.cos(rotation) * rx * 0.4, 0.04, 0.96),
                rx * 0.6,
                ry * 0.66,
                rotation + float(self._rng.uniform(-0.55, 0.55)),
            )
            plate = np.maximum.reduce([primary, shoulder * 0.86, tail * 0.72])
            field += gaussian_filter(plate.astype(np.float32), sigma=1.1) * weight

        for _ in range(ocean_count):
            cy = float(self._rng.uniform(0.18, 0.82))
            cx = float(self._rng.uniform(0.16, 0.84))
            rx = float(self._rng.uniform(0.15, 0.32))
            ry = float(self._rng.uniform(0.12, 0.28))
            rotation = float(self._rng.uniform(-1.1, 1.1))
            weight = float(self._rng.uniform(0.35, 0.7))
            field -= self._elliptic_gaussian(cy, cx, rx, ry, rotation) * weight

        field += self._fbm(scale=120.0, octaves=2, persistence=0.5, lacunarity=2.0, offset=37.0) * 0.16
        field -= self._ridged_noise(scale=165.0, octaves=2, offset=83.0) * 0.08
        return self._normalize(gaussian_filter(field, sigma=3.2))

    def _sample_warped_noise(self, warp_x: np.ndarray, warp_y: np.ndarray) -> np.ndarray:
        x = np.mod(np.rint(self._x + warp_x * 16.0).astype(np.int32), max(self.width, 1))
        y = np.clip(self._y + warp_y * 16.0, 0, self.height - 1).astype(np.int32)
        base = self._fbm(scale=92.0, octaves=3, persistence=0.52, lacunarity=2.0, offset=51.0)
        return base[y, x]

    def _fbm(
        self,
        scale: float,
        octaves: int,
        persistence: float,
        lacunarity: float,
        offset: float = 0.0,
    ) -> np.ndarray:
        accum = np.zeros((self.height, self.width), dtype=np.float32)
        amplitude = 1.0
        frequency = 1.0 / max(scale, 1.0)
        amplitude_sum = 0.0

        for octave in range(octaves):
            sample = self._noise(self._x * frequency + offset + octave * 13.37, self._y * frequency + offset)
            accum += sample.astype(np.float32) * amplitude
            amplitude_sum += amplitude
            amplitude *= persistence
            frequency *= lacunarity

        return self._normalize(accum / max(amplitude_sum, 1e-6))

    def _ridged_noise(self, scale: float, octaves: int, offset: float = 0.0) -> np.ndarray:
        base = self._fbm(scale=scale, octaves=octaves, persistence=0.58, lacunarity=2.15, offset=offset)
        ridged = 1.0 - np.abs(base * 2.0 - 1.0)
        ridged = ridged**1.7
        return self._normalize(ridged)

    def _noise(self, x_coords: np.ndarray, y_coords: np.ndarray) -> np.ndarray:
        x_array = np.asarray(x_coords, dtype=np.float32)
        y_array = np.asarray(y_coords, dtype=np.float32)

        if self._simplex_noise2array is not None and x_array.ndim == 2 and y_array.ndim == 2:
            try:
                x_axis = np.asarray(x_array[0, :], dtype=np.float64)
                y_axis = np.asarray(y_array[:, 0], dtype=np.float64)
                values = self._simplex_noise2array(x_axis, y_axis)
                return np.asarray(values, dtype=np.float32)
            except Exception:
                self._simplex_noise2array = None

        return self._value_noise(x_array, y_array)

    def _value_noise(self, x_coords: np.ndarray, y_coords: np.ndarray) -> np.ndarray:
        x0 = np.floor(x_coords).astype(np.int32)
        y0 = np.floor(y_coords).astype(np.int32)
        x1 = x0 + 1
        y1 = y0 + 1

        xf = x_coords - x0
        yf = y_coords - y0
        u = xf * xf * (3.0 - 2.0 * xf)
        v = yf * yf * (3.0 - 2.0 * yf)

        n00 = self._hash_noise(x0, y0)
        n10 = self._hash_noise(x1, y0)
        n01 = self._hash_noise(x0, y1)
        n11 = self._hash_noise(x1, y1)

        nx0 = n00 + (n10 - n00) * u
        nx1 = n01 + (n11 - n01) * u
        return (nx0 + (nx1 - nx0) * v).astype(np.float32)

    def _hash_noise(self, x_coords: np.ndarray, y_coords: np.ndarray) -> np.ndarray:
        x = np.asarray(x_coords, dtype=np.uint32)
        y = np.asarray(y_coords, dtype=np.uint32)
        value = x * np.uint32(374761393) + y * np.uint32(668265263) + np.uint32(self.seed * 1442695041)
        value = (value ^ (value >> np.uint32(13))) * np.uint32(1274126177)
        value = value ^ (value >> np.uint32(16))
        return (value.astype(np.float32) / np.float32(np.iinfo(np.uint32).max)) * 2.0 - 1.0

    def _fallback_noise(self, x_coords: np.ndarray, y_coords: np.ndarray) -> np.ndarray:
        return (
            np.sin((x_coords + self.seed) * 0.11)
            + np.cos((y_coords - self.seed) * 0.09)
            + np.sin((x_coords + y_coords) * 0.05 + self.seed * 0.17)
        ).astype(np.float32) / 3.0

    def _create_location_mask(self, location: str, radius: float = 0.22, sigma: float = 0.45) -> np.ndarray:
        center_y, center_x = self._resolve_position(location)
        dx = self._wrapped_dx(self._x_norm - center_x)
        dy = self._y_norm - center_y
        distance = np.sqrt(dx * dx + dy * dy) / max(radius, 1e-4)
        return np.exp(-(distance**2) / (2.0 * sigma * sigma)).astype(np.float32)

    def _create_continent_mask(self, position: str, size: float, sigma: float = 0.38) -> np.ndarray:
        size = float(np.clip(size, 0.12, 0.85))
        center_y, center_x = self._resolve_position(position)
        radius_x = max(0.14, size * (0.84 if position in {"west", "east"} else 0.68))
        radius_y = max(0.14, size * (0.84 if position in {"north", "south"} else 0.62))
        rotation = self._position_rotation(position)
        mask = self._elliptic_gaussian(center_y, center_x, radius_x, radius_y, rotation)
        lobe_dx = np.cos(rotation) * radius_x * 0.36
        lobe_dy = np.sin(rotation) * radius_y * 0.36
        secondary = self._elliptic_gaussian(
            np.clip(center_y + lobe_dy, 0.08, 0.92),
            np.clip(center_x + lobe_dx, 0.08, 0.92),
            radius_x * 0.62,
            radius_y * 0.72,
            rotation - 0.4,
        )
        tertiary = self._elliptic_gaussian(
            np.clip(center_y - lobe_dy * 0.8, 0.08, 0.92),
            np.clip(center_x - lobe_dx * 0.8, 0.08, 0.92),
            radius_x * 0.48,
            radius_y * 0.54,
            rotation + 0.55,
        )
        jagged = self._fbm(scale=72.0, octaves=3, persistence=0.52, lacunarity=2.1, offset=len(position) * 11.0)
        shoreline = self._ridged_noise(scale=66.0, octaves=3, offset=len(position) * 9.0)
        archipelago = self._fbm(scale=48.0, octaves=2, persistence=0.48, lacunarity=2.0, offset=len(position) * 4.0)
        combined = np.maximum(mask, secondary * 0.82)
        combined = np.maximum(combined, tertiary * 0.62)
        combined = self._normalize(gaussian_filter(combined, sigma=1.8))
        dx = self._wrapped_dx(self._x_norm - center_x)
        dy = self._y_norm - center_y
        cos_r = np.cos(rotation)
        sin_r = np.sin(rotation)
        x_rot = dx * cos_r - dy * sin_r
        y_rot = dx * sin_r + dy * cos_r
        radial = np.sqrt((x_rot / max(radius_x, 1e-4)) ** 2 + (y_rot / max(radius_y, 1e-4)) ** 2)
        coastal_variation = (jagged * 0.14 + shoreline * 0.1 + archipelago * 0.08) - 0.14
        shelf = gaussian_filter(combined, sigma=max(3.2, min(self.width, self.height) * 0.016))
        contour = 1.0 - radial + combined * 0.72 + shelf * 0.34 + coastal_variation
        contour = gaussian_filter(contour, sigma=max(2.4, min(self.width, self.height) * 0.01))
        shoreline_profile = self._sigmoid((contour - 0.52) * 5.8)
        shoreline_profile = shoreline_profile * (0.82 + gaussian_filter(archipelago, sigma=1.2) * 0.18)
        return gaussian_filter(
            shoreline_profile.astype(np.float32),
            sigma=max(5.2, min(self.width, self.height) * 0.024),
        )

    def _create_mountain_chain_mask(self, location: str, height: float = 0.78) -> np.ndarray:
        center_y, center_x = self._resolve_position(location)
        orientation = self._position_rotation(location)
        h_scale = 0.6 + 0.6 * min(height, 1.5)
        primary = self._elliptic_gaussian(center_y, center_x, 0.26 * h_scale, 0.08 * h_scale, orientation)
        secondary = self._elliptic_gaussian(center_y, center_x, 0.32 * h_scale, 0.06 * h_scale, orientation + 0.45) * 0.55
        tail = self._elliptic_gaussian(
            np.clip(center_y + np.sin(orientation) * 0.12 * h_scale, 0.08, 0.92),
            np.clip(center_x + np.cos(orientation) * 0.12 * h_scale, 0.08, 0.92),
            0.21 * h_scale,
            0.055 * h_scale,
            orientation - 0.18,
        ) * 0.62
        warped = self._fbm(scale=56.0, octaves=3, persistence=0.54, lacunarity=2.0, offset=len(location) * 7.0)
        chain = np.clip(np.maximum.reduce([primary, secondary, tail]) * (0.78 + warped * 0.42), 0.0, 1.0)
        return gaussian_filter(chain.astype(np.float32), sigma=max(1.4, min(self.width, self.height) * 0.008))

    def _sea_corridor_mask(self, left_mask: np.ndarray, right_mask: np.ndarray, width_hint: float = 0.085) -> np.ndarray:
        left_y, left_x = self._mask_centroid(left_mask)
        right_y, right_x = self._mask_centroid(right_mask)
        dx = right_x - left_x
        dy = right_y - left_y
        denom = dx * dx + dy * dy
        if denom < 1e-6:
            return np.zeros_like(left_mask, dtype=np.float32)

        projection = ((self._x_norm - left_x) * dx + (self._y_norm - left_y) * dy) / denom
        projection = np.clip(projection, 0.0, 1.0)
        closest_x = left_x + projection * dx
        closest_y = left_y + projection * dy
        distance = np.sqrt(self._wrapped_dx(self._x_norm - closest_x) ** 2 + (self._y_norm - closest_y) ** 2)
        corridor = np.exp(-(distance**2) / (2.0 * width_hint * width_hint))
        ocean_window = np.clip(1.0 - np.maximum(left_mask, right_mask), 0.0, 1.0)
        return gaussian_filter((corridor * ocean_window).astype(np.float32), sigma=2.0)

    @staticmethod
    def _mask_centroid(mask: np.ndarray) -> Tuple[float, float]:
        total = float(np.sum(mask))
        if total <= 1e-6:
            return 0.5, 0.5
        height, width = mask.shape
        y_idx, x_idx = np.mgrid[0:height, 0:width]
        center_y = float(np.sum((y_idx / max(1, height - 1)) * mask) / total)
        center_x = float(np.sum((x_idx / max(1, width - 1)) * mask) / total)
        return center_y, center_x

    def _elliptic_gaussian(
        self,
        center_y: float,
        center_x: float,
        radius_x: float,
        radius_y: float,
        rotation: float,
    ) -> np.ndarray:
        dx = self._wrapped_dx(self._x_norm - center_x)
        dy = self._y_norm - center_y
        cos_r = np.cos(rotation)
        sin_r = np.sin(rotation)
        x_rot = dx * cos_r - dy * sin_r
        y_rot = dx * sin_r + dy * cos_r
        return np.exp(-((x_rot / max(radius_x, 1e-4)) ** 2 + (y_rot / max(radius_y, 1e-4)) ** 2) / 2.0).astype(np.float32)

    @staticmethod
    def _wrapped_dx(delta_x: np.ndarray) -> np.ndarray:
        return ((delta_x + 0.5) % 1.0) - 0.5

    def _edge_falloff(self, power: float = 1.8) -> np.ndarray:
        dy = np.abs(self._y_norm * 2.0 - 1.0)
        distance = dy
        falloff = np.clip(1.0 - distance**power, 0.0, 1.0)
        return gaussian_filter(falloff, sigma=1.8).astype(np.float32)

    @staticmethod
    def _sigmoid(values: np.ndarray) -> np.ndarray:
        return (1.0 / (1.0 + np.exp(-values))).astype(np.float32)

    @classmethod
    def _signed_mask(cls, values: np.ndarray, midpoint: float = 0.5, gain: float = 4.0) -> np.ndarray:
        normalized = cls._normalize(values)
        return np.tanh((normalized - midpoint) * gain).astype(np.float32)

    @classmethod
    def _shoreline_profile(cls, values: np.ndarray, water_level: float = 0.5, gain: float = 2.0) -> np.ndarray:
        normalized = cls._normalize(values)
        expanded = cls._sigmoid((normalized - water_level) * gain * 4.0)
        return gaussian_filter(expanded, sigma=1.1).astype(np.float32)

    @classmethod
    def _rebalance_sea_level(cls, elev: np.ndarray, target_ocean: float = 0.56) -> np.ndarray:
        sea_level = float(np.quantile(elev, np.clip(target_ocean, 0.3, 0.7)))
        shifted = elev.astype(np.float32) - sea_level
        scale = max(float(np.max(np.abs(shifted))), 1e-6)
        return np.clip(shifted / scale, -1.0, 1.0).astype(np.float32)

    def _enforce_longitude_wrap(self, elev: np.ndarray, band_ratio: float = 0.06) -> np.ndarray:
        band = max(4, int(self.width * band_ratio))
        band = min(band, max(4, self.width // 6))
        if band * 2 >= self.width:
            return elev.astype(np.float32)

        wrapped = elev.astype(np.float32)
        padded = np.concatenate([wrapped[:, -band:], wrapped, wrapped[:, :band]], axis=1)
        periodic = gaussian_filter(padded, sigma=(0.0, 0.85)).astype(np.float32)[:, band:-band]

        weights = np.zeros_like(wrapped, dtype=np.float32)
        ramp = np.linspace(1.0, 0.0, band, dtype=np.float32)
        weights[:, :band] = ramp
        weights[:, -band:] = ramp[::-1]

        result = wrapped * (1.0 - weights * 0.42) + periodic * (weights * 0.42)
        seam = (result[:, 0] + result[:, -1]) * 0.5
        result[:, 0] = seam
        result[:, -1] = seam
        return result.astype(np.float32)

    def _position_rotation(self, position: str) -> float:
        normalized = (position or "center").lower().replace("-", "")
        mapping = {
            "northwest": -0.55,
            "north": 0.05,
            "northeast": 0.55,
            "west": -0.25,
            "center": 0.2,
            "east": 0.25,
            "southwest": 0.55,
            "south": -0.05,
            "southeast": -0.55,
            "central": 0.2,
        }
        return mapping.get(normalized, 0.2)

    def _resolve_position(self, position: str) -> Tuple[float, float]:
        from app.core.semantic_mapper import resolve_position_continuous
        return resolve_position_continuous(position)

    @staticmethod
    def _normalize(values: np.ndarray) -> np.ndarray:
        min_value = float(np.min(values))
        max_value = float(np.max(values))
        if max_value - min_value < 1e-8:
            return np.zeros_like(values, dtype=np.float32)
        return ((values - min_value) / (max_value - min_value)).astype(np.float32)

    @classmethod
    def _normalize_signed(cls, values: np.ndarray, midpoint: float = 0.5, gain: float = 2.0) -> np.ndarray:
        normalized = cls._normalize(values)
        return np.tanh((normalized - midpoint) * gain).astype(np.float32)

    def _naturalize_relief(
        self,
        elev: np.ndarray,
        sea_level: float = 0.0,
        coast_sigma: float = 5.0,
    ) -> np.ndarray:
        land_mask = elev > sea_level
        pixel_scale = max(1.0, min(self.width, self.height) * 0.08)
        distance_land = distance_transform_edt(land_mask).astype(np.float32) / pixel_scale
        distance_ocean = distance_transform_edt(~land_mask).astype(np.float32) / pixel_scale
        signed_distance = np.clip(distance_land - distance_ocean, -1.6, 1.6)
        coast_band = np.exp(-(signed_distance**2) / max(0.2, coast_sigma / max(min(self.width, self.height), 1)))
        smooth = gaussian_filter(elev, sigma=1.5)
        shaped = np.tanh((elev * 0.72 + signed_distance * 0.28) * 1.15).astype(np.float32)
        return (
            shaped * (1.0 - coast_band * 0.32) + smooth.astype(np.float32) * (coast_band * 0.32)
        ).astype(np.float32)

    def _relax_land_edges(
        self,
        elev: np.ndarray,
        sea_level: float = 0.0,
        coastal_blend: float = 0.45,
        distance_gain: float = 1.1,
    ) -> np.ndarray:
        land_mask = elev > sea_level
        pixel_scale = max(1.0, min(self.width, self.height) * 0.06)
        distance_land = distance_transform_edt(land_mask).astype(np.float32) / pixel_scale
        distance_ocean = distance_transform_edt(~land_mask).astype(np.float32) / pixel_scale
        signed_distance = np.clip(distance_land - distance_ocean, -2.4, 2.4)
        coast_band = np.exp(-(signed_distance**2) / 0.26).astype(np.float32)
        relaxed_profile = np.tanh(signed_distance * distance_gain).astype(np.float32)
        smoothed = gaussian_filter(elev, sigma=1.1).astype(np.float32)
        blended = elev * (1.0 - coast_band * coastal_blend) + relaxed_profile * (coast_band * coastal_blend)
        return (blended * 0.92 + smoothed * 0.08).astype(np.float32)
