from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from app.core.terrain import TerrainGenerator

ALLOWED_MODULE_NAMES = {
    "noise",
    "ridged_noise",
    "continent",
    "gaussian_mountain",
    "ridge",
    "water_body",
    "strait",
    "plateau",
    "smooth",
}
ALLOWED_MODULE_OPERATIONS = {"add", "subtract", "max", "min", "replace"}


class ModularTerrainGenerator:
    """Minimal executable modular terrain backend for the beta P2 path."""

    def __init__(self, width: int, height: int, seed: int):
        self.width = int(width)
        self.height = int(height)
        self.seed = int(seed)
        self.terrain = TerrainGenerator(width=width, height=height, seed=seed)

    def generate(self, plan: dict) -> np.ndarray:
        modules = list((plan or {}).get("module_sequence") or [])
        if not modules:
            modules = self._default_sequence(plan or {})

        field = np.full((self.height, self.width), -0.42, dtype=np.float32)
        for spec in modules:
            module = str(spec.get("module", "")).lower()
            if not module:
                continue
            params = dict(spec.get("params") or {})
            component = self._render_module(module, params)
            operation = str(params.get("operation", "add")).lower()
            weight = float(np.clip(params.get("weight", 1.0), -2.0, 2.0))
            field = self._combine(field, component, operation, weight)

        field = gaussian_filter(field, sigma=1.0).astype(np.float32)
        land_ratio = float(((plan or {}).get("profile") or {}).get("land_ratio", 0.46))
        field = self.terrain._rebalance_sea_level(field, target_ocean=float(np.clip(1.0 - land_ratio, 0.28, 0.78)))
        field = self.terrain.apply_edge_smoothing(field, border_width=0.14)
        field = self.terrain.apply_coastal_smoothing(field, sea_level=0.0, smoothing_radius=5)
        field = np.tanh(field * 1.05).astype(np.float32)
        return np.clip(field, -1.0, 1.0).astype(np.float32)

    def _default_sequence(self, plan: dict) -> list[dict]:
        profile = plan.get("profile") or {}
        constraints = plan.get("constraints") or {}
        continents = list(plan.get("continents") or constraints.get("continents") or [])
        mountains = list(plan.get("mountains") or constraints.get("mountains") or [])
        water_bodies = list(plan.get("water_bodies") or [])
        sequence: list[dict] = [
            {"module": "noise", "params": {"scale": 170.0, "octaves": 4, "amplitude": 0.18, "operation": "add"}},
            {"module": "ridged_noise", "params": {"scale": 72.0, "octaves": 4, "amplitude": 0.16, "operation": "add"}},
        ]

        if continents:
            for continent in continents:
                sequence.append(
                    {
                        "module": "continent",
                        "params": {
                            "position": continent.get("position", "center"),
                            "size": continent.get("size", 0.35),
                            "height": 0.92,
                            "operation": "add",
                        },
                    }
                )

        for mountain in mountains:
            sequence.append(
                {
                    "module": "gaussian_mountain",
                    "params": {
                        "location": mountain.get("location", "center"),
                        "height": mountain.get("height", 0.8),
                        "sigma": 0.14,
                        "operation": "add",
                    },
                }
            )
            sequence.append(
                {
                    "module": "ridge",
                    "params": {
                        "location": mountain.get("location", "center"),
                        "height": mountain.get("height", 0.8),
                        "operation": "add",
                    },
                }
            )

        for water_body in water_bodies:
            body_type = str(water_body.get("type", "ocean")).lower()
            module = "strait" if "strait" in body_type else "water_body"
            sequence.append(
                {
                    "module": module,
                    "params": {
                        "position": water_body.get("position", "center"),
                        "coverage": water_body.get("coverage", 0.25),
                        "depth": 0.88 if "ocean" in body_type else 0.72,
                        "connection": water_body.get("connection"),
                        "operation": "subtract",
                    },
                }
            )

        if str(profile.get("layout_template", "default")) in {"mediterranean", "single_island"}:
            sequence.append(
                {
                    "module": "plateau",
                    "params": {
                        "position": "center" if profile.get("layout_template") == "single_island" else "north",
                        "height": 0.18,
                        "radius_y": 0.16,
                        "radius_x": 0.24,
                        "operation": "add",
                    },
                }
            )

        sequence.append({"module": "smooth", "params": {"sigma": 1.4, "operation": "replace"}})
        return sequence

    def _render_module(self, module: str, params: dict) -> np.ndarray:
        if module == "noise":
            return (self.terrain._fbm(
                scale=float(params.get("scale", 160.0)),
                octaves=int(params.get("octaves", 4)),
                persistence=float(params.get("persistence", 0.55)),
                lacunarity=float(params.get("lacunarity", 2.05)),
                offset=float(params.get("offset", 0.0)),
            ) * 2.0 - 1.0).astype(np.float32) * float(params.get("amplitude", 0.2))

        if module == "ridged_noise":
            return (self.terrain._ridged_noise(
                scale=float(params.get("scale", 64.0)),
                octaves=int(params.get("octaves", 4)),
                offset=float(params.get("offset", 0.0)),
            ) * 2.0 - 1.0).astype(np.float32) * float(params.get("amplitude", 0.18))

        if module == "continent":
            position = str(params.get("position", "center"))
            size = float(np.clip(params.get("size", 0.35), 0.16, 0.72))
            height = float(params.get("height", 0.9))
            mask = self.terrain._create_continent_mask(position, size)
            return self.terrain._signed_mask(mask, midpoint=0.34, gain=3.4) * height

        if module == "gaussian_mountain":
            location = str(params.get("location", "center"))
            sigma = float(np.clip(params.get("sigma", 0.14), 0.05, 0.3))
            height = float(params.get("height", 0.8))
            cy, cx = self.terrain._resolve_position(location)
            peak = self.terrain._elliptic_gaussian(cy, cx, sigma, sigma * 0.66, self.terrain._position_rotation(location))
            return gaussian_filter(peak * (0.48 + height * 0.62), sigma=1.0).astype(np.float32)

        if module == "ridge":
            location = str(params.get("location", "center"))
            height = float(params.get("height", 0.8))
            chain = self.terrain._create_mountain_chain_mask(location)
            detail = self.terrain._ridged_noise(scale=36.0, octaves=3, offset=len(location) * 11.0)
            return gaussian_filter(chain * (0.18 + height * 0.34 + detail * 0.1), sigma=1.2).astype(np.float32)

        if module == "water_body":
            position = str(params.get("position", "center"))
            coverage = float(np.clip(params.get("coverage", 0.25), 0.08, 0.68))
            depth = float(params.get("depth", 0.8))
            radius = 0.12 + coverage * 0.24
            sigma = 0.28 + coverage * 0.54
            basin = self.terrain._create_location_mask(position, radius=radius, sigma=sigma)
            return gaussian_filter(basin * depth, sigma=1.4).astype(np.float32)

        if module == "strait":
            position = str(params.get("position", "center"))
            coverage = float(np.clip(params.get("coverage", 0.14), 0.06, 0.32))
            depth = float(params.get("depth", 0.78))
            cy, cx = self.terrain._resolve_position(position)
            rotation = self.terrain._position_rotation(position)
            channel = self.terrain._elliptic_gaussian(cy, cx, 0.05 + coverage * 0.08, 0.22 + coverage * 0.22, rotation)
            return gaussian_filter(channel * depth, sigma=1.0).astype(np.float32)

        if module == "plateau":
            position = str(params.get("position", "center"))
            height = float(params.get("height", 0.22))
            cy, cx = self.terrain._resolve_position(position)
            radius_y = float(np.clip(params.get("radius_y", 0.14), 0.05, 0.3))
            radius_x = float(np.clip(params.get("radius_x", 0.2), 0.08, 0.36))
            plateau = self.terrain._elliptic_gaussian(cy, cx, radius_y, radius_x, 0.0)
            stepped = np.clip(np.floor(plateau * 4.0) / 4.0, 0.0, 1.0)
            return gaussian_filter(stepped * height, sigma=1.0).astype(np.float32)

        if module == "smooth":
            sigma = float(np.clip(params.get("sigma", 1.2), 0.4, 4.0))
            return gaussian_filter(np.zeros((self.height, self.width), dtype=np.float32), sigma=sigma)

        raise ValueError(f"Unsupported modular terrain module: {module}")

    def _combine(self, field: np.ndarray, component: np.ndarray, operation: str, weight: float) -> np.ndarray:
        if operation == "replace":
            if not np.any(component):
                return gaussian_filter(field, sigma=1.2).astype(np.float32)
            return component.astype(np.float32)
        if operation == "subtract":
            return (field - component * abs(weight)).astype(np.float32)
        if operation == "max":
            return np.maximum(field, component * weight).astype(np.float32)
        if operation == "min":
            return np.minimum(field, component * weight).astype(np.float32)
        return (field + component * weight).astype(np.float32)
