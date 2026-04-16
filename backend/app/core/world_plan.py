from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

from app.core.llm_parser import MapConstraints, WorldProfile
from app.core.terrain_modular import ALLOWED_MODULE_NAMES, ALLOWED_MODULE_OPERATIONS


class Continent(BaseModel):
    position: str
    size: float = 0.3


class Mountain(BaseModel):
    location: str
    height: float = 0.7
    orientation: str | None = None


class IslandChain(BaseModel):
    position: str
    density: float = 0.5


class Peninsula(BaseModel):
    location: str
    size: float = 0.2


class InlandSea(BaseModel):
    position: str
    connection: str | None = None


class RiverHint(BaseModel):
    region: str
    length: str = "long"


class WaterBody(BaseModel):
    type: str = "ocean"
    position: str = "center"
    coverage: float = 0.25
    connection: str | None = None


class RegionalRelation(BaseModel):
    relation: str
    subject: str
    object: str
    strength: float = 1.0


class GenerationModuleSpec(BaseModel):
    module: str
    enabled: bool = True
    params: dict = Field(default_factory=dict)

    @field_validator("module")
    @classmethod
    def validate_module(cls, value: str) -> str:
        normalized = str(value).strip().lower()
        if normalized not in ALLOWED_MODULE_NAMES:
            allowed = ", ".join(sorted(ALLOWED_MODULE_NAMES))
            raise ValueError(f"Unsupported terrain module '{value}'. Allowed modules: {allowed}")
        return normalized

    @field_validator("params")
    @classmethod
    def validate_params(cls, value: dict) -> dict:
        params = dict(value or {})
        operation = str(params.get("operation", "add")).strip().lower()
        if operation not in ALLOWED_MODULE_OPERATIONS:
            allowed = ", ".join(sorted(ALLOWED_MODULE_OPERATIONS))
            raise ValueError(f"Unsupported terrain operation '{operation}'. Allowed operations: {allowed}")
        params["operation"] = operation
        return params

    @field_validator("params")
    @classmethod
    def validate_module_specific_params(cls, value: dict, info) -> dict:
        params = dict(value or {})
        module = str(info.data.get("module", "")).strip().lower()
        validators = {
            "noise": cls._validate_noise_like_params,
            "ridged_noise": cls._validate_noise_like_params,
            "continent": cls._validate_continent_params,
            "gaussian_mountain": cls._validate_mountain_params,
            "ridge": cls._validate_ridge_params,
            "water_body": cls._validate_water_params,
            "strait": cls._validate_water_params,
            "plateau": cls._validate_plateau_params,
            "smooth": cls._validate_smooth_params,
        }
        validator = validators.get(module)
        if validator is not None:
            validator(params, module)
        return params

    @staticmethod
    def _require_string(params: dict[str, Any], key: str, module: str) -> None:
        value = params.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Module '{module}' requires a non-empty string parameter '{key}'")

    @staticmethod
    def _require_number(
        params: dict[str, Any],
        key: str,
        module: str,
        minimum: float | None = None,
        maximum: float | None = None,
    ) -> None:
        value = params.get(key)
        if not isinstance(value, int | float):
            raise ValueError(f"Module '{module}' requires numeric parameter '{key}'")
        numeric = float(value)
        if minimum is not None and numeric < minimum:
            raise ValueError(f"Module '{module}' parameter '{key}' must be >= {minimum}")
        if maximum is not None and numeric > maximum:
            raise ValueError(f"Module '{module}' parameter '{key}' must be <= {maximum}")

    @classmethod
    def _validate_noise_like_params(cls, params: dict[str, Any], module: str) -> None:
        cls._require_number(params, "scale", module, minimum=1.0)
        cls._require_number(params, "octaves", module, minimum=1.0, maximum=8.0)

    @classmethod
    def _validate_continent_params(cls, params: dict[str, Any], module: str) -> None:
        cls._require_string(params, "position", module)
        cls._require_number(params, "size", module, minimum=0.05, maximum=1.0)

    @classmethod
    def _validate_mountain_params(cls, params: dict[str, Any], module: str) -> None:
        cls._require_string(params, "location", module)
        cls._require_number(params, "height", module, minimum=0.0, maximum=2.0)

    @classmethod
    def _validate_ridge_params(cls, params: dict[str, Any], module: str) -> None:
        cls._require_string(params, "location", module)
        cls._require_number(params, "height", module, minimum=0.0, maximum=2.0)

    @classmethod
    def _validate_water_params(cls, params: dict[str, Any], module: str) -> None:
        cls._require_string(params, "position", module)
        cls._require_number(params, "coverage", module, minimum=0.01, maximum=1.0)

    @classmethod
    def _validate_plateau_params(cls, params: dict[str, Any], module: str) -> None:
        cls._require_string(params, "position", module)
        cls._require_number(params, "height", module, minimum=0.0, maximum=1.0)

    @classmethod
    def _validate_smooth_params(cls, params: dict[str, Any], module: str) -> None:
        cls._require_number(params, "sigma", module, minimum=0.1, maximum=10.0)


class WorldPlan(BaseModel):
    prompt: str
    summary: str
    constraints: MapConstraints = Field(default_factory=MapConstraints)
    profile: WorldProfile = Field(default_factory=WorldProfile)
    generation_backend: str = "gaussian_voronoi"
    continents: list[Continent] = Field(default_factory=list)
    mountains: list[Mountain] = Field(default_factory=list)
    island_chains: list[IslandChain] = Field(default_factory=list)
    peninsulas: list[Peninsula] = Field(default_factory=list)
    inland_seas: list[InlandSea] = Field(default_factory=list)
    river_hints: list[RiverHint] = Field(default_factory=list)
    water_bodies: list[WaterBody] = Field(default_factory=list)
    regional_relations: list[RegionalRelation] = Field(default_factory=list)
    module_sequence: list[GenerationModuleSpec] = Field(default_factory=list)
    climate_hints: list[str] = Field(default_factory=list)
    rag_meta: dict = Field(default_factory=dict)
