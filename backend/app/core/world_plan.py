from __future__ import annotations

from pydantic import BaseModel, Field

from app.core.llm_parser import MapConstraints, WorldProfile


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


class WorldPlan(BaseModel):
    prompt: str
    summary: str
    constraints: MapConstraints = Field(default_factory=MapConstraints)
    profile: WorldProfile = Field(default_factory=WorldProfile)
    continents: list[Continent] = Field(default_factory=list)
    mountains: list[Mountain] = Field(default_factory=list)
    island_chains: list[IslandChain] = Field(default_factory=list)
    peninsulas: list[Peninsula] = Field(default_factory=list)
    inland_seas: list[InlandSea] = Field(default_factory=list)
    river_hints: list[RiverHint] = Field(default_factory=list)
    climate_hints: list[str] = Field(default_factory=list)
    rag_meta: dict = Field(default_factory=dict)
