from __future__ import annotations

from pydantic import BaseModel, Field

from app.core.llm_parser import MapConstraints, WorldProfile


class WorldPlan(BaseModel):
    prompt: str
    summary: str
    constraints: MapConstraints = Field(default_factory=MapConstraints)
    profile: WorldProfile = Field(default_factory=WorldProfile)
