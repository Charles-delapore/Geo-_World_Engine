from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field


class TerrainRecipe(BaseModel):
    id: str
    name: str
    description: str
    world_plan: dict
    params: dict | None = None
    tags: list[str] = Field(default_factory=list)
    source: str = "builtin"
    success_count: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
