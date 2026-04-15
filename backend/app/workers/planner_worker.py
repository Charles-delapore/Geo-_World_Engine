from __future__ import annotations

from app.core.llm_parser import get_constraints_summary, parse_constraints
from app.core.world_plan import WorldPlan


def build_world_plan(prompt: str, params: dict) -> WorldPlan:
    constraints = parse_constraints(
        prompt,
        api_key=params.get("llm_api_key"),
        base_url=params.get("llm_base_url"),
        model=params.get("llm_model"),
    )
    summary = get_constraints_summary(constraints)
    return WorldPlan(prompt=prompt, summary=summary, constraints=constraints)
