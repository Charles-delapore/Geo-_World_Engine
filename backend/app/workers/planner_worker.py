from __future__ import annotations

from app.core.llm_parser import get_constraints_summary, infer_world_profile, parse_constraints
from app.core.world_plan import WorldPlan


def build_world_plan(prompt: str, params: dict) -> WorldPlan:
    constraints = parse_constraints(
        prompt,
        api_key=params.get("llm_api_key"),
        base_url=params.get("llm_base_url"),
        model=params.get("llm_model"),
    )
    profile = infer_world_profile(prompt, constraints)
    summary = get_constraints_summary(constraints)
    summary = (
        f"{summary}\n"
        f"Profile: layout={profile.layout_template}, land={profile.land_ratio:.2f}, "
        f"sea={profile.sea_style}, ruggedness={profile.ruggedness:.2f}, coast={profile.coast_complexity:.2f}, "
        f"moisture={profile.moisture:.2f}, palette={profile.palette_hint}"
    )
    return WorldPlan(prompt=prompt, summary=summary, constraints=constraints, profile=profile)
