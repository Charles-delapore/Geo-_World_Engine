from __future__ import annotations

import logging

from app.config import settings
from app.core.llm_parser import _build_module_sequence, get_constraints_summary, parse_with_rag
from app.core.world_plan import (
    Continent,
    InlandSea,
    IslandChain,
    Mountain,
    Peninsula,
    RegionalRelation,
    RiverHint,
    WaterBody,
    TopologyIntent,
    GenerationModuleSpec,
    WorldPlan,
)
from app.rag.init_kb import init_builtin_knowledge_base
from app.rag.retriever import RecipeRetriever
from app.rag.failure_cases import FailureCaseDB
from app.utils.metrics import rag_parse_counter


logger = logging.getLogger(__name__)
logger.setLevel(getattr(logging, settings.RAG_LOG_LEVEL.upper(), logging.INFO))

_retriever: RecipeRetriever | None = None
_failure_db: FailureCaseDB | None = None
if settings.ENABLE_RAG:
    try:
        init_builtin_knowledge_base(force=False)
        _retriever = RecipeRetriever()
        _failure_db = FailureCaseDB()
        logger.info("RAG retriever initialized, failure_cases=%d", _failure_db.count())
    except Exception as exc:
        logger.warning("RAG init failed, running without retrieval: %s", exc)


def build_world_plan(prompt: str, params: dict) -> WorldPlan:
    examples: list[dict] = []
    rag_meta = {"enabled": settings.ENABLE_RAG and _retriever is not None}
    if _retriever is not None:
        examples, meta = _retriever.retrieve_for_prompt(prompt)
        rag_meta.update(meta)
        logger.info("RAG prompt=%s meta=%s", prompt[:80], rag_meta)

    parsed = parse_with_rag(
        user_prompt=prompt,
        examples=examples,
        api_key=params.get("llm_api_key") or settings.OPENAI_API_KEY,
        base_url=params.get("llm_base_url") or settings.OPENAI_BASE_URL,
        model=params.get("llm_model") or settings.OPENAI_MODEL,
    )

    if _failure_db is not None:
        matched_failures = _failure_db.find_by_prompt(prompt)
        if matched_failures:
            for fc in matched_failures:
                fix = fc.get("fix", {})
                if fix.get("topology_intent") and parsed.get("topology_intent"):
                    for k, v in fix["topology_intent"].items():
                        if k not in parsed["topology_intent"] or parsed["topology_intent"][k] is None:
                            parsed["topology_intent"][k] = v
                if fix.get("modifiers") and parsed.get("topology_intent"):
                    modifiers = parsed["topology_intent"].setdefault("modifiers", {})
                    for k, v in fix["modifiers"].items():
                        if k not in modifiers:
                            modifiers[k] = v
            rag_meta["failure_case_matches"] = len(matched_failures)
            logger.info("Applied %d failure case fixes for prompt=%s", len(matched_failures), prompt[:50])
    requested_modules = list(params.get("module_sequence") or [])
    requested_backend = str(params.get("generation_backend") or "").strip().lower()
    if requested_modules and not requested_backend:
        requested_backend = "modular"
    if requested_backend in {"gaussian_voronoi", "modular"}:
        parsed["generation_backend"] = requested_backend
        if requested_backend == "gaussian_voronoi":
            parsed["module_sequence"] = []
        elif requested_modules:
            parsed["module_sequence"] = requested_modules
        elif not parsed.get("module_sequence"):
            constraints_obj = parsed["constraints"]
            profile_obj = parsed["profile"]
            if isinstance(constraints_obj, dict):
                from app.core.llm_parser import MapConstraints

                constraints_obj = MapConstraints(**constraints_obj)
            if isinstance(profile_obj, dict):
                from app.core.llm_parser import WorldProfile

                profile_obj = WorldProfile(**profile_obj)
            parsed["module_sequence"] = _build_module_sequence(prompt, constraints_obj, profile_obj, requested_backend)
    constraints = parsed["constraints"] if isinstance(parsed["constraints"], dict) else parsed["constraints"].model_dump(mode="json")
    profile = parsed["profile"] if isinstance(parsed["profile"], dict) else parsed["profile"].model_dump(mode="json")
    plan = WorldPlan(
        prompt=prompt,
        summary="",
        constraints=constraints,
        profile=profile,
        generation_backend=str(parsed.get("generation_backend") or "gaussian_voronoi"),
        continents=[Continent(**item) for item in parsed.get("continents") or []],
        mountains=[Mountain(**item) for item in parsed.get("mountains") or []],
        island_chains=[IslandChain(**item) for item in parsed.get("island_chains") or []],
        peninsulas=[Peninsula(**item) for item in parsed.get("peninsulas") or []],
        inland_seas=[InlandSea(**item) for item in parsed.get("inland_seas") or []],
        river_hints=[RiverHint(**item) for item in parsed.get("river_hints") or []],
        water_bodies=[WaterBody(**item) for item in parsed.get("water_bodies") or []],
        regional_relations=[RegionalRelation(**item) for item in parsed.get("regional_relations") or []],
        topology_intent=TopologyIntent(**parsed["topology_intent"]) if parsed.get("topology_intent") else None,
        module_sequence=[GenerationModuleSpec(**item) for item in parsed.get("module_sequence") or []],
        climate_hints=list(parsed.get("climate_hints") or []),
        rag_meta=rag_meta,
    )

    summary = get_constraints_summary(plan.constraints)
    summary = (
        f"{summary}\n"
        f"Backend: {plan.generation_backend}, water_bodies={len(plan.water_bodies)}, "
        f"regional_relations={len(plan.regional_relations)}, modules={len(plan.module_sequence)}, "
        f"topology={plan.topology_intent.kind if plan.topology_intent else 'none'}, "
        f"modifiers={plan.topology_intent.modifiers if plan.topology_intent else {}}\n"
        f"Profile: layout={plan.profile.layout_template}, land={plan.profile.land_ratio:.2f}, "
        f"sea={plan.profile.sea_style}, ruggedness={plan.profile.ruggedness:.2f}, coast={plan.profile.coast_complexity:.2f}, "
        f"moisture={plan.profile.moisture:.2f}, palette={plan.profile.palette_hint}\n"
        f"RAG: enabled={rag_meta.get('enabled')}, examples={len(examples)}, top_similarity={rag_meta.get('top_similarity')}, fallback={rag_meta.get('fallback_reason')}"
    )
    plan.summary = summary
    rag_parse_counter.labels(
        success="true",
        fallback=str(bool(rag_meta.get("fallback_reason"))).lower(),
        examples_count=str(len(examples)),
    ).inc()

    plan = _critic_validate_plan(plan, prompt, rag_meta)

    return plan


def _critic_validate_plan(plan: "WorldPlan", prompt: str, rag_meta: dict) -> "WorldPlan":
    from app.core.topology_validator import validate_world_plan, auto_fix_world_plan
    from app.core.geometry_metrics import count_components

    plan_dict = plan.model_dump(mode="json")
    issues = validate_world_plan(plan_dict)
    if not issues:
        return plan

    logger.info("Critic found %d issues: %s", len(issues), issues)
    fixed = auto_fix_world_plan(plan_dict)

    ti = fixed.get("topology_intent") or {}
    kind = str(ti.get("kind", "")).lower()
    if kind == "single_island":
        if ti.get("target_land_component_count") is None:
            ti["target_land_component_count"] = 1
        if not ti.get("forbid_cross_cut"):
            ti["forbid_cross_cut"] = True
    elif kind == "two_continents_with_rift_sea":
        if ti.get("target_land_component_count") is None:
            ti["target_land_component_count"] = 2
        if not ti.get("forbid_cross_cut"):
            ti["forbid_cross_cut"] = True
    elif kind == "archipelago_chain":
        if ti.get("min_land_component_count") is None:
            ti["min_land_component_count"] = 3

    profile = fixed.get("profile") or {}
    land_ratio = float(profile.get("land_ratio", 0.5))
    if kind == "single_island" and land_ratio > 0.65:
        profile["land_ratio"] = 0.55
        logger.info("Critic: capped land_ratio=%.2f to 0.55 for single_island", land_ratio)
    elif kind == "archipelago_chain" and land_ratio > 0.55:
        profile["land_ratio"] = 0.45
        logger.info("Critic: capped land_ratio=%.2f to 0.45 for archipelago", land_ratio)

    recheck = validate_world_plan(fixed)
    if recheck:
        logger.warning("Critic: %d issues remain after auto-fix: %s", len(recheck), recheck)

    try:
        from app.core.world_plan import WorldPlan as WP
        updated = WP(**{k: fixed[k] for k in fixed if k in WP.model_fields})
        updated.summary = plan.summary
        rag_meta["critic_issues"] = len(issues)
        rag_meta["critic_fixes"] = len(issues) - len(recheck)
        return updated
    except Exception as exc:
        logger.warning("Critic: failed to rebuild plan: %s", exc)
        return plan
