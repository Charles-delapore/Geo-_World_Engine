from __future__ import annotations

import logging

from app.config import settings
from app.core.llm_parser import get_constraints_summary, parse_with_rag
from app.core.world_plan import (
    Continent,
    InlandSea,
    IslandChain,
    Mountain,
    Peninsula,
    RiverHint,
    WorldPlan,
)
from app.rag.init_kb import init_builtin_knowledge_base
from app.rag.retriever import RecipeRetriever
from app.utils.metrics import rag_parse_counter


logger = logging.getLogger(__name__)
logger.setLevel(getattr(logging, settings.RAG_LOG_LEVEL.upper(), logging.INFO))

_retriever: RecipeRetriever | None = None
if settings.ENABLE_RAG:
    try:
        init_builtin_knowledge_base(force=False)
        _retriever = RecipeRetriever()
        logger.info("RAG retriever initialized")
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
    constraints = parsed["constraints"] if isinstance(parsed["constraints"], dict) else parsed["constraints"].model_dump(mode="json")
    profile = parsed["profile"] if isinstance(parsed["profile"], dict) else parsed["profile"].model_dump(mode="json")
    plan = WorldPlan(
        prompt=prompt,
        summary="",
        constraints=constraints,
        profile=profile,
        continents=[Continent(**item) for item in parsed.get("continents") or []],
        mountains=[Mountain(**item) for item in parsed.get("mountains") or []],
        island_chains=[IslandChain(**item) for item in parsed.get("island_chains") or []],
        peninsulas=[Peninsula(**item) for item in parsed.get("peninsulas") or []],
        inland_seas=[InlandSea(**item) for item in parsed.get("inland_seas") or []],
        river_hints=[RiverHint(**item) for item in parsed.get("river_hints") or []],
        climate_hints=list(parsed.get("climate_hints") or []),
        rag_meta=rag_meta,
    )

    summary = get_constraints_summary(plan.constraints)
    summary = (
        f"{summary}\n"
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
    return plan
