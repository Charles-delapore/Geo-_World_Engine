from __future__ import annotations

import logging
from typing import Any

from app.config import settings
from app.rag.knowledge_base import TerrainKnowledgeBase

logger = logging.getLogger(__name__)


class RecipeRetriever:
    def __init__(self):
        self.kb = TerrainKnowledgeBase()
        self.min_similarity = settings.RAG_MIN_SIMILARITY
        self.second_diff_threshold = settings.RAG_SECOND_DIFF_THRESHOLD
        self.top_k = settings.RAG_TOP_K

    def retrieve_for_prompt(self, user_prompt: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        results = self.kb.retrieve_with_distance(user_prompt, top_k=self.top_k)
        meta: dict[str, Any] = {
            "retrieved_count": len(results),
            "top_similarity": None,
            "fallback_reason": None,
            "deduped": False,
        }
        if not results:
            meta["fallback_reason"] = "no_results"
            return [], meta

        meta["top_similarity"] = results[0]["similarity"]
        if results[0]["similarity"] < self.min_similarity:
            meta["fallback_reason"] = f"low_similarity ({results[0]['similarity']:.3f} < {self.min_similarity:.3f})"
            return [], meta

        if len(results) >= 2 and (results[0]["similarity"] - results[1]["similarity"]) >= self.second_diff_threshold:
            results = results[:1]
            meta["deduped"] = True

        examples = [{"description": result["description"], "world_plan": result["world_plan"]} for result in results]
        return examples, meta
