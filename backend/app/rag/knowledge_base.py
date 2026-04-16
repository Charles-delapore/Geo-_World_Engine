from __future__ import annotations

import hashlib
import math
import re
from collections import Counter

import numpy as np

from app.config import settings
from app.rag.recipe_store import RecipeDB
from app.rag.schemas import TerrainRecipe


class TerrainKnowledgeBase:
    def __init__(self, db_url: str | None = None):
        self.db = RecipeDB(db_url or settings.RECIPE_DB_URL)
        self._recipes: list[TerrainRecipe] = []
        self._vectors: np.ndarray | None = None
        self.reload()

    def reload(self) -> None:
        self._recipes = self.db.list_all()
        self._vectors = self._build_matrix(self._recipes) if self._recipes else None

    def retrieve_with_distance(self, user_prompt: str, top_k: int = 2) -> list[dict]:
        if not self._recipes or self._vectors is None:
            return []

        query_vector = self._embed_text(user_prompt)
        scores = self._vectors @ query_vector
        ranked_indices = np.argsort(scores)[::-1][: max(1, top_k)]
        results: list[dict] = []
        for index in ranked_indices:
            recipe = self._recipes[int(index)]
            similarity = float(scores[int(index)])
            results.append(
                {
                    "id": recipe.id,
                    "name": recipe.name,
                    "description": recipe.description,
                    "world_plan": recipe.world_plan,
                    "similarity": similarity,
                    "source": recipe.source,
                }
            )
        return results

    def _build_matrix(self, recipes: list[TerrainRecipe]) -> np.ndarray:
        if not recipes:
            return np.zeros((0, 512), dtype=np.float32)
        vectors = [self._embed_recipe(recipe) for recipe in recipes]
        return np.vstack(vectors).astype(np.float32)

    def _embed_text(self, text: str, dimensions: int = 512) -> np.ndarray:
        tokens = _tokenize(text)
        counts = Counter(tokens)
        vector = np.zeros(dimensions, dtype=np.float32)
        for token, weight in counts.items():
            bucket = _stable_bucket(token, dimensions)
            vector[bucket] += 1.0 + math.log1p(weight)
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-8:
            return vector
        return (vector / norm).astype(np.float32)

    def _embed_recipe(self, recipe: TerrainRecipe, dimensions: int = 512) -> np.ndarray:
        profile = (recipe.world_plan or {}).get("profile") or {}
        climate_hints = (recipe.world_plan or {}).get("climate_hints") or []
        structured = " ".join(
            [
                recipe.name,
                recipe.description,
                " ".join(recipe.tags),
                profile.get("layout_template", ""),
                profile.get("sea_style", ""),
                " ".join(str(item.get("position", item.get("location", ""))) for item in recipe.world_plan.get("continents", [])),
                " ".join(str(item.get("location", "")) for item in recipe.world_plan.get("mountains", [])),
                " ".join(str(item.get("position", "")) for item in recipe.world_plan.get("inland_seas", [])),
                " ".join(climate_hints),
            ]
        )
        return self._embed_text(structured, dimensions=dimensions)


def _tokenize(text: str) -> list[str]:
    normalized = (text or "").lower()
    latin_tokens = re.findall(r"[a-z0-9]+", normalized)
    cjk_chars = [char for char in normalized if "\u4e00" <= char <= "\u9fff"]
    cjk_bigrams = [f"{cjk_chars[index]}{cjk_chars[index + 1]}" for index in range(len(cjk_chars) - 1)]
    return latin_tokens + cjk_chars + cjk_bigrams + _semantic_alias_tokens(normalized)


def _stable_bucket(token: str, dimensions: int) -> int:
    digest = hashlib.sha1(token.encode("utf-8")).hexdigest()
    return int(digest[:10], 16) % dimensions


def _semantic_alias_tokens(text: str) -> list[str]:
    aliases = {
        "内海": ["inland-sea", "mediterranean", "enclosed-sea"],
        "中间有内海": ["inland-sea", "mediterranean", "central-basin"],
        "中央内海": ["inland-sea", "mediterranean", "central-basin"],
        "中部内海": ["inland-sea", "mediterranean", "central-basin"],
        "被海隔开": ["open-sea", "split-continent"],
        "隔着海": ["open-sea", "split-continent"],
        "中间隔着海": ["open-sea", "split-continent", "ocean-corridor"],
        "中间被海隔开": ["open-sea", "split-continent", "ocean-corridor"],
        "中间被开阔海洋隔开": ["open-sea", "split-continent", "ocean-corridor"],
        "开阔海洋": ["open-sea", "open-ocean"],
        "外海": ["open-sea", "open-ocean"],
        "海峡": ["strait", "narrow-connection"],
        "半岛": ["peninsula"],
        "东侧伸出半岛": ["east-peninsula"],
        "西侧有半岛": ["west-peninsula"],
        "群岛": ["archipelago", "island-chain"],
        "四面环海": ["single-island", "island-continent"],
        "一块四面环海的大陆": ["single-island", "island-continent"],
        "东部大陆": ["east-continent"],
        "西部大陆": ["west-continent"],
        "一东一西两块大陆": ["split-east-west", "dual-continent"],
        "一南一北两块大陆": ["split-north-south", "dual-continent"],
        "北部山脉": ["north-mountains"],
        "东部有山脉": ["east-mountains"],
        "东部大陆有山脉": ["east-mountains", "east-orogen"],
        "西侧有山脉": ["west-mountains", "west-orogen"],
        "南部有山脉": ["south-mountains", "south-rim"],
        "北部有山脉": ["north-mountains", "north-rim"],
        "北缘有山脉": ["north-mountains", "north-rim"],
        "西侧伸出半岛": ["west-peninsula"],
        "超大陆": ["supercontinent"],
        "干旱": ["arid", "dry"],
        "热带": ["tropical"],
        "寒冷": ["frozen", "cold"],
    }
    derived: list[str] = []
    for phrase, tokens in aliases.items():
        if phrase in text:
            derived.extend(tokens)
    return derived
