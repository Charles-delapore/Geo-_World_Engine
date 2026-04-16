from __future__ import annotations

from app.config import settings
from app.rag.default_recipes import DEFAULT_RECIPES
from app.rag.recipe_store import RecipeDB
from app.rag.schemas import TerrainRecipe


def init_builtin_knowledge_base(force: bool = False) -> int:
    db = RecipeDB(settings.RECIPE_DB_URL)
    recipes = [TerrainRecipe(**recipe) for recipe in DEFAULT_RECIPES]
    if force:
        db.save_many(recipes)
        return db.count()
    db.save_many(recipes)
    return db.count()


if __name__ == "__main__":
    total = init_builtin_knowledge_base(force=False)
    print(f"recipes_ready={total}")
