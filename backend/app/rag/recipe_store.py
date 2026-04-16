from __future__ import annotations

import json
from datetime import datetime

from sqlalchemy import DateTime, Integer, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from app.rag.schemas import TerrainRecipe


class Base(DeclarativeBase):
    pass


class RecipeModel(Base):
    __tablename__ = "recipes"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    name: Mapped[str] = mapped_column(Text)
    description: Mapped[str] = mapped_column(Text)
    world_plan: Mapped[str] = mapped_column(Text)
    params: Mapped[str] = mapped_column(Text, default="{}")
    tags: Mapped[str] = mapped_column(Text, default="")
    source: Mapped[str] = mapped_column(Text, default="builtin")
    success_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class RecipeDB:
    def __init__(self, db_url: str):
        self.engine = create_engine(db_url, future=True)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine, autoflush=False, autocommit=False, future=True)

    def load(self, recipe_id: str) -> TerrainRecipe | None:
        with self.Session() as session:
            model = session.get(RecipeModel, recipe_id)
            return self._model_to_recipe(model) if model else None

    def list_all(self) -> list[TerrainRecipe]:
        with self.Session() as session:
            models = session.query(RecipeModel).all()
            return [self._model_to_recipe(model) for model in models]

    def count(self) -> int:
        with self.Session() as session:
            return int(session.query(RecipeModel).count())

    def save_many(self, recipes: list[TerrainRecipe]) -> None:
        with self.Session() as session:
            for recipe in recipes:
                session.merge(
                    RecipeModel(
                        id=recipe.id,
                        name=recipe.name,
                        description=recipe.description,
                        world_plan=json.dumps(recipe.world_plan, ensure_ascii=False),
                        params=json.dumps(recipe.params or {}, ensure_ascii=False),
                        tags=",".join(recipe.tags),
                        source=recipe.source,
                        success_count=recipe.success_count,
                        created_at=recipe.created_at,
                    )
                )
            session.commit()

    @staticmethod
    def _model_to_recipe(model: RecipeModel) -> TerrainRecipe:
        return TerrainRecipe(
            id=model.id,
            name=model.name,
            description=model.description,
            world_plan=json.loads(model.world_plan),
            params=json.loads(model.params) if model.params else None,
            tags=model.tags.split(",") if model.tags else [],
            source=model.source,
            success_count=model.success_count,
            created_at=model.created_at,
        )
