from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True, extra="ignore")

    PROJECT_NAME: str = "Geo-WorldEngine Beta"
    API_V1_STR: str = "/api"
    RUN_MODE: str = "local"
    DATABASE_URL: str = f"sqlite:///{(DATA_DIR / 'geo_world_beta.db').as_posix()}"
    RUN_DB_MIGRATIONS: bool = True
    ARTIFACT_ROOT: Path = DATA_DIR / "artifacts"
    ARTIFACT_BACKEND: str = "local"
    DEFAULT_WIDTH: int = 1024
    DEFAULT_HEIGHT: int = 512
    MAX_WIDTH: int = 2048
    MAX_HEIGHT: int = 1024
    DEFAULT_TILE_SIZE: int = 256
    MAX_TILE_ZOOM: int = 0
    DEFAULT_AUTO_CONFIRM: bool = True
    DEFAULT_GENERATE_TILES: bool = True
    ENABLE_BACKGROUND_PIPELINE: bool = True
    REDIS_URL: str = "redis://127.0.0.1:6379/0"
    CELERY_BROKER_URL: str = "redis://127.0.0.1:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://127.0.0.1:6379/1"
    CELERY_DEFAULT_QUEUE: str = "celery"
    CELERY_TILE_QUEUE: str = "tile"
    MINIO_ENDPOINT: str = "127.0.0.1:9000"
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "minioadmin"
    MINIO_BUCKET: str = "geo-world-beta"
    MINIO_USE_SSL: bool = False
    OPENAI_API_KEY: str | None = None
    OPENAI_BASE_URL: str | None = None
    OPENAI_MODEL: str = "gpt-4o-mini"
    CORS_ORIGINS: list[str] = Field(default_factory=lambda: ["*"])


settings = Settings()
DATA_DIR.mkdir(parents=True, exist_ok=True)
settings.ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
