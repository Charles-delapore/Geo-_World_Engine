from __future__ import annotations

from celery import Celery

from app.config import settings


celery_app = Celery(
    "geo_world_beta",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=["app.orchestrator.tasks"],
)

celery_app.conf.update(
    task_default_queue=settings.CELERY_DEFAULT_QUEUE,
    task_routes={
        "geo_world_beta.orchestrate": {"queue": settings.CELERY_DEFAULT_QUEUE},
        "geo_world_beta.execute_generation": {"queue": settings.CELERY_DEFAULT_QUEUE},
        "geo_world_beta.generate_tiles": {"queue": settings.CELERY_TILE_QUEUE},
    },
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    broker_connection_retry_on_startup=True,
)
