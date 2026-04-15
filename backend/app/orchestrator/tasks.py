from __future__ import annotations

from app.celery_app import celery_app
from app.orchestrator.orchestrator import execute_generation, generate_tiles_for_task, orchestrate_task


@celery_app.task(name="geo_world_beta.orchestrate")
def orchestrate_task_job(task_id: str) -> None:
    orchestrate_task(task_id)


@celery_app.task(name="geo_world_beta.execute_generation")
def execute_generation_job(task_id: str) -> None:
    execute_generation(task_id)


@celery_app.task(name="geo_world_beta.generate_tiles")
def generate_tiles_job(task_id: str) -> None:
    generate_tiles_for_task(task_id)
