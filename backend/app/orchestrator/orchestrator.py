from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO

from PIL import Image

from app.config import settings
from app.orchestrator.transitions import transition_task
from app.storage.artifact_repo import ArtifactRepository
from app.storage.models import TaskRecord, TaskStatus, session_scope
from app.workers.planner_worker import build_world_plan
from app.workers.render_worker import render_world
from app.workers.tile_worker import generate_tiles


repo = ArtifactRepository()


def orchestrate_task(task_id: str) -> None:
    try:
        transition_task(task_id, TaskStatus.PARSING, progress=5, reason="planner")
        with session_scope() as db:
            task = db.get(TaskRecord, task_id)
            if task is None:
                raise ValueError(f"Task {task_id} not found")
            plan = build_world_plan(task.prompt, task.params or {})
            task.plan_json = plan.model_dump(mode="json")
            task.plan_summary = plan.summary
            if not bool(task.params.get("auto_confirm", settings.DEFAULT_AUTO_CONFIRM)):
                task.status = TaskStatus.AWAITING_CONFIRM.value
                task.current_stage = "等待用户确认世界规划"
                task.progress = 15
                return
            task.confirmed_at = datetime.now(timezone.utc)

        execute_generation(task_id)
    except Exception as exc:
        transition_task(task_id, TaskStatus.FAILED, progress=100, reason="pipeline_error", error_msg=str(exc))


def execute_generation(task_id: str) -> None:
    transition_task(task_id, TaskStatus.GENERATING_TERRAIN, progress=35, reason="render")
    with session_scope() as db:
        task = db.get(TaskRecord, task_id)
        if task is None:
            raise ValueError(f"Task {task_id} not found")
        width = int(task.params.get("width", settings.DEFAULT_WIDTH))
        height = int(task.params.get("height", settings.DEFAULT_HEIGHT))
        seed = int(task.params.get("seed") or abs(hash(task.task_id)) % 100000)
        plan_json = task.plan_json or {}

    transition_task(task_id, TaskStatus.SIMULATING, progress=60, reason="render")
    arrays, preview = render_world(plan_json, width=width, height=height, seed=seed)
    repo.save_world(task_id, **arrays)

    transition_task(task_id, TaskStatus.RENDERING_IMAGE, progress=80, reason="preview")
    repo.save_preview(task_id, preview)
    transition_task(task_id, TaskStatus.READY, progress=100, reason="preview", preview_ready=True)

    with session_scope() as db:
        task = db.get(TaskRecord, task_id)
        if task is None:
            raise ValueError(f"Task {task_id} not found")
        generate_tiles_enabled = bool(task.params.get("generate_tiles", settings.DEFAULT_GENERATE_TILES))

    if generate_tiles_enabled:
        queue_tiles(task_id)


def queue_orchestrator(task_id: str):
    if settings.RUN_MODE.lower() == "celery":
        from app.orchestrator.tasks import orchestrate_task_job

        return orchestrate_task_job.delay(task_id)
    orchestrate_task(task_id)
    return None


def queue_generation(task_id: str):
    if settings.RUN_MODE.lower() == "celery":
        from app.orchestrator.tasks import execute_generation_job

        return execute_generation_job.delay(task_id)
    execute_generation(task_id)
    return None


def generate_tiles_for_task(task_id: str) -> None:
    preview = Image.open(BytesIO(repo.read_preview_bytes(task_id))).convert("RGB")
    transition_task(task_id, TaskStatus.RENDERING_TILES, progress=100, reason="tiles", preview_ready=True)
    generate_tiles(task_id, repo, preview, max_zoom=settings.MAX_TILE_ZOOM)
    transition_task(
        task_id,
        TaskStatus.READY_INTERACTIVE,
        progress=100,
        reason="tiles",
        preview_ready=True,
        tiles_ready=True,
    )


def queue_tiles(task_id: str):
    if settings.RUN_MODE.lower() == "celery":
        from app.orchestrator.tasks import generate_tiles_job

        return generate_tiles_job.delay(task_id)
    generate_tiles_for_task(task_id)
    return None
