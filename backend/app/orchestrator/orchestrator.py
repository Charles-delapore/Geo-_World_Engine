from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
from uuid import UUID

from PIL import Image

from app.config import settings
from app.orchestrator.transitions import transition_task
from app.storage.artifact_repo import ArtifactRepository
from app.storage.models import TaskRecord, TaskStatus, session_scope
from app.workers.planner_worker import build_world_plan
from app.workers.render_worker import render_world
from app.workers.tile_worker import generate_tiles


repo = ArtifactRepository()


def _resolve_seed(task_id: str, params: dict) -> int:
    if params.get("seed") is not None:
        return int(params["seed"])
    try:
        return int(UUID(task_id).int % 100000)
    except ValueError:
        return sum(ord(char) for char in task_id) % 100000


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
        seed = _resolve_seed(task.task_id, task.params or {})
        plan_json = task.plan_json or {}

    arrays, preview = render_world(plan_json, width=width, height=height, seed=seed)
    metric_report_raw = arrays.pop("metric_report", None)
    repo.save_world(task_id, **arrays)

    if metric_report_raw is not None:
        metric_str = metric_report_raw.decode("utf-8") if isinstance(metric_report_raw, bytes) else str(metric_report_raw)
        with session_scope() as db:
            task = db.get(TaskRecord, task_id)
            if task is not None:
                task.metric_report = metric_str

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
