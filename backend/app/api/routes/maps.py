from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.config import settings
from app.orchestrator.orchestrator import queue_generation, queue_orchestrator
from app.orchestrator.state_machine import PUBLIC_STATUS
from app.storage.artifact_repo import ArtifactRepository
from app.storage.models import TaskRecord, TaskStatus, get_db, session_scope

router = APIRouter(prefix="/maps", tags=["maps"])
repo = ArtifactRepository()


class CreateMapRequest(BaseModel):
    prompt: str = Field(min_length=1)
    width: int = Field(default_factory=lambda: settings.DEFAULT_WIDTH, ge=256, le=settings.MAX_WIDTH)
    height: int = Field(default_factory=lambda: settings.DEFAULT_HEIGHT, ge=256, le=settings.MAX_HEIGHT)
    seed: int | None = None
    auto_confirm: bool = settings.DEFAULT_AUTO_CONFIRM
    generate_tiles: bool = settings.DEFAULT_GENERATE_TILES
    llm_api_key: str | None = None
    llm_base_url: str | None = None
    llm_model: str | None = None


class MapResource(BaseModel):
    taskId: str
    status: str
    currentStage: str | None
    progress: int
    previewUrl: str | None
    manifestUrl: str | None
    errorMsg: str | None
    planSummary: str | None
    createdAt: datetime
    updatedAt: datetime


def _artifact_url(request: Request, path: str) -> str:
    return str(request.url_for("root")).rstrip("/") + path


def _serialize_task(task: TaskRecord, request: Request) -> MapResource:
    preview_url = _artifact_url(request, f"{settings.API_V1_STR}/maps/{task.task_id}/preview.png") if task.preview_ready else None
    manifest_url = (
        _artifact_url(request, f"{settings.API_V1_STR}/maps/{task.task_id}/tiles/manifest.json") if task.tiles_ready else None
    )
    return MapResource(
        taskId=task.task_id,
        status=PUBLIC_STATUS[TaskStatus(task.status)],
        currentStage=task.current_stage,
        progress=task.progress,
        previewUrl=preview_url,
        manifestUrl=manifest_url,
        errorMsg=task.error_msg,
        planSummary=task.plan_summary,
        createdAt=task.created_at,
        updatedAt=task.updated_at,
    )


@router.post("", response_model=MapResource, status_code=201)
def create_map(payload: CreateMapRequest, request: Request, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    task_id = str(uuid4())
    record = TaskRecord(
        task_id=task_id,
        prompt=payload.prompt,
        status=TaskStatus.QUEUED.value,
        current_stage="任务已排队",
        progress=0,
        params=payload.model_dump(mode="json"),
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    if settings.ENABLE_BACKGROUND_PIPELINE:
        if settings.RUN_MODE.lower() == "celery":
            queue_orchestrator(task_id)
        else:
            background_tasks.add_task(queue_orchestrator, task_id)
    return _serialize_task(record, request)


@router.get("/{task_id}", response_model=MapResource)
def get_map(task_id: str, request: Request, db: Session = Depends(get_db)):
    task = db.get(TaskRecord, task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return _serialize_task(task, request)


@router.post("/{task_id}/confirm", status_code=204)
def confirm_map(task_id: str, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    task = db.get(TaskRecord, task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    if task.status != TaskStatus.AWAITING_CONFIRM.value:
        raise HTTPException(status_code=409, detail=f"Task is not awaiting confirmation: {task.status}")
    task.confirmed_at = datetime.now(timezone.utc)
    task.progress = max(task.progress, 20)
    db.add(task)
    db.commit()
    if settings.RUN_MODE.lower() == "celery":
        queue_generation(task_id)
    else:
        background_tasks.add_task(queue_generation, task_id)
    return Response(status_code=204)
