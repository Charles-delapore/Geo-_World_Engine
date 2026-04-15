from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.orchestrator.state_machine import STAGE_LABELS
from app.storage.models import TaskRecord, TaskStatus, get_db

router = APIRouter(prefix="/internal/tasks", tags=["internal"])


class TransitionRequest(BaseModel):
    status: TaskStatus
    current_stage: str | None = None
    progress: int | None = None
    error_msg: str | None = None
    preview_ready: bool | None = None
    tiles_ready: bool | None = None


@router.post("/{task_id}/transition")
def transition_task_internal(task_id: str, payload: TransitionRequest, db: Session = Depends(get_db)):
    task = db.get(TaskRecord, task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")

    task.status = payload.status.value
    task.current_stage = payload.current_stage or STAGE_LABELS[payload.status]
    if payload.progress is not None:
        task.progress = payload.progress
    if payload.error_msg is not None:
        task.error_msg = payload.error_msg
    if payload.preview_ready is not None:
        task.preview_ready = payload.preview_ready
    if payload.tiles_ready is not None:
        task.tiles_ready = payload.tiles_ready

    db.add(task)
    db.commit()
    db.refresh(task)
    return {"taskId": task.task_id, "status": task.status}
