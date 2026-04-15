from __future__ import annotations

from app.orchestrator.state_machine import STAGE_LABELS
from app.storage.models import TaskRecord, TaskStatus, TaskTransition, session_scope


def transition_task(
    task_id: str,
    status: TaskStatus,
    *,
    progress: int | None = None,
    reason: str | None = None,
    error_msg: str | None = None,
    preview_ready: bool | None = None,
    tiles_ready: bool | None = None,
) -> None:
    with session_scope() as db:
        task = db.get(TaskRecord, task_id)
        if task is None:
            raise ValueError(f"Task {task_id} not found")
        previous = task.status
        task.status = status.value
        task.current_stage = STAGE_LABELS[status]
        if progress is not None:
            task.progress = progress
        if error_msg is not None:
            task.error_msg = error_msg
        if preview_ready is not None:
            task.preview_ready = preview_ready
        if tiles_ready is not None:
            task.tiles_ready = tiles_ready
        db.add(TaskTransition(task_id=task_id, from_state=previous, to_state=status.value, reason=reason))
