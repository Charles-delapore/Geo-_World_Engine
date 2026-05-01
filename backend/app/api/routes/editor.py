from __future__ import annotations

import logging
from datetime import datetime, timezone
from collections import defaultdict
from threading import Lock

import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.config import settings
from app.orchestrator.artifact_orchestrator import ArtifactOrchestrator
from app.orchestrator.editor_orchestrator import EditorOrchestrator
from app.storage.models import TaskRecord, TaskStatus, get_db
from app.utils.helpers import safe_dict

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/maps", tags=["editor"])
_edit_locks: defaultdict[str, Lock] = defaultdict(Lock)


def task_edit_lock(task_id: str):
    lock = _edit_locks[task_id]
    lock.acquire()
    try:
        yield
    finally:
        lock.release()


class EditRequest(BaseModel):
    instructions: list[dict] = Field(default_factory=list)
    text_instruction: str | None = None
    sketch_geojson: dict | None = None


class EditResponse(BaseModel):
    taskId: str
    versionNum: int
    editSummary: str
    metricReport: dict | None = None
    dirtyBounds: list[dict] | None = None


class VersionInfo(BaseModel):
    versionNum: int
    editSummary: str | None
    editType: str | None
    parentVersion: int | None
    createdAt: object


class RevertRequest(BaseModel):
    version_num: int


class FeatureResponse(BaseModel):
    taskId: str
    features: list[dict]


class SketchPayload(BaseModel):
    geojson: dict
    instruction_type: str = "elevation_offset"
    brush_size: int = 5


def _resolve_orchestrators(task_id: str, plan: dict, params: dict | None = None):
    projection = safe_dict(plan).get("projection") or safe_dict(params).get("projection") or "flat"
    editor_orch = EditorOrchestrator(task_id=task_id, plan=plan, projection=projection)
    artifact_orch = ArtifactOrchestrator(task_id=task_id, projection=projection)
    return editor_orch, artifact_orch, projection


def _touch_task(task: TaskRecord, db: Session):
    task.updated_at = datetime.now(timezone.utc)
    db.commit()


@router.put("/{task_id}/edit", response_model=EditResponse)
def edit_map(
    task_id: str,
    payload: EditRequest,
    db: Session = Depends(get_db),
    _lock: None = Depends(task_edit_lock),
):
    task = db.get(TaskRecord, task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    if task.status not in (TaskStatus.READY.value, TaskStatus.READY_INTERACTIVE.value):
        raise HTTPException(status_code=409, detail=f"Task is not in editable state: {task.status}")

    plan = safe_dict(task.plan_json)
    editor_orch, artifact_orch, _ = _resolve_orchestrators(task_id, plan, task.params)
    elevation, world = editor_orch.load_elevation()

    new_elevation, next_version, dirty_bounds = editor_orch.apply_edit(
        elevation=elevation,
        world=world,
        instructions=payload.instructions or None,
        text_instruction=payload.text_instruction,
        sketch_geojson=payload.sketch_geojson,
        edit_summary=payload.text_instruction or f"Edit with {len(payload.instructions)} constraints",
        edit_type="text" if payload.text_instruction else ("sketch" if payload.sketch_geojson else "constraints"),
    )

    profile = safe_dict(plan.get("profile"))
    artifact_orch.refresh_tiles(dirty_bounds, profile)

    metric_report = None
    try:
        from app.core.geometry_metrics import compute_metric_report
        metric_report = compute_metric_report(new_elevation, safe_dict(plan.get("topology_intent")))
    except Exception as exc:
        logger.warning("Metric report failed after edit: %s", exc)

    _touch_task(task, db)

    return EditResponse(
        taskId=task_id,
        versionNum=next_version,
        editSummary=payload.text_instruction or f"Edit with {len(payload.instructions)} constraints",
        metricReport=metric_report,
        dirtyBounds=dirty_bounds if dirty_bounds else None,
    )


@router.post("/{task_id}/sketch", response_model=EditResponse)
def apply_sketch(
    task_id: str,
    payload: SketchPayload,
    db: Session = Depends(get_db),
    _lock: None = Depends(task_edit_lock),
):
    task = db.get(TaskRecord, task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")

    plan = safe_dict(task.plan_json)
    editor_orch, artifact_orch, _ = _resolve_orchestrators(task_id, plan, task.params)
    elevation, world = editor_orch.load_elevation()

    new_elevation, next_version, dirty_bounds = editor_orch.apply_sketch(
        elevation=elevation,
        world=world,
        geojson=payload.geojson,
        instruction_type=payload.instruction_type,
        brush_size=payload.brush_size,
    )

    profile = safe_dict(plan.get("profile"))
    artifact_orch.refresh_tiles(dirty_bounds, profile)

    metric_report = None
    try:
        from app.core.geometry_metrics import compute_metric_report
        metric_report = compute_metric_report(new_elevation, safe_dict(plan.get("topology_intent")))
    except Exception as exc:
        logger.warning("Metric report failed after sketch: %s", exc)

    _touch_task(task, db)

    return EditResponse(
        taskId=task_id,
        versionNum=next_version,
        editSummary=f"Brush {payload.instruction_type}",
        metricReport=metric_report,
        dirtyBounds=dirty_bounds if dirty_bounds else None,
    )


@router.get("/{task_id}/features", response_model=FeatureResponse)
def get_features(task_id: str, types: str = "ridge,river,lake", db: Session = Depends(get_db)):
    task = db.get(TaskRecord, task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")

    plan = safe_dict(task.plan_json)
    editor_orch, _, _ = _resolve_orchestrators(task_id, plan, task.params)
    elevation, _ = editor_orch.load_elevation()

    requested_types = set(types.split(","))
    features: list[dict] = []

    try:
        from app.core.hydrology_advanced import detect_ridges, detect_rivers

        if "ridge" in requested_types:
            ridge_mask = detect_ridges(elevation, accumulation_threshold=100)
            if np.any(ridge_mask):
                features.append({"type": "ridge", "count": int(np.sum(ridge_mask)), "bounds": _mask_bounds(ridge_mask)})

        if "river" in requested_types:
            river_mask = detect_rivers(elevation, accumulation_threshold=500)
            if np.any(river_mask):
                features.append({"type": "river", "count": int(np.sum(river_mask)), "bounds": _mask_bounds(river_mask)})
    except Exception as exc:
        logger.warning("Feature extraction failed: %s", exc)

    if "lake" in requested_types:
        water_mask = elevation <= 0
        from scipy.ndimage import label as ndimage_label
        labeled, num = ndimage_label(water_mask.astype(np.int8))
        for i in range(1, num + 1):
            comp = labeled == i
            if np.sum(comp) > 20:
                features.append({"type": "lake", "area": int(np.sum(comp)), "bounds": _mask_bounds(comp)})

    if "land" in requested_types:
        land_mask = elevation > 0
        from scipy.ndimage import label as ndimage_label
        labeled, num = ndimage_label(land_mask.astype(np.int8))
        for i in range(1, num + 1):
            comp = labeled == i
            if np.sum(comp) > 20:
                features.append({"type": "land_component", "area": int(np.sum(comp)), "bounds": _mask_bounds(comp)})

    return FeatureResponse(taskId=task_id, features=features)


@router.get("/{task_id}/history", response_model=list[VersionInfo])
def get_history(task_id: str, db: Session = Depends(get_db)):
    task = db.get(TaskRecord, task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")

    plan = safe_dict(task.plan_json)
    editor_orch, _, _ = _resolve_orchestrators(task_id, plan, task.params)
    history = editor_orch.get_version_history()

    return [
        VersionInfo(
            versionNum=entry["versionNum"],
            editSummary=entry.get("editSummary"),
            editType=entry.get("editType"),
            parentVersion=entry.get("parentVersion"),
            createdAt=entry.get("createdAt"),
        )
        for entry in history
    ]


@router.post("/{task_id}/revert", response_model=EditResponse)
def revert_to_version(
    task_id: str,
    payload: RevertRequest,
    db: Session = Depends(get_db),
    _lock: None = Depends(task_edit_lock),
):
    task = db.get(TaskRecord, task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")

    plan = safe_dict(task.plan_json)
    editor_orch, artifact_orch, _ = _resolve_orchestrators(task_id, plan, task.params)

    try:
        elevation, dirty_bounds = editor_orch.revert_to_version(payload.version_num)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Version {payload.version_num} not found")

    profile = safe_dict(plan.get("profile"))
    artifact_orch.refresh_tiles(dirty_bounds, profile)

    next_version = editor_orch.get_next_version()

    _touch_task(task, db)

    return EditResponse(
        taskId=task_id,
        versionNum=next_version,
        editSummary=f"Reverted to version {payload.version_num}",
        dirtyBounds=dirty_bounds if dirty_bounds else None,
    )


def _mask_bounds(mask: np.ndarray) -> dict:
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not np.any(rows):
        return {"min_y": 0, "min_x": 0, "max_y": 0, "max_x": 0}
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return {"min_y": int(rmin), "min_x": int(cmin), "max_y": int(rmax), "max_x": int(cmax)}
