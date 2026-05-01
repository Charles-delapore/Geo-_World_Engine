from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Literal
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.config import settings
from app.core.world_plan import GenerationModuleSpec
from app.orchestrator.orchestrator import _resolve_seed, queue_generation, queue_orchestrator
from app.orchestrator.state_machine import PUBLIC_STATUS
from app.storage.artifact_repo import ArtifactRepository
from app.storage.models import TaskRecord, TaskStatus, get_db, session_scope
from app.utils.helpers import safe_dict

router = APIRouter(prefix="/maps", tags=["maps"])
repo = ArtifactRepository()
logger = logging.getLogger(__name__)


class CreateMapRequest(BaseModel):
    prompt: str = Field(min_length=1)
    width: int = Field(default_factory=lambda: settings.DEFAULT_WIDTH, ge=256, le=settings.MAX_WIDTH)
    height: int = Field(default_factory=lambda: settings.DEFAULT_HEIGHT, ge=256, le=settings.MAX_HEIGHT)
    seed: int | None = None
    generation_backend: Literal["gaussian_voronoi", "modular"] | None = None
    module_sequence: list[GenerationModuleSpec] = Field(default_factory=list)
    auto_confirm: bool = settings.DEFAULT_AUTO_CONFIRM
    generate_tiles: bool = settings.DEFAULT_GENERATE_TILES
    projection: Literal["flat", "planet"] = "planet"
    llm_api_key: str | None = None
    llm_base_url: str | None = None
    llm_model: str | None = None
    asset_ids: list[str] = Field(default_factory=list)


class MapResource(BaseModel):
    taskId: str
    status: str
    currentStage: str | None
    progress: int
    previewUrl: str | None
    manifestUrl: str | None
    errorMsg: str | None
    planSummary: str | None
    diagnostics: dict
    createdAt: datetime
    updatedAt: datetime


def _artifact_url(request: Request, path: str, version: str | None = None) -> str:
    forwarded_host = request.headers.get("x-forwarded-host") or request.headers.get("host")
    forwarded_proto = request.headers.get("x-forwarded-proto", "http")
    if forwarded_host:
        base = f"{forwarded_proto}://{forwarded_host}"
    else:
        base = str(request.base_url).rstrip("/")
    url = f"{base}{settings.API_V1_STR}{path}"
    return f"{url}?v={version}" if version else url


def _serialize_task(task: TaskRecord, request: Request) -> MapResource:
    artifact_version = str(int(task.updated_at.timestamp() * 1000)) if task.updated_at else None
    preview_url = _artifact_url(request, f"/maps/{task.task_id}/preview.png", artifact_version) if task.preview_ready else None
    manifest_url = (
        _artifact_url(request, f"/maps/{task.task_id}/tiles/manifest.json", artifact_version) if task.tiles_ready else None
    )
    plan = safe_dict(task.plan_json)
    profile = safe_dict(plan.get("profile"))
    constraints = safe_dict(plan.get("constraints"))
    rag_meta = safe_dict(plan.get("rag_meta"))
    params = safe_dict(task.params)
    topology_intent = safe_dict(plan.get("topology_intent"))
    diagnostics = {
        "seed": _resolve_seed(task.task_id, params),
        "width": int(params.get("width", settings.DEFAULT_WIDTH)),
        "height": int(params.get("height", settings.DEFAULT_HEIGHT)),
        "generationBackend": plan.get("generation_backend", "gaussian_voronoi"),
        "projection": params.get("projection", "planet"),
        "generateTiles": bool(params.get("generate_tiles", settings.DEFAULT_GENERATE_TILES)),
        "topologyIntent": topology_intent.get("kind"),
        "topologyModifiers": topology_intent.get("modifiers") or {},
        "layoutTemplate": profile.get("layout_template", "default"),
        "seaStyle": profile.get("sea_style", "open"),
        "landRatio": float(profile.get("land_ratio", 0.44)),
        "ruggedness": float(profile.get("ruggedness", 0.55)),
        "coastComplexity": float(profile.get("coast_complexity", 0.5)),
        "islandFactor": float(profile.get("island_factor", 0.25)),
        "moisture": float(profile.get("moisture", 1.0)),
        "temperatureBias": float(profile.get("temperature_bias", 0.0)),
        "windDirection": profile.get("wind_direction", "westerly"),
        "continentCount": len(constraints.get("continents") or []),
        "mountainCount": len(constraints.get("mountains") or []),
        "seaZoneCount": len(constraints.get("sea_zones") or []),
        "waterBodyCount": len(plan.get("water_bodies") or []),
        "regionalRelationCount": len(plan.get("regional_relations") or []),
        "ragEnabled": bool(rag_meta.get("enabled")),
        "ragExamples": int(rag_meta.get("retrieved_count", 0)),
        "ragTopSimilarity": rag_meta.get("top_similarity"),
        "ragFallbackReason": rag_meta.get("fallback_reason"),
    }
    metric_report_raw = task.metric_report
    if metric_report_raw:
        try:
            import json as _json
            diagnostics["metricReport"] = _json.loads(metric_report_raw) if isinstance(metric_report_raw, str) else metric_report_raw
        except Exception:
            diagnostics["metricReport"] = None
    return MapResource(
        taskId=task.task_id,
        status=PUBLIC_STATUS[TaskStatus(task.status)],
        currentStage=task.current_stage,
        progress=task.progress,
        previewUrl=preview_url,
        manifestUrl=manifest_url,
        errorMsg=task.error_msg,
        planSummary=task.plan_summary,
        diagnostics=diagnostics,
        createdAt=task.created_at,
        updatedAt=task.updated_at,
    )


def _sync_artifact_state(task: TaskRecord, db: Session) -> None:
    if task.status == TaskStatus.FAILED.value:
        return
    preview_ready = repo.has_preview(task.task_id)
    tiles_ready = repo.has_manifest(task.task_id)
    if task.status == TaskStatus.READY.value and not task.tiles_ready:
        tiles_ready = False
    changed = False
    if task.preview_ready != preview_ready:
        task.preview_ready = preview_ready
        changed = True
    if task.tiles_ready != tiles_ready:
        task.tiles_ready = tiles_ready
        changed = True
    if tiles_ready and task.status != TaskStatus.READY_INTERACTIVE.value:
        task.status = TaskStatus.READY_INTERACTIVE.value
        task.current_stage = "交互地图已就绪"
        task.progress = 100
        changed = True
    elif preview_ready and task.status in {
        TaskStatus.QUEUED.value,
        TaskStatus.PARSING.value,
        TaskStatus.GENERATING_TERRAIN.value,
        TaskStatus.RENDERING_IMAGE.value,
    }:
        task.status = TaskStatus.READY.value
        task.current_stage = "预览图已就绪"
        task.progress = max(task.progress, 100)
        changed = True
    if changed:
        db.add(task)
        db.commit()
        db.refresh(task)


@router.post("", response_model=MapResource, status_code=201)
def create_map(payload: CreateMapRequest, request: Request, db: Session = Depends(get_db)):
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
    logger.info("Task created: task_id=%s projection=%s", task_id, payload.projection)

    if settings.ENABLE_BACKGROUND_PIPELINE:
        queue_orchestrator(task_id)
    return _serialize_task(record, request)


@router.get("/{task_id}", response_model=MapResource)
def get_map(task_id: str, request: Request, db: Session = Depends(get_db)):
    task = db.get(TaskRecord, task_id)
    if task is None:
        logger.warning("Task not found: task_id=%s host=%s xfwd=%s", task_id,
                       request.headers.get("host", "-"),
                       request.headers.get("x-forwarded-host", "-"))
        raise HTTPException(status_code=404, detail="Task not found")
    _sync_artifact_state(task, db)
    return _serialize_task(task, request)


@router.post("/{task_id}/confirm", status_code=204)
def confirm_map(task_id: str, db: Session = Depends(get_db)):
    task = db.get(TaskRecord, task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    if task.status != TaskStatus.AWAITING_CONFIRM.value:
        raise HTTPException(status_code=409, detail=f"Task is not awaiting confirmation: {task.status}")
    task.confirmed_at = datetime.now(timezone.utc)
    task.progress = max(task.progress, 20)
    db.add(task)
    db.commit()
    queue_generation(task_id)
    return Response(status_code=204)
