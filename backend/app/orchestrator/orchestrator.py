from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from io import BytesIO
from uuid import UUID

import numpy as np
from PIL import Image

from app.config import settings
from app.orchestrator.transitions import transition_task
from app.pipelines.plan_compiler import PlanCompiler
from app.storage.artifact_repo import ArtifactRepository
from app.storage.models import TaskRecord, TaskStatus, session_scope
from app.utils.helpers import safe_dict
from app.workers.planner_worker import build_world_plan
from app.workers.tile_worker import generate_tiles


repo = ArtifactRepository()
logger = logging.getLogger(__name__)
_local_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="geo-world-local")


def render_world(
    plan: dict,
    width: int,
    height: int,
    seed: int,
    emit_debug_artifacts: bool = False,
    task_id: str | None = None,
    projection: str = "planet",
    enable_nl_verify: bool = True,
) -> tuple[dict[str, np.ndarray], Image.Image]:
    from app.core.terrain_shaping import normalize_generation_plan
    from app.core.geometry_metrics import component_labels

    plan = normalize_generation_plan(plan)
    prompt = plan.get("prompt", "")
    profile = plan.get("profile") or {}
    topology_intent = plan.get("topology_intent") or {}

    if projection == "planet":
        from app.pipelines.planet_pipeline import PlanetTerrainPipeline
        pipeline = PlanetTerrainPipeline()
    else:
        from app.pipelines.flat_pipeline import FlatTerrainPipeline
        pipeline = FlatTerrainPipeline()

    bundle = pipeline.generate(plan, seed=seed, width=width, height=height)

    if enable_nl_verify and prompt:
        nl_report = _run_nl_verification(
            prompt, plan, bundle, pipeline, seed, width, height
        )
        if nl_report:
            if nl_report.get("regenerated"):
                bundle = pipeline.generate(plan, seed=seed, width=width, height=height)
            profile = plan.get("profile") or {}
            topology_intent = plan.get("topology_intent") or {}
            plan["_nl_verify_report"] = nl_report

    from app.orchestrator.validation_orchestrator import ValidationOrchestrator
    validator = ValidationOrchestrator()
    validation = validator.validate(bundle.elevation, plan, topology_intent)

    metric_report = validation.metric_report
    if validation.consistency:
        metric_report["consistency"] = validation.consistency
    if validation.srg_consistency:
        metric_report["srg_consistency"] = validation.srg_consistency

    nl_report = plan.pop("_nl_verify_report", None)
    if nl_report:
        metric_report["nl_verify"] = nl_report

    arrays = bundle.to_arrays()
    arrays["metric_report"] = json.dumps(metric_report, ensure_ascii=False).encode("utf-8")

    if emit_debug_artifacts and task_id:
        _emit_debug_artifacts(arrays, task_id, component_labels)

    preview = pipeline.build_preview(bundle, profile)
    return arrays, preview


def _run_nl_verification(
    prompt: str,
    plan: dict,
    bundle,
    pipeline,
    seed: int,
    width: int,
    height: int,
) -> dict | None:
    from app.core.nl_verifier import critique_terrain, iterative_regenerate

    try:
        critique = critique_terrain(prompt, plan, bundle.elevation)
        report = {
            "pre_verify": critique.to_dict(),
            "regenerated": False,
            "iterations": 0,
            "final_grade": critique.grade,
            "final_score": critique.overall_score,
        }

        if critique.passed:
            return report

        logger.info(
            "NL verify failed (grade=%s score=%.3f), attempting regeneration",
            critique.grade, critique.overall_score,
        )

        def _gen_fn(p: dict):
            return pipeline.generate(p, seed=seed, width=width, height=height).elevation

        best_elevation, history, updated_plan = iterative_regenerate(
            prompt, plan, _gen_fn, max_iterations=3,
        )
        if best_elevation is not None:
            bundle.elevation = best_elevation

        for key in ("profile", "constraints", "topology_intent", "mountains"):
            if key in updated_plan:
                plan[key] = updated_plan[key]

        final_critique = history[-1] if history else critique
        report["regenerated"] = True
        report["iterations"] = len(history)
        report["final_grade"] = final_critique.grade
        report["final_score"] = final_critique.overall_score
        report["history"] = [h.to_dict() for h in history]
        report["issues"] = final_critique.issues
        report["adjustments"] = final_critique.param_adjustments

        return report
    except Exception as exc:
        logger.warning("NL verification failed with error: %s", exc)
        return {"error": str(exc), "regenerated": False}


def _emit_debug_artifacts(
    arrays: dict[str, np.ndarray],
    task_id: str,
    component_labels_fn,
) -> None:
    try:
        from app.storage.artifact_repo import ArtifactRepository as _Repo
        _repo = _Repo()
        elevation = arrays.get("elevation")
        if elevation is None:
            return

        land_mask = (elevation > 0.0).astype(np.uint8) * 255
        water_mask = (elevation <= 0.0).astype(np.uint8) * 255
        labels = component_labels_fn(elevation > 0.0, min_cells=20)
        label_vis = (labels % 7 * 36).astype(np.uint8)

        _repo.save_debug_image(task_id, "target_land_mask", Image.fromarray(land_mask, mode="L"))
        _repo.save_debug_image(task_id, "target_water_mask", Image.fromarray(water_mask, mode="L"))
        _repo.save_debug_image(task_id, "component_labels", Image.fromarray(label_vis, mode="L"))
        _repo.save_debug_image(task_id, "final_land_mask", Image.fromarray(land_mask, mode="L"))

        _repo.save_debug_array(task_id, "land_mask", (elevation > 0.0).astype(np.float32))
        _repo.save_debug_array(task_id, "water_mask", (elevation <= 0.0).astype(np.float32))
        _repo.save_debug_array(task_id, "component_labels_raw", labels.astype(np.int32))

        metric_bytes = arrays.get("metric_report")
        if metric_bytes is not None:
            metric_data = json.loads(metric_bytes)
            _repo.save_debug_json(task_id, "metric_report", metric_data)

        logger.info("Debug artifacts persisted for task=%s shape=%s", task_id, elevation.shape)
    except Exception as exc:
        logger.warning("Failed to emit debug artifacts: %s", exc)


def _run_local_async(func, *args):
    future = _local_executor.submit(func, *args)

    def _log_failure(done):
        try:
            done.result()
        except Exception as exc:
            logger.exception("Local background job failed: %s", getattr(func, "__name__", func))
            if args and isinstance(args[0], str):
                task_id = args[0]
                func_name = getattr(func, "__name__", "")
                if func_name in {"orchestrate_task", "execute_generation"}:
                    transition_task(
                        task_id,
                        TaskStatus.FAILED,
                        progress=100,
                        reason="pipeline_error",
                        error_msg=str(exc),
                    )
                elif func_name == "generate_tiles_for_task":
                    transition_task(
                        task_id,
                        TaskStatus.READY,
                        progress=100,
                        reason="tile_error",
                        preview_ready=True,
                        tiles_ready=False,
                        error_msg=str(exc),
                    )

    future.add_done_callback(_log_failure)
    return future


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
            params = safe_dict(task.params)
            asset_ids = params.get("asset_ids") or []

            if asset_ids:
                compiler = PlanCompiler(
                    api_key=params.get("llm_api_key") or settings.OPENAI_API_KEY,
                    base_url=params.get("llm_base_url") or settings.OPENAI_BASE_URL,
                    model=params.get("llm_model") or settings.OPENAI_MODEL,
                )
                plan_dict, constraint_graph = compiler.compile(
                    prompt=task.prompt,
                    params=params,
                    asset_ids=asset_ids,
                )
                plan = build_world_plan(task.prompt, params)
                plan_dict.update({k: v for k, v in plan.model_dump(mode="json").items() if k not in plan_dict})
                task.plan_json = plan_dict
                task.plan_summary = plan.summary
            else:
                plan = build_world_plan(task.prompt, params)
                task.plan_json = plan.model_dump(mode="json")
                task.plan_summary = plan.summary

            if not bool(params.get("auto_confirm", settings.DEFAULT_AUTO_CONFIRM)):
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
        width = int(safe_dict(task.params).get("width", settings.DEFAULT_WIDTH))
        height = int(safe_dict(task.params).get("height", settings.DEFAULT_HEIGHT))
        seed = _resolve_seed(task.task_id, safe_dict(task.params))
        plan_json = safe_dict(task.plan_json)
        projection = safe_dict(task.params).get("projection", "planet")

    arrays, preview = render_world(
        plan_json, width=width, height=height, seed=seed,
        emit_debug_artifacts=True, task_id=task_id,
        projection=projection,
    )
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
        generate_tiles_enabled = bool(safe_dict(task.params).get("generate_tiles", settings.DEFAULT_GENERATE_TILES))

    if generate_tiles_enabled:
        queue_tiles(task_id)


def queue_orchestrator(task_id: str):
    if settings.RUN_MODE.lower() == "celery":
        from app.orchestrator.tasks import orchestrate_task_job

        return orchestrate_task_job.delay(task_id)
    return _run_local_async(orchestrate_task, task_id)


def queue_generation(task_id: str):
    if settings.RUN_MODE.lower() == "celery":
        from app.orchestrator.tasks import execute_generation_job

        return execute_generation_job.delay(task_id)
    return _run_local_async(execute_generation, task_id)


def generate_tiles_for_task(task_id: str) -> None:
    preview = Image.open(BytesIO(repo.read_preview_bytes(task_id))).convert("RGB")
    with session_scope() as db:
        task = db.get(TaskRecord, task_id)
        projection = safe_dict(task.params).get("projection", "planet") if task else "planet"
        profile = safe_dict(safe_dict(task.plan_json).get("profile")) if task else {}
    generate_tiles(task_id, repo, preview, max_zoom=settings.MAX_TILE_ZOOM, projection=projection, profile=profile)
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
    return _run_local_async(generate_tiles_for_task, task_id)
