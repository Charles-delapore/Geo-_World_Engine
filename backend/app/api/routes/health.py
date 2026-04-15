from __future__ import annotations

from fastapi import APIRouter
from sqlalchemy import text

from app.config import settings
from app.storage.models import SessionLocal
from app.storage.s3_client import s3_client

try:
    from redis import Redis
except Exception:  # pragma: no cover - redis import should exist in production
    Redis = None

router = APIRouter()


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/ready")
def ready():
    checks: dict[str, str] = {"database": "ok"}

    with SessionLocal() as db:
        db.execute(text("SELECT 1"))

    if settings.RUN_MODE == "celery" and Redis is not None:
        redis_client = Redis.from_url(settings.REDIS_URL)
        redis_client.ping()
        checks["redis"] = "ok"

    if settings.ARTIFACT_BACKEND == "s3":
        s3_client.ensure_bucket()
        checks["artifacts"] = "ok"
    else:
        checks["artifacts"] = "ok"

    return {"status": "ready", "checks": checks}
