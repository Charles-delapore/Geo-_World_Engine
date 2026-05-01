from __future__ import annotations

import os

os.environ.setdefault("NUMBA_THREADING_LAYER", "tbb")

from app.celery_app import celery_app
from celery.signals import worker_process_init


@worker_process_init.connect
def _precompile_numba(**kwargs) -> None:
    try:
        from app.core.terrain import TerrainGenerator
        TerrainGenerator(width=256, height=128)
        print("✅ Numba functions precompiled successfully")
    except Exception as exc:
        print(f"⚠️ Numba precompile skipped: {exc}")


if __name__ == "__main__":
    celery_app.worker_main(
        [
            "worker",
            "--loglevel=info",
            "-Q",
            "celery",
            "--concurrency=2",
            "--pool=prefork",
            "--prefetch-multiplier=1",
            "--max-tasks-per-child=5",
        ]
    )
