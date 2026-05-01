from __future__ import annotations

import os

os.environ.setdefault("NUMBA_THREADING_LAYER", "tbb")

from app.celery_app import celery_app


if __name__ == "__main__":
    celery_app.worker_main(
        [
            "worker",
            "--loglevel=info",
            "-Q",
            "tile",
            "--concurrency=1",
            "--pool=prefork",
            "--prefetch-multiplier=1",
            "--max-tasks-per-child=10",
        ]
    )
