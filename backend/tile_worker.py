from __future__ import annotations

from app.celery_app import celery_app


if __name__ == "__main__":
    celery_app.worker_main(
        [
            "worker",
            "--loglevel=info",
            "-Q",
            "tile",
            "--concurrency=1",
        ]
    )
