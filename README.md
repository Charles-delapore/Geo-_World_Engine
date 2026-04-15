
=======

# Geo_World_Engine
Geo-WorldEngine(Beta) 是一个调试中的AI驱动程序化幻想地图生成系统。融合LLM语义理解、计算几何、物理模拟和分布式任务调度,支持自然语言约束的交互式世界构建。
这个项目仍然在草创阶段，而且大部分工作由agent完成（我指的是codex），后续将会持续优化。【welcome stars and contributions!】
=======

# Geo-WorldEngine Beta

Geo-WorldEngine Beta is the beta-generation workspace for the world map pipeline. It currently includes:

- FastAPI backend with a unified `/api/maps/{taskId}` task resource model
- Vue 3 + Vite frontend for task submission, polling, preview display, and interactive map switching
- Local development mode for rapid iteration
- Containerized deployment baseline with PostgreSQL, Redis, MinIO, Celery, and Nginx

## Repository Safety

This repository is prepared for public or shared version control with a conservative ignore policy:

- local secrets are excluded through `.gitignore`
- generated preview images, tiles, databases, and Docker volume data are excluded
- virtual environments, caches, editor folders, logs, and local temp files are excluded

Before publishing, verify that no real API keys, user prompts, exported datasets, or local debug dumps were manually added outside the ignored paths.

## Project Layout

- `backend/`: FastAPI app, orchestration, workers, storage adapters, Alembic migrations
- `frontend/Geo-world/`: Vite + Vue frontend
- `frontend/nginx.conf`: production reverse proxy configuration
- `deployment/`: Compose files and helper scripts
- `start-beta.ps1`: local development launcher
- `.env.example`: safe template for required environment variables

## Local Development

Prerequisites:

- Python 3.13
- Node.js with npm

Start both local services:

```powershell
Set-Location .\GEO
.\start-beta.ps1
```

Default local addresses:

- frontend: `http://127.0.0.1:5173`
- backend: `http://127.0.0.1:8000`

The Vite dev server proxies `/api` to the backend so preview images and tile assets resolve correctly in local mode.

## Containerized Deployment

The repository also includes a containerized beta baseline.

Core services:

- `backend-api`
- `celery-worker`
- `celery-tile`
- `postgres`
- `redis`
- `minio`
- `frontend`

Setup:

1. Copy `.env.example` to `.env`
2. Review the values and replace placeholders
3. Start the stack with the compose file in `deployment/`

Current deployment behavior:

- backend API container performs database bootstrap on startup
- readiness checks validate database connectivity and, in distributed mode, Redis and artifact storage
- Nginx resolves backend service names through Docker DNS to avoid stale upstream IPs after container recreation
- backend containers disable Numba JIT by default to avoid Python 3.13 container cache failures during generation

## Current Beta Scope

Implemented baseline features:

- task creation and polling
- task status progression
- static preview generation
- interactive tile manifest generation
- separate default worker and tile worker queues
- PostgreSQL/Redis/MinIO-ready deployment structure

Known limitations at this stage:

- generated geography may still diverge from prompt intent
- world preview fidelity and semantic alignment need further tuning
- interactive map currently uses the generated raster tile output, not a richer vector map stack

## Recommended Next Work

- improve prompt-to-world semantic alignment
- refine terrain, climate, and biome rendering quality
- add structured logging, metrics, and retry policy controls
- tighten production deployment assumptions around PostgreSQL-only execution

