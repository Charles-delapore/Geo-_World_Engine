from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from app.api.routes.health import health as api_health
from app.api.routes.health import ready as api_ready
from app.api.routes import router
from app.config import settings
from app.rag.init_kb import init_builtin_knowledge_base
from app.storage.models import init_db, session_scope


app = FastAPI(title=settings.PROJECT_NAME, version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router, prefix=settings.API_V1_STR)


@app.on_event("startup")
def startup() -> None:
    if settings.DATABASE_URL.startswith("sqlite"):
        init_db()
        with session_scope() as db:
            db.execute(text("SELECT 1"))
    if settings.ENABLE_RAG:
        init_builtin_knowledge_base(force=False)


@app.get("/")
def root():
    return {"name": settings.PROJECT_NAME, "status": "ok"}


@app.get("/health")
def health():
    return api_health()


@app.get("/ready")
def ready():
    return api_ready()
