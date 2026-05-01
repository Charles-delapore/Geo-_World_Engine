from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from enum import StrEnum
from typing import Iterator

from sqlalchemy import JSON, Boolean, DateTime, Integer, Text, create_engine, event
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from app.config import settings

_HAS_GEOALCHEMY2 = False
try:
    from geoalchemy2 import Geometry
    _HAS_GEOALCHEMY2 = True
except ImportError:
    pass


class Base(DeclarativeBase):
    pass


class TaskStatus(StrEnum):
    QUEUED = "QUEUED"
    PARSING = "PARSING"
    AWAITING_CONFIRM = "AWAITING_CONFIRM"
    GENERATING_TERRAIN = "GENERATING_TERRAIN"
    RENDERING_IMAGE = "RENDERING_IMAGE"
    READY = "READY"
    READY_INTERACTIVE = "READY_INTERACTIVE"
    FAILED = "FAILED"


class TaskRecord(Base):
    __tablename__ = "tasks"

    task_id: Mapped[str] = mapped_column(Text, primary_key=True)
    status: Mapped[str] = mapped_column(Text, index=True, default=TaskStatus.QUEUED.value)
    current_stage: Mapped[str | None] = mapped_column(Text, nullable=True)
    progress: Mapped[int] = mapped_column(Integer, default=0)
    prompt: Mapped[str] = mapped_column(Text)
    params: Mapped[dict] = mapped_column(JSON, default=dict)
    plan_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    plan_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    confirmed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    preview_ready: Mapped[bool] = mapped_column(Boolean, default=False)
    tiles_ready: Mapped[bool] = mapped_column(Boolean, default=False)
    error_msg: Mapped[str | None] = mapped_column(Text, nullable=True)
    metric_report: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )


class TaskTransition(Base):
    __tablename__ = "task_state_transitions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    task_id: Mapped[str] = mapped_column(Text, index=True)
    from_state: Mapped[str | None] = mapped_column(Text, nullable=True)
    to_state: Mapped[str] = mapped_column(Text)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class MapVersion(Base):
    __tablename__ = "map_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    task_id: Mapped[str] = mapped_column(Text, index=True)
    version_num: Mapped[int] = mapped_column(Integer)
    edit_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    edit_type: Mapped[str | None] = mapped_column(Text, nullable=True)
    parent_version: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


if _HAS_GEOALCHEMY2:
    class TerrainGeometry(Base):
        __tablename__ = "terrain_geometries"

        id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
        task_id: Mapped[str] = mapped_column(Text, index=True)
        geometry_type: Mapped[str] = mapped_column(Text)
        label: Mapped[str | None] = mapped_column(Text, nullable=True)
        properties: Mapped[dict | None] = mapped_column(JSON, nullable=True)
        geom = mapped_column(Geometry(geometry_type="GEOMETRY", srid=4326), nullable=True)
        created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


connect_args = {"check_same_thread": False} if settings.DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(
    settings.DATABASE_URL,
    future=True,
    connect_args=connect_args,
    pool_pre_ping=True,
    pool_recycle=300,
)


def _set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.execute("PRAGMA busy_timeout=10000")
    cursor.execute("PRAGMA cache_size=-64000")
    cursor.close()


if settings.DATABASE_URL.startswith("sqlite"):
    event.listen(engine, "connect", _set_sqlite_pragma)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


def get_db() -> Iterator:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def session_scope():
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
