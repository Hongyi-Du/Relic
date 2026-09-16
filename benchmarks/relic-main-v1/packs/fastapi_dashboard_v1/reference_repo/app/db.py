"""SQLAlchemy engine, session factory, and Base for ORM models."""

from __future__ import annotations

from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from .config import get_settings


class Base(DeclarativeBase):
    """Declarative base for ORM models."""


_engine: Engine | None = None
_SessionLocal: sessionmaker[Session] | None = None


def _make_engine(database_url: str) -> Engine:
    """Construct a SQLAlchemy engine, applying SQLite-specific options."""
    if database_url.startswith("sqlite"):
        return create_engine(
            database_url,
            connect_args={"check_same_thread": False},
            future=True,
        )
    return create_engine(database_url, future=True)


def get_engine() -> Engine:
    """Return the process-wide SQLAlchemy engine, building it on first call."""
    global _engine, _SessionLocal
    if _engine is None:
        settings = get_settings()
        _engine = _make_engine(settings.database_url)
        _SessionLocal = sessionmaker(
            bind=_engine, autoflush=False, autocommit=False, future=True
        )
    return _engine


def get_sessionmaker() -> sessionmaker[Session]:
    """Return the process-wide session factory."""
    if _SessionLocal is None:
        get_engine()
    assert _SessionLocal is not None
    return _SessionLocal


def reset_engine() -> None:
    """Drop the cached engine and session factory. Used by the test suite."""
    global _engine, _SessionLocal
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _SessionLocal = None


def init_db() -> None:
    """Create all tables registered on ``Base.metadata``."""
    from . import models  # noqa: F401 - register models

    engine = get_engine()
    Base.metadata.create_all(bind=engine)


def get_db() -> Iterator[Session]:
    """FastAPI dependency that yields a database session and closes it."""
    SessionLocal = get_sessionmaker()
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
