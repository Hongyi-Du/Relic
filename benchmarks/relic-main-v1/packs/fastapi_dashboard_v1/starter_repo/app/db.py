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
    raise NotImplementedError('_make_engine is not implemented yet')

def get_engine() -> Engine:
    """Return the process-wide SQLAlchemy engine, building it on first call."""
    raise NotImplementedError('get_engine is not implemented yet')

def get_sessionmaker() -> sessionmaker[Session]:
    """Return the process-wide session factory."""
    raise NotImplementedError('get_sessionmaker is not implemented yet')

def reset_engine() -> None:
    """Drop the cached engine and session factory. Used by the test suite."""
    raise NotImplementedError('reset_engine is not implemented yet')

def init_db() -> None:
    """Create all tables registered on ``Base.metadata``."""
    raise NotImplementedError('init_db is not implemented yet')

def get_db() -> Iterator[Session]:
    """FastAPI dependency that yields a database session and closes it."""
    raise NotImplementedError('get_db is not implemented yet')
