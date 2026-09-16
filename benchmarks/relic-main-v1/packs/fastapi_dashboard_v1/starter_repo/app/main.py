"""FastAPI application factory: wires routers, error handlers, and startup."""
from __future__ import annotations
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from .config import get_settings
from .db import get_sessionmaker, init_db
from .errors import register_error_handlers
from .routers import auth, charts, users
from .seed import seed_sample_data
_STATIC_DIR = Path(__file__).resolve().parent / 'static'

@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Create tables and seed sample data on application startup."""
    raise NotImplementedError('_lifespan is not implemented yet')

def create_app() -> FastAPI:
    """Construct the FastAPI app, mount routes, register handlers, seed data."""
    raise NotImplementedError('create_app is not implemented yet')
app = create_app()
