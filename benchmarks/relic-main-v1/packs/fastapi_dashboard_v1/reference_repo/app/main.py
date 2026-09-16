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


_STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Create tables and seed sample data on application startup."""
    init_db()
    SessionLocal = get_sessionmaker()
    db = SessionLocal()
    try:
        seed_sample_data(db)
    finally:
        db.close()
    yield


def create_app() -> FastAPI:
    """Construct the FastAPI app, mount routes, register handlers, seed data."""
    settings = get_settings()
    app = FastAPI(
        title="FastAPI Auth & Visualization Demo",
        version="1.0.0",
        docs_url="/docs",
        openapi_url="/openapi.json",
        lifespan=_lifespan,
    )

    register_error_handlers(app)

    prefix = settings.api_prefix
    app.include_router(auth.router, prefix=prefix)
    app.include_router(users.router, prefix=prefix)
    app.include_router(charts.router, prefix=prefix)

    if _STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    @app.get("/health", tags=["meta"])
    def health() -> dict[str, str]:
        """Liveness probe used by Docker and the test harness."""
        return {"status": "ok"}

    @app.get("/", response_class=HTMLResponse, tags=["meta"])
    def index() -> HTMLResponse:
        """Serve the visualization dashboard if available, else a stub page."""
        index_file = _STATIC_DIR / "index.html"
        if index_file.is_file():
            return HTMLResponse(index_file.read_text(encoding="utf-8"))
        return HTMLResponse("<h1>FastAPI Auth &amp; Visualization Demo</h1>")

    return app


app = create_app()
