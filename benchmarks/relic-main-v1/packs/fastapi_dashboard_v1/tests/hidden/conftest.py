"""Shared pytest fixtures: per-test SQLite DB, FastAPI client, seeded data."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Iterator

import pytest


# Configure environment BEFORE importing the app.
_TMP_DIR = Path(tempfile.mkdtemp(prefix="fastapi-task-"))
os.environ["DATABASE_URL"] = f"sqlite:///{_TMP_DIR / 'session.db'}"
os.environ["JWT_SECRET"] = "test-secret"
os.environ["JWT_ALGORITHM"] = "HS256"
os.environ["JWT_EXPIRES_MINUTES"] = "60"


@pytest.fixture()
def fresh_db(monkeypatch, tmp_path) -> Iterator[Path]:
    """Point the app at a brand-new SQLite DB for the duration of the test."""
    from app import db as db_module

    db_path = tmp_path / "test.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    db_module.reset_engine()
    yield db_path
    db_module.reset_engine()


@pytest.fixture()
def app(fresh_db):
    """Build a FastAPI app instance bound to the fresh DB and seed it."""
    from app.main import create_app
    from app.db import get_sessionmaker, init_db
    from app.seed import seed_sample_data

    application = create_app()
    init_db()
    SessionLocal = get_sessionmaker()
    session = SessionLocal()
    try:
        seed_sample_data(session)
    finally:
        session.close()
    return application


@pytest.fixture()
def client(app):
    """A ``TestClient`` wrapping the freshly-built app."""
    from fastapi.testclient import TestClient

    with TestClient(app) as c:
        yield c


@pytest.fixture()
def admin_token(client) -> str:
    """Return a JWT bearer token for the seeded admin user."""
    resp = client.post("/api/v1/auth/login", json={"username": "admin", "password": "adminpass"})
    assert resp.status_code == 200, resp.text
    return resp.json()["access_token"]


@pytest.fixture()
def user_token(client) -> str:
    """Sign up a fresh regular user and return their JWT bearer token."""
    payload = {
        "username": "alice",
        "email": "alice@example.com",
        "password": "alicepass1",
        "role": "user",
    }
    signup = client.post("/api/v1/auth/signup", json=payload)
    assert signup.status_code == 201, signup.text
    login = client.post(
        "/api/v1/auth/login",
        json={"username": "alice", "password": "alicepass1"},
    )
    assert login.status_code == 200, login.text
    return login.json()["access_token"]


def auth_headers(token: str) -> dict[str, str]:
    """Return the ``Authorization`` header for a bearer token."""
    return {"Authorization": f"Bearer {token}"}
