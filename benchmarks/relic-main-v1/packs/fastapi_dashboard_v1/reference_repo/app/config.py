"""Configuration loaded from environment variables."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings for the FastAPI application.

    Values are read from environment variables; defaults make the service
    runnable out of the box for tests.
    """

    database_url: str = "sqlite:///./app.db"
    jwt_secret: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expires_minutes: int = 60
    api_prefix: str = "/api/v1"

    model_config = SettingsConfigDict(env_file=None, env_prefix="", extra="ignore")


def get_settings() -> Settings:
    """Return a fresh ``Settings`` instance from the current environment."""
    return Settings()
