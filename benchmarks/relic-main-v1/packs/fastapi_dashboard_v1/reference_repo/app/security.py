"""Password hashing and JWT helpers."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from jose import JWTError, jwt
from passlib.context import CryptContext

from .config import get_settings


_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    """Hash ``password`` with bcrypt and return the encoded digest."""
    return _pwd_context.hash(password)


def verify_password(password: str, hashed: str) -> bool:
    """Return ``True`` if ``password`` matches the bcrypt ``hashed`` digest."""
    try:
        return _pwd_context.verify(password, hashed)
    except ValueError:
        return False


def create_access_token(subject: str, role: str, expires_minutes: int | None = None) -> str:
    """Create a signed JWT containing the subject username and role."""
    settings = get_settings()
    expires_in = expires_minutes if expires_minutes is not None else settings.jwt_expires_minutes
    now = datetime.now(timezone.utc)
    payload: dict[str, Any] = {
        "sub": subject,
        "role": role,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=expires_in)).timestamp()),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def decode_access_token(token: str) -> dict[str, Any]:
    """Decode a JWT, raising ``JWTError`` if invalid or expired."""
    settings = get_settings()
    return jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])


__all__ = [
    "hash_password",
    "verify_password",
    "create_access_token",
    "decode_access_token",
    "JWTError",
]
