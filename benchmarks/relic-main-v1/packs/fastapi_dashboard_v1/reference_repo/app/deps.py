"""Reusable FastAPI dependencies for authentication and role gating."""

from __future__ import annotations

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from .config import get_settings
from .db import get_db
from .models import User
from .security import JWTError, decode_access_token


oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl=f"{get_settings().api_prefix}/auth/token",
    auto_error=False,
)


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail={"error": "unauthorized", "detail": detail},
        headers={"WWW-Authenticate": "Bearer"},
    )


def _forbidden(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail={"error": "forbidden", "detail": detail},
    )


def get_current_user(
    token: str | None = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> User:
    """Resolve the JWT in the ``Authorization`` header to a ``User`` row."""
    if not token:
        raise _unauthorized("missing bearer token")
    try:
        payload = decode_access_token(token)
    except JWTError:
        raise _unauthorized("invalid or expired token")
    username = payload.get("sub")
    if not isinstance(username, str):
        raise _unauthorized("token missing subject")
    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise _unauthorized("user no longer exists")
    return user


def require_admin(current_user: User = Depends(get_current_user)) -> User:
    """Allow only users whose role is ``admin``."""
    if current_user.role != "admin":
        raise _forbidden("admin role required")
    return current_user
