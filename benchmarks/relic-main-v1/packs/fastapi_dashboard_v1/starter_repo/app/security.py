"""Password hashing and JWT helpers."""
from __future__ import annotations
from datetime import datetime, timedelta, timezone
from typing import Any
from jose import JWTError, jwt
from passlib.context import CryptContext
from .config import get_settings
_pwd_context = CryptContext(schemes=['bcrypt'], deprecated='auto')

def hash_password(password: str) -> str:
    """Hash ``password`` with bcrypt and return the encoded digest."""
    raise NotImplementedError('hash_password is not implemented yet')

def verify_password(password: str, hashed: str) -> bool:
    """Return ``True`` if ``password`` matches the bcrypt ``hashed`` digest."""
    raise NotImplementedError('verify_password is not implemented yet')

def create_access_token(subject: str, role: str, expires_minutes: int | None=None) -> str:
    """Create a signed JWT containing the subject username and role."""
    raise NotImplementedError('create_access_token is not implemented yet')

def decode_access_token(token: str) -> dict[str, Any]:
    """Decode a JWT, raising ``JWTError`` if invalid or expired."""
    raise NotImplementedError('decode_access_token is not implemented yet')
__all__ = ['hash_password', 'verify_password', 'create_access_token', 'decode_access_token', 'JWTError']
