"""Reusable FastAPI dependencies for authentication and role gating."""
from __future__ import annotations
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from .config import get_settings
from .db import get_db
from .models import User
from .security import JWTError, decode_access_token
oauth2_scheme = OAuth2PasswordBearer(tokenUrl=f'{get_settings().api_prefix}/auth/token', auto_error=False)

def _unauthorized(detail: str) -> HTTPException:
    raise NotImplementedError('_unauthorized is not implemented yet')

def _forbidden(detail: str) -> HTTPException:
    raise NotImplementedError('_forbidden is not implemented yet')

def get_current_user(token: str | None=Depends(oauth2_scheme), db: Session=Depends(get_db)) -> User:
    """Resolve the JWT in the ``Authorization`` header to a ``User`` row."""
    raise NotImplementedError('get_current_user is not implemented yet')

def require_admin(current_user: User=Depends(get_current_user)) -> User:
    """Allow only users whose role is ``admin``."""
    raise NotImplementedError('require_admin is not implemented yet')
