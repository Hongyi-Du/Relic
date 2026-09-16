"""Authentication routes: signup, login (JSON), and OAuth2 token endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from ..db import get_db
from ..models import User
from ..schemas import LoginRequest, TokenResponse, UserCreate, UserRead
from ..security import create_access_token, hash_password, verify_password


router = APIRouter(prefix="/auth", tags=["auth"])


def _conflict(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_409_CONFLICT,
        detail={"error": "conflict", "detail": detail},
    )


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail={"error": "unauthorized", "detail": detail},
        headers={"WWW-Authenticate": "Bearer"},
    )


@router.post("/signup", response_model=UserRead, status_code=status.HTTP_201_CREATED)
def signup(payload: UserCreate, db: Session = Depends(get_db)) -> User:
    """Create a new user with a hashed password.

    Returns the persisted user row. Raises 409 if the username or email
    is already taken.
    """
    if db.query(User).filter(User.username == payload.username).first() is not None:
        raise _conflict("username already exists")
    if db.query(User).filter(User.email == payload.email).first() is not None:
        raise _conflict("email already registered")

    user = User(
        username=payload.username,
        email=payload.email,
        hashed_password=hash_password(payload.password),
        role=payload.role,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def _authenticate(db: Session, username: str, password: str) -> User:
    user = db.query(User).filter(User.username == username).first()
    if user is None or not verify_password(password, user.hashed_password):
        raise _unauthorized("invalid credentials")
    return user


@router.post("/login", response_model=TokenResponse)
def login(payload: LoginRequest, db: Session = Depends(get_db)) -> TokenResponse:
    """Validate username/password and return a fresh JWT bearer token."""
    user = _authenticate(db, payload.username, payload.password)
    token = create_access_token(subject=user.username, role=user.role)
    return TokenResponse(access_token=token)


@router.post("/token", response_model=TokenResponse)
def token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
) -> TokenResponse:
    """OAuth2 password flow endpoint used by Swagger UI's "Authorize" dialog."""
    user = _authenticate(db, form_data.username, form_data.password)
    access = create_access_token(subject=user.username, role=user.role)
    return TokenResponse(access_token=access)
