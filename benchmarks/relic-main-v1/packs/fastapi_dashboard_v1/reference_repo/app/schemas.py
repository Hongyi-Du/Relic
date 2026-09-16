"""Pydantic request/response models."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, EmailStr, Field

Role = Literal["user", "admin"]


class ErrorResponse(BaseModel):
    """Consistent error envelope returned for every non-2xx response."""

    error: str = Field(..., description="Machine-readable error code.")
    detail: str = Field(..., description="Human-readable error message.")


class UserCreate(BaseModel):
    """Payload for ``POST /auth/signup``."""

    username: str = Field(..., min_length=3, max_length=64)
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=128)
    role: Role = "user"


class UserRead(BaseModel):
    """User representation returned to clients (no password)."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    username: str
    email: EmailStr
    role: Role
    created_at: datetime


class LoginRequest(BaseModel):
    """Payload for ``POST /auth/login``."""

    username: str
    password: str


class TokenResponse(BaseModel):
    """JWT bearer token returned on successful login."""

    access_token: str
    token_type: Literal["bearer"] = "bearer"


class SalesPoint(BaseModel):
    """One row of the sales dataset."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    category: str
    month: str
    amount: float


class SalesByCategory(BaseModel):
    """Aggregated sales total for one category."""

    category: str
    total: float


class SalesByMonth(BaseModel):
    """Aggregated sales total for one month."""

    month: str
    total: float
