"""Exception handlers that enforce the ``{"error", "detail"}`` envelope."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException


_DEFAULT_CODES = {
    400: "bad_request",
    401: "unauthorized",
    403: "forbidden",
    404: "not_found",
    405: "method_not_allowed",
    409: "conflict",
    422: "validation_error",
    500: "internal_server_error",
}


def _code_for_status(status_code: int) -> str:
    """Map an HTTP status code to a stable error code string."""
    return _DEFAULT_CODES.get(status_code, "http_error")


def _envelope(status_code: int, code: str | None, detail: object) -> JSONResponse:
    """Return a JSONResponse with the canonical ``{error, detail}`` body."""
    if isinstance(detail, dict) and "error" in detail and "detail" in detail:
        body = {"error": str(detail["error"]), "detail": str(detail["detail"])}
    else:
        body = {
            "error": code or _code_for_status(status_code),
            "detail": detail if isinstance(detail, str) else str(detail),
        }
    return JSONResponse(status_code=status_code, content=body)


async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """Render HTTPException using the error envelope."""
    return _envelope(exc.status_code, None, exc.detail)


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Render request validation errors using the error envelope."""
    detail = "; ".join(
        f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in exc.errors()
    ) or "validation error"
    return _envelope(422, "validation_error", detail)


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all handler so even unexpected errors stay inside the envelope."""
    return _envelope(500, "internal_server_error", "internal server error")


def register_error_handlers(app: FastAPI) -> None:
    """Install the envelope handlers on a FastAPI application."""
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)
