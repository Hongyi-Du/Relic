"""Exception handlers that enforce the ``{"error", "detail"}`` envelope."""
from __future__ import annotations
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
_DEFAULT_CODES = {400: 'bad_request', 401: 'unauthorized', 403: 'forbidden', 404: 'not_found', 405: 'method_not_allowed', 409: 'conflict', 422: 'validation_error', 500: 'internal_server_error'}

def _code_for_status(status_code: int) -> str:
    """Map an HTTP status code to a stable error code string."""
    raise NotImplementedError('_code_for_status is not implemented yet')

def _envelope(status_code: int, code: str | None, detail: object) -> JSONResponse:
    """Return a JSONResponse with the canonical ``{error, detail}`` body."""
    raise NotImplementedError('_envelope is not implemented yet')

async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """Render HTTPException using the error envelope."""
    raise NotImplementedError('http_exception_handler is not implemented yet')

async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Render request validation errors using the error envelope."""
    raise NotImplementedError('validation_exception_handler is not implemented yet')

async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all handler so even unexpected errors stay inside the envelope."""
    raise NotImplementedError('unhandled_exception_handler is not implemented yet')

def register_error_handlers(app: FastAPI) -> None:
    """Install the envelope handlers on a FastAPI application."""
    raise NotImplementedError('register_error_handlers is not implemented yet')
