"""Credential redaction for durable evaluator and trace payloads."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any


_SECRET_TEXT_PATTERNS = (
    re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/-]{20,}=*"),
)


def redact_sensitive_payload(value: Any, *, key: str = "") -> Any:
    """Remove credential-like material before persistence or prompt-visible use."""

    lowered = key.lower()
    sensitive = (
        "api_key" in lowered
        or lowered == "token"
        or lowered.endswith("_token")
        or "secret" in lowered
        or "password" in lowered
        or "authorization" in lowered
    )
    if sensitive:
        return "[REDACTED]"
    if isinstance(value, Mapping):
        return {
            str(item_key): redact_sensitive_payload(item_value, key=str(item_key))
            for item_key, item_value in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact_sensitive_payload(item) for item in value]
    if isinstance(value, str):
        redacted = value
        for pattern in _SECRET_TEXT_PATTERNS:
            redacted = pattern.sub("[REDACTED]", redacted)
        return redacted
    return value
