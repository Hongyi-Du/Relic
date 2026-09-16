"""Stable canonical hashing for the shared public trace contract."""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import json
from pathlib import Path
from typing import Any


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _jsonable(getattr(value, field.name)) for field in dataclasses.fields(value)
        }
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_jsonable(item) for item in value), key=lambda item: repr(item))
    return value


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def stable_fingerprint(value: Any) -> str:
    """Compatibility name used by the extracted B3 protocol registry."""

    return canonical_sha256(value)
