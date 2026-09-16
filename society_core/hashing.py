"""Canonical serialization and hashing for replay checks."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from enum import Enum
from typing import Any


def canonicalize(value: Any) -> Any:
    """Convert dataclass-heavy state into deterministic JSON-compatible data."""
    if dataclasses.is_dataclass(value):
        return {
            field.name: canonicalize(getattr(value, field.name))
            for field in dataclasses.fields(value)
            if field.repr
        }
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {
            str(key): canonicalize(val)
            for key, val in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [canonicalize(item) for item in value]
    if isinstance(value, set):
        return sorted(canonicalize(item) for item in value)
    if isinstance(value, float):
        return round(value, 12)
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(canonicalize(value), sort_keys=True, separators=(",", ":"))


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
