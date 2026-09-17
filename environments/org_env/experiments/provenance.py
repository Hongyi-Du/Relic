"""Deterministic provenance helpers for experiment run records."""

from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import json
import os
import re
from collections.abc import Mapping
from enum import Enum
from pathlib import Path
from typing import Any

from relic.research.repository_digest import repository_digest as _source_repository_digest

MappingSource = Mapping[str, Any] | str | os.PathLike[str]
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _json_compatible(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {
            field.name: _json_compatible(getattr(value, field.name))
            for field in dataclasses.fields(value)
            if field.repr
        }
    if isinstance(value, Enum):
        return _json_compatible(value.value)
    if isinstance(value, Mapping):
        return {
            str(key): _json_compatible(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    if isinstance(value, (set, frozenset)):
        normalized = [_json_compatible(item) for item in value]
        return sorted(
            normalized,
            key=lambda item: json.dumps(item, sort_keys=True, default=str),
        )
    if isinstance(value, (Path, os.PathLike)):
        return os.fspath(value)
    if isinstance(value, (dt.datetime, dt.date, dt.time)):
        return value.isoformat()
    if isinstance(value, float):
        return round(value, 12)
    return value


def stable_fingerprint(value: Any) -> str:
    """Hash JSON-like evidence after deterministic normalization."""

    payload = json.dumps(
        _json_compatible(value),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def require_sha256(
    value: Any,
    *,
    label: str,
    allow_none: bool = True,
) -> str | None:
    """Validate the canonical lowercase SHA-256 representation."""

    if value in (None, ""):
        if allow_none:
            return None
        raise ValueError(f"{label} must be a SHA-256 digest")
    if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
        raise ValueError(f"{label} must be 64 lowercase hexadecimal characters")
    return value


def load_mapping(source: MappingSource, *, label: str) -> dict[str, Any]:
    """Load a JSON/YAML object or copy an already-parsed mapping."""

    if isinstance(source, Mapping):
        return dict(source)
    path = Path(source).expanduser()
    text = path.read_text(encoding="utf-8")
    data: Any
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - project dependency
            raise ValueError(f"{label} requires PyYAML: {path}") from exc
        data = yaml.safe_load(text)
    else:
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            try:
                import yaml
            except ImportError as exc:  # pragma: no cover - project dependency
                raise ValueError(f"{label} is not valid JSON: {path}") from exc
            data = yaml.safe_load(text)
    if not isinstance(data, Mapping):
        raise ValueError(f"{label} must contain a JSON/YAML object")
    return dict(data)


def mapping_source_hash(
    source: MappingSource, *, label: str
) -> tuple[dict[str, Any], str]:
    payload = load_mapping(source, label=label)
    return payload, stable_fingerprint(payload)


def resolve_fingerprint(
    payload: Any,
    supplied: str | None,
    *,
    label: str,
) -> str | None:
    """Compute a fingerprint when evidence exists and reject conflicting claims."""

    declared = require_sha256(supplied, label=label)
    if payload is None:
        return declared
    computed = stable_fingerprint(payload)
    if declared is not None and declared != computed:
        raise ValueError(f"{label} mismatch: supplied={declared} computed={computed}")
    return computed


def file_sha256(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repository_digest(path: str | os.PathLike[str]) -> str:
    """Use the same repository digest as the frozen final evaluator."""

    return _source_repository_digest(path)


def directory_content_hash(path: str | os.PathLike[str]) -> str:
    """Hash evaluator source content using the final evaluator's algorithm."""

    root = Path(path).expanduser()
    if not root.is_dir():
        raise ValueError(f"content path is not a directory: {root}")
    files: dict[str, str] = {}
    skipped_directories = {
        "__pycache__",
        ".git",
        ".hg",
        ".svn",
        ".mypy_cache",
        ".pytest_cache",
    }
    for directory, directory_names, file_names in os.walk(root):
        directory_names[:] = [
            name for name in directory_names if name not in skipped_directories
        ]
        for file_name in sorted(file_names):
            if file_name.endswith((".pyc", ".pyo")) or file_name == ".DS_Store":
                continue
            candidate = Path(directory) / file_name
            relative = candidate.relative_to(root).as_posix()
            try:
                files[relative] = candidate.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
    return stable_fingerprint(files)


def checkpoint_metadata(
    checkpoint: Mapping[str, Any] | str | os.PathLike[str] | None,
) -> dict[str, Any] | None:
    """Normalize checkpoint evidence and hash a locally available checkpoint."""

    if checkpoint is None:
        return None
    direct_path = not isinstance(checkpoint, Mapping)
    metadata = {"path": os.fspath(checkpoint)} if direct_path else dict(checkpoint)
    raw_path = metadata.get("path")
    if raw_path in (None, ""):
        declared = require_sha256(
            metadata.get("sha256"),
            label="checkpoint.sha256",
        )
        if declared is not None:
            metadata["sha256"] = declared
        return sanitize_provenance(metadata)
    path = Path(str(raw_path)).expanduser()
    if not path.is_file():
        if direct_path:
            raise FileNotFoundError(path)
        metadata.pop("path", None)
        metadata["source_id"] = f"checkpoint:{path.name}:unavailable"
        declared = require_sha256(
            metadata.get("sha256"),
            label="checkpoint.sha256",
        )
        if declared is not None:
            metadata["sha256"] = declared
        return sanitize_provenance(metadata)
    computed = file_sha256(path)
    declared = require_sha256(
        metadata.get("sha256"),
        label="checkpoint.sha256",
    )
    if declared not in (None, computed):
        raise ValueError(
            f"checkpoint sha256 mismatch: supplied={declared} computed={computed}"
        )
    metadata.pop("path", None)
    metadata["source_id"] = f"checkpoint:{path.name}:sha256:{computed}"
    metadata["sha256"] = computed
    metadata["size_bytes"] = path.stat().st_size
    return sanitize_provenance(metadata)


def portable_source_identifier(source: Any) -> str:
    """Describe a mapping/path without embedding an absolute host path."""

    if isinstance(source, Mapping):
        digest = require_sha256(
            source.get("artifact_hash"),
            label="source.artifact_hash",
        )
        return f"inline-json:sha256:{digest or stable_fingerprint(source)}"
    path = Path(os.fspath(source)).expanduser()
    name = path.name or "root"
    if path.is_file():
        return f"file:{name}:sha256:{file_sha256(path)}"
    if path.is_dir():
        return f"directory:{name}"
    return f"unavailable:{name}"


def sanitize_provenance(value: Any) -> Any:
    """Recursively replace absolute host paths with portable identifiers."""

    if isinstance(value, Mapping):
        return {str(key): sanitize_provenance(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_provenance(item) for item in value]
    if isinstance(value, (Path, os.PathLike)):
        return portable_source_identifier(value)
    if isinstance(value, str):
        path = Path(value).expanduser()
        if path.is_absolute():
            return portable_source_identifier(path)
    return _json_compatible(value)


def event_graph_fingerprint(graph: Any) -> str | None:
    """Hash the full typed graph, never only its lossy summary."""

    if graph is None:
        return None
    if isinstance(graph, Mapping):
        payload: Any = dict(graph)
    else:
        nodes = getattr(graph, "nodes", None)
        edges = getattr(graph, "edges", None)
        if nodes is None and edges is None:
            raise ValueError("event graph must expose nodes/edges or be a mapping")
        payload = {
            "nodes": sorted(
                (str(key), str(value)) for key, value in (nodes or {}).items()
            ),
            "edges": sorted(
                tuple(str(part) for part in edge) for edge in (edges or ())
            ),
        }
    return stable_fingerprint(payload)


__all__ = [
    "MappingSource",
    "checkpoint_metadata",
    "directory_content_hash",
    "event_graph_fingerprint",
    "file_sha256",
    "load_mapping",
    "mapping_source_hash",
    "portable_source_identifier",
    "repository_digest",
    "require_sha256",
    "resolve_fingerprint",
    "sanitize_provenance",
    "stable_fingerprint",
]
