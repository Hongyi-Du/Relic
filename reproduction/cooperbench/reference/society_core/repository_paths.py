"""Shared repository path visibility and mutation policy."""

from __future__ import annotations

from pathlib import PurePath, PurePosixPath
from typing import Iterable

from .hashing import stable_hash


REPOSITORY_PATH_POLICY_VERSION = "2"
_IGNORED_ANYWHERE_COMPONENTS = frozenset(
    {
        ".git",
        ".hg",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".svn",
        ".venv",
        "__pycache__",
        "node_modules",
    }
)
_IGNORED_ROOT_COMPONENTS = frozenset({"coverage", "outputs"})


def repository_path_parts(path: str | PurePath | Iterable[str]) -> tuple[str, ...]:
    """Normalize an already-relative repository path into stable components."""

    if isinstance(path, str):
        value = PurePosixPath(path)
        parts = value.parts
    elif isinstance(path, PurePath):
        parts = path.parts
    else:
        parts = tuple(str(part) for part in path)
    return tuple(str(part) for part in parts if str(part) not in {"", "."})


def ignored_repository_path(path: str | PurePath | Iterable[str]) -> bool:
    """Return whether a relative path is harness/runtime state, not source."""

    parts = repository_path_parts(path)
    if not parts:
        return False
    return parts[0] in _IGNORED_ROOT_COMPONENTS or any(
        part in _IGNORED_ANYWHERE_COMPONENTS for part in parts
    )


def repository_path_exclusion_reason(
    path: str | PurePath | Iterable[str],
) -> str | None:
    """Return the stable exclusion class for a relative path."""

    parts = repository_path_parts(path)
    if not parts:
        return None
    if parts[0] in _IGNORED_ROOT_COMPONENTS:
        return "root_runtime_artifact"
    if any(part in _IGNORED_ANYWHERE_COMPONENTS for part in parts):
        return "runtime_or_dependency_component"
    return None


def repository_path_policy_payload() -> dict[str, object]:
    """Describe the versioned policy embedded in content-addressed evidence."""

    payload: dict[str, object] = {
        "version": REPOSITORY_PATH_POLICY_VERSION,
        "ignored_anywhere_components": tuple(sorted(_IGNORED_ANYWHERE_COMPONENTS)),
        "ignored_root_components": tuple(sorted(_IGNORED_ROOT_COMPONENTS)),
        "nested_semantic_names_preserved": True,
    }
    return {**payload, "policy_hash": stable_hash(payload)}


__all__ = [
    "REPOSITORY_PATH_POLICY_VERSION",
    "ignored_repository_path",
    "repository_path_exclusion_reason",
    "repository_path_parts",
    "repository_path_policy_payload",
]
