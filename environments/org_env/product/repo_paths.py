"""Canonical, cross-platform paths for repository artifacts.

Repository paths can be proposed by an LLM, so treating them as ordinary host
paths is both ambiguous and unsafe. This module defines the one serialized form
used by product artifacts: a non-empty, repository-relative POSIX path. It
validates before normalizing; ``a/../b`` is rejected instead of silently
changing which file it names.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Final


class InvalidRepoPath(ValueError):
    """A proposed repository path is unsafe or not canonicalizable."""


_WINDOWS_INVALID_CHARS: Final = frozenset('<>:"|?*')
_WINDOWS_RESERVED_NAMES: Final = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{number}" for number in range(1, 10)}
    | {f"LPT{number}" for number in range(1, 10)}
)
_VCS_ADMIN_COMPONENTS: Final = frozenset({".git", ".hg", ".svn"})
_ARTIFACT_SLUG = re.compile(r"[^a-z0-9]+")
_MAX_COMPONENT_BYTES: Final = 255
_MAX_PATH_BYTES: Final = 4096


def _invalid(raw: object, reason: str) -> InvalidRepoPath:
    return InvalidRepoPath(f"invalid_repo_path:{reason}:{raw!r}")


def normalize_repo_relative_path(raw: str) -> str:
    """Return the canonical repository-relative form of ``raw``.

    Backslashes are accepted as path separators so a decision made on Windows
    replays identically on Linux. Absolute paths, traversal, ambiguous Windows
    aliases and repository-control directories are rejected. No filesystem is
    consulted; collision checks against an existing tree remain the caller's
    responsibility.
    """
    if not isinstance(raw, str):
        raise _invalid(raw, "not_a_string")
    if not raw:
        raise _invalid(raw, "empty")
    if raw != raw.strip():
        raise _invalid(raw, "surrounding_whitespace")
    normalized = unicodedata.normalize("NFC", raw)
    if any(unicodedata.category(char) in {"Cc", "Cf", "Cs"} for char in normalized):
        raise _invalid(raw, "unicode_control_character")

    normalized = normalized.replace("\\", "/")
    if normalized.startswith("/"):
        raise _invalid(raw, "absolute")
    if re.match(r"^[A-Za-z]:", normalized):
        raise _invalid(raw, "drive_qualified")
    if len(normalized.encode("utf-8")) > _MAX_PATH_BYTES:
        raise _invalid(raw, "path_too_long")

    parts = normalized.split("/")
    if any(part == "" for part in parts):
        raise _invalid(raw, "empty_component")
    for part in parts:
        if len(part.encode("utf-8")) > _MAX_COMPONENT_BYTES:
            raise _invalid(raw, "component_too_long")
        if part in {".", ".."}:
            raise _invalid(raw, "dot_component")
        if part.casefold() in _VCS_ADMIN_COMPONENTS:
            raise _invalid(raw, "vcs_admin_component")
        if part.endswith((" ", ".")):
            raise _invalid(raw, "windows_ambiguous_component")
        if any(char in _WINDOWS_INVALID_CHARS for char in part):
            raise _invalid(raw, "windows_invalid_character")
        stem = part.split(".", 1)[0].upper()
        if stem in _WINDOWS_RESERVED_NAMES:
            raise _invalid(raw, "windows_reserved_name")

    return "/".join(parts)


def repo_artifact_id(path: str) -> str:
    """Return a stable, readable, collision-resistant id for a repo path."""
    canonical = normalize_repo_relative_path(path)
    collision_key = canonical.casefold()
    slug = _ARTIFACT_SLUG.sub("_", collision_key).strip("_")
    slug = (slug or "file")[:48].rstrip("_")
    digest = hashlib.sha256(collision_key.encode("utf-8")).hexdigest()[:20]
    return f"art_repo_{slug}_{digest}"


__all__ = [
    "InvalidRepoPath",
    "normalize_repo_relative_path",
    "repo_artifact_id",
]
