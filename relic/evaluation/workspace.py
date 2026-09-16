"""Disposable evaluator workspaces and deterministic repository digests."""

from __future__ import annotations

import hashlib
import shutil
import stat
import tempfile
from pathlib import Path
from types import TracebackType
from typing import Self

_IGNORED_NAMES = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "__pycache__",
    }
)


def _ignore(_directory: str, names: list[str]) -> set[str]:
    return {name for name in names if name in _IGNORED_NAMES or name.endswith((".pyc", ".pyo"))}


def _entry_records(root: Path) -> tuple[tuple[object, ...], ...]:
    records: list[tuple[object, ...]] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if any(part in _IGNORED_NAMES for part in path.relative_to(root).parts):
            continue
        try:
            metadata = path.lstat()
        except OSError:
            records.append((relative, "unreadable"))
            continue
        mode = stat.S_IMODE(metadata.st_mode)
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"candidate_workspace_symlink_forbidden:{relative}")
        elif stat.S_ISREG(metadata.st_mode):
            digest = hashlib.sha256()
            try:
                with path.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(chunk)
            except OSError:
                records.append((relative, "unreadable_file", mode, metadata.st_size))
            else:
                records.append((relative, "file", mode, metadata.st_size, digest.hexdigest()))
        elif stat.S_ISDIR(metadata.st_mode):
            records.append((relative, "directory", mode))
        else:
            records.append((relative, "other", mode))
    return tuple(records)


def repository_digest(root: Path) -> str:
    """Hash repository paths, types, modes, link targets, and file contents."""

    resolved = root.resolve()
    if not resolved.is_dir():
        return hashlib.sha256(f"missing-repo:{root.name}".encode()).hexdigest()
    digest = hashlib.sha256()
    for record in _entry_records(resolved):
        for field in record:
            digest.update(str(field).encode("utf-8", errors="surrogateescape"))
            digest.update(b"\0")
        digest.update(b"\n")
    return digest.hexdigest()


class CandidateWorkspace:
    """Context-managed, isolated copy used for one evaluator invocation."""

    def __init__(self, root: Path, temporary: tempfile.TemporaryDirectory[str]) -> None:
        self.root = root
        self._temporary = temporary

    @classmethod
    def create(
        cls,
        source_root: Path,
        *,
        recover_interrupted_promotion: bool = True,
    ) -> CandidateWorkspace:
        del recover_interrupted_promotion
        source = source_root.resolve()
        if not source.is_dir():
            raise ValueError("candidate_workspace_source_not_directory")
        before = repository_digest(source)
        temporary = tempfile.TemporaryDirectory(prefix="relic_candidate_")
        root = Path(temporary.name) / "workspace"
        try:
            shutil.copytree(
                source,
                root,
                symlinks=True,
                ignore=_ignore,
            )
            if repository_digest(source) != before:
                raise RuntimeError("candidate_workspace_source_changed_during_clone")
            if repository_digest(root) != before:
                raise RuntimeError("candidate_workspace_clone_mismatch")
        except BaseException:
            temporary.cleanup()
            raise
        return cls(root, temporary)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback
        self._temporary.cleanup()


__all__ = ["CandidateWorkspace", "repository_digest"]
