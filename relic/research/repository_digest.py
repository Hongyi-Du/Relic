"""Final-evaluator repository digest, distilled from the source closure.

The implementation below is the pure ``repo_hash`` portion of
``society_core.code_landing.environment.build_workspace_execution_profile`` at
``SocioGenesis@dda36fb563375060ae8d8850300db01eb4695d29``.  It deliberately
does not import the excluded SocietyCore / ProgramBench runtime: all required
dependencies are the neutral hashing, safe-file, and path-policy utilities.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

from .hashing import stable_hash
from .repository_paths import ignored_repository_path
from .safe_files import UnsafeRegularFileError, regular_file_fingerprint


def repository_digest(path: str | os.PathLike[str]) -> str:
    """Return the source final evaluator's canonical repository digest."""

    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"repository path is not a directory: {root}")
    return stable_hash(_repo_hash_inputs(root))


def _repo_hash_inputs(root: Path) -> dict[str, object]:
    files: list[tuple[object, ...]] = []
    for path in sorted(root.rglob("*")):
        if _ignored(path, root):
            continue
        relative = path.relative_to(root).as_posix()
        try:
            metadata = path.lstat()
        except OSError:
            files.append((relative, "unreadable_entry"))
            continue
        if stat.S_ISLNK(metadata.st_mode):
            try:
                target = os.readlink(path)
            except OSError:
                target = "unreadable"
            files.append((relative, "symlink", target, metadata.st_mode))
            continue
        if stat.S_ISREG(metadata.st_mode):
            try:
                mode, digest = regular_file_fingerprint(root, path)
            except (OSError, UnsafeRegularFileError):
                files.append(
                    (
                        relative,
                        "unreadable_file",
                        metadata.st_size,
                        metadata.st_mtime_ns,
                        metadata.st_mode,
                    )
                )
                continue
            files.append((relative, "file", metadata.st_size, mode, digest))
            continue
        entry_type = "directory" if stat.S_ISDIR(metadata.st_mode) else "other"
        files.append((relative, entry_type, metadata.st_mode))
    return {
        "files": files,
        "dependency_state": _dependency_state_inputs(root),
    }


def _dependency_state_inputs(root: Path) -> tuple[tuple[object, ...], ...]:
    entries: list[tuple[object, ...]] = []
    for name in ("node_modules", ".venv"):
        dependency_root = root / name
        if dependency_root.is_symlink() or not dependency_root.is_dir():
            continue
        for path in sorted(dependency_root.rglob("*")):
            relative = path.relative_to(root).as_posix()
            try:
                metadata = path.lstat()
            except OSError:
                continue
            if path.is_symlink():
                try:
                    target = path.readlink().as_posix()
                except OSError:
                    target = "unreadable"
                entries.append((relative, "symlink", target, metadata.st_mtime_ns))
            elif path.is_file():
                entries.append(
                    (
                        relative,
                        "file",
                        metadata.st_size,
                        metadata.st_mtime_ns,
                        metadata.st_mode,
                    )
                )
            elif path.is_dir():
                entries.append((relative, "directory", metadata.st_mode))
    return tuple(entries)


def _ignored(path: Path, root: Path) -> bool:
    return ignored_repository_path(path.relative_to(root))


__all__ = ["repository_digest"]
