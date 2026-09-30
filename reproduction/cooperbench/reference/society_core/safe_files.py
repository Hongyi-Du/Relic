"""No-follow regular-file access inside a trusted workspace root."""

from __future__ import annotations

import os
import stat
import hashlib
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


class UnsafeRegularFileError(ValueError):
    pass


def read_regular_file_bytes(
    root: Path,
    path: Path | str,
    *,
    max_bytes: int,
) -> bytes:
    if max_bytes < 0:
        raise ValueError("max_bytes_must_be_nonnegative")
    with _open_regular_file(root, path) as descriptor:
        metadata = os.fstat(descriptor)
        if metadata.st_size > max_bytes:
            raise UnsafeRegularFileError("unsafe_regular_file:file_too_large")
        chunks: list[bytes] = []
        remaining = max_bytes + 1
        while remaining > 0:
            chunk = os.read(descriptor, min(64 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        if len(payload) > max_bytes:
            raise UnsafeRegularFileError("unsafe_regular_file:file_too_large")
        return payload


def read_regular_file_text(
    root: Path,
    path: Path | str,
    *,
    max_bytes: int,
) -> str:
    return read_regular_file_bytes(root, path, max_bytes=max_bytes).decode("utf-8")


def regular_file_fingerprint(root: Path, path: Path | str) -> tuple[int, str]:
    with _open_regular_file(root, path) as descriptor:
        metadata = os.fstat(descriptor)
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 64 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        return metadata.st_mode, digest.hexdigest()


@contextmanager
def _open_regular_file(root: Path, path: Path | str) -> Iterator[int]:
    lexical_root = root.absolute()
    resolved_root = lexical_root.resolve(strict=True)
    relative = _relative_path(lexical_root, resolved_root, path)
    if os.name != "posix":  # pragma: no cover - Windows compatibility.
        target = (resolved_root / relative).resolve(strict=True)
        if target.is_symlink() or not target.is_file():
            raise UnsafeRegularFileError("unsafe_regular_file:not_regular")
        try:
            target.relative_to(resolved_root)
        except ValueError:
            raise UnsafeRegularFileError("unsafe_regular_file:outside_root") from None
        descriptor = os.open(target, os.O_RDONLY)
        try:
            yield descriptor
        finally:
            os.close(descriptor)
        return

    directory_flags = os.O_RDONLY | os.O_DIRECTORY
    no_follow = getattr(os, "O_NOFOLLOW", 0)
    close_descriptors: list[int] = []
    try:
        directory_fd = os.open(resolved_root, directory_flags)
        close_descriptors.append(directory_fd)
        for component in relative.parts[:-1]:
            directory_fd = os.open(
                component,
                directory_flags | no_follow,
                dir_fd=directory_fd,
            )
            close_descriptors.append(directory_fd)
        descriptor = os.open(
            relative.parts[-1],
            os.O_RDONLY | no_follow,
            dir_fd=directory_fd,
        )
        close_descriptors.append(descriptor)
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise UnsafeRegularFileError("unsafe_regular_file:not_regular")
        yield descriptor
    except (FileNotFoundError, NotADirectoryError, OSError) as exc:
        if isinstance(exc, UnsafeRegularFileError):
            raise
        raise UnsafeRegularFileError(
            f"unsafe_regular_file:{type(exc).__name__}"
        ) from None
    finally:
        for descriptor_to_close in reversed(close_descriptors):
            try:
                os.close(descriptor_to_close)
            except OSError:
                pass


def _relative_path(lexical_root: Path, resolved_root: Path, path: Path | str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        try:
            relative = candidate.relative_to(lexical_root)
        except ValueError:
            try:
                relative = candidate.relative_to(resolved_root)
            except ValueError:
                raise UnsafeRegularFileError("unsafe_regular_file:outside_root") from None
    else:
        relative = candidate
    if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise UnsafeRegularFileError("unsafe_regular_file:invalid_path")
    return relative
