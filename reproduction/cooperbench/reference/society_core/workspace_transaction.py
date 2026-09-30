"""Transactional candidate workspaces and atomic verified promotion."""

from __future__ import annotations

import hashlib
import base64
import ctypes
import errno
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Iterable, Iterator

from .execution import CommandExecutor
from .repository_paths import ignored_repository_path
from .safe_files import (
    UnsafeRegularFileError,
    read_regular_file_bytes,
    regular_file_fingerprint,
)
from .workspace_update import (
    WorkspaceFilePatch,
    WorkspacePatchResult,
    WorkspaceVerificationResult,
    apply_workspace_patches,
    run_workspace_verification,
)


_THREAD_LOCK_GUARD = threading.Lock()
_THREAD_LOCKS: dict[str, threading.Lock] = {}
_CANDIDATE_CLONE_ATTEMPTS = 3
_MAX_PROMOTION_SNAPSHOT_BYTES = 80_000_000


@dataclass(frozen=True)
class WorkspacePromotionResult:
    status: str
    changed_paths: tuple[str, ...]
    verification_results: tuple[WorkspaceVerificationResult, ...]
    baseline_digest: str
    candidate_digest: str
    promoted_digest: str | None
    rollback_performed: bool
    blocked_reason: str | None = None


@dataclass(frozen=True)
class WorkspaceCommitReceipt:
    canonical_root: str
    baseline_digest: str
    candidate_digest: str
    changed_paths: tuple[str, ...]


@dataclass(frozen=True)
class _FileSnapshot:
    existed: bool
    content: bytes
    mode: int | None


class _ConcurrentWorkspaceModification(RuntimeError):
    pass


class CandidateWorkspace:
    """A disposable repository copy whose patches can be promoted as one unit."""

    def __init__(
        self,
        *,
        canonical_root: Path,
        root: Path,
        temporary_directory: tempfile.TemporaryDirectory[str],
        baseline_files: dict[str, str],
    ) -> None:
        self.canonical_root = canonical_root
        self.root = root
        self._temporary_directory = temporary_directory
        self._baseline_files = baseline_files
        self._baseline_path_fingerprints: dict[str, str | None] = {}
        self._patch_paths: list[str] = []
        self._closed = False
        self._promoted = False

    @classmethod
    def create(
        cls,
        canonical_root: Path,
        *,
        recover_interrupted_promotion: bool = True,
    ) -> CandidateWorkspace:
        resolved_root = canonical_root.resolve()
        if not resolved_root.is_dir():
            raise ValueError("canonical_workspace_not_directory")
        if recover_interrupted_promotion:
            recover_interrupted_workspace_promotion(resolved_root)
        for _ in range(_CANDIDATE_CLONE_ATTEMPTS):
            baseline_files = _workspace_file_map(resolved_root)
            temporary_directory = tempfile.TemporaryDirectory(
                prefix="society_candidate_",
                dir=_candidate_workspace_parent(),
            )
            candidate_root = Path(temporary_directory.name) / "workspace"
            try:
                shutil.copytree(
                    resolved_root,
                    candidate_root,
                    symlinks=True,
                    ignore=_copy_ignore_for_root(resolved_root),
                )
                _clone_dependency_directories(
                    source_root=resolved_root,
                    candidate_root=candidate_root,
                )
                candidate_files = _workspace_file_map(candidate_root)
                final_canonical_files = _workspace_file_map(resolved_root)
            except BaseException:
                temporary_directory.cleanup()
                raise
            if (
                baseline_files == candidate_files
                and baseline_files == final_canonical_files
            ):
                return cls(
                    canonical_root=resolved_root,
                    root=candidate_root,
                    temporary_directory=temporary_directory,
                    baseline_files=baseline_files,
                )
            temporary_directory.cleanup()
        raise RuntimeError("canonical_workspace_changed_during_candidate_clone")

    @property
    def baseline_digest(self) -> str:
        return _file_map_digest(self._baseline_files)

    @property
    def candidate_digest(self) -> str:
        self._ensure_open()
        return _file_map_digest(self._projected_candidate_files(self.changed_paths()))

    def apply(
        self,
        patches: tuple[WorkspaceFilePatch, ...],
        *,
        gate_policy: str = "claim",
    ) -> tuple[WorkspacePatchResult, ...]:
        self._ensure_open()
        for patch in patches:
            relative_path = _safe_relative_path(patch.path)
            self._baseline_path_fingerprints.setdefault(
                relative_path,
                self._baseline_files.get(relative_path),
            )
        results = apply_workspace_patches(
            self.root,
            patches,
            gate_policy=gate_policy,
        )
        for result in results:
            if result.status == "applied" and result.path not in self._patch_paths:
                self._patch_paths.append(result.path)
        return results

    def verify(
        self,
        commands: tuple[str, ...],
        *,
        timeout_seconds: float,
        executor: CommandExecutor,
    ) -> tuple[WorkspaceVerificationResult, ...]:
        self._ensure_open()
        unique_commands = tuple(dict.fromkeys(commands))
        if not unique_commands or len(unique_commands) > 32:
            with CandidateWorkspace.create(self.root) as verification_workspace:
                return run_workspace_verification(
                    verification_workspace.root,
                    unique_commands,
                    timeout_seconds=timeout_seconds,
                    executor=executor,
                )
        results: list[WorkspaceVerificationResult] = []
        for command in unique_commands:
            with CandidateWorkspace.create(self.root) as verification_workspace:
                results.extend(
                    run_workspace_verification(
                        verification_workspace.root,
                        (command,),
                        timeout_seconds=timeout_seconds,
                        executor=executor,
                    )
                )
        return tuple(results)

    def promote(
        self,
        *,
        verification_commands: tuple[str, ...],
        timeout_seconds: float,
        executor: CommandExecutor,
        retain_commit_receipt: bool = False,
    ) -> WorkspacePromotionResult:
        self._ensure_open()
        changed_paths = self.changed_paths()
        candidate_files = self._projected_candidate_files(changed_paths)
        candidate_digest = _file_map_digest(candidate_files)
        if self._promoted:
            return self._result(
                status="blocked",
                changed_paths=changed_paths,
                candidate_digest=candidate_digest,
                blocked_reason="candidate_already_promoted",
            )
        if not verification_commands:
            return self._result(
                status="blocked",
                changed_paths=changed_paths,
                candidate_digest=candidate_digest,
                blocked_reason="promotion_requires_verification",
            )
        if not changed_paths:
            return self._result(
                status="no_changes",
                changed_paths=(),
                candidate_digest=candidate_digest,
                blocked_reason="candidate_has_no_effective_changes",
            )
        candidate_verification = self.verify(
            verification_commands,
            timeout_seconds=timeout_seconds,
            executor=executor,
        )
        if not candidate_verification or any(
            result.status != "passed" for result in candidate_verification
        ):
            return self._result(
                status="verification_failed",
                changed_paths=changed_paths,
                candidate_digest=candidate_digest,
                verification_results=candidate_verification,
                blocked_reason="candidate_verification_failed",
            )
        promoted_fingerprints = {
            path: _path_fingerprint(self.root / path, root=self.root)
            for path in changed_paths
        }
        with _workspace_promotion_lock(self.canonical_root):
            _recover_interrupted_promotion(self.canonical_root)
            concurrency_error = self._concurrency_error(changed_paths)
            if concurrency_error is not None:
                return self._result(
                    status="concurrent_modification",
                    changed_paths=changed_paths,
                    candidate_digest=candidate_digest,
                    blocked_reason=concurrency_error,
                )

            snapshots: dict[str, _FileSnapshot] = {}
            created_directories: list[Path] = []
            staging_relative_paths = _promotion_staging_paths(changed_paths)
            staging_paths = {
                path: self.canonical_root / staging_relative_paths[path]
                for path in changed_paths
            }
            journal_prepared = False
            try:
                snapshots = {
                    path: _snapshot_path(
                        self.canonical_root / path,
                        root=self.canonical_root,
                    )
                    for path in changed_paths
                }
                if sum(len(snapshot.content) for snapshot in snapshots.values()) > (
                    _MAX_PROMOTION_SNAPSHOT_BYTES
                ):
                    raise ValueError("promotion_snapshot_budget_exceeded")
                missing_parent_directories = _missing_parent_directories(
                    self.canonical_root,
                    changed_paths,
                )
                _write_promotion_journal(
                    root=self.canonical_root,
                    state="prepared",
                    baseline_digest=self.baseline_digest,
                    candidate_digest=candidate_digest,
                    snapshots=snapshots,
                    missing_parent_directories=missing_parent_directories,
                    promoted_fingerprints=promoted_fingerprints,
                    staging_paths=staging_relative_paths,
                )
                journal_prepared = True
                for relative_path in changed_paths:
                    _promote_path(
                        source_root=self.root,
                        target_root=self.canonical_root,
                        relative_path=relative_path,
                        created_directories=created_directories,
                        expected_target_fingerprint=(
                            self._baseline_path_fingerprints.get(relative_path)
                        ),
                        staging_path=staging_paths[relative_path],
                    )
                _write_promotion_journal(
                    root=self.canonical_root,
                    state="canonical_written",
                    baseline_digest=self.baseline_digest,
                    candidate_digest=candidate_digest,
                    snapshots=snapshots,
                    missing_parent_directories=missing_parent_directories,
                    promoted_fingerprints=promoted_fingerprints,
                    staging_paths=staging_relative_paths,
                )
                with CandidateWorkspace.create(
                    self.canonical_root,
                    recover_interrupted_promotion=False,
                ) as verification_workspace:
                    verification_results = verification_workspace.verify(
                        verification_commands,
                        timeout_seconds=timeout_seconds,
                        executor=executor,
                    )
                if not verification_results or any(
                    result.status != "passed" for result in verification_results
                ):
                    conflicts, restored_count = _restore_paths(
                        root=self.canonical_root,
                        snapshots=snapshots,
                        created_directories=created_directories,
                        expected_current=promoted_fingerprints,
                        staging_paths=staging_paths,
                    )
                    _remove_promotion_journal(self.canonical_root)
                    if conflicts:
                        return self._result(
                            status="concurrent_modification",
                            changed_paths=changed_paths,
                            candidate_digest=candidate_digest,
                            verification_results=verification_results,
                            rollback_performed=restored_count > 0,
                            blocked_reason=f"rollback_conflict:{conflicts[0]}",
                        )
                    return self._result(
                        status="verification_failed",
                        changed_paths=changed_paths,
                        candidate_digest=candidate_digest,
                        verification_results=verification_results,
                        rollback_performed=True,
                        blocked_reason="post_promotion_verification_failed",
                    )
                promoted_digest = _file_map_digest(
                    _workspace_file_map(self.canonical_root)
                )
                if promoted_digest != candidate_digest:
                    conflicts, restored_count = _restore_paths(
                        root=self.canonical_root,
                        snapshots=snapshots,
                        created_directories=created_directories,
                        expected_current=promoted_fingerprints,
                        staging_paths=staging_paths,
                    )
                    _remove_promotion_journal(self.canonical_root)
                    if conflicts:
                        return self._result(
                            status="concurrent_modification",
                            changed_paths=changed_paths,
                            candidate_digest=candidate_digest,
                            verification_results=verification_results,
                            rollback_performed=restored_count > 0,
                            blocked_reason=f"rollback_conflict:{conflicts[0]}",
                        )
                    return self._result(
                        status="verification_failed",
                        changed_paths=changed_paths,
                        candidate_digest=candidate_digest,
                        verification_results=verification_results,
                        rollback_performed=True,
                        blocked_reason="canonical_digest_mismatch_after_verification",
                    )
                _write_promotion_journal(
                    root=self.canonical_root,
                    state="committed",
                    baseline_digest=self.baseline_digest,
                    candidate_digest=candidate_digest,
                    snapshots=snapshots,
                    missing_parent_directories=missing_parent_directories,
                    promoted_fingerprints=promoted_fingerprints,
                    staging_paths=staging_relative_paths,
                )
            except Exception as exc:
                rollback_performed = False
                rollback_conflicts: tuple[str, ...] = ()
                try:
                    if journal_prepared:
                        rollback_conflicts, restored_count = _restore_paths(
                            root=self.canonical_root,
                            snapshots=snapshots,
                            created_directories=created_directories,
                            expected_current=promoted_fingerprints,
                            staging_paths=staging_paths,
                        )
                        _remove_promotion_journal(self.canonical_root)
                        rollback_performed = restored_count > 0
                except Exception:
                    rollback_performed = False
                if rollback_conflicts:
                    return self._result(
                        status="concurrent_modification",
                        changed_paths=changed_paths,
                        candidate_digest=candidate_digest,
                        rollback_performed=rollback_performed,
                        blocked_reason=f"rollback_conflict:{rollback_conflicts[0]}",
                    )
                if isinstance(exc, _ConcurrentWorkspaceModification):
                    try:
                        concurrent_path = (
                            Path(str(exc)).relative_to(self.canonical_root).as_posix()
                        )
                    except ValueError:
                        concurrent_path = changed_paths[0]
                    return self._result(
                        status="concurrent_modification",
                        changed_paths=changed_paths,
                        candidate_digest=candidate_digest,
                        rollback_performed=rollback_performed,
                        blocked_reason=(
                            f"canonical_workspace_changed:{concurrent_path}"
                        ),
                    )
                return self._result(
                    status="infra_error",
                    changed_paths=changed_paths,
                    candidate_digest=candidate_digest,
                    rollback_performed=rollback_performed,
                    blocked_reason=(
                        f"promotion_failed:{str(exc)}"
                        if isinstance(exc, (ValueError, UnsafeRegularFileError))
                        else f"promotion_failed:{type(exc).__name__}"
                    ),
                )
            if not retain_commit_receipt:
                _remove_promotion_journal(self.canonical_root)

        self._promoted = True
        return self._result(
            status="release_candidate",
            changed_paths=changed_paths,
            candidate_digest=candidate_digest,
            verification_results=verification_results,
            promoted_digest=promoted_digest,
        )

    def changed_paths(self) -> tuple[str, ...]:
        self._ensure_open()
        return tuple(
            path
            for path in self._patch_paths
            if _path_fingerprint(self.root / path, root=self.root)
            != self._baseline_path_fingerprints.get(path)
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._temporary_directory.cleanup()

    def __enter__(self) -> CandidateWorkspace:
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def _projected_candidate_files(
        self, changed_paths: tuple[str, ...]
    ) -> dict[str, str]:
        projected = dict(self._baseline_files)
        for path in changed_paths:
            fingerprint = _path_fingerprint(self.root / path, root=self.root)
            if fingerprint is None:
                projected.pop(path, None)
            else:
                projected[path] = fingerprint
        return projected

    def _concurrency_error(self, changed_paths: tuple[str, ...]) -> str | None:
        observed_files = _workspace_file_map(self.canonical_root)
        if observed_files != self._baseline_files:
            differing_paths = sorted(
                path
                for path in set(observed_files).union(self._baseline_files)
                if observed_files.get(path) != self._baseline_files.get(path)
            )
            changed_path = differing_paths[0] if differing_paths else "workspace"
            return f"canonical_workspace_changed:{changed_path}"
        for path in changed_paths:
            expected = self._baseline_path_fingerprints.get(path)
            observed = _path_fingerprint(
                self.canonical_root / path,
                root=self.canonical_root,
            )
            if observed != expected:
                return f"canonical_workspace_changed:{path}"
        return None

    def _result(
        self,
        *,
        status: str,
        changed_paths: tuple[str, ...],
        candidate_digest: str,
        verification_results: tuple[WorkspaceVerificationResult, ...] = (),
        promoted_digest: str | None = None,
        rollback_performed: bool = False,
        blocked_reason: str | None = None,
    ) -> WorkspacePromotionResult:
        return WorkspacePromotionResult(
            status=status,
            changed_paths=changed_paths,
            verification_results=verification_results,
            baseline_digest=self.baseline_digest,
            candidate_digest=candidate_digest,
            promoted_digest=promoted_digest,
            rollback_performed=rollback_performed,
            blocked_reason=blocked_reason,
        )

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("candidate_workspace_closed")


def recover_interrupted_workspace_promotion(
    canonical_root: Path,
    *,
    retain_committed: bool = False,
) -> WorkspaceCommitReceipt | None:
    resolved_root = canonical_root.resolve()
    if not resolved_root.is_dir():
        raise ValueError("canonical_workspace_not_directory")
    with _workspace_promotion_lock(resolved_root):
        return _recover_interrupted_promotion(
            resolved_root,
            retain_committed=retain_committed,
        )


def acknowledge_committed_workspace_promotion(
    canonical_root: Path,
    *,
    candidate_digest: str,
) -> None:
    resolved_root = canonical_root.resolve()
    if not resolved_root.is_dir():
        raise ValueError("canonical_workspace_not_directory")
    with _workspace_promotion_lock(resolved_root):
        receipt = _recover_interrupted_promotion(
            resolved_root,
            retain_committed=True,
        )
        if receipt is None:
            raise RuntimeError("committed_promotion_receipt_missing")
        if receipt.candidate_digest != candidate_digest:
            raise RuntimeError("committed_promotion_receipt_mismatch")
        _remove_promotion_journal(resolved_root)


def _copy_ignore_for_root(source_root: Path):
    resolved_root = source_root.resolve()

    def ignore(directory: str, names: list[str]) -> set[str]:
        relative_directory = Path(directory).resolve().relative_to(resolved_root)
        return {
            name
            for name in names
            if ignored_repository_path((*relative_directory.parts, name))
        }

    return ignore


def _candidate_workspace_parent() -> Path:
    configured = os.environ.get("SOCIETY_CORE_CANDIDATE_ROOT")
    root = (
        Path(configured).expanduser()
        if configured
        else Path.home() / ".society_core" / "candidate-workspaces"
    )
    if not root.is_absolute() or root.is_symlink() or _is_junction(root):
        raise RuntimeError("unsafe_candidate_workspace_root")
    try:
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        metadata = root.lstat()
        if not stat.S_ISDIR(metadata.st_mode):
            raise RuntimeError("unsafe_candidate_workspace_root")
        if hasattr(os, "getuid") and metadata.st_uid != os.getuid():
            raise RuntimeError("unsafe_candidate_workspace_root")
        if os.name == "posix":
            root.chmod(0o700)
            if stat.S_IMODE(root.lstat().st_mode) != 0o700:
                raise RuntimeError("unsafe_candidate_workspace_root")
    except OSError as exc:
        raise RuntimeError("unsafe_candidate_workspace_root") from exc
    return root


def _clone_dependency_directories(
    *,
    source_root: Path,
    candidate_root: Path,
) -> None:
    for name in ("node_modules", ".venv"):
        source = source_root / name
        if source.is_symlink() or not source.is_dir():
            continue
        target = candidate_root / name
        try:
            _clone_directory(source, target)
            _confine_cloned_symlinks(
                source_root=source_root,
                candidate_root=candidate_root,
                cloned_root=target,
            )
        except (OSError, subprocess.SubprocessError, UnsafeRegularFileError):
            shutil.rmtree(target, ignore_errors=True)


def _clone_directory(source: Path, target: Path) -> None:
    copy_command: tuple[str, ...] | None = None
    if sys.platform == "darwin" and shutil.which("cp"):
        copy_command = ("cp", "-cR", str(source), str(target))
    elif shutil.which("cp"):
        copy_command = (
            "cp",
            "-a",
            "--reflink=auto",
            str(source),
            str(target),
        )
    if copy_command is not None:
        completed = subprocess.run(
            copy_command,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if completed.returncode == 0:
            return
        shutil.rmtree(target, ignore_errors=True)
    shutil.copytree(source, target, symlinks=True)


def _confine_cloned_symlinks(
    *,
    source_root: Path,
    candidate_root: Path,
    cloned_root: Path,
) -> None:
    for cloned_path in sorted(cloned_root.rglob("*")):
        if not cloned_path.is_symlink():
            continue
        source_path = source_root / cloned_path.relative_to(candidate_root)
        source_target = source_path.resolve(strict=True)
        try:
            source_relative = source_target.relative_to(source_root)
        except ValueError:
            raise UnsafeRegularFileError(
                "unsafe_regular_file:external_dependency_symlink"
            ) from None
        candidate_target = candidate_root / source_relative
        replacement = os.path.relpath(candidate_target, start=cloned_path.parent)
        cloned_path.unlink()
        cloned_path.symlink_to(replacement)


@contextmanager
def _workspace_promotion_lock(root: Path) -> Iterator[None]:
    lock_key = hashlib.sha256(str(root.resolve()).encode("utf-8")).hexdigest()
    with _THREAD_LOCK_GUARD:
        thread_lock = _THREAD_LOCKS.setdefault(lock_key, threading.Lock())
    lock_root = _private_runtime_directory(
        "society_core_workspace_locks",
        error_name="unsafe_workspace_lock_root",
    )
    lock_path = lock_root / f"{lock_key}.lock"
    with thread_lock:
        with _open_workspace_lock_file(lock_path) as handle:
            _lock_file(handle)
            try:
                yield
            finally:
                _unlock_file(handle)


def _private_runtime_directory(name: str, *, error_name: str) -> Path:
    root = Path(tempfile.gettempdir()) / name
    if root.is_symlink() or _is_junction(root):
        raise RuntimeError(error_name)
    try:
        root.mkdir(mode=0o700, parents=False, exist_ok=False)
    except FileExistsError:
        pass
    except OSError as exc:
        raise RuntimeError(error_name) from exc
    try:
        metadata = root.lstat()
        if not stat.S_ISDIR(metadata.st_mode):
            raise RuntimeError(error_name)
        if hasattr(os, "getuid") and metadata.st_uid != os.getuid():
            raise RuntimeError(error_name)
        if os.name == "posix":
            root.chmod(0o700)
            metadata = root.lstat()
            if not stat.S_ISDIR(metadata.st_mode):
                raise RuntimeError(error_name)
            if stat.S_IMODE(metadata.st_mode) != 0o700:
                raise RuntimeError(error_name)
    except OSError as exc:
        raise RuntimeError(error_name) from exc
    return root


def _is_junction(path: Path) -> bool:
    """Recognize Windows directory junctions, which ``is_symlink`` omits."""

    detector = getattr(path, "is_junction", None)
    return bool(detector is not None and detector())


def _open_workspace_lock_file(path: Path) -> IO[bytes]:
    if path.is_symlink():
        raise RuntimeError("unsafe_workspace_lock_file")
    flags = os.O_RDWR | os.O_CREAT | os.O_APPEND
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        raise RuntimeError("unsafe_workspace_lock_file") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise RuntimeError("unsafe_workspace_lock_file")
        if hasattr(os, "getuid") and metadata.st_uid != os.getuid():
            raise RuntimeError("unsafe_workspace_lock_file")
        if hasattr(os, "fchmod"):
            os.fchmod(descriptor, 0o600)
        return os.fdopen(descriptor, "a+b", closefd=True)
    except BaseException:
        os.close(descriptor)
        raise


def _lock_file(handle: IO[bytes]) -> None:
    if os.name == "posix":
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        return
    if os.name == "nt":  # pragma: no cover - Windows compatibility.
        import msvcrt

        handle.seek(0)
        if handle.read(1) == b"":
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        return
    raise RuntimeError("unsupported_file_lock_platform")


def _unlock_file(handle: IO[bytes]) -> None:
    if os.name == "posix":
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return
    if os.name == "nt":  # pragma: no cover - Windows compatibility.
        import msvcrt

        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


def _workspace_file_map(root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if ignored_repository_path(relative):
            continue
        try:
            metadata = path.lstat()
        except OSError:
            result[relative.as_posix()] = "unreadable"
            continue
        if stat.S_ISDIR(metadata.st_mode):
            continue
        result[relative.as_posix()] = _path_fingerprint(path, root=root) or "missing"
    return result


def _file_map_digest(files: dict[str, str]) -> str:
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _path_fingerprint(path: Path, *, root: Path | None = None) -> str | None:
    if path.is_symlink():
        payload = f"symlink:{os.readlink(path)}".encode("utf-8")
        mode = path.lstat().st_mode
    elif path.is_file():
        mode, digest = regular_file_fingerprint(root or path.parent, path)
        return f"{mode:o}:{digest}"
    elif path.exists():
        payload = b"other"
        mode = path.stat().st_mode
    else:
        return None
    digest = hashlib.sha256(payload).hexdigest()
    return f"{mode:o}:{digest}"


def _safe_relative_path(value: str) -> str:
    path = Path(value)
    if not value or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe_path:{value}")
    return path.as_posix()


def _snapshot_path(path: Path, *, root: Path) -> _FileSnapshot:
    if path.is_symlink():
        raise UnsafeRegularFileError("unsafe_regular_file:canonical_symlink")
    if path.is_file():
        return _FileSnapshot(
            existed=True,
            content=read_regular_file_bytes(root, path, max_bytes=16_000_000),
            mode=path.lstat().st_mode,
        )
    return _FileSnapshot(existed=False, content=b"", mode=None)


def _promote_path(
    *,
    source_root: Path,
    target_root: Path,
    relative_path: str,
    created_directories: list[Path],
    expected_target_fingerprint: str | None = None,
    staging_path: Path | None = None,
) -> None:
    source = source_root / relative_path
    target = target_root / relative_path
    if source.is_symlink():
        raise UnsafeRegularFileError("unsafe_regular_file:candidate_symlink")
    if not source.exists():
        if target.is_symlink():
            raise UnsafeRegularFileError("unsafe_regular_file:canonical_symlink")
        if target.exists():
            _conditional_remove_path(
                target,
                expected_fingerprint=expected_target_fingerprint,
                root=target_root,
                staging_path=staging_path,
            )
        return
    payload = read_regular_file_bytes(
        source_root,
        source,
        max_bytes=16_000_000,
    )
    _ensure_safe_parent_directories(
        root=target_root,
        parent=target.parent,
        created_directories=created_directories,
    )
    if target.is_symlink():
        raise UnsafeRegularFileError("unsafe_regular_file:canonical_symlink")
    _conditional_replace_bytes(
        target,
        payload,
        mode=source.lstat().st_mode,
        expected_fingerprint=expected_target_fingerprint,
        root=target_root,
        staging_path=staging_path,
    )


def _ensure_safe_parent_directories(
    *,
    root: Path,
    parent: Path,
    created_directories: list[Path],
) -> None:
    relative = parent.relative_to(root)
    cursor = root
    for component in relative.parts:
        cursor = cursor / component
        if cursor.is_symlink():
            raise UnsafeRegularFileError("unsafe_regular_file:symlink_parent")
        if cursor.exists():
            if not cursor.is_dir():
                raise UnsafeRegularFileError("unsafe_regular_file:non_directory_parent")
            continue
        cursor.mkdir(mode=0o700)
        created_directories.append(cursor)


def _atomic_replace_bytes(target: Path, content: bytes, *, mode: int) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.promotion-",
        dir=target.parent,
    )
    temporary = Path(temporary_name)
    try:
        mode_bits = stat.S_IMODE(mode)
        fchmod = getattr(os, "fchmod", None)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(content)
            handle.flush()
            if callable(fchmod):
                fchmod(handle.fileno(), mode_bits)
            os.fsync(handle.fileno())
        if not callable(fchmod):
            os.chmod(temporary, mode_bits)
        os.replace(temporary, target)
        _fsync_directory(target.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _conditional_replace_bytes(
    target: Path,
    content: bytes,
    *,
    mode: int,
    expected_fingerprint: str | None,
    root: Path,
    staging_path: Path | None = None,
) -> None:
    temporary = staging_path or (
        target.parent / f".{target.name}.promotion-cas-{uuid.uuid4().hex}"
    )
    if temporary.parent != target.parent:
        raise ValueError("invalid_promotion_staging_parent")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(temporary, flags, 0o600)
    mutation_possible = False
    try:
        mode_bits = stat.S_IMODE(mode)
        fchmod = getattr(os, "fchmod", None)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(content)
            handle.flush()
            if callable(fchmod):
                fchmod(handle.fileno(), mode_bits)
            os.fsync(handle.fileno())
        if not callable(fchmod):
            os.chmod(temporary, mode_bits)
        candidate_fingerprint = _path_fingerprint(temporary, root=target.parent)
        observed = _path_fingerprint(target, root=root)
        if observed != expected_fingerprint:
            raise _ConcurrentWorkspaceModification(str(target))
        if expected_fingerprint is None:
            mutation_possible = True
            try:
                os.link(temporary, target)
            except FileExistsError as exc:
                raise _ConcurrentWorkspaceModification(str(target)) from exc
            temporary.unlink()
            mutation_possible = False
            _fsync_directory(target.parent)
            return
        mutation_possible = True
        try:
            _atomic_exchange_paths(temporary, target)
        except FileNotFoundError as exc:
            raise _ConcurrentWorkspaceModification(str(target)) from exc
        displaced_fingerprint = _path_fingerprint(temporary, root=target.parent)
        if displaced_fingerprint != expected_fingerprint:
            if _path_fingerprint(target, root=root) == candidate_fingerprint:
                _atomic_exchange_paths(temporary, target)
                mutation_possible = False
                temporary.unlink()
            raise _ConcurrentWorkspaceModification(str(target))
        temporary.unlink()
        mutation_possible = False
        _fsync_directory(target.parent)
    except BaseException:
        if not mutation_possible:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
        raise


def _conditional_remove_path(
    target: Path,
    *,
    expected_fingerprint: str | None,
    root: Path,
    staging_path: Path | None = None,
) -> None:
    observed = _path_fingerprint(target, root=root)
    if observed != expected_fingerprint:
        raise _ConcurrentWorkspaceModification(str(target))
    if expected_fingerprint is None:
        return
    quarantine = staging_path or (
        target.parent / f".{target.name}.promotion-cas-{uuid.uuid4().hex}"
    )
    if quarantine.parent != target.parent:
        raise ValueError("invalid_promotion_staging_parent")
    if quarantine.exists() or quarantine.is_symlink():
        raise RuntimeError("promotion_staging_path_exists")
    mutation_possible = True
    try:
        try:
            os.rename(target, quarantine)
        except FileNotFoundError as exc:
            raise _ConcurrentWorkspaceModification(str(target)) from exc
        displaced_fingerprint = _path_fingerprint(quarantine, root=target.parent)
        if displaced_fingerprint != expected_fingerprint:
            try:
                os.link(quarantine, target)
            except FileExistsError:
                pass
            else:
                quarantine.unlink()
                mutation_possible = False
            raise _ConcurrentWorkspaceModification(str(target))
        quarantine.unlink()
        mutation_possible = False
        _fsync_directory(target.parent)
    except BaseException:
        if not mutation_possible:
            try:
                quarantine.unlink()
            except FileNotFoundError:
                pass
        raise


def _atomic_exchange_paths(first: Path, second: Path) -> None:
    if os.name == "nt":  # pragma: no cover - exercised on Windows CI.
        _windows_atomic_exchange_paths(first, second)
        return
    library = ctypes.CDLL(None, use_errno=True)
    first_bytes = os.fsencode(first)
    second_bytes = os.fsencode(second)
    if sys.platform == "darwin":
        rename = getattr(library, "renamex_np", None)
        if rename is None:
            raise OSError(errno.ENOTSUP, "atomic_path_exchange_unavailable")
        rename.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint)
        rename.restype = ctypes.c_int
        result = rename(first_bytes, second_bytes, 0x00000002)
    elif sys.platform.startswith("linux"):
        rename = getattr(library, "renameat2", None)
        if rename is None:
            raise OSError(errno.ENOTSUP, "atomic_path_exchange_unavailable")
        rename.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        rename.restype = ctypes.c_int
        result = rename(-100, first_bytes, -100, second_bytes, 0x00000002)
    else:
        raise OSError(errno.ENOTSUP, "atomic_path_exchange_unavailable")
    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number))
    _fsync_directory(first.parent)


def _windows_exchange_backup_path(staging_path: Path) -> Path:
    """A short, deterministic crash-recovery path for ``ReplaceFileW``."""
    name_hash = hashlib.sha256(staging_path.name.encode("utf-8")).hexdigest()[:24]
    return staging_path.with_name(f".promotion-win-{name_hash}.bak")


def _assert_safe_windows_exchange_file(path: Path, *, error_name: str) -> None:
    """Reject path aliases that could make an exchange mutate outside state."""
    is_junction = getattr(path, "is_junction", None)
    if path.is_symlink() or (callable(is_junction) and is_junction()):
        raise RuntimeError(error_name)
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise RuntimeError(error_name) from exc
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise RuntimeError(error_name)


def _windows_atomic_exchange_paths(first: Path, second: Path) -> None:
    """Atomically put ``first`` at ``second`` and retain displaced bytes.

    Windows has no rename-exchange primitive. ``ReplaceFileW`` is the native
    atomic replacement API and can capture the displaced target into a backup
    in the same operation. The backup name is derivable from the journaled
    staging path, so recovery can finish the bookkeeping if the process dies
    after the kernel replacement but before the backup is renamed to ``first``.
    """
    if first.parent != second.parent:
        raise ValueError("atomic_path_exchange_requires_same_parent")
    _assert_safe_windows_exchange_file(
        first, error_name="unsafe_promotion_exchange_replacement"
    )
    _assert_safe_windows_exchange_file(
        second, error_name="unsafe_promotion_exchange_target"
    )
    backup = _windows_exchange_backup_path(first)
    is_backup_junction = getattr(backup, "is_junction", None)
    if (
        backup.exists()
        or backup.is_symlink()
        or (callable(is_backup_junction) and is_backup_junction())
    ):
        raise RuntimeError("promotion_exchange_backup_exists")

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    replace_file = kernel32.ReplaceFileW
    replace_file.argtypes = (
        ctypes.c_wchar_p,
        ctypes.c_wchar_p,
        ctypes.c_wchar_p,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_void_p,
    )
    replace_file.restype = ctypes.c_int
    replace_file_write_through = 0x00000001
    replaced = replace_file(
        str(second),
        str(first),
        str(backup),
        replace_file_write_through,
        None,
        None,
    )
    if not replaced:
        error_number = ctypes.get_last_error()
        message = ctypes.FormatError(error_number).strip()
        if error_number in {2, 3}:
            raise FileNotFoundError(error_number, message, str(second))
        raise OSError(error_number, message, str(second))

    # ReplaceFileW removed ``first`` and atomically captured the old ``second``
    # at ``backup``. Windows os.rename is no-replace, so an unexpected actor
    # planting ``first`` cannot be silently overwritten here.
    _assert_safe_windows_exchange_file(
        backup, error_name="unsafe_promotion_exchange_backup"
    )
    os.rename(backup, first)
    _assert_safe_windows_exchange_file(
        first, error_name="unsafe_promotion_exchange_replacement"
    )
    _fsync_directory(first.parent)


def _fsync_directory(directory: Path) -> None:
    if os.name != "posix":  # pragma: no cover - Windows compatibility.
        return
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _restore_paths(
    *,
    root: Path,
    snapshots: dict[str, _FileSnapshot],
    created_directories: Iterable[Path],
    expected_current: dict[str, str | None] | None = None,
    staging_paths: dict[str, Path] | None = None,
) -> tuple[tuple[str, ...], int]:
    conflicts = list(
        _reconcile_promotion_staging(
            root=root,
            snapshots=snapshots,
            promoted_fingerprints=expected_current or {},
            staging_paths=staging_paths or {},
        )
    )
    restored_count = 0
    for relative_path, snapshot in reversed(tuple(snapshots.items())):
        if relative_path in conflicts:
            continue
        target = root / relative_path
        if expected_current is not None:
            observed = _path_fingerprint(target, root=root)
            baseline_fingerprint = _snapshot_fingerprint(snapshot)
            if observed == baseline_fingerprint:
                continue
            if observed != expected_current.get(relative_path):
                conflicts.append(relative_path)
                continue
        if snapshot.existed:
            _ensure_safe_parent_directories(
                root=root,
                parent=target.parent,
                created_directories=[],
            )
            try:
                if expected_current is None:
                    _atomic_replace_bytes(
                        target,
                        snapshot.content,
                        mode=snapshot.mode or 0o600,
                    )
                else:
                    _conditional_replace_bytes(
                        target,
                        snapshot.content,
                        mode=snapshot.mode or 0o600,
                        expected_fingerprint=expected_current.get(relative_path),
                        root=root,
                        staging_path=(staging_paths or {}).get(relative_path),
                    )
            except _ConcurrentWorkspaceModification:
                conflicts.append(relative_path)
                continue
            restored_count += 1
        elif target.exists() or target.is_symlink():
            try:
                if expected_current is None:
                    target.unlink()
                else:
                    _conditional_remove_path(
                        target,
                        expected_fingerprint=expected_current.get(relative_path),
                        root=root,
                        staging_path=(staging_paths or {}).get(relative_path),
                    )
            except _ConcurrentWorkspaceModification:
                conflicts.append(relative_path)
                continue
            restored_count += 1
    for directory in created_directories:
        try:
            directory.rmdir()
        except OSError:
            pass
    return tuple(conflicts), restored_count


def _reconcile_promotion_staging(
    *,
    root: Path,
    snapshots: dict[str, _FileSnapshot],
    promoted_fingerprints: dict[str, str | None],
    staging_paths: dict[str, Path],
) -> tuple[str, ...]:
    conflicts: list[str] = []
    for relative_path, staging_path in staging_paths.items():
        if _recover_windows_exchange_backup(
            root=root,
            relative_path=relative_path,
            staging_path=staging_path,
        ):
            conflicts.append(relative_path)
            continue
        if not staging_path.exists() and not staging_path.is_symlink():
            continue
        target = root / relative_path
        snapshot = snapshots[relative_path]
        baseline_fingerprint = _snapshot_fingerprint(snapshot)
        promoted_fingerprint = promoted_fingerprints.get(relative_path)
        if staging_path.is_symlink() or not staging_path.is_file():
            staging_path.unlink()
            conflicts.append(relative_path)
            continue
        staging_fingerprint = _path_fingerprint(staging_path, root=root)
        target_fingerprint = _path_fingerprint(target, root=root)
        known_staging = staging_fingerprint in {
            baseline_fingerprint,
            promoted_fingerprint,
        }
        if target_fingerprint == promoted_fingerprint and not known_staging:
            _restore_displaced_staging(staging_path=staging_path, target=target)
            conflicts.append(relative_path)
            continue
        if target_fingerprint == baseline_fingerprint and not known_staging:
            _restore_displaced_staging(staging_path=staging_path, target=target)
            conflicts.append(relative_path)
            continue
        if (
            target_fingerprint not in {baseline_fingerprint, promoted_fingerprint}
            and not known_staging
        ):
            _archive_staging_conflict(
                root=root,
                relative_path=relative_path,
                staging_path=staging_path,
            )
            conflicts.append(relative_path)
            continue
        staging_path.unlink()
        _fsync_directory(staging_path.parent)
    return tuple(conflicts)


def _recover_windows_exchange_backup(
    *,
    root: Path,
    relative_path: str,
    staging_path: Path,
) -> bool:
    """Finish ``ReplaceFileW`` bookkeeping after a process interruption.

    Returns True only for an ambiguous/hostile state that recovery must report
    as a conflict. A lone backup is the expected crash window: it is the target
    displaced atomically by the kernel and becomes the journaled staging file.
    """
    if os.name != "nt":
        return False
    backup = _windows_exchange_backup_path(staging_path)
    is_backup_junction = getattr(backup, "is_junction", None)
    if (
        not backup.exists()
        and not backup.is_symlink()
        and not (callable(is_backup_junction) and is_backup_junction())
    ):
        return False
    _assert_safe_windows_exchange_file(
        backup, error_name="unsafe_promotion_exchange_backup"
    )
    is_staging_junction = getattr(staging_path, "is_junction", None)
    if (
        staging_path.exists()
        or staging_path.is_symlink()
        or (callable(is_staging_junction) and is_staging_junction())
    ):
        _assert_safe_windows_exchange_file(
            staging_path, error_name="unsafe_promotion_staging_path"
        )
        _archive_staging_conflict(
            root=root,
            relative_path=relative_path,
            staging_path=backup,
        )
        _archive_staging_conflict(
            root=root,
            relative_path=relative_path,
            staging_path=staging_path,
        )
        return True
    os.rename(backup, staging_path)
    _assert_safe_windows_exchange_file(
        staging_path, error_name="unsafe_promotion_staging_path"
    )
    _fsync_directory(staging_path.parent)
    return False


def _restore_displaced_staging(*, staging_path: Path, target: Path) -> None:
    if os.name == "nt":
        _assert_safe_windows_exchange_file(
            staging_path, error_name="unsafe_promotion_staging_path"
        )
    if target.exists() or target.is_symlink():
        if os.name == "nt":
            _assert_safe_windows_exchange_file(
                target, error_name="unsafe_promotion_exchange_target"
            )
        _atomic_exchange_paths(staging_path, target)
        staging_path.unlink()
    else:
        os.link(staging_path, target)
        staging_path.unlink()
    _fsync_directory(target.parent)


def _archive_staging_conflict(
    *,
    root: Path,
    relative_path: str,
    staging_path: Path,
) -> None:
    conflict_root = _private_runtime_directory(
        "society_core_workspace_conflicts",
        error_name="unsafe_workspace_conflict_root",
    )
    root_hash = hashlib.sha256(str(root.resolve()).encode("utf-8")).hexdigest()[:16]
    path_hash = hashlib.sha256(relative_path.encode("utf-8")).hexdigest()[:16]
    destination = conflict_root / f"{root_hash}-{path_hash}-{uuid.uuid4().hex}.bin"
    payload = read_regular_file_bytes(
        staging_path.parent,
        staging_path,
        max_bytes=_MAX_PROMOTION_SNAPSHOT_BYTES,
    )
    _atomic_replace_bytes(destination, payload, mode=staging_path.lstat().st_mode)
    staging_path.unlink()
    _fsync_directory(staging_path.parent)


def _snapshot_fingerprint(snapshot: _FileSnapshot) -> str | None:
    if not snapshot.existed:
        return None
    digest = hashlib.sha256(snapshot.content).hexdigest()
    return f"{(snapshot.mode or 0o600):o}:{digest}"


def _promotion_journal_path(root: Path) -> Path:
    root_hash = hashlib.sha256(str(root.resolve()).encode("utf-8")).hexdigest()
    journal_root = _private_runtime_directory(
        "society_core_workspace_transactions",
        error_name="unsafe_promotion_journal_root",
    )
    path = journal_root / f"{root_hash}.json"
    if path.exists() or path.is_symlink():
        try:
            metadata = path.lstat()
        except OSError as exc:
            raise RuntimeError("unsafe_promotion_journal") from exc
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or (hasattr(os, "getuid") and metadata.st_uid != os.getuid())
        ):
            raise RuntimeError("unsafe_promotion_journal")
    return path


def _missing_parent_directories(
    root: Path,
    changed_paths: tuple[str, ...],
) -> tuple[str, ...]:
    missing: list[str] = []
    for relative_path in changed_paths:
        parent = Path(relative_path).parent
        parts = () if parent == Path(".") else parent.parts
        cursor = root
        for part in parts:
            cursor = cursor / part
            if not cursor.exists() and not cursor.is_symlink():
                value = cursor.relative_to(root).as_posix()
                if value not in missing:
                    missing.append(value)
    return tuple(missing)


def _promotion_staging_paths(
    changed_paths: tuple[str, ...],
) -> dict[str, str]:
    return {
        relative_path: (
            Path(relative_path).parent
            / (f".{Path(relative_path).name}.promotion-cas-{uuid.uuid4().hex}")
        ).as_posix()
        for relative_path in changed_paths
    }


def _write_promotion_journal(
    *,
    root: Path,
    state: str,
    baseline_digest: str,
    candidate_digest: str,
    snapshots: dict[str, _FileSnapshot],
    missing_parent_directories: tuple[str, ...],
    promoted_fingerprints: dict[str, str | None],
    staging_paths: dict[str, str],
) -> None:
    if state not in {"prepared", "canonical_written", "committed"}:
        raise ValueError("invalid_promotion_journal_state")
    body = {
        "schema_version": 3,
        "canonical_root": str(root.resolve()),
        "state": state,
        "baseline_digest": baseline_digest,
        "candidate_digest": candidate_digest,
        "snapshots": [
            {
                "path": path,
                "existed": snapshot.existed,
                "content_base64": base64.b64encode(snapshot.content).decode("ascii"),
                "mode": snapshot.mode,
            }
            for path, snapshot in snapshots.items()
        ],
        "missing_parent_directories": missing_parent_directories,
        "promoted_fingerprints": promoted_fingerprints,
        "staging_paths": staging_paths,
    }
    payload = dict(body)
    payload["integrity_hash"] = _journal_integrity_hash(body)
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    path = _promotion_journal_path(root)
    _atomic_replace_bytes(path, encoded, mode=0o600)


def _remove_promotion_journal(root: Path) -> None:
    path = _promotion_journal_path(root)
    try:
        path.unlink()
    except FileNotFoundError:
        return
    _fsync_directory(path.parent)


def _recover_interrupted_promotion(
    root: Path,
    *,
    retain_committed: bool = False,
) -> WorkspaceCommitReceipt | None:
    journal_path = _promotion_journal_path(root)
    if not journal_path.exists() and not journal_path.is_symlink():
        return None
    if journal_path.is_symlink() or not journal_path.is_file():
        raise RuntimeError("unsafe_promotion_journal")
    raw = read_regular_file_bytes(
        journal_path.parent,
        journal_path,
        max_bytes=128_000_000,
    )
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("invalid_promotion_journal") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("invalid_promotion_journal")
    integrity_hash = str(payload.pop("integrity_hash", ""))
    if not integrity_hash or integrity_hash != _journal_integrity_hash(payload):
        raise RuntimeError("promotion_journal_integrity_failure")
    schema_version = payload.get("schema_version")
    if schema_version not in {2, 3}:
        raise RuntimeError("unsupported_promotion_journal_schema")
    if str(payload.get("canonical_root")) != str(root.resolve()):
        raise RuntimeError("promotion_journal_root_mismatch")
    state = str(payload.get("state") or "")
    if state == "committed":
        if _file_map_digest(_workspace_file_map(root)) != str(
            payload.get("candidate_digest") or ""
        ):
            raise RuntimeError("promotion_recovery_committed_digest_mismatch")
        try:
            changed_paths = tuple(
                _safe_relative_path(str(item["path"]))
                for item in payload.get("snapshots", ())
                if isinstance(item, dict)
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("invalid_promotion_journal") from exc
        receipt = WorkspaceCommitReceipt(
            canonical_root=str(root.resolve()),
            baseline_digest=str(payload.get("baseline_digest") or ""),
            candidate_digest=str(payload.get("candidate_digest") or ""),
            changed_paths=changed_paths,
        )
        if not retain_committed:
            _remove_promotion_journal(root)
        return receipt
    if state not in {"prepared", "canonical_written"}:
        raise RuntimeError("invalid_promotion_journal_state")
    snapshots: dict[str, _FileSnapshot] = {}
    try:
        for item in payload.get("snapshots", ()):
            if not isinstance(item, dict):
                raise ValueError("snapshot_not_object")
            path = _safe_relative_path(str(item["path"]))
            snapshots[path] = _FileSnapshot(
                existed=bool(item["existed"]),
                content=base64.b64decode(
                    str(item.get("content_base64") or ""),
                    validate=True,
                ),
                mode=int(item["mode"]) if item.get("mode") is not None else None,
            )
        missing_directories = tuple(
            root / _safe_relative_path(str(item))
            for item in payload.get("missing_parent_directories", ())
        )
        raw_fingerprints = payload.get("promoted_fingerprints")
        if not isinstance(raw_fingerprints, dict):
            raise ValueError("promoted_fingerprints_not_object")
        promoted_fingerprints = {
            _safe_relative_path(str(path)): (
                str(fingerprint) if fingerprint is not None else None
            )
            for path, fingerprint in raw_fingerprints.items()
        }
        if set(promoted_fingerprints) != set(snapshots):
            raise ValueError("promoted_fingerprint_paths_mismatch")
        raw_staging_paths = payload.get("staging_paths", {})
        if not isinstance(raw_staging_paths, dict):
            raise ValueError("staging_paths_not_object")
        staging_paths = {
            _safe_relative_path(str(path)): root
            / _validated_staging_relative_path(
                target_path=_safe_relative_path(str(path)),
                staging_path=_safe_relative_path(str(staging_path)),
            )
            for path, staging_path in raw_staging_paths.items()
        }
        if schema_version == 3 and set(staging_paths) != set(snapshots):
            raise ValueError("staging_path_targets_mismatch")
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("invalid_promotion_journal") from exc
    conflicts, _ = _restore_paths(
        root=root,
        snapshots=snapshots,
        created_directories=sorted(
            missing_directories,
            key=lambda path: len(path.parts),
            reverse=True,
        ),
        expected_current=promoted_fingerprints,
        staging_paths=staging_paths,
    )
    if conflicts:
        _remove_promotion_journal(root)
        raise RuntimeError(f"promotion_recovery_conflict:{conflicts[0]}")
    for relative_path, snapshot in snapshots.items():
        if _path_fingerprint(root / relative_path, root=root) != _snapshot_fingerprint(
            snapshot
        ):
            raise RuntimeError(
                f"promotion_recovery_owned_path_mismatch:{relative_path}"
            )
    _remove_promotion_journal(root)
    return None


def _validated_staging_relative_path(
    *,
    target_path: str,
    staging_path: str,
) -> str:
    target = Path(target_path)
    staging = Path(staging_path)
    expected_prefix = f".{target.name}.promotion-cas-"
    if staging.parent != target.parent or not staging.name.startswith(expected_prefix):
        raise ValueError("invalid_promotion_staging_path")
    return staging.as_posix()


def _journal_integrity_hash(payload: dict[str, object]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
