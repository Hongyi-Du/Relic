"""Guarded workspace patching and verification for real project updates."""

from __future__ import annotations

import json
import os
import shlex
import stat
import sys
import tempfile
import tomllib
from dataclasses import dataclass, replace
from email.parser import Parser
from pathlib import Path

from .execution import CommandExecutor
from .hashing import stable_hash
from .repository_paths import ignored_repository_path
from .safe_files import (
    UnsafeRegularFileError,
    read_regular_file_bytes,
    read_regular_file_text,
)
from .verification_policy import VERIFICATION_EXECUTABLES


_MAX_WORKSPACE_FILE_BYTES = 16_000_000


@dataclass(frozen=True)
class WorkspacePackageInfo:
    name: str
    version: str
    manifest_hash: str


@dataclass(frozen=True)
class WorkspaceFilePatch:
    patch_id: str
    path: str
    operation: str
    content: str = ""
    old: str = ""
    new: str = ""
    rationale: str = ""


@dataclass(frozen=True)
class WorkspacePatchResult:
    patch_id: str
    path: str
    operation: str
    status: str
    before_hash: str | None
    after_hash: str | None
    blocked_reason: str | None = None


@dataclass(frozen=True)
class WorkspaceVerificationResult:
    command: str
    status: str
    exit_code: int | None
    elapsed_sec: float
    stdout_tail: str = ""
    stderr_tail: str = ""
    blocked_reason: str | None = None
    backend: str = ""
    stdout_hash: str | None = None
    stderr_hash: str | None = None
    runtime_ref: str | None = None


@dataclass(frozen=True)
class _WorkspaceFileSnapshot:
    existed: bool
    content: bytes
    mode: int | None


def load_workspace_package_info(root: Path) -> WorkspacePackageInfo:
    resolved_root = root.resolve()
    package_path = resolved_root / "package.json"
    if package_path.exists():
        raw = read_regular_file_text(
            resolved_root,
            package_path,
            max_bytes=2_000_000,
        )
        package = json.loads(raw)
        return WorkspacePackageInfo(
            name=str(package.get("name", "")),
            version=str(package.get("version", "")),
            manifest_hash=stable_hash(json.loads(raw)),
        )

    pyproject_path = resolved_root / "pyproject.toml"
    if pyproject_path.exists():
        raw_bytes = read_regular_file_bytes(
            resolved_root,
            pyproject_path,
            max_bytes=2_000_000,
        )
        pyproject = tomllib.loads(raw_bytes.decode("utf-8"))
        project = pyproject.get("project", {}) if isinstance(pyproject, dict) else {}
        return WorkspacePackageInfo(
            name=str(project.get("name", "")),
            version=str(project.get("version", "")),
            manifest_hash=stable_hash(pyproject),
        )

    metadata_path = _find_python_wheel_metadata(resolved_root)
    if metadata_path is not None:
        try:
            raw = read_regular_file_bytes(
                resolved_root,
                metadata_path,
                max_bytes=2_000_000,
            ).decode("utf-8", errors="replace")
        except UnsafeRegularFileError:
            raw = ""
        metadata = Parser().parsestr(raw)
        return WorkspacePackageInfo(
            name=str(metadata.get("Name", "")),
            version=str(metadata.get("Version", "")),
            manifest_hash=stable_hash(
                {"metadata_path": metadata_path.name, "metadata": raw}
            ),
        )

    raise FileNotFoundError(
        f"Missing package.json, pyproject.toml, or dist-info METADATA under {root}"
    )


def _find_python_wheel_metadata(root: Path) -> Path | None:
    metadata_paths = sorted(root.glob("*.dist-info/METADATA"))
    if not metadata_paths:
        return None
    return metadata_paths[0]


def validate_workspace_package(
    root: Path,
    *,
    expected_name: str,
    expected_version: str,
) -> WorkspacePackageInfo:
    info = load_workspace_package_info(root)
    if info.name != expected_name or info.version != expected_version:
        raise ValueError(
            f"Expected {expected_name}@{expected_version}, "
            f"got {info.name}@{info.version}"
        )
    return info


def apply_workspace_patches(
    root: Path,
    patches: tuple[WorkspaceFilePatch, ...],
    *,
    dry_run: bool = False,
    gate_policy: str = "claim",
    max_patches: int = 64,
    max_total_bytes: int = 500_000,
) -> tuple[WorkspacePatchResult, ...]:
    resolved_root = root.resolve()
    if len(patches) > max_patches:
        return (
            WorkspacePatchResult(
                patch_id="patch_set",
                path="",
                operation="",
                status="blocked",
                before_hash=None,
                after_hash=None,
                blocked_reason="too_many_patches",
            ),
        )
    total_bytes = sum(
        len(patch.content.encode("utf-8"))
        + len(patch.old.encode("utf-8"))
        + len(patch.new.encode("utf-8"))
        for patch in patches
    )
    if total_bytes > max_total_bytes:
        return (
            WorkspacePatchResult(
                patch_id="patch_set",
                path="",
                operation="",
                status="blocked",
                before_hash=None,
                after_hash=None,
                blocked_reason="patch_set_too_large",
            ),
        )
    results: list[WorkspacePatchResult] = []
    snapshots: dict[Path, _WorkspaceFileSnapshot] = {}
    for patch in patches:
        snapshot_error = _snapshot_target_before_patch(
            resolved_root,
            patch,
            snapshots=snapshots,
        )
        if snapshot_error is not None:
            result = WorkspacePatchResult(
                patch_id=patch.patch_id,
                path=patch.path,
                operation=patch.operation,
                status="blocked",
                before_hash=None,
                after_hash=None,
                blocked_reason=snapshot_error,
            )
            results.append(result)
            if gate_policy != "explore" and not dry_run:
                _restore_patch_snapshots(resolved_root, snapshots)
                results = _mark_applied_results_rolled_back(
                    results,
                    blocked_patch_id=result.patch_id,
                )
            break
        result = _apply_single_patch(
            root=resolved_root,
            patch=patch,
            dry_run=dry_run,
            gate_policy=gate_policy,
        )
        results.append(result)
        if results[-1].status == "blocked":
            if gate_policy != "explore" and not dry_run:
                _restore_patch_snapshots(resolved_root, snapshots)
                results = _mark_applied_results_rolled_back(
                    results,
                    blocked_patch_id=result.patch_id,
                )
            break
    return tuple(results)


def _snapshot_target_before_patch(
    root: Path,
    patch: WorkspaceFilePatch,
    *,
    snapshots: dict[Path, _WorkspaceFileSnapshot],
) -> str | None:
    try:
        target = _safe_target(root, patch.path)
    except ValueError:
        return None
    if target in snapshots:
        return None
    if target.exists():
        try:
            content = read_regular_file_bytes(
                root,
                target,
                max_bytes=_MAX_WORKSPACE_FILE_BYTES,
            )
        except (UnsafeRegularFileError, OSError) as exc:
            return _workspace_file_error(exc, patch.path)
        snapshots[target] = _WorkspaceFileSnapshot(
            existed=True,
            content=content,
            mode=target.lstat().st_mode,
        )
    else:
        snapshots[target] = _WorkspaceFileSnapshot(
            existed=False,
            content=b"",
            mode=None,
        )
    return None


def _restore_patch_snapshots(
    root: Path,
    snapshots: dict[Path, _WorkspaceFileSnapshot],
) -> None:
    for target, snapshot in reversed(tuple(snapshots.items())):
        if snapshot.existed:
            _atomic_write_workspace_bytes(
                root=root,
                target=target,
                content=snapshot.content,
                mode=snapshot.mode or 0o600,
            )
        elif target.exists() or target.is_symlink():
            target.unlink()


def _mark_applied_results_rolled_back(
    results: list[WorkspacePatchResult],
    *,
    blocked_patch_id: str,
) -> list[WorkspacePatchResult]:
    rolled_back: list[WorkspacePatchResult] = []
    for result in results:
        if result.status == "applied":
            rolled_back.append(
                replace(
                    result,
                    status="rolled_back",
                    after_hash=result.before_hash,
                    blocked_reason=f"rolled_back_after_blocked_patch:{blocked_patch_id}",
                )
            )
        else:
            rolled_back.append(result)
    return rolled_back


def run_workspace_verification(
    root: Path,
    commands: tuple[str, ...],
    *,
    timeout_seconds: float = 60.0,
    max_commands: int = 32,
    allowed_executables: tuple[str, ...] = VERIFICATION_EXECUTABLES,
    executor: CommandExecutor | None = None,
) -> tuple[WorkspaceVerificationResult, ...]:
    resolved_root = root.resolve()
    unique_commands = tuple(dict.fromkeys(commands))
    if not unique_commands:
        return (
            _blocked_verification_result(
                command="verification_command_set",
                reason="verification_commands_required",
            ),
        )
    if len(unique_commands) > max_commands:
        return (
            _blocked_verification_result(
                command="verification_command_set",
                reason=(
                    f"verification_command_limit_exceeded:"
                    f"{len(unique_commands)}>{max_commands}"
                ),
            ),
        )
    if executor is None:
        return (
            _blocked_verification_result(
                command="verification_command_set",
                reason="verification_executor_required",
            ),
        )
    results: list[WorkspaceVerificationResult] = []
    for command in unique_commands:
        results.append(
            _run_single_command(
                root=resolved_root,
                command=command,
                timeout_seconds=timeout_seconds,
                allowed_executables=allowed_executables,
                executor=executor,
            )
        )
    return tuple(results)


def _blocked_verification_result(
    *,
    command: str,
    reason: str,
) -> WorkspaceVerificationResult:
    return WorkspaceVerificationResult(
        command=command,
        status="blocked",
        exit_code=None,
        elapsed_sec=0.0,
        blocked_reason=reason,
    )


def workspace_update_mode(
    patch_results: tuple[WorkspacePatchResult, ...],
    verification_results: tuple[WorkspaceVerificationResult, ...],
) -> str:
    if any(result.status == "blocked" for result in patch_results):
        return "blocked"
    if patch_results and all(result.status == "no_op" for result in patch_results):
        return "no_effect"
    if verification_results and all(
        result.status == "passed" for result in verification_results
    ):
        return "verified"
    if verification_results:
        return "verification_failed"
    return "applied"


def file_hash(path: Path, *, root: Path | None = None) -> str | None:
    if not path.exists() and not path.is_symlink():
        return None
    payload = read_regular_file_text(
        root or path.parent,
        path,
        max_bytes=_MAX_WORKSPACE_FILE_BYTES,
    )
    return stable_hash(payload)


def _apply_single_patch(
    *,
    root: Path,
    patch: WorkspaceFilePatch,
    dry_run: bool,
    gate_policy: str,
) -> WorkspacePatchResult:
    try:
        target = _safe_target(root, patch.path)
    except ValueError as exc:
        return WorkspacePatchResult(
            patch_id=patch.patch_id,
            path=patch.path,
            operation=patch.operation,
            status="blocked",
            before_hash=None,
            after_hash=None,
            blocked_reason=str(exc),
        )
    try:
        before_hash = file_hash(target, root=root)
        next_content = _next_content(
            root,
            target,
            patch,
            gate_policy=gate_policy,
        )
        if gate_policy != "explore":
            _validate_next_content(target, next_content)
    except UnsafeRegularFileError as exc:
        return WorkspacePatchResult(
            patch_id=patch.patch_id,
            path=patch.path,
            operation=patch.operation,
            status="blocked",
            before_hash=None,
            after_hash=None,
            blocked_reason=_workspace_file_error(exc, patch.path),
        )
    except UnicodeDecodeError as exc:
        return WorkspacePatchResult(
            patch_id=patch.patch_id,
            path=patch.path,
            operation=patch.operation,
            status="blocked",
            before_hash=None,
            after_hash=None,
            blocked_reason=_workspace_file_error(exc, patch.path),
        )
    except OSError as exc:
        return WorkspacePatchResult(
            patch_id=patch.patch_id,
            path=patch.path,
            operation=patch.operation,
            status="blocked",
            before_hash=None,
            after_hash=None,
            blocked_reason=_workspace_file_error(exc, patch.path),
        )
    except ValueError as exc:
        return WorkspacePatchResult(
            patch_id=patch.patch_id,
            path=patch.path,
            operation=patch.operation,
            status="blocked",
            before_hash=before_hash,
            after_hash=before_hash,
            blocked_reason=str(exc),
        )
    after_hash = stable_hash(next_content)
    if before_hash == after_hash:
        return WorkspacePatchResult(
            patch_id=patch.patch_id,
            path=patch.path,
            operation=patch.operation,
            status="no_op",
            before_hash=before_hash,
            after_hash=after_hash,
        )
    if not dry_run:
        _atomic_write_workspace_bytes(
            root=root,
            target=target,
            content=next_content.encode("utf-8"),
            mode=target.lstat().st_mode if target.exists() else 0o644,
        )
    return WorkspacePatchResult(
        patch_id=patch.patch_id,
        path=patch.path,
        operation=patch.operation,
        status="dry_run" if dry_run else "applied",
        before_hash=before_hash,
        after_hash=after_hash,
    )


def _next_content(
    root: Path,
    target: Path,
    patch: WorkspaceFilePatch,
    *,
    gate_policy: str,
) -> str:
    operation = patch.operation
    if operation == "create_or_replace":
        return patch.content
    existing = (
        read_regular_file_text(
            root,
            target,
            max_bytes=_MAX_WORKSPACE_FILE_BYTES,
        )
        if target.exists()
        else ""
    )
    if operation == "replace_fragment":
        if not patch.old:
            raise ValueError("empty_replace_fragment")
        if patch.old not in existing:
            if patch.new and patch.new in existing:
                return existing
            raise ValueError(f"replace_fragment_not_found:{patch.path}")
        if gate_policy != "explore":
            _validate_replace_fragment_shape(patch)
        return existing.replace(patch.old, patch.new, 1)
    if operation == "append_if_missing":
        if patch.content in existing:
            return existing
        separator = "" if not existing or existing.endswith("\n") else "\n"
        return existing + separator + patch.content
    raise ValueError(f"unsupported_operation:{operation}")


def _validate_replace_fragment_shape(patch: WorkspaceFilePatch) -> None:
    old_stripped = patch.old.strip()
    new_stripped = patch.new.strip()
    old_lines = [line for line in patch.old.splitlines() if line.strip()]
    new_lines = [line for line in patch.new.splitlines() if line.strip()]
    if old_stripped and not new_stripped:
        raise ValueError(f"destructive_replace_fragment:{patch.path}:empty_new")
    if len(old_lines) >= 8 and len(new_lines) < max(2, int(len(old_lines) * 0.5)):
        raise ValueError(
            f"destructive_replace_fragment:{patch.path}:"
            f"old_lines={len(old_lines)}:new_lines={len(new_lines)}"
        )


def _validate_next_content(target: Path, content: str) -> None:
    if target.suffix.lower() != ".json":
        return
    try:
        json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"invalid_json:{target.name}:{exc.lineno}:{exc.colno}"
        ) from None


def _safe_target(root: Path, relative_path: str) -> Path:
    relative = Path(relative_path)
    if (
        not relative_path
        or relative.is_absolute()
        or ".." in relative.parts
        or not relative.parts
    ):
        raise ValueError(f"unsafe_path:{relative_path}")
    if ignored_repository_path(relative):
        raise ValueError(f"unsafe_path:reserved_component:{relative_path}")
    resolved_root = root.resolve()
    target = resolved_root / relative
    cursor = resolved_root
    for index, component in enumerate(relative.parts):
        cursor = cursor / component
        if cursor.is_symlink():
            raise ValueError(f"unsafe_path:symlink_component:{relative_path}")
        if cursor.exists():
            is_final = index == len(relative.parts) - 1
            if not is_final and not cursor.is_dir():
                raise ValueError(f"unsafe_path:non_directory_parent:{relative_path}")
            if is_final and not cursor.is_file():
                raise ValueError(f"unsafe_path:not_regular:{relative_path}")
    return target


def _workspace_file_error(error: BaseException, path: str) -> str:
    if isinstance(error, UnicodeDecodeError):
        return f"workspace_file_not_utf8:{path}"
    if isinstance(error, UnsafeRegularFileError):
        return str(error)
    return f"workspace_file_access_failed:{type(error).__name__}:{path}"


def _atomic_write_workspace_bytes(
    *,
    root: Path,
    target: Path,
    content: bytes,
    mode: int,
) -> None:
    relative_path = target.relative_to(root.resolve()).as_posix()
    checked_target = _safe_target(root, relative_path)
    checked_target.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
    checked_target = _safe_target(root, relative_path)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{checked_target.name}.workspace-patch-",
        dir=checked_target.parent,
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
            # Windows has no os.fchmod. The mkstemp path is still private and
            # in the target directory, so apply its mode before replacement.
            os.chmod(temporary, mode_bits)
        os.replace(temporary, checked_target)
        _fsync_directory(checked_target.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _fsync_directory(directory: Path) -> None:
    if os.name != "posix":  # pragma: no cover - Windows compatibility.
        return
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _run_single_command(
    *,
    root: Path,
    command: str,
    timeout_seconds: float,
    allowed_executables: tuple[str, ...],
    executor: CommandExecutor,
) -> WorkspaceVerificationResult:
    shell_operator = _unsupported_shell_operator(command)
    if shell_operator is not None:
        return WorkspaceVerificationResult(
            command=command,
            status="blocked",
            exit_code=None,
            elapsed_sec=0.0,
            blocked_reason=f"shell_syntax_not_supported:{shell_operator}",
        )
    try:
        argv = shlex.split(command)
    except ValueError as exc:
        return WorkspaceVerificationResult(
            command=command,
            status="blocked",
            exit_code=None,
            elapsed_sec=0.0,
            blocked_reason=f"invalid_command:{type(exc).__name__}",
        )
    if not argv:
        return WorkspaceVerificationResult(
            command=command,
            status="blocked",
            exit_code=None,
            elapsed_sec=0.0,
            blocked_reason="empty_command",
        )
    executable = Path(argv[0]).name
    if executable not in allowed_executables and argv[0] != sys.executable:
        return WorkspaceVerificationResult(
            command=command,
            status="blocked",
            exit_code=None,
            elapsed_sec=0.0,
            blocked_reason=f"unsupported_executable:{executable}",
        )
    outcome = executor.run(
        root=root,
        argv=tuple(argv),
        timeout_seconds=timeout_seconds,
    )
    return WorkspaceVerificationResult(
        command=command,
        status=outcome.status,
        exit_code=outcome.exit_code,
        elapsed_sec=outcome.elapsed_sec,
        stdout_tail=outcome.stdout_tail,
        stderr_tail=outcome.stderr_tail,
        blocked_reason=outcome.blocked_reason,
        backend=outcome.backend,
        stdout_hash=outcome.stdout_hash,
        stderr_hash=outcome.stderr_hash,
        runtime_ref=outcome.runtime_ref,
    )


def _unsupported_shell_operator(command: str) -> str | None:
    try:
        lexer = shlex.shlex(
            command,
            posix=True,
            punctuation_chars="|&;<>()",
        )
        lexer.whitespace_split = True
        lexer.commenters = ""
        tokens = tuple(lexer)
    except ValueError:
        return None
    operators = {"|", "||", "&", "&&", ";", "<", "<<", ">", ">>", "(", ")"}
    return next((token for token in tokens if token in operators), None)


def _tail_text(value: bytes | str, *, max_chars: int = 2000) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    return value[-max_chars:]
