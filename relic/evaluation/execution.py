"""Explicit command-execution boundary for workspace verification.

Controller secrets and model credentials belong to the parent process. Commands
that inspect or test a repository receive a minimal environment and, for
untrusted repositories, must run in a restricted container backend.
"""

from __future__ import annotations

import hashlib
import math
import os
import platform
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from time import monotonic
from typing import Literal, Protocol

from .verification_policy import VERIFICATION_EXECUTABLES

ExecutionStatus = Literal["passed", "failed", "blocked", "timeout", "infra_error"]
ExecutionBackend = Literal["local", "docker", "apptainer"]


_IMAGE_WORKSPACE_STAGE_SCRIPT = r"""
set -euo pipefail
candidate="$1"
workspace="$2"
shift 2
test -d "$candidate"
test -d "$workspace/.git"
test -n "$(find "$candidate" -type f -print -quit)"
cd "$workspace"
mkdir -p .relic-runtime-tmp
export TMPDIR="$workspace/.relic-runtime-tmp"
export GOTMPDIR="$TMPDIR"
while IFS= read -r -d '' relative; do
    if [ ! -e "$candidate/$relative" ] && [ ! -L "$candidate/$relative" ]; then
        rm -rf -- "$relative"
    fi
done < <(git ls-files -z)
while IFS= read -r -d '' source; do
    relative="${source#"$candidate"/}"
    target="$workspace/$relative"
    if [ ! -f "$target" ] || ! cmp -s -- "$source" "$target"; then
        mkdir -p -- "$(dirname "$target")"
        cp -f -- "$source" "$target"
    fi
done < <(find "$candidate" -type f -print0)
exec "$@"
""".strip()


@dataclass(frozen=True)
class ExecutionPolicy:
    trust_level: Literal["trusted", "untrusted"] = "untrusted"
    backend: ExecutionBackend = "docker"
    container_image: str | None = None
    container_platform: str | None = None
    # CooperBench task images can install a benchmark runner as ENTRYPOINT.
    # This opt-in lets the verified repository command run directly instead.
    clear_container_entrypoint: bool = False
    docker_host: str | None = None
    network_enabled: bool = False
    memory_limit_mb: int = 2048
    cpu_limit: float = 2.0
    pids_limit: int = 128
    max_open_files: int = 1024
    max_output_bytes: int = 10_000_000
    output_tail_bytes: int = 16_384
    min_free_disk_bytes: int = 268_435_456
    environment_allowlist: tuple[str, ...] = (
        "PATH",
        "LANG",
        "LC_ALL",
        "TERM",
        "SYSTEMROOT",
        "WINDIR",
    )

    def __post_init__(self) -> None:
        if self.trust_level not in {"trusted", "untrusted"}:
            raise ValueError("unsupported_trust_level")
        if self.backend not in {"local", "docker", "apptainer"}:
            raise ValueError("unsupported_execution_backend")
        if self.backend in {"docker", "apptainer"}:
            if not self.container_image:
                raise ValueError("container_image_required")
            if (
                self.backend == "apptainer"
                and "://" in self.container_image
                and not self.container_image.startswith("docker://")
            ):
                raise ValueError("apptainer_container_scheme_unsupported")
            immutable_registry_digest = re.search(r"@sha256:[0-9a-fA-F]{64}$", self.container_image)
            immutable_local_image_id = re.fullmatch(r"sha256:[0-9a-fA-F]{64}", self.container_image)
            immutable_image = bool(immutable_registry_digest) or bool(
                self.backend == "docker" and immutable_local_image_id
            )
            if self.trust_level == "untrusted" and not immutable_image:
                raise ValueError("container_image_must_be_digest_pinned")
            if self.backend == "docker" and self.docker_host is not None:
                _validate_local_docker_host(self.docker_host)
            if self.backend == "apptainer" and self.docker_host is not None:
                raise ValueError("docker_host_requires_docker_backend")
            if self.backend != "docker" and self.clear_container_entrypoint:
                raise ValueError("clear_container_entrypoint_requires_docker_backend")
            if self.container_platform is not None and not re.fullmatch(
                r"[a-z0-9][a-z0-9._-]*/[a-z0-9][a-z0-9._-]*"
                r"(?:/[a-z0-9][a-z0-9._-]*)?",
                self.container_platform,
            ):
                raise ValueError("invalid_container_platform")
        else:
            if self.docker_host is not None:
                raise ValueError("docker_host_requires_docker_backend")
            if self.container_platform is not None:
                raise ValueError("container_platform_requires_docker_backend")
            if self.clear_container_entrypoint:
                raise ValueError("clear_container_entrypoint_requires_docker_backend")
        if self.memory_limit_mb <= 0:
            raise ValueError("memory_limit_mb_must_be_positive")
        if self.cpu_limit <= 0:
            raise ValueError("cpu_limit_must_be_positive")
        if self.pids_limit <= 0:
            raise ValueError("pids_limit_must_be_positive")
        if self.max_open_files <= 0:
            raise ValueError("max_open_files_must_be_positive")
        if self.max_output_bytes <= 0:
            raise ValueError("max_output_bytes_must_be_positive")
        if self.output_tail_bytes <= 0:
            raise ValueError("output_tail_bytes_must_be_positive")
        if self.output_tail_bytes > self.max_output_bytes:
            raise ValueError("output_tail_bytes_exceeds_output_limit")
        if self.min_free_disk_bytes <= 0:
            raise ValueError("min_free_disk_bytes_must_be_positive")


@dataclass(frozen=True)
class CommandOutcome:
    status: ExecutionStatus
    exit_code: int | None
    elapsed_sec: float
    stdout_tail: str = ""
    stderr_tail: str = ""
    blocked_reason: str | None = None
    backend: str = ""
    stdout_hash: str | None = None
    stderr_hash: str | None = None
    runtime_ref: str | None = None


class CommandExecutor(Protocol):
    policy: ExecutionPolicy

    def run(
        self,
        *,
        root: Path,
        argv: tuple[str, ...],
        timeout_seconds: float,
    ) -> CommandOutcome: ...


class LocalCommandExecutor:
    """Hardened process runner for explicitly trusted workspaces only."""

    def __init__(self, *, policy: ExecutionPolicy) -> None:
        if policy.backend != "local":
            raise ValueError("local_executor_requires_local_policy")
        self.policy = policy

    def run(
        self,
        *,
        root: Path,
        argv: tuple[str, ...],
        timeout_seconds: float,
    ) -> CommandOutcome:
        if self.policy.trust_level != "trusted":
            return CommandOutcome(
                status="blocked",
                exit_code=None,
                elapsed_sec=0.0,
                blocked_reason="untrusted_local_execution_forbidden",
                backend="local",
            )
        with tempfile.TemporaryDirectory(prefix="relic_exec_home_") as home:
            environment = _sanitized_environment(self.policy, home=Path(home))
            return _run_process(
                argv=argv,
                cwd=root.resolve(),
                environment=environment,
                timeout_seconds=timeout_seconds,
                policy=self.policy,
                backend="local",
            )

    def run_with_read_only_evaluator(
        self,
        *,
        root: Path,
        evaluator_root: Path,
        argv: tuple[str, ...],
        timeout_seconds: float,
    ) -> CommandOutcome:
        # Local execution is exploratory-only and never passes the formal
        # container gate. Keeping assets outside the workspace still prevents
        # candidate-tree hooks from replacing the evaluator bundle.
        _validate_evaluator_asset_root(root, evaluator_root)
        return self.run(
            root=root,
            argv=argv,
            timeout_seconds=timeout_seconds,
        )


class DockerCommandExecutor:
    """Restricted container runner for untrusted repository commands."""

    def __init__(self, *, policy: ExecutionPolicy, docker_binary: str = "docker") -> None:
        if policy.backend != "docker":
            raise ValueError("docker_executor_requires_docker_policy")
        self.docker_binary = docker_binary
        docker_host = policy.docker_host
        if docker_host is None and docker_binary == "docker":
            docker_host = _discover_local_docker_host(docker_binary)
        self.policy = (
            replace(policy, docker_host=docker_host) if docker_host is not None else policy
        )

    def build_command(
        self,
        *,
        root: Path,
        argv: tuple[str, ...],
        cidfile: Path | None = None,
        container_name: str | None = None,
        read_only_mounts: tuple[tuple[Path, str], ...] = (),
        workspace_read_only: bool = False,
        image_workspace: str | None = None,
    ) -> tuple[str, ...]:
        assert self.policy.container_image is not None
        if image_workspace is not None:
            _validate_docker_container_path(image_workspace)
            if (
                image_workspace == "/candidate"
                or image_workspace.startswith("/candidate/")
                or "/candidate".startswith(image_workspace + "/")
            ):
                raise ValueError("image_workspace_conflicts_with_candidate_mount")
        network = "bridge" if self.policy.network_enabled else "none"
        workspace_root = image_workspace or "/workspace"
        command_argv = argv
        if image_workspace is not None:
            command_argv = (
                "/bin/bash",
                "-ceu",
                _IMAGE_WORKSPACE_STAGE_SCRIPT,
                "relic-image-workspace",
                "/candidate",
                image_workspace,
                *argv,
            )
        return (
            self.docker_binary,
            "run",
            "--rm",
            "--pull",
            "never",
            *(
                ("--platform", self.policy.container_platform)
                if self.policy.container_platform
                else ()
            ),
            *(("--cidfile", str(cidfile)) if cidfile is not None else ()),
            *(("--name", container_name) if container_name is not None else ()),
            "--network",
            network,
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--user",
            f"{os.getuid()}:{os.getgid()}" if hasattr(os, "getuid") else "65534:65534",
            "--pids-limit",
            str(self.policy.pids_limit),
            "--ulimit",
            f"nofile={self.policy.max_open_files}:{self.policy.max_open_files}",
            "--memory",
            f"{self.policy.memory_limit_mb}m",
            "--cpus",
            f"{self.policy.cpu_limit:g}",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=256m",
            "--workdir",
            workspace_root,
            "--env",
            "HOME=/tmp/home",
            "--env",
            "PYTHONNOUSERSITE=1",
            "--env",
            f"PYTHONPATH={workspace_root}:{workspace_root}/src",
            *(
                (
                    "--mount",
                    f"type=volume,target={image_workspace}",
                    "-v",
                    f"{root.resolve()}:/candidate:ro",
                )
                if image_workspace is not None
                else (
                    "-v",
                    f"{root.resolve()}:/workspace:{'ro' if workspace_read_only else 'rw'}",
                )
            ),
            *(
                token
                for host_root, container_root in read_only_mounts
                for token in ("-v", f"{host_root.resolve()}:{container_root}:ro")
            ),
            *( ("--entrypoint", "") if self.policy.clear_container_entrypoint else () ),
            self.policy.container_image,
            *command_argv,
        )

    def run(
        self,
        *,
        root: Path,
        argv: tuple[str, ...],
        timeout_seconds: float,
    ) -> CommandOutcome:
        return self._run_with_mounts(
            root=root,
            argv=argv,
            timeout_seconds=timeout_seconds,
            read_only_mounts=(),
        )

    def run_in_image_workspace(
        self,
        *,
        root: Path,
        argv: tuple[str, ...],
        timeout_seconds: float,
        image_workspace: str,
        read_only_mounts: tuple[tuple[Path, str], ...] = (),
    ) -> CommandOutcome:
        """Stage candidate bytes into an image-provided repository workspace."""

        if not root.is_dir() or root.is_symlink():
            raise ValueError("invalid_image_workspace_candidate")
        _validate_docker_container_path(image_workspace)
        for host_root, container_root in read_only_mounts:
            if not host_root.is_dir() or host_root.is_symlink():
                raise ValueError("invalid_read_only_mount_source")
            _validate_docker_container_path(container_root)
            if (
                container_root == "/candidate"
                or container_root.startswith("/candidate/")
                or container_root == image_workspace
                or container_root.startswith(image_workspace + "/")
            ):
                raise ValueError("read_only_mount_shadows_image_workspace")
        return self._run_with_mounts(
            root=root,
            argv=argv,
            timeout_seconds=timeout_seconds,
            read_only_mounts=read_only_mounts,
            image_workspace=image_workspace,
        )

    def run_with_read_only_evaluator(
        self,
        *,
        root: Path,
        evaluator_root: Path,
        argv: tuple[str, ...],
        timeout_seconds: float,
    ) -> CommandOutcome:
        _validate_evaluator_asset_root(root, evaluator_root)
        return self._run_with_mounts(
            root=root,
            argv=argv,
            timeout_seconds=timeout_seconds,
            read_only_mounts=((evaluator_root, "/evaluator"),),
            workspace_read_only=True,
        )

    def _run_with_mounts(
        self,
        *,
        root: Path,
        argv: tuple[str, ...],
        timeout_seconds: float,
        read_only_mounts: tuple[tuple[Path, str], ...],
        workspace_read_only: bool = False,
        image_workspace: str | None = None,
    ) -> CommandOutcome:
        if shutil.which(self.docker_binary) is None:
            return CommandOutcome(
                status="blocked",
                exit_code=None,
                elapsed_sec=0.0,
                blocked_reason="docker_executable_not_found",
                backend="docker",
            )
        with tempfile.TemporaryDirectory(prefix="relic_docker_home_") as home:
            home_path = Path(home)
            cidfile = home_path / "container.cid"
            container_name = f"relic-{uuid.uuid4().hex}"
            environment = _sanitized_environment(self.policy, home=home_path)
            termination_cleanup: CommandOutcome | None = None

            def cleanup_terminated_container() -> None:
                nonlocal termination_cleanup
                cleanup_identifier = _read_container_id(cidfile) or container_name
                termination_cleanup = self._remove_container(
                    identifier=cleanup_identifier,
                    cwd=root.resolve(),
                    environment=environment,
                )

            outcome = _run_process(
                argv=self.build_command(
                    root=root,
                    argv=argv,
                    cidfile=cidfile,
                    container_name=container_name,
                    read_only_mounts=read_only_mounts,
                    workspace_read_only=workspace_read_only,
                    image_workspace=image_workspace,
                ),
                cwd=root.resolve(),
                environment=environment,
                timeout_seconds=timeout_seconds,
                policy=self.policy,
                backend="docker",
                termination_callback=cleanup_terminated_container,
            )
            outcome = replace(outcome, runtime_ref=container_name)
            outcome = _classify_docker_infrastructure_failure(
                outcome,
                container_started=bool(_read_container_id(cidfile)),
            )
            if outcome.blocked_reason in {
                "docker_daemon_unavailable",
                "docker_image_unavailable",
            }:
                return outcome
            cleanup_required = outcome.status in {"timeout", "infra_error"} or (
                outcome.status == "blocked"
                and outcome.blocked_reason in {"output_limit_exceeded", "disk_reserve_exhausted"}
            )
            if not cleanup_required:
                return outcome
            cleanup = termination_cleanup
            if cleanup is None:
                cleanup_identifier = _read_container_id(cidfile) or container_name
                cleanup = self._remove_container(
                    identifier=cleanup_identifier,
                    cwd=root.resolve(),
                    environment=environment,
                )
            if cleanup.status == "passed":
                return outcome
            reason = cleanup.blocked_reason or cleanup.stderr_tail or cleanup.status
            return replace(
                outcome,
                status="infra_error",
                blocked_reason=f"container_cleanup_failed:{reason}",
            )

    def _remove_container(
        self,
        *,
        identifier: str,
        cwd: Path,
        environment: dict[str, str],
    ) -> CommandOutcome:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", identifier):
            return CommandOutcome(
                status="infra_error",
                exit_code=None,
                elapsed_sec=0.0,
                blocked_reason="invalid_container_identifier",
                backend="docker_cleanup",
            )
        last: CommandOutcome | None = None
        for _ in range(3):
            last = _run_process(
                argv=(self.docker_binary, "rm", "-f", identifier),
                cwd=cwd,
                environment=environment,
                timeout_seconds=10.0,
                policy=self.policy,
                backend="docker_cleanup",
                enforce_disk_reserve=False,
            )
            if last.status == "passed" or "No such container" in last.stderr_tail:
                return replace(last, status="passed", blocked_reason=None)
            time.sleep(0.1)
        assert last is not None
        return last


class ApptainerCommandExecutor:
    """Restricted Apptainer runner backed by a Slurm allocation.

    Apptainer supplies filesystem, namespace, environment, and network
    isolation. Slurm supplies CPU and memory cgroups because unprivileged
    Apptainer cgroups are not consistently enabled on shared clusters.
    """

    def __init__(
        self,
        *,
        policy: ExecutionPolicy,
        apptainer_binary: str = "apptainer",
        slurm_binary: str | None = None,
    ) -> None:
        if policy.backend != "apptainer":
            raise ValueError("apptainer_executor_requires_apptainer_policy")
        self.policy = policy
        self.apptainer_binary = shutil.which(apptainer_binary) or apptainer_binary
        self.slurm_binary = slurm_binary or _discover_slurm_step_binary()

    def build_command(
        self,
        *,
        root: Path,
        argv: tuple[str, ...],
        read_only_mounts: tuple[tuple[Path, str], ...] = (),
        workspace_read_only: bool = False,
    ) -> tuple[str, ...]:
        assert self.policy.container_image is not None
        image = self.policy.container_image
        if not image.startswith("docker://"):
            image = f"docker://{image}"
        container_command = (
            self.apptainer_binary,
            "exec",
            "--containall",
            "--cleanenv",
            "--no-home",
            "--writable-tmpfs",
            *(() if self.policy.network_enabled else ("--net", "--network", "none")),
            "--bind",
            (f"{root.resolve()}:/workspace:{'ro' if workspace_read_only else 'rw'}"),
            *(
                token
                for host_root, container_root in read_only_mounts
                for token in (
                    "--bind",
                    f"{host_root.resolve()}:{container_root}:ro",
                )
            ),
            "--pwd",
            "/workspace",
            "--env",
            "PYTHONNOUSERSITE=1",
            "--env",
            "PYTHONPATH=/workspace:/workspace/src",
            image,
            "/bin/bash",
            "-c",
            ('set -eu; ulimit -n "$1"; ulimit -u "$2"; shift 2; exec "$@"'),
            "relic-resource-limits",
            str(self.policy.max_open_files),
            str(self.policy.pids_limit),
            *argv,
        )
        if self.slurm_binary is None:
            return container_command
        return (
            self.slurm_binary,
            "--exclusive",
            "--nodes=1",
            "--ntasks=1",
            f"--cpus-per-task={max(1, math.ceil(self.policy.cpu_limit))}",
            f"--mem={self.policy.memory_limit_mb}M",
            "--cpu-bind=cores",
            "--kill-on-bad-exit=1",
            "--wait=0",
            "--export=ALL",
            *container_command,
        )

    def run(
        self,
        *,
        root: Path,
        argv: tuple[str, ...],
        timeout_seconds: float,
    ) -> CommandOutcome:
        return self._run_with_mounts(
            root=root,
            argv=argv,
            timeout_seconds=timeout_seconds,
            read_only_mounts=(),
        )

    def run_with_read_only_evaluator(
        self,
        *,
        root: Path,
        evaluator_root: Path,
        argv: tuple[str, ...],
        timeout_seconds: float,
    ) -> CommandOutcome:
        _validate_evaluator_asset_root(root, evaluator_root)
        return self._run_with_mounts(
            root=root,
            argv=argv,
            timeout_seconds=timeout_seconds,
            read_only_mounts=((evaluator_root, "/evaluator"),),
            workspace_read_only=True,
        )

    def _run_with_mounts(
        self,
        *,
        root: Path,
        argv: tuple[str, ...],
        timeout_seconds: float,
        read_only_mounts: tuple[tuple[Path, str], ...],
        workspace_read_only: bool = False,
    ) -> CommandOutcome:
        if shutil.which(self.apptainer_binary) is None:
            return CommandOutcome(
                status="blocked",
                exit_code=None,
                elapsed_sec=0.0,
                blocked_reason="apptainer_executable_not_found",
                backend="apptainer",
            )
        if self.policy.trust_level == "untrusted" and self.slurm_binary is None:
            return CommandOutcome(
                status="blocked",
                exit_code=None,
                elapsed_sec=0.0,
                blocked_reason="apptainer_untrusted_requires_slurm_allocation",
                backend="apptainer",
            )
        if self.slurm_binary is not None and shutil.which(self.slurm_binary) is None:
            return CommandOutcome(
                status="blocked",
                exit_code=None,
                elapsed_sec=0.0,
                blocked_reason="slurm_step_executable_not_found",
                backend="apptainer",
            )
        expected_platform = self.policy.container_platform
        actual_platform = _native_container_platform()
        if expected_platform is not None and expected_platform != actual_platform:
            return CommandOutcome(
                status="blocked",
                exit_code=None,
                elapsed_sec=0.0,
                blocked_reason=(
                    "apptainer_platform_mismatch:"
                    f"expected={expected_platform}:actual={actual_platform}"
                ),
                backend="apptainer",
            )
        with tempfile.TemporaryDirectory(prefix="relic_apptainer_home_") as home:
            environment = _sanitized_environment(self.policy, home=Path(home))
            outcome = _run_process(
                argv=self.build_command(
                    root=root,
                    argv=argv,
                    read_only_mounts=read_only_mounts,
                    workspace_read_only=workspace_read_only,
                ),
                cwd=root.resolve(),
                environment=environment,
                timeout_seconds=timeout_seconds,
                policy=self.policy,
                backend="apptainer",
            )
        return replace(
            _classify_apptainer_infrastructure_failure(outcome),
            runtime_ref=self.policy.container_image,
        )


def _validate_evaluator_asset_root(root: Path, evaluator_root: Path) -> None:
    workspace = root.resolve()
    evaluator = evaluator_root.resolve()
    if not evaluator.is_dir():
        raise ValueError("evaluator_asset_root_missing")
    if evaluator == workspace or evaluator.is_relative_to(workspace):
        raise ValueError("evaluator_assets_must_be_outside_workspace")


def _validate_docker_container_path(container_root: str) -> None:
    if (
        not re.fullmatch(r"/[A-Za-z0-9_./-]+", container_root)
        or ".." in PurePosixPath(container_root).parts
        or container_root == "/"
    ):
        raise ValueError("invalid_docker_container_path")


def _read_container_id(cidfile: Path) -> str | None:
    try:
        identifier = cidfile.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not re.fullmatch(r"[a-fA-F0-9]{12,64}", identifier):
        return None
    return identifier


def build_command_executor(policy: ExecutionPolicy) -> CommandExecutor:
    if policy.backend == "local":
        return LocalCommandExecutor(policy=policy)
    if policy.backend == "docker":
        return DockerCommandExecutor(policy=policy)
    return ApptainerCommandExecutor(policy=policy)


def ensure_command_executor_ready(
    executor: CommandExecutor,
    *,
    root: Path,
    timeout_seconds: float = 30.0,
) -> CommandOutcome:
    """Prove that the executor sees the intended writable workspace."""

    resolved_root = root.resolve()
    probe_id = uuid.uuid4().hex
    token = uuid.uuid4().hex
    input_name = f".execution-probe-{probe_id}.in"
    output_name = f".execution-probe-{probe_id}.out"
    input_path = resolved_root / input_name
    output_path = resolved_root / output_name
    try:
        try:
            descriptor = os.open(
                input_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(token)
                handle.flush()
                os.fsync(handle.fileno())
        except OSError as exc:
            raise RuntimeError(
                f"execution_workspace_probe_setup_failed:{type(exc).__name__}"
            ) from exc

        script = (
            'set -eu; observed="$(cat -- "$1")"; '
            'test "$observed" = "$3"; umask 077; set -C; '
            'printf "%s" "$3" > "$2"'
        )
        outcome = executor.run(
            root=resolved_root,
            argv=(
                "/bin/sh",
                "-c",
                script,
                "execution-probe",
                input_name,
                output_name,
                token,
            ),
            timeout_seconds=timeout_seconds,
        )
        if outcome.status != "passed":
            reason = outcome.blocked_reason or (
                f"{outcome.backend or 'execution'}_{outcome.status}"
                + (f"_exit_{outcome.exit_code}" if outcome.exit_code is not None else "")
            )
            raise RuntimeError(f"execution_backend_preflight_failed:{reason}")
        try:
            observed = output_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise RuntimeError("execution_workspace_roundtrip_failed:output_missing") from exc
        if observed != token:
            raise RuntimeError("execution_workspace_roundtrip_failed:token_mismatch")
        return outcome
    finally:
        input_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)


def probe_command_executor_executables(
    executor: CommandExecutor,
    *,
    root: Path,
    candidates: tuple[str, ...] = VERIFICATION_EXECUTABLES,
    timeout_seconds: float = 30.0,
) -> tuple[str, ...]:
    """Observe verification runtimes inside the configured execution boundary."""

    if not candidates or any(
        not re.fullmatch(r"[A-Za-z0-9_.+-]+", candidate) for candidate in candidates
    ):
        raise ValueError("invalid_execution_capability_candidate")
    script = (
        "for executable in "
        + " ".join(candidates)
        + '; do if command -v "$executable" >/dev/null 2>&1; then '
        + "printf '%s\\n' \"$executable\"; fi; done"
    )
    outcome = executor.run(
        root=root,
        argv=("/bin/sh", "-lc", script),
        timeout_seconds=timeout_seconds,
    )
    if outcome.status != "passed":
        reason = outcome.blocked_reason or (
            f"{outcome.backend or 'execution'}_{outcome.status}"
            + (f"_exit_{outcome.exit_code}" if outcome.exit_code is not None else "")
        )
        raise RuntimeError(f"execution_capability_probe_failed:{reason}")
    observed = set(outcome.stdout_tail.splitlines())
    return tuple(candidate for candidate in candidates if candidate in observed)


def _classify_docker_infrastructure_failure(
    outcome: CommandOutcome,
    *,
    container_started: bool = False,
) -> CommandOutcome:
    if outcome.status != "failed":
        return outcome
    # Once a cidfile proves that Docker started the container, daemon-looking
    # text is candidate output rather than an infrastructure diagnosis.
    if container_started:
        return outcome
    combined = "\n".join(
        part for part in (outcome.stdout_tail, outcome.stderr_tail) if part
    ).casefold()
    modern_windows_daemon_error = (
        "failed to connect to the docker api at npipe:" in combined
        and "dockerdesktoplinuxengine" in combined
        and ("daemon is running" in combined or "system cannot find the file specified" in combined)
    )
    if modern_windows_daemon_error or any(
        marker in combined
        for marker in (
            "cannot connect to the docker daemon",
            "is the docker daemon running",
            "error during connect",
            "docker daemon is not running",
        )
    ):
        return replace(
            outcome,
            status="infra_error",
            blocked_reason="docker_daemon_unavailable",
        )
    if any(
        marker in combined
        for marker in (
            "unable to find image",
            "pull access denied",
            "manifest unknown",
            "no such image",
        )
    ):
        return replace(
            outcome,
            status="infra_error",
            blocked_reason="docker_image_unavailable",
        )
    return outcome


def _classify_apptainer_infrastructure_failure(
    outcome: CommandOutcome,
) -> CommandOutcome:
    if outcome.status != "failed":
        return outcome
    combined = "\n".join(
        part for part in (outcome.stdout_tail, outcome.stderr_tail) if part
    ).casefold()
    if any(
        marker in combined
        for marker in (
            "failed to get checksum for",
            "while fetching image",
            "while performing build",
            "manifest unknown",
            "unable to retrieve auth token",
        )
    ):
        return replace(
            outcome,
            status="infra_error",
            blocked_reason="apptainer_image_unavailable",
        )
    if any(
        marker in combined
        for marker in (
            "unable to create user namespace",
            "network requires root",
            "while applying cgroups config",
        )
    ):
        return replace(
            outcome,
            status="infra_error",
            blocked_reason="apptainer_isolation_unavailable",
        )
    return outcome


def production_execution_policy(
    *,
    backend: ExecutionBackend,
    container_image: str | None,
    allow_trusted_local: bool,
    container_platform: str | None = None,
    memory_limit_mb: int = 4096,
    cpu_limit: float = 4.0,
    pids_limit: int = 256,
    max_open_files: int = 1024,
    min_free_disk_bytes: int = 2_147_483_648,
) -> ExecutionPolicy:
    """Build the fail-closed policy shared by production entry points."""

    if backend == "local" and not allow_trusted_local:
        raise ValueError(
            "local backend requires allow_trusted_local=True; use Docker for untrusted repositories"
        )
    if backend == "docker" and not container_image:
        raise ValueError("Docker backend requires a digest-pinned container_image")
    if backend == "apptainer" and not container_image:
        raise ValueError("Apptainer backend requires a digest-pinned container_image")
    docker_host = _discover_local_docker_host("docker") if backend == "docker" else None
    return ExecutionPolicy(
        trust_level="trusted" if backend == "local" else "untrusted",
        backend=backend,
        container_image=container_image,
        container_platform=container_platform,
        docker_host=docker_host,
        network_enabled=False,
        memory_limit_mb=memory_limit_mb,
        cpu_limit=cpu_limit,
        pids_limit=pids_limit,
        max_open_files=max_open_files,
        min_free_disk_bytes=min_free_disk_bytes,
    )


def trusted_local_executor() -> LocalCommandExecutor:
    """Compatibility backend for controller-owned fixtures and repositories."""

    return LocalCommandExecutor(policy=ExecutionPolicy(trust_level="trusted", backend="local"))


def _sanitized_environment(policy: ExecutionPolicy, *, home: Path) -> dict[str, str]:
    environment = {
        key: os.environ[key] for key in policy.environment_allowlist if key in os.environ
    }
    environment.update(
        {
            "HOME": str(home),
            "TMPDIR": str(home),
            "PYTHONNOUSERSITE": "1",
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        }
    )
    if policy.backend == "docker" and policy.docker_host is not None:
        environment["DOCKER_HOST"] = policy.docker_host
    if policy.backend == "apptainer":
        for key in ("APPTAINER_CACHEDIR", "APPTAINER_TMPDIR"):
            if key in os.environ:
                environment[key] = os.environ[key]
        environment.update(
            {key: value for key, value in os.environ.items() if key.startswith("SLURM_")}
        )
    return environment


def _discover_slurm_step_binary() -> str | None:
    job_id = os.environ.get("SLURM_JOB_ID", "")
    if not re.fullmatch(r"[0-9]+", job_id):
        return None
    return shutil.which("srun")


def _native_container_platform() -> str:
    machine = platform.machine().casefold()
    architectures = {
        "aarch64": "arm64",
        "arm64": "arm64",
        "x86_64": "amd64",
        "amd64": "amd64",
    }
    architecture = architectures.get(machine, machine or "unknown")
    system = platform.system().casefold() or "unknown"
    return f"{system}/{architecture}"


def _discover_local_docker_host(docker_binary: str) -> str | None:
    configured = os.environ.get("DOCKER_HOST")
    if configured:
        _validate_local_docker_host(configured)
        return configured
    if shutil.which(docker_binary) is None:
        return None
    try:
        completed = subprocess.run(
            (
                docker_binary,
                "context",
                "inspect",
                "--format",
                "{{.Endpoints.docker.Host}}",
            ),
            check=False,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    endpoint = completed.stdout.strip()
    if not endpoint:
        return None
    _validate_local_docker_host(endpoint)
    return endpoint


def _validate_local_docker_host(value: str) -> None:
    if "\n" in value or "\r" in value:
        raise ValueError("docker_host_must_be_local_socket")
    if value.startswith("unix://"):
        socket_path = value.removeprefix("unix://")
        if not socket_path.startswith("/"):
            raise ValueError("docker_host_must_be_local_socket")
        return
    if os.name == "nt" and value.startswith("npipe://"):
        return
    raise ValueError("docker_host_must_be_local_socket")


def _run_process(
    *,
    argv: tuple[str, ...],
    cwd: Path,
    environment: dict[str, str],
    timeout_seconds: float,
    policy: ExecutionPolicy,
    backend: str,
    termination_callback: Callable[[], None] | None = None,
    enforce_disk_reserve: bool = True,
) -> CommandOutcome:
    started = monotonic()
    if not argv:
        return CommandOutcome(
            status="blocked",
            exit_code=None,
            elapsed_sec=0.0,
            blocked_reason="empty_command",
            backend=backend,
        )
    if enforce_disk_reserve:
        try:
            free_bytes = shutil.disk_usage(cwd).free
        except OSError as exc:
            return CommandOutcome(
                status="infra_error",
                exit_code=None,
                elapsed_sec=round(monotonic() - started, 6),
                blocked_reason=(f"disk_reserve_probe_failed:{type(exc).__name__}"),
                backend=backend,
            )
        if free_bytes < policy.min_free_disk_bytes:
            return CommandOutcome(
                status="blocked",
                exit_code=None,
                elapsed_sec=round(monotonic() - started, 6),
                blocked_reason="disk_reserve_exhausted",
                backend=backend,
            )
    try:
        process = subprocess.Popen(
            argv,
            cwd=cwd,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=False,
            bufsize=0,
            start_new_session=(os.name == "posix"),
        )
    except FileNotFoundError:
        return CommandOutcome(
            status="blocked",
            exit_code=None,
            elapsed_sec=round(monotonic() - started, 6),
            blocked_reason=f"executable_not_found:{Path(argv[0]).name}",
            backend=backend,
        )
    except OSError as exc:
        return CommandOutcome(
            status="infra_error",
            exit_code=None,
            elapsed_sec=round(monotonic() - started, 6),
            stderr_tail=_tail_text(str(exc)),
            blocked_reason=f"process_start_failed:{type(exc).__name__}",
            backend=backend,
        )
    budget = _OutputBudget(policy.max_output_bytes)
    stdout_capture = _BoundedOutputCapture(
        budget=budget,
        tail_bytes=policy.output_tail_bytes,
    )
    stderr_capture = _BoundedOutputCapture(
        budget=budget,
        tail_bytes=policy.output_tail_bytes,
    )
    assert process.stdout is not None
    assert process.stderr is not None
    readers = (
        threading.Thread(
            target=_drain_stream,
            args=(process.stdout, stdout_capture),
            daemon=True,
        ),
        threading.Thread(
            target=_drain_stream,
            args=(process.stderr, stderr_capture),
            daemon=True,
        ),
    )
    for reader in readers:
        reader.start()

    deadline = started + timeout_seconds
    termination_reason: str | None = None
    next_disk_check = started
    while process.poll() is None:
        if budget.exceeded:
            termination_reason = "output_limit_exceeded"
            break
        if monotonic() >= deadline:
            termination_reason = f"timeout_after_seconds:{timeout_seconds:g}"
            break
        if enforce_disk_reserve and monotonic() >= next_disk_check:
            try:
                free_bytes = shutil.disk_usage(cwd).free
            except OSError:
                termination_reason = "disk_reserve_probe_failed"
                break
            if free_bytes < policy.min_free_disk_bytes:
                termination_reason = "disk_reserve_exhausted"
                break
            next_disk_check = monotonic() + 0.25
        time.sleep(0.01)
    if termination_reason is not None:
        _terminate_process_group(process)
        if termination_callback is not None:
            termination_callback()
        try:
            process.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=1.0)
    else:
        process.wait()
    for reader in readers:
        reader.join(timeout=0.5)
    process.stdout.close()
    process.stderr.close()
    for reader in readers:
        reader.join(timeout=0.5)

    if budget.exceeded and termination_reason is None:
        termination_reason = "output_limit_exceeded"
    if termination_reason is not None:
        resource_limited = termination_reason in {
            "output_limit_exceeded",
            "disk_reserve_exhausted",
        }
        infrastructure_failure = termination_reason == "disk_reserve_probe_failed"
        return CommandOutcome(
            status=(
                "blocked"
                if resource_limited
                else "infra_error"
                if infrastructure_failure
                else "timeout"
            ),
            exit_code=None,
            elapsed_sec=round(monotonic() - started, 6),
            stdout_tail=stdout_capture.tail_text,
            stderr_tail=stderr_capture.tail_text,
            blocked_reason=termination_reason,
            backend=backend,
            stdout_hash=stdout_capture.digest,
            stderr_hash=stderr_capture.digest,
        )
    return CommandOutcome(
        status="passed" if process.returncode == 0 else "failed",
        exit_code=process.returncode,
        elapsed_sec=round(monotonic() - started, 6),
        stdout_tail=stdout_capture.tail_text,
        stderr_tail=stderr_capture.tail_text,
        backend=backend,
        stdout_hash=stdout_capture.digest,
        stderr_hash=stderr_capture.digest,
    )


class _OutputBudget:
    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.total = 0
        self.exceeded = False
        self.lock = threading.Lock()

    def add(self, size: int) -> None:
        with self.lock:
            self.total += size
            if self.total > self.limit:
                self.exceeded = True


class _BoundedOutputCapture:
    def __init__(self, *, budget: _OutputBudget, tail_bytes: int) -> None:
        self.budget = budget
        self.tail_bytes = tail_bytes
        self.tail = bytearray()
        self.hasher = hashlib.sha256()
        self.lock = threading.Lock()

    def append(self, chunk: bytes) -> None:
        self.budget.add(len(chunk))
        with self.lock:
            self.hasher.update(chunk)
            self.tail.extend(chunk)
            if len(self.tail) > self.tail_bytes:
                del self.tail[: len(self.tail) - self.tail_bytes]

    @property
    def tail_text(self) -> str:
        with self.lock:
            return bytes(self.tail).decode("utf-8", errors="replace")

    @property
    def digest(self) -> str:
        with self.lock:
            return self.hasher.hexdigest()


def _drain_stream(
    stream: object,
    capture: _BoundedOutputCapture,
) -> None:
    try:
        while True:
            chunk = stream.read(64 * 1024)  # type: ignore[attr-defined]
            if not chunk:
                return
            capture.append(chunk)
    except (OSError, ValueError):
        return


def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGKILL)
            return
        except ProcessLookupError:
            return
        except PermissionError:
            pass
    try:
        process.kill()
    except ProcessLookupError:
        pass


def _tail_text(value: str, *, max_chars: int = 4000) -> str:
    return value[-max_chars:]
