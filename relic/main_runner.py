"""Canonical 120-cell main-study subprocess scheduler.

The scheduler owns orchestration metadata only.  Every experiment cell is
compiled by :mod:`relic.cell_spec` and executed by the existing ``run-cell``
command in a dedicated process.  This keeps provider clients, evaluator state,
and the identity environment isolated between cells.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator, Mapping, TextIO

from relic.cell_spec import (
    SOURCE_BRANCH,
    SOURCE_COMMIT,
    SOURCE_GIT_TREE,
    SOURCE_REPOSITORY,
    compile_cell_spec,
    release_runtime_tree,
    stable_sha256,
)
from relic.manifest import load_yaml, recommended_parallelism, visible_memory_gib, write_manifest
from relic.paths import config_root, default_output_root, project_root

RUN_MANIFEST_SCHEMA_VERSION = "relic-run-manifest-v2"
_RETRYABLE_FAILURE_CLASSES = frozenset(
    {"model_failure", "evaluator_failure", "infrastructure_failure"}
)
_RESUMABLE_STATUSES = frozenset({"pending", "launching", "running", "interrupted"})
_RUNTIME_STATUSES = frozenset(
    {"pending", "launching", "running", "completed", "failed", "interrupted"}
)
_SAFE_FAILURE_CODE = re.compile(r"^[a-z0-9][a-z0-9_.:-]{0,159}$")


class MainRunnerError(RuntimeError):
    """A stable, user-facing scheduler failure."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class MainRunResult:
    manifest_path: Path
    output_root: Path
    model: str
    status: str
    total_cells: int
    selected_cells: int
    completed_cells: int
    failed_cells: int
    interrupted_cells: int
    max_parallel: int
    visible_memory_gib: float
    recommended_max_parallel: int
    dry_run: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest": str(self.manifest_path),
            "output_root": str(self.output_root),
            "model": self.model,
            "status": self.status,
            "paper_design": "10 workloads x 3 seeds x 4 arms",
            "total_cells": self.total_cells,
            "selected_cells": self.selected_cells,
            "completed_cells": self.completed_cells,
            "failed_cells": self.failed_cells,
            "interrupted_cells": self.interrupted_cells,
            "max_parallel": self.max_parallel,
            "visible_memory_gib": self.visible_memory_gib,
            "recommended_max_parallel": self.recommended_max_parallel,
            "dry_run": self.dry_run,
        }


@dataclass
class _ActiveChild:
    cell_id: str
    process: subprocess.Popen[bytes]
    stdout: TextIO
    stderr: TextIO
    started_at: str


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _plan_digest(plan: Mapping[str, Any]) -> str:
    return stable_sha256({key: value for key, value in plan.items() if key != "plan_sha256"})


def _canonical_study_axes() -> tuple[list[str], list[int], list[str]]:
    study = load_yaml(config_root() / "main-study.yaml")
    workloads = [str(value).lower() for value in study["workloads"]]
    seeds = [int(value) for value in study["seeds"]]
    arms = [str(value).lower() for value in study["arms"]]
    if len(workloads) * len(seeds) * len(arms) != 120:
        raise MainRunnerError("canonical_main_study_cell_count_invalid")
    return workloads, seeds, arms


def build_run_manifest(
    *,
    model: str,
    output_root: Path | None = None,
    max_parallel: int = 1,
) -> dict[str, Any]:
    """Freeze the immutable plan and initial runtime state for one model."""

    if max_parallel < 1:
        raise MainRunnerError("max_parallel_must_be_positive")
    destination = (output_root or default_output_root() / "main-study").expanduser().resolve()
    workloads, seeds, arms = _canonical_study_axes()
    cells: list[dict[str, Any]] = []
    runtime_cells: dict[str, dict[str, Any]] = {}
    source: dict[str, Any] | None = None
    normalized_model: str | None = None
    for workload in workloads:
        for seed in seeds:
            for arm in arms:
                try:
                    spec = compile_cell_spec(
                        model=model,
                        workload=workload,
                        arm=arm,
                        seed=seed,
                        output_root=destination,
                    )
                except ValueError as exc:
                    raise MainRunnerError(str(exc)) from exc
                normalized_model = spec.model
                source = dict(spec.source)
                cells.append(
                    {
                        "cell_id": spec.cell_id,
                        "model": spec.model,
                        "workload": spec.workload.upper(),
                        "arm": spec.arm.upper(),
                        "seed": spec.seed,
                        "output_path": str(spec.cell_dir),
                        "cell_spec_sha256": spec.fingerprint,
                        "cell_spec": spec.document(),
                    }
                )
                runtime_cells[spec.cell_id] = {
                    "status": "pending",
                    "attempt_count": 0,
                    "failure_class": None,
                    "failure_code": None,
                    "attempts": [],
                }
    if len(cells) != 120 or len(runtime_cells) != 120:
        raise MainRunnerError("canonical_main_study_cells_not_unique")
    assert source is not None and normalized_model is not None
    expected_source = {
        "repository": SOURCE_REPOSITORY,
        "branch": SOURCE_BRANCH,
        "commit": SOURCE_COMMIT,
        "git_tree": SOURCE_GIT_TREE,
    }
    if source != expected_source:
        raise MainRunnerError("canonical_source_provenance_mismatch")
    plan: dict[str, Any] = {
        "study": "relic-main-v1",
        "model": normalized_model,
        "output_root": str(destination),
        "source": source,
        "cells": cells,
    }
    plan["plan_sha256"] = _plan_digest(plan)
    memory_gib = visible_memory_gib()
    recommended = recommended_parallelism(memory_gib)
    return {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "plan": plan,
        "runtime": {
            "revision": 0,
            "status": "planned",
            "run_origin": {
                "kind": "user_new_run",
                "instance_id": uuid.uuid4().hex,
                "created_at": _utc_now(),
            },
            "sessions": [],
            "cells": runtime_cells,
        },
        "resource_advice": {
            "visible_memory_gib": round(memory_gib, 2),
            "recommended_max_parallel": recommended,
            "budget_gib_per_active_cell": 16,
            "requested_max_parallel": max_parallel,
            "warning": max_parallel > recommended,
        },
    }


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise MainRunnerError("run_manifest_missing") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise MainRunnerError("run_manifest_unreadable") from exc
    if not isinstance(payload, dict):
        raise MainRunnerError("run_manifest_invalid")
    return payload


def _safe_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _validate_manifest(payload: Mapping[str, Any]) -> None:
    """Validate the entire frozen plan before any child can be launched."""

    if payload.get("schema_version") != RUN_MANIFEST_SCHEMA_VERSION:
        raise MainRunnerError("run_manifest_schema_mismatch")
    plan = payload.get("plan")
    runtime = payload.get("runtime")
    if not isinstance(plan, Mapping) or not isinstance(runtime, Mapping):
        raise MainRunnerError("run_manifest_structure_invalid")
    try:
        revision = int(runtime.get("revision", -1))
    except (TypeError, ValueError) as exc:
        raise MainRunnerError("run_manifest_revision_invalid") from exc
    if revision < 0 or not isinstance(runtime.get("sessions"), list):
        raise MainRunnerError("run_manifest_runtime_invalid")
    run_origin = runtime.get("run_origin")
    if not isinstance(run_origin, Mapping):
        raise MainRunnerError("run_manifest_origin_missing")
    if run_origin.get("kind") != "user_new_run":
        raise MainRunnerError("run_manifest_origin_invalid")
    instance_id = str(run_origin.get("instance_id") or "")
    created_at = str(run_origin.get("created_at") or "")
    if not re.fullmatch(r"[0-9a-f]{32}", instance_id) or not created_at:
        raise MainRunnerError("run_manifest_origin_invalid")
    if plan.get("plan_sha256") != _plan_digest(plan):
        raise MainRunnerError("run_manifest_plan_hash_mismatch")
    if plan.get("study") != "relic-main-v1":
        raise MainRunnerError("run_manifest_study_mismatch")
    source = plan.get("source")
    if not isinstance(source, Mapping):
        raise MainRunnerError("run_manifest_source_invalid")
    expected_source = {
        "repository": SOURCE_REPOSITORY,
        "branch": SOURCE_BRANCH,
        "commit": SOURCE_COMMIT,
        "git_tree": SOURCE_GIT_TREE,
    }
    if dict(source) != expected_source:
        raise MainRunnerError("run_manifest_source_mismatch")
    raw_cells = plan.get("cells")
    runtime_cells = runtime.get("cells")
    if not isinstance(raw_cells, list) or len(raw_cells) != 120:
        raise MainRunnerError("run_manifest_plan_cells_invalid")
    if not isinstance(runtime_cells, Mapping) or len(runtime_cells) != 120:
        raise MainRunnerError("run_manifest_runtime_cells_invalid")
    output_root = Path(str(plan.get("output_root") or "")).expanduser().resolve()
    model = str(plan.get("model") or "")
    observed_ids: set[str] = set()
    for raw_cell in raw_cells:
        if not isinstance(raw_cell, Mapping):
            raise MainRunnerError("run_manifest_cell_invalid")
        cell_id = str(raw_cell.get("cell_id") or "")
        if not cell_id or cell_id in observed_ids:
            raise MainRunnerError("run_manifest_cell_id_invalid")
        observed_ids.add(cell_id)
        try:
            spec = compile_cell_spec(
                model=model,
                workload=str(raw_cell.get("workload") or ""),
                arm=str(raw_cell.get("arm") or ""),
                seed=int(raw_cell.get("seed")),
                output_root=output_root,
            )
        except (TypeError, ValueError) as exc:
            raise MainRunnerError("run_manifest_cell_compile_failed") from exc
        declared_output = Path(str(raw_cell.get("output_path") or "")).expanduser().resolve()
        if declared_output != spec.cell_dir or not _safe_relative_to(declared_output, output_root):
            raise MainRunnerError("run_manifest_cell_output_mismatch")
        if raw_cell.get("cell_spec_sha256") != spec.fingerprint:
            raise MainRunnerError("run_manifest_cell_spec_hash_mismatch")
        if raw_cell.get("cell_spec") != spec.document():
            raise MainRunnerError("run_manifest_cell_spec_mismatch")
        cell_runtime = runtime_cells.get(cell_id)
        if not isinstance(cell_runtime, Mapping):
            raise MainRunnerError("run_manifest_cell_runtime_missing")
        if cell_runtime.get("status") not in _RUNTIME_STATUSES:
            raise MainRunnerError("run_manifest_cell_status_invalid")
        attempts = cell_runtime.get("attempts")
        if not isinstance(attempts, list):
            raise MainRunnerError("run_manifest_cell_attempts_invalid")
        try:
            attempt_count = int(cell_runtime.get("attempt_count", -1))
        except (TypeError, ValueError) as exc:
            raise MainRunnerError("run_manifest_cell_attempt_count_invalid") from exc
        if attempt_count != len(attempts):
            raise MainRunnerError("run_manifest_cell_attempt_count_invalid")
    if observed_ids != set(runtime_cells):
        raise MainRunnerError("run_manifest_cell_sets_mismatch")


@contextmanager
def _manifest_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.parent.is_symlink():
        raise MainRunnerError("run_manifest_parent_symlink_forbidden")
    if path.is_symlink():
        raise MainRunnerError("run_manifest_symlink_forbidden")
    lock_path = path.with_name(f".{path.name}.lock")
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as exc:
        raise MainRunnerError("run_manifest_lock_open_failed") from exc
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise MainRunnerError("run_manifest_locked") from exc
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _bump_and_write(payload: dict[str, Any], path: Path) -> None:
    runtime = payload["runtime"]
    runtime["revision"] = int(runtime["revision"]) + 1
    write_manifest(payload, path)


def _selected_cell_ids(
    payload: Mapping[str, Any], *, resume: bool, retry_failed: bool, cell_id: str | None
) -> list[str]:
    plan_cells = payload["plan"]["cells"]
    runtime_cells = payload["runtime"]["cells"]
    known = {str(cell["cell_id"]) for cell in plan_cells}
    if cell_id is not None and cell_id not in known:
        raise MainRunnerError("requested_cell_id_unknown")
    selected: list[str] = []
    for cell in plan_cells:
        current_id = str(cell["cell_id"])
        if cell_id is not None and current_id != cell_id:
            continue
        state = runtime_cells[current_id]
        status = state["status"]
        if status == "completed":
            continue
        if retry_failed:
            if status == "failed" and state.get("failure_class") in _RETRYABLE_FAILURE_CLASSES:
                selected.append(current_id)
        elif resume:
            if status in _RESUMABLE_STATUSES:
                selected.append(current_id)
        elif status == "pending":
            selected.append(current_id)
    return selected


def _cell_index(payload: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(cell["cell_id"]): cell for cell in payload["plan"]["cells"]}


def _provider(payload: Mapping[str, Any]) -> str:
    first = payload["plan"]["cells"][0]
    return str(first["cell_spec"]["configs"]["model"].get("provider") or "")


def _safe_failure_code(value: Any, fallback: str) -> str:
    code = str(value or "")
    return code if _SAFE_FAILURE_CODE.fullmatch(code) else fallback


def _classify_child(cell: Mapping[str, Any], return_code: int) -> tuple[str, str | None]:
    if return_code < 0 or return_code in {128 + signal.SIGINT, 128 + signal.SIGTERM}:
        return "interrupted", "child_interrupted"
    status_path = Path(str(cell["output_path"])) / "public" / "status.json"
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "infrastructure_failure", "child_status_missing"
    if not isinstance(status, dict) or status.get("cell_id") != cell["cell_id"]:
        return "infrastructure_failure", "child_status_identity_mismatch"
    if return_code == 0 and status.get("status") == "completed":
        return "completed", None
    failure_code = _safe_failure_code(status.get("failure_code"), "child_process_failed")
    if failure_code == "model_failure" or failure_code.startswith("unsupported_model_provider"):
        return "model_failure", failure_code
    if status.get("stage") == "evaluation" or failure_code.startswith("evaluator_"):
        return "evaluator_failure", failure_code
    return "infrastructure_failure", failure_code


def _attempt_log_paths(output_root: Path, cell_id: str, attempt_no: int) -> tuple[Path, Path]:
    directory = output_root / "private" / "scheduler" / cell_id
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    return directory / f"attempt-{attempt_no}.stdout.log", directory / f"attempt-{attempt_no}.stderr.log"


def _open_private_log(path: Path) -> TextIO:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    return os.fdopen(descriptor, "w", encoding="utf-8")


@contextmanager
def _defer_sigint_until_registered() -> Iterator[None]:
    """Prevent SIGINT from landing between Popen return and child registration."""

    if threading.current_thread() is not threading.main_thread():
        raise MainRunnerError("scheduler_execution_requires_main_thread")
    previous = signal.getsignal(signal.SIGINT)
    if previous == signal.SIG_IGN:
        yield
        return
    interrupted = False

    def defer(_signum: int, _frame: Any) -> None:
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, defer)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous)
        if interrupted:
            if callable(previous):
                previous(signal.SIGINT, None)
            else:
                raise KeyboardInterrupt


def _revalidate_dispatch_cell(payload: Mapping[str, Any], cell: Mapping[str, Any]) -> None:
    """Recompile identity immediately before dispatch, not only at session start."""

    release_runtime_tree.cache_clear()
    try:
        spec = compile_cell_spec(
            model=str(cell["model"]),
            workload=str(cell["workload"]),
            arm=str(cell["arm"]),
            seed=int(cell["seed"]),
            output_root=Path(str(payload["plan"]["output_root"])),
        )
    except (TypeError, ValueError) as exc:
        raise MainRunnerError("dispatch_cell_compile_failed") from exc
    if (
        spec.cell_id != cell.get("cell_id")
        or spec.cell_dir != Path(str(cell.get("output_path"))).resolve()
        or spec.fingerprint != cell.get("cell_spec_sha256")
        or spec.document() != cell.get("cell_spec")
    ):
        raise MainRunnerError("dispatch_cell_identity_mismatch")


def _launch_child(
    payload: dict[str, Any],
    manifest_path: Path,
    cell: Mapping[str, Any],
    *,
    python_executable: str,
    active: dict[str, _ActiveChild],
) -> _ActiveChild | None:
    cell_id = str(cell["cell_id"])
    _revalidate_dispatch_cell(payload, cell)
    state = payload["runtime"]["cells"][cell_id]
    attempt_no = int(state["attempt_count"]) + 1
    output_root = Path(payload["plan"]["output_root"])
    stdout_path, stderr_path = _attempt_log_paths(output_root, cell_id, attempt_no)
    relative_stdout = stdout_path.relative_to(output_root).as_posix()
    relative_stderr = stderr_path.relative_to(output_root).as_posix()
    started_at = _utc_now()
    attempt = {
        "attempt_no": attempt_no,
        "started_at": started_at,
        "ended_at": None,
        "pid": None,
        "exit_code": None,
        "stage": "launching",
        "failure_class": None,
        "failure_code": None,
        "stdout_path": relative_stdout,
        "stderr_path": relative_stderr,
    }
    state.update(
        {
            "status": "launching",
            "attempt_count": attempt_no,
            "failure_class": None,
            "failure_code": None,
        }
    )
    state["attempts"].append(attempt)
    _bump_and_write(payload, manifest_path)
    stdout = _open_private_log(stdout_path)
    stderr = _open_private_log(stderr_path)
    output_path = Path(str(cell["output_path"]))
    argv = [
        python_executable,
        "-m",
        "relic.cli",
        "run-cell",
        "--model",
        str(cell["model"]),
        "--workload",
        str(cell["workload"]),
        "--arm",
        str(cell["arm"]),
        "--seed",
        str(cell["seed"]),
        "--output-dir",
        str(output_path),
    ]
    if (output_path / "cell-spec.json").is_file():
        argv.append("--resume")
    try:
        with _defer_sigint_until_registered():
            process = subprocess.Popen(
                argv,
                cwd=project_root(),
                env=dict(os.environ),
                start_new_session=True,
                stdout=stdout,
                stderr=stderr,
            )
            child = _ActiveChild(cell_id, process, stdout, stderr, started_at)
            # Registration happens while SIGINT is deferred. Any later
            # exception is therefore guaranteed to see and terminate it.
            active[cell_id] = child
    except OSError:
        stdout.close()
        stderr.close()
        ended_at = _utc_now()
        attempt.update(
            {
                "ended_at": ended_at,
                "stage": "launch_failed",
                "failure_class": "infrastructure_failure",
                "failure_code": "child_spawn_failed",
            }
        )
        state.update(
            {
                "status": "failed",
                "failure_class": "infrastructure_failure",
                "failure_code": "child_spawn_failed",
            }
        )
        _bump_and_write(payload, manifest_path)
        return None
    except BaseException:
        if cell_id not in active:
            stdout.close()
            stderr.close()
        raise
    attempt.update({"pid": process.pid, "stage": "running"})
    state["status"] = "running"
    _bump_and_write(payload, manifest_path)
    return child


def _record_child_exit(
    payload: dict[str, Any], manifest_path: Path, child: _ActiveChild, cell: Mapping[str, Any]
) -> None:
    return_code = int(child.process.returncode)
    child.stdout.close()
    child.stderr.close()
    classification, failure_code = _classify_child(cell, return_code)
    state = payload["runtime"]["cells"][child.cell_id]
    attempt = state["attempts"][-1]
    attempt.update(
        {
            "ended_at": _utc_now(),
            "exit_code": return_code,
            "stage": "complete" if classification == "completed" else "failed",
            "failure_class": None if classification == "completed" else classification,
            "failure_code": failure_code,
        }
    )
    state.update(
        {
            "status": "completed" if classification == "completed" else classification,
            "failure_class": None if classification == "completed" else classification,
            "failure_code": failure_code,
        }
    )
    if classification not in {"completed", "interrupted"}:
        state["status"] = "failed"
    _bump_and_write(payload, manifest_path)


def _signal_process_group(child: _ActiveChild, sig: signal.Signals) -> None:
    try:
        os.killpg(child.process.pid, sig)
    except ProcessLookupError:
        return


def _terminate_children(active: dict[str, _ActiveChild], grace_seconds: float) -> None:
    for child in active.values():
        _signal_process_group(child, signal.SIGINT)
    deadline = time.monotonic() + grace_seconds
    while active and time.monotonic() < deadline:
        if all(child.process.poll() is not None for child in active.values()):
            return
        time.sleep(0.05)
    for child in active.values():
        if child.process.poll() is None:
            _signal_process_group(child, signal.SIGTERM)
    deadline = time.monotonic() + min(grace_seconds, 2.0)
    while active and time.monotonic() < deadline:
        if all(child.process.poll() is not None for child in active.values()):
            return
        time.sleep(0.05)
    for child in active.values():
        if child.process.poll() is None:
            _signal_process_group(child, signal.SIGKILL)
            child.process.wait()


def _mark_children_interrupted(
    payload: dict[str, Any], active: Mapping[str, _ActiveChild], *, failure_code: str
) -> None:
    runtime = payload["runtime"]
    for current_id, child in active.items():
        child.process.poll()
        if not child.stdout.closed:
            child.stdout.close()
        if not child.stderr.closed:
            child.stderr.close()
        state = runtime["cells"][current_id]
        attempt = state["attempts"][-1]
        attempt.update(
            {
                "ended_at": _utc_now(),
                "exit_code": child.process.returncode,
                "stage": "interrupted",
                "failure_class": "interrupted",
                "failure_code": failure_code,
            }
        )
        state.update(
            {
                "status": "interrupted",
                "failure_class": "interrupted",
                "failure_code": failure_code,
            }
        )


def _runtime_counts(payload: Mapping[str, Any]) -> dict[str, int]:
    states = payload["runtime"]["cells"].values()
    return {
        "completed": sum(state["status"] == "completed" for state in states),
        "failed": sum(state["status"] == "failed" for state in states),
        "interrupted": sum(state["status"] == "interrupted" for state in states),
    }


def _final_runtime_status(payload: Mapping[str, Any], *, interrupted: bool) -> str:
    if interrupted:
        return "interrupted"
    statuses = [state["status"] for state in payload["runtime"]["cells"].values()]
    if all(status == "completed" for status in statuses):
        return "complete"
    if any(status == "failed" for status in statuses):
        return "complete_with_failures"
    return "partial"


def run_main(
    *,
    model: str | None = None,
    output_root: Path | None = None,
    manifest_path: Path | None = None,
    max_parallel: int = 1,
    dry_run: bool = False,
    resume: bool = False,
    retry_failed: bool = False,
    cell_id: str | None = None,
    python_executable: str | None = None,
    poll_interval: float = 0.05,
    interrupt_grace_seconds: float = 5.0,
) -> MainRunResult:
    """Create or resume one canonical single-model 120-cell run."""

    if max_parallel < 1:
        raise MainRunnerError("max_parallel_must_be_positive")
    if retry_failed and not resume:
        raise MainRunnerError("retry_failed_requires_resume")
    if not resume and not model:
        raise MainRunnerError("model_required_for_new_run")
    resolved_output = (
        output_root.expanduser().resolve()
        if output_root is not None
        else default_output_root().joinpath("main-study").resolve()
    )
    resolved_manifest = (
        manifest_path.expanduser().resolve()
        if manifest_path is not None
        else resolved_output / "run_manifest.json"
    )
    with _manifest_lock(resolved_manifest):
        if resume:
            payload = _read_manifest(resolved_manifest)
            _validate_manifest(payload)
            planned_root = Path(payload["plan"]["output_root"]).resolve()
            if output_root is not None and resolved_output != planned_root:
                raise MainRunnerError("resume_output_root_mismatch")
            if model is not None and model.strip().lower() != payload["plan"]["model"]:
                raise MainRunnerError("resume_model_mismatch")
        else:
            if resolved_manifest.exists():
                raise MainRunnerError("run_manifest_already_exists")
            payload = build_run_manifest(
                model=str(model), output_root=resolved_output, max_parallel=max_parallel
            )
            _validate_manifest(payload)
            write_manifest(payload, resolved_manifest)
        selected = _selected_cell_ids(
            payload, resume=resume, retry_failed=retry_failed, cell_id=cell_id
        )
        if dry_run:
            counts = _runtime_counts(payload)
            return MainRunResult(
                manifest_path=resolved_manifest,
                output_root=Path(payload["plan"]["output_root"]),
                model=str(payload["plan"]["model"]),
                status="planned",
                total_cells=120,
                selected_cells=len(selected),
                completed_cells=counts["completed"],
                failed_cells=counts["failed"],
                interrupted_cells=counts["interrupted"],
                max_parallel=max_parallel,
                visible_memory_gib=float(payload["resource_advice"]["visible_memory_gib"]),
                recommended_max_parallel=int(
                    payload["resource_advice"]["recommended_max_parallel"]
                ),
                dry_run=True,
            )
        provider = _provider(payload)
        if provider != "openai":
            raise MainRunnerError(f"unsupported_model_provider:{provider or 'unknown'}")

        runtime = payload["runtime"]
        session = {
            "session_id": len(runtime["sessions"]) + 1,
            "started_at": _utc_now(),
            "ended_at": None,
            "status": "running",
            "selected_cells": len(selected),
            "max_parallel": max_parallel,
            "retry_failed": retry_failed,
            "cell_id": cell_id,
        }
        runtime["sessions"].append(session)
        runtime["status"] = "running"
        _bump_and_write(payload, resolved_manifest)
        cells = _cell_index(payload)
        queue = list(selected)
        active: dict[str, _ActiveChild] = {}
        was_interrupted = False
        executable = python_executable or sys.executable
        try:
            while queue or active:
                while queue and len(active) < max_parallel:
                    next_id = queue.pop(0)
                    child = _launch_child(
                        payload,
                        resolved_manifest,
                        cells[next_id],
                        python_executable=executable,
                        active=active,
                    )
                finished = [
                    current_id
                    for current_id, child in active.items()
                    if child.process.poll() is not None
                ]
                for current_id in finished:
                    child = active.pop(current_id)
                    _record_child_exit(payload, resolved_manifest, child, cells[current_id])
                if active and not finished:
                    time.sleep(poll_interval)
        except KeyboardInterrupt:
            was_interrupted = True
            _terminate_children(active, interrupt_grace_seconds)
            _mark_children_interrupted(
                payload, active, failure_code="scheduler_interrupted"
            )
            _bump_and_write(payload, resolved_manifest)
        except Exception as exc:
            # A scheduler-side persistence or bookkeeping failure must never
            # leave expensive model/evaluator children running unattended.
            try:
                _terminate_children(active, interrupt_grace_seconds)
            finally:
                _mark_children_interrupted(
                    payload, active, failure_code="scheduler_runtime_failure"
                )
                runtime["status"] = "interrupted"
                session["ended_at"] = _utc_now()
                session["status"] = "interrupted"
                try:
                    _bump_and_write(payload, resolved_manifest)
                except Exception:
                    pass
            if isinstance(exc, MainRunnerError):
                raise
            raise MainRunnerError("scheduler_runtime_failure") from exc
        finally:
            for child in active.values():
                if not child.stdout.closed:
                    child.stdout.close()
                if not child.stderr.closed:
                    child.stderr.close()

        final_status = _final_runtime_status(payload, interrupted=was_interrupted)
        runtime["status"] = final_status
        session["ended_at"] = _utc_now()
        session["status"] = final_status
        _bump_and_write(payload, resolved_manifest)
        counts = _runtime_counts(payload)
        result = MainRunResult(
            manifest_path=resolved_manifest,
            output_root=Path(payload["plan"]["output_root"]),
            model=str(payload["plan"]["model"]),
            status=final_status,
            total_cells=120,
            selected_cells=len(selected),
            completed_cells=counts["completed"],
            failed_cells=counts["failed"],
            interrupted_cells=counts["interrupted"],
            max_parallel=max_parallel,
            visible_memory_gib=float(payload["resource_advice"]["visible_memory_gib"]),
            recommended_max_parallel=int(
                payload["resource_advice"]["recommended_max_parallel"]
            ),
            dry_run=False,
        )
        if was_interrupted:
            raise KeyboardInterrupt
        return result


__all__ = [
    "MainRunResult",
    "MainRunnerError",
    "RUN_MANIFEST_SCHEMA_VERSION",
    "build_run_manifest",
    "run_main",
]
