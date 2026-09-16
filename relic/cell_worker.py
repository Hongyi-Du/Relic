"""Canonical single-cell execution, resume, status, and evaluation.

Private checkpoints and evaluator evidence never flow through the public trace
writer.  Public JSON is constructed from a small explicit allow-list.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import fcntl
import json
import os
import stat
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

from environments.org_env.backend.simulation.world import OrgWorld
from environments.org_env.backend.entities.work import TaskStatus
from environments.org_env.config.scenarios import oss_time_machine_formal
from environments.org_env.experiments.resources import (
    ExperimentResourceExhausted,
    attach_metered_llm_client,
)
from environments.org_env.llm.client import LLMError, OpenAIOrgLLMClient
from environments.org_env.product.substrates.eval_assets import validate_formal_oss_world
from environments.org_env.product.substrates.final_evaluation import (
    run_final_evaluation,
    write_experiment_run_record,
)
from environments.org_env.runtime_adapter.checkpoint import (
    checkpoint_info,
    load_world_checkpoint,
    save_world_checkpoint,
)
from relic.cell_spec import CellSpec, load_frozen_cell_spec, stable_sha256
from relic.evaluation.execution import ExecutionPolicy, build_command_executor
from relic.evaluation.time_machine import build_time_machine_evaluation_plan
from relic.research.openai_runtime import configured_openai_default_headers

PUBLIC_STATUS_SCHEMA_VERSION = "relic-cell-status-v1"
PUBLIC_TRACE_SCHEMA_VERSION = "relic-public-trace-v1"
EXECUTION_BINDING_SCHEMA_VERSION = "relic-execution-binding-v1"
PRIVATE_STATE_SCHEMA_VERSION = "relic-private-cell-state-v1"

_TERMINAL_STATUSES = frozenset({"completed", "failed", "infra_error", "blocked"})
_ENVIRONMENT_LOCK = threading.RLock()


class CellWorkerError(RuntimeError):
    """A fail-closed runner error carrying a stable, non-secret code."""

    def __init__(self, code: str):
        self.code = str(code)
        super().__init__(self.code)


@dataclass(frozen=True)
class CellResult:
    cell_id: str
    cell_dir: Path
    status: str
    stage: str
    tick: int
    failure_code: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "cell_id": self.cell_id,
            "cell_dir": str(self.cell_dir),
            "status": self.status,
            "stage": self.stage,
            "tick": self.tick,
            "failure_code": self.failure_code,
        }


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: Mapping[str, Any], *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        if os.name == "posix":
            directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path) -> dict[str, Any]:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        details = os.fstat(descriptor)
        if not stat.S_ISREG(details.st_mode) or details.st_size > 16 * 1024 * 1024:
            raise CellWorkerError(f"invalid_json_file:{path.name}")
        with os.fdopen(descriptor, "r", encoding="utf-8") as handle:
            descriptor = -1
            payload = json.load(handle)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if not isinstance(payload, dict):
        raise CellWorkerError(f"invalid_json_object:{path.name}")
    return payload


def _write_immutable_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        if _read_json(path) != dict(payload):
            raise CellWorkerError(f"immutable_file_mismatch:{path.name}")
        return
    _atomic_write_json(path, payload)


def _ensure_layout(cell_dir: Path) -> None:
    if cell_dir.is_symlink():
        raise CellWorkerError("cell_directory_symlink_forbidden")
    cell_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    _validate_owned_directory(cell_dir, label="cell_directory")
    for relative in (
        "private",
        "private/checkpoints",
        "private/evaluator",
        "public",
    ):
        path = cell_dir / relative
        if path.is_symlink():
            raise CellWorkerError(f"cell_subdirectory_symlink_forbidden:{relative}")
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
        _validate_owned_directory(path, label=f"cell_subdirectory:{relative}")
        try:
            path.chmod(0o700)
        except OSError:
            pass


def _validate_owned_directory(path: Path, *, label: str) -> None:
    details = path.lstat()
    if not stat.S_ISDIR(details.st_mode):
        raise CellWorkerError(f"{label}_not_directory")
    if hasattr(os, "getuid") and details.st_uid != os.getuid():
        raise CellWorkerError(f"{label}_owner_mismatch")
    if details.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        raise CellWorkerError(f"{label}_permissions_too_open")


@contextlib.contextmanager
def _cell_lock(cell_dir: Path) -> Iterator[None]:
    lock_path = cell_dir / ".cell.lock"
    descriptor = os.open(
        lock_path,
        os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        details = os.fstat(descriptor)
        if not stat.S_ISREG(details.st_mode):
            raise CellWorkerError("cell_lock_not_regular_file")
        if hasattr(os, "getuid") and details.st_uid != os.getuid():
            raise CellWorkerError("cell_lock_owner_mismatch")
        with os.fdopen(descriptor, "a+") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            yield
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        # fdopen owns the descriptor after it succeeds.
        pass


@contextlib.contextmanager
def _scoped_environment(values: Mapping[str, str]) -> Iterator[None]:
    # The canonical scheduler launches one process per cell.  This lock also
    # prevents accidental same-process threads from interleaving identity env.
    with _ENVIRONMENT_LOCK:
        previous = {key: os.environ.get(key) for key in values}
        try:
            for key, value in values.items():
                os.environ[key] = str(value)
            yield
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


def _runtime_model(model_config: Mapping[str, Any]) -> str:
    default = str(model_config.get("runtime_model_default") or "").strip()
    if default:
        return default
    environment_name = str(model_config.get("runtime_model_env") or "").strip()
    resolved = str(os.environ.get(environment_name) or "").strip() if environment_name else ""
    if not resolved:
        raise CellWorkerError("runtime_model_binding_missing")
    return resolved


def _build_openai_client(spec: CellSpec) -> tuple[OpenAIOrgLLMClient, dict[str, Any]]:
    provider = str(spec.model_config["provider"]).strip().lower()
    if provider != "openai":
        raise CellWorkerError(f"unsupported_model_provider:{provider}")
    api_key = os.environ.get("ORG_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise CellWorkerError("model_credential_missing:openai")
    runtime_model = _runtime_model(spec.model_config)
    wire_api = str(os.environ.get("ORG_LLM_WIRE_API") or "responses").strip().lower()
    base_url = os.environ.get("ORG_LLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
    try:
        default_headers = configured_openai_default_headers()
    except ValueError as exc:
        raise CellWorkerError("model_default_headers_invalid") from exc
    client = OpenAIOrgLLMClient(
        model=runtime_model,
        api_key=api_key,
        base_url=base_url,
        reasoning_effort=str(spec.model_config["reasoning_effort"]),
        max_retries=int(spec.model_config["max_retries"]),
        retry_backoff_seconds=float(spec.model_config["retry_backoff_seconds"]),
        wire_api=wire_api,
        request_timeout_seconds=float(spec.model_config["request_timeout_seconds"]),
        store_responses=False,
        default_headers=default_headers,
    )
    binding = {
        "provider": provider,
        "runtime_model": runtime_model,
        "reasoning_effort": str(spec.model_config["reasoning_effort"]),
        "request_timeout_seconds": float(spec.model_config["request_timeout_seconds"]),
        "max_retries": int(spec.model_config["max_retries"]),
        "retry_backoff_seconds": float(spec.model_config["retry_backoff_seconds"]),
        "wire_api": wire_api,
        "routing_context_fingerprint": client.routing_context_fingerprint,
    }
    return client, binding


def _preflight_evaluator(spec: CellSpec) -> dict[str, Any]:
    values = {
        "backend": str(os.environ.get("RELIC_EVALUATOR_BACKEND") or "").strip().lower(),
        "container_image": str(
            os.environ.get("RELIC_EVALUATOR_CONTAINER_IMAGE") or ""
        ).strip(),
        "container_platform": str(
            os.environ.get("RELIC_EVALUATOR_CONTAINER_PLATFORM") or ""
        ).strip(),
    }
    missing = sorted(key for key, value in values.items() if not value)
    if missing:
        raise CellWorkerError("formal_evaluator_runtime_binding_missing:" + ",".join(missing))
    allowed = tuple(spec.study_config["formal_evaluator_policy"]["allowed_backends"])
    if values["backend"] not in allowed:
        raise CellWorkerError("formal_evaluator_backend_forbidden")
    expected_platform = str(spec.study_config["formal_evaluator_policy"]["platform"])
    if values["container_platform"] != expected_platform:
        raise CellWorkerError("formal_evaluator_platform_mismatch")
    try:
        policy = ExecutionPolicy(
            trust_level="untrusted",
            backend=values["backend"],
            container_image=values["container_image"],
            container_platform=values["container_platform"],
            network_enabled=False,
        )
        executor = build_command_executor(policy)
        plan = build_time_machine_evaluation_plan(
            dataset_id=spec.dataset_id,
            timeout_seconds=int(os.environ.get("ORG_OSS_QUALIFICATION_TIMEOUT", "180")),
            executor=executor,
        )
    except CellWorkerError:
        raise
    except Exception as exc:
        raise CellWorkerError("formal_evaluator_preflight_error") from exc
    if not plan.formal_ready:
        raise CellWorkerError("formal_evaluator_not_ready")
    policy_payload = {
        "trust_level": policy.trust_level,
        "backend": policy.backend,
        "container_image": policy.container_image,
        "container_platform": policy.container_platform,
        "network_enabled": policy.network_enabled,
    }
    return {
        "execution_policy": policy_payload,
        "execution_policy_sha256": stable_sha256(policy_payload),
        "qualification_plan_sha256": plan.plan_hash,
        "evaluator_environment_sha256": plan.evaluator_environment_hash,
        "dataset_id": plan.dataset_id,
    }


def _execution_binding(
    spec: CellSpec,
    *,
    model_binding: Mapping[str, Any],
    evaluator_binding: Mapping[str, Any],
) -> dict[str, Any]:
    model_payload = dict(model_binding)
    payload = {
        "schema_version": EXECUTION_BINDING_SCHEMA_VERSION,
        "cell_spec_sha256": spec.fingerprint,
        "model": model_payload,
        "model_binding_sha256": stable_sha256(model_payload),
        "evaluator": dict(evaluator_binding),
        "source_provenance_sha256": spec.source_provenance_fingerprint,
        "resource_budget_sha256": spec.resource_budget.fingerprint,
    }
    return {**payload, "execution_binding_sha256": stable_sha256(payload)}


def _validate_execution_binding(spec: CellSpec, binding: Mapping[str, Any]) -> None:
    if binding.get("schema_version") != EXECUTION_BINDING_SCHEMA_VERSION:
        raise CellWorkerError("execution_binding_schema_mismatch")
    declared_hash = str(binding.get("execution_binding_sha256") or "")
    unhashed = {
        key: value for key, value in binding.items() if key != "execution_binding_sha256"
    }
    if stable_sha256(unhashed) != declared_hash:
        raise CellWorkerError("execution_binding_hash_mismatch")
    if binding.get("cell_spec_sha256") != spec.fingerprint:
        raise CellWorkerError("execution_binding_cell_spec_mismatch")
    if binding.get("source_provenance_sha256") != spec.source_provenance_fingerprint:
        raise CellWorkerError("execution_binding_source_mismatch")
    if binding.get("resource_budget_sha256") != spec.resource_budget.fingerprint:
        raise CellWorkerError("execution_binding_resource_budget_mismatch")
    model = binding.get("model")
    evaluator = binding.get("evaluator")
    if not isinstance(model, Mapping) or not isinstance(evaluator, Mapping):
        raise CellWorkerError("execution_binding_payload_invalid")
    if stable_sha256(model) != binding.get("model_binding_sha256"):
        raise CellWorkerError("execution_binding_model_hash_mismatch")
    required_evaluator_fields = {
        "execution_policy",
        "execution_policy_sha256",
        "qualification_plan_sha256",
        "evaluator_environment_sha256",
        "dataset_id",
    }
    if set(evaluator) != required_evaluator_fields:
        raise CellWorkerError("execution_binding_evaluator_fields_mismatch")
    policy = evaluator.get("execution_policy")
    if not isinstance(policy, Mapping):
        raise CellWorkerError("execution_binding_policy_invalid")
    if stable_sha256(policy) != evaluator.get("execution_policy_sha256"):
        raise CellWorkerError("execution_binding_policy_hash_mismatch")
    if evaluator.get("dataset_id") != spec.dataset_id:
        raise CellWorkerError("execution_binding_dataset_mismatch")


def _identity_environment(spec: CellSpec, binding: Mapping[str, Any]) -> dict[str, str]:
    evaluator = binding["evaluator"]
    return {
        "ORG_OSS_MODE": "formal",
        "ORG_OSS_DATASET": spec.dataset_id,
        "ORG_EXPERIMENT_CONDITION": spec.condition_id,
        "ORG_ACTION_SELECTION_MODE": str(spec.arm_config["action_selection"]),
        "ORG_EXPERIMENT_TARGET_TICK": str(spec.ticks),
        "ORG_EXPERIMENT_PAIRED_SEED": str(spec.seed),
        "ORG_EXPERIMENT_ARM_ID": spec.arm,
        "ORG_CASE_PLAN_FINGERPRINT": spec.fingerprint,
        "ORG_SOURCE_PROVENANCE_FINGERPRINT": spec.source_provenance_fingerprint,
        "ORG_MODEL_BINDING_FINGERPRINT": str(binding["model_binding_sha256"]),
        "ORG_EXECUTION_RESOURCE_BUDGET_FINGERPRINT": spec.resource_budget.fingerprint,
        "RELIC_EVALUATOR_EXPECTED_ENVIRONMENT_HASH": str(
            evaluator["evaluator_environment_sha256"]
        ),
        "RELIC_EVALUATOR_EXPECTED_QUALIFICATION_HASH": str(
            evaluator["qualification_plan_sha256"]
        ),
    }


def _build_world(spec: CellSpec, client: OpenAIOrgLLMClient) -> OrgWorld:
    scenario = oss_time_machine_formal(seed=spec.seed, dataset_id=spec.dataset_id)
    scenario.params.update(
        {
            "experiment_condition": spec.condition_id,
            "experiment_condition_explicit": True,
            "baseline_sprint_ticks": spec.sprint_ticks,
            "approval_mode": spec.approval_mode,
            "mechanism_ablations": list(spec.mechanism_ablations),
            "frozen_resource_budget": spec.resource_budget.to_dict(),
            "experiment_phase": "main_study",
            "arm_id": spec.arm,
            "oss_control": "none",
            "evaluation_perturbation": "none",
            "action_selection_mode": spec.arm_config["action_selection"],
            "reasoning_effort": spec.model_config["reasoning_effort"],
            "run_manifest": spec.document(),
        }
    )
    world = OrgWorld(scenario).build()
    world.run_id = spec.run_id
    world.set_approval_mode(spec.approval_mode)
    world.llm_client = attach_metered_llm_client(world, client)
    world.ensure_action_selection_ready()
    _validate_world_binding(world, spec, require_client=True)
    return world


def _validate_world_binding(world: OrgWorld, spec: CellSpec, *, require_client: bool) -> None:
    condition = world.condition_spec
    checks = {
        "condition_id": (condition.condition_id, spec.condition_id),
        "short_name": (condition.short_name, spec.arm),
        "roster_size": (len(world.agents), int(spec.arm_config["roster_size"])),
        "condition_roster_size": (condition.roster_size, int(spec.arm_config["roster_size"])),
        "profile_conditioning": (
            world.profile_conditioning_enabled,
            bool(spec.arm_config["profile_conditioning"]),
        ),
        "capability_learning": (
            world.capability_learning_enabled,
            bool(spec.arm_config["capability_learning"]),
        ),
        "institutionalization": (
            world.institutionalization_enabled,
            bool(spec.arm_config["institutionalization"]),
        ),
        "temporary_team": (
            world.temporary_team_enabled,
            bool(spec.arm_config["temporary_team"]),
        ),
        "profile_assignment": (
            condition.profile_assignment,
            str(spec.arm_config["profile_assignment"]),
        ),
        "runtime_protocol_binding": (
            world.institutionalization_enabled,
            bool(spec.arm_config["runtime_protocol_binding"]),
        ),
        "organization_reflection_to_institution_path": (
            bool(world.auto_propose),
            bool(spec.arm_config["organization_reflection_to_institution_path"]),
        ),
        "semi_auto_disables_blanket_approval": (
            bool(world.auto_approve),
            False,
        ),
        "shared_workspace": (
            len(world.agents) > 1,
            bool(spec.arm_config["shared_workspace_channels_meetings"]),
        ),
        "approval_mode": (world.approval_mode, spec.approval_mode),
        "sprint_ticks": (world.baseline_sprint_ticks, spec.sprint_ticks),
        "experiment_mode": (world.experiment_mode, "formal"),
        "work_rhythm": (world.time.rhythm_enabled, False),
        "mechanism_ablations": (
            tuple(sorted(world.mechanism_ablations.disabled)),
            tuple(sorted(spec.mechanism_ablations)),
        ),
    }
    if require_client:
        checks["action_selection"] = (
            world.action_selection_mode,
            str(spec.arm_config["action_selection"]),
        )
    for field, (observed, expected) in checks.items():
        if observed != expected:
            raise CellWorkerError(f"world_binding_mismatch:{field}")


def _checkpoint_path(cell_dir: Path, tick: int) -> Path:
    return cell_dir / "private" / "checkpoints" / f"checkpoint-t{tick:06d}.pkl"


def _checkpoint_expected(spec: CellSpec, binding: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "expected_source_provenance_fingerprint": spec.source_provenance_fingerprint,
        "expected_model_binding_fingerprint": str(binding["model_binding_sha256"]),
        "expected_resource_budget_fingerprint": spec.resource_budget.fingerprint,
        "expected_case_plan_fingerprint": spec.fingerprint,
        "expected_target_tick": spec.ticks,
        "expected_seed": spec.seed,
    }


def _validate_checkpoint_metadata(
    metadata: Mapping[str, Any], spec: CellSpec, binding: Mapping[str, Any]
) -> None:
    expected = {
        "seed": spec.seed,
        "experiment_condition": spec.condition_id,
        "action_selection_mode": spec.arm_config["action_selection"],
        "target_tick": spec.ticks,
        "case_plan_fingerprint": spec.fingerprint,
        "source_provenance_fingerprint": spec.source_provenance_fingerprint,
        "model_binding_fingerprint": binding["model_binding_sha256"],
        "resource_budget_fingerprint": spec.resource_budget.fingerprint,
        "approval_mode": spec.approval_mode,
    }
    for field, value in expected.items():
        if metadata.get(field) != value:
            raise CellWorkerError(f"checkpoint_identity_mismatch:{field}")
    meta = metadata.get("meta")
    if not isinstance(meta, Mapping):
        raise CellWorkerError("checkpoint_cell_metadata_missing")
    if meta.get("cell_id") != spec.cell_id or meta.get("cell_spec_sha256") != spec.fingerprint:
        raise CellWorkerError("checkpoint_cell_identity_mismatch")
    tick = int(metadata.get("tick", -1))
    if tick <= 0 or tick > spec.ticks or tick % spec.checkpoint_every:
        raise CellWorkerError("checkpoint_tick_invalid")


def _latest_checkpoint(
    cell_dir: Path, spec: CellSpec, binding: Mapping[str, Any]
) -> tuple[Path, dict[str, Any]] | None:
    candidates: list[tuple[int, Path, dict[str, Any]]] = []
    for path in sorted((cell_dir / "private" / "checkpoints").glob("checkpoint-t*.pkl")):
        try:
            metadata = checkpoint_info(str(path))
        except Exception as exc:
            raise CellWorkerError("checkpoint_verification_failed") from exc
        _validate_checkpoint_metadata(metadata, spec, binding)
        candidates.append((int(metadata["tick"]), path, metadata))
    if not candidates:
        return None
    _, path, metadata = max(candidates, key=lambda item: item[0])
    return path, metadata


def _public_frame(world: OrgWorld) -> dict[str, Any]:
    task_statuses: dict[str, int] = {}
    for task in world.tasks.values():
        status = str(getattr(getattr(task, "status", None), "value", None) or "unknown")
        if status not in {item.value for item in TaskStatus}:
            raise CellWorkerError("public_trace_task_status_forbidden")
        task_statuses[status] = task_statuses.get(status, 0) + 1
    proposal_manager = getattr(world, "proposal_manager", None)
    proposals = getattr(proposal_manager, "proposals", {}) if proposal_manager else {}
    return {
        "tick": int(world.world_tick),
        "counts": {
            "members": len(world.agents),
            "tasks": len(world.tasks),
            "task_statuses": dict(sorted(task_statuses.items())),
            "episodes": len(getattr(world.episode_manager, "episodes", {}) or {}),
            "proposals": len(proposals or {}),
            "protocols": len(world.protocol_registry.protocols),
            "pull_requests": len(world.repo_system.repo.pull_requests),
            "releases": len(getattr(world.repo_system.repo, "releases", {}) or {}),
        },
    }


def _validate_public_frame(frame: Mapping[str, Any]) -> dict[str, Any]:
    if set(frame) != {"tick", "counts"} or not isinstance(frame.get("tick"), int):
        raise CellWorkerError("public_trace_frame_invalid")
    counts = frame.get("counts")
    count_fields = {
        "members",
        "tasks",
        "task_statuses",
        "episodes",
        "proposals",
        "protocols",
        "pull_requests",
        "releases",
    }
    if not isinstance(counts, Mapping) or set(counts) != count_fields:
        raise CellWorkerError("public_trace_counts_invalid")
    for field in count_fields - {"task_statuses"}:
        value = counts.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise CellWorkerError(f"public_trace_count_invalid:{field}")
    statuses = counts.get("task_statuses")
    allowed_statuses = {item.value for item in TaskStatus}
    if not isinstance(statuses, Mapping) or not set(statuses).issubset(allowed_statuses):
        raise CellWorkerError("public_trace_task_statuses_invalid")
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0
        for value in statuses.values()
    ):
        raise CellWorkerError("public_trace_task_status_count_invalid")
    return {
        "tick": int(frame["tick"]),
        "counts": {
            **{field: int(counts[field]) for field in count_fields - {"task_statuses"}},
            "task_statuses": {
                str(key): int(value) for key, value in sorted(statuses.items())
            },
        },
    }


def _write_public_trace(
    cell_dir: Path,
    spec: CellSpec,
    *,
    status: str,
    world: OrgWorld | None = None,
) -> None:
    path = cell_dir / "public" / "trace.json"
    frames: list[dict[str, Any]] = []
    if path.exists():
        prior = _read_json(path)
        expected_fields = {
            "schema_version",
            "projection_profile",
            "cell_id",
            "study",
            "model_label",
            "workload",
            "arm",
            "seed",
            "target_tick",
            "terminal_status",
            "frames",
        }
        if set(prior) != expected_fields:
            raise CellWorkerError("public_trace_fields_mismatch")
        if prior.get("schema_version") != PUBLIC_TRACE_SCHEMA_VERSION:
            raise CellWorkerError("public_trace_schema_mismatch")
        if prior.get("projection_profile") != "relic-public-allowlist-v1":
            raise CellWorkerError("public_trace_projection_profile_mismatch")
        if prior.get("cell_id") != spec.cell_id:
            raise CellWorkerError("public_trace_cell_mismatch")
        prior_frames = prior.get("frames")
        if not isinstance(prior_frames, list):
            raise CellWorkerError("public_trace_frames_invalid")
        frames = [
            _validate_public_frame(frame)
            for frame in prior_frames
            if isinstance(frame, Mapping)
        ]
        if len(frames) != len(prior_frames):
            raise CellWorkerError("public_trace_frame_invalid")
    if world is not None:
        frame = _validate_public_frame(_public_frame(world))
        frames = [existing for existing in frames if existing.get("tick") != frame["tick"]]
        frames.append(frame)
        frames.sort(key=lambda value: int(value["tick"]))
    payload = {
        "schema_version": PUBLIC_TRACE_SCHEMA_VERSION,
        "projection_profile": "relic-public-allowlist-v1",
        "cell_id": spec.cell_id,
        "study": spec.study,
        "model_label": spec.model_config["paper_label"],
        "workload": spec.workload.upper(),
        "arm": spec.arm.upper(),
        "seed": spec.seed,
        "target_tick": spec.ticks,
        "terminal_status": status if status in _TERMINAL_STATUSES else None,
        "frames": frames,
    }
    _atomic_write_json(path, payload, mode=0o644)


def _write_status(
    cell_dir: Path,
    spec: CellSpec,
    *,
    status: str,
    stage: str,
    tick: int,
    started_at: str,
    failure_code: str | None = None,
) -> dict[str, Any]:
    now = _utc_now()
    prior_latest: int | None = None
    existing_path = cell_dir / "public" / "status.json"
    if existing_path.exists():
        existing = _read_json(existing_path)
        raw_latest = existing.get("latest_checkpoint_tick")
        if isinstance(raw_latest, int) and raw_latest > 0:
            prior_latest = raw_latest
    current_latest = int(tick) if tick > 0 and tick % spec.checkpoint_every == 0 else None
    latest_checkpoint_tick = max(
        value for value in (prior_latest, current_latest) if value is not None
    ) if prior_latest is not None or current_latest is not None else None
    payload = {
        "schema_version": PUBLIC_STATUS_SCHEMA_VERSION,
        "cell_id": spec.cell_id,
        "status": status,
        "stage": stage,
        "tick": int(tick),
        "target_tick": spec.ticks,
        "latest_checkpoint_tick": latest_checkpoint_tick,
        "started_at": started_at,
        "updated_at": now,
        "completed_at": now if status == "completed" else None,
        "failure_code": failure_code,
    }
    _atomic_write_json(cell_dir / "public" / "status.json", payload, mode=0o644)
    private = {
        "schema_version": PRIVATE_STATE_SCHEMA_VERSION,
        **payload,
    }
    _atomic_write_json(cell_dir / "private" / "state.json", private)
    return payload


def _started_at(cell_dir: Path) -> str:
    path = cell_dir / "private" / "state.json"
    if path.exists():
        value = _read_json(path).get("started_at")
        if isinstance(value, str) and value:
            return value
    return _utc_now()


def _prior_tick(cell_dir: Path) -> int:
    path = cell_dir / "public" / "status.json"
    if not path.exists():
        return 0
    value = _read_json(path).get("tick")
    return max(0, int(value)) if isinstance(value, int) else 0


def _write_private_diagnostic(cell_dir: Path, *, code: str) -> None:
    path = cell_dir / "private" / "state.json"
    payload = _read_json(path) if path.exists() else {
        "schema_version": PRIVATE_STATE_SCHEMA_VERSION
    }
    diagnostics = payload.get("diagnostics")
    if not isinstance(diagnostics, list):
        diagnostics = []
    if code not in diagnostics:
        diagnostics.append(code)
    payload["diagnostics"] = diagnostics
    _atomic_write_json(path, payload)


def _checkpoint_world(
    world: OrgWorld, cell_dir: Path, spec: CellSpec
) -> tuple[Path, dict[str, Any]]:
    path = _checkpoint_path(cell_dir, world.world_tick)
    receipt = save_world_checkpoint(
        world,
        str(path),
        meta={
            "cell_id": spec.cell_id,
            "run_id": spec.run_id,
            "cell_spec_sha256": spec.fingerprint,
        },
    )
    return path, receipt


def _completed_resume_result(
    cell_dir: Path, spec: CellSpec
) -> CellResult | None:
    status_path = cell_dir / "public" / "status.json"
    if not status_path.exists():
        return None
    status = _read_json(status_path)
    if status.get("status") != "completed":
        return None
    if (
        status.get("schema_version") != PUBLIC_STATUS_SCHEMA_VERSION
        or status.get("cell_id") != spec.cell_id
        or int(status.get("tick", -1)) != spec.ticks
    ):
        raise CellWorkerError("completed_status_invalid")
    binding_path = cell_dir / "private" / "execution-binding.json"
    if not binding_path.exists():
        raise CellWorkerError("completed_execution_binding_missing")
    binding = _read_json(binding_path)
    _validate_execution_binding(spec, binding)
    latest = _latest_checkpoint(cell_dir, spec, binding)
    if latest is None or int(latest[1]["tick"]) != spec.ticks:
        raise CellWorkerError("completed_final_checkpoint_missing")
    record_path = cell_dir / "run-record.json"
    if not record_path.exists() or record_path.is_symlink():
        raise CellWorkerError("completed_run_record_missing")
    record = _read_json(record_path)
    if (
        record.get("schema_version") != "orgenv_experiment_run_v2"
        or record.get("status") != "completed"
        or record.get("replication_id") != spec.cell_id
        or int(record.get("seed", -1)) != spec.seed
    ):
        raise CellWorkerError("completed_run_record_invalid")
    return CellResult(spec.cell_id, cell_dir, "completed", "complete", spec.ticks)


def _write_run_record(
    world: OrgWorld,
    cell_dir: Path,
    spec: CellSpec,
    binding: Mapping[str, Any],
    *,
    started_at: str,
    status: str,
    checkpoint: Mapping[str, Any] | str | Path | None,
    final_evaluator: Any = None,
    failure_reason: str | None = None,
) -> dict[str, Any]:
    model = binding["model"]
    return write_experiment_run_record(
        world,
        cell_dir / "run-record.json",
        final_evaluator=final_evaluator,
        started_at=started_at,
        ended_at=_utc_now(),
        status=status,
        checkpoint=checkpoint,
        failure_reason=failure_reason,
        provider=str(model["provider"]),
        model=str(model["runtime_model"]),
        replication_id=spec.cell_id,
        randomization_block=f"{spec.model}__{spec.workload.upper()}",
        provenance={
            "cell_spec_sha256": spec.fingerprint,
            "execution_binding_sha256": binding["execution_binding_sha256"],
            "source_branch": spec.source["branch"],
            "source_commit": spec.source["commit"],
            "qualification_plan_sha256": binding["evaluator"][
                "qualification_plan_sha256"
            ],
        },
    )


def _failure_code(exc: BaseException, stage: str) -> tuple[str, str]:
    if isinstance(exc, CellWorkerError):
        code = exc.code
    elif isinstance(exc, LLMError):
        code = "model_failure"
    elif isinstance(exc, ExperimentResourceExhausted):
        code = "resource_exhausted"
    elif stage == "evaluation":
        code = "evaluator_failure"
    elif stage == "rollout":
        code = "rollout_failure"
    else:
        code = "infrastructure_failure"
    status = "failed" if code in {"model_failure", "rollout_failure"} else "infra_error"
    return status, code


def run_cell(spec: CellSpec, *, cell_dir: Path | None = None, resume: bool = False) -> CellResult:
    """Run, checkpoint, evaluate, and record exactly one canonical cell."""

    requested = (cell_dir or spec.cell_dir).expanduser().absolute()
    if requested.is_symlink():
        raise CellWorkerError("cell_directory_symlink_forbidden")
    requested.mkdir(parents=True, exist_ok=True, mode=0o700)
    destination = requested.resolve()
    _validate_owned_directory(destination, label="cell_directory")
    with _cell_lock(destination):
        existing = [path for path in destination.iterdir() if path.name != ".cell.lock"]
        if not resume and existing:
            raise CellWorkerError("cell_directory_not_empty")
        _ensure_layout(destination)
        spec_path = destination / "cell-spec.json"
        if resume:
            if not spec_path.exists():
                raise CellWorkerError("resume_cell_spec_missing")
            if spec_path.is_symlink():
                raise CellWorkerError("resume_cell_spec_symlink_forbidden")
            frozen = load_frozen_cell_spec(destination)
            if frozen.fingerprint != spec.fingerprint:
                raise CellWorkerError("resume_cell_spec_mismatch")
        _write_immutable_json(spec_path, spec.document())
        if resume:
            completed = _completed_resume_result(destination, spec)
            if completed is not None:
                return completed
        started_at = _started_at(destination)
        initial_tick = _prior_tick(destination)
        stage = "preflight"
        world: OrgWorld | None = None
        last_checkpoint: Path | None = None
        last_receipt: Mapping[str, Any] | None = None
        try:
            _write_status(
                destination,
                spec,
                status="running",
                stage=stage,
                tick=initial_tick,
                started_at=started_at,
            )
            evaluator_binding = _preflight_evaluator(spec)
            client, model_binding = _build_openai_client(spec)
            binding = _execution_binding(
                spec,
                model_binding=model_binding,
                evaluator_binding=evaluator_binding,
            )
            _validate_execution_binding(spec, binding)
            _write_immutable_json(destination / "private" / "execution-binding.json", binding)
            environment = _identity_environment(spec, binding)
            with _scoped_environment(environment):
                latest = _latest_checkpoint(destination, spec, binding)
                if latest is not None:
                    if not resume:
                        raise CellWorkerError("checkpoint_exists_without_resume")
                    last_checkpoint, metadata = latest
                    world, _ = load_world_checkpoint(
                        str(last_checkpoint),
                        llm_client=client,
                        load_llm=False,
                        **_checkpoint_expected(spec, binding),
                    )
                    validate_formal_oss_world(world, require_formal=True)
                    _validate_world_binding(world, spec, require_client=True)
                    last_receipt = checkpoint_info(str(last_checkpoint))
                else:
                    world = _build_world(spec, client)

                stage = "rollout"
                _write_status(
                    destination,
                    spec,
                    status="running",
                    stage=stage,
                    tick=world.world_tick,
                    started_at=started_at,
                )
                while world.world_tick < spec.ticks:
                    world.step()
                    if world.world_tick % spec.checkpoint_every == 0:
                        last_checkpoint, last_receipt = _checkpoint_world(
                            world, destination, spec
                        )
                        _write_public_trace(
                            destination, spec, status="running", world=world
                        )
                        _write_status(
                            destination,
                            spec,
                            status="running",
                            stage=stage,
                            tick=world.world_tick,
                            started_at=started_at,
                        )
                if last_checkpoint is None or world.world_tick != spec.ticks:
                    raise CellWorkerError("final_checkpoint_missing")

                stage = "evaluation"
                _write_status(
                    destination,
                    spec,
                    status="running",
                    stage=stage,
                    tick=world.world_tick,
                    started_at=started_at,
                )
                artifact = run_final_evaluation(
                    world,
                    output_dir=destination / "private" / "evaluator",
                    run_tag=spec.cell_id,
                )
                if artifact is None:
                    raise CellWorkerError("final_evaluator_not_configured")
                _write_run_record(
                    world,
                    destination,
                    spec,
                    binding,
                    started_at=started_at,
                    status="completed",
                    checkpoint=last_receipt or last_checkpoint,
                    final_evaluator=artifact,
                )
                _write_public_trace(destination, spec, status="completed", world=world)
                _write_status(
                    destination,
                    spec,
                    status="completed",
                    stage="complete",
                    tick=world.world_tick,
                    started_at=started_at,
                )
                return CellResult(
                    spec.cell_id, destination, "completed", "complete", world.world_tick
                )
        except Exception as exc:
            status, code = _failure_code(exc, stage)
            tick = int(getattr(world, "world_tick", initial_tick))
            record_failure = False
            if world is not None and "binding" in locals():
                try:
                    _write_run_record(
                        world,
                        destination,
                        spec,
                        binding,
                        started_at=started_at,
                        status=status,
                        checkpoint=last_receipt or last_checkpoint,
                        failure_reason=code,
                    )
                except Exception:
                    record_failure = True
            _write_public_trace(destination, spec, status=status, world=world)
            _write_status(
                destination,
                spec,
                status=status,
                stage=stage,
                tick=tick,
                started_at=started_at,
                failure_code=code,
            )
            if record_failure:
                _write_private_diagnostic(
                    destination, code="failure_run_record_write_failed"
                )
            if isinstance(exc, CellWorkerError):
                raise
            raise CellWorkerError(code) from exc


def evaluate_cell(*, cell_dir: Path) -> CellResult:
    """Evaluate an already completed rollout without creating an LLM client."""

    requested = Path(cell_dir).expanduser().absolute()
    if requested.is_symlink():
        raise CellWorkerError("cell_directory_symlink_forbidden")
    if not requested.is_dir():
        raise CellWorkerError("cell_directory_missing")
    destination = requested.resolve()
    _validate_owned_directory(destination, label="cell_directory")
    with _cell_lock(destination):
        _ensure_layout(destination)
        spec: CellSpec | None = None
        frozen_binding: dict[str, Any] | None = None
        world: OrgWorld | None = None
        metadata: Mapping[str, Any] | None = None
        started_at = _started_at(destination)
        try:
            if (destination / "cell-spec.json").is_symlink():
                raise CellWorkerError("cell_spec_symlink_forbidden")
            spec = load_frozen_cell_spec(destination)
            binding_path = destination / "private" / "execution-binding.json"
            if not binding_path.exists():
                raise CellWorkerError("execution_binding_missing")
            frozen_binding = _read_json(binding_path)
            _validate_execution_binding(spec, frozen_binding)
            evaluator_binding = _preflight_evaluator(spec)
            if dict(evaluator_binding) != frozen_binding.get("evaluator"):
                raise CellWorkerError("evaluator_binding_changed")
            latest = _latest_checkpoint(destination, spec, frozen_binding)
            if latest is None or int(latest[1]["tick"]) != spec.ticks:
                raise CellWorkerError("final_checkpoint_missing")
            checkpoint_path, metadata = latest
            environment = _identity_environment(spec, frozen_binding)
            with _scoped_environment(environment):
                world, _ = load_world_checkpoint(
                    str(checkpoint_path),
                    llm_client=None,
                    load_llm=False,
                    **_checkpoint_expected(spec, frozen_binding),
                )
                validate_formal_oss_world(world, require_formal=True)
                _validate_world_binding(world, spec, require_client=False)
                artifact = run_final_evaluation(
                    world,
                    output_dir=destination / "private" / "evaluator",
                    run_tag=spec.cell_id,
                )
                if artifact is None:
                    raise CellWorkerError("final_evaluator_not_configured")
                _write_run_record(
                    world,
                    destination,
                    spec,
                    frozen_binding,
                    started_at=started_at,
                    status="completed",
                    checkpoint=metadata,
                    final_evaluator=artifact,
                )
                _write_public_trace(destination, spec, status="completed", world=world)
                _write_status(
                    destination,
                    spec,
                    status="completed",
                    stage="complete",
                    tick=spec.ticks,
                    started_at=started_at,
                )
                return CellResult(
                    spec.cell_id, destination, "completed", "complete", spec.ticks
                )
        except Exception as exc:
            status, code = _failure_code(exc, "evaluation")
            record_failure = False
            if spec is not None and world is not None and frozen_binding is not None:
                try:
                    _write_run_record(
                        world,
                        destination,
                        spec,
                        frozen_binding,
                        started_at=started_at,
                        status=status,
                        checkpoint=metadata,
                        failure_reason=code,
                    )
                except Exception:
                    record_failure = True
            if spec is not None:
                _write_public_trace(destination, spec, status=status, world=world)
                _write_status(
                    destination,
                    spec,
                    status=status,
                    stage="evaluation",
                    tick=int(getattr(world, "world_tick", _prior_tick(destination))),
                    started_at=started_at,
                    failure_code=code,
                )
                if record_failure:
                    _write_private_diagnostic(
                        destination, code="failure_run_record_write_failed"
                    )
            if isinstance(exc, CellWorkerError):
                raise
            raise CellWorkerError(code) from exc


def inspect_cell(cell_dir: Path) -> dict[str, Any]:
    """Read public status and verified checkpoint sidecars; never unpickle."""

    requested = Path(cell_dir).expanduser().absolute()
    if requested.is_symlink():
        raise CellWorkerError("cell_directory_symlink_forbidden")
    if not requested.is_dir():
        raise CellWorkerError("cell_directory_missing")
    destination = requested.resolve()
    _validate_owned_directory(destination, label="cell_directory")
    for relative in ("public", "private", "private/checkpoints"):
        path = destination / relative
        if path.is_symlink():
            raise CellWorkerError(f"cell_subdirectory_symlink_forbidden:{relative}")
        if path.exists():
            _validate_owned_directory(path, label=f"cell_subdirectory:{relative}")
    status_path = destination / "public" / "status.json"
    if not status_path.exists():
        raise CellWorkerError("cell_status_missing")
    status = _read_json(status_path)
    if status.get("schema_version") != PUBLIC_STATUS_SCHEMA_VERSION:
        raise CellWorkerError("cell_status_schema_mismatch")
    checkpoints: list[dict[str, Any]] = []
    checkpoint_root = destination / "private" / "checkpoints"
    if checkpoint_root.is_dir():
        spec_path = destination / "cell-spec.json"
        binding_path = destination / "private" / "execution-binding.json"
        checkpoint_paths = sorted(checkpoint_root.glob("checkpoint-t*.pkl"))
        if checkpoint_paths and (not spec_path.exists() or not binding_path.exists()):
            raise CellWorkerError("checkpoint_identity_context_missing")
        spec = load_frozen_cell_spec(destination) if checkpoint_paths else None
        binding = _read_json(binding_path) if checkpoint_paths else None
        if spec is not None and binding is not None:
            _validate_execution_binding(spec, binding)
        for path in checkpoint_paths:
            metadata = checkpoint_info(str(path))
            assert spec is not None and binding is not None
            _validate_checkpoint_metadata(metadata, spec, binding)
            checkpoints.append(
                {
                    "tick": int(metadata["tick"]),
                    "created": metadata.get("created"),
                    "verified": True,
                }
            )
    return {**status, "checkpoints": checkpoints}


__all__ = [
    "CellResult",
    "CellWorkerError",
    "evaluate_cell",
    "inspect_cell",
    "run_cell",
]
