"""Batch evaluation receipts for a user's newly-created canonical runs.

This module deliberately does *not* discover arbitrary result folders or read
paper-era raw runs.  A caller must name a v2 scheduler manifest, or name the
output root whose exact ``run_manifest.json`` is the v2 scheduler manifest.
The manifest's frozen 120-cell plan is revalidated before a cell can be sent to
the existing single-cell ``relic evaluate`` command.

The resulting ``evaluation_manifest.json`` is a receipt for local, new runs.
It contains only a small aggregate-oriented allow-list; evaluator outcomes,
private traces, model prompts, and evaluator stdout/stderr never enter it.
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

from environments.org_env.experiments.records import validate_experiment_run_record_schema
from environments.org_env.product.substrates.final_evaluation import (
    validate_final_evaluation_evidence,
)
from environments.org_env.product.substrates.loader import load_oss_substrate_spec
from relic.cell_spec import CellSpec, compile_cell_spec, stable_sha256
from relic.cell_worker import (
    CellWorkerError,
    _validate_execution_binding,
    inspect_cell,
)
from relic.main_runner import MainRunnerError, _validate_manifest
from relic.paths import project_root


EVALUATION_MANIFEST_SCHEMA_VERSION = "relic-user-run-evaluation-batch-v1"
_RUN_MANIFEST_NAME = "run_manifest.json"
_EVALUATION_MANIFEST_NAME = "evaluation_manifest.json"
_SELECTION_POLICIES = frozenset({"completed", "failed-evaluation", "all-eligible"})
_MAX_JSON_BYTES = 20 * 1024 * 1024


class UserRunBatchError(RuntimeError):
    """A stable error raised when even a local receipt cannot be safely written."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class UserRunBatchResult:
    """Compact command-facing result; full details live in the receipt."""

    manifest_path: Path
    receipt_path: Path
    status: str
    selected_cells: int
    succeeded_cells: int
    failed_cells: int
    excluded_cells: int
    dry_run: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest": str(self.manifest_path),
            "evaluation_manifest": str(self.receipt_path),
            "status": self.status,
            "selected_cells": self.selected_cells,
            "succeeded_cells": self.succeeded_cells,
            "failed_cells": self.failed_cells,
            "excluded_cells": self.excluded_cells,
            "dry_run": self.dry_run,
        }


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _error_code(exc: BaseException, fallback: str) -> str:
    """Return a bounded, non-secret diagnostic code for a public receipt."""

    if isinstance(exc, (UserRunBatchError, CellWorkerError, MainRunnerError)):
        value = str(getattr(exc, "code", str(exc)))
        if value and len(value) <= 200 and "\n" not in value and "\r" not in value:
            return value
    return fallback


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _has_symlink_component(path: Path) -> bool:
    """Return true when any existing component of an absolute path is a symlink."""

    absolute = path.expanduser().absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        if current.is_symlink():
            return True
    return False


def _read_json_object(path: Path, *, code: str) -> dict[str, Any]:
    """Read a bounded regular JSON object without following a final symlink."""

    if path.is_symlink():
        raise UserRunBatchError(f"{code}_symlink_forbidden")
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except FileNotFoundError as exc:
        raise UserRunBatchError(f"{code}_missing") from exc
    except OSError as exc:
        raise UserRunBatchError(f"{code}_unreadable") from exc
    try:
        details = os.fstat(descriptor)
        if not stat.S_ISREG(details.st_mode) or details.st_size > _MAX_JSON_BYTES:
            raise UserRunBatchError(f"{code}_invalid_file")
        with os.fdopen(descriptor, "r", encoding="utf-8") as handle:
            descriptor = -1
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        raise UserRunBatchError(f"{code}_invalid_json") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if not isinstance(payload, dict):
        raise UserRunBatchError(f"{code}_invalid_object")
    return payload


def _file_sha256(path: Path) -> str:
    """Hash a validated receipt input without retaining its contents."""

    # The manifest must already have passed _read_json_object, so ordinary
    # byte reading here cannot turn a symlink into a trusted input.
    import hashlib

    digest = hashlib.sha256()
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return digest.hexdigest()


def resolve_run_manifest(
    *, manifest_path: Path | None = None, output_root: Path | None = None
) -> Path:
    """Resolve an explicit v2-manifest input without recursive discovery."""

    if (manifest_path is None) == (output_root is None):
        raise UserRunBatchError("exactly_one_manifest_or_output_root_required")
    if manifest_path is not None:
        requested = Path(manifest_path).expanduser().absolute()
        if _has_symlink_component(requested):
            raise UserRunBatchError("run_manifest_symlink_forbidden")
        resolved_manifest = requested.resolve()
        for forbidden in (
            project_root() / "artifacts" / "paper_results",
            project_root() / "reproduction",
        ):
            if _is_within(resolved_manifest, forbidden.resolve()):
                raise UserRunBatchError("historical_or_paper_manifest_input_forbidden")
        return resolved_manifest
    root = Path(output_root).expanduser().absolute()
    if _has_symlink_component(root):
        raise UserRunBatchError("output_root_symlink_forbidden")
    # This is intentionally not rglob/glob: output_root has one exact,
    # documented location for its scheduler manifest.
    resolved_root = root.resolve()
    for forbidden in (
        project_root() / "artifacts" / "paper_results",
        project_root() / "reproduction",
    ):
        if _is_within(resolved_root, forbidden.resolve()):
            raise UserRunBatchError("historical_or_paper_manifest_input_forbidden")
    candidate = resolved_root / _RUN_MANIFEST_NAME
    if candidate.is_symlink():
        raise UserRunBatchError("run_manifest_symlink_forbidden")
    return candidate


def _receipt_path(manifest_path: Path, receipt_directory: Path | None) -> Path:
    directory = (
        Path(receipt_directory).expanduser().absolute()
        if receipt_directory is not None
        else manifest_path.parent
    )
    if _has_symlink_component(directory):
        raise UserRunBatchError("evaluation_receipt_directory_symlink_forbidden")
    resolved = directory.resolve()
    for forbidden in (
        project_root() / "artifacts" / "paper_results",
        project_root() / "reproduction",
    ):
        if _is_within(resolved, forbidden.resolve()):
            raise UserRunBatchError("evaluation_receipt_paper_path_forbidden")
    return resolved / _EVALUATION_MANIFEST_NAME


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically write a local receipt with private-by-default permissions."""

    parent = path.parent
    if parent.is_symlink():
        raise UserRunBatchError("evaluation_manifest_parent_symlink_forbidden")
    try:
        parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    except OSError as exc:
        raise UserRunBatchError("evaluation_manifest_parent_create_failed") from exc
    if not parent.is_dir() or parent.is_symlink():
        raise UserRunBatchError("evaluation_manifest_parent_invalid")
    if path.is_symlink():
        raise UserRunBatchError("evaluation_manifest_symlink_forbidden")
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=parent
        )
    except OSError as exc:
        raise UserRunBatchError("evaluation_manifest_temp_create_failed") from exc
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        if os.name == "posix":
            directory_fd = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    except OSError as exc:
        raise UserRunBatchError("evaluation_manifest_write_failed") from exc
    finally:
        temporary.unlink(missing_ok=True)


def _empty_analysis(
    spec: CellSpec, binding: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, str]]:
    """Build the fixed public allow-list with values filled only after validation."""

    evaluator = binding["evaluator"]
    analysis = {
        "model": spec.model,
        "workload": spec.workload.upper(),
        "arm": spec.arm.upper(),
        "seed": spec.seed,
        "dataset_id": spec.dataset_id,
        "run_record_status": None,
        "llm_usage": {"total_tokens": None},
        "final_evaluation": {
            "candidate_pass_rate": None,
            "causal_fix_count": None,
            "causal_fix_rate": None,
            "unresolved_count": None,
            "regression_count": None,
            "infrastructure_error_count": None,
            "formal_claim_ready": None,
            "status": None,
        },
        "lineage": {
            "source_provenance_sha256": spec.source_provenance_fingerprint,
            "runtime_tree_sha256": spec.runtime_source.get("tree_sha256"),
            # This is the canonical model-config binding, rather than the
            # private, effective-provider binding.  The latter is proved by
            # _validate_execution_binding above but can include deployment
            # details.  Keeping this canonical value lets the aggregator
            # derive it again from compile_cell_spec without importing the
            # private execution binding into a public receipt.
            "model_binding_sha256": spec.model_binding_fingerprint,
            "execution_policy_sha256": evaluator.get("execution_policy_sha256"),
            "qualification_plan_sha256": evaluator.get("qualification_plan_sha256"),
            "evaluator_environment_sha256": evaluator.get("evaluator_environment_sha256"),
            "starter_repo_digest": None,
            "reference_repo_digest": None,
            "hidden_suite_hash": None,
        },
        "evaluator": {
            "artifact_hash": None,
            "result_hash": None,
            "candidate_repo_digest": None,
            "status": None,
        },
    }
    return analysis, {}


def _analysis_reason(reasons: dict[str, str], field: str, reason: str) -> None:
    # Reasons are codes, not exception strings or evaluator output.
    reasons[field] = reason


def _analysis_from_verified_artifacts(
    *, spec: CellSpec, binding: Mapping[str, Any], cell_dir: Path
) -> dict[str, Any]:
    """Extract only the aggregation allow-list from a verified local run.

    The evaluator file is located by its content-addressed name derived from
    the validated run record.  No directory scans are used, and the receipt
    never includes the artifact's raw oracle outcomes or evidence records.
    """

    analysis, reasons = _empty_analysis(spec, binding)
    record_path = cell_dir / "run-record.json"
    try:
        record = _read_json_object(record_path, code="run_record")
        validate_experiment_run_record_schema(record)
    except (UserRunBatchError, ValueError, TypeError) as exc:
        reason = _error_code(exc, "run_record_invalid")
        for field in (
            "run_record_status",
            "llm_usage.total_tokens",
            "final_evaluation",
            "lineage.starter_repo_digest",
            "lineage.reference_repo_digest",
            "lineage.hidden_suite_hash",
        ):
            _analysis_reason(reasons, field, reason)
        analysis["missing_reasons"] = reasons
        return analysis

    if (
        record.get("schema_version") != "orgenv_experiment_run_v2"
        or record.get("replication_id") != spec.cell_id
        or record.get("seed") != spec.seed
        or record.get("pack") != spec.dataset_id
    ):
        for field in (
            "run_record_status",
            "llm_usage.total_tokens",
            "final_evaluation",
            "lineage.starter_repo_digest",
            "lineage.reference_repo_digest",
            "lineage.hidden_suite_hash",
        ):
            _analysis_reason(reasons, field, "run_record_cell_identity_mismatch")
        analysis["missing_reasons"] = reasons
        return analysis

    status = record.get("status")
    analysis["run_record_status"] = status if isinstance(status, str) else None
    if analysis["run_record_status"] is None:
        _analysis_reason(reasons, "run_record_status", "run_record_status_missing")
    usage = record.get("llm_usage")
    total_tokens = usage.get("total_tokens") if isinstance(usage, Mapping) else None
    if isinstance(total_tokens, int) and not isinstance(total_tokens, bool):
        analysis["llm_usage"]["total_tokens"] = total_tokens
    else:
        _analysis_reason(reasons, "llm_usage.total_tokens", "run_record_total_tokens_missing")

    final = record.get("final_evaluation")
    if not isinstance(final, Mapping):
        _analysis_reason(reasons, "final_evaluation", "run_record_final_evaluation_missing")
        for field in (
            "lineage.starter_repo_digest",
            "lineage.reference_repo_digest",
            "lineage.hidden_suite_hash",
        ):
            _analysis_reason(reasons, field, "verified_evaluator_artifact_missing")
        analysis["missing_reasons"] = reasons
        return analysis

    plan_hash = final.get("plan_hash")
    candidate_digest = final.get("candidate_repo_digest")
    if not isinstance(plan_hash, str) or not isinstance(candidate_digest, str):
        _analysis_reason(reasons, "final_evaluation", "run_record_evaluator_locator_invalid")
        analysis["missing_reasons"] = reasons
        return analysis
    evaluator_binding = binding.get("evaluator")
    if not isinstance(evaluator_binding, Mapping) or plan_hash != evaluator_binding.get(
        "qualification_plan_sha256"
    ):
        _analysis_reason(reasons, "final_evaluation", "evaluator_binding_plan_mismatch")
        analysis["missing_reasons"] = reasons
        return analysis
    artifact_path = (
        cell_dir
        / "private"
        / "evaluator"
        / f"final_evaluation_{spec.dataset_id}_{plan_hash}_{candidate_digest}.json"
    )
    try:
        artifact = _read_json_object(artifact_path, code="final_evaluator_artifact")
        dataset_manifest = load_oss_substrate_spec(spec.dataset_id).manifest
        manual_checks_enabled = bool(
            (dataset_manifest.get("evaluation") or {}).get("manual_release_checks") or ()
        )
        verified_artifact = validate_final_evaluation_evidence(
            artifact,
            dataset_id=spec.dataset_id,
            plan_hash=plan_hash,
            candidate_digest=candidate_digest,
            manifest=dataset_manifest,
            manual_checks_enabled=manual_checks_enabled,
        )
    except (UserRunBatchError, ValueError, TypeError, OSError) as exc:
        reason = _error_code(exc, "final_evaluator_artifact_invalid")
        _analysis_reason(reasons, "final_evaluation", reason)
        for field in (
            "lineage.starter_repo_digest",
            "lineage.reference_repo_digest",
            "lineage.hidden_suite_hash",
        ):
            _analysis_reason(reasons, field, reason)
        analysis["missing_reasons"] = reasons
        return analysis

    result = verified_artifact.get("result")
    if not isinstance(result, Mapping):  # Defensive after the validator above.
        _analysis_reason(reasons, "final_evaluation", "final_evaluator_result_missing")
    else:
        for field in analysis["final_evaluation"]:
            value = result.get(field)
            if field == "formal_claim_ready":
                allowed = isinstance(value, bool)
            elif field in {
                "candidate_pass_rate",
                "causal_fix_rate",
            }:
                allowed = isinstance(value, (int, float)) and not isinstance(value, bool)
            elif field in {
                "causal_fix_count",
                "unresolved_count",
                "regression_count",
                "infrastructure_error_count",
            }:
                allowed = isinstance(value, int) and not isinstance(value, bool)
            else:
                allowed = isinstance(value, str)
            if allowed:
                analysis["final_evaluation"][field] = value
            else:
                _analysis_reason(reasons, f"final_evaluation.{field}", "evaluator_field_missing")

    if verified_artifact.get("evaluator_environment_hash") != evaluator_binding.get(
        "evaluator_environment_sha256"
    ):
        _analysis_reason(
            reasons,
            "final_evaluation",
            "evaluator_binding_environment_mismatch",
        )
    else:
        analysis["evaluator"] = {
            "artifact_hash": verified_artifact.get("artifact_hash"),
            "result_hash": result.get("result_hash") if isinstance(result, Mapping) else None,
            "candidate_repo_digest": candidate_digest,
            "status": result.get("status") if isinstance(result, Mapping) else None,
        }

    for field in ("starter_repo_digest", "reference_repo_digest", "hidden_suite_hash"):
        artifact_value = verified_artifact.get(field)
        record_value = record.get(field)
        if isinstance(artifact_value, str) and artifact_value == record_value:
            analysis["lineage"][field] = artifact_value
        else:
            _analysis_reason(reasons, f"lineage.{field}", "evaluator_record_lineage_mismatch")
    analysis["missing_reasons"] = reasons
    return analysis


def _validate_cell_artifacts(
    *,
    raw_cell: Mapping[str, Any],
    runtime_cell: Mapping[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    """Verify identity and checkpoint sidecars without deserializing a checkpoint."""

    try:
        spec = compile_cell_spec(
            model=str(raw_cell.get("model") or ""),
            workload=str(raw_cell.get("workload") or ""),
            arm=str(raw_cell.get("arm") or ""),
            seed=int(raw_cell.get("seed")),
            output_root=output_root,
        )
    except (TypeError, ValueError) as exc:
        raise UserRunBatchError("batch_cell_compile_failed") from exc
    declared_input = Path(str(raw_cell.get("output_path") or "")).expanduser().absolute()
    if _has_symlink_component(declared_input):
        raise UserRunBatchError("batch_cell_directory_symlink_forbidden")
    declared_path = declared_input.resolve()
    if declared_path != spec.cell_dir or not _is_within(declared_path, output_root):
        raise UserRunBatchError("batch_cell_output_path_mismatch")
    if raw_cell.get("cell_id") != spec.cell_id:
        raise UserRunBatchError("batch_cell_identity_mismatch")
    if raw_cell.get("cell_spec_sha256") != spec.fingerprint:
        raise UserRunBatchError("batch_cell_spec_hash_mismatch")
    if raw_cell.get("cell_spec") != spec.document():
        raise UserRunBatchError("batch_cell_spec_mismatch")
    # This is a byte-for-byte check against the manifest before the worker's
    # sidecar verifier is called.  A copied cell-spec cannot retarget a path.
    frozen_spec = _read_json_object(declared_path / "cell-spec.json", code="cell_spec")
    if frozen_spec != spec.document():
        raise UserRunBatchError("batch_frozen_cell_spec_mismatch")
    binding = _read_json_object(
        declared_path / "private" / "execution-binding.json", code="execution_binding"
    )
    try:
        _validate_execution_binding(spec, binding)
        inspected = inspect_cell(declared_path)
    except (CellWorkerError, ValueError, OSError) as exc:
        raise UserRunBatchError(_error_code(exc, "batch_checkpoint_sidecar_invalid")) from exc
    if inspected.get("cell_id") != spec.cell_id:
        raise UserRunBatchError("batch_public_status_identity_mismatch")
    if inspected.get("target_tick") != spec.ticks:
        raise UserRunBatchError("batch_public_status_target_tick_mismatch")
    checkpoints = inspected.get("checkpoints")
    if not isinstance(checkpoints, list) or not any(
        isinstance(item, Mapping)
        and item.get("verified") is True
        and item.get("tick") == spec.ticks
        for item in checkpoints
    ):
        raise UserRunBatchError("batch_final_checkpoint_sidecar_missing")
    return {
        "spec": spec,
        "binding": binding,
        "cell_dir": declared_path,
        "status": inspected,
        "scheduler_status": runtime_cell.get("status"),
        "scheduler_failure_class": runtime_cell.get("failure_class"),
        "checkpoint_validation": {
            "final_checkpoint_sidecar_verified": True,
            "final_tick": spec.ticks,
            "unpickle_performed_by_batch": False,
        },
    }


def _scheduler_preselection_reason(*, policy: str, runtime_cell: Mapping[str, Any]) -> str | None:
    """Reject cells which cannot be eligible without opening any artifacts.

    This deliberately comes before checkpoint or status-file inspection.  A
    normal partial 120-cell run has many pending directories, and a batch
    receipt for the completed subset must remain possible without treating
    those absent directories as corrupt artifacts.
    """

    scheduler_status = runtime_cell.get("status")
    failure_class = runtime_cell.get("failure_class")
    if scheduler_status == "completed":
        if policy in {"completed", "all-eligible"}:
            return None
        return "selection_policy_excludes_completed"
    if scheduler_status == "failed":
        if failure_class != "evaluator_failure":
            return "scheduler_failure_not_evaluator"
        if policy in {"failed-evaluation", "all-eligible"}:
            return None
        return "selection_policy_excludes_failed_evaluation"
    return "scheduler_cell_not_terminal"


def _selection_reason(
    *, policy: str, runtime_cell: Mapping[str, Any], status: Mapping[str, Any]
) -> str | None:
    """Check public pre-evaluation state for a scheduler-eligible cell."""

    preselection = _scheduler_preselection_reason(policy=policy, runtime_cell=runtime_cell)
    if preselection is not None:
        return preselection
    scheduler_status = runtime_cell.get("status")
    failure_class = runtime_cell.get("failure_class")
    public_status = status.get("status")
    stage = status.get("stage")
    completed = (
        scheduler_status == "completed" and public_status == "completed" and stage == "complete"
    )
    failed_evaluation = (
        scheduler_status == "failed"
        and failure_class == "evaluator_failure"
        and public_status in {"failed", "infra_error"}
        and stage == "evaluation"
    )
    if completed or failed_evaluation:
        return None
    return "scheduler_public_status_not_eligible"


def _receipt_hash(payload: Mapping[str, Any]) -> str:
    return stable_sha256({key: value for key, value in payload.items() if key != "receipt_sha256"})


def _write_receipt(path: Path, payload: dict[str, Any]) -> None:
    payload["receipt_sha256"] = _receipt_hash(payload)
    _atomic_write_json(path, payload)


def evaluate_user_run_batch(
    *,
    manifest_path: Path | None = None,
    output_root: Path | None = None,
    receipt_directory: Path | None = None,
    selection: str = "completed",
    dry_run: bool = False,
) -> UserRunBatchResult:
    """Serially evaluate eligible cells from one user's frozen v2 run manifest.

    This function does not edit ``run_manifest.json`` or any scheduler runtime
    field.  It always attempts to persist an ``evaluation_manifest.json`` once
    the target location is known; individual cell failures are recorded there
    and do not prevent later eligible cells from being attempted.
    """

    manifest = resolve_run_manifest(manifest_path=manifest_path, output_root=output_root)
    receipt_path = _receipt_path(manifest, receipt_directory)
    receipt: dict[str, Any] = {
        "schema_version": EVALUATION_MANIFEST_SCHEMA_VERSION,
        "created_at": _utc_now(),
        "status": "failed",
        "dry_run": bool(dry_run),
        "selection": {"policy": selection, "eligible": [], "excluded": []},
        "evaluations": [],
        "cells": [],
        "failures": [],
        "paper_snapshot_inputs": [],
        "run_origin": {
            "kind": "user_new_run",
            "historical_author_run_inputs": [],
            "historical_raw_run_inputs": [],
        },
        "trusted_checkpoint_boundary": {
            "local_trusted_checkpoints_only": True,
            "warning": (
                "The evaluation subprocess deserializes checkpoints. Run this only on "
                "locally created, trusted checkpoints; do not evaluate downloaded or "
                "otherwise untrusted checkpoint files."
            ),
        },
        "scheduler_runtime_mutation": "batch_never_writes_scheduler_runtime",
    }
    try:
        if selection not in _SELECTION_POLICIES:
            raise UserRunBatchError("batch_selection_policy_invalid")
        payload = _read_json_object(manifest, code="run_manifest")
        try:
            _validate_manifest(payload)
        except MainRunnerError as exc:
            raise UserRunBatchError(_error_code(exc, "run_manifest_invalid")) from exc
        plan = payload["plan"]
        runtime = payload["runtime"]
        plan_root = Path(str(plan["output_root"])).expanduser().resolve()
        if output_root is not None and plan_root != Path(output_root).expanduser().resolve():
            raise UserRunBatchError("output_root_manifest_mismatch")
        receipt["manifest_snapshot"] = {
            "path": str(manifest),
            "sha256": _file_sha256(manifest),
            "schema_version": payload["schema_version"],
            "plan_sha256": plan["plan_sha256"],
        }
        receipt["source_snapshot"] = dict(plan["source"])
        manifest_origin = runtime["run_origin"]
        receipt_manifest_origin = {
            **dict(manifest_origin),
            "historical_author_run_inputs": [],
            "historical_raw_run_inputs": [],
        }
        receipt["run_origin"].update(
            {
                "instance_id": manifest_origin["instance_id"],
                "created_at": manifest_origin["created_at"],
                "manifest_sha256": receipt["manifest_snapshot"]["sha256"],
            }
        )
        receipt["manifest"] = {
            "schema_version": payload["schema_version"],
            "study": plan["study"],
            "model": plan["model"],
            # The original scheduler plan contains this value.  It is kept so
            # aggregation can reconstruct (rather than merely trust) every
            # canonical cell and validate the original plan digest.
            "output_root": str(plan_root),
            "plan_sha256": plan["plan_sha256"],
            "manifest_snapshot_sha256": receipt["manifest_snapshot"]["sha256"],
            "runtime_revision": runtime["revision"],
            "run_origin": receipt_manifest_origin,
            "upstream_source": dict(plan["source"]),
            "plan_cells": [
                {
                    "cell_id": cell["cell_id"],
                    "model": cell["model"],
                    "workload": cell["workload"],
                    "arm": cell["arm"],
                    "seed": cell["seed"],
                    "cell_spec_sha256": cell["cell_spec_sha256"],
                }
                for cell in plan["cells"]
            ],
        }
        runtime_before = stable_sha256(runtime)
        receipt["scheduler_runtime_before_sha256"] = runtime_before

        for raw_cell in plan["cells"]:
            cell_id = str(raw_cell["cell_id"])
            runtime_cell = runtime["cells"][cell_id]
            cell_receipt: dict[str, Any] = {
                "cell_id": cell_id,
                "model": raw_cell["model"],
                "workload": raw_cell["workload"],
                "arm": raw_cell["arm"],
                "seed": raw_cell["seed"],
                "cell_spec_sha256": raw_cell["cell_spec_sha256"],
                "scheduler_status": runtime_cell.get("status"),
                "scheduler_failure_class": runtime_cell.get("failure_class"),
            }
            # Never touch cell files unless the scheduler state and requested
            # policy make a cell potentially evaluator-eligible.  In
            # particular, this makes an ordinary incomplete run receipt
            # aggregate-able with --allow-partial.
            preselection_reason = _scheduler_preselection_reason(
                policy=selection, runtime_cell=runtime_cell
            )
            if preselection_reason is not None:
                receipt["selection"]["excluded"].append(
                    {"cell_id": cell_id, "reason": preselection_reason}
                )
                cell_receipt.update(outcome="not_selected", reason_code=preselection_reason)
                receipt["cells"].append(cell_receipt)
                continue
            try:
                verified = _validate_cell_artifacts(
                    raw_cell=raw_cell, runtime_cell=runtime_cell, output_root=plan_root
                )
            except (UserRunBatchError, OSError) as exc:
                reason = _error_code(exc, "cell_artifact_invalid")
                receipt["selection"]["excluded"].append({"cell_id": cell_id, "reason": reason})
                cell_receipt.update(outcome="invalid", reason_code=reason)
                receipt["cells"].append(cell_receipt)
                if runtime_cell.get("status") in {"completed", "failed"}:
                    receipt["failures"].append({"cell_id": cell_id, "reason": reason})
                continue
            reason = _selection_reason(
                policy=selection, runtime_cell=runtime_cell, status=verified["status"]
            )
            cell_receipt.update(
                public_pre_status=verified["status"].get("status"),
                public_pre_stage=verified["status"].get("stage"),
            )
            if reason is not None:
                receipt["selection"]["excluded"].append({"cell_id": cell_id, "reason": reason})
                cell_receipt.update(outcome="not_selected", reason_code=reason)
                receipt["cells"].append(cell_receipt)
                continue
            receipt["selection"]["eligible"].append(
                {
                    "cell_id": cell_id,
                    "cell_dir": str(verified["cell_dir"]),
                    "scheduler_status": verified["scheduler_status"],
                    "scheduler_failure_class": verified["scheduler_failure_class"],
                    "public_pre_status": verified["status"].get("status"),
                    "public_pre_stage": verified["status"].get("stage"),
                    "checkpoint_validation": verified["checkpoint_validation"],
                }
            )
            if dry_run:
                receipt["evaluations"].append(
                    {"cell_id": cell_id, "status": "dry_run", "return_code": None}
                )
                cell_receipt.update(outcome="dry_run", reason_code=None)
                receipt["cells"].append(cell_receipt)
                continue
            try:
                completed = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "relic.cli",
                        "evaluate",
                        "--cell-dir",
                        str(verified["cell_dir"]),
                    ],
                    cwd=project_root(),
                    env=dict(os.environ),
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
            except OSError:
                failure = {"cell_id": cell_id, "reason": "evaluator_subprocess_launch_failed"}
                receipt["evaluations"].append({**failure, "status": "failed", "return_code": None})
                receipt["failures"].append(failure)
                cell_receipt.update(outcome="failed", reason_code=failure["reason"])
                receipt["cells"].append(cell_receipt)
                continue
            if completed.returncode != 0:
                failure = {"cell_id": cell_id, "reason": "evaluator_subprocess_failed"}
                receipt["evaluations"].append(
                    {**failure, "status": "failed", "return_code": int(completed.returncode)}
                )
                receipt["failures"].append(failure)
                cell_receipt.update(outcome="failed", reason_code=failure["reason"])
                receipt["cells"].append(cell_receipt)
                continue
            try:
                # Recheck identities/sidecars after the child has written its
                # evaluation result.  Do not reapply the pre-child scheduler
                # selection rule: scheduler runtime intentionally remains old.
                after = _validate_cell_artifacts(
                    raw_cell=raw_cell, runtime_cell=runtime_cell, output_root=plan_root
                )
                if (
                    after["status"].get("status") != "completed"
                    or after["status"].get("stage") != "complete"
                ):
                    raise UserRunBatchError("evaluator_success_status_not_completed")
                analysis = _analysis_from_verified_artifacts(
                    spec=after["spec"], binding=after["binding"], cell_dir=after["cell_dir"]
                )
            except (UserRunBatchError, OSError) as exc:
                failure = {
                    "cell_id": cell_id,
                    "reason": _error_code(exc, "post_evaluation_invalid"),
                }
                receipt["evaluations"].append(
                    {**failure, "status": "failed", "return_code": int(completed.returncode)}
                )
                receipt["failures"].append(failure)
                cell_receipt.update(outcome="failed", reason_code=failure["reason"])
                receipt["cells"].append(cell_receipt)
                continue
            receipt["evaluations"].append(
                {
                    "cell_id": cell_id,
                    "status": "completed",
                    "return_code": int(completed.returncode),
                    "analysis": analysis,
                }
            )
            cell_receipt.update(
                outcome="evaluated",
                reason_code=None,
                analysis=analysis,
                evaluator=dict(analysis["evaluator"]),
            )
            receipt["cells"].append(cell_receipt)

        # Re-read only the scheduler metadata to document that this batch did
        # not update it.  This also catches a concurrent scheduler before a
        # receipt is later used for aggregation.
        after_payload = _read_json_object(manifest, code="run_manifest")
        runtime_after = after_payload.get("runtime")
        receipt["scheduler_runtime_after_sha256"] = (
            stable_sha256(runtime_after) if isinstance(runtime_after, Mapping) else None
        )
        if receipt["scheduler_runtime_after_sha256"] != runtime_before:
            receipt["scheduler_runtime_unchanged"] = False
            receipt["failures"].append(
                {"cell_id": None, "reason": "scheduler_runtime_changed_during_batch"}
            )
        else:
            receipt["scheduler_runtime_unchanged"] = True

        attempted_failures = len(receipt["failures"])
        if dry_run:
            receipt["status"] = "dry_run"
        elif attempted_failures:
            receipt["status"] = "completed_with_failures"
        else:
            receipt["status"] = "completed"
    except (UserRunBatchError, OSError) as exc:
        receipt["failures"].append({"cell_id": None, "reason": _error_code(exc, "batch_failed")})
        receipt["status"] = "failed"
    finally:
        receipt["completed_at"] = _utc_now()
        _write_receipt(receipt_path, receipt)

    selected = len(receipt["selection"]["eligible"])
    succeeded = sum(item.get("status") == "completed" for item in receipt["evaluations"])
    failed = len(receipt["failures"])
    excluded = len(receipt["selection"]["excluded"])
    return UserRunBatchResult(
        manifest_path=manifest,
        receipt_path=receipt_path,
        status=str(receipt["status"]),
        selected_cells=selected,
        succeeded_cells=succeeded,
        failed_cells=failed,
        excluded_cells=excluded,
        dry_run=bool(dry_run),
    )


__all__ = [
    "EVALUATION_MANIFEST_SCHEMA_VERSION",
    "UserRunBatchError",
    "UserRunBatchResult",
    "evaluate_user_run_batch",
    "resolve_run_manifest",
]
