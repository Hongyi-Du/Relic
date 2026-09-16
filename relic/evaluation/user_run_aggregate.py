"""Auditable summaries for evaluation receipts from user-created runs.

This module never reads checkpoints, raw run directories, the canonical paper
snapshot, or historical author outputs.  Its only accepted input is the
bounded, public-safe receipt written by :mod:`relic.evaluation.user_run_batch`.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from environments.org_env.experiments.statistics import (
    RunObservation,
    build_paired_units,
    paired_block_bootstrap_interval,
    paired_block_weighted_mean,
    validate_observations,
)
from relic.cell_spec import (
    SOURCE_BRANCH,
    SOURCE_COMMIT,
    SOURCE_GIT_TREE,
    SOURCE_REPOSITORY,
    CellSpec,
    compile_cell_spec,
    stable_sha256,
)
from relic.evaluation.user_run_batch import _scheduler_preselection_reason, _selection_reason
from relic.paths import project_root

USER_RUN_AGGREGATE_SCHEMA_VERSION = "relic-user-run-aggregate-v1"
EVALUATION_BATCH_SCHEMA_VERSION = "relic-user-run-evaluation-batch-v1"
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 1729
CANONICAL_MODELS = ("gpt-5.6-terra", "claude-opus-4.6")
CANONICAL_WORKLOADS = tuple(f"W{index:02d}" for index in range(1, 11))
CANONICAL_ARMS = ("B0", "B1", "B2", "B3")
CANONICAL_SEEDS = (1401, 2711, 4013)
_MAX_RECEIPT_BYTES = 32 * 1024 * 1024
_PAPER_SCORING_METRICS = (
    "complete_contracts",
    "held_out_cases",
    "exposed_cases",
    "evaluator_confirmed_seeded_issues",
    "tokens_per_confirmed_issue",
)
_LINEAGE_FIELDS = (
    "source_provenance_sha256",
    "runtime_tree_sha256",
    "model_binding_sha256",
    "execution_policy_sha256",
    "qualification_plan_sha256",
    "evaluator_environment_sha256",
    "starter_repo_digest",
    "reference_repo_digest",
    "hidden_suite_hash",
)
_SOURCE_FIELDS = ("repository", "branch", "commit", "git_tree")
_HISTORICAL_INPUT_FIELDS = (
    "historical_author_run_inputs",
    "historical_raw_run_inputs",
)
_FINAL_EVALUATOR_STATUSES = frozenset(
    {
        "passed",
        "incomplete",
        "failed",
        "timeout",
        "infra_error",
        "blocked",
        "blocked_invalid_plan",
        "not_run",
    }
)
_AGGREGATABLE_EVALUATOR_STATUSES = frozenset({"passed", "incomplete"})
_SELECTION_POLICIES = frozenset({"completed", "failed-evaluation", "all-eligible"})
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_RUN_ORIGIN_INSTANCE_RE = re.compile(r"[0-9a-f]{32}\Z")
_SCHEDULER_STATUSES = frozenset(
    {"pending", "launching", "running", "completed", "failed", "interrupted"}
)
_SCHEDULER_FAILURE_CLASSES = frozenset(
    {None, "model_failure", "evaluator_failure", "infrastructure_failure"}
)
_TOP_LEVEL_RECEIPT_KEYS = frozenset(
    {
        "schema_version",
        "created_at",
        "completed_at",
        "status",
        "dry_run",
        "selection",
        "evaluations",
        "cells",
        "failures",
        "paper_snapshot_inputs",
        "run_origin",
        "trusted_checkpoint_boundary",
        "scheduler_runtime_mutation",
        "scheduler_runtime_before_sha256",
        "scheduler_runtime_after_sha256",
        "scheduler_runtime_unchanged",
        "manifest_snapshot",
        "source_snapshot",
        "manifest",
        "receipt_sha256",
    }
)
_ROOT_ORIGIN_KEYS = frozenset(
    {
        "kind",
        "instance_id",
        "created_at",
        "manifest_sha256",
        "historical_author_run_inputs",
        "historical_raw_run_inputs",
    }
)
_MANIFEST_ORIGIN_KEYS = frozenset(
    {
        "kind",
        "instance_id",
        "created_at",
        "historical_author_run_inputs",
        "historical_raw_run_inputs",
    }
)
_MANIFEST_SNAPSHOT_KEYS = frozenset({"path", "sha256", "schema_version", "plan_sha256"})
_RECEIPT_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "study",
        "model",
        "output_root",
        "plan_sha256",
        "manifest_snapshot_sha256",
        "runtime_revision",
        "run_origin",
        "upstream_source",
        "plan_cells",
    }
)
_PLAN_CELL_KEYS = frozenset({"cell_id", "model", "workload", "arm", "seed", "cell_spec_sha256"})
_CELL_BASE_KEYS = frozenset(
    {
        "cell_id",
        "model",
        "workload",
        "arm",
        "seed",
        "cell_spec_sha256",
        "scheduler_status",
        "scheduler_failure_class",
        "outcome",
        "reason_code",
    }
)
_EVALUATED_CELL_KEYS = _CELL_BASE_KEYS | frozenset(
    {"public_pre_status", "public_pre_stage", "analysis", "evaluator"}
)
_NOT_SELECTED_WITH_PUBLIC_KEYS = _CELL_BASE_KEYS | frozenset(
    {"public_pre_status", "public_pre_stage"}
)
_SELECTION_KEYS = frozenset({"policy", "eligible", "excluded"})
_ELIGIBLE_SELECTION_KEYS = frozenset(
    {
        "cell_id",
        "cell_dir",
        "scheduler_status",
        "scheduler_failure_class",
        "public_pre_status",
        "public_pre_stage",
        "checkpoint_validation",
    }
)
_EXCLUDED_SELECTION_KEYS = frozenset({"cell_id", "reason"})
_CHECKPOINT_VALIDATION_KEYS = frozenset(
    {"final_checkpoint_sidecar_verified", "final_tick", "unpickle_performed_by_batch"}
)
_EVALUATION_KEYS = frozenset({"cell_id", "status", "return_code", "analysis"})
_ANALYSIS_KEYS = frozenset(
    {
        "model",
        "workload",
        "arm",
        "seed",
        "dataset_id",
        "run_record_status",
        "llm_usage",
        "final_evaluation",
        "lineage",
        "evaluator",
        "missing_reasons",
    }
)
_USAGE_KEYS = frozenset({"total_tokens"})
_FINAL_EVALUATION_KEYS = frozenset(
    {
        "candidate_pass_rate",
        "causal_fix_count",
        "causal_fix_rate",
        "unresolved_count",
        "regression_count",
        "infrastructure_error_count",
        "formal_claim_ready",
        "status",
    }
)
_EVALUATOR_KEYS = frozenset({"artifact_hash", "result_hash", "candidate_repo_digest", "status"})
_MISSING_REASON_KEYS = frozenset(
    {
        "run_record_status",
        "llm_usage.total_tokens",
        "final_evaluation",
        "final_evaluation.candidate_pass_rate",
        "final_evaluation.causal_fix_count",
        "final_evaluation.causal_fix_rate",
        "final_evaluation.unresolved_count",
        "final_evaluation.regression_count",
        "final_evaluation.infrastructure_error_count",
        "final_evaluation.formal_claim_ready",
        "final_evaluation.status",
        "lineage.starter_repo_digest",
        "lineage.reference_repo_digest",
        "lineage.hidden_suite_hash",
    }
)


class UserRunAggregateError(RuntimeError):
    """A stable, non-secret aggregate failure."""

    def __init__(self, code: str):
        self.code = str(code)
        super().__init__(self.code)


@dataclass(frozen=True)
class UserRunAggregateResult:
    json_path: Path
    markdown_path: Path
    analysis_status: str
    expected_cells: int
    evaluated_cells: int
    aggregate_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "json_path": str(self.json_path),
            "markdown_path": str(self.markdown_path),
            "analysis_status": self.analysis_status,
            "expected_cells": self.expected_cells,
            "evaluated_cells": self.evaluated_cells,
            "aggregate_sha256": self.aggregate_sha256,
        }


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _has_symlink_component(path: Path) -> bool:
    absolute = path.expanduser().absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        if current.is_symlink():
            return True
    return False


def _read_receipt(path: Path) -> tuple[dict[str, Any], str]:
    requested = path.expanduser().absolute()
    if _has_symlink_component(requested):
        raise UserRunAggregateError("evaluation_manifest_symlink_forbidden")
    resolved = requested.resolve()
    for forbidden in (
        project_root() / "artifacts" / "paper_results",
        project_root() / "reproduction",
    ):
        try:
            resolved.relative_to(forbidden.resolve())
        except ValueError:
            continue
        raise UserRunAggregateError("paper_artifact_input_forbidden")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(requested, flags)
    except OSError as exc:
        raise UserRunAggregateError("evaluation_manifest_unreadable") from exc
    try:
        details = os.fstat(descriptor)
        if not stat.S_ISREG(details.st_mode) or details.st_size > _MAX_RECEIPT_BYTES:
            raise UserRunAggregateError("evaluation_manifest_invalid_file")
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            body = handle.read(_MAX_RECEIPT_BYTES + 1)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if len(body) > _MAX_RECEIPT_BYTES:
        raise UserRunAggregateError("evaluation_manifest_too_large")
    try:
        payload = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise UserRunAggregateError("evaluation_manifest_invalid_json") from exc
    if not isinstance(payload, dict):
        raise UserRunAggregateError("evaluation_manifest_invalid")
    return payload, hashlib.sha256(body).hexdigest()


def _canonical_output_directory(path: Path) -> Path:
    requested = path.expanduser().absolute()
    if _has_symlink_component(requested):
        raise UserRunAggregateError("aggregate_output_symlink_forbidden")
    resolved = requested.resolve()
    forbidden = (
        project_root() / "artifacts" / "paper_results",
        project_root() / "reproduction",
    )
    for root in forbidden:
        try:
            resolved.relative_to(root.resolve())
        except ValueError:
            continue
        raise UserRunAggregateError("aggregate_output_reserved_for_paper_artifacts")
    return resolved


def _as_finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _identity(cell: Mapping[str, Any]) -> tuple[str, str, str, int] | None:
    analysis = cell.get("analysis")
    source = analysis if isinstance(analysis, Mapping) else cell
    try:
        identity = (
            str(source["model"]),
            str(source["workload"]).upper(),
            str(source["arm"]).upper(),
            int(source["seed"]),
        )
    except (KeyError, TypeError, ValueError):
        return None
    model, workload, arm, seed = identity
    if (
        model not in CANONICAL_MODELS
        or workload not in CANONICAL_WORKLOADS
        or arm not in CANONICAL_ARMS
        or seed not in CANONICAL_SEEDS
    ):
        return None
    expected_id = f"{model}__{workload}__{arm}__seed{seed}"
    if cell.get("cell_id") != expected_id:
        return None
    return identity


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _expected_source() -> dict[str, str]:
    return {
        "repository": SOURCE_REPOSITORY,
        "branch": SOURCE_BRANCH,
        "commit": SOURCE_COMMIT,
        "git_tree": SOURCE_GIT_TREE,
    }


def _identity_from_plan_cell(raw: Mapping[str, Any]) -> tuple[str, str, str, int] | None:
    try:
        identity = (
            str(raw["model"]),
            str(raw["workload"]).upper(),
            str(raw["arm"]).upper(),
            int(raw["seed"]),
        )
    except (KeyError, TypeError, ValueError):
        return None
    model, workload, arm, seed = identity
    if (
        model not in CANONICAL_MODELS
        or workload not in CANONICAL_WORKLOADS
        or arm not in CANONICAL_ARMS
        or seed not in CANONICAL_SEEDS
    ):
        return None
    if raw.get("cell_id") != f"{model}__{workload}__{arm}__seed{seed}":
        return None
    return identity


def _expected_model_identities(model: str) -> set[tuple[str, str, str, int]]:
    return {
        (model, workload, arm, seed)
        for workload in CANONICAL_WORKLOADS
        for arm in CANONICAL_ARMS
        for seed in CANONICAL_SEEDS
    }


def _validate_empty_historical_inputs(origin: Mapping[str, Any], *, scope: str) -> None:
    if origin.get("kind") != "user_new_run":
        raise UserRunAggregateError(f"{scope}_not_user_new_run")
    if not _RUN_ORIGIN_INSTANCE_RE.fullmatch(str(origin.get("instance_id") or "")):
        raise UserRunAggregateError(f"{scope}_instance_id_invalid")
    if not isinstance(origin.get("created_at"), str) or not origin["created_at"]:
        raise UserRunAggregateError(f"{scope}_created_at_invalid")
    for field in _HISTORICAL_INPUT_FIELDS:
        if origin.get(field) != []:
            raise UserRunAggregateError(f"{scope}_historical_input_forbidden")


def _require_exact_keys(value: Any, expected: frozenset[str], *, code: str) -> Mapping[str, Any]:
    """Require a public receipt object to contain exactly its safe schema.

    A receipt is deliberately a narrow, aggregate-only boundary.  Accepting
    arbitrary future keys here would make it possible to smuggle raw evidence,
    historical rows, prompts, or other non-release artifacts through a
    self-hashed JSON document.
    """

    if not isinstance(value, Mapping) or set(value) != expected:
        raise UserRunAggregateError(code)
    return value


def _validate_analysis(
    analysis: Mapping[str, Any],
    *,
    identity: tuple[str, str, str, int],
    spec: CellSpec,
) -> None:
    _require_exact_keys(
        analysis, _ANALYSIS_KEYS, code="evaluation_manifest_analysis_fields_invalid"
    )
    model, workload, arm, seed = identity
    if (
        analysis.get("model") != model
        or analysis.get("workload") != workload
        or analysis.get("arm") != arm
        or analysis.get("seed") != seed
        or analysis.get("dataset_id") != spec.dataset_id
        or analysis.get("run_record_status") != "completed"
    ):
        raise UserRunAggregateError("evaluation_manifest_analysis_identity_invalid")
    usage = _require_exact_keys(
        analysis.get("llm_usage"), _USAGE_KEYS, code="evaluation_manifest_analysis_usage_invalid"
    )
    tokens = usage.get("total_tokens")
    if tokens is not None and (
        isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 0
    ):
        raise UserRunAggregateError("evaluation_manifest_analysis_usage_invalid")
    final = _require_exact_keys(
        analysis.get("final_evaluation"),
        _FINAL_EVALUATION_KEYS,
        code="evaluation_manifest_analysis_final_invalid",
    )
    status = final.get("status")
    if status is not None and status not in _FINAL_EVALUATOR_STATUSES:
        raise UserRunAggregateError("evaluation_manifest_analysis_final_status_invalid")
    for field in (
        "causal_fix_count",
        "unresolved_count",
        "regression_count",
        "infrastructure_error_count",
    ):
        value = final.get(field)
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 0
        ):
            raise UserRunAggregateError("evaluation_manifest_analysis_final_metric_invalid")
    for field in ("candidate_pass_rate", "causal_fix_rate"):
        value = _as_finite(final.get(field))
        if final.get(field) is not None and (value is None or not 0.0 <= value <= 1.0):
            raise UserRunAggregateError("evaluation_manifest_analysis_final_metric_invalid")
    if final.get("formal_claim_ready") is not None and not isinstance(
        final.get("formal_claim_ready"), bool
    ):
        raise UserRunAggregateError("evaluation_manifest_analysis_final_metric_invalid")
    evaluator = _require_exact_keys(
        analysis.get("evaluator"),
        _EVALUATOR_KEYS,
        code="evaluation_manifest_analysis_evaluator_invalid",
    )
    evaluator_status = evaluator.get("status")
    if evaluator_status is not None and evaluator_status not in _FINAL_EVALUATOR_STATUSES:
        raise UserRunAggregateError("evaluation_manifest_analysis_evaluator_status_invalid")
    if evaluator_status != status:
        raise UserRunAggregateError("evaluation_manifest_analysis_evaluator_status_mismatch")
    for field in ("artifact_hash", "result_hash", "candidate_repo_digest"):
        value = evaluator.get(field)
        if value is not None and not _is_sha256(value):
            raise UserRunAggregateError("evaluation_manifest_analysis_evaluator_hash_invalid")
    lineage = _require_exact_keys(
        analysis.get("lineage"),
        frozenset(_LINEAGE_FIELDS),
        code="evaluation_manifest_analysis_lineage_invalid",
    )
    for field in _LINEAGE_FIELDS:
        value = lineage.get(field)
        if value is not None and not _is_sha256(value):
            raise UserRunAggregateError("evaluation_manifest_analysis_lineage_invalid")
    expected_derivable = {
        "source_provenance_sha256": spec.source_provenance_fingerprint,
        "runtime_tree_sha256": spec.runtime_source.get("tree_sha256"),
        "model_binding_sha256": spec.model_binding_fingerprint,
    }
    if any(lineage.get(field) != value for field, value in expected_derivable.items()):
        raise UserRunAggregateError("evaluation_manifest_analysis_derived_lineage_mismatch")
    reasons = analysis.get("missing_reasons")
    if (
        not isinstance(reasons, Mapping)
        or not set(reasons).issubset(_MISSING_REASON_KEYS)
        or not all(
            isinstance(key, str)
            and isinstance(value, str)
            and 0 < len(value) <= 200
            and "\n" not in value
            and "\r" not in value
            for key, value in reasons.items()
        )
    ):
        raise UserRunAggregateError("evaluation_manifest_analysis_reasons_invalid")


def _rebuild_and_validate_plan(manifest: Mapping[str, Any]) -> dict[str, CellSpec]:
    """Reconstruct the frozen scheduler plan from canonical current configs.

    The batch receipt does not copy the source manifest's raw cell documents.
    Instead it holds exactly enough plan identity data to compile all 120
    cells with pack hashing disabled, re-create the original ``plan`` object,
    and verify its digest.  This keeps aggregation independent of arbitrary
    user directories while still binding it to canonical release code.
    """

    _require_exact_keys(
        manifest, _RECEIPT_MANIFEST_KEYS, code="evaluation_manifest_manifest_fields_invalid"
    )
    output_root_raw = manifest.get("output_root")
    if not isinstance(output_root_raw, str) or not output_root_raw:
        raise UserRunAggregateError("evaluation_manifest_plan_invalid")
    output_root = Path(output_root_raw).expanduser()
    if not output_root.is_absolute() or _has_symlink_component(output_root):
        raise UserRunAggregateError("evaluation_manifest_plan_output_root_invalid")
    output_root = output_root.resolve()
    if str(output_root) != output_root_raw:
        raise UserRunAggregateError("evaluation_manifest_plan_output_root_invalid")
    plan_cells = manifest.get("plan_cells")
    if not isinstance(plan_cells, list) or len(plan_cells) != 120:
        raise UserRunAggregateError("evaluation_manifest_plan_cells_invalid")

    compiled_by_id: dict[str, CellSpec] = {}
    rebuilt_cells: list[dict[str, Any]] = []
    for raw in plan_cells:
        _require_exact_keys(
            raw, _PLAN_CELL_KEYS, code="evaluation_manifest_plan_cell_fields_invalid"
        )
        identity = _identity_from_plan_cell(raw)
        cell_id = str(raw.get("cell_id") or "")
        if identity is None or cell_id in compiled_by_id:
            raise UserRunAggregateError("evaluation_manifest_plan_cell_invalid")
        try:
            spec = compile_cell_spec(
                model=str(raw.get("model") or ""),
                workload=str(raw.get("workload") or ""),
                arm=str(raw.get("arm") or ""),
                seed=int(raw.get("seed")),
                output_root=output_root,
                verify_pack=False,
            )
        except (TypeError, ValueError) as exc:
            raise UserRunAggregateError("evaluation_manifest_plan_cell_compile_failed") from exc
        if (
            spec.model != manifest.get("model")
            or spec.study != manifest.get("study")
            or spec.cell_id != cell_id
            or raw.get("cell_spec_sha256") != spec.fingerprint
        ):
            raise UserRunAggregateError("evaluation_manifest_plan_cell_invalid")
        compiled_by_id[cell_id] = spec
        rebuilt_cells.append(
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
    if set(_identity_from_plan_cell(cell) for cell in plan_cells) != _expected_model_identities(
        str(manifest.get("model") or "")
    ):
        raise UserRunAggregateError("evaluation_manifest_plan_cells_invalid")
    source = manifest.get("upstream_source")
    if not isinstance(source, Mapping) or dict(source) != _expected_source():
        raise UserRunAggregateError("evaluation_manifest_source_mismatch")
    if any(dict(spec.source) != dict(source) for spec in compiled_by_id.values()):
        raise UserRunAggregateError("evaluation_manifest_plan_source_mismatch")
    rebuilt_plan = {
        "study": manifest["study"],
        "model": manifest["model"],
        "output_root": str(output_root),
        "source": dict(source),
        "cells": rebuilt_cells,
    }
    if stable_sha256(rebuilt_plan) != manifest.get("plan_sha256"):
        raise UserRunAggregateError("evaluation_manifest_plan_hash_mismatch")
    return compiled_by_id


def _validate_receipt(payload: Mapping[str, Any]) -> dict[str, CellSpec]:
    _require_exact_keys(
        payload, _TOP_LEVEL_RECEIPT_KEYS, code="evaluation_manifest_top_level_fields_invalid"
    )
    if payload.get("schema_version") != EVALUATION_BATCH_SCHEMA_VERSION:
        raise UserRunAggregateError("evaluation_manifest_schema_mismatch")
    declared_hash = payload.get("receipt_sha256")
    unhashed = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    if not isinstance(declared_hash, str) or declared_hash != stable_sha256(unhashed):
        raise UserRunAggregateError("evaluation_manifest_hash_mismatch")
    if payload.get("paper_snapshot_inputs") != []:
        raise UserRunAggregateError("paper_snapshot_input_forbidden")
    if payload.get("dry_run") is not False:
        raise UserRunAggregateError("evaluation_manifest_dry_run_forbidden")
    if payload.get("status") != "completed":
        raise UserRunAggregateError("evaluation_manifest_batch_not_completed")
    if payload.get("scheduler_runtime_unchanged") is not True:
        raise UserRunAggregateError("evaluation_manifest_scheduler_runtime_changed")
    if payload.get("failures") != []:
        raise UserRunAggregateError("evaluation_manifest_batch_failures_present")
    if not all(
        isinstance(payload.get(field), str) and payload[field]
        for field in ("created_at", "completed_at")
    ):
        raise UserRunAggregateError("evaluation_manifest_timestamp_invalid")
    root_origin = _require_exact_keys(
        payload.get("run_origin"), _ROOT_ORIGIN_KEYS, code="evaluation_manifest_origin_missing"
    )
    _validate_empty_historical_inputs(root_origin, scope="evaluation_manifest_origin")
    trusted_boundary = _require_exact_keys(
        payload.get("trusted_checkpoint_boundary"),
        frozenset({"local_trusted_checkpoints_only", "warning"}),
        code="evaluation_manifest_trusted_boundary_invalid",
    )
    if (
        trusted_boundary.get("local_trusted_checkpoints_only") is not True
        or not isinstance(trusted_boundary.get("warning"), str)
        or not trusted_boundary["warning"]
    ):
        raise UserRunAggregateError("evaluation_manifest_trusted_boundary_invalid")
    source_snapshot = payload.get("source_snapshot")
    if not isinstance(source_snapshot, Mapping) or dict(source_snapshot) != _expected_source():
        raise UserRunAggregateError("evaluation_manifest_source_mismatch")
    snapshot = _require_exact_keys(
        payload.get("manifest_snapshot"),
        _MANIFEST_SNAPSHOT_KEYS,
        code="evaluation_manifest_snapshot_invalid",
    )
    if (
        not isinstance(snapshot.get("path"), str)
        or not snapshot["path"]
        or not _is_sha256(snapshot.get("sha256"))
        or snapshot.get("schema_version") != "relic-run-manifest-v2"
        or not _is_sha256(snapshot.get("plan_sha256"))
    ):
        raise UserRunAggregateError("evaluation_manifest_snapshot_invalid")
    runtime_before = payload.get("scheduler_runtime_before_sha256")
    runtime_after = payload.get("scheduler_runtime_after_sha256")
    if (
        not _is_sha256(runtime_before)
        or runtime_after != runtime_before
        or payload.get("scheduler_runtime_mutation") != "batch_never_writes_scheduler_runtime"
    ):
        raise UserRunAggregateError("evaluation_manifest_scheduler_runtime_invalid")
    manifest = _require_exact_keys(
        payload.get("manifest"),
        _RECEIPT_MANIFEST_KEYS,
        code="evaluation_manifest_source_missing",
    )
    source = manifest.get("upstream_source")
    if not isinstance(source, Mapping) or dict(source) != _expected_source():
        raise UserRunAggregateError("evaluation_manifest_source_mismatch")
    origin = _require_exact_keys(
        manifest.get("run_origin"),
        _MANIFEST_ORIGIN_KEYS,
        code="evaluation_manifest_not_user_new_run",
    )
    _validate_empty_historical_inputs(origin, scope="evaluation_manifest_manifest_origin")
    if (
        origin.get("instance_id") != root_origin.get("instance_id")
        or origin.get("created_at") != root_origin.get("created_at")
        or root_origin.get("manifest_sha256") != snapshot.get("sha256")
    ):
        raise UserRunAggregateError("evaluation_manifest_origin_mismatch")
    if (
        manifest.get("schema_version") != "relic-run-manifest-v2"
        or manifest.get("study") != "relic-main-v1"
        or manifest.get("model") not in CANONICAL_MODELS
        or not _is_sha256(manifest.get("plan_sha256"))
        or manifest.get("plan_sha256") != snapshot.get("plan_sha256")
        or manifest.get("manifest_snapshot_sha256") != snapshot.get("sha256")
        or isinstance(manifest.get("runtime_revision"), bool)
        or not isinstance(manifest.get("runtime_revision"), int)
        or manifest["runtime_revision"] < 0
    ):
        raise UserRunAggregateError("evaluation_manifest_plan_invalid")
    compiled_by_id = _rebuild_and_validate_plan(manifest)
    planned_by_id = {
        cell_id: (spec.model, spec.workload.upper(), spec.arm.upper(), spec.seed)
        for cell_id, spec in compiled_by_id.items()
    }
    planned_hashes = {cell_id: spec.fingerprint for cell_id, spec in compiled_by_id.items()}
    selection = _require_exact_keys(
        payload.get("selection"), _SELECTION_KEYS, code="evaluation_manifest_selection_invalid"
    )
    if selection.get("policy") not in _SELECTION_POLICIES:
        raise UserRunAggregateError("evaluation_manifest_selection_invalid")
    policy = str(selection["policy"])
    cells = payload.get("cells")
    if not isinstance(cells, list) or len(cells) != len(planned_by_id):
        raise UserRunAggregateError("evaluation_manifest_cells_invalid")
    cells_by_id: dict[str, Mapping[str, Any]] = {}
    evaluated_ids: set[str] = set()
    excluded_ids: set[str] = set()
    for cell in cells:
        if not isinstance(cell, Mapping):
            raise UserRunAggregateError("evaluation_manifest_cell_invalid")
        cell_id = str(cell.get("cell_id") or "")
        planned_identity = planned_by_id.get(cell_id)
        if cell_id in cells_by_id or planned_identity is None:
            raise UserRunAggregateError("evaluation_manifest_cell_invalid")
        if (
            cell.get("model") != planned_identity[0]
            or cell.get("workload") != planned_identity[1]
            or cell.get("arm") != planned_identity[2]
            or cell.get("seed") != planned_identity[3]
            or cell.get("cell_spec_sha256") != planned_hashes[cell_id]
        ):
            raise UserRunAggregateError("evaluation_manifest_cell_identity_invalid")
        scheduler_status = cell.get("scheduler_status")
        failure_class = cell.get("scheduler_failure_class")
        if (
            not isinstance(scheduler_status, str)
            or scheduler_status not in _SCHEDULER_STATUSES
            or (
                failure_class is not None
                and (
                    not isinstance(failure_class, str)
                    or failure_class not in _SCHEDULER_FAILURE_CLASSES
                )
            )
        ):
            raise UserRunAggregateError("evaluation_manifest_cell_scheduler_state_invalid")
        outcome = cell.get("outcome")
        if outcome == "evaluated":
            _require_exact_keys(
                cell, _EVALUATED_CELL_KEYS, code="evaluation_manifest_evaluated_cell_invalid"
            )
            analysis = cell.get("analysis")
            if not isinstance(analysis, Mapping):
                raise UserRunAggregateError("evaluation_manifest_analysis_missing")
            if _identity(cell) != planned_identity:
                raise UserRunAggregateError("evaluation_manifest_analysis_identity_invalid")
            _validate_analysis(analysis, identity=planned_identity, spec=compiled_by_id[cell_id])
            if cell.get("reason_code") is not None or cell.get("evaluator") != analysis.get(
                "evaluator"
            ):
                raise UserRunAggregateError("evaluation_manifest_evaluated_cell_invalid")
            expected_reason = _selection_reason(
                policy=policy,
                runtime_cell={"status": scheduler_status, "failure_class": failure_class},
                status={
                    "status": cell.get("public_pre_status"),
                    "stage": cell.get("public_pre_stage"),
                },
            )
            if expected_reason is not None:
                raise UserRunAggregateError("evaluation_manifest_evaluated_selection_invalid")
            evaluated_ids.add(cell_id)
        elif outcome == "not_selected":
            keys = set(cell)
            if keys not in {_CELL_BASE_KEYS, _NOT_SELECTED_WITH_PUBLIC_KEYS}:
                raise UserRunAggregateError("evaluation_manifest_not_selected_cell_invalid")
            if not isinstance(cell.get("reason_code"), str) or not cell["reason_code"]:
                raise UserRunAggregateError("evaluation_manifest_not_selected_cell_invalid")
            preselection = _scheduler_preselection_reason(
                policy=policy,
                runtime_cell={"status": scheduler_status, "failure_class": failure_class},
            )
            if keys == _CELL_BASE_KEYS:
                if cell["reason_code"] != preselection:
                    raise UserRunAggregateError("evaluation_manifest_not_selected_cell_invalid")
            else:
                if preselection is not None:
                    raise UserRunAggregateError("evaluation_manifest_not_selected_cell_invalid")
                expected_reason = _selection_reason(
                    policy=policy,
                    runtime_cell={"status": scheduler_status, "failure_class": failure_class},
                    status={
                        "status": cell.get("public_pre_status"),
                        "stage": cell.get("public_pre_stage"),
                    },
                )
                if cell["reason_code"] != expected_reason or expected_reason is None:
                    raise UserRunAggregateError("evaluation_manifest_not_selected_cell_invalid")
            excluded_ids.add(cell_id)
        else:
            # A completed, non-dry-run batch cannot legally aggregate failed,
            # invalid, or dry-run rows as if they were observations.
            raise UserRunAggregateError("evaluation_manifest_outcome_invalid")
        cells_by_id[cell_id] = cell
    if set(cells_by_id) != set(planned_by_id):
        raise UserRunAggregateError("evaluation_manifest_cells_invalid")
    eligible = selection.get("eligible")
    excluded = selection.get("excluded")
    if not isinstance(eligible, list) or not isinstance(excluded, list):
        raise UserRunAggregateError("evaluation_manifest_selection_invalid")
    eligible_ids: list[Any] = []
    for item in eligible:
        item = _require_exact_keys(
            item, _ELIGIBLE_SELECTION_KEYS, code="evaluation_manifest_selection_invalid"
        )
        cell_id = item.get("cell_id")
        cell = cells_by_id.get(str(cell_id) if cell_id is not None else "")
        if (
            not isinstance(cell_id, str)
            or cell is None
            or cell.get("outcome") != "evaluated"
            or item.get("cell_dir") != str(compiled_by_id[cell_id].cell_dir)
            or item.get("scheduler_status") != cell.get("scheduler_status")
            or item.get("scheduler_failure_class") != cell.get("scheduler_failure_class")
            or item.get("public_pre_status") != cell.get("public_pre_status")
            or item.get("public_pre_stage") != cell.get("public_pre_stage")
        ):
            raise UserRunAggregateError("evaluation_manifest_selection_invalid")
        checkpoint = _require_exact_keys(
            item.get("checkpoint_validation"),
            _CHECKPOINT_VALIDATION_KEYS,
            code="evaluation_manifest_selection_invalid",
        )
        if checkpoint != {
            "final_checkpoint_sidecar_verified": True,
            "final_tick": compiled_by_id[cell_id].ticks,
            "unpickle_performed_by_batch": False,
        }:
            raise UserRunAggregateError("evaluation_manifest_selection_invalid")
        eligible_ids.append(cell_id)
    excluded_pairs: list[tuple[Any, Any]] = []
    for item in excluded:
        item = _require_exact_keys(
            item, _EXCLUDED_SELECTION_KEYS, code="evaluation_manifest_selection_invalid"
        )
        excluded_pairs.append((item.get("cell_id"), item.get("reason")))
    if (
        len(eligible_ids) != len(eligible)
        or len(set(eligible_ids)) != len(eligible_ids)
        or set(eligible_ids) != evaluated_ids
        or len(excluded_pairs) != len(excluded)
        or len({cell_id for cell_id, _reason in excluded_pairs}) != len(excluded_pairs)
        or {cell_id for cell_id, _reason in excluded_pairs} != excluded_ids
        or any(
            not isinstance(cell_id, str)
            or not isinstance(reason, str)
            or not reason
            or cell_id not in cells_by_id
            or cells_by_id[cell_id].get("reason_code") != reason
            for cell_id, reason in excluded_pairs
        )
    ):
        raise UserRunAggregateError("evaluation_manifest_selection_invalid")
    evaluations = payload.get("evaluations")
    if not isinstance(evaluations, list) or len(evaluations) != len(evaluated_ids):
        raise UserRunAggregateError("evaluation_manifest_evaluations_invalid")
    evaluation_ids: list[Any] = []
    for item in evaluations:
        item = _require_exact_keys(
            item, _EVALUATION_KEYS, code="evaluation_manifest_evaluations_invalid"
        )
        evaluation_ids.append(item.get("cell_id"))
    if len(evaluation_ids) != len(evaluations) or set(evaluation_ids) != evaluated_ids:
        raise UserRunAggregateError("evaluation_manifest_evaluations_invalid")
    for item in evaluations:
        assert isinstance(item, Mapping)
        cell_id = str(item["cell_id"])
        if (
            item.get("status") != "completed"
            or item.get("return_code") != 0
            or item.get("analysis") != cells_by_id[cell_id].get("analysis")
        ):
            raise UserRunAggregateError("evaluation_manifest_evaluations_invalid")
    return compiled_by_id


def _metric_value(cell: Mapping[str, Any], metric: str) -> float | None:
    analysis = cell.get("analysis")
    if not isinstance(analysis, Mapping):
        return None
    if metric in {"provider_tokens_per_run", "average_tokens_per_run"}:
        usage = analysis.get("llm_usage")
        value = usage.get("total_tokens") if isinstance(usage, Mapping) else None
        tokens = _as_finite(value)
        if tokens is not None and metric == "average_tokens_per_run":
            return tokens / 1_000_000.0
        return tokens
    final = analysis.get("final_evaluation")
    if not isinstance(final, Mapping):
        return None
    mapping = {
        "generic_candidate_pass_rate_percent": "candidate_pass_rate",
        "generic_causal_fix_rate_percent": "causal_fix_rate",
        "generic_causal_fix_count_per_run": "causal_fix_count",
        "generic_unresolved_count_per_run": "unresolved_count",
        "generic_regression_count_per_run": "regression_count",
        "generic_infrastructure_error_count_per_run": "infrastructure_error_count",
    }
    raw = _as_finite(final.get(mapping[metric])) if metric in mapping else None
    if raw is not None and metric.endswith("_percent"):
        return raw * 100.0
    return raw


def _block_macro(cells: Sequence[Mapping[str, Any]], metric: str) -> dict[str, Any]:
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for cell in cells:
        identity = _identity(cell)
        value = _metric_value(cell, metric)
        if identity is None or value is None:
            continue
        model, workload, arm, _seed = identity
        grouped[(model, workload, arm)].append(value)
    blocks = [
        {
            "model": model,
            "workload": workload,
            "arm": arm,
            "seed_count": len(values),
            "value": sum(values) / len(values),
        }
        for (model, workload, arm), values in sorted(grouped.items())
    ]
    by_arm: dict[str, dict[str, Any]] = {}
    for arm in CANONICAL_ARMS:
        selected = [row for row in blocks if row["arm"] == arm]
        values = [float(row["value"]) for row in selected]
        by_arm[arm] = {
            "value": sum(values) / len(values) if values else None,
            "eligible_blocks": len(values),
            "contributing_cells": sum(int(row["seed_count"]) for row in selected),
            "blocks": selected,
        }
    return by_arm


def _lineage(cell: Mapping[str, Any]) -> tuple[Any, ...] | None:
    analysis = cell.get("analysis")
    raw = analysis.get("lineage") if isinstance(analysis, Mapping) else None
    if not isinstance(raw, Mapping):
        return None
    values = tuple(raw.get(field) for field in _LINEAGE_FIELDS)
    return values if all(isinstance(value, str) and value for value in values) else None


def _aggregate_exclusion_reason(cell: Mapping[str, Any]) -> str | None:
    """Return a bounded reason when an evaluated row is not a measurement.

    A successful evaluator process is not itself evidence that the evaluator
    produced a verdict.  In particular, an infrastructure result has numeric
    fields in the historical artifact schema, but those fields must never be
    interpreted as a poor candidate score.
    """

    analysis = cell.get("analysis")
    if not isinstance(analysis, Mapping) or analysis.get("run_record_status") != "completed":
        return "run_record_unavailable"
    final = analysis.get("final_evaluation")
    evaluator = analysis.get("evaluator")
    if not isinstance(final, Mapping) or not isinstance(evaluator, Mapping):
        return "evaluator_unavailable"
    status = final.get("status")
    if status not in _AGGREGATABLE_EVALUATOR_STATUSES:
        return "evaluator_status_unavailable"
    if evaluator.get("status") != status:
        return "evaluator_status_unavailable"
    if final.get("infrastructure_error_count") != 0:
        return "evaluator_infrastructure_error"
    if not all(
        _is_sha256(evaluator.get(field))
        for field in ("artifact_hash", "result_hash", "candidate_repo_digest")
    ):
        return "evaluator_provenance_unavailable"
    if _lineage(cell) is None:
        return "evaluator_lineage_unavailable"
    return None


def _validate_full_design_receipts(
    receipts: Sequence[tuple[dict[str, Any], str, Path]],
) -> None:
    """Ensure a 240-cell claim consists of two whole single-model batches.

    The aggregate must never turn several cherry-picked partial batches into a
    nominal full design.  Each receipt already has a canonical 120-cell plan;
    here we additionally require exactly one complete receipt per canonical
    model and stable release provenance within and across those receipts.
    """

    if len(receipts) != len(CANONICAL_MODELS):
        raise UserRunAggregateError("full_design_requires_two_complete_receipts")
    by_model: dict[str, dict[str, Any]] = {}
    cross_model_fields = (
        "source_provenance_sha256",
        "runtime_tree_sha256",
        "execution_policy_sha256",
    )
    receipt_fields = (*cross_model_fields, "model_binding_sha256")
    workload_fields = (
        "qualification_plan_sha256",
        "evaluator_environment_sha256",
        "starter_repo_digest",
        "reference_repo_digest",
        "hidden_suite_hash",
    )
    cross_model_values: dict[str, set[str]] = {field: set() for field in cross_model_fields}
    evaluator_by_model: dict[str, dict[str, tuple[str, ...]]] = {}
    for receipt, _file_hash, _path in receipts:
        manifest = receipt["manifest"]
        assert isinstance(manifest, Mapping)
        model = str(manifest["model"])
        if model in by_model:
            raise UserRunAggregateError("full_design_duplicate_model_receipt")
        by_model[model] = receipt
        expected = _expected_model_identities(model)
        receipt_cells = receipt["cells"]
        assert isinstance(receipt_cells, list)
        if len(receipt_cells) != len(expected):
            raise UserRunAggregateError("full_design_receipt_cells_incomplete")
        observed: set[tuple[str, str, str, int]] = set()
        per_receipt_values: dict[str, set[str]] = {field: set() for field in receipt_fields}
        per_workload_values: dict[str, dict[str, set[str]]] = {
            workload: {field: set() for field in workload_fields}
            for workload in CANONICAL_WORKLOADS
        }
        for cell in receipt_cells:
            assert isinstance(cell, Mapping)
            identity = _identity(cell)
            if cell.get("outcome") != "evaluated" or identity is None:
                raise UserRunAggregateError("full_design_receipt_cells_incomplete")
            if _aggregate_exclusion_reason(cell) is not None:
                raise UserRunAggregateError("full_design_evaluator_measurement_unavailable")
            observed.add(identity)
            lineage = cell["analysis"]["lineage"]
            assert isinstance(lineage, Mapping)
            for field in receipt_fields:
                per_receipt_values[field].add(str(lineage[field]))
            for field in workload_fields:
                per_workload_values[identity[1]][field].add(str(lineage[field]))
        if observed != expected or any(len(values) != 1 for values in per_receipt_values.values()):
            raise UserRunAggregateError("full_design_receipt_provenance_mismatch")
        if any(
            len(values) != 1
            for by_field in per_workload_values.values()
            for values in by_field.values()
        ):
            raise UserRunAggregateError("full_design_receipt_provenance_mismatch")
        for field in cross_model_fields:
            cross_model_values[field].update(per_receipt_values[field])
        evaluator_by_model[model] = {
            workload: tuple(
                next(iter(per_workload_values[workload][field])) for field in workload_fields
            )
            for workload in CANONICAL_WORKLOADS
        }
    if set(by_model) != set(CANONICAL_MODELS):
        raise UserRunAggregateError("full_design_models_mismatch")
    if any(len(values) != 1 for values in cross_model_values.values()):
        raise UserRunAggregateError("full_design_cross_model_provenance_mismatch")
    first, second = CANONICAL_MODELS
    if evaluator_by_model[first] != evaluator_by_model[second]:
        raise UserRunAggregateError("full_design_evaluator_provenance_mismatch")


def _source_statistics_observation(
    cell: Mapping[str, Any],
    *,
    spec: CellSpec,
    metric: str,
) -> RunObservation:
    """Adapt one verified public receipt row to the source statistics contract.

    The receipt has already passed Relic's formal-record, evaluator-digest, and
    lineage gates.  This deliberately does not invent a second evaluator or
    score: it presents the verified numeric field to hci's paired-statistics
    implementation with its canonical model/provider/resource identities.
    """

    identity = _identity(cell)
    analysis = cell.get("analysis")
    if identity is None or not isinstance(analysis, Mapping):
        raise UserRunAggregateError("source_statistics_adapter_identity_invalid")
    value = _metric_value(cell, metric)
    if value is None:
        raise UserRunAggregateError("source_statistics_adapter_metric_unavailable")
    lineage = analysis.get("lineage")
    evaluator = analysis.get("evaluator")
    final = analysis.get("final_evaluation")
    if (
        not isinstance(lineage, Mapping)
        or not isinstance(evaluator, Mapping)
        or not isinstance(final, Mapping)
    ):
        raise UserRunAggregateError("source_statistics_adapter_lineage_invalid")
    model, workload, arm, seed = identity
    if (
        spec.model != model
        or spec.workload.upper() != workload
        or spec.arm.upper() != arm
        or spec.seed != seed
    ):
        raise UserRunAggregateError("source_statistics_adapter_spec_mismatch")
    provider = str(spec.model_config.get("provider") or "").strip().lower()
    if not provider:
        raise UserRunAggregateError("source_statistics_adapter_provider_missing")
    resource_budget = stable_sha256(spec.resource_budget.to_dict())
    mechanism_ablations = stable_sha256(
        {"mechanism_ablations": list(spec.mechanism_ablations)}
    )
    return RunObservation(
        run_id=str(cell["cell_id"]),
        pack=spec.dataset_id,
        condition=arm,
        provider=provider,
        model=model,
        seed=seed,
        metric=metric,
        value=value,
        metric_family="relic_user_run_release_adapter",
        resource_budget_fingerprint=resource_budget,
        ablation_fingerprint=mechanism_ablations,
        # The public receipt intentionally omits raw run records.  Keep the
        # adapter on source's v1 input schema; formal eligibility was proved at
        # the receipt boundary above rather than reconstructed from redacted
        # fields here.
        schema_version="orgenv_experiment_run_v1",
        status="completed",
        final_status=(
            str(final["status"]) if isinstance(final.get("status"), str) else None
        ),
        formal_claim_ready=(
            final["formal_claim_ready"]
            if isinstance(final.get("formal_claim_ready"), bool)
            else None
        ),
        dataset_manifest_hash=stable_sha256(spec.benchmark_entry),
        starter_repo_digest=(
            str(lineage["starter_repo_digest"])
            if isinstance(lineage.get("starter_repo_digest"), str)
            else None
        ),
        reference_repo_digest=(
            str(lineage["reference_repo_digest"])
            if isinstance(lineage.get("reference_repo_digest"), str)
            else None
        ),
        candidate_repo_digest=(
            str(evaluator["candidate_repo_digest"])
            if isinstance(evaluator.get("candidate_repo_digest"), str)
            else None
        ),
        hidden_suite_hash=(
            str(lineage["hidden_suite_hash"])
            if isinstance(lineage.get("hidden_suite_hash"), str)
            else None
        ),
        evaluator_environment_hash=(
            str(lineage["evaluator_environment_sha256"])
            if isinstance(lineage.get("evaluator_environment_sha256"), str)
            else None
        ),
        information_budget_fingerprint=resource_budget,
        llm_runtime_fingerprint=(
            str(lineage["model_binding_sha256"])
            if isinstance(lineage.get("model_binding_sha256"), str)
            else None
        ),
        final_plan_hash=(
            str(lineage["qualification_plan_sha256"])
            if isinstance(lineage.get("qualification_plan_sha256"), str)
            else None
        ),
        final_result_hash=(
            str(evaluator["result_hash"])
            if isinstance(evaluator.get("result_hash"), str)
            else None
        ),
        final_artifact_hash=(
            str(evaluator["artifact_hash"])
            if isinstance(evaluator.get("artifact_hash"), str)
            else None
        ),
    )


def _paired_contrast(
    cells: Sequence[Mapping[str, Any]],
    metric: str,
    *,
    specs_by_cell_id: Mapping[str, CellSpec],
) -> dict[str, Any]:
    """Report B3−B2 through the hci fixed-block paired implementation."""

    index: dict[tuple[str, str, str, int], Mapping[str, Any]] = {}
    for cell in cells:
        identity = _identity(cell)
        if identity is not None and identity not in index:
            index[identity] = cell
    exclusions: Counter[str] = Counter()
    paired_cells: list[Mapping[str, Any]] = []
    for model in CANONICAL_MODELS:
        for workload in CANONICAL_WORKLOADS:
            for seed in CANONICAL_SEEDS:
                b2 = index.get((model, workload, "B2", seed))
                b3 = index.get((model, workload, "B3", seed))
                if b2 is None or b3 is None:
                    exclusions["missing_counterpart"] += 1
                    continue
                if _lineage(b2) is None or _lineage(b2) != _lineage(b3):
                    exclusions["provenance_mismatch"] += 1
                    continue
                low = _metric_value(b2, metric)
                high = _metric_value(b3, metric)
                if low is None or high is None:
                    exclusions["metric_unavailable"] += 1
                    continue
                paired_cells.extend((b2, b3))
    if not paired_cells:
        return {
            "b3_minus_b2": None,
            "ci95": None,
            "eligible_blocks": 0,
            "paired_cells": 0,
            "excluded_pairs": dict(sorted(exclusions.items())),
        }
    try:
        observations = tuple(
            _source_statistics_observation(
                cell,
                spec=specs_by_cell_id[str(cell["cell_id"])],
                metric=metric,
            )
            for cell in paired_cells
        )
        # Partial aggregates may contain only B2/B3.  Source validation still
        # proves the paired resource/ablation/lineage constraints before the
        # source pairing function constructs the contrast units.
        validate_observations(
            observations,
            require_complete_conditions=False,
            require_matched_resources=True,
            require_matched_ablations=True,
        )
        units = build_paired_units(
            observations,
            metric=metric,
            treatment="B3",
            control="B2",
            require_complete_pairs=True,
        )
        estimate = paired_block_weighted_mean(units)
        confidence_low, confidence_high = paired_block_bootstrap_interval(
            units,
            samples=BOOTSTRAP_DRAWS,
            seed=BOOTSTRAP_SEED,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise UserRunAggregateError("source_statistics_adapter_invalid") from exc
    return {
        "b3_minus_b2": estimate,
        "ci95": [confidence_low, confidence_high],
        "eligible_blocks": len({unit.block_key for unit in units}),
        "paired_cells": len(units),
        "excluded_pairs": dict(sorted(exclusions.items())),
    }


def _tokens_per_generic_causal_fix(
    cells: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for cell in cells:
        identity = _identity(cell)
        if identity is None:
            continue
        grouped[identity[:3]].append(cell)
    by_arm: dict[str, dict[str, Any]] = {}
    for arm in CANONICAL_ARMS:
        rows: list[dict[str, Any]] = []
        for (model, workload, candidate_arm), block_cells in sorted(grouped.items()):
            if candidate_arm != arm:
                continue
            tokens = [_metric_value(cell, "provider_tokens_per_run") for cell in block_cells]
            fixes = [
                _metric_value(cell, "generic_causal_fix_count_per_run") for cell in block_cells
            ]
            if any(value is None for value in tokens + fixes):
                continue
            total_fixes = sum(float(value) for value in fixes if value is not None)
            rows.append(
                {
                    "model": model,
                    "workload": workload,
                    "seed_count": len(block_cells),
                    "value": (
                        sum(float(value) for value in tokens if value is not None) / total_fixes
                        if total_fixes > 0
                        else None
                    ),
                    "availability": "measured" if total_fixes > 0 else "zero_denominator",
                }
            )
        values = [float(row["value"]) for row in rows if row["value"] is not None]
        by_arm[arm] = {
            "value": sum(values) / len(values) if values else None,
            "eligible_blocks": len(values),
            "blocks": rows,
        }
    return by_arm


def _atomic_write(path: Path, body: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o644)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(body)
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


def _render_markdown(payload: Mapping[str, Any]) -> str:
    coverage = payload["coverage"]
    lines = [
        "# User-run aggregate — not paper results",
        "",
        str(payload["noncomparability_notice"]),
        "",
        "## Coverage",
        "",
        f"- Analysis status: `{payload['analysis_status']}`",
        f"- Expected canonical cells: {coverage['expected_cells']}",
        f"- Evaluated or reused cells: {coverage['evaluated_cells']}",
        f"- Unique planned cells represented: {coverage['represented_cells']}",
        "",
        "## Scoring boundary",
        "",
        (
            "The current evaluator artifact has no versioned case/contract visibility ledger. "
            "Paper-named contract, exposed-case, held-out-case, confirmed-issue, and "
            "tokens-per-confirmed-issue values therefore remain NA; they are never inferred."
        ),
        "",
        "## Available descriptive metrics",
        "",
        "| Metric | B0 | B1 | B2 | B3 | B3−B2 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for metric, result in payload["results"]["available_metrics"].items():
        arms = result["by_arm"]

        def display(value: Any) -> str:
            return "NA" if value is None else f"{float(value):.6g}"

        lines.append(
            f"| {metric} | "
            + " | ".join(display(arms[arm]["value"]) for arm in CANONICAL_ARMS)
            + f" | {display(result['b3_minus_b2']['b3_minus_b2'])} |"
        )
    lines.extend(
        [
            "",
            "Aggregation averages seeds within each model × workload × arm block, then "
            "weights eligible model × workload blocks equally. B3−B2 uses common paired "
            f"seeds and {BOOTSTRAP_DRAWS:,} fixed-block bootstrap draws with seed "
            f"{BOOTSTRAP_SEED} (percentile 95% interval).",
            "",
        ]
    )
    return "\n".join(lines)


def build_user_run_aggregate(
    evaluation_manifests: Iterable[Path],
    *,
    output_directory: Path,
    allow_partial: bool = False,
) -> UserRunAggregateResult:
    """Build a user-run-only aggregate from validated batch receipts."""

    manifest_paths = [Path(path) for path in evaluation_manifests]
    if not manifest_paths:
        raise UserRunAggregateError("evaluation_manifest_required")
    receipts: list[tuple[dict[str, Any], str, Path]] = []
    cells: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    specs_by_cell_id: dict[str, CellSpec] = {}
    for path in manifest_paths:
        payload, file_hash = _read_receipt(path)
        receipt_specs = _validate_receipt(payload)
        receipts.append((payload, file_hash, path.resolve()))
        for raw_cell in payload["cells"]:
            if not isinstance(raw_cell, dict):
                raise UserRunAggregateError("evaluation_manifest_cell_invalid")
            cell_id = str(raw_cell.get("cell_id") or "")
            if not cell_id or cell_id in seen_ids:
                raise UserRunAggregateError("duplicate_evaluation_cell")
            seen_ids.add(cell_id)
            cells.append(raw_cell)
            specs_by_cell_id[cell_id] = receipt_specs[cell_id]

    represented = {_identity(cell) for cell in cells}
    represented.discard(None)
    aggregate_exclusions: Counter[str] = Counter()
    successful: list[dict[str, Any]] = []
    for cell in cells:
        if cell.get("outcome") != "evaluated" or _identity(cell) is None:
            continue
        reason = _aggregate_exclusion_reason(cell)
        if reason is not None:
            aggregate_exclusions[reason] += 1
            continue
        successful.append(cell)
    expected_identities = {
        (model, workload, arm, seed)
        for model in CANONICAL_MODELS
        for workload in CANONICAL_WORKLOADS
        for arm in CANONICAL_ARMS
        for seed in CANONICAL_SEEDS
    }
    full_design = set(represented) == expected_identities and len(successful) == 240
    if full_design:
        _validate_full_design_receipts(receipts)
    if not full_design and not allow_partial:
        raise UserRunAggregateError("incomplete_user_run_design_use_allow_partial")
    analysis_status = (
        "full_design_limited_metrics_scoring_ledger_missing"
        if full_design
        else "partial_descriptive_not_paper_comparable"
    )

    metric_specs = {
        "average_tokens_per_run": "million_tokens",
        "generic_candidate_pass_rate_percent": "percent",
        "generic_causal_fix_rate_percent": "percent",
        "generic_causal_fix_count_per_run": "count",
        "generic_unresolved_count_per_run": "count",
        "generic_regression_count_per_run": "count",
        "generic_infrastructure_error_count_per_run": "count",
    }
    available: dict[str, Any] = {}
    for metric, unit in metric_specs.items():
        available[metric] = {
            "unit": unit,
            "paper_metric": metric == "average_tokens_per_run",
            "estimand": "seed_mean_then_equal_weight_model_workload_blocks",
            "by_arm": _block_macro(successful, metric),
            "b3_minus_b2": _paired_contrast(
                successful,
                metric,
                specs_by_cell_id=specs_by_cell_id,
            ),
        }
    available["generic_tokens_per_causal_fix_outcome"] = {
        "unit": "tokens",
        "paper_metric": False,
        "warning": "generic evaluator outcome; not evaluator-confirmed issue scoring",
        "estimand": "block_sum_tokens_divided_by_block_sum_generic_causal_fixes",
        "by_arm": _tokens_per_generic_causal_fix(successful),
        "b3_minus_b2": {
            "b3_minus_b2": None,
            "ci95": None,
            "reason": "paper_confirmed_issue_scoring_ledger_missing",
        },
    }

    unavailable: dict[str, Any] = {}
    for metric in _PAPER_SCORING_METRICS:
        unavailable[metric] = {
            "value": None,
            "availability": "scoring_ledger_missing",
            "reason": "evaluator_does_not_emit_hash_bound_paper_scoring_rows",
            "workload_exception": (
                {"W01": "inapplicable_no_held_out_scoring_units"}
                if metric == "held_out_cases"
                else {}
            ),
        }

    payload: dict[str, Any] = {
        "schema_version": USER_RUN_AGGREGATE_SCHEMA_VERSION,
        "artifact_kind": "user_new_runs_aggregate",
        "generated_at": _utc_now(),
        "analysis_status": analysis_status,
        "paper_snapshot": {"mixed": False, "used_as_input": False},
        "source_scope": {
            "source_repository": SOURCE_REPOSITORY,
            "source_branch": SOURCE_BRANCH,
            "source_commit": SOURCE_COMMIT,
            "source_git_tree": SOURCE_GIT_TREE,
            "evaluation_manifests": [
                {
                    "sha256": file_hash,
                    "plan_sha256": receipt["manifest"].get("plan_sha256"),
                    "run_origin_instance_id": receipt["manifest"]["run_origin"].get("instance_id"),
                }
                for receipt, file_hash, _path in receipts
            ],
            "cell_spec_fingerprints": sorted(
                {
                    str(cell["cell_spec_sha256"])
                    for cell in successful
                    if cell.get("cell_spec_sha256")
                }
            ),
            "evaluator_artifact_hashes": sorted(
                {
                    str(cell["evaluator"]["artifact_hash"])
                    for cell in successful
                    if isinstance(cell.get("evaluator"), Mapping)
                    and cell["evaluator"].get("artifact_hash")
                }
            ),
        },
        "design": {
            "models": list(CANONICAL_MODELS),
            "workloads": list(CANONICAL_WORKLOADS),
            "arms": list(CANONICAL_ARMS),
            "seeds": list(CANONICAL_SEEDS),
        },
        "coverage": {
            "expected_cells": 240,
            "represented_cells": len(represented),
            "evaluated_cells": len(successful),
            "outcomes": dict(sorted(Counter(str(cell.get("outcome")) for cell in cells).items())),
            "aggregate_exclusions": dict(sorted(aggregate_exclusions.items())),
            "missing_cell_ids": sorted(
                f"{model}__{workload}__{arm}__seed{seed}"
                for model, workload, arm, seed in expected_identities - set(represented)
            ),
        },
        "scoring_ledger": {
            "schema_version": None,
            "sha256": None,
            "availability": "missing",
            "required_fields": [
                "metric_id",
                "scoring_unit_id",
                "contract_id",
                "issue_id",
                "visibility",
                "numerator",
                "denominator",
                "status",
                "reason",
            ],
        },
        "aggregation": {
            "unit_order": [
                "cell",
                "seed_mean_within_model_workload_arm",
                "equal_weight_model_workload_blocks",
            ],
            "missingness_policy": "fail_closed_na_no_imputation",
            "contrast": "paired_b3_minus_b2_common_model_workload_seed",
            "bootstrap": {
                "draws": BOOTSTRAP_DRAWS,
                "rng": "python_random_mt19937",
                "rng_seed": BOOTSTRAP_SEED,
                "resampling": "seeds_with_replacement_within_fixed_model_workload_blocks",
                "ci": "percentile_95",
                "quantiles": [0.025, 0.975],
            },
            "source_statistics": {
                "module": "environments.org_env.experiments.statistics",
                "pairing": "build_paired_units",
                "point_estimate": "paired_block_weighted_mean",
                "interval": "paired_block_bootstrap_interval",
                "receipt_adapter": "verified_public_receipt_to_orgenv_v1_observation",
            },
        },
        "results": {
            "available_metrics": available,
            "paper_metrics_unavailable": unavailable,
        },
        "noncomparability_notice": (
            "User-generated results only. The canonical paper snapshot was not read, mixed, "
            "or used to fill missing cells. Available generic evaluator metrics are not a "
            "substitute for the paper's missing scoring ledger."
        ),
    }
    payload["aggregate_sha256"] = stable_sha256(payload)
    destination = _canonical_output_directory(output_directory)
    json_path = destination / "user_run_aggregate.json"
    markdown_path = destination / "user_run_aggregate.md"
    _atomic_write(
        json_path,
        (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
    )
    _atomic_write(markdown_path, _render_markdown(payload).encode("utf-8"))
    return UserRunAggregateResult(
        json_path=json_path,
        markdown_path=markdown_path,
        analysis_status=analysis_status,
        expected_cells=240,
        evaluated_cells=len(successful),
        aggregate_sha256=str(payload["aggregate_sha256"]),
    )


__all__ = [
    "BOOTSTRAP_DRAWS",
    "BOOTSTRAP_SEED",
    "UserRunAggregateError",
    "UserRunAggregateResult",
    "build_user_run_aggregate",
]
