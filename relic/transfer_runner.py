"""Thin source-backed orchestration for the final Relic transfer comparison.

The paper's final transfer comparison has two *new* target arms: fresh-roster
Text and fresh-roster Exec.  Both use the existing B2 source runner and the
same canonical v2 capability bundle.  Fresh is the already-existing B2
main-study reference, not a third transfer target to generate here.

This module deliberately owns only planning, selection, and manifest state.
Every actual target invocation goes back through
``tools.run_org_baselines.py`` via :mod:`relic.source_runner`; it does not
replace the source runner's fresh-process isolation or formal evaluator gate.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from environments.org_env.experiments.capability_transfer import (
    BUNDLE_SCHEMA_VERSION_V2,
    CANONICAL_V2_BUNDLE_SHA256,
    CAPABILITY_FORM_EXECUTABLE,
    CAPABILITY_FORM_TEXT_ONLY,
    ROSTER_ORIGIN_FRESH,
    capability_bundle_sha256,
    validate_capability_bundle,
)
from relic.manifest import write_manifest
from relic.paths import default_output_root, project_root
from relic.research.hashing import stable_hash
from relic.source_runner import (
    SOURCE_BASELINE_RUNNER,
    EvaluatorBinding,
    _batch_status,
    _invoke_source_batch,
    _require_selected_bindings,
    _select_batches,
    _source_batch_argv,
    build_source_main_manifest,
    load_evaluator_bindings,
)


TRANSFER_RUN_MANIFEST_SCHEMA_VERSION = "relic-transfer-run-manifest-v1"
TRANSFER_PLAN_SCHEMA_VERSION = "relic-transfer-plan-v1"
TRANSFER_MODEL = "gpt-5.6-terra"
TRANSFER_ARMS = ("text", "exec")
_CANONICAL_BUNDLE_RELATIVE_PATH = Path(
    "environments/org_env/data/capability_bundles/canonical_v2.json"
)
_TRANSFER_MODULE_SOURCE = (
    "origin/codex/transfer-v4-fixed-protocol@"
    "1a49b4821ab207b012d95057d2151c5cfaf1bc38"
)

_ARM_CONFIGURATION: Mapping[str, Mapping[str, str]] = {
    "text": {
        "arm_id": "Text",
        "capability_form": CAPABILITY_FORM_TEXT_ONLY,
    },
    "exec": {
        "arm_id": "Exec",
        "capability_form": CAPABILITY_FORM_EXECUTABLE,
    },
}


class TransferRunnerError(RuntimeError):
    """Stable non-secret error for the final transfer adapter."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class TransferRunResult:
    manifest_path: Path
    output_root: Path
    model: str
    status: str
    total_target_runs: int
    selected_target_runs: int
    selected_workload_seed_batches: int
    completed_target_runs: int
    failed_target_runs: int
    dry_run: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest": str(self.manifest_path),
            "output_root": str(self.output_root),
            "model": self.model,
            "status": self.status,
            "paper_design": (
                "10 workloads x 3 seeds x 2 fresh-B2 transfer target arms "
                "(Text, Exec); Fresh reuses the main-study B2 reference"
            ),
            "total_target_runs": self.total_target_runs,
            "selected_target_runs": self.selected_target_runs,
            "selected_workload_seed_batches": self.selected_workload_seed_batches,
            "completed_target_runs": self.completed_target_runs,
            "failed_target_runs": self.failed_target_runs,
            "dry_run": self.dry_run,
            "executor": SOURCE_BASELINE_RUNNER,
        }


def _plan_digest(plan: Mapping[str, Any]) -> str:
    return stable_hash(
        {key: value for key, value in plan.items() if key != "plan_sha256"}
    )


def _canonical_bundle() -> tuple[Path, Mapping[str, Any], str]:
    path = project_root() / _CANONICAL_BUNDLE_RELATIVE_PATH
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise TransferRunnerError("transfer_canonical_bundle_missing") from exc
    except (OSError, ValueError) as exc:
        raise TransferRunnerError("transfer_canonical_bundle_unreadable") from exc
    if not isinstance(payload, Mapping):
        raise TransferRunnerError("transfer_canonical_bundle_invalid")
    try:
        validate_capability_bundle(payload)
    except ValueError as exc:
        raise TransferRunnerError("transfer_canonical_bundle_invalid") from exc
    if payload.get("schema_version") != BUNDLE_SCHEMA_VERSION_V2:
        raise TransferRunnerError("transfer_canonical_bundle_schema_mismatch")
    digest = capability_bundle_sha256(payload)
    if digest != CANONICAL_V2_BUNDLE_SHA256:
        raise TransferRunnerError("transfer_canonical_bundle_digest_mismatch")
    return path.resolve(), payload, digest


def _require_model(model: str | None) -> str:
    normalized = str(model or TRANSFER_MODEL).strip().lower()
    if normalized != TRANSFER_MODEL:
        raise TransferRunnerError("transfer_model_must_be_gpt_5_6_terra")
    return normalized


def _normalize_arm_selection(arm: str) -> tuple[str, ...]:
    normalized = str(arm or "both").strip().lower()
    if normalized == "both":
        return TRANSFER_ARMS
    if normalized in TRANSFER_ARMS:
        return (normalized,)
    raise TransferRunnerError("transfer_arm_unknown")


def build_transfer_manifest(
    *,
    model: str | None = None,
    output_root: Path | None = None,
    max_parallel: int = 1,
) -> dict[str, Any]:
    """Build the fixed 60-target-run Text/Exec transfer plan.

    The source main manifest supplies the frozen workload, seed, resource, and
    runtime identity.  We replace only its four-condition cell matrix with the
    paper-final B2 target matrix.
    """

    if max_parallel < 1:
        raise TransferRunnerError("transfer_max_parallel_must_be_positive")
    canonical_model = _require_model(model)
    destination = (
        output_root or default_output_root() / "transfer-v2"
    ).expanduser().resolve()
    _, bundle, bundle_digest = _canonical_bundle()
    source_payload = build_source_main_manifest(
        model=canonical_model,
        output_root=destination,
        max_parallel=max_parallel,
    )
    source_plan = source_payload["plan"]
    source_batches = source_plan.get("batches")
    if not isinstance(source_batches, list) or len(source_batches) != 30:
        raise TransferRunnerError("transfer_source_batches_invalid")

    batches: list[dict[str, Any]] = []
    for source_batch in source_batches:
        if not isinstance(source_batch, Mapping):
            raise TransferRunnerError("transfer_source_batches_invalid")
        batch_id = str(source_batch.get("batch_id") or "")
        workload = str(source_batch.get("workload") or "")
        pack = str(source_batch.get("pack") or "")
        try:
            seed = int(source_batch.get("seed"))
        except (TypeError, ValueError) as exc:
            raise TransferRunnerError("transfer_source_batch_seed_invalid") from exc
        if not batch_id or not workload or not pack:
            raise TransferRunnerError("transfer_source_batches_invalid")
        targets: list[dict[str, Any]] = []
        for arm in TRANSFER_ARMS:
            arm_config = _ARM_CONFIGURATION[arm]
            targets.append(
                {
                    "target_id": (
                        f"{canonical_model}__{workload.upper()}__"
                        f"TRANSFER_{arm_config['arm_id'].upper()}__seed{seed}"
                    ),
                    "arm": arm,
                    "arm_id": arm_config["arm_id"],
                    "source_case": "b2",
                    "condition_id": "b2_policy_conditioned_org",
                    "roster_origin": ROSTER_ORIGIN_FRESH,
                    "capability_form": arm_config["capability_form"],
                    "fixed_protocol_landscape": True,
                    "output_root": str(destination / "batches" / arm / batch_id),
                }
            )
        batches.append(
            {
                "batch_id": batch_id,
                "workload": workload,
                "pack": pack,
                "seed": seed,
                "targets": targets,
            }
        )

    total_targets = sum(len(batch["targets"]) for batch in batches)
    if len(batches) != 30 or total_targets != 60:
        raise TransferRunnerError("transfer_target_count_invalid")
    plan: dict[str, Any] = {
        "schema_version": TRANSFER_PLAN_SCHEMA_VERSION,
        "study": "relic-transfer-v2",
        "paper_design": (
            "10 workloads x 3 seeds x 2 fresh-B2 transfer target arms "
            "(Text, Exec); Fresh reuses the main-study B2 reference"
        ),
        "output_root": str(destination),
        "source": {
            **dict(source_plan["source"]),
            "transfer_module_source": _TRANSFER_MODULE_SOURCE,
        },
        "model": dict(source_plan["model"]),
        "ticks": int(source_plan["ticks"]),
        "checkpoint_every": int(source_plan["checkpoint_every"]),
        "sprint_ticks": int(source_plan["sprint_ticks"]),
        "mechanism_ablations": list(source_plan["mechanism_ablations"]),
        "resource_ceilings": dict(source_plan["resource_ceilings"]),
        "canonical_bundle": {
            "relative_path": str(_CANONICAL_BUNDLE_RELATIVE_PATH),
            "schema_version": str(bundle["schema_version"]),
            "bundle_sha256": bundle_digest,
            "source_repository_id": str(bundle["source_repository_id"]),
        },
        "target": {
            "source_case": "b2",
            "condition_id": "b2_policy_conditioned_org",
            "roster_origin": ROSTER_ORIGIN_FRESH,
            "fixed_protocol_landscape": True,
            "new_target_arms": ["Text", "Exec"],
            "fresh_reference": "existing_main_study_b2_only",
        },
        "evaluator_binding": {
            "required_for_formal_execution": True,
            "binding_file_required": True,
            "per_pack": True,
            "unpublished_values_are_not_fabricated": True,
        },
        "batches": batches,
    }
    plan["plan_sha256"] = _plan_digest(plan)
    return {
        "schema_version": TRANSFER_RUN_MANIFEST_SCHEMA_VERSION,
        "plan": plan,
        "execution": {
            "status": "planned",
            "source_dry_run_root": None,
            "targets": {},
        },
        "resource_policy": {
            "max_parallel_scope": "within_each_single-B2 source invocation",
            "requested_max_parallel": max_parallel,
        },
    }


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise TransferRunnerError("transfer_manifest_missing") from exc
    except (OSError, ValueError) as exc:
        raise TransferRunnerError("transfer_manifest_unreadable") from exc
    if not isinstance(payload, dict):
        raise TransferRunnerError("transfer_manifest_invalid")
    return payload


def _validate_manifest(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    if payload.get("schema_version") != TRANSFER_RUN_MANIFEST_SCHEMA_VERSION:
        raise TransferRunnerError("transfer_manifest_schema_mismatch")
    plan = payload.get("plan")
    if not isinstance(plan, Mapping):
        raise TransferRunnerError("transfer_manifest_plan_missing")
    if plan.get("schema_version") != TRANSFER_PLAN_SCHEMA_VERSION:
        raise TransferRunnerError("transfer_plan_schema_mismatch")
    if plan.get("plan_sha256") != _plan_digest(plan):
        raise TransferRunnerError("transfer_plan_hash_mismatch")
    model = plan.get("model")
    if not isinstance(model, Mapping) or model.get("canonical_model") != TRANSFER_MODEL:
        raise TransferRunnerError("transfer_plan_model_mismatch")
    target = plan.get("target")
    expected_target = {
        "source_case": "b2",
        "condition_id": "b2_policy_conditioned_org",
        "roster_origin": ROSTER_ORIGIN_FRESH,
        "fixed_protocol_landscape": True,
        "new_target_arms": ["Text", "Exec"],
        "fresh_reference": "existing_main_study_b2_only",
    }
    if target != expected_target:
        raise TransferRunnerError("transfer_plan_target_mismatch")
    canonical = plan.get("canonical_bundle")
    if not isinstance(canonical, Mapping) or canonical.get("bundle_sha256") != CANONICAL_V2_BUNDLE_SHA256:
        raise TransferRunnerError("transfer_plan_bundle_mismatch")
    batches = plan.get("batches")
    if not isinstance(batches, list) or len(batches) != 30:
        raise TransferRunnerError("transfer_plan_batches_invalid")
    targets = [
        item
        for batch in batches
        if isinstance(batch, Mapping)
        for item in (batch.get("targets") or [])
    ]
    if len(targets) != 60:
        raise TransferRunnerError("transfer_plan_targets_invalid")
    if {
        str(item.get("arm") or "")
        for item in targets
        if isinstance(item, Mapping)
    } != set(TRANSFER_ARMS):
        raise TransferRunnerError("transfer_plan_arms_invalid")
    return plan


def _selected_targets(
    plan: Mapping[str, Any],
    *,
    arms: Sequence[str],
    batch_ids: Sequence[str],
    workloads: Sequence[str],
    seeds: Sequence[int],
) -> tuple[list[Mapping[str, Any]], list[tuple[Mapping[str, Any], Mapping[str, Any]]]]:
    selected_batches = _select_batches(
        plan,
        batch_ids=batch_ids,
        workloads=workloads,
        seeds=seeds,
    )
    selected: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    wanted = set(arms)
    for batch in selected_batches:
        targets = batch.get("targets")
        if not isinstance(targets, list):
            raise TransferRunnerError("transfer_plan_targets_invalid")
        for target in targets:
            if not isinstance(target, Mapping):
                raise TransferRunnerError("transfer_plan_targets_invalid")
            if target.get("arm") in wanted:
                selected.append((batch, target))
    if not selected:
        raise TransferRunnerError("transfer_selection_empty")
    return selected_batches, selected


def _target_source_argv(
    *,
    plan: Mapping[str, Any],
    batch: Mapping[str, Any],
    target: Mapping[str, Any],
    output_root: Path,
    max_parallel: int,
    dry_run: bool,
    resume: bool,
    binding: EvaluatorBinding | None,
    bundle_path: Path,
) -> list[str]:
    """Specialize the canonical source argv to exactly one B2 transfer arm."""

    argv = _source_batch_argv(
        plan=plan,
        batch=batch,
        output_root=output_root,
        max_parallel=max_parallel,
        dry_run=dry_run,
        resume=resume,
        binding=binding,
    )
    try:
        argv[argv.index("--cases") + 1] = "b2"
    except ValueError as exc:
        raise TransferRunnerError("transfer_source_argv_cases_missing") from exc
    arm = str(target.get("arm") or "")
    config = _ARM_CONFIGURATION.get(arm)
    if config is None:
        raise TransferRunnerError("transfer_plan_arms_invalid")
    argv.extend(
        (
            "--experiment-phase",
            "capability_transfer",
            "--arm-map",
            json.dumps({"b2": config["arm_id"]}, separators=(",", ":")),
            "--transfer-arm",
            config["arm_id"],
            "--transfer-source-repository",
            "canonical_v2",
            "--transfer-roster-origin",
            ROSTER_ORIGIN_FRESH,
            "--transfer-capability-form",
            config["capability_form"],
            "--transfer-source-bundle",
            str(bundle_path),
            "--fixed-protocol-landscape",
        )
    )
    return argv


def run_transfer(
    *,
    arm: str = "both",
    model: str | None = None,
    output_root: Path | None = None,
    manifest_path: Path | None = None,
    max_parallel: int = 1,
    dry_run: bool = False,
    resume: bool = False,
    retry_failed: bool = False,
    evaluator_bindings_path: Path | None = None,
    batch_ids: Sequence[str] = (),
    workloads: Sequence[str] = (),
    seeds: Sequence[int] = (),
) -> TransferRunResult:
    """Run selected final transfer targets serially through the canonical runner."""

    if max_parallel < 1:
        raise TransferRunnerError("transfer_max_parallel_must_be_positive")
    if retry_failed and not resume:
        raise TransferRunnerError("transfer_retry_failed_requires_resume")
    if dry_run and resume:
        raise TransferRunnerError("transfer_dry_run_resume_not_supported")
    requested_arms = _normalize_arm_selection(arm)
    canonical_model = _require_model(model)
    default_root = (
        output_root or default_output_root() / "transfer-v2"
    ).expanduser().resolve()
    destination_manifest = (
        manifest_path.expanduser().resolve()
        if manifest_path is not None
        else default_root / "transfer_manifest.json"
    )
    if resume:
        payload = _read_manifest(destination_manifest)
        plan = _validate_manifest(payload)
        plan_root = Path(str(plan.get("output_root") or "")).expanduser().resolve()
        if output_root is not None and default_root != plan_root:
            raise TransferRunnerError("transfer_resume_output_root_mismatch")
        if str(plan["model"].get("canonical_model") or "") != canonical_model:
            raise TransferRunnerError("transfer_resume_model_mismatch")
        active_root = plan_root
    else:
        payload = build_transfer_manifest(
            model=canonical_model,
            output_root=default_root,
            max_parallel=max_parallel,
        )
        plan = _validate_manifest(payload)
        active_root = default_root
        if destination_manifest.exists():
            raise TransferRunnerError("transfer_manifest_already_exists")

    selected_batches, selected = _selected_targets(
        plan,
        arms=requested_arms,
        batch_ids=batch_ids,
        workloads=workloads,
        seeds=seeds,
    )
    bindings = (
        load_evaluator_bindings(evaluator_bindings_path)
        if evaluator_bindings_path is not None
        else None
    )
    if not dry_run:
        # This happens before the first source argv is built/invoked.  Missing
        # author-published formal evaluator identities therefore cannot trigger
        # a source child or provider request.
        bindings = _require_selected_bindings(selected_batches, bindings)
    bundle_path, _, bundle_digest = _canonical_bundle()
    if bundle_digest != str(plan["canonical_bundle"].get("bundle_sha256") or ""):
        raise TransferRunnerError("transfer_plan_bundle_mismatch")
    runtime = plan.get("model", {}).get("runtime")
    if not isinstance(runtime, Mapping):
        raise TransferRunnerError("transfer_plan_runtime_missing")

    execution = payload.get("execution")
    if not isinstance(execution, dict):
        raise TransferRunnerError("transfer_execution_missing")
    execution_targets = execution.get("targets")
    if not isinstance(execution_targets, dict):
        raise TransferRunnerError("transfer_execution_targets_invalid")
    dry_root = active_root / "source-dry-run" if dry_run else active_root / "batches"
    if dry_run:
        execution["source_dry_run_root"] = str(dry_root)

    completed = 0
    failed = 0
    stop_after_failure = False
    for batch, target in selected:
        target_id = str(target.get("target_id") or "")
        if not target_id:
            raise TransferRunnerError("transfer_plan_targets_invalid")
        if stop_after_failure:
            execution_targets[target_id] = {
                "status": "not_run",
                "reason": "prior_source_target_failed",
            }
            continue
        arm_name = str(target["arm"])
        batch_id = str(batch["batch_id"])
        target_root = (
            dry_root / arm_name / batch_id
            if dry_run
            else Path(str(target["output_root"])).expanduser().resolve()
        )
        binding = bindings.get(str(batch["pack"])) if bindings is not None else None
        argv = _target_source_argv(
            plan=plan,
            batch=batch,
            target=target,
            output_root=target_root,
            max_parallel=max_parallel,
            dry_run=dry_run,
            resume=resume and not dry_run,
            binding=binding,
            bundle_path=bundle_path,
        )
        result_code = _invoke_source_batch(argv, runtime)
        status = _batch_status(target_root, result_code)
        status["arm"] = target["arm_id"]
        status["source_case"] = "b2"
        status["fixed_protocol_landscape"] = True
        if dry_run:
            status["status"] = "dry_run" if result_code == 0 else "failed"
        execution_targets[target_id] = status
        if result_code == 0:
            completed += 1
        else:
            failed += 1
            stop_after_failure = True
        payload["resource_policy"] = {
            "max_parallel_scope": "within_each_single-B2 source invocation",
            "requested_max_parallel": max_parallel,
        }
        execution["status"] = (
            "planned" if dry_run and failed == 0 else ("failed" if failed else "running")
        )
        write_manifest(payload, destination_manifest)

    if failed:
        execution["status"] = "failed"
    elif dry_run:
        execution["status"] = "planned"
    elif len(selected) == 60:
        execution["status"] = "completed"
    else:
        execution["status"] = "partial"
    write_manifest(payload, destination_manifest)
    return TransferRunResult(
        manifest_path=destination_manifest,
        output_root=active_root,
        model=canonical_model,
        status=str(execution["status"]),
        total_target_runs=60,
        selected_target_runs=len(selected),
        selected_workload_seed_batches=len(selected_batches),
        completed_target_runs=completed,
        failed_target_runs=failed,
        dry_run=dry_run,
    )


__all__ = [
    "TRANSFER_ARMS",
    "TRANSFER_MODEL",
    "TRANSFER_PLAN_SCHEMA_VERSION",
    "TRANSFER_RUN_MANIFEST_SCHEMA_VERSION",
    "TransferRunResult",
    "TransferRunnerError",
    "build_transfer_manifest",
    "run_transfer",
]
