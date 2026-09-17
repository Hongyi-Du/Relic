"""Source-backed orchestration for the Relic paper's paired main-study cells.

The public release retains the hci baseline runner as the execution authority:
one invocation of :mod:`tools.run_org_baselines` owns a fresh-process,
deterministically ordered B0--B3 pack/seed block.  This module is deliberately
thin.  It expands the paper's 10-pack x 3-seed design into 30 such source
batches; it does not substitute the older per-cell compatibility worker.

Formal execution can use the public evaluator source on the host by default.
An explicit per-pack container binding remains available as reproducibility
provenance, and ``strict_reproducibility`` restores the digest/platform/hash
gate for operators who need the historical formal boundary.
"""

from __future__ import annotations

import json
import os
import re
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from environments.org_env.config.baseline_conditions import resolve_condition
from relic.cell_spec import (
    SOURCE_BRANCH,
    SOURCE_COMMIT,
    SOURCE_GIT_TREE,
    SOURCE_REPOSITORY,
)
from relic.manifest import load_yaml, recommended_parallelism, visible_memory_gib, write_manifest
from relic.paths import config_root, default_output_root, project_root
from relic.research.hashing import stable_hash


SOURCE_MAIN_MANIFEST_SCHEMA_VERSION = "relic-source-main-run-manifest-v1"
SOURCE_MAIN_PLAN_SCHEMA_VERSION = "relic-source-main-plan-v1"
SOURCE_BASELINE_RUNNER = "tools/run_org_baselines.py"
SOURCE_CASES = ("b0", "b1", "b2", "b3")
_IMAGE_DIGEST = re.compile(r"^.+@sha256:[0-9a-fA-F]{64}$")
_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")
_CONTAINER_PLATFORM = re.compile(
    r"^[a-z0-9][a-z0-9._-]*/[a-z0-9][a-z0-9._-]*"
    r"(?:/[a-z0-9][a-z0-9._-]*)?$"
)
_ALLOWED_EVALUATOR_BACKENDS = frozenset({"docker", "apptainer"})
_SCOPED_RUNTIME_KEYS = frozenset(
    {
        "ORG_LLM_PROVIDER",
        "ORG_LLM_MODEL",
        "ORG_LLM_WIRE_API",
        "ORG_LLM_JSON_TRANSPORT",
        "ORG_LLM_REASONING_EFFORT",
        "ORG_LLM_REQUEST_TIMEOUT_SECONDS",
        "ORG_LLM_MAX_RETRIES",
        "ORG_LLM_RETRY_BACKOFF_SECONDS",
        "ORG_LLM_STORE_RESPONSES",
    }
)
_SCOPED_PROVIDER_BRIDGE_KEYS = frozenset(
    {
        "ORG_LLM_API_KEY",
        "ORG_LLM_BASE_URL",
        "ORG_LLM_DEFAULT_HEADERS_JSON",
    }
)


class SourceMainRunnerError(RuntimeError):
    """Stable, non-secret failure code for the source-backed main runner."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class EvaluatorBinding:
    """The explicit, per-pack evaluator identity supplied by an operator.

    Hashes and platform are retained as metadata even in the default
    non-strict mode.  Empty values mean that the operator supplied a useful
    runtime locator without author-published qualification values.
    """

    backend: str
    container_image: str
    container_platform: str
    environment_hash: str
    qualification_plan_hash: str

    def document(self) -> dict[str, str]:
        return {
            "backend": self.backend,
            "container_image": self.container_image,
            "container_platform": self.container_platform,
            "environment_hash": self.environment_hash,
            "qualification_plan_hash": self.qualification_plan_hash,
        }


@dataclass(frozen=True)
class SourceMainRunResult:
    manifest_path: Path
    output_root: Path
    model: str
    status: str
    total_cells: int
    selected_batches: int
    completed_batches: int
    failed_batches: int
    dry_run: bool
    visible_memory_gib: float
    recommended_max_parallel: int
    requested_max_parallel: int
    parallelism_warning: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest": str(self.manifest_path),
            "output_root": str(self.output_root),
            "model": self.model,
            "status": self.status,
            "paper_design": "10 workloads x 3 seeds x 4 arms",
            "total_cells": self.total_cells,
            "selected_batches": self.selected_batches,
            "selected_cells": self.selected_batches * len(SOURCE_CASES),
            "completed_batches": self.completed_batches,
            "failed_batches": self.failed_batches,
            "dry_run": self.dry_run,
            "executor": SOURCE_BASELINE_RUNNER,
            "resource_advice": {
                "visible_memory_gib": self.visible_memory_gib,
                "budget_gib_per_active_cell": 16,
                "recommended_max_parallel": self.recommended_max_parallel,
                "requested_max_parallel": self.requested_max_parallel,
                "warning": self.parallelism_warning,
            },
        }


def _require_mapping(value: object, code: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SourceMainRunnerError(code)
    return value


def _nonempty_text(value: object, code: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise SourceMainRunnerError(code)
    return text


def _plan_digest(plan: Mapping[str, Any]) -> str:
    return stable_hash({key: value for key, value in plan.items() if key != "plan_sha256"})


def _resource_policy(max_parallel: int) -> dict[str, Any]:
    """Record the handoff's visible-RAM guidance for every launcher invocation."""

    memory_gib = round(max(0.0, visible_memory_gib()), 2)
    recommended = recommended_parallelism(memory_gib)
    return {
        "max_parallel_scope": "cases_within_each_serial_source_batch",
        "requested_max_parallel": max_parallel,
        "visible_memory_gib": memory_gib,
        "budget_gib_per_active_cell": 16,
        "recommended_max_parallel": recommended,
        "warning": max_parallel > recommended,
    }


def _load_study() -> dict[str, Any]:
    try:
        study = load_yaml(config_root() / "main-study.yaml")
    except (OSError, TypeError, ValueError) as exc:
        raise SourceMainRunnerError("source_main_study_config_unreadable") from exc
    if study.get("schema_version") != "relic-main-study-v1":
        raise SourceMainRunnerError("source_main_study_schema_mismatch")
    if study.get("study") != "relic-main-v1":
        raise SourceMainRunnerError("source_main_study_identity_mismatch")
    return study


def _source_runner_config(study: Mapping[str, Any]) -> Mapping[str, Any]:
    source = _require_mapping(study.get("source_runner"), "source_runner_config_missing")
    expected_fields = {
        "schema_version",
        "baseline_runner",
        "execution_profile",
        "mechanism_ablations",
        "resource_ceilings",
    }
    if set(source) != expected_fields:
        raise SourceMainRunnerError("source_runner_config_shape_invalid")
    if source.get("schema_version") != "relic-source-runner-v1":
        raise SourceMainRunnerError("source_runner_config_schema_mismatch")
    if source.get("baseline_runner") != SOURCE_BASELINE_RUNNER:
        raise SourceMainRunnerError("source_runner_path_mismatch")
    if source.get("execution_profile") != "native":
        raise SourceMainRunnerError("programbench_execution_profile_not_available_in_relic_release")
    ceilings = _require_mapping(source.get("resource_ceilings"), "source_runner_ceilings_missing")
    required = {
        "max_llm_calls",
        "max_llm_requested_tokens",
        "max_llm_prompt_characters",
        "max_primary_actions",
        "max_ticks",
    }
    if set(ceilings) != required:
        raise SourceMainRunnerError("source_runner_ceilings_shape_invalid")
    try:
        normalized = {key: int(ceilings[key]) for key in required}
    except (TypeError, ValueError) as exc:
        raise SourceMainRunnerError("source_runner_ceilings_invalid") from exc
    if any(value <= 0 for value in normalized.values()):
        raise SourceMainRunnerError("source_runner_ceilings_invalid")
    if normalized["max_ticks"] != int(study.get("ticks", 0) or 0):
        raise SourceMainRunnerError("source_runner_max_ticks_mismatch")
    ablations = source.get("mechanism_ablations")
    if ablations != ["work_rhythm"] or bool(study.get("work_rhythm_enabled")):
        raise SourceMainRunnerError("source_runner_work_rhythm_policy_mismatch")
    return {
        **source,
        "mechanism_ablations": ("work_rhythm",),
        "resource_ceilings": normalized,
    }


def _load_model_config(model: str, study: Mapping[str, Any]) -> dict[str, Any]:
    requested = _nonempty_text(model, "source_main_model_required").lower()
    models = [str(value).lower() for value in study.get("models", [])]
    if requested not in models:
        raise SourceMainRunnerError("source_main_model_unknown")
    try:
        payload = load_yaml(config_root() / "models" / f"{requested}.yaml")
    except (OSError, TypeError, ValueError) as exc:
        raise SourceMainRunnerError("source_main_model_config_unreadable") from exc
    if str(payload.get("model") or "").lower() != requested:
        raise SourceMainRunnerError("source_main_model_config_identity_mismatch")
    return payload


def _runtime_model(
    model_config: Mapping[str, Any],
    override: str | None = None,
    *,
    allow_unresolved: bool = False,
) -> str:
    explicit = str(override or "").strip()
    if explicit:
        return explicit
    # Keep paper model IDs stable while allowing an operator's gateway to use
    # its own deployment name.  The generic variables cover GPT and custom
    # gateways; the model-specific variable remains the compatibility path for
    # the Claude arm.
    for environment_name in (
        "RELIC_RUNTIME_MODEL",
        "ORG_LLM_RUNTIME_MODEL",
        "OPENAI_MODEL",
        "ORG_LLM_MODEL",
    ):
        value = os.environ.get(environment_name, "").strip()
        if value:
            return value
    configured_env = str(model_config.get("runtime_model_env") or "").strip()
    if configured_env:
        value = os.environ.get(configured_env, "").strip()
        if not value:
            if allow_unresolved:
                return ""
            raise SourceMainRunnerError(
                "runtime_model_binding_missing:"
                f"{configured_env}; set {configured_env} to the provider "
                "deployment name or pass --runtime-model"
            )
        return value
    return _nonempty_text(
        model_config.get("runtime_model_default") or model_config.get("model"),
        "source_main_runtime_model_missing",
    )


def _runtime_environment(
    model_config: Mapping[str, Any],
    *,
    runtime_model: str,
) -> dict[str, str]:
    provider = _nonempty_text(model_config.get("provider"), "source_main_provider_missing")
    required = (
        "wire_api",
        "json_transport",
        "reasoning_effort",
        "request_timeout_seconds",
        "max_retries",
        "retry_backoff_seconds",
    )
    missing = [key for key in required if model_config.get(key) in (None, "")]
    if missing:
        raise SourceMainRunnerError("source_main_model_runtime_incomplete")
    environment = {
        "ORG_LLM_PROVIDER": provider,
        "ORG_LLM_MODEL": runtime_model,
        "ORG_LLM_WIRE_API": str(model_config["wire_api"]),
        "ORG_LLM_JSON_TRANSPORT": str(model_config["json_transport"]),
        "ORG_LLM_REASONING_EFFORT": str(model_config["reasoning_effort"]),
        "ORG_LLM_REQUEST_TIMEOUT_SECONDS": str(model_config["request_timeout_seconds"]),
        "ORG_LLM_MAX_RETRIES": str(model_config["max_retries"]),
        "ORG_LLM_RETRY_BACKOFF_SECONDS": str(model_config["retry_backoff_seconds"]),
        # Never let a local config flip a paper cell to persisted provider
        # responses.  This is a non-secret transport identity field.
        "ORG_LLM_STORE_RESPONSES": "0",
    }
    if set(environment) != _SCOPED_RUNTIME_KEYS:
        raise AssertionError("source_main_runtime_environment_shape")
    return environment


def _load_workloads(study: Mapping[str, Any]) -> tuple[dict[str, str], ...]:
    names = tuple(str(value).lower() for value in study.get("workloads", []))
    if len(names) != 10 or len(set(names)) != 10:
        raise SourceMainRunnerError("source_main_workload_count_invalid")
    workloads: list[dict[str, str]] = []
    seen_packs: set[str] = set()
    for name in names:
        try:
            payload = load_yaml(config_root() / "workloads" / f"{name}.yaml")
        except (OSError, TypeError, ValueError) as exc:
            raise SourceMainRunnerError("source_main_workload_config_unreadable") from exc
        if str(payload.get("workload") or "").lower() != name:
            raise SourceMainRunnerError("source_main_workload_config_identity_mismatch")
        pack = _nonempty_text(payload.get("frozen_pack_id"), "source_main_pack_missing")
        if pack in seen_packs:
            raise SourceMainRunnerError("source_main_pack_duplicate")
        seen_packs.add(pack)
        workloads.append({"workload": name, "pack": pack})
    return tuple(workloads)


def _source_conditions(study: Mapping[str, Any]) -> tuple[dict[str, str], ...]:
    observed = tuple(str(value).lower() for value in study.get("arms", []))
    if observed != SOURCE_CASES:
        raise SourceMainRunnerError("source_main_arms_mismatch")
    conditions = []
    for arm in SOURCE_CASES:
        condition = resolve_condition(arm)
        conditions.append({"arm": arm, "condition_id": condition.condition_id})
    if conditions[-1]["condition_id"] != "b3_full_sociogenesis":
        raise SourceMainRunnerError("source_main_b3_condition_mismatch")
    return tuple(conditions)


def build_source_main_manifest(
    *,
    model: str,
    output_root: Path | None = None,
    max_parallel: int = 1,
    strict_reproducibility: bool = False,
    runtime_model: str | None = None,
) -> dict[str, Any]:
    """Freeze the 30 paired source batches / 120 paper cells for one model."""

    if max_parallel < 1:
        raise SourceMainRunnerError("source_main_max_parallel_must_be_positive")
    study = _load_study()
    source_config = _source_runner_config(study)
    model_config = _load_model_config(model, study)
    # Planning is intentionally provider-agnostic.  A model-specific gateway
    # alias may be unavailable while the user is still materializing a dry
    # plan; the non-dry runner checks the empty value before any child starts.
    resolved_runtime_model = _runtime_model(
        model_config,
        runtime_model,
        allow_unresolved=True,
    )
    runtime_environment = _runtime_environment(
        model_config,
        runtime_model=resolved_runtime_model,
    )
    workloads = _load_workloads(study)
    conditions = _source_conditions(study)
    try:
        seeds = tuple(int(value) for value in study["seeds"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SourceMainRunnerError("source_main_seeds_invalid") from exc
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise SourceMainRunnerError("source_main_seeds_invalid")
    destination = (output_root or default_output_root() / "main-study").expanduser().resolve()
    batches: list[dict[str, Any]] = []
    for workload in workloads:
        for seed in seeds:
            batch_id = f"{workload['workload']}__seed{seed}"
            cells = [
                {
                    "cell_id": f"{model_config['model']}__{workload['workload'].upper()}__{condition['arm'].upper()}__seed{seed}",
                    "arm": condition["arm"],
                    "condition_id": condition["condition_id"],
                    "workload": workload["workload"],
                    "pack": workload["pack"],
                    "seed": seed,
                    "source_case": condition["arm"],
                }
                for condition in conditions
            ]
            batches.append(
                {
                    "batch_id": batch_id,
                    "workload": workload["workload"],
                    "pack": workload["pack"],
                    "seed": seed,
                    "source_runner": SOURCE_BASELINE_RUNNER,
                    "output_root": str(destination / "batches" / batch_id),
                    "cases": cells,
                }
            )
    if len(batches) != 30 or sum(len(batch["cases"]) for batch in batches) != 120:
        raise SourceMainRunnerError("source_main_cell_count_invalid")
    plan: dict[str, Any] = {
        "schema_version": SOURCE_MAIN_PLAN_SCHEMA_VERSION,
        "study": str(study["study"]),
        "paper_design": "10 workloads x 3 seeds x 4 arms",
        "output_root": str(destination),
        "source": {
            "repository": SOURCE_REPOSITORY,
            "branch": SOURCE_BRANCH,
            "commit": SOURCE_COMMIT,
            "git_tree": SOURCE_GIT_TREE,
            "baseline_runner": SOURCE_BASELINE_RUNNER,
            "execution_profile": "native",
        },
        "model": {
            "canonical_model": str(model_config["model"]),
            "paper_label": str(model_config.get("paper_label") or ""),
            "runtime_model_env": str(model_config.get("runtime_model_env") or ""),
            "runtime_model": resolved_runtime_model,
            "runtime": runtime_environment,
        },
        "ticks": int(study["ticks"]),
        "checkpoint_every": int(study["checkpoint_every"]),
        "sprint_ticks": int(study["sprint_ticks"]),
        "mechanism_ablations": list(source_config["mechanism_ablations"]),
        "resource_ceilings": dict(source_config["resource_ceilings"]),
        "evaluator_binding": {
            "required_for_formal_execution": bool(strict_reproducibility),
            "binding_file_required": bool(strict_reproducibility),
            "per_pack": True,
            "default_mode": "local",
            "strict_reproducibility": bool(strict_reproducibility),
            "unpublished_values_are_not_fabricated": True,
        },
        "batches": batches,
    }
    plan["plan_sha256"] = _plan_digest(plan)
    return {
        "schema_version": SOURCE_MAIN_MANIFEST_SCHEMA_VERSION,
        "plan": plan,
        "execution": {
            "status": "planned",
            "source_dry_run_root": None,
            "batches": {},
        },
        "resource_policy": _resource_policy(max_parallel),
    }


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SourceMainRunnerError("source_main_manifest_missing") from exc
    except (OSError, ValueError) as exc:
        raise SourceMainRunnerError("source_main_manifest_unreadable") from exc
    if not isinstance(payload, dict):
        raise SourceMainRunnerError("source_main_manifest_invalid")
    return payload


def _validate_manifest(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    if payload.get("schema_version") != SOURCE_MAIN_MANIFEST_SCHEMA_VERSION:
        raise SourceMainRunnerError("source_main_manifest_schema_mismatch")
    plan = _require_mapping(payload.get("plan"), "source_main_manifest_plan_missing")
    if plan.get("schema_version") != SOURCE_MAIN_PLAN_SCHEMA_VERSION:
        raise SourceMainRunnerError("source_main_plan_schema_mismatch")
    if plan.get("plan_sha256") != _plan_digest(plan):
        raise SourceMainRunnerError("source_main_plan_hash_mismatch")
    source = _require_mapping(plan.get("source"), "source_main_plan_source_missing")
    expected_source = {
        "repository": SOURCE_REPOSITORY,
        "branch": SOURCE_BRANCH,
        "commit": SOURCE_COMMIT,
        "git_tree": SOURCE_GIT_TREE,
        "baseline_runner": SOURCE_BASELINE_RUNNER,
        "execution_profile": "native",
    }
    if dict(source) != expected_source:
        raise SourceMainRunnerError("source_main_plan_source_mismatch")
    if plan.get("mechanism_ablations") != ["work_rhythm"]:
        raise SourceMainRunnerError("source_main_plan_work_rhythm_policy_mismatch")
    evaluator_binding = plan.get("evaluator_binding")
    if not isinstance(evaluator_binding, Mapping):
        raise SourceMainRunnerError("source_main_plan_evaluator_binding_missing")
    if "strict_reproducibility" in evaluator_binding and not isinstance(
        evaluator_binding.get("strict_reproducibility"), bool
    ):
        raise SourceMainRunnerError("source_main_plan_evaluator_binding_invalid")
    batches = plan.get("batches")
    if not isinstance(batches, list) or len(batches) != 30:
        raise SourceMainRunnerError("source_main_plan_batches_invalid")
    cases = [case for batch in batches if isinstance(batch, Mapping) for case in batch.get("cases", [])]
    if len(cases) != 120:
        raise SourceMainRunnerError("source_main_plan_cells_invalid")
    return plan


def _binding_from_mapping(
    pack: str,
    value: object,
    *,
    strict_reproducibility: bool = True,
) -> EvaluatorBinding:
    raw = _require_mapping(value, f"evaluator_binding_invalid:{pack}")
    expected = {
        "backend",
        "container_image",
        "container_platform",
        "environment_hash",
        "qualification_plan_hash",
    }
    unknown = set(raw) - expected
    if unknown or not {"backend", "container_image"} <= set(raw):
        raise SourceMainRunnerError(f"evaluator_binding_shape_invalid:{pack}")
    binding = EvaluatorBinding(
        backend=_nonempty_text(raw.get("backend"), f"evaluator_binding_backend_missing:{pack}"),
        container_image=_nonempty_text(
            raw.get("container_image"), f"evaluator_binding_image_missing:{pack}"
        ),
        container_platform=str(raw.get("container_platform") or "").strip(),
        environment_hash=str(raw.get("environment_hash") or "").strip(),
        qualification_plan_hash=str(raw.get("qualification_plan_hash") or "").strip(),
    )
    if binding.backend not in _ALLOWED_EVALUATOR_BACKENDS:
        raise SourceMainRunnerError(f"evaluator_binding_backend_invalid:{pack}")
    if binding.container_platform and not _CONTAINER_PLATFORM.fullmatch(
        binding.container_platform
    ):
        raise SourceMainRunnerError(f"evaluator_binding_platform_invalid:{pack}")
    if strict_reproducibility and binding.container_platform != "linux/amd64":
        raise SourceMainRunnerError(f"evaluator_binding_platform_invalid:{pack}")
    if strict_reproducibility and not _IMAGE_DIGEST.fullmatch(binding.container_image):
        raise SourceMainRunnerError(f"evaluator_binding_image_not_digest_pinned:{pack}")
    if strict_reproducibility and not _SHA256.fullmatch(binding.environment_hash):
        raise SourceMainRunnerError(f"evaluator_binding_environment_hash_invalid:{pack}")
    if strict_reproducibility and not _SHA256.fullmatch(binding.qualification_plan_hash):
        raise SourceMainRunnerError(f"evaluator_binding_qualification_hash_invalid:{pack}")
    return binding


def load_evaluator_bindings(
    path: Path,
    *,
    strict_reproducibility: bool = True,
) -> dict[str, EvaluatorBinding]:
    """Load an explicit JSON mapping of frozen pack IDs to evaluator bindings.

    The accepted file shape is either ``{"pack": {...}}`` or the same mapping
    nested under ``{"bindings": {...}}`` with an optional schema version.
    """

    try:
        payload = json.loads(path.expanduser().read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SourceMainRunnerError("evaluator_bindings_file_missing") from exc
    except (OSError, ValueError) as exc:
        raise SourceMainRunnerError("evaluator_bindings_file_invalid") from exc
    raw = _require_mapping(payload, "evaluator_bindings_file_invalid")
    if "bindings" in raw:
        allowed = {"schema_version", "bindings"}
        if set(raw) - allowed:
            raise SourceMainRunnerError("evaluator_bindings_file_shape_invalid")
        raw = _require_mapping(raw["bindings"], "evaluator_bindings_file_invalid")
    if not raw:
        raise SourceMainRunnerError("evaluator_bindings_file_empty")
    return {
        _nonempty_text(pack, "evaluator_binding_pack_empty"): _binding_from_mapping(
            str(pack),
            binding,
            strict_reproducibility=strict_reproducibility,
        )
        for pack, binding in raw.items()
    }


def _select_batches(
    plan: Mapping[str, Any],
    *,
    batch_ids: Sequence[str] = (),
    workloads: Sequence[str] = (),
    seeds: Sequence[int] = (),
) -> list[Mapping[str, Any]]:
    requested_batches = {str(value).strip().lower() for value in batch_ids if str(value).strip()}
    requested_workloads = {str(value).strip().lower() for value in workloads if str(value).strip()}
    requested_seeds = {int(value) for value in seeds}
    selected: list[Mapping[str, Any]] = []
    available_batches: set[str] = set()
    available_workloads: set[str] = set()
    available_seeds: set[int] = set()
    for batch in plan["batches"]:
        item = _require_mapping(batch, "source_main_plan_batches_invalid")
        batch_id = str(item.get("batch_id") or "").lower()
        workload = str(item.get("workload") or "").lower()
        try:
            seed = int(item.get("seed"))
        except (TypeError, ValueError) as exc:
            raise SourceMainRunnerError("source_main_plan_batch_seed_invalid") from exc
        available_batches.add(batch_id)
        available_workloads.add(workload)
        available_seeds.add(seed)
        if requested_batches and batch_id not in requested_batches:
            continue
        if requested_workloads and workload not in requested_workloads:
            continue
        if requested_seeds and seed not in requested_seeds:
            continue
        selected.append(item)
    unknown_batches = requested_batches - available_batches
    unknown_workloads = requested_workloads - available_workloads
    unknown_seeds = requested_seeds - available_seeds
    if unknown_batches:
        raise SourceMainRunnerError("source_main_batch_unknown")
    if unknown_workloads:
        raise SourceMainRunnerError("source_main_workload_unknown")
    if unknown_seeds:
        raise SourceMainRunnerError("source_main_seed_unknown")
    if not selected:
        raise SourceMainRunnerError("source_main_selection_empty")
    return selected


def _require_selected_bindings(
    selected: Sequence[Mapping[str, Any]],
    bindings: Mapping[str, EvaluatorBinding] | None,
) -> Mapping[str, EvaluatorBinding]:
    if bindings is None:
        raise SourceMainRunnerError("formal_evaluator_bindings_required")
    required = {str(batch["pack"]) for batch in selected}
    missing = sorted(required - set(bindings))
    if missing:
        raise SourceMainRunnerError(
            "formal_evaluator_binding_missing:" + ",".join(missing)
        )
    return bindings


@contextmanager
def _scoped_runtime_environment(values: Mapping[str, str]) -> Iterator[None]:
    """Set source-runner transport identity without leaking it to the caller."""

    allowed = _SCOPED_RUNTIME_KEYS | _SCOPED_PROVIDER_BRIDGE_KEYS
    if not _SCOPED_RUNTIME_KEYS <= set(values) or set(values) - allowed:
        raise SourceMainRunnerError("source_main_runtime_environment_invalid")
    previous = {key: os.environ.get(key) for key in values}
    try:
        os.environ.update(values)
        yield
    finally:
        for key, old_value in previous.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


def _source_invocation_environment(runtime: Mapping[str, str]) -> dict[str, str]:
    """Bridge documented Relic OpenAI variables into source child allow-lists.

    The hci runner deliberately passes only ``ORG_LLM_*`` provider variables to
    a fresh condition process.  Relic documents the standard ``OPENAI_*`` names,
    so add aliases only for the duration of source planning/launch.  They never
    enter the outer plan or the source runner's public case environment.
    """

    values = dict(runtime)
    aliases = {
        "ORG_LLM_API_KEY": "OPENAI_API_KEY",
        "ORG_LLM_BASE_URL": "OPENAI_BASE_URL",
        "ORG_LLM_DEFAULT_HEADERS_JSON": "RELIC_OPENAI_DEFAULT_HEADERS_JSON",
    }
    for destination, source in aliases.items():
        if os.environ.get(destination):
            continue
        source_value = os.environ.get(source)
        if source_value:
            values[destination] = source_value
    return values


def _source_batch_argv(
    *,
    plan: Mapping[str, Any],
    batch: Mapping[str, Any],
    output_root: Path,
    max_parallel: int,
    dry_run: bool,
    resume: bool,
    binding: EvaluatorBinding | None,
    strict_reproducibility: bool = False,
) -> list[str]:
    model = _require_mapping(plan.get("model"), "source_main_plan_model_missing")
    runtime = _require_mapping(model.get("runtime"), "source_main_plan_model_runtime_missing")
    runtime_model = str(model.get("runtime_model") or "").strip()
    if not runtime_model and not dry_run:
        configured_env = str(model.get("runtime_model_env") or "").strip()
        raise SourceMainRunnerError(
            "runtime_model_binding_missing:"
            + (configured_env or "RELIC_RUNTIME_MODEL")
            + "; set the named variable to the provider deployment name or "
            "pass --runtime-model"
        )
    # Child dry-run manifests still need a non-empty identity for their
    # canonical case plans.  They never construct a provider, so use the paper
    # model ID as a visible placeholder while keeping the outer plan's runtime
    # model empty and actionable for the eventual run.
    runtime_model_for_argv = runtime_model or str(model.get("canonical_model") or "")
    ceilings = _require_mapping(plan.get("resource_ceilings"), "source_main_plan_ceilings_missing")
    argv = [
        "--cases",
        ",".join(SOURCE_CASES),
        "--dataset",
        str(batch["pack"]),
        "--repository-id",
        str(batch["pack"]),
        "--seed",
        str(batch["seed"]),
        "--ticks",
        str(plan["ticks"]),
        "--checkpoint-every",
        str(plan["checkpoint_every"]),
        "--sprint-ticks",
        str(plan["sprint_ticks"]),
        "--llm",
        "--provider",
        str(runtime["ORG_LLM_PROVIDER"]),
        "--model",
        runtime_model_for_argv,
        "--max-parallel",
        str(max_parallel),
        "--max-llm-calls",
        str(ceilings["max_llm_calls"]),
        "--max-llm-requested-tokens",
        str(ceilings["max_llm_requested_tokens"]),
        "--max-llm-prompt-characters",
        str(ceilings["max_llm_prompt_characters"]),
        "--max-primary-actions",
        str(ceilings["max_primary_actions"]),
        "--max-ticks",
        str(ceilings["max_ticks"]),
        "--mechanism-ablations",
        ",".join(str(value) for value in plan["mechanism_ablations"]),
        "--execution-profile",
        "native",
        "--output-root",
        str(output_root),
    ]
    if dry_run:
        argv.append("--dry-run")
    if resume:
        argv.append("--resume")
    argv.extend(
        (
            "--evaluator-mode",
            "container" if binding is not None else "local",
        )
    )
    if strict_reproducibility:
        argv.append("--strict-reproducibility")
    if binding is not None:
        argv.extend(["--evaluator-backend", binding.backend])
        argv.extend(["--evaluator-container-image", binding.container_image])
        if binding.container_platform:
            argv.extend(["--evaluator-container-platform", binding.container_platform])
        if binding.environment_hash:
            argv.extend(
                ["--expected-evaluator-environment-hash", binding.environment_hash]
            )
        if binding.qualification_plan_hash:
            argv.extend(
                ["--expected-qualification-plan-hash", binding.qualification_plan_hash]
            )
    return argv


def _invoke_source_batch(argv: Sequence[str], runtime: Mapping[str, str]) -> int:
    root = project_root()
    root_text = str(root)
    inserted = False
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
        inserted = True
    try:
        from tools.run_org_baselines import main as source_main

        with _scoped_runtime_environment(_source_invocation_environment(runtime)):
            result = source_main(list(argv))
    except SystemExit as exc:
        if isinstance(exc.code, int):
            return exc.code
        return 1
    finally:
        if inserted:
            try:
                sys.path.remove(root_text)
            except ValueError:
                pass
    return int(result)


def _batch_status(batch_root: Path, returncode: int) -> dict[str, Any]:
    manifest_path = batch_root / "manifest.json"
    result: dict[str, Any] = {
        "returncode": returncode,
        "source_manifest": str(manifest_path),
        "status": "passed" if returncode == 0 else "failed",
    }
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return result
    cases = manifest.get("cases") if isinstance(manifest, Mapping) else None
    evaluator = manifest.get("evaluator") if isinstance(manifest, Mapping) else None
    if isinstance(evaluator, Mapping):
        result["evaluator"] = dict(evaluator)
    if isinstance(cases, list):
        result["cases"] = [
            {
                "case": item.get("case"),
                "condition_id": item.get("condition_id"),
                "status": item.get("status"),
                "failure_reason": item.get("failure_reason"),
            }
            for item in cases
            if isinstance(item, Mapping)
        ]
    return result


def run_source_main(
    *,
    model: str | None = None,
    output_root: Path | None = None,
    manifest_path: Path | None = None,
    max_parallel: int = 1,
    dry_run: bool = False,
    resume: bool = False,
    retry_failed: bool = False,
    evaluator_bindings_path: Path | None = None,
    strict_reproducibility: bool = False,
    runtime_model: str | None = None,
    batch_ids: Sequence[str] = (),
    workloads: Sequence[str] = (),
    seeds: Sequence[int] = (),
) -> SourceMainRunResult:
    """Run selected paired source batches serially, or materialize their dry plan.

    ``max_parallel`` is deliberately forwarded only to each source batch's four
    fresh condition processes.  Cross-batch scheduling would introduce a second
    execution regime beyond the hci runner, so this adapter runs batches in plan
    order and stops at the first failing batch.
    """

    if max_parallel < 1:
        raise SourceMainRunnerError("source_main_max_parallel_must_be_positive")
    if retry_failed and not resume:
        raise SourceMainRunnerError("source_main_retry_failed_requires_resume")
    if dry_run and resume:
        raise SourceMainRunnerError("source_main_dry_run_resume_not_supported")
    default_root = (output_root or default_output_root() / "main-study").expanduser().resolve()
    destination_manifest = (
        manifest_path.expanduser().resolve()
        if manifest_path is not None
        else default_root / "source_main_manifest.json"
    )
    if resume:
        payload = _read_manifest(destination_manifest)
        plan = _validate_manifest(payload)
        plan_root = Path(str(plan.get("output_root") or "")).expanduser().resolve()
        if output_root is not None and default_root != plan_root:
            raise SourceMainRunnerError("source_main_resume_output_root_mismatch")
        if model is not None and str(plan["model"].get("canonical_model") or "") != model:
            raise SourceMainRunnerError("source_main_resume_model_mismatch")
        planned_runtime_model = str(plan["model"].get("runtime_model") or "")
        if not planned_runtime_model and payload.get("execution", {}).get("status") == "planned":
            # A provider-free dry plan may leave the deployment name unset.
            # Resolve it once before execution; an executed plan keeps its identity.
            planned_runtime_model = _runtime_model(
                _load_model_config(str(plan["model"]["canonical_model"]), _load_study()),
                runtime_model,
            )
            plan["model"]["runtime_model"] = planned_runtime_model
            plan["model"]["runtime"]["ORG_LLM_MODEL"] = planned_runtime_model
            plan["plan_sha256"] = _plan_digest(plan)
        if runtime_model is not None and str(runtime_model).strip() != planned_runtime_model:
            raise SourceMainRunnerError("source_main_resume_runtime_model_mismatch")
        output_root = plan_root
    else:
        if model is None:
            raise SourceMainRunnerError("source_main_model_required")
        payload = build_source_main_manifest(
            model=model,
            output_root=default_root,
            max_parallel=max_parallel,
            strict_reproducibility=strict_reproducibility,
            runtime_model=runtime_model,
        )
        plan = _validate_manifest(payload)
        output_root = default_root
        if destination_manifest.exists():
            raise SourceMainRunnerError("source_main_manifest_already_exists")
    payload["resource_policy"] = _resource_policy(max_parallel)
    assert output_root is not None
    selected = _select_batches(
        plan,
        batch_ids=batch_ids,
        workloads=workloads,
        seeds=seeds,
    )
    plan_evaluator = _require_mapping(
        plan.get("evaluator_binding"), "source_main_plan_evaluator_binding_missing"
    )
    planned_strict = bool(plan_evaluator.get("strict_reproducibility", False))
    if planned_strict != bool(strict_reproducibility):
        raise SourceMainRunnerError("source_main_reproducibility_mode_mismatch")
    bindings = (
        load_evaluator_bindings(
            evaluator_bindings_path,
            strict_reproducibility=strict_reproducibility,
        )
        if evaluator_bindings_path is not None
        else None
    )
    # A binding file is an explicit provenance request even in non-strict mode,
    # so a missing selected pack is still diagnosed.  With no file, local host
    # evaluation is the documented default.  Strict mode additionally requires
    # the file before a source child can start.
    if bindings is not None or strict_reproducibility:
        bindings = _require_selected_bindings(selected, bindings)

    execution = _require_mapping(payload.get("execution"), "source_main_execution_missing")
    execution_batches = execution.get("batches")
    if not isinstance(execution_batches, dict):
        raise SourceMainRunnerError("source_main_execution_batches_invalid")
    execution["evaluator"] = {
        "mode": "container" if bindings is not None else "local",
        "strict_reproducibility": bool(strict_reproducibility),
        "bindings": {
            pack: binding.document()
            for pack, binding in (bindings or {}).items()
            if pack in {str(batch["pack"]) for batch in selected}
        },
    }
    runtime = _require_mapping(plan["model"].get("runtime"), "source_main_plan_model_runtime_missing")
    if set(runtime) != _SCOPED_RUNTIME_KEYS or not all(
        isinstance(value, str)
        and (bool(value) or key == "ORG_LLM_MODEL")
        for key, value in runtime.items()
    ):
        raise SourceMainRunnerError("source_main_plan_model_runtime_invalid")

    if dry_run:
        # A source dry-run contains real source case plans/manifests, but it is
        # intentionally separate from ``batches/`` so a later formal run can
        # supply the then-published evaluator binding without a resume-identity
        # collision with an unbound planning artefact.
        dry_root = output_root / "source-dry-run"
        execution["source_dry_run_root"] = str(dry_root)
    else:
        dry_root = output_root / "batches"
    completed = 0
    failed = 0
    stop_after_failure = False
    for batch in selected:
        batch_id = str(batch["batch_id"])
        if stop_after_failure:
            execution_batches[batch_id] = {
                "status": "not_run",
                "reason": "prior_source_batch_failed",
            }
            continue
        batch_root = dry_root / batch_id if dry_run else Path(str(batch["output_root"])).resolve()
        binding = bindings.get(str(batch["pack"])) if bindings is not None else None
        argv = _source_batch_argv(
            plan=plan,
            batch=batch,
            output_root=batch_root,
            max_parallel=max_parallel,
            dry_run=dry_run,
            resume=resume and not dry_run,
            binding=binding,
            strict_reproducibility=strict_reproducibility,
        )
        result_code = _invoke_source_batch(argv, runtime)
        status = _batch_status(batch_root, result_code)
        if dry_run:
            status["status"] = "dry_run" if result_code == 0 else "failed"
        observed = status.get("evaluator")
        if isinstance(observed, Mapping):
            observed_batches = execution["evaluator"].setdefault(
                "observed_batches", {}
            )
            if isinstance(observed_batches, dict):
                observed_batches[batch_id] = dict(observed)
        execution_batches[batch_id] = status
        if result_code == 0:
            completed += 1
        else:
            failed += 1
            stop_after_failure = True
        payload["resource_policy"] = _resource_policy(max_parallel)
        execution["status"] = (
            "planned" if dry_run and failed == 0 else ("failed" if failed else "running")
        )
        write_manifest(payload, destination_manifest)
    if failed:
        execution["status"] = "failed"
    elif dry_run:
        execution["status"] = "planned"
    elif len(selected) == len(plan["batches"]):
        execution["status"] = "completed"
    else:
        execution["status"] = "partial"
    write_manifest(payload, destination_manifest)
    resource_policy = _require_mapping(
        payload.get("resource_policy"), "source_main_resource_policy_missing"
    )
    return SourceMainRunResult(
        manifest_path=destination_manifest,
        output_root=output_root,
        model=str(plan["model"]["canonical_model"]),
        status=str(execution["status"]),
        total_cells=120,
        selected_batches=len(selected),
        completed_batches=completed,
        failed_batches=failed,
        dry_run=dry_run,
        visible_memory_gib=float(resource_policy["visible_memory_gib"]),
        recommended_max_parallel=int(resource_policy["recommended_max_parallel"]),
        requested_max_parallel=int(resource_policy["requested_max_parallel"]),
        parallelism_warning=bool(resource_policy["warning"]),
    )


__all__ = [
    "EvaluatorBinding",
    "SOURCE_MAIN_MANIFEST_SCHEMA_VERSION",
    "SOURCE_MAIN_PLAN_SCHEMA_VERSION",
    "SourceMainRunResult",
    "SourceMainRunnerError",
    "build_source_main_manifest",
    "load_evaluator_bindings",
    "run_source_main",
]
