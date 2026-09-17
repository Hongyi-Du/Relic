"""Canonical, paper-auditable records for controlled OrgEnv runs.

Version 2 keeps every version-1 analysis field at the top level.  New
provenance is populated only from observed world state, local frozen assets, or
explicit caller evidence; unavailable evidence remains null and is reported by
``completeness``/``caveats``.
"""

from __future__ import annotations

import datetime as dt
import math
import os
import platform
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from environments.org_env.config.metrics import ALL_METRICS, OrgMetrics
from environments.org_env.experiments.provenance import (
    MappingSource,
    checkpoint_metadata,
    directory_content_hash,
    event_graph_fingerprint,
    load_mapping,
    mapping_source_hash,
    portable_source_identifier,
    repository_digest,
    require_sha256,
    resolve_fingerprint,
    sanitize_provenance,
    stable_fingerprint,
)
from environments.org_env.experiments.resources import experiment_resource_snapshot

LEGACY_RUN_RECORD_SCHEMA_VERSION = "orgenv_experiment_run_v1"
RUN_RECORD_SCHEMA_VERSION = "orgenv_experiment_run_v2"
KNOWN_RUN_RECORD_SCHEMA_VERSIONS = frozenset(
    {LEGACY_RUN_RECORD_SCHEMA_VERSION, RUN_RECORD_SCHEMA_VERSION}
)

_RUN_STATUSES = frozenset(
    {
        "pending",
        "running",
        "completed",
        "failed",
        "cancelled",
        "timeout",
        "infra_error",
        "blocked",
    }
)
_FINAL_STATUSES = frozenset(
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
_CONTAMINATION_STATUSES = frozenset(
    {
        "pending",
        "probes_defined_not_executed",
        "cleared",
        "flagged",
        "failed",
    }
)

_DEFAULT_PRODUCT_METRICS = (
    "oss_hidden_pass_rate",
    "oss_issue_fix_rate",
    "oss_regression_pass_rate",
    "oss_public_test_pass_rate",
    "oss_cli_ok",
    "quality",
)

_CAPABILITY_METRICS = {
    "artifact_creation_rate",
    "artifact_adoption_rate",
    "protocol_proposal_rate",
    "protocol_use_count",
    "violation_count",
    "enforcement_count",
    "ownership_clarity",
    "weak_protocol_emergence_rate",
    "strong_protocol_emergence_rate",
    "time_to_emergence",
    "persistence",
    "repeated_protocol_use_rate",
    "cross_context_protocol_reuse_rate",
    "protocol_persistence_rate",
    "valid_third_party_enforcement_rate",
    "protocol_amendment_rate",
    "protocol_repair_rate",
    "measurable_impact",
    # The preregistration's MIDDLE tier — used beyond the episode it was adopted
    # in — and the share of carriers for which that question is answerable at
    # all, so a 0.0 reads as "did not transfer" rather than "not measured".
    "episode_transfer_rate",
    "episode_transfer_measurable_rate",
    # Stage flag (0.0 until the final-evaluation oracle attaches); grouped with
    # the capability family so it travels next to the rates it disambiguates.
    "strong_emergence_oracle_attached",
    # The denominator every rate above divides by. Without it a run with no
    # protocols and a run whose protocols all failed both read 0.0, and three of
    # the four conditions disable institutionalization by definition.
    "protocol_carrier_count",
}
_EFFICIENCY_METRICS = {
    "expensive_experiment_count",
    "cheap_pilot_count",
    "duplicated_experiment_count",
    "unlogged_experiment_count",
    "budget_use_per_completed_task",
    "review_backlog",
    "task_cycle_time_avg",
    "pr_review_latency_avg",
    "pr_merge_latency_avg",
    "rework_rate",
    "token_burn_total",
    # In-world treasury currency. Real provider cost is llm_tokens_per_merged_pr.
    "budget_burn_per_merged_pr",
    "llm_tokens_per_merged_pr",
}
_EXTERNAL_METRICS = {
    "signal_exposure_count",
    "external_post_reach",
    "expert_advice_retrieved",
    "customer_complaint_pressure",
    "reputation_change",
}
_PROFILE_METRICS = {
    "profile_policy_mean_js_divergence",
    "profile_policy_argmax_shift_rate",
    "profile_policy_mean_chosen_probability_delta",
    "profile_policy_monotonicity_violation_rate",
}

_PAPER_TOP_LEVEL_FIELDS = (
    "run_id",
    "pack",
    "condition",
    "experiment_phase",
    "arm_id",
    "oss_control",
    "evaluation_perturbation",
    "provider",
    "model",
    "llm_runtime_identity",
    "llm_runtime_fingerprint",
    "seed",
    "metrics",
    "resource_budget_fingerprint",
    "ablation_fingerprint",
    "action_selection_mode",
    "replication_id",
    "randomization_block",
    "randomization_order",
    "run_manifest_hash",
    "dataset_manifest_hash",
    "starter_repo_digest",
    "reference_repo_digest",
    "candidate_repo_digest",
    "hidden_suite_hash",
    "evaluator_environment_hash",
    "tool_surface_fingerprint",
    "information_budget_fingerprint",
    "model_cutoff_policy",
    "contamination_status",
    "contamination_probes",
    "contamination_clearance",
    "started_at",
    "ended_at",
    "duration_seconds",
    "status",
    "checkpoint",
    "provenance",
    "event_graph_hash",
    "organizational_capability_evidence_hash",
    "organizational_capability_evidence",
    "profile_causality_evidence_hash",
    "profile_causality_evidence",
    "capability_evidence_refs",
    "capability_evidence_refs_truncated",
    "final_evaluation",
)
_FINAL_EVALUATION_FIELDS = (
    "dataset_id",
    "plan_hash",
    "candidate_repo_digest",
    "causal_fix_count",
    "causal_fix_rate",
    "unresolved_count",
    "regression_count",
    "candidate_pass_rate",
    "infrastructure_error_count",
    "formal_claim_ready",
    "result_hash",
    "evidence_kind",
    "artifact_hash",
    "manual_check_declarations",
    "manual_checks_enabled",
    "manual_checks",
    "formal_release_ready",
    "evidence_hashes",
)
_OUTCOME_FIELDS = (
    "test_id",
    "issue_ids",
    "baseline_status",
    "reference_status",
    "candidate_status",
    "causal_fix",
    "unresolved",
    "regression",
)
_PROVENANCE_HASH_FIELDS = (
    "run_manifest_hash",
    "dataset_manifest_hash",
    "starter_repo_digest",
    "reference_repo_digest",
    "candidate_repo_digest",
    "hidden_suite_hash",
    "evaluator_environment_hash",
    "tool_surface_fingerprint",
    "information_budget_fingerprint",
)
_PLAN_HASH_FIELDS = (
    "dataset_id",
    "product_name",
    "starter_ref",
    "reference_ref",
    "starter_repo_digest",
    "reference_repo_digest",
    "hidden_suite_hash",
    "evaluator_environment_hash",
    "required_release_coverage",
    "covered_release_versions",
    "oracles",
    "public_contracts",
    "operational_ready",
    "formal_environment_ready",
    "blocking_reasons",
)
_EVIDENCE_HASH_FIELDS = (
    "task_id",
    "command",
    "owner",
    "kind",
    "base_status",
    "candidate_status",
    "required",
    "baseline_repo_digest",
    "candidate_repo_digest",
    "execution_policy_hash",
    "exit_code",
    "stdout_hash",
    "stderr_hash",
)
_TOP_LEVEL_SHA256_FIELDS = (
    "resource_budget_fingerprint",
    "ablation_fingerprint",
    "run_manifest_hash",
    "dataset_manifest_hash",
    "starter_repo_digest",
    "reference_repo_digest",
    "candidate_repo_digest",
    "hidden_suite_hash",
    "evaluator_environment_hash",
    "tool_surface_fingerprint",
    "information_budget_fingerprint",
    "event_graph_hash",
    "organizational_capability_evidence_hash",
    "profile_causality_evidence_hash",
    "llm_runtime_fingerprint",
)


def _default_metric_family(metric: str) -> str:
    if metric in _CAPABILITY_METRICS:
        return "organizational_capability"
    if metric in _EFFICIENCY_METRICS:
        return "resource_efficiency"
    if metric in _EXTERNAL_METRICS:
        return "external_adaptation"
    if metric in _PROFILE_METRICS:
        return "profile_causality"
    return "product_effectiveness"


def _infer_pack(world: Any) -> str:
    config = getattr(world, "company_config", {}) or {}
    substrate = config.get("product_substrate") or {}
    # The declared repository id wins over the locator. `dataset_id` may be an
    # absolute path — a dataset resolves from either form — but `pack` is
    # cross-checked against the manifest's project_id further down, so a path
    # here fails the record for every run the DAG launches.
    repository_id = substrate.get("repository_id")
    if repository_id:
        return str(repository_id)
    dataset_id = substrate.get("dataset_id")
    if dataset_id:
        return str(dataset_id)
    product = getattr(world, "product", None)
    metadata = getattr(product, "substrate_meta", {}) or {}
    return str(
        metadata.get("dataset_id")
        or metadata.get("project_id")
        or getattr(product, "name", "")
        or "unknown_pack"
    )


def _scenario_params(world: Any) -> Mapping[str, Any]:
    scenario = getattr(world, "scenario", None)
    params = getattr(scenario, "params", None)
    return params if isinstance(params, Mapping) else {}


def _nonempty(value: Any) -> bool:
    return value is not None and value != ""


def _first(*values: Any) -> Any:
    return next((value for value in values if _nonempty(value)), None)


_LLM_USAGE_COUNTERS = (
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "cached_prompt_tokens",
)


def _llm_usage_payload(client: Any) -> dict[str, int] | None:
    """Actual API token usage accumulated by the cognitive-layer LLM client.

    None for rule-based runs (no client). Zeros are honest for providers that
    expose no usage (mock / generic HTTP); call/failure/retry counts are real
    either way.
    """
    if client is None:
        return None
    totals = getattr(client, "usage_totals", None)
    if not isinstance(totals, Mapping):
        return None

    def _count(value: Any) -> int:
        try:
            return max(0, int(value or 0))
        except (TypeError, ValueError):
            return 0

    payload = {name: _count(totals.get(name)) for name in _LLM_USAGE_COUNTERS}
    payload["calls"] = _count(getattr(client, "calls", 0))
    payload["failures"] = _count(getattr(client, "failures", 0))
    payload["retries"] = _count(getattr(client, "retries", 0))
    payload["provider_attempts"] = _count(
        getattr(client, "provider_attempts", 0)
    )
    payload["resource_denials"] = _count(
        getattr(client, "resource_denials", 0)
    )
    payload["prompt_visibility_denials"] = _count(
        getattr(client, "prompt_visibility_denials", 0)
    )
    return payload


def _prompt_visibility_audit_payload(
    client: Any,
) -> dict[str, Any] | None:
    if client is None:
        return None
    snapshot = getattr(client, "prompt_visibility_audit", None)
    if not callable(snapshot):
        return None
    payload = snapshot()
    return dict(payload) if isinstance(payload, Mapping) else None


def _llm_runtime_payload(
    client: Any,
    *,
    provider: str,
    model: str,
) -> tuple[dict[str, Any], str]:
    """Return a non-secret identity for the effective cognitive runtime."""

    endpoint = (
        getattr(client, "_base_url", None)
        or getattr(client, "endpoint", None)
        or ""
    )
    header_names = tuple(
        sorted(
            str(name).strip().lower()
            for name in (
                getattr(client, "default_headers", {}) or {}
            )
            if str(name).strip()
        )
    )
    endpoint_kind = (
        "custom_compatible"
        if endpoint
        else (
            "official_default"
            if provider == "openai"
            else "not_applicable"
        )
    )
    payload = {
        "provider": str(provider),
        "model": str(model),
        "wire_api": str(
            getattr(client, "wire_api", None) or "not_applicable"
        ),
        "json_transport": str(
            getattr(client, "json_transport", None) or "not_applicable"
        ),
        "endpoint_kind": endpoint_kind,
        "endpoint_hash": stable_fingerprint(
            str(endpoint) or endpoint_kind
        ),
        "routing_context_fingerprint": str(
            getattr(client, "routing_context_fingerprint", "")
            or stable_fingerprint(
                {
                    "endpoint": str(endpoint) or endpoint_kind,
                    "header_names": header_names,
                }
            )
        ),
        "default_header_names": list(header_names),
        "observed_response_models": dict(
            sorted(
                (
                    str(name),
                    int(count),
                )
                for name, count in (
                    getattr(client, "response_model_counts", {}) or {}
                ).items()
            )
        ),
        "response_id_digest": (
            str(getattr(client, "response_id_digest", "") or "") or None
        ),
        "response_storage_enabled": bool(
            getattr(client, "store_responses", False)
        ),
        "request_timeout_seconds": (
            float(getattr(client, "request_timeout_seconds"))
            if getattr(client, "request_timeout_seconds", None) is not None
            else None
        ),
        "max_retries": (
            int(getattr(client, "max_retries"))
            if getattr(client, "max_retries", None) is not None
            else None
        ),
        "retry_backoff_seconds": (
            float(getattr(client, "retry_backoff_seconds"))
            if getattr(client, "retry_backoff_seconds", None) is not None
            else None
        ),
    }
    return payload, stable_fingerprint(payload)


def _manifest_project_id(manifest: Mapping[str, Any] | None) -> str | None:
    if not manifest:
        return None
    value = _first(manifest.get("project_id"), manifest.get("dataset_id"))
    return str(value) if value is not None else None


def _source_description(source: Any) -> str:
    return portable_source_identifier(source)


def _resolve_mapping_hash(
    source: MappingSource | None,
    supplied: str | None,
    *,
    label: str,
) -> tuple[dict[str, Any] | None, str | None]:
    if source is None:
        return None, resolve_fingerprint(None, supplied, label=label)
    payload, computed = mapping_source_hash(source, label=label)
    resolved = resolve_fingerprint(payload, supplied, label=label)
    if resolved != computed:  # defensive: both use the same canonical algorithm
        raise ValueError(f"{label} internal hash disagreement")
    return payload, resolved


def _resolve_repo_digest(
    path: str | os.PathLike[str] | None,
    supplied: str | None,
    *,
    label: str,
) -> str | None:
    declared = require_sha256(supplied, label=label)
    if path is None:
        return declared
    computed = repository_digest(path)
    if declared is not None and declared != computed:
        raise ValueError(f"{label} mismatch: supplied={declared} computed={computed}")
    return computed


def _resolve_hidden_suite_hash(
    source: Any,
    supplied: str | None,
) -> str | None:
    declared = require_sha256(supplied, label="hidden_suite_hash")
    if source is None:
        return declared
    if isinstance(source, (str, os.PathLike)) and Path(source).expanduser().is_dir():
        hidden_root = Path(source).expanduser()
        if (hidden_root / "programbench.json").exists():
            # ProgramBench's identity includes opaque binary branch archives,
            # file kinds, and sizes.  The generic text-directory hash silently
            # skips those bytes and therefore cannot be compared with the
            # frozen evaluator's plan/artifact identity.  Keep this import
            # local so ordinary record construction does not load the official
            # evaluator adapter.
            from society_core.programbench_evaluation import (
                programbench_hidden_suite_hash,
            )

            computed = programbench_hidden_suite_hash(
                SimpleNamespace(hidden_tests_dir=str(hidden_root))
            )
        else:
            computed = directory_content_hash(hidden_root)
    else:
        computed = stable_fingerprint(source)
    if declared is not None and declared != computed:
        raise ValueError(
            f"hidden_suite_hash mismatch: supplied={declared} computed={computed}"
        )
    return computed


def _derived_starter_path(
    assets: Mapping[str, Any],
    manifest: Mapping[str, Any] | None,
) -> str | None:
    direct = assets.get("starter_repo_dir")
    if direct:
        return str(direct)
    reference = assets.get("reference_repo_dir")
    visible = (manifest or {}).get("agent_visible") or {}
    relative = visible.get("starter_repo") if isinstance(visible, Mapping) else None
    if reference and relative:
        return str(Path(str(reference)).parent / str(relative))
    return None


def _derived_contamination_probe_path(
    assets: Mapping[str, Any],
    manifest: Mapping[str, Any] | None,
) -> str | None:
    direct = assets.get("contamination_probes_path")
    if direct:
        return str(direct)
    reference = assets.get("reference_repo_dir")
    private = (manifest or {}).get("private_evaluator_only") or {}
    relative = (
        private.get("contamination_probes") if isinstance(private, Mapping) else None
    )
    if not reference or not relative:
        return None
    path = Path(str(reference)).parent / str(relative) / "probes.json"
    return str(path) if path.is_file() else None


def _normalize_contamination_probes(
    source: MappingSource | None,
) -> dict[str, Any] | None:
    if source is None:
        return None
    payload = load_mapping(source, label="contamination probes")
    if {"probe_ids", "probe_count", "probe_hash"} <= set(payload):
        probe_ids = payload["probe_ids"]
        probe_count = payload["probe_count"]
        probe_hash = require_sha256(
            payload["probe_hash"],
            label="contamination_probes.probe_hash",
            allow_none=False,
        )
        if (
            not isinstance(probe_ids, Sequence)
            or isinstance(probe_ids, (str, bytes))
            or not all(isinstance(item, str) and item for item in probe_ids)
        ):
            raise ValueError("contamination_probes.probe_ids must be strings")
        if (
            isinstance(probe_count, bool)
            or not isinstance(probe_count, int)
            or probe_count < 0
            or probe_count != len(probe_ids)
        ):
            raise ValueError("contamination_probes.probe_count must match probe_ids")
        result = {
            "probe_ids": list(probe_ids),
            "probe_count": probe_count,
            "probe_hash": probe_hash,
        }
        if "results" in payload:
            if not isinstance(payload["results"], Mapping):
                raise ValueError("contamination_probes.results must be an object")
            result["results"] = sanitize_provenance(payload["results"])
        return result
    probe_ids = sorted(str(key) for key in payload if str(key).endswith("_probe"))
    return {
        "probe_ids": probe_ids,
        "probe_count": len(probe_ids),
        "probe_hash": stable_fingerprint(payload),
    }


def _strict_optional_bool(value: Any, *, field: str) -> bool | None:
    if value in (None, ""):
        return None
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be a boolean or null")
    return value


def _normalize_timestamp(value: Any) -> str | None:
    if value in (None, ""):
        return None
    if isinstance(value, dt.datetime):
        parsed = value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        parsed = dt.datetime.fromtimestamp(float(value), tz=dt.timezone.utc)
    elif isinstance(value, str):
        parsed = _parse_timestamp(value)
        if parsed is None:
            raise ValueError("timestamp must be valid ISO-8601")
    elif not isinstance(value, dt.datetime):
        raise ValueError("timestamp must be datetime, epoch seconds, or ISO-8601")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timestamp must include a UTC offset")
    return parsed.isoformat()


def _parse_timestamp(value: str | None) -> dt.datetime | None:
    if value is None:
        return None
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        return dt.datetime.fromisoformat(normalized)
    except ValueError:
        return None


def _resolve_timing(
    started_at: Any,
    ended_at: Any,
    duration_seconds: float | None,
) -> tuple[str | None, str | None, float | None]:
    start = _normalize_timestamp(started_at)
    end = _normalize_timestamp(ended_at)
    if duration_seconds is None:
        duration = None
    elif (
        isinstance(duration_seconds, bool)
        or not isinstance(duration_seconds, (int, float))
        or not math.isfinite(float(duration_seconds))
        or float(duration_seconds) < 0
    ):
        raise ValueError("duration_seconds must be a finite non-negative number")
    else:
        duration = float(duration_seconds)
    parsed_start = _parse_timestamp(start)
    parsed_end = _parse_timestamp(end)
    if parsed_start is not None and parsed_end is not None:
        if parsed_start.tzinfo is None or parsed_end.tzinfo is None:
            raise ValueError("timestamps must include UTC offsets")
        observed = (parsed_end - parsed_start).total_seconds()
        if observed < 0:
            raise ValueError("ended_at must not precede started_at")
        if duration is not None and abs(duration - observed) > 0.001:
            raise ValueError("duration_seconds conflicts with started_at/ended_at")
        duration = observed
    elif parsed_start is not None and duration is not None and end is None:
        end = (parsed_start + dt.timedelta(seconds=duration)).isoformat()
    elif parsed_end is not None and duration is not None and start is None:
        start = (parsed_end - dt.timedelta(seconds=duration)).isoformat()
    return start, end, duration


def _capability_refs(
    world: Any,
    supplied: Sequence[Any] | str | None,
) -> list[Any]:
    if supplied is not None:
        raw = [supplied] if isinstance(supplied, str) else list(supplied)
    else:
        raw = []
        for skill in getattr(world, "company_skills", ()) or ():
            if not isinstance(skill, Mapping):
                continue
            raw.extend(skill.get("evidence_event_ids") or ())
        registry = getattr(world, "protocol_registry", None)
        raw.extend(
            str(getattr(event, "event_id", ""))
            for event in (getattr(registry, "events", ()) or ())
            if str(getattr(event, "event_id", ""))
        )
        # Document-borne capabilities carry their own synthesized event ids.
        # They are part of the capability packet's event index, so a refs list
        # built from protocols alone would fail the packet-subset invariant and,
        # worse, leave seven of the ten preregistered capabilities without a
        # traceable event reference in the run record.
        try:
            from environments.org_env.experiments.capability_evidence import (
                _document_carrier_rows,
            )

            for row in _document_carrier_rows(world):
                raw.extend(str(item.get("event_id") or "") for item in row["events"])
        except Exception:
            pass
    result: list[Any] = []
    seen: set[str] = set()
    for item in raw:
        marker = stable_fingerprint(item)
        if marker not in seen:
            seen.add(marker)
            result.append(item)
    return result


def _capability_refs_are_truncated(world: Any) -> bool:
    for skill in getattr(world, "company_skills", ()) or ():
        if not isinstance(skill, Mapping):
            continue
        recorded = len(skill.get("evidence_event_ids") or ())
        expected = int(skill.get("use_count") or 0) + int(
            skill.get("enforcement_count") or 0
        )
        if expected > recorded:
            return True
    return False


def _evaluation_sources(
    payload: Mapping[str, Any],
) -> tuple[tuple[str, Mapping[str, Any]], ...]:
    sources: list[tuple[str, Mapping[str, Any]]] = []
    seen: set[int] = set()

    def add(label: str, value: Any) -> None:
        if isinstance(value, Mapping) and id(value) not in seen:
            seen.add(id(value))
            sources.append((label, value))

    add("root", payload)
    for key in (
        "time_machine_evaluation",
        "final_evaluation",
        "evaluation",
        "evaluation_plan",
        "qualified_plan",
        "plan",
        "evaluation_result",
        "result",
    ):
        add(key, payload.get(key))
    for parent_key in (
        "run_config",
        "report",
        "workspace_development_run",
        "development_run",
    ):
        parent = payload.get(parent_key)
        add(parent_key, parent)
        if not isinstance(parent, Mapping):
            continue
        for key in (
            "time_machine_evaluation",
            "final_evaluation",
            "evaluation_plan",
            "qualified_plan",
            "plan",
            "evaluation_result",
            "result",
        ):
            add(f"{parent_key}.{key}", parent.get(key))
    return tuple(sources)


def _consensus(
    sources: Sequence[tuple[str, Mapping[str, Any]]],
    field: str,
    *aliases: str,
) -> Any:
    values: list[tuple[str, str, Any]] = []
    for label, source in sources:
        for key in (field, *aliases):
            if key in source and _nonempty(source.get(key)):
                values.append((label, key, source[key]))
    if not values:
        return None
    expected = stable_fingerprint(values[0][2])
    for label, key, value in values[1:]:
        if stable_fingerprint(value) != expected:
            locations = ", ".join(
                f"{item_label}.{item_key}={item_value!r}"
                for item_label, item_key, item_value in values
            )
            raise ValueError(f"final evaluator {field} mismatch across {locations}")
    return values[0][2]


def _as_count(value: Any, *, field: str) -> int | None:
    if value in (None, ""):
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"final evaluator {field} must be a non-negative integer")
    return value


def _as_rate(value: Any, *, field: str) -> float | None:
    if value in (None, ""):
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise ValueError(f"final evaluator {field} must be between 0 and 1")
    return float(value)


def _as_optional_bool(value: Any, *, field: str) -> bool | None:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        return value
    raise ValueError(f"final evaluator {field} must be a boolean")


def _as_status(
    value: Any,
    *,
    field: str,
    allowed: frozenset[str],
) -> str | None:
    if value in (None, ""):
        return None
    if not isinstance(value, str) or value not in allowed:
        raise ValueError(f"{field} must be one of {sorted(allowed)} or null")
    return value


def _outcomes_from_sources(
    sources: Sequence[tuple[str, Mapping[str, Any]]],
) -> tuple[Mapping[str, Any], ...] | None:
    for _, source in sources:
        outcomes = source.get("outcomes")
        if isinstance(outcomes, Sequence) and not isinstance(outcomes, (str, bytes)):
            if not all(isinstance(item, Mapping) for item in outcomes):
                raise ValueError("final evaluator outcomes must contain objects")
            return tuple(outcomes)
    return None


def _validate_derived_counts(
    outcomes: tuple[Mapping[str, Any], ...] | None,
    *,
    causal_fix_count: int | None,
    unresolved_count: int | None,
    regression_count: int | None,
) -> tuple[int | None, int | None, int | None]:
    if outcomes is None:
        return causal_fix_count, unresolved_count, regression_count
    derived = (
        sum(bool(item.get("causal_fix")) for item in outcomes),
        sum(bool(item.get("unresolved")) for item in outcomes),
        sum(bool(item.get("regression")) for item in outcomes),
    )
    declared = (causal_fix_count, unresolved_count, regression_count)
    names = ("causal_fix_count", "unresolved_count", "regression_count")
    for name, claimed, observed in zip(names, declared, derived):
        if claimed is not None and claimed != observed:
            raise ValueError(
                f"final evaluator {name} mismatch: "
                f"supplied={claimed} outcomes={observed}"
            )
    return tuple(
        observed if claimed is None else claimed
        for claimed, observed in zip(declared, derived)
    )


def _evidence_hashes(source: Mapping[str, Any]) -> list[str] | None:
    declared: list[str] | None = None
    if "evidence_hashes" in source:
        raw = source.get("evidence_hashes") or ()
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            declared = [
                require_sha256(
                    value,
                    label="final_evaluation.evidence_hashes[]",
                    allow_none=False,
                )
                for value in raw
            ]
        else:
            raise ValueError("final evaluator evidence_hashes must be a list")
    records = source.get("evidence_records")
    if isinstance(records, Sequence) and not isinstance(records, (str, bytes)):
        values: list[str] = []
        for item in records:
            if not isinstance(item, Mapping):
                raise ValueError("final evaluator evidence record must be an object")
            for field in (
                "baseline_repo_digest",
                "candidate_repo_digest",
                "execution_policy_hash",
                "stdout_hash",
                "stderr_hash",
            ):
                require_sha256(
                    item.get(field),
                    label=f"final evaluator evidence {field}",
                )
            if not isinstance(item.get("required"), bool):
                raise ValueError("final evaluator evidence required must be boolean")
            exit_code = item.get("exit_code")
            if exit_code is not None and (
                isinstance(exit_code, bool) or not isinstance(exit_code, int)
            ):
                raise ValueError(
                    "final evaluator evidence exit_code must be integer or null"
                )
            observed = require_sha256(
                item.get("evidence_hash"),
                label="final evaluator evidence_hash",
                allow_none=False,
            )
            expected = stable_fingerprint(
                {field: item.get(field) for field in _EVIDENCE_HASH_FIELDS}
            )
            if observed != expected:
                raise ValueError("final evaluator evidence_hash mismatch")
            evidence_id = item.get("evidence_id")
            if evidence_id != f"verification_evidence_{expected[:24]}":
                raise ValueError("final evaluator evidence_id mismatch")
            values.append(observed)
        if declared is not None and declared != values:
            raise ValueError("final evaluator evidence_hashes mismatch")
        return values
    return declared


def _verify_plan_hash(
    sources: Sequence[tuple[str, Mapping[str, Any]]],
) -> None:
    for label, source in sources:
        if not all(key in source for key in (*_PLAN_HASH_FIELDS, "plan_hash")):
            if label.endswith("qualified_plan"):
                raise ValueError(
                    "final evaluator qualified_plan must contain all "
                    f"{len(_PLAN_HASH_FIELDS)} hash fields"
                )
            continue
        observed = require_sha256(
            source["plan_hash"],
            label=f"{label}.plan_hash",
            allow_none=False,
        )
        payload = {key: source[key] for key in _PLAN_HASH_FIELDS}
        computed = stable_fingerprint(payload)
        if observed != computed:
            raise ValueError(
                f"final evaluator plan_hash mismatch in {label}: "
                f"supplied={source['plan_hash']} computed={computed}"
            )
        if "formal_ready" in source:
            formal_ready = _as_optional_bool(
                source["formal_ready"],
                field=f"{label}.formal_ready",
            )
            if formal_ready is not (not bool(source["blocking_reasons"])):
                raise ValueError(f"final evaluator formal_ready mismatch in {label}")


def _verify_result_hash(
    sources: Sequence[tuple[str, Mapping[str, Any]]],
) -> None:
    for label, source in sources:
        outcomes = source.get("outcomes")
        evidence_hashes = _evidence_hashes(source)
        required = (
            "dataset_id",
            "plan_hash",
            "candidate_repo_digest",
            "status",
            "formal_claim_ready",
            "result_hash",
        )
        if (
            not all(key in source for key in required)
            or not isinstance(outcomes, Sequence)
            or isinstance(outcomes, (str, bytes))
            or evidence_hashes is None
        ):
            continue
        normalized_outcomes = []
        complete = True
        for outcome in outcomes:
            if not isinstance(outcome, Mapping) or not all(
                key in outcome for key in _OUTCOME_FIELDS
            ):
                complete = False
                break
            normalized_outcomes.append({key: outcome[key] for key in _OUTCOME_FIELDS})
        if not complete:
            continue
        observed = require_sha256(
            source["result_hash"],
            label=f"{label}.result_hash",
            allow_none=False,
        )
        status = _as_status(
            source["status"],
            field=f"{label}.status",
            allowed=_FINAL_STATUSES,
        )
        formal_claim_ready = _as_optional_bool(
            source["formal_claim_ready"],
            field=f"{label}.formal_claim_ready",
        )
        result_payload = {
            "dataset_id": source["dataset_id"],
            "plan_hash": source["plan_hash"],
            "candidate_repo_digest": source["candidate_repo_digest"],
            "status": status,
            "outcomes": normalized_outcomes,
            "evidence_hashes": evidence_hashes,
            "formal_claim_ready": formal_claim_ready,
        }
        computed = stable_fingerprint(result_payload)
        if observed != computed:
            raise ValueError(
                f"final evaluator result_hash mismatch in {label}: "
                f"supplied={source['result_hash']} computed={computed}"
            )


def _verify_artifact_hash(payload: Mapping[str, Any]) -> str | None:
    if "artifact_hash" not in payload:
        return None
    observed = require_sha256(
        payload.get("artifact_hash"),
        label="final evaluator artifact_hash",
        allow_none=False,
    )
    expected = stable_fingerprint(
        {key: value for key, value in payload.items() if key != "artifact_hash"}
    )
    if observed != expected:
        raise ValueError(
            "final evaluator artifact_hash mismatch: "
            f"supplied={observed} computed={expected}"
        )
    result = payload.get("result")
    evaluation = payload.get("evaluation")
    if evaluation is not None and result != evaluation:
        raise ValueError("final evaluator evaluation alias mismatch")
    return observed


def _evidence_hashes_from_sources(
    sources: Sequence[tuple[str, Mapping[str, Any]]],
) -> list[str] | None:
    values: list[tuple[str, list[str]]] = []
    for label, source in sources:
        hashes = _evidence_hashes(source)
        if hashes is not None:
            values.append((label, hashes))
    if not values:
        return None
    expected = values[0][1]
    for label, hashes in values[1:]:
        if hashes != expected:
            raise ValueError(
                "final evaluator evidence hashes mismatch across "
                f"{values[0][0]} and {label}"
            )
    return expected


def _manual_evidence_fields(payload: Mapping[str, Any]) -> dict[str, Any]:
    declarations = payload.get("manual_check_declarations")
    enabled = payload.get("manual_checks_enabled")
    checks = payload.get("manual_checks")
    release_ready = payload.get("formal_release_ready")
    if all(value is None for value in (declarations, enabled, checks, release_ready)):
        return {
            "manual_check_declarations": None,
            "manual_checks_enabled": None,
            "manual_checks": None,
            "formal_release_ready": None,
        }
    if (
        not isinstance(declarations, Sequence)
        or isinstance(declarations, (str, bytes))
        or not all(isinstance(item, Mapping) for item in declarations)
    ):
        raise ValueError("final evaluator manual_check_declarations must be a list")
    if not isinstance(enabled, bool):
        raise ValueError("final evaluator manual_checks_enabled must be a boolean")
    if (
        not isinstance(checks, Sequence)
        or isinstance(checks, (str, bytes))
        or not all(isinstance(item, Mapping) for item in checks)
    ):
        raise ValueError("final evaluator manual_checks must be a list")
    if not isinstance(release_ready, bool):
        raise ValueError("final evaluator formal_release_ready must be a boolean")
    for check in checks:
        if not isinstance(check.get("check_id"), str) or not check["check_id"]:
            raise ValueError("final evaluator manual check has no check_id")
        if check.get("status") not in {"passed", "failed", "infra_error"}:
            raise ValueError("final evaluator manual check has invalid status")
    return {
        "manual_check_declarations": [
            sanitize_provenance(item) for item in declarations
        ],
        "manual_checks_enabled": enabled,
        "manual_checks": [sanitize_provenance(item) for item in checks],
        "formal_release_ready": release_ready,
    }


def _manifest_evidence_kind(
    manifest: Mapping[str, Any] | None,
) -> str | None:
    if not manifest:
        return None
    strategy = manifest.get("test_strategy") or {}
    if isinstance(strategy, Mapping) and strategy.get("kind"):
        return str(strategy["kind"])
    if manifest.get("hidden_tests"):
        return "behavior"
    return None


def _evidence_kind_from_sources(
    sources: Sequence[tuple[str, Mapping[str, Any]]],
    manifest: Mapping[str, Any] | None,
) -> str | None:
    explicit = _consensus(sources, "evidence_kind")
    if explicit is not None:
        return str(explicit)
    manifest_kind = _manifest_evidence_kind(manifest)
    if manifest_kind is not None:
        return manifest_kind
    kinds: set[str] = set()
    for _, source in sources:
        records = source.get("evidence_records")
        if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
            continue
        kinds.update(
            str(item["kind"])
            for item in records
            if isinstance(item, Mapping) and item.get("kind")
        )
    return next(iter(kinds)) if len(kinds) == 1 else None


def normalize_final_evaluation(
    final_evaluator: MappingSource,
    *,
    expected_dataset_id: str | None = None,
    expected_plan_hash: str | None = None,
    expected_candidate_repo_digest: str | None = None,
    dataset_manifest: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load and validate evaluator-owned dataset/plan/candidate/result evidence."""

    payload = load_mapping(final_evaluator, label="final evaluator")
    sources = _evaluation_sources(payload)
    artifact_hash = _verify_artifact_hash(payload)
    if "qualified_plan" in payload and payload.get("schema_version") != (
        "orgenv_oss_final_evaluation_v1"
    ):
        raise ValueError("canonical final evaluator schema mismatch")
    _verify_plan_hash(sources)
    _verify_result_hash(sources)
    evidence_hashes = _evidence_hashes_from_sources(sources)
    manual_fields = _manual_evidence_fields(payload)

    dataset_id_value = _consensus(sources, "dataset_id", "pack")
    dataset_id = str(dataset_id_value) if dataset_id_value is not None else None
    manifest_id = _manifest_project_id(dataset_manifest)
    expected_ids = {
        value
        for value in (expected_dataset_id, manifest_id)
        if value not in (None, "", "unknown_pack")
    }
    if len(expected_ids) > 1:
        raise ValueError(f"record/dataset manifest mismatch: {sorted(expected_ids)}")
    expected_dataset = next(iter(expected_ids), None)
    if (
        expected_dataset is not None
        and dataset_id is not None
        and dataset_id != expected_dataset
    ):
        raise ValueError(
            "final evaluator dataset mismatch: "
            f"record={expected_dataset} evaluator={dataset_id}"
        )
    dataset_id = dataset_id or expected_dataset

    plan_hash = require_sha256(
        _consensus(sources, "plan_hash"),
        label="final_evaluation.plan_hash",
    )
    expected_plan_hash = require_sha256(
        expected_plan_hash,
        label="expected_plan_hash",
    )
    if expected_plan_hash and plan_hash and plan_hash != expected_plan_hash:
        raise ValueError(
            "final evaluator plan mismatch: "
            f"expected={expected_plan_hash} evaluator={plan_hash}"
        )

    candidate_digest = require_sha256(
        _consensus(sources, "candidate_repo_digest"),
        label="final_evaluation.candidate_repo_digest",
    )
    expected_candidate_repo_digest = require_sha256(
        expected_candidate_repo_digest,
        label="expected_candidate_repo_digest",
    )
    if (
        expected_candidate_repo_digest
        and candidate_digest
        and candidate_digest != expected_candidate_repo_digest
    ):
        raise ValueError(
            "final evaluator candidate mismatch: "
            f"expected={expected_candidate_repo_digest} "
            f"evaluator={candidate_digest}"
        )
    candidate_digest = candidate_digest or expected_candidate_repo_digest

    result_hash = require_sha256(
        _consensus(sources, "result_hash"),
        label="final_evaluation.result_hash",
    )
    causal_fix_count = _as_count(
        _consensus(sources, "causal_fix_count"),
        field="causal_fix_count",
    )
    unresolved_count = _as_count(
        _consensus(sources, "unresolved_count", "unresolved"),
        field="unresolved_count",
    )
    regression_count = _as_count(
        _consensus(sources, "regression_count", "regression"),
        field="regression_count",
    )
    infrastructure_error_count = _as_count(
        _consensus(sources, "infrastructure_error_count"),
        field="infrastructure_error_count",
    )
    candidate_pass_rate = _as_rate(
        _consensus(sources, "candidate_pass_rate"),
        field="candidate_pass_rate",
    )
    outcomes = _outcomes_from_sources(sources)
    (
        causal_fix_count,
        unresolved_count,
        regression_count,
    ) = _validate_derived_counts(
        outcomes,
        causal_fix_count=causal_fix_count,
        unresolved_count=unresolved_count,
        regression_count=regression_count,
    )
    declared_causal_fix_rate = _as_rate(
        _consensus(sources, "causal_fix_rate"),
        field="causal_fix_rate",
    )
    derived_causal_fix_rate = (
        causal_fix_count / len(outcomes)
        if outcomes is not None and outcomes
        else None
    )
    if (
        declared_causal_fix_rate is not None
        and derived_causal_fix_rate is not None
        and abs(declared_causal_fix_rate - derived_causal_fix_rate) > 1e-12
    ):
        raise ValueError("final evaluator causal_fix_rate mismatch")
    causal_fix_rate = (
        declared_causal_fix_rate
        if declared_causal_fix_rate is not None
        else derived_causal_fix_rate
    )
    formal_claim_ready = _as_optional_bool(
        _consensus(sources, "formal_claim_ready"),
        field="formal_claim_ready",
    )
    plan_formal_ready = _as_optional_bool(
        _consensus(sources, "formal_ready"),
        field="formal_ready",
    )
    status_sources = tuple(
        (label, source)
        for label, source in sources
        if label != "root"
        or any(
            key in source
            for key in (
                "result_hash",
                "formal_claim_ready",
                "causal_fix_count",
                "outcomes",
            )
        )
    )
    result_status = _as_status(
        _consensus(status_sources, "status"),
        field="final_evaluation.status",
        allowed=_FINAL_STATUSES,
    )
    if formal_claim_ready is True:
        if plan_formal_ready is False:
            raise ValueError(
                "formal_claim_ready conflicts with a non-formal evaluation plan"
            )
        if unresolved_count not in (None, 0):
            raise ValueError("formal_claim_ready conflicts with unresolved outcomes")
        if regression_count not in (None, 0):
            raise ValueError("formal_claim_ready conflicts with regressions")
        if result_status not in (None, "passed"):
            raise ValueError(
                "formal_claim_ready conflicts with final evaluator status "
                f"{result_status!r}"
            )
    formal_release_ready = manual_fields["formal_release_ready"]
    if formal_release_ready is True:
        if formal_claim_ready is not True:
            raise ValueError("formal_release_ready requires formal_claim_ready")
        if manual_fields["manual_checks_enabled"] and any(
            check.get("status") != "passed"
            for check in (manual_fields["manual_checks"] or ())
        ):
            raise ValueError("formal_release_ready conflicts with failed manual checks")

    # Route-agnostic functional score, carried through verbatim when the
    # evaluator produced one. Older artifacts predate it; absence is normal.
    functional_overlap = payload.get("functional_overlap")
    if not isinstance(functional_overlap, Mapping):
        functional_overlap = None
    functional_overlap_rate = None
    if functional_overlap is not None:
        raw_rate = functional_overlap.get("overlap_rate")
        if isinstance(raw_rate, (int, float)) and not isinstance(raw_rate, bool):
            functional_overlap_rate = float(raw_rate)

    return (
        {
            "dataset_id": dataset_id,
            "plan_hash": plan_hash or expected_plan_hash,
            "candidate_repo_digest": candidate_digest,
            "status": result_status,
            "functional_overlap": dict(functional_overlap) if functional_overlap else None,
            "functional_overlap_rate": functional_overlap_rate,
            "causal_fix_count": causal_fix_count,
            "causal_fix_rate": causal_fix_rate,
            "unresolved_count": unresolved_count,
            "regression_count": regression_count,
            "candidate_pass_rate": candidate_pass_rate,
            "infrastructure_error_count": infrastructure_error_count,
            "formal_claim_ready": formal_claim_ready,
            "result_hash": result_hash,
            "evidence_kind": _evidence_kind_from_sources(sources, dataset_manifest),
            "artifact_hash": artifact_hash,
            **manual_fields,
            "evidence_hashes": evidence_hashes,
        },
        payload,
    )


def _path_value(record: Mapping[str, Any], path: str) -> Any:
    value: Any = record
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return value


def _field_is_complete(record: Mapping[str, Any], path: str) -> bool:
    value = _path_value(record, path)
    if path in {
        "capability_evidence_refs",
        "final_evaluation.manual_check_declarations",
        "final_evaluation.manual_checks",
        "final_evaluation.evidence_hashes",
    }:
        return isinstance(value, Sequence) and not isinstance(value, (str, bytes))
    if path == "metrics":
        return isinstance(value, Mapping) and bool(value)
    if path == "provenance":
        return isinstance(value, Mapping) and bool(value)
    if isinstance(value, Mapping):
        return bool(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return bool(value)
    return value is not None and value != ""


def assess_run_record_completeness(
    record: Mapping[str, Any],
) -> dict[str, Any]:
    """Return an explicit paper-readiness inventory without filling evidence."""

    expected = list(_PAPER_TOP_LEVEL_FIELDS)
    expected.extend(f"final_evaluation.{field}" for field in _FINAL_EVALUATION_FIELDS)
    status = str(record.get("status") or "").lower()
    if status in {"failed", "error", "timeout", "infra_error", "blocked"}:
        expected.append("failure_reason")
    present = [path for path in expected if _field_is_complete(record, path)]
    missing = [path for path in expected if path not in present]
    fraction = len(present) / len(expected) if expected else 1.0
    return {
        "profile": "paper_run_record_v2",
        "is_complete": not missing,
        "present_field_count": len(present),
        "expected_field_count": len(expected),
        "fraction": round(fraction, 4),
        "present_fields": present,
        "missing_fields": missing,
    }


def validate_capability_transfer_receipt(record: Mapping[str, Any]) -> None:
    """A transfer arm has to say what it inherited.

    The four arms of the transfer design differ only in what they receive, and
    the receipt is the record's only evidence of which arm a run actually was.
    Without it an arm whose injection never ran is indistinguishable from one
    that ran, and a comparison across four arms quietly becomes a comparison of
    four identical runs.
    """
    if str(record.get("experiment_phase") or "") != "capability_transfer":
        return
    receipt = record.get("capability_transfer")
    if not isinstance(receipt, Mapping) or not receipt:
        raise ValueError(
            "a capability_transfer run must record its transfer receipt; "
            "without one there is no evidence the arm inherited anything"
        )
    for field in ("roster_origin", "capability_form", "source_repository_id"):
        if not str(receipt.get(field) or "").strip():
            raise ValueError(f"capability_transfer.{field} must be recorded")
    if str(receipt.get("source_repository_id")) == str(record.get("pack") or ""):
        raise ValueError(
            "capability_transfer.source_repository_id equals the target pack; "
            "a transfer that does not cross repositories measures nothing"
        )
    if str(receipt.get("schema_version") or "") == "org_capability_bundle_v2":
        require_sha256(
            receipt.get("bundle_sha256"),
            label="capability_transfer.bundle_sha256",
            allow_none=False,
        )
        bundle_snapshot = receipt.get("bundle_snapshot")
        if not isinstance(bundle_snapshot, Mapping):
            raise ValueError(
                "capability_transfer v2 must record its validated bundle_snapshot"
            )
        from environments.org_env.experiments.capability_transfer import (
            capability_bundle_sha256,
            validate_capability_bundle,
        )

        validate_capability_bundle(bundle_snapshot)
        if capability_bundle_sha256(bundle_snapshot) != str(
            receipt.get("bundle_sha256") or ""
        ):
            raise ValueError(
                "capability_transfer v2 bundle_snapshot does not match bundle_sha256"
            )
        capability_form = str(receipt.get("capability_form") or "")
        compiled = receipt.get("compiled_protocols")
        injected = receipt.get("protocols_injected")
        if not isinstance(compiled, list) or not isinstance(injected, list):
            raise ValueError(
                "capability_transfer v2 must record compiled_protocols and protocols_injected lists"
            )
        if capability_form == "executable":
            if not compiled or not injected:
                raise ValueError(
                    "capability_transfer v2 executable receipt has an empty compiled manifest"
                )
            mirror_ids: set[str] = set()
            spec_ids: set[str] = set()
            for index, row in enumerate(compiled):
                if not isinstance(row, Mapping):
                    raise ValueError(
                        f"capability_transfer.compiled_protocols[{index}] must be an object"
                    )
                mirror_id = str(row.get("protocol_id") or "")
                spec_id = str(row.get("protocol_spec_id") or "")
                binding_ids = row.get("binding_ids")
                require_sha256(
                    row.get("binding_hash"),
                    label=f"capability_transfer.compiled_protocols[{index}].binding_hash",
                    allow_none=False,
                )
                if (
                    not mirror_id
                    or not spec_id
                    or not isinstance(binding_ids, list)
                    or not binding_ids
                    or any(not str(value or "") for value in binding_ids)
                    or mirror_id in mirror_ids
                    or spec_id in spec_ids
                ):
                    raise ValueError(
                        f"capability_transfer.compiled_protocols[{index}] is malformed or duplicated"
                    )
                mirror_ids.add(mirror_id)
                spec_ids.add(spec_id)
            if mirror_ids != {str(value) for value in injected}:
                raise ValueError(
                    "capability_transfer v2 compiled/injected protocol manifests differ"
                )
        elif compiled:
            raise ValueError(
                "capability_transfer v2 non-executable receipt must not claim compiled protocols"
            )


def validate_experiment_run_record_schema(
    record: Mapping[str, Any],
) -> None:
    """Validate types and content-addressed fields without requiring completeness."""

    schema_version = record.get("schema_version")
    if schema_version not in KNOWN_RUN_RECORD_SCHEMA_VERSIONS:
        raise ValueError(
            f"unknown or missing run record schema_version: {schema_version!r}"
        )
    for field in ("run_id", "pack", "condition", "provider", "model"):
        value = record.get(field)
        if not isinstance(value, str) or not value:
            raise ValueError(f"{field} must be a non-empty string")
    runtime_identity = record.get("llm_runtime_identity")
    if schema_version == RUN_RECORD_SCHEMA_VERSION:
        runtime_hash = record.get("llm_runtime_fingerprint")
        if runtime_identity is not None or runtime_hash is not None:
            if not isinstance(runtime_identity, Mapping):
                raise ValueError("llm_runtime_identity must be an object")
            declared_runtime_hash = require_sha256(
                runtime_hash,
                label="llm_runtime_fingerprint",
                allow_none=False,
            )
            if stable_fingerprint(runtime_identity) != declared_runtime_hash:
                raise ValueError("llm_runtime_fingerprint mismatch")
            header_names = runtime_identity.get("default_header_names")
            if (
                not isinstance(header_names, Sequence)
                or isinstance(header_names, (str, bytes))
                or not all(
                    isinstance(name, str) and name.strip()
                    for name in header_names
                )
            ):
                raise ValueError(
                    "llm_runtime_identity.default_header_names must be a list"
                )
            require_sha256(
                runtime_identity.get("routing_context_fingerprint"),
                label="llm_runtime_identity.routing_context_fingerprint",
                allow_none=False,
            )
            response_models = runtime_identity.get(
                "observed_response_models"
            )
            if (
                not isinstance(response_models, Mapping)
                or any(
                    not isinstance(name, str)
                    or not name
                    or isinstance(count, bool)
                    or not isinstance(count, int)
                    or count < 0
                    for name, count in response_models.items()
                )
            ):
                raise ValueError(
                    "llm_runtime_identity.observed_response_models "
                    "must be a count mapping"
                )
            response_id_digest = runtime_identity.get("response_id_digest")
            if response_id_digest is not None:
                require_sha256(
                    response_id_digest,
                    label="llm_runtime_identity.response_id_digest",
                    allow_none=False,
                )
    seed = record.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    metrics = record.get("metrics")
    if not isinstance(metrics, Mapping) or not metrics:
        raise ValueError("metrics must be a non-empty object")
    for metric, value in metrics.items():
        if not isinstance(metric, str) or not metric:
            raise ValueError("metric names must be non-empty strings")
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ValueError(f"metric {metric!r} must be a finite number")
    if schema_version == LEGACY_RUN_RECORD_SCHEMA_VERSION:
        for field in ("resource_budget_fingerprint", "ablation_fingerprint"):
            value = record.get(field)
            if value is not None and not isinstance(value, str):
                raise ValueError(f"{field} must be a string in v1 records")
        return

    for field in _TOP_LEVEL_SHA256_FIELDS:
        require_sha256(record.get(field), label=field)
    for field in (
        "action_selection_mode",
        "reasoning_effort",
        "experiment_phase",
        "arm_id",
        "oss_control",
        "evaluation_perturbation",
        "replication_id",
        "randomization_block",
        "model_cutoff_policy",
    ):
        value = record.get(field)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"{field} must be a non-empty string or null")

    validate_capability_transfer_receipt(record)

    llm_usage = record.get("llm_usage")
    if llm_usage is not None:
        if not isinstance(llm_usage, Mapping):
            raise ValueError("llm_usage must be an object or null")
        for key, value in llm_usage.items():
            if (
                not isinstance(key, str)
                or isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise ValueError(
                    "llm_usage values must be non-negative integers"
                )
    prompt_audit = record.get("prompt_visibility_audit")
    if prompt_audit is not None:
        if not isinstance(prompt_audit, Mapping):
            raise ValueError(
                "prompt_visibility_audit must be an object or null"
            )
        if (
            prompt_audit.get("schema_version")
            != "orgenv_prompt_visibility_audit_v1"
        ):
            raise ValueError("prompt_visibility_audit schema mismatch")
        for field in (
            "calls_audited",
            "characters_audited",
            "violation_count",
        ):
            value = prompt_audit.get(field)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise ValueError(
                    f"prompt_visibility_audit.{field} must be non-negative"
                )
        codes = prompt_audit.get("violation_codes")
        if (
            not isinstance(codes, Sequence)
            or isinstance(codes, (str, bytes))
            or not all(isinstance(value, str) and value for value in codes)
            or len(set(codes)) != len(codes)
        ):
            raise ValueError(
                "prompt_visibility_audit.violation_codes invalid"
            )
        if prompt_audit.get("clear") is not (
            prompt_audit.get("violation_count") == 0
        ):
            raise ValueError("prompt_visibility_audit.clear inconsistent")
        require_sha256(
            prompt_audit.get("audit_chain_hash"),
            label="prompt_visibility_audit.audit_chain_hash",
            allow_none=False,
        )
        declared_audit_hash = require_sha256(
            prompt_audit.get("audit_hash"),
            label="prompt_visibility_audit.audit_hash",
            allow_none=False,
        )
        unhashed_audit = dict(prompt_audit)
        unhashed_audit.pop("audit_hash", None)
        if stable_fingerprint(unhashed_audit) != declared_audit_hash:
            raise ValueError("prompt_visibility_audit hash mismatch")
    contamination_status = record.get("contamination_status")
    if (
        contamination_status is not None
        and contamination_status not in _CONTAMINATION_STATUSES
    ):
        raise ValueError(
            "contamination_status must be one of "
            f"{sorted(_CONTAMINATION_STATUSES)} or null"
        )
    order = record.get("randomization_order")
    if order is not None:
        if isinstance(order, bool):
            raise ValueError("randomization_order must not be boolean")
        if isinstance(order, int):
            if order < 0:
                raise ValueError("randomization_order must be non-negative")
        elif isinstance(order, Sequence) and not isinstance(order, (str, bytes)):
            if (
                not order
                or not all(isinstance(item, str) and item for item in order)
                or len(set(order)) != len(order)
            ):
                raise ValueError(
                    "randomization_order sequence must contain unique strings"
                )
        else:
            raise ValueError(
                "randomization_order must be an integer, string sequence, or null"
            )
    _as_status(
        record.get("status"),
        field="status",
        allowed=_RUN_STATUSES,
    )
    for field in ("started_at", "ended_at"):
        if record.get(field) is not None:
            _normalize_timestamp(record[field])
    if record.get("duration_seconds") is not None:
        _resolve_timing(None, None, record["duration_seconds"])
    failure_reason = record.get("failure_reason")
    if failure_reason is not None and not isinstance(failure_reason, str):
        raise ValueError("failure_reason must be a string or null")
    clearance = record.get("contamination_clearance")
    _strict_optional_bool(clearance, field="contamination_clearance")
    probes = record.get("contamination_probes")
    if probes is not None:
        probes = _normalize_contamination_probes(probes)
    if clearance is True:
        if contamination_status != "cleared":
            raise ValueError(
                "contamination_clearance=true requires contamination_status=cleared"
            )
        if not probes or not isinstance(probes.get("results"), Mapping):
            raise ValueError(
                "contamination_clearance=true requires recorded probe results"
            )
    checkpoint = record.get("checkpoint")
    if checkpoint is not None:
        if not isinstance(checkpoint, Mapping):
            raise ValueError("checkpoint must be an object or null")
        require_sha256(
            checkpoint.get("sha256"),
            label="checkpoint.sha256",
        )
    if not isinstance(record.get("provenance"), Mapping):
        raise ValueError("provenance must be an object")
    capability_evidence = record.get("organizational_capability_evidence")
    capability_evidence_hash = record.get(
        "organizational_capability_evidence_hash"
    )
    if capability_evidence is not None:
        if not isinstance(capability_evidence, Mapping):
            raise ValueError(
                "organizational_capability_evidence must be an object or null"
            )
        from environments.org_env.experiments.capability_evidence import (
            validate_organizational_capability_evidence,
        )

        validate_organizational_capability_evidence(capability_evidence)
        if capability_evidence_hash != capability_evidence.get("evidence_hash"):
            raise ValueError(
                "organizational_capability_evidence_hash mismatch"
            )
    elif capability_evidence_hash is not None:
        raise ValueError(
            "organizational_capability_evidence_hash requires evidence"
        )
    profile_evidence = record.get("profile_causality_evidence")
    profile_evidence_hash = record.get("profile_causality_evidence_hash")
    if profile_evidence is not None:
        if not isinstance(profile_evidence, Mapping):
            raise ValueError(
                "profile_causality_evidence must be an object or null"
            )
        from environments.org_env.experiments.profile_causality import (
            validate_profile_causality_evidence,
        )

        validate_profile_causality_evidence(profile_evidence)
        if profile_evidence_hash != profile_evidence.get("evidence_hash"):
            raise ValueError("profile_causality_evidence_hash mismatch")
    elif profile_evidence_hash is not None:
        raise ValueError("profile_causality_evidence_hash requires evidence")
    references = record.get("capability_evidence_refs")
    if not isinstance(references, Sequence) or isinstance(references, (str, bytes)):
        raise ValueError("capability_evidence_refs must be a list")
    _strict_optional_bool(
        record.get("capability_evidence_refs_truncated"),
        field="capability_evidence_refs_truncated",
    )

    final = record.get("final_evaluation")
    if final is None:
        return
    if not isinstance(final, Mapping):
        raise ValueError("final_evaluation must be an object or null")
    for field in (
        "plan_hash",
        "candidate_repo_digest",
        "result_hash",
        "artifact_hash",
    ):
        require_sha256(
            final.get(field),
            label=f"final_evaluation.{field}",
        )
    _as_status(
        final.get("status"),
        field="final_evaluation.status",
        allowed=_FINAL_STATUSES,
    )
    for field in (
        "causal_fix_count",
        "unresolved_count",
        "regression_count",
        "infrastructure_error_count",
    ):
        _as_count(final.get(field), field=field)
    for field in ("causal_fix_rate", "candidate_pass_rate"):
        _as_rate(final.get(field), field=field)
    for field in ("formal_claim_ready", "formal_release_ready"):
        _as_optional_bool(final.get(field), field=field)
    kind = final.get("evidence_kind")
    if kind is not None and (not isinstance(kind, str) or not kind):
        raise ValueError("final_evaluation.evidence_kind must be a string")
    evidence_hashes = final.get("evidence_hashes")
    if evidence_hashes is not None:
        if not isinstance(evidence_hashes, Sequence) or isinstance(
            evidence_hashes, (str, bytes)
        ):
            raise ValueError("final_evaluation.evidence_hashes must be a list")
        for evidence_hash in evidence_hashes:
            require_sha256(
                evidence_hash,
                label="final_evaluation.evidence_hashes[]",
                allow_none=False,
            )
    _manual_evidence_fields(final)


def run_record_caveats(record: Mapping[str, Any]) -> list[str]:
    completeness = assess_run_record_completeness(record)
    caveats: list[str] = []
    provenance = record.get("provenance")
    if isinstance(provenance, Mapping) and provenance.get(
        "upgraded_from_schema_version"
    ):
        caveats.append(
            "legacy_schema_upgraded:" + str(provenance["upgraded_from_schema_version"])
        )
    if completeness["missing_fields"]:
        caveats.append("missing_fields:" + ",".join(completeness["missing_fields"]))
    if record.get("final_evaluation") is None:
        caveats.append("final_evaluation_not_supplied")
    if record.get("contamination_probes") is None:
        caveats.append("contamination_probes_not_recorded")
    if record.get("contamination_clearance") is not True:
        caveats.append("contamination_not_cleared")
    checkpoint = record.get("checkpoint")
    if isinstance(checkpoint, Mapping) and not checkpoint.get("sha256"):
        caveats.append("checkpoint_not_content_addressed")
    status = str(record.get("status") or "").lower()
    if status in {
        "failed",
        "error",
        "timeout",
        "infra_error",
        "blocked",
    } and not record.get("failure_reason"):
        caveats.append("failed_run_without_failure_reason")
    return list(dict.fromkeys(caveats))


def _annotate_record(
    record: Mapping[str, Any],
    *,
    preserve_caveats: Sequence[str] = (),
) -> dict[str, Any]:
    result = dict(record)
    result["provenance"] = sanitize_provenance(result.get("provenance") or {})
    validate_experiment_run_record_schema(result)
    result["completeness"] = assess_run_record_completeness(result)
    generated = run_record_caveats(result)
    result["caveats"] = list(
        dict.fromkeys([str(item) for item in preserve_caveats] + generated)
    )
    return result


def merge_final_evaluator_evidence(
    record: Mapping[str, Any],
    final_evaluator: MappingSource,
    *,
    dataset_manifest: Mapping[str, Any] | None = None,
    expected_plan_hash: str | None = None,
    expected_candidate_repo_digest: str | None = None,
) -> dict[str, Any]:
    """Merge evaluator-owned final evidence and reject contradictory lineage."""

    merged = dict(record)
    candidate_expected = _first(
        expected_candidate_repo_digest,
        merged.get("candidate_repo_digest"),
    )
    final, payload = normalize_final_evaluation(
        final_evaluator,
        expected_dataset_id=str(merged.get("pack") or "") or None,
        expected_plan_hash=expected_plan_hash,
        expected_candidate_repo_digest=(
            str(candidate_expected) if candidate_expected is not None else None
        ),
        dataset_manifest=dataset_manifest,
    )
    existing_final = merged.get("final_evaluation")
    if existing_final is not None:
        if not isinstance(existing_final, Mapping):
            raise ValueError("existing final_evaluation must be an object")
        if stable_fingerprint(existing_final) != stable_fingerprint(final):
            raise ValueError(
                "refusing to overwrite a different existing final_evaluation"
            )
    sources = _evaluation_sources(payload)
    for field in _PROVENANCE_HASH_FIELDS:
        evaluator_value = _consensus(sources, field)
        if evaluator_value is None and field == "candidate_repo_digest":
            evaluator_value = final.get(field)
        if evaluator_value is None:
            continue
        evaluator_text = require_sha256(
            evaluator_value,
            label=f"final evaluator {field}",
            allow_none=False,
        )
        existing = merged.get(field)
        if _nonempty(existing) and str(existing) != evaluator_text:
            raise ValueError(
                f"final evaluator {field} mismatch: "
                f"record={existing} evaluator={evaluator_text}"
            )
        merged[field] = evaluator_text
    merged["final_evaluation"] = (
        dict(existing_final) if existing_final is not None else final
    )
    metrics = dict(merged.get("metrics") or {})
    metric_families = dict(merged.get("metric_families") or {})
    capability_evidence = merged.get(
        "organizational_capability_evidence"
    )
    if not isinstance(capability_evidence, Mapping):
        # A record without the capability-evidence packet can never have oracle
        # outcomes attached, so strong emergence stays 0.0 forever. Record that
        # loudly instead of silently skipping the attach step.
        caveats = list(merged.get("caveats") or ())
        caveat = "capability_evidence_missing_independent_outcomes_not_attached"
        if caveat not in caveats:
            caveats.append(caveat)
        merged["caveats"] = caveats
    if isinstance(capability_evidence, Mapping):
        from environments.org_env.experiments.capability_evidence import (
            attach_independent_outcomes,
        )

        capability_evidence = attach_independent_outcomes(
            capability_evidence,
            payload,
        )
        merged["organizational_capability_evidence"] = (
            capability_evidence
        )
        merged["organizational_capability_evidence_hash"] = (
            capability_evidence["evidence_hash"]
        )
        for metric, value in (
            capability_evidence.get("metrics") or {}
        ).items():
            if isinstance(value, (int, float)) and not isinstance(
                value,
                bool,
            ):
                metrics[str(metric)] = float(value)
                metric_families[str(metric)] = (
                    "organizational_capability"
                )
        # The stage flag lives at the packet top level (not in its metrics
        # map): reflect the post-attach stage into the record metrics so
        # strong_protocol_emergence_rate is interpretable in isolation.
        metrics["strong_emergence_oracle_attached"] = (
            1.0
            if capability_evidence.get("oracle_attachment") == "attached"
            else 0.0
        )
        metric_families["strong_emergence_oracle_attached"] = (
            "organizational_capability"
        )
    final_metric_map = {
        "oss_hidden_pass_rate": final.get("candidate_pass_rate"),
        # The number the 90%-functional-overlap goal is stated in: of the
        # user-reachable capabilities the next release added, the share the
        # candidate also exposes - implementation route deliberately ignored.
        "functional_overlap_rate": final.get("functional_overlap_rate"),
        "causal_fix_count": final.get("causal_fix_count"),
        "causal_fix_rate": final.get("causal_fix_rate"),
        "unresolved_count": final.get("unresolved_count"),
        "hidden_regression_count": final.get("regression_count"),
        "hidden_infrastructure_error_count": final.get(
            "infrastructure_error_count"
        ),
    }
    for metric, value in final_metric_map.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            metrics[metric] = float(value)
            metric_families[metric] = "product_effectiveness"
    merged["metrics"] = metrics
    merged["metric_families"] = metric_families
    provenance = dict(merged.get("provenance") or {})
    provenance["final_evaluator_source"] = _source_description(final_evaluator)
    provenance["final_evaluator_result_hash"] = final.get("result_hash")
    merged["provenance"] = provenance
    preserved = [
        item
        for item in (record.get("caveats") or ())
        if not str(item).startswith("missing_fields:")
        and item != "final_evaluation_not_supplied"
    ]
    return _annotate_record(merged, preserve_caveats=preserved)


def upgrade_experiment_run_record(
    record: Mapping[str, Any],
    *,
    final_evaluator: MappingSource | None = None,
    dataset_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Upgrade a v1/wide record without inventing unavailable provenance."""

    upgraded = dict(record)
    prior_version = upgraded.get("schema_version")
    if prior_version not in KNOWN_RUN_RECORD_SCHEMA_VERSIONS:
        raise ValueError(
            f"unknown or missing run record schema_version: {prior_version!r}"
        )
    validate_experiment_run_record_schema(upgraded)
    legacy_evidence = upgraded.pop("time_machine_evaluation", None)
    if legacy_evidence is not None and not isinstance(legacy_evidence, Mapping):
        raise ValueError("time_machine_evaluation must be an object")
    if final_evaluator is not None and legacy_evidence is not None:
        explicit_final, _ = normalize_final_evaluation(
            final_evaluator,
            expected_dataset_id=str(upgraded.get("pack") or "") or None,
            dataset_manifest=dataset_manifest,
        )
        legacy_final, _ = normalize_final_evaluation(
            legacy_evidence,
            expected_dataset_id=str(upgraded.get("pack") or "") or None,
            dataset_manifest=dataset_manifest,
        )
        if stable_fingerprint(explicit_final) != stable_fingerprint(legacy_final):
            raise ValueError("conflicting duplicate legacy final evidence")
    legacy_fingerprints: dict[str, str] = {}
    if prior_version == LEGACY_RUN_RECORD_SCHEMA_VERSION:
        for field in (
            "resource_budget_fingerprint",
            "ablation_fingerprint",
        ):
            value = upgraded.get(field)
            if value in (None, ""):
                upgraded[field] = None
                continue
            try:
                require_sha256(value, label=field, allow_none=False)
            except ValueError:
                legacy_fingerprints[field] = str(value)
                upgraded[field] = None
    upgraded["schema_version"] = RUN_RECORD_SCHEMA_VERSION
    for field in _PAPER_TOP_LEVEL_FIELDS:
        if field == "capability_evidence_refs":
            upgraded.setdefault(field, [])
        else:
            upgraded.setdefault(field, None)
    upgraded.setdefault("failure_reason", None)
    provenance = dict(upgraded.get("provenance") or {})
    provenance.setdefault(
        "record_builder",
        "environments.org_env.experiments.records.upgrade_experiment_run_record",
    )
    if prior_version != RUN_RECORD_SCHEMA_VERSION:
        provenance["upgraded_from_schema_version"] = prior_version
    if legacy_fingerprints:
        provenance["legacy_non_sha256_fingerprints"] = legacy_fingerprints
    upgraded["provenance"] = provenance
    if final_evaluator is None and legacy_evidence is not None:
        final_evaluator = legacy_evidence
    if final_evaluator is not None:
        return merge_final_evaluator_evidence(
            upgraded,
            final_evaluator,
            dataset_manifest=dataset_manifest,
        )
    return _annotate_record(
        upgraded,
        preserve_caveats=record.get("caveats") or (),
    )


def build_experiment_run_record(
    world: Any,
    *,
    pack: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    metric_families: Mapping[str, str] | None = None,
    reasoning_effort: str | None = None,
    action_selection_mode: str | None = None,
    replication_id: str | None = None,
    randomization_block: Any = None,
    randomization_order: Any = None,
    run_manifest: MappingSource | None = None,
    run_manifest_hash: str | None = None,
    dataset_manifest: MappingSource | None = None,
    dataset_manifest_hash: str | None = None,
    starter_repo: str | os.PathLike[str] | None = None,
    starter_repo_digest: str | None = None,
    reference_repo: str | os.PathLike[str] | None = None,
    reference_repo_digest: str | None = None,
    candidate_repo: str | os.PathLike[str] | None = None,
    candidate_repo_digest: str | None = None,
    hidden_suite: Any = None,
    hidden_suite_hash: str | None = None,
    evaluator_environment: Any = None,
    evaluator_environment_hash: str | None = None,
    tool_surface: Any = None,
    tool_surface_fingerprint: str | None = None,
    information_budget: Any = None,
    information_budget_fingerprint: str | None = None,
    model_cutoff_policy: str | None = None,
    contamination_status: str | None = None,
    contamination_probes: MappingSource | None = None,
    contamination_clearance: bool | None = None,
    started_at: Any = None,
    ended_at: Any = None,
    duration_seconds: float | None = None,
    status: str | None = None,
    failure_reason: str | None = None,
    checkpoint: Mapping[str, Any] | str | os.PathLike[str] | None = None,
    provenance: Mapping[str, Any] | None = None,
    event_graph: Any = None,
    event_graph_hash: str | None = None,
    capability_evidence_refs: Sequence[Any] | str | None = None,
    capability_evidence_refs_truncated: bool | None = None,
    final_evaluator: MappingSource | None = None,
) -> dict[str, Any]:
    """Build one v2 wide record accepted by the unchanged statistics parser."""

    collected = OrgMetrics().compute(state=world)
    quality = collected.get("product_quality") or {}
    org = collected.get("org_metrics") or {}
    capability_evidence = collected.get(
        "organizational_capability_evidence"
    )
    profile_causality_evidence = collected.get(
        "profile_causality_evidence"
    )
    metrics = {
        name: float(collected[name])
        for name in ALL_METRICS
        if isinstance(collected.get(name), (int, float))
    }
    for name in _DEFAULT_PRODUCT_METRICS:
        if isinstance(quality.get(name), (int, float, bool)):
            metrics[name] = float(quality[name])
    for name in (
        "task_cycle_time_avg",
        "pr_review_latency_avg",
        "pr_merge_latency_avg",
        "rework_rate",
        "token_burn_total",
        "budget_burn_per_merged_pr",
        "merged_pr_count",
        "release_count",
    ):
        if isinstance(org.get(name), (int, float)):
            metrics[name] = float(org[name])

    params = _scenario_params(world)
    oss_control_state = getattr(world, "_oss_control", {}) or {}
    client = getattr(world, "llm_client", None)
    inner = getattr(client, "inner", client)
    resolved_provider = provider or getattr(inner, "provider", None) or "none"
    resolved_model = model or getattr(inner, "model", None) or "rules"
    (
        resolved_llm_runtime_identity,
        resolved_llm_runtime_fingerprint,
    ) = _llm_runtime_payload(
        inner,
        provider=str(resolved_provider),
        model=str(resolved_model),
    )
    # P4a: record the EFFECTIVE reasoning effort (None for non-reasoning models
    # and rule-based runs) so the model-family axis is auditable per run.
    resolved_reasoning_effort = _first(
        reasoning_effort,
        getattr(inner, "effective_reasoning_effort", None),
    )
    resolved_pack = pack or _infer_pack(world)
    resources = experiment_resource_snapshot(world) or {}
    # P4b: actual API token usage from the cognitive-layer client (None for
    # rule-based runs). Distinct from resource_usage, which is the evaluator's
    # frozen REQUESTED-token ledger, and from token_burn_total, which is the
    # simulated in-world budget — this field is the only real-token axis.
    llm_usage = _llm_usage_payload(client)
    # The cost axis a paper can publish: real tokens the provider billed,
    # divided by work that reached mainline. The in-world ledger cannot stand
    # in for it — it charges a flat 0.4 per call whatever the call cost, so a
    # whole-file rewrite and a one-line edit are priced the same, and it adds
    # a constant daily burn that a quiet arm cannot amortise.
    merged_for_cost = org.get("merged_pr_count")
    real_tokens = (llm_usage or {}).get("total_tokens")
    if isinstance(merged_for_cost, (int, float)) and merged_for_cost:
        if isinstance(real_tokens, (int, float)):
            metrics["llm_tokens_per_merged_pr"] = round(
                float(real_tokens) / float(merged_for_cost), 1
            )
    prompt_visibility_audit = _prompt_visibility_audit_payload(client)
    ablations = getattr(world, "mechanism_ablations", None)
    resolved_families = {name: _default_metric_family(name) for name in metrics}
    resolved_families.update(dict(metric_families or {}))

    from environments.org_env.product.substrates.eval_assets import (
        oss_eval_assets,
    )

    assets_raw = oss_eval_assets(world)
    assets = assets_raw if isinstance(assets_raw, Mapping) else {}
    resolved_dataset_source = (
        dataset_manifest if dataset_manifest is not None else assets.get("manifest")
    )
    manifest_payload, resolved_dataset_hash = _resolve_mapping_hash(
        resolved_dataset_source,
        dataset_manifest_hash,
        label="dataset_manifest_hash",
    )
    manifest_id = _manifest_project_id(manifest_payload)
    if (
        manifest_id is not None
        and resolved_pack not in ("", "unknown_pack")
        and manifest_id != resolved_pack
    ):
        raise ValueError(
            "record/dataset manifest mismatch: "
            f"pack={resolved_pack} manifest={manifest_id}"
        )
    contamination_controls = (manifest_payload or {}).get(
        "contamination_controls"
    ) or {}
    if not isinstance(contamination_controls, Mapping):
        raise ValueError("manifest contamination_controls must be an object")
    resolved_model_cutoff = _first(
        model_cutoff_policy,
        params.get("model_cutoff_policy"),
        (manifest_payload or {}).get("model_cutoff_policy"),
        contamination_controls.get("model_cutoff_policy"),
    )
    resolved_probe_source = _first(
        contamination_probes,
        params.get("contamination_probes"),
        _derived_contamination_probe_path(assets, manifest_payload),
    )
    resolved_contamination_probes = _normalize_contamination_probes(
        resolved_probe_source
    )
    resolved_contamination_status = _first(
        contamination_status,
        params.get("contamination_status"),
        (manifest_payload or {}).get("contamination_status"),
        contamination_controls.get("status"),
        (
            "probes_defined_not_executed"
            if resolved_contamination_probes is not None
            and "results" not in resolved_contamination_probes
            else None
        ),
    )
    resolved_contamination_clearance = _strict_optional_bool(
        _first(
            contamination_clearance,
            params.get("contamination_clearance"),
            (manifest_payload or {}).get("contamination_clearance"),
            contamination_controls.get("clearance"),
        ),
        field="contamination_clearance",
    )
    if (
        resolved_contamination_status is not None
        and resolved_contamination_status not in _CONTAMINATION_STATUSES
    ):
        raise ValueError(
            "contamination_status must be one of "
            f"{sorted(_CONTAMINATION_STATUSES)}"
        )
    if resolved_contamination_clearance is True:
        if resolved_contamination_status != "cleared":
            raise ValueError(
                "contamination_clearance=true requires contamination_status=cleared"
            )
        if (
            not resolved_contamination_probes
            or not isinstance(
                resolved_contamination_probes.get("results"),
                Mapping,
            )
        ):
            raise ValueError(
                "contamination_clearance=true requires recorded probe results"
            )

    configured_run_manifest = (
        run_manifest if run_manifest is not None else params.get("run_manifest")
    )
    run_manifest_was_derived = False
    resolved_run_source = configured_run_manifest
    if resolved_run_source is None and run_manifest_hash is None:
        scenario = getattr(world, "scenario", None)
        resolved_run_source = {
            "run_id": str(getattr(world, "run_id", "")),
            "pack": str(resolved_pack),
            "condition": str(getattr(world, "experiment_condition", "B3")),
            "provider": str(resolved_provider),
            "model": str(resolved_model),
            "seed": int(getattr(scenario, "seed", 0)),
            "action_selection_mode": _first(
                action_selection_mode,
                params.get("action_selection_mode"),
                getattr(world, "action_selection_mode", None),
            ),
            "replication_id": _first(
                replication_id,
                params.get("replication_id"),
            ),
            "randomization_block": _first(
                randomization_block,
                params.get("randomization_block"),
            ),
            "randomization_order": _first(
                randomization_order,
                params.get("randomization_order"),
            ),
            "scenario": {
                "name": str(getattr(scenario, "name", "")),
                "corpus_version": str(getattr(scenario, "corpus_version", "")),
                "params": dict(params),
            },
            "llm_decides_actions": bool(getattr(world, "llm_decides_actions", False)),
        }
        run_manifest_was_derived = True
    _, resolved_run_hash = _resolve_mapping_hash(
        resolved_run_source,
        run_manifest_hash,
        label="run_manifest_hash",
    )

    starter_path = _first(
        starter_repo,
        _derived_starter_path(assets, manifest_payload),
    )
    reference_path = _first(
        reference_repo,
        assets.get("reference_repo_dir"),
    )
    hidden_source = _first(
        hidden_suite,
        assets.get("hidden_tests_dir"),
    )
    resolved_starter_digest = _resolve_repo_digest(
        starter_path,
        starter_repo_digest,
        label="starter_repo_digest",
    )
    resolved_reference_digest = _resolve_repo_digest(
        reference_path,
        reference_repo_digest,
        label="reference_repo_digest",
    )
    resolved_candidate_digest = _resolve_repo_digest(
        candidate_repo,
        candidate_repo_digest,
        label="candidate_repo_digest",
    )
    resolved_hidden_hash = _resolve_hidden_suite_hash(
        hidden_source,
        hidden_suite_hash,
    )

    resolved_evaluator_environment = _first(
        evaluator_environment,
        params.get("evaluator_environment"),
    )
    if (
        resolved_evaluator_environment is None
        and evaluator_environment_hash is None
        and final_evaluator is None
        and resolved_hidden_hash is not None
    ):
        resolved_evaluator_environment = {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": sys.platform,
            "runner": (
                "environments.org_env.product.substrates.eval_assets."
                "run_oss_hidden_tests_for_spec"
            ),
            "hidden_suite_hash": resolved_hidden_hash,
        }
    resolved_tool_surface = _first(
        tool_surface,
        params.get("tool_surface"),
    )
    if resolved_tool_surface is None:
        from environments.org_env.backend.actions import registered_action_types

        resolved_tool_surface = {
            "registry": "orgenv_action_registry",
            "actions": sorted(registered_action_types()),
        }
    resolved_information_budget = _first(
        information_budget,
        params.get("information_budget"),
    )
    if resolved_information_budget is None and resources:
        resolved_information_budget = {
            "visibility_model": "agent_local_perception_v1",
            "frozen_resource_budget": resources.get("budget"),
        }
    resolved_environment_hash = resolve_fingerprint(
        resolved_evaluator_environment,
        evaluator_environment_hash,
        label="evaluator_environment_hash",
    )
    resolved_tool_fingerprint = resolve_fingerprint(
        resolved_tool_surface,
        tool_surface_fingerprint,
        label="tool_surface_fingerprint",
    )
    resolved_information_fingerprint = resolve_fingerprint(
        resolved_information_budget,
        information_budget_fingerprint,
        label="information_budget_fingerprint",
    )

    observed_graph = (
        event_graph if event_graph is not None else getattr(world, "event_graph", None)
    )
    computed_graph_hash = event_graph_fingerprint(observed_graph)
    if (
        _nonempty(event_graph_hash)
        and computed_graph_hash is not None
        and str(event_graph_hash) != computed_graph_hash
    ):
        raise ValueError(
            "event_graph_hash mismatch: "
            f"supplied={event_graph_hash} computed={computed_graph_hash}"
        )
    resolved_graph_hash = computed_graph_hash or (
        str(event_graph_hash) if _nonempty(event_graph_hash) else None
    )

    resolved_start, resolved_end, resolved_duration = _resolve_timing(
        _first(
            started_at,
            getattr(world, "run_started_at", None),
            getattr(world, "started_at", None),
        ),
        _first(
            ended_at,
            getattr(world, "run_ended_at", None),
            getattr(world, "ended_at", None),
        ),
        _first(
            duration_seconds,
            getattr(world, "run_duration_seconds", None),
        ),
    )
    resolved_status = _first(
        status,
        getattr(world, "run_status", None),
    )
    resolved_failure = _first(
        failure_reason,
        getattr(world, "failure_reason", None),
    )
    resolved_action_mode = _first(
        action_selection_mode,
        params.get("action_selection_mode"),
        getattr(world, "action_selection_mode", None),
        "llm" if bool(getattr(world, "llm_decides_actions", False)) else "policy",
    )
    resolved_replication = _first(
        replication_id,
        params.get("replication_id"),
    )
    resolved_randomization_block = _first(
        randomization_block,
        params.get("randomization_block"),
    )
    resolved_randomization_order = _first(
        randomization_order,
        params.get("randomization_order"),
    )

    provenance_payload = dict(provenance or {})
    provenance_payload.setdefault(
        "record_builder",
        "environments.org_env.experiments.records.build_experiment_run_record",
    )
    provenance_payload.setdefault("record_schema_version", RUN_RECORD_SCHEMA_VERSION)
    provenance_payload.setdefault(
        "execution_job_id",
        os.environ.get("ORG_EXECUTION_JOB_ID"),
    )
    provenance_payload.setdefault(
        "source_provenance_fingerprint",
        os.environ.get("ORG_SOURCE_PROVENANCE_FINGERPRINT"),
    )
    provenance_payload.setdefault(
        "model_binding_fingerprint",
        os.environ.get("ORG_MODEL_BINDING_FINGERPRINT"),
    )
    provenance_payload.setdefault(
        "execution_resource_budget_fingerprint",
        os.environ.get("ORG_EXECUTION_RESOURCE_BUDGET_FINGERPRINT"),
    )
    if resolved_run_source is not None:
        provenance_payload.setdefault(
            "run_manifest_source",
            (
                "derived_world_configuration"
                if run_manifest_was_derived
                else _source_description(resolved_run_source)
            ),
        )
    if resolved_dataset_source is not None:
        provenance_payload.setdefault(
            "dataset_manifest_source",
            _source_description(resolved_dataset_source),
        )
    if resolved_probe_source is not None:
        provenance_payload.setdefault(
            "contamination_probe_source",
            _source_description(resolved_probe_source),
        )

    record = {
        "schema_version": RUN_RECORD_SCHEMA_VERSION,
        "run_id": str(getattr(world, "run_id", "")),
        "pack": str(resolved_pack),
        "condition": str(getattr(world, "experiment_condition", "B3")),
        # Keep the execution boundary explicit even for host-controlled pilots.
        # The evaluator environment hash binds exact runtime contents, but does
        # not by itself reveal whether those contents ran under Docker,
        # Apptainer, or Bubblewrap.  Mixed continuation batches need both axes
        # in every arm rather than relying on the launcher log.
        "experiment_mode": _first(
            params.get("experiment_mode"),
            os.environ.get("ORG_OSS_MODE"),
        ),
        "evaluator_backend": _first(
            os.environ.get("ORG_EVALUATOR_BACKEND"),
            params.get("evaluator_backend"),
        ),
        "experiment_phase": _first(
            params.get("experiment_phase"),
            os.environ.get("ORG_EXPERIMENT_PHASE"),
        ),
        "arm_id": _first(
            params.get("arm_id"),
            os.environ.get("ORG_EXPERIMENT_ARM_ID"),
            getattr(getattr(world, "condition_spec", None), "short_name", None),
        ),
        "oss_control": _first(
            oss_control_state.get("control")
            if isinstance(oss_control_state, Mapping)
            else None,
            params.get("oss_control"),
            os.environ.get("ORG_OSS_CONTROL"),
            "none",
        ),
        "evaluation_perturbation": _first(
            params.get("evaluation_perturbation"),
            os.environ.get("ORG_EXPERIMENT_EVALUATION_PERTURBATION"),
            "none",
        ),
        # What this arm inherited, written by inject_capability_bundle. A
        # transfer arm whose record does not say what it received cannot be
        # told apart from one that inherited nothing, and an arm that silently
        # inherited nothing is not a transfer arm at all.
        "capability_transfer": (
            getattr(world, "__dict__", {}).get("_capability_transfer_receipt")
        ),
        "provider": str(resolved_provider),
        "model": str(resolved_model),
        "llm_runtime_identity": resolved_llm_runtime_identity,
        "llm_runtime_fingerprint": resolved_llm_runtime_fingerprint,
        "reasoning_effort": (
            str(resolved_reasoning_effort)
            if resolved_reasoning_effort is not None
            else None
        ),
        "seed": int(getattr(getattr(world, "scenario", None), "seed", 0)),
        "metrics": metrics,
        "metric_families": resolved_families,
        "resource_budget_fingerprint": str(resources.get("budget_fingerprint", "")),
        "resource_usage": resources,
        "llm_usage": llm_usage,
        "prompt_visibility_audit": prompt_visibility_audit,
        "ablation_fingerprint": str(getattr(ablations, "fingerprint", "")),
        "mechanism_ablations": (ablations.to_dict() if ablations is not None else None),
        "action_selection_mode": resolved_action_mode,
        "profile_assignment": str(
            getattr(
                getattr(world, "condition_spec", None), "profile_assignment", "aligned"
            )
        ),
        "replication_id": resolved_replication,
        "randomization_block": resolved_randomization_block,
        "randomization_order": resolved_randomization_order,
        "run_manifest_hash": resolved_run_hash,
        "dataset_manifest_hash": resolved_dataset_hash,
        "starter_repo_digest": resolved_starter_digest,
        "reference_repo_digest": resolved_reference_digest,
        "candidate_repo_digest": resolved_candidate_digest,
        "hidden_suite_hash": resolved_hidden_hash,
        "evaluator_environment_hash": resolved_environment_hash,
        "tool_surface_fingerprint": resolved_tool_fingerprint,
        "information_budget_fingerprint": resolved_information_fingerprint,
        "model_cutoff_policy": resolved_model_cutoff,
        "contamination_status": resolved_contamination_status,
        "contamination_probes": resolved_contamination_probes,
        "contamination_clearance": resolved_contamination_clearance,
        "started_at": resolved_start,
        "ended_at": resolved_end,
        "duration_seconds": resolved_duration,
        "status": resolved_status,
        "failure_reason": resolved_failure,
        "checkpoint": checkpoint_metadata(checkpoint),
        "provenance": provenance_payload,
        "event_graph_hash": resolved_graph_hash,
        "organizational_capability_evidence_hash": (
            capability_evidence.get("evidence_hash")
            if isinstance(capability_evidence, Mapping)
            else None
        ),
        "organizational_capability_evidence": capability_evidence,
        "profile_causality_evidence_hash": (
            profile_causality_evidence.get("evidence_hash")
            if isinstance(profile_causality_evidence, Mapping)
            else None
        ),
        "profile_causality_evidence": profile_causality_evidence,
        "capability_evidence_refs": _capability_refs(world, capability_evidence_refs),
        "capability_evidence_refs_truncated": _strict_optional_bool(
            (
                capability_evidence_refs_truncated
                if capability_evidence_refs_truncated is not None
                else _capability_refs_are_truncated(world)
            ),
            field="capability_evidence_refs_truncated",
        ),
        # Descriptive, deliberately outside `metrics`: the funnel is not a
        # confirmatory quantity and must not enter a metric family's
        # multiplicity correction. It travels whole because its content is the
        # SHAPE of the drop-off, which a run reporting only a terminal zero
        # cannot be read for.
        "delivery_funnel": collected.get("delivery_funnel") or {},
        "final_evaluation": None,
    }
    if final_evaluator is not None:
        return merge_final_evaluator_evidence(
            record,
            final_evaluator,
            dataset_manifest=manifest_payload,
            expected_candidate_repo_digest=resolved_candidate_digest,
        )
    return _annotate_record(record)


__all__ = [
    "KNOWN_RUN_RECORD_SCHEMA_VERSIONS",
    "LEGACY_RUN_RECORD_SCHEMA_VERSION",
    "RUN_RECORD_SCHEMA_VERSION",
    "assess_run_record_completeness",
    "build_experiment_run_record",
    "merge_final_evaluator_evidence",
    "normalize_final_evaluation",
    "run_record_caveats",
    "upgrade_experiment_run_record",
    "validate_experiment_run_record_schema",
]
