"""Paired statistical analysis for the controlled OSS organization experiment.

The implementation is dependency-free and treats Pack x Provider x Model as
fixed blocks. Seeds are paired within the preregistered design. Estimates are
conditional on the enumerated repositories, model bindings, and seeds; the
repository axis is not sampled and therefore does not support
repository-population inference.

Legacy B4 (shuffled-profile) and B5 (random-profile) controls are optional but
must be present uniformly across paired cells. Their B3-B4 and B3-B5 reports
use a separate Holm family so they cannot perturb the preregistered primary
ladder.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from itertools import product
from random import Random
from typing import Any, Iterable, Mapping, Optional, Sequence

from environments.org_env.experiments.provenance import require_sha256

ORGANIZATION_CONDITIONS = ("B0", "B1", "B2", "B3")
# Retained under the old name because run records and analysis fixtures written
# before the ladder was consolidated still import it.
LEGACY_CONDITIONS = ORGANIZATION_CONDITIONS
KNOWN_CONDITIONS = ORGANIZATION_CONDITIONS
CONDITIONS = ORGANIZATION_CONDITIONS
PRIMARY_CONTRASTS = (("B3", "B2"), ("B2", "B1"), ("B1", "B0"))
FROZEN_DESIGN_INFERENCE_SCOPE = (
    "conditional_on_preregistered_frozen_repositories_models_and_seeds"
)
_KNOWN_RUN_SCHEMAS = frozenset({"orgenv_experiment_run_v1", "orgenv_experiment_run_v2"})
_SUCCESSFUL_RUN_STATUSES = frozenset({"completed"})
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
_LINEAGE_FIELDS = (
    "dataset_manifest_hash",
    "starter_repo_digest",
    "reference_repo_digest",
    "hidden_suite_hash",
    "evaluator_environment_hash",
    "tool_surface_fingerprint",
    "information_budget_fingerprint",
    "llm_runtime_fingerprint",
    "model_cutoff_policy",
    "contamination_status",
    "contamination_probe_hash",
    "contamination_clearance",
    "randomization_block",
    "final_plan_hash",
    "evidence_kind",
)

_CONDITION_ALIASES = {
    "b0": "B0",
    "b0_single_agent_founder": "B0",
    "single_agent_founder": "B0",
    "b1": "B1",
    "b1_persistent_role_org": "B1",
    "b2": "B2",
    "b2_policy_conditioned_org": "B2",
    "b3": "B3",
    "b3_relic_organization": "B3",
    "relic_organization": "B3",
}

# Rungs from the previous ladder. They keep their own labels rather than
# folding into whichever rung inherited their short name: a run recorded as
# b2_persistent_role_org selected actions directly through the LLM, and
# counting it as today's policy-conditioned B2 would pair two different
# treatments as if they were replicates of one.
_RETIRED_CONDITION_ALIASES = {
    "b1_temporary_specialist_team": "B1_TEMPORARY",
    "temporary_specialist_team": "B1_TEMPORARY",
    "b2_persistent_role_org": "B1_LLM_DIRECT_LEGACY",
    "b2_persistent_role_based_organization": "B1_LLM_DIRECT_LEGACY",
    "persistent_role_based_organization": "B1_LLM_DIRECT_LEGACY",
}


def normalize_condition(value: Any) -> str:
    raw = str(value).strip()
    if raw.upper() in KNOWN_CONDITIONS:
        return raw.upper()
    normalized = raw.lower().replace("-", "_").replace(" ", "_")
    if normalized in _CONDITION_ALIASES:
        return _CONDITION_ALIASES[normalized]
    if normalized in _RETIRED_CONDITION_ALIASES:
        return _RETIRED_CONDITION_ALIASES[normalized]
    raise ValueError(f"unknown organization condition: {value!r}")


@dataclass(frozen=True)
class RunObservation:
    run_id: str
    pack: str
    condition: str
    provider: str
    model: str
    seed: int
    metric: str
    value: float
    metric_family: str = "primary"
    resource_budget_fingerprint: str = ""
    ablation_fingerprint: str = ""
    schema_version: str = "orgenv_experiment_run_v1"
    experiment_phase: Optional[str] = None
    arm_id: Optional[str] = None
    oss_control: Optional[str] = None
    evaluation_perturbation: Optional[str] = None
    replication_id: Optional[str] = None
    randomization_block: Optional[str] = None
    randomization_order: int | tuple[str, ...] | None = None
    action_selection_mode: Optional[str] = None
    status: Optional[str] = None
    final_status: Optional[str] = None
    formal_claim_ready: Optional[bool] = None
    formal_release_ready: Optional[bool] = None
    model_cutoff_policy: Optional[str] = None
    contamination_status: Optional[str] = None
    contamination_probe_hash: Optional[str] = None
    contamination_probe_ids: tuple[str, ...] = ()
    contamination_probe_count: Optional[int] = None
    contamination_clearance: Optional[bool] = None
    run_manifest_hash: Optional[str] = None
    dataset_manifest_hash: Optional[str] = None
    starter_repo_digest: Optional[str] = None
    reference_repo_digest: Optional[str] = None
    candidate_repo_digest: Optional[str] = None
    hidden_suite_hash: Optional[str] = None
    evaluator_environment_hash: Optional[str] = None
    tool_surface_fingerprint: Optional[str] = None
    information_budget_fingerprint: Optional[str] = None
    llm_runtime_fingerprint: Optional[str] = None
    event_graph_hash: Optional[str] = None
    final_plan_hash: Optional[str] = None
    final_result_hash: Optional[str] = None
    final_artifact_hash: Optional[str] = None
    evidence_kind: Optional[str] = None
    evidence_hashes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "condition", normalize_condition(self.condition))
        if (
            not self.run_id
            or not self.pack
            or not self.provider
            or not self.model
            or not self.metric
        ):
            raise ValueError("run_id, pack, provider, model, and metric are required")
        if self.schema_version not in _KNOWN_RUN_SCHEMAS:
            raise ValueError(f"unknown run schema: {self.schema_version!r}")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError(f"seed must be an integer for {self.run_id}")
        if isinstance(self.value, bool):
            raise ValueError(f"boolean metric value for {self.run_id}:{self.metric}")
        if not math.isfinite(float(self.value)):
            raise ValueError(f"non-finite value for {self.run_id}:{self.metric}")
        if self.schema_version == "orgenv_experiment_run_v2":
            for field in (
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
                "llm_runtime_fingerprint",
                "event_graph_hash",
                "final_plan_hash",
                "final_result_hash",
                "final_artifact_hash",
                "contamination_probe_hash",
            ):
                require_sha256(getattr(self, field), label=field)
        for field in (
            "formal_claim_ready",
            "formal_release_ready",
            "contamination_clearance",
        ):
            value = getattr(self, field)
            if value is not None and not isinstance(value, bool):
                raise ValueError(f"{field} must be boolean or null")
        if self.final_status is not None and self.final_status not in {
            "passed",
            "incomplete",
            "failed",
            "timeout",
            "infra_error",
            "blocked",
            "blocked_invalid_plan",
            "not_run",
        }:
            raise ValueError(
                f"unknown final evaluator status: {self.final_status!r}"
            )
        if self.randomization_order is not None:
            if isinstance(self.randomization_order, bool):
                raise ValueError("randomization_order must not be boolean")
            if isinstance(self.randomization_order, int):
                if self.randomization_order < 0:
                    raise ValueError("randomization_order must be non-negative")
            elif (
                not isinstance(self.randomization_order, tuple)
                or not self.randomization_order
                or len(set(self.randomization_order)) != len(self.randomization_order)
            ):
                raise ValueError("invalid randomization_order")
        for evidence_hash in self.evidence_hashes:
            require_sha256(
                evidence_hash,
                label="evidence_hashes[]",
                allow_none=False,
            )
        if self.contamination_probe_count is not None and (
            isinstance(self.contamination_probe_count, bool)
            or not isinstance(self.contamination_probe_count, int)
            or self.contamination_probe_count < 0
            or self.contamination_probe_count != len(self.contamination_probe_ids)
        ):
            raise ValueError(
                "contamination_probe_count must match contamination_probe_ids"
            )

    @property
    def cell_key(self) -> tuple[str, str, str, int, str, str, str, str]:
        return (
            self.pack,
            self.provider,
            self.model,
            int(self.seed),
            self.experiment_phase or "",
            self.arm_id or self.condition,
            self.condition,
            self.metric,
        )

    @property
    def pairing_key(self) -> tuple[str, str, str, int, str]:
        return self.pack, self.provider, self.model, int(self.seed), self.metric

    @property
    def block_key(self) -> tuple[str, str, str]:
        return self.pack, self.provider, self.model

    @property
    def lineage_key(self) -> tuple[Any, ...]:
        return tuple(getattr(self, field) for field in _LINEAGE_FIELDS)


@dataclass(frozen=True)
class PairedUnit:
    pack: str
    provider: str
    model: str
    seed: int
    metric: str
    treatment: str
    control: str
    treatment_value: float
    control_value: float

    @property
    def difference(self) -> float:
        return self.treatment_value - self.control_value

    @property
    def block_key(self) -> tuple[str, str, str]:
        return self.pack, self.provider, self.model


@dataclass(frozen=True)
class PairedContrastReport:
    report_id: str
    scope: str
    pack: Optional[str]
    provider: Optional[str]
    model: Optional[str]
    metric: str
    metric_family: str
    treatment: str
    control: str
    pair_count: int
    block_count: int
    mean_difference: float
    confidence_low: float
    confidence_high: float
    confidence_level: float
    bootstrap_samples: int
    p_value: float
    adjusted_p_value: float
    randomization_method: str
    resource_budget_fingerprints: tuple[str, ...]
    caveats: tuple[str, ...] = ()
    inference_scope: str = FROZEN_DESIGN_INFERENCE_SCOPE
    repository_population_inference: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_id": self.report_id,
            "scope": self.scope,
            "pack": self.pack,
            "provider": self.provider,
            "model": self.model,
            "metric": self.metric,
            "metric_family": self.metric_family,
            "treatment": self.treatment,
            "control": self.control,
            "pair_count": self.pair_count,
            "block_count": self.block_count,
            "mean_difference": self.mean_difference,
            "confidence_low": self.confidence_low,
            "confidence_high": self.confidence_high,
            "confidence_level": self.confidence_level,
            "bootstrap_samples": self.bootstrap_samples,
            "p_value": self.p_value,
            "adjusted_p_value": self.adjusted_p_value,
            "randomization_method": self.randomization_method,
            "resource_budget_fingerprints": list(self.resource_budget_fingerprints),
            "caveats": list(self.caveats),
            "inference_scope": self.inference_scope,
            "repository_population_inference": (
                self.repository_population_inference
            ),
        }


def _optional_string(record: Mapping[str, Any], field: str) -> str | None:
    value = record.get(field)
    if value in (None, ""):
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string or null")
    return value


def _optional_boolean(record: Mapping[str, Any], field: str) -> bool | None:
    value = record.get(field)
    if value in (None, ""):
        return None
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be a boolean or null")
    return value


def _randomization_order(value: Any) -> int | tuple[str, ...] | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("randomization_order must not be boolean")
    if isinstance(value, int):
        if value < 0:
            raise ValueError("randomization_order must be non-negative")
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        order = tuple(value)
        if (
            not order
            or not all(isinstance(item, str) and item for item in order)
            or len(set(order)) != len(order)
        ):
            raise ValueError(
                "randomization_order must contain unique condition strings"
            )
        return order
    raise ValueError("randomization_order must be an integer or string list")


def observations_from_payload(payload: Any) -> tuple[RunObservation, ...]:
    """Parse a JSON list, ``{"runs": [...]}``, or JSONL-style record iterable.

    Each run may be wide (``metrics`` mapping) or long (``metric`` + ``value``).
    """

    if isinstance(payload, Mapping):
        records = payload.get("runs")
        if records is None:
            records = [payload]
    else:
        records = payload
    if not isinstance(records, Iterable) or isinstance(records, (str, bytes)):
        raise ValueError("analysis payload must contain an iterable of run records")

    observations: list[RunObservation] = []
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("every run record must be an object")
        schema_version = record.get("schema_version")
        if schema_version not in _KNOWN_RUN_SCHEMAS:
            raise ValueError(
                f"unknown or missing run record schema_version: {schema_version!r}"
            )
        if "seed" not in record:
            raise ValueError(f"run {record.get('run_id')!r} has no seed")
        seed = record["seed"]
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        for field in ("run_id", "pack", "condition", "provider", "model"):
            if not isinstance(record.get(field), str) or not record[field]:
                raise ValueError(f"{field} must be a non-empty string")
        final = record.get("final_evaluation")
        if final is not None and not isinstance(final, Mapping):
            raise ValueError("final_evaluation must be an object or null")
        final = final if isinstance(final, Mapping) else {}
        probes = record.get("contamination_probes")
        if probes is not None and not isinstance(probes, Mapping):
            raise ValueError("contamination_probes must be an object or null")
        probes = probes if isinstance(probes, Mapping) else {}
        probe_ids_raw = probes.get("probe_ids") or ()
        if not isinstance(probe_ids_raw, Sequence) or isinstance(
            probe_ids_raw, (str, bytes)
        ):
            raise ValueError("contamination_probes.probe_ids must be a list")
        probe_ids = tuple(probe_ids_raw)
        if not all(isinstance(item, str) and item for item in probe_ids):
            raise ValueError("contamination_probes.probe_ids must contain strings")
        probe_count = probes.get("probe_count")
        if probe_count is not None and (
            isinstance(probe_count, bool)
            or not isinstance(probe_count, int)
            or probe_count < 0
        ):
            raise ValueError(
                "contamination_probes.probe_count must be non-negative integer"
            )
        evidence_hashes = final.get("evidence_hashes") or ()
        if not isinstance(evidence_hashes, Sequence) or isinstance(
            evidence_hashes, (str, bytes)
        ):
            raise ValueError("final_evaluation.evidence_hashes must be a list")
        status = _optional_string(record, "status")
        if status is not None and status not in _RUN_STATUSES:
            raise ValueError(f"unknown run status: {status!r}")
        common = {
            "run_id": str(record.get("run_id", "")),
            "pack": str(record.get("pack", "")),
            "condition": str(record.get("condition", "")),
            "provider": str(record.get("provider", "")),
            "model": str(record.get("model", "")),
            "seed": seed,
            "resource_budget_fingerprint": (
                _optional_string(record, "resource_budget_fingerprint") or ""
            ),
            "ablation_fingerprint": (
                _optional_string(record, "ablation_fingerprint") or ""
            ),
            "schema_version": str(schema_version),
            "experiment_phase": _optional_string(
                record, "experiment_phase"
            ),
            "arm_id": _optional_string(record, "arm_id"),
            "oss_control": _optional_string(record, "oss_control"),
            "evaluation_perturbation": _optional_string(
                record, "evaluation_perturbation"
            ),
            "replication_id": _optional_string(record, "replication_id"),
            "randomization_block": _optional_string(record, "randomization_block"),
            "randomization_order": _randomization_order(
                record.get("randomization_order")
            ),
            "action_selection_mode": _optional_string(record, "action_selection_mode"),
            "status": status,
            "final_status": _optional_string(final, "status"),
            "formal_claim_ready": _optional_boolean(final, "formal_claim_ready"),
            "formal_release_ready": _optional_boolean(final, "formal_release_ready"),
            "model_cutoff_policy": _optional_string(record, "model_cutoff_policy"),
            "contamination_status": _optional_string(record, "contamination_status"),
            "contamination_probe_hash": _optional_string(probes, "probe_hash"),
            "contamination_probe_ids": probe_ids,
            "contamination_probe_count": probe_count,
            "contamination_clearance": _optional_boolean(
                record, "contamination_clearance"
            ),
            "run_manifest_hash": _optional_string(record, "run_manifest_hash"),
            "dataset_manifest_hash": _optional_string(record, "dataset_manifest_hash"),
            "starter_repo_digest": _optional_string(record, "starter_repo_digest"),
            "reference_repo_digest": _optional_string(record, "reference_repo_digest"),
            "candidate_repo_digest": _optional_string(record, "candidate_repo_digest"),
            "hidden_suite_hash": _optional_string(record, "hidden_suite_hash"),
            "evaluator_environment_hash": _optional_string(
                record, "evaluator_environment_hash"
            ),
            "tool_surface_fingerprint": _optional_string(
                record, "tool_surface_fingerprint"
            ),
            "information_budget_fingerprint": _optional_string(
                record, "information_budget_fingerprint"
            ),
            "llm_runtime_fingerprint": _optional_string(
                record, "llm_runtime_fingerprint"
            ),
            "event_graph_hash": _optional_string(record, "event_graph_hash"),
            "final_plan_hash": _optional_string(final, "plan_hash"),
            "final_result_hash": _optional_string(final, "result_hash"),
            "final_artifact_hash": _optional_string(final, "artifact_hash"),
            "evidence_kind": _optional_string(final, "evidence_kind"),
            "evidence_hashes": tuple(evidence_hashes),
        }
        metrics = record.get("metrics")
        families = record.get("metric_families") or {}
        if not isinstance(families, Mapping):
            raise ValueError("metric_families must be an object")
        if isinstance(metrics, Mapping):
            if not metrics:
                raise ValueError(f"run {common['run_id']!r} has empty metrics")
            for metric, value in metrics.items():
                if not isinstance(metric, str) or not metric:
                    raise ValueError("metric names must be non-empty strings")
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                ):
                    raise ValueError(
                        f"run {common['run_id']!r} metric {metric!r} "
                        "must be finite numeric"
                    )
                observations.append(
                    RunObservation(
                        **common,
                        metric=str(metric),
                        value=float(value),
                        metric_family=str(
                            families.get(metric, record.get("metric_family", "primary"))
                        ),
                    )
                )
        elif "metric" in record and "value" in record:
            value = record["value"]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValueError(f"run {common['run_id']!r} has invalid metric value")
            observations.append(
                RunObservation(
                    **common,
                    metric=str(record["metric"]),
                    value=float(value),
                    metric_family=str(record.get("metric_family", "primary")),
                )
            )
        else:
            raise ValueError(f"run {common['run_id']!r} has no metrics")
    return tuple(observations)


# Provenance sentinel written by tools/oss_run.py. Kept here (not imported from
# tools/) so the analysis layer never depends on a script module.
ADHOC_RUN_SENTINEL = "adhoc_single_run"


def validate_observations(
    observations: Sequence[RunObservation],
    *,
    require_complete_conditions: bool = True,
    require_matched_resources: bool = True,
    require_matched_ablations: bool = True,
    require_complete_run_records: bool = False,
    require_contamination_clearance: bool = False,
    require_v2_records: bool = False,
    expected_conditions: Sequence[str] = CONDITIONS,
    expected_packs: Optional[Sequence[str]] = None,
    expected_seeds_per_block: Optional[int] = None,
) -> None:
    if not observations:
        raise ValueError("no run observations supplied")
    normalized_expected_conditions = tuple(
        normalize_condition(condition) for condition in expected_conditions
    )
    if not normalized_expected_conditions:
        raise ValueError("expected_conditions must not be empty")
    if len(set(normalized_expected_conditions)) != len(
        normalized_expected_conditions
    ):
        raise ValueError("expected_conditions must be unique")
    expected_condition_set = set(normalized_expected_conditions)
    by_cell: dict[tuple[Any, ...], RunObservation] = {}
    by_pairing: dict[tuple[str, str, str, int, str], list[RunObservation]] = {}
    metric_families: dict[str, str] = {}
    for observation in observations:
        if (
            require_v2_records
            and observation.schema_version != "orgenv_experiment_run_v2"
        ):
            raise ValueError(
                f"run {observation.run_id!r} is not a v2 paper run record"
            )
        # A run produced by the direct driver (tools/oss_run.py) carries an
        # explicit ad-hoc provenance sentinel instead of a matrix position. It is
        # useful for debugging but must never enter a preregistered contrast:
        # it has no randomization block, so pairing it would silently break the
        # paired-cell design.
        if ADHOC_RUN_SENTINEL in {
            str(getattr(observation, "randomization_block", "") or ""),
            str(getattr(observation, "experiment_phase", "") or ""),
        }:
            raise ValueError(
                f"run {observation.run_id!r} is an ad-hoc single run "
                f"({ADHOC_RUN_SENTINEL}) and is not eligible for a "
                "preregistered contrast; re-run it through "
                "tools/run_org_baselines.py to obtain a matrix cell"
            )
        is_v2 = observation.schema_version == "orgenv_experiment_run_v2"
        if require_complete_run_records and is_v2:
            required = (
                "experiment_phase",
                "arm_id",
                "oss_control",
                "evaluation_perturbation",
                "replication_id",
                "randomization_block",
                "randomization_order",
                "action_selection_mode",
                "status",
                "final_status",
                "formal_claim_ready",
                "formal_release_ready",
                "model_cutoff_policy",
                "contamination_status",
                "contamination_probe_hash",
                "contamination_probe_count",
                "contamination_clearance",
                "run_manifest_hash",
                "dataset_manifest_hash",
                "starter_repo_digest",
                "reference_repo_digest",
                "candidate_repo_digest",
                "hidden_suite_hash",
                "evaluator_environment_hash",
                "tool_surface_fingerprint",
                "information_budget_fingerprint",
                "llm_runtime_fingerprint",
                "event_graph_hash",
                "final_plan_hash",
                "final_result_hash",
                "final_artifact_hash",
                "evidence_kind",
            )
            missing = [
                field for field in required if getattr(observation, field) is None
            ]
            if not observation.contamination_probe_ids:
                missing.append("contamination_probe_ids")
            if not observation.evidence_hashes:
                missing.append("evidence_hashes")
            if missing:
                raise ValueError(
                    f"run {observation.run_id!r} has incomplete metadata: "
                    f"{sorted(missing)}"
                )
            if observation.status not in _SUCCESSFUL_RUN_STATUSES:
                raise ValueError(
                    f"run {observation.run_id!r} is not analysis-eligible: "
                    f"status={observation.status!r}"
                )
            if observation.final_status not in {"passed", "incomplete"}:
                raise ValueError(
                    f"run {observation.run_id!r} has ineligible final evaluator "
                    f"status={observation.final_status!r}"
                )
        if (
            require_contamination_clearance
            and is_v2
            and observation.contamination_clearance is not True
        ):
            raise ValueError(
                f"run {observation.run_id!r} lacks contamination clearance"
            )
        if observation.cell_key in by_cell:
            prior = by_cell[observation.cell_key]
            raise ValueError(
                "duplicate experiment cell: "
                f"{observation.cell_key} ({prior.run_id}, {observation.run_id})"
            )
        by_cell[observation.cell_key] = observation
        by_pairing.setdefault(observation.pairing_key, []).append(observation)
        prior_family = metric_families.setdefault(
            observation.metric, observation.metric_family
        )
        if prior_family != observation.metric_family:
            raise ValueError(f"metric {observation.metric!r} has inconsistent families")

    if expected_packs is not None:
        present = {observation.pack for observation in observations}
        missing = set(expected_packs) - present
        if missing:
            raise ValueError(f"missing expected packs: {sorted(missing)}")

    for pairing_key, records in by_pairing.items():
        present_conditions = {record.condition for record in records}
        if require_complete_conditions:
            missing = expected_condition_set - present_conditions
            extra = present_conditions - expected_condition_set
            if missing or extra:
                raise ValueError(
                    f"incomplete paired cell {pairing_key}: "
                    f"missing={sorted(missing)} extra={sorted(extra)}"
                )
    for pairing_key, records in by_pairing.items():
        if require_matched_resources:
            fingerprints = {record.resource_budget_fingerprint for record in records}
            if "" in fingerprints:
                raise ValueError(
                    f"unfrozen resource budget in paired cell {pairing_key}"
                )
            if len(fingerprints) != 1:
                raise ValueError(
                    f"resource budgets differ across conditions in paired cell {pairing_key}"
                )
        if require_matched_ablations:
            ablation_fingerprints = {record.ablation_fingerprint for record in records}
            if len(ablation_fingerprints) != 1:
                raise ValueError(
                    f"mechanism ablations differ across conditions in paired cell {pairing_key}"
                )
        for field in _LINEAGE_FIELDS:
            values = {
                getattr(record, field)
                for record in records
                if getattr(record, field) is not None
            }
            if len(values) > 1:
                raise ValueError(
                    f"{field} differs across conditions in paired cell {pairing_key}"
                )
        replication_ids = {
            record.replication_id
            for record in records
            if record.replication_id is not None
        }
        if len(replication_ids) > 1:
            raise ValueError(
                f"replication_id differs across conditions in paired cell {pairing_key}"
            )
        orders = [
            record.randomization_order
            for record in records
            if record.randomization_order is not None
        ]
        if orders:
            if all(isinstance(order, int) for order in orders):
                if len(set(orders)) != len(orders):
                    raise ValueError(
                        f"duplicate randomization_order in paired cell {pairing_key}"
                    )
            elif all(isinstance(order, tuple) for order in orders):
                if len(set(orders)) != 1:
                    raise ValueError(
                        "randomization order schedules differ in paired cell "
                        f"{pairing_key}"
                    )
            else:
                raise ValueError(
                    "mixed randomization_order representations in paired cell "
                    f"{pairing_key}"
                )
    if expected_seeds_per_block is not None:
        if expected_seeds_per_block <= 0:
            raise ValueError("expected_seeds_per_block must be positive")
        seeds_by_block: dict[tuple[str, str, str, str], set[int]] = {}
        for observation in observations:
            key = (
                observation.pack,
                observation.provider,
                observation.model,
                observation.metric,
            )
            seeds_by_block.setdefault(key, set()).add(int(observation.seed))
        for block, seeds in seeds_by_block.items():
            if len(seeds) != expected_seeds_per_block:
                raise ValueError(
                    f"block {block} has {len(seeds)} seeds; "
                    f"expected {expected_seeds_per_block}"
                )


def build_paired_units(
    observations: Sequence[RunObservation],
    *,
    metric: str,
    treatment: str,
    control: str,
    require_complete_pairs: bool = True,
    experiment_phase: str | None = None,
) -> tuple[PairedUnit, ...]:
    treatment = normalize_condition(treatment)
    control = normalize_condition(control)
    selected = [
        observation
        for observation in observations
        if observation.metric == metric
        and (
            experiment_phase is None
            or observation.experiment_phase == experiment_phase
        )
    ]
    cells: dict[tuple[str, str, str, int, str, str], RunObservation] = {}
    for observation in selected:
        key = (
            observation.pack,
            observation.provider,
            observation.model,
            observation.seed,
            observation.condition,
            observation.metric,
        )
        if key in cells:
            raise ValueError(
                "duplicate condition contrast cell; filter by experiment_phase"
            )
        cells[key] = observation
    pairing_keys = sorted(
        {
            observation.pairing_key
            for observation in selected
        }
    )
    units: list[PairedUnit] = []
    for pack, provider, model, seed, metric_name in pairing_keys:
        treatment_key = (pack, provider, model, seed, treatment, metric_name)
        control_key = (pack, provider, model, seed, control, metric_name)
        if treatment_key not in cells or control_key not in cells:
            if not require_complete_pairs:
                continue
            raise ValueError(
                f"missing paired contrast {treatment}-{control} for "
                f"{pack}/{provider}/{model}/seed={seed}/{metric_name}"
            )
        treated = cells[treatment_key]
        baseline = cells[control_key]
        units.append(
            PairedUnit(
                pack=pack,
                provider=provider,
                model=model,
                seed=seed,
                metric=metric_name,
                treatment=treatment,
                control=control,
                treatment_value=float(treated.value),
                control_value=float(baseline.value),
            )
        )
    return tuple(units)


def build_paired_arm_units(
    observations: Sequence[RunObservation],
    *,
    metric: str,
    treatment: str,
    control: str,
    experiment_phase: str,
    require_complete_pairs: bool = True,
) -> tuple[PairedUnit, ...]:
    """Pair arbitrary registered arms without overloading organization condition."""

    selected = [
        observation
        for observation in observations
        if observation.metric == metric
        and observation.experiment_phase == experiment_phase
    ]
    cells: dict[tuple[str, str, str, int, str, str], RunObservation] = {}
    for observation in selected:
        arm = str(observation.arm_id or "")
        if not arm:
            raise ValueError("arm contrast requires arm_id on every run")
        key = (
            observation.pack,
            observation.provider,
            observation.model,
            observation.seed,
            arm,
            observation.metric,
        )
        if key in cells:
            raise ValueError(f"duplicate arm contrast cell:{key}")
        cells[key] = observation
    pairing_keys = sorted(
        {
            (
                observation.pack,
                observation.provider,
                observation.model,
                observation.seed,
                observation.metric,
            )
            for observation in selected
        }
    )
    units: list[PairedUnit] = []
    for pack, provider, model, seed, metric_name in pairing_keys:
        treatment_key = (
            pack,
            provider,
            model,
            seed,
            treatment,
            metric_name,
        )
        control_key = (
            pack,
            provider,
            model,
            seed,
            control,
            metric_name,
        )
        if treatment_key not in cells or control_key not in cells:
            if not require_complete_pairs:
                continue
            raise ValueError(
                f"missing paired arm contrast {treatment}-{control} for "
                f"{pack}/{provider}/{model}/seed={seed}/{metric_name}"
            )
        treated = cells[treatment_key]
        baseline = cells[control_key]
        if (
            treated.resource_budget_fingerprint
            != baseline.resource_budget_fingerprint
        ):
            raise ValueError("paired arm resource budgets differ")
        for field in _LINEAGE_FIELDS:
            if getattr(treated, field) != getattr(baseline, field):
                raise ValueError(f"paired arm lineage differs:{field}")
        units.append(
            PairedUnit(
                pack=pack,
                provider=provider,
                model=model,
                seed=seed,
                metric=metric_name,
                treatment=treatment,
                control=control,
                treatment_value=float(treated.value),
                control_value=float(baseline.value),
            )
        )
    return tuple(units)


def _block_weighted_mean(
    units: Sequence[PairedUnit], signs: Optional[Sequence[int]] = None
) -> float:
    if not units:
        raise ValueError("paired contrast has no units")
    by_block: dict[tuple[str, str, str], list[float]] = {}
    for index, unit in enumerate(units):
        sign = signs[index] if signs is not None else 1
        by_block.setdefault(unit.block_key, []).append(sign * unit.difference)
    block_means = [sum(values) / len(values) for values in by_block.values()]
    return sum(block_means) / len(block_means)


def paired_block_weighted_mean(units: Sequence[PairedUnit]) -> float:
    """Return the fixed-block mean used by intervals and sign-flip tests."""

    return _block_weighted_mean(units)


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot take quantile of empty values")
    if len(ordered) == 1:
        return ordered[0]
    position = max(0.0, min(1.0, probability)) * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def paired_block_bootstrap_interval(
    units: Sequence[PairedUnit],
    *,
    confidence_level: float = 0.95,
    samples: int = 10_000,
    seed: int = 1729,
) -> tuple[float, float]:
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between zero and one")
    if samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    by_block: dict[tuple[str, str, str], list[PairedUnit]] = {}
    for unit in units:
        by_block.setdefault(unit.block_key, []).append(unit)
    if not by_block:
        raise ValueError("paired contrast has no units")
    rng = Random(seed)
    draws: list[float] = []
    for _ in range(samples):
        block_means: list[float] = []
        for block_units in by_block.values():
            sampled = [
                block_units[rng.randrange(len(block_units))] for _ in block_units
            ]
            block_means.append(sum(unit.difference for unit in sampled) / len(sampled))
        draws.append(sum(block_means) / len(block_means))
    alpha = 1.0 - confidence_level
    return _quantile(draws, alpha / 2), _quantile(draws, 1 - alpha / 2)


def paired_sign_flip_test(
    units: Sequence[PairedUnit],
    *,
    max_exact_pairs: int = 20,
    monte_carlo_samples: int = 100_000,
    seed: int = 2718,
    alternative: str = "two_sided",
) -> tuple[float, str]:
    if not units:
        raise ValueError("paired contrast has no units")
    if alternative not in {"two_sided", "greater", "less"}:
        raise ValueError(
            "alternative must be two_sided, greater, or less"
        )
    observed = _block_weighted_mean(units)

    def is_extreme(candidate: float) -> bool:
        if alternative == "greater":
            return candidate >= observed - 1e-12
        if alternative == "less":
            return candidate <= observed + 1e-12
        return abs(candidate) >= abs(observed) - 1e-12

    n = len(units)
    extreme = 0
    if n <= max_exact_pairs:
        total = 2**n
        for signs in product((-1, 1), repeat=n):
            if is_extreme(_block_weighted_mean(units, signs)):
                extreme += 1
        suffix = "" if alternative == "two_sided" else f"_{alternative}"
        return extreme / total, f"exact_sign_flip{suffix}_{total}"
    if monte_carlo_samples <= 0:
        raise ValueError("monte_carlo_samples must be positive")
    rng = Random(seed)
    for _ in range(monte_carlo_samples):
        signs = tuple(1 if rng.random() >= 0.5 else -1 for _ in units)
        if is_extreme(_block_weighted_mean(units, signs)):
            extreme += 1
    suffix = "" if alternative == "two_sided" else f"_{alternative}"
    return (
        (extreme + 1) / (monte_carlo_samples + 1),
        "deterministic_sign_flip"
        f"{suffix}_monte_carlo_{monte_carlo_samples}",
    )


def minimum_attainable_two_sided_sign_flip_p(pair_count: int) -> float:
    """Smallest exact two-sided sign-flip p-value with ``pair_count`` pairs."""

    if isinstance(pair_count, bool) or not isinstance(pair_count, int):
        raise ValueError("pair_count must be an integer")
    if pair_count <= 0:
        raise ValueError("pair_count must be positive")
    return min(1.0, 2.0 / (2**pair_count))


def _single_report(
    units: Sequence[PairedUnit],
    observations: Sequence[RunObservation],
    *,
    scope: str,
    metric_family: str,
    bootstrap_samples: int,
    confidence_level: float,
    seed: int,
    max_exact_pairs: int,
    randomization_samples: int,
    selection_axis: str = "condition",
) -> PairedContrastReport:
    if selection_axis not in {"condition", "arm_id"}:
        raise ValueError("selection_axis must be condition or arm_id")
    estimate = _block_weighted_mean(units)
    low, high = paired_block_bootstrap_interval(
        units,
        confidence_level=confidence_level,
        samples=bootstrap_samples,
        seed=seed,
    )
    p_value, method = paired_sign_flip_test(
        units,
        max_exact_pairs=max_exact_pairs,
        monte_carlo_samples=randomization_samples,
        seed=seed + 1,
    )
    first = units[0]
    block_keys = {unit.block_key for unit in units}
    relevant_cells = {
        (unit.pack, unit.provider, unit.model, unit.seed, unit.metric) for unit in units
    }
    fingerprints = tuple(
        sorted(
            {
                observation.resource_budget_fingerprint
                for observation in observations
                if observation.pairing_key in relevant_cells
                and getattr(observation, selection_axis)
                in (first.treatment, first.control)
                and observation.resource_budget_fingerprint
            }
        )
    )
    relevant_observations = [
        observation
        for observation in observations
        if observation.pairing_key in relevant_cells
        and getattr(observation, selection_axis)
        in (first.treatment, first.control)
    ]
    caveats: list[str] = []
    if len(units) < 5:
        caveats.append("fewer_than_five_paired_seeds")
    if minimum_attainable_two_sided_sign_flip_p(len(units)) >= 0.05:
        caveats.append(
            "two_sided_exact_test_cannot_reach_alpha_0.05_with_pair_count"
        )
    if len(set(round(unit.difference, 12) for unit in units)) == 1:
        caveats.append("identical_paired_differences")
    if any(item.formal_claim_ready is False for item in relevant_observations):
        caveats.append("includes_non_formal_claim_ready_runs")
    if any(item.formal_release_ready is False for item in relevant_observations):
        caveats.append("includes_non_formal_release_ready_runs")
    if any(item.contamination_clearance is not True for item in relevant_observations):
        caveats.append("includes_runs_without_contamination_clearance")
    pack = first.pack if scope == "block" else None
    provider = first.provider if scope == "block" else None
    model = first.model if scope == "block" else None
    scope_id = (
        f"{first.pack}:{first.provider}:{first.model}"
        if scope == "block"
        else "all_blocks"
    )
    return PairedContrastReport(
        report_id=(
            f"{scope}:{scope_id}:{first.metric}:{first.treatment}_minus_{first.control}"
        ),
        scope=scope,
        pack=pack,
        provider=provider,
        model=model,
        metric=first.metric,
        metric_family=metric_family,
        treatment=first.treatment,
        control=first.control,
        pair_count=len(units),
        block_count=len(block_keys),
        mean_difference=estimate,
        confidence_low=low,
        confidence_high=high,
        confidence_level=confidence_level,
        bootstrap_samples=bootstrap_samples,
        p_value=p_value,
        adjusted_p_value=p_value,
        randomization_method=method,
        resource_budget_fingerprints=fingerprints,
        caveats=tuple(caveats),
    )


def holm_adjust(
    reports: Sequence[PairedContrastReport],
) -> tuple[PairedContrastReport, ...]:
    """Holm-adjust p-values within each declared metric-family/scope family."""

    by_family: dict[tuple[str, str], list[tuple[int, PairedContrastReport]]] = {}
    for index, report in enumerate(reports):
        by_family.setdefault((report.metric_family, report.scope), []).append(
            (index, report)
        )
    adjusted: dict[int, float] = {}
    for family_reports in by_family.values():
        ordered = sorted(family_reports, key=lambda item: item[1].p_value)
        running_max = 0.0
        count = len(ordered)
        for rank, (index, report) in enumerate(ordered):
            candidate = min(1.0, (count - rank) * report.p_value)
            running_max = max(running_max, candidate)
            adjusted[index] = running_max
    return tuple(
        replace(report, adjusted_p_value=adjusted[index])
        for index, report in enumerate(reports)
    )


def holm_adjust_global(
    reports: Sequence[PairedContrastReport],
) -> tuple[PairedContrastReport, ...]:
    """Holm-adjust one explicitly preregistered cross-metric family."""

    ordered = sorted(enumerate(reports), key=lambda item: item[1].p_value)
    adjusted: dict[int, float] = {}
    running_max = 0.0
    count = len(ordered)
    for rank, (index, report) in enumerate(ordered):
        candidate = min(1.0, (count - rank) * report.p_value)
        running_max = max(running_max, candidate)
        adjusted[index] = running_max
    return tuple(
        replace(report, adjusted_p_value=adjusted[index])
        for index, report in enumerate(reports)
    )


def analyze_primary_contrasts(
    observations: Sequence[RunObservation],
    *,
    metrics: Optional[Sequence[str]] = None,
    bootstrap_samples: int = 10_000,
    randomization_samples: int = 100_000,
    max_exact_pairs: int = 20,
    confidence_level: float = 0.95,
    seed: int = 314159,
    include_block_reports: bool = True,
    include_pooled_reports: bool = True,
    require_complete_conditions: bool = True,
    require_matched_resources: bool = True,
    require_matched_ablations: bool = True,
    require_complete_run_records: bool = True,
    require_contamination_clearance: bool = True,
    expected_conditions: Sequence[str] = CONDITIONS,
    contrasts: Sequence[tuple[str, str]] = PRIMARY_CONTRASTS,
    experiment_phase: str | None = None,
    expected_packs: Optional[Sequence[str]] = None,
    expected_seeds_per_block: Optional[int] = 5,
) -> tuple[PairedContrastReport, ...]:
    selected_observations = tuple(
        observation
        for observation in observations
        if (
            experiment_phase is None
            or observation.experiment_phase == experiment_phase
        )
    )
    validate_observations(
        selected_observations,
        require_complete_conditions=require_complete_conditions,
        require_matched_resources=require_matched_resources,
        require_matched_ablations=require_matched_ablations,
        require_complete_run_records=require_complete_run_records,
        require_contamination_clearance=require_contamination_clearance,
        expected_conditions=expected_conditions,
        expected_packs=expected_packs,
        expected_seeds_per_block=expected_seeds_per_block,
    )
    selected_metrics = tuple(
        metrics or sorted({item.metric for item in selected_observations})
    )
    family_by_metric = {
        item.metric: item.metric_family
        for item in selected_observations
        if item.metric in selected_metrics
    }
    selected_contrasts = tuple(contrasts)
    reports: list[PairedContrastReport] = []
    report_seed = seed
    for metric in selected_metrics:
        for treatment, control in selected_contrasts:
            contrast_family = family_by_metric[metric]
            units = build_paired_units(
                selected_observations,
                metric=metric,
                treatment=treatment,
                control=control,
                require_complete_pairs=require_complete_conditions,
                experiment_phase=experiment_phase,
            )
            if not units:
                continue
            if include_block_reports:
                by_block: dict[tuple[str, str, str], list[PairedUnit]] = {}
                for unit in units:
                    by_block.setdefault(unit.block_key, []).append(unit)
                for block_units in by_block.values():
                    reports.append(
                        _single_report(
                            block_units,
                            selected_observations,
                            scope="block",
                            metric_family=contrast_family,
                            bootstrap_samples=bootstrap_samples,
                            confidence_level=confidence_level,
                            seed=report_seed,
                            max_exact_pairs=max_exact_pairs,
                            randomization_samples=randomization_samples,
                        )
                    )
                    report_seed += 17
            if include_pooled_reports:
                reports.append(
                    _single_report(
                        units,
                        selected_observations,
                        scope="pooled",
                        metric_family=contrast_family,
                        bootstrap_samples=bootstrap_samples,
                        confidence_level=confidence_level,
                        seed=report_seed,
                        max_exact_pairs=max_exact_pairs,
                        randomization_samples=randomization_samples,
                    )
                )
                report_seed += 17
    return holm_adjust(reports)


def _validate_arm_observations(
    observations: Sequence[RunObservation],
    *,
    experiment_phase: str,
    expected_arms: Sequence[str],
    require_complete_arms: bool,
    require_complete_run_records: bool,
    require_contamination_clearance: bool,
    expected_packs: Optional[Sequence[str]],
    expected_seeds_per_block: Optional[int],
) -> tuple[RunObservation, ...]:
    selected = tuple(
        observation
        for observation in observations
        if observation.experiment_phase == experiment_phase
    )
    validate_observations(
        selected,
        require_complete_conditions=True,
        require_matched_resources=True,
        require_matched_ablations=False,
        require_complete_run_records=require_complete_run_records,
        require_contamination_clearance=require_contamination_clearance,
        require_v2_records=require_complete_run_records,
        expected_conditions=("B3",),
        expected_packs=expected_packs,
        expected_seeds_per_block=expected_seeds_per_block,
    )
    expected = set(expected_arms)
    if not expected or len(expected) != len(tuple(expected_arms)):
        raise ValueError("expected_arms must contain unique values")
    by_pairing: dict[
        tuple[str, str, str, int, str], set[str]
    ] = {}
    for observation in selected:
        if not observation.arm_id:
            raise ValueError("arm analysis requires arm_id")
        by_pairing.setdefault(observation.pairing_key, set()).add(
            observation.arm_id
        )
    for pairing_key, observed in by_pairing.items():
        if require_complete_arms and observed != expected:
            raise ValueError(
                f"incomplete arm cell {pairing_key}:"
                f"missing={sorted(expected - observed)} "
                f"extra={sorted(observed - expected)}"
            )
        if not observed <= expected:
            raise ValueError(
                f"unknown arm in cell {pairing_key}:"
                f"{sorted(observed - expected)}"
            )
    return selected


def analyze_arm_contrasts(
    observations: Sequence[RunObservation],
    *,
    experiment_phase: str,
    expected_arms: Sequence[str],
    contrasts: Sequence[tuple[str, str]],
    metrics: Optional[Sequence[str]] = None,
    bootstrap_samples: int = 10_000,
    randomization_samples: int = 100_000,
    max_exact_pairs: int = 20,
    confidence_level: float = 0.95,
    seed: int = 314159,
    include_block_reports: bool = True,
    include_pooled_reports: bool = True,
    require_complete_arms: bool = True,
    require_complete_run_records: bool = True,
    require_contamination_clearance: bool = True,
    expected_packs: Optional[Sequence[str]] = None,
    expected_seeds_per_block: Optional[int] = 5,
) -> tuple[PairedContrastReport, ...]:
    """Analyze preregistered mechanism arms without treating them as P5 clones."""

    selected = _validate_arm_observations(
        observations,
        experiment_phase=experiment_phase,
        expected_arms=expected_arms,
        require_complete_arms=require_complete_arms,
        require_complete_run_records=require_complete_run_records,
        require_contamination_clearance=require_contamination_clearance,
        expected_packs=expected_packs,
        expected_seeds_per_block=expected_seeds_per_block,
    )
    selected_metrics = tuple(metrics or sorted({item.metric for item in selected}))
    family_by_metric = {
        item.metric: item.metric_family
        for item in selected
        if item.metric in selected_metrics
    }
    reports: list[PairedContrastReport] = []
    report_seed = seed
    for metric in selected_metrics:
        for treatment, control in contrasts:
            units = build_paired_arm_units(
                selected,
                metric=metric,
                treatment=treatment,
                control=control,
                experiment_phase=experiment_phase,
                require_complete_pairs=require_complete_arms,
            )
            if not units:
                continue
            if include_block_reports:
                by_block: dict[tuple[str, str, str], list[PairedUnit]] = {}
                for unit in units:
                    by_block.setdefault(unit.block_key, []).append(unit)
                for block_units in by_block.values():
                    reports.append(
                        _single_report(
                            block_units,
                            selected,
                            scope="block",
                            metric_family=family_by_metric[metric],
                            bootstrap_samples=bootstrap_samples,
                            confidence_level=confidence_level,
                            seed=report_seed,
                            max_exact_pairs=max_exact_pairs,
                            randomization_samples=randomization_samples,
                            selection_axis="arm_id",
                        )
                    )
                    report_seed += 17
            if include_pooled_reports:
                reports.append(
                    _single_report(
                        units,
                        selected,
                        scope="pooled",
                        metric_family=family_by_metric[metric],
                        bootstrap_samples=bootstrap_samples,
                        confidence_level=confidence_level,
                        seed=report_seed,
                        max_exact_pairs=max_exact_pairs,
                        randomization_samples=randomization_samples,
                        selection_axis="arm_id",
                    )
                )
                report_seed += 17
    return holm_adjust(reports)


def reports_to_json(reports: Sequence[PairedContrastReport]) -> str:
    return json.dumps(
        {
            "schema_version": "orgenv_paired_contrasts_v1",
            "reports": [r.to_dict() for r in reports],
        },
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
    )


__all__ = [
    "CONDITIONS",
    "FROZEN_DESIGN_INFERENCE_SCOPE",
    "KNOWN_CONDITIONS",
    "LEGACY_CONDITIONS",
    "ORGANIZATION_CONDITIONS",
    "PRIMARY_CONTRASTS",
    "PairedContrastReport",
    "PairedUnit",
    "RunObservation",
    "analyze_arm_contrasts",
    "analyze_primary_contrasts",
    "build_paired_arm_units",
    "build_paired_units",
    "holm_adjust",
    "holm_adjust_global",
    "minimum_attainable_two_sided_sign_flip_p",
    "normalize_condition",
    "observations_from_payload",
    "paired_block_bootstrap_interval",
    "paired_block_weighted_mean",
    "paired_sign_flip_test",
    "reports_to_json",
    "validate_observations",
]
