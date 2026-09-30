"""Parameter, bounded-score, and claim-audit registries."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .hashing import stable_hash


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    symbol: str
    module: str
    meaning: str
    type: str
    unit_or_scale: str
    default: Any
    allowed_range: tuple[float, float] | None
    source_category: str
    source_note: str
    used_in_main_result: bool
    sensitivity_values: tuple[Any, ...] = ()
    ablation_value: Any | None = None
    affects: tuple[str, ...] = ()
    forbidden_interpretations: tuple[str, ...] = ()


@dataclass
class ParameterRegistry:
    specs: dict[str, ParameterSpec] = field(default_factory=dict)
    frozen: bool = False

    def register(self, spec: ParameterSpec) -> None:
        if self.frozen:
            raise ValueError("Parameter registry is frozen")
        self.specs[spec.name] = spec

    def require(self, name: str) -> ParameterSpec:
        if name not in self.specs:
            raise KeyError(f"Parameter not registered: {name}")
        return self.specs[name]

    def freeze(self) -> str:
        self.frozen = True
        return self.hash()

    def hash(self) -> str:
        return stable_hash(self.specs)


@dataclass(frozen=True)
class BoundedScoreSpec:
    component_name: str
    raw_event_predicates: tuple[str, ...]
    numerator_query: str
    denominator_query_or_exposure_set: str
    normalization_family: str
    normalization_parameters: dict[str, float]
    sample_size_minimum: int
    missing_data_behavior: str
    uncertainty_method: str


class BoundedScoreRegistry:
    allowed_families = {
        "saturating_count",
        "logistic_threshold",
        "clipped_ratio",
        "rank_percentile",
    }

    def __init__(self) -> None:
        self.specs: dict[str, BoundedScoreSpec] = {}
        self.frozen = False

    def register(self, spec: BoundedScoreSpec) -> None:
        if self.frozen:
            raise ValueError("Bounded-score registry is frozen")
        if spec.normalization_family not in self.allowed_families:
            raise ValueError(f"Unsupported normalization family: {spec.normalization_family}")
        self.specs[spec.component_name] = spec

    def require(self, component_name: str) -> BoundedScoreSpec:
        if component_name not in self.specs:
            raise KeyError(f"Bounded score not registered: {component_name}")
        return self.specs[component_name]

    def freeze(self) -> str:
        self.frozen = True
        return self.hash()

    def hash(self) -> str:
        return stable_hash(self.specs)


@dataclass(frozen=True)
class ClaimRecord:
    claim_id: str
    statement: str
    claim_level: int
    evidence_level: int
    supporting_artifacts: tuple[str, ...]
    evidence_gate: str = ""
    uncertainty_method: str = "seed_level_interval"
    limitations: tuple[str, ...] = ()


class ClaimAudit:
    def __init__(self) -> None:
        self.records: dict[str, ClaimRecord] = {}

    def add(self, record: ClaimRecord) -> None:
        if record.claim_level > record.evidence_level:
            raise ValueError(
                f"Claim {record.claim_id} exceeds evidence: "
                f"L{record.claim_level} > L{record.evidence_level}"
            )
        self.records[record.claim_id] = record

    def hash(self) -> str:
        return stable_hash(self.records)


def default_parameter_registry() -> ParameterRegistry:
    registry = ParameterRegistry()
    registry.register(
        ParameterSpec(
            name="body_hunger_drift",
            symbol="δ_hunger",
            module="body",
            meaning="per-tick hunger increase without eating",
            type="float",
            unit_or_scale="[0,1] per tick",
            default=0.03,
            allowed_range=(0.0, 0.2),
            source_category="fixed_by_design",
            source_note="MVR default, swept later",
            used_in_main_result=True,
            sensitivity_values=(0.01, 0.03, 0.06),
            ablation_value=0.0,
            affects=("hunger", "eat_probability"),
        )
    )
    registry.register(
        ParameterSpec(
            name="attention_read_cost",
            symbol="c_read",
            module="action_budget",
            meaning="attention cost of reading/inspection actions",
            type="float",
            unit_or_scale="[0,1] per action",
            default=0.04,
            allowed_range=(0.0, 0.3),
            source_category="fixed_by_design",
            source_note="MVR default, swept later",
            used_in_main_result=True,
            sensitivity_values=(0.02, 0.04, 0.08),
            ablation_value=0.0,
            affects=("attention_budget",),
        )
    )
    return registry


def default_bounded_score_registry() -> BoundedScoreRegistry:
    registry = BoundedScoreRegistry()
    registry.register(
        BoundedScoreSpec(
            component_name="attention_cluster_unique_agents",
            raw_event_predicates=("Posted", "Replied", "Tried"),
            numerator_query="count_distinct_agents_engaged(topic, window)",
            denominator_query_or_exposure_set="active_agents_in_window",
            normalization_family="clipped_ratio",
            normalization_parameters={},
            sample_size_minimum=5,
            missing_data_behavior="return_zero_with_high_uncertainty",
            uncertainty_method="seed_level_interval",
        )
    )
    registry.register(
        BoundedScoreSpec(
            component_name="artifact_independent_trial",
            raw_event_predicates=("Tried",),
            numerator_query="count_independent_trials(artifact, window)",
            denominator_query_or_exposure_set="agents_with_access_and_task_opportunity",
            normalization_family="clipped_ratio",
            normalization_parameters={},
            sample_size_minimum=5,
            missing_data_behavior="return_zero_with_high_uncertainty",
            uncertainty_method="seed_level_interval",
        )
    )
    return registry


def default_claim_audit() -> ClaimAudit:
    audit = ClaimAudit()
    audit.add(
        ClaimRecord(
            claim_id="typed_preinstitutional_runtime",
            statement=(
                "Society-Core implements a typed pre-institutional external society "
                "runtime with primitive actions and rejected macro actions."
            ),
            claim_level=1,
            evidence_level=1,
            supporting_artifacts=(
                "tests/society_core/test_no_initial_institutions.py",
                "tests/society_core/test_no_macro_actions.py",
                "tests/society_core/test_seed_replay_exact.py",
            ),
            evidence_gate="unit_tests_pass",
            limitations=("not an empirical validity claim",),
        )
    )
    audit.add(
        ClaimRecord(
            claim_id="artifact_neutral_release_path",
            statement=(
                "Company and non-company artifacts enter the society through the same "
                "artifact release and trial path without product-specific adoption actions."
            ),
            claim_level=1,
            evidence_level=1,
            supporting_artifacts=("tests/society_core/test_artifact_neutrality.py",),
            evidence_gate="artifact_path_equivalence_test_passes",
            limitations=("diffusion is detector output, not a runtime goal",),
        )
    )
    audit.add(
        ClaimRecord(
            claim_id="detectors_read_only",
            statement=(
                "Detector outputs are post-hoc summaries and do not feed back into "
                "runtime transitions, observations, or feed ranking."
            ),
            claim_level=1,
            evidence_level=1,
            supporting_artifacts=("tests/society_core/test_detector_non_interference.py",),
            evidence_gate="detector_enabled_disabled_event_hash_equality",
            limitations=("detector construct validity is audited separately",),
        )
    )
    audit.add(
        ClaimRecord(
            claim_id="seed_level_uncertainty_primary",
            statement=(
                "Main uncertainty reports aggregate independent seed or perturbation "
                "runs rather than naive event-level bootstrap."
            ),
            claim_level=1,
            evidence_level=1,
            supporting_artifacts=("society_core/experiments.py",),
            evidence_gate="uncertainty_report_primary_unit_is_seed",
            limitations=("intervals are internal simulation uncertainty only",),
        )
    )
    return audit
