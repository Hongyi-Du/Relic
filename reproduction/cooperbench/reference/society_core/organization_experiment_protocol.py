"""Machine-validated protocol for the organizational-capability experiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from environments.org_env.config.baseline_conditions import (
    organization_condition_ids,
    resolve_condition,
)
from environments.org_env.product.substrates.controls import NONE

from .hashing import stable_hash
from .organization_experiment_estimands import estimand_manifest
from .organization_experiment_power import (
    audit_preregistered_prospective_power,
    audit_preregistered_resolution,
)


PROTOCOL_SCHEMA_VERSION = "sociogenesis_organization_experiment_v4"
PROTOCOL_ID = "execution_grounded_oss_organization_v1"

# The paper evaluates ten real open-source repositories. The ids are declared
# rather than discovered so the matrix stays frozen, and the count is gated so a
# partially prepared corpus fails loudly instead of silently running a smaller
# experiment than the one that was preregistered.
PAPER_REPOSITORY_COUNT = 10
# The ten frozen Packs expanded in benchmarks/org_env_pack_expansion/windows.yaml.
# The order is the declared portfolio order and is load-bearing: transfer uses a
# cyclic shift over this tuple.
MAIN_REPOSITORIES = (
    "pydantic_v2140a1_to_main2607",
    "tenacity_v823_to_v914",
    "anyio_v4130_to_v4142",
    "boltons_v2400_to_v2610",
    "black_v2610_to_v2651",
    "urllib3_v250_to_v270",
    "cattrs_v2510_to_v2610",
    "aiohttp_v3140_to_v3143",
    "celery_v560_to_v563",
    "soupsieve_v26_to_v291",
)
SEEDS_PER_CONDITION = 3
MODEL_SLOTS = ("M1", "M2", "M3")
ANCHOR_MODEL_SLOT = "M1"
# Cross-repository transfer runs on one model so the arms differ only in the
# organizational state they inherit.
CANONICAL_MODEL_SLOT = "M1"

FULL_ORGANIZATION_CONDITION = "b3"

# Four post-formation arms. A capability formed in a source repository is
# carried into a target repository, and each arm is a COMPLETE target run from
# the frozen initial snapshot at episode one - the source checkpoint only
# exports the capability bundle, the original roster's member state, and the
# capability provenance. The target repository's own B3 run from the main sweep
# serves as the fresh-from-scratch reference, so no blank arm is needed.
ROSTER_ORIGIN_RETAINED = "retained_source_roster"
ROSTER_ORIGIN_FRESH = "fresh_roster"
CAPABILITY_FORM_EXECUTABLE = "executable"
CAPABILITY_FORM_REMOVED = "removed"
CAPABILITY_FORM_TEXT_ONLY = "content_matched_text_only"

TRANSFER_ARMS: dict[str, dict[str, str]] = {
    "R_Exec": {
        "roster_origin": ROSTER_ORIGIN_RETAINED,
        "capability_form": CAPABILITY_FORM_EXECUTABLE,
    },
    "R_Removed": {
        "roster_origin": ROSTER_ORIGIN_RETAINED,
        "capability_form": CAPABILITY_FORM_REMOVED,
    },
    "F_Exec": {
        "roster_origin": ROSTER_ORIGIN_FRESH,
        "capability_form": CAPABILITY_FORM_EXECUTABLE,
    },
    "F_Text": {
        "roster_origin": ROSTER_ORIGIN_FRESH,
        "capability_form": CAPABILITY_FORM_TEXT_ONLY,
    },
}

# Episodes at the start of a transfer run during which capability compilation is
# frozen. Without it F_Text can recompile its prose into rules and the reference
# can relearn an equivalent capability, washing out the inherited-capability
# effect the arms exist to measure.
TRANSFER_EVALUATION_WINDOW_EPISODES = 2
NO_EVALUATION_PERTURBATION = "none"

MAIN_SEEDS_BY_REPOSITORY = {
    repository_id: tuple(
        1101 + repository_index * 100 + offset
        for offset in range(SEEDS_PER_CONDITION)
    )
    for repository_index, repository_id in enumerate(MAIN_REPOSITORIES)
}
TRANSFER_SEEDS_BY_MAPPING_INDEX = {
    index: tuple(6101 + index * 100 + offset for offset in range(SEEDS_PER_CONDITION))
    for index in range(len(MAIN_REPOSITORIES))
}


def transfer_mappings() -> tuple[tuple[str, str], ...]:
    """One frozen source->target pair per repository.

    A cyclic shift is used so every repository is a source exactly once and a
    target exactly once, and nothing transfers to itself. That keeps the ten
    mappings balanced without picking favourites among the corpus.
    """
    count = len(MAIN_REPOSITORIES)
    if count < 2:
        return ()
    return tuple(
        (MAIN_REPOSITORIES[index], MAIN_REPOSITORIES[(index + 1) % count])
        for index in range(count)
    )

PAPER_CONDITIONS = tuple(
    resolve_condition(condition_id).short_name
    for condition_id in organization_condition_ids()
)

@dataclass(frozen=True)
class FrozenResourceBudget:
    budget_id: str = "paper_budget_v2"
    max_ticks: int = 336
    max_primary_actions: int = 2688
    max_llm_calls: int = 6000
    max_llm_requested_tokens: int = 20_000_000
    max_llm_prompt_characters: int = 100_000_000
    # 0 = uncapped, and that is the default. The other ceilings here are
    # properties of the experiment: the same trajectory consumes the same ticks,
    # actions, calls and tokens every time it is run. Wall clock is a property
    # of the machine and of the provider that day, so capping it made membership
    # in the dataset depend on gateway latency — a b0 case measured at 336 ticks
    # needs roughly eight hours at observed latency and a 6h ceiling truncated
    # it, while the identical trajectory would have finished on a faster
    # evening. A positive value is still honoured as an ops guard against a
    # wedged process; it is simply not something the design should fix.
    max_case_wall_clock_seconds: float = 0.0
    # How many cases one job may run at once. Serial by default because
    # concurrency is not free of the measurement: parallel cases contend for the
    # LLM endpoint and the Docker daemon, and max_case_wall_clock_seconds is
    # measured in wall clock, so a case can time out purely because siblings
    # were running. Raising it is a legitimate choice for a 220-cell sweep -
    # fully serial is roughly two weeks - but it is an execution CONDITION, so
    # it belongs in the frozen budget and its fingerprint rather than being
    # hardcoded at the call site where no record of it survives.
    max_parallel_cases: int = 1

    def __post_init__(self) -> None:
        if not self.budget_id.strip():
            raise ValueError("resource_budget_id_required")
        for name in (
            "max_ticks",
            "max_primary_actions",
            "max_llm_calls",
            "max_llm_requested_tokens",
            "max_llm_prompt_characters",
            "max_parallel_cases",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"resource_budget_limit_invalid:{name}")
        if self.max_case_wall_clock_seconds < 0:
            raise ValueError(
                "resource_budget_limit_invalid:max_case_wall_clock_seconds"
            )


@dataclass(frozen=True)
class ExperimentCell:
    cell_id: str
    phase: str
    repository_id: str
    model_slot: str
    condition: str
    arm_id: str
    seed: int
    paired_block: str
    execution_kind: str = "agent_execution"
    disabled_mechanisms: tuple[str, ...] = ()
    oss_control: str = NONE
    evaluation_perturbation: str = NO_EVALUATION_PERTURBATION
    # Derived, never a second literal: the cell names the budget it ran under
    # and FrozenResourceBudget IS that budget. Written out separately, the two
    # drifted the moment the wall-clock cap was lifted — every cell claimed
    # paper_budget_v1 while the budget itself had become paper_budget_v2, so a
    # reader joining runs by budget identity would have seen a version that no
    # longer existed. The execution fingerprint is hashed from the budget object
    # and was always right, which is exactly why the label could rot unnoticed.
    resource_budget_id: str = FrozenResourceBudget().budget_id
    tool_surface_id: str = "org_env_tools_v1"
    information_budget_id: str = "time_machine_visible_v1"
    external_signal_panel_id: str = ""
    future_vault_access: str = "evaluator_only"
    agent_visible_future_cards: bool = False
    # Transfer-only. A transfer cell is a complete target-repository run that is
    # seeded at initialization with organizational state exported from a source
    # repository's B3 formation run.
    source_repository_id: str = ""
    roster_origin: str = ""
    capability_form: str = ""
    reference_cell_id: str = ""
    source_formation_cell_id: str = ""
    capability_compilation_frozen_episodes: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProtocolAudit:
    valid: bool
    cell_count: int
    agent_execution_count: int
    phase_counts: dict[str, int]
    protocol_hash: str
    blocking_reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _cell(
    *,
    phase: str,
    repository_id: str,
    model_slot: str,
    condition: str,
    arm_id: str,
    seed: int,
    execution_kind: str = "agent_execution",
    disabled_mechanisms: Sequence[str] = (),
    oss_control: str = NONE,
    evaluation_perturbation: str = NO_EVALUATION_PERTURBATION,
    source_repository_id: str = "",
    roster_origin: str = "",
    capability_form: str = "",
    reference_cell_id: str = "",
    source_formation_cell_id: str = "",
    capability_compilation_frozen_episodes: int = 0,
) -> ExperimentCell:
    paired_block = f"phase={phase}|repo={repository_id}|model={model_slot}|seed={seed}"
    cell_id = f"{phase}__{repository_id}__{model_slot}__{condition}__{arm_id}__s{seed}"
    return ExperimentCell(
        cell_id=cell_id,
        phase=phase,
        repository_id=repository_id,
        model_slot=model_slot,
        condition=condition,
        arm_id=arm_id,
        seed=int(seed),
        paired_block=paired_block,
        execution_kind=execution_kind,
        disabled_mechanisms=tuple(sorted(disabled_mechanisms)),
        oss_control=oss_control,
        evaluation_perturbation=evaluation_perturbation,
        external_signal_panel_id=f"{repository_id}:seed={seed}:frozen",
        source_repository_id=source_repository_id,
        roster_origin=roster_origin,
        capability_form=capability_form,
        reference_cell_id=reference_cell_id,
        source_formation_cell_id=source_formation_cell_id,
        capability_compilation_frozen_episodes=int(
            capability_compilation_frozen_episodes
        ),
    )


def build_experiment_matrix() -> tuple[ExperimentCell, ...]:
    """The frozen matrix: the organization ladder, controls, and transfer arms.

    The main sweep is every repository x every model x every seed x the four
    organization conditions. Transfer adds one frozen source->target mapping per
    repository; each arm is a complete target run, and the target's own B3 cell
    on the canonical model is referenced as the fresh-from-scratch comparison
    rather than duplicated as a blank arm.
    """
    cells: list[ExperimentCell] = []
    for repository_id in MAIN_REPOSITORIES:
        for model_slot in MODEL_SLOTS:
            for seed in MAIN_SEEDS_BY_REPOSITORY[repository_id]:
                for condition in PAPER_CONDITIONS:
                    cells.append(
                        _cell(
                            phase="main_organization_sweep",
                            repository_id=repository_id,
                            model_slot=model_slot,
                            condition=condition,
                            arm_id=condition,
                            seed=seed,
                        )
                    )
    for index, (source_id, target_id) in enumerate(transfer_mappings()):
        for replicate, seed in enumerate(TRANSFER_SEEDS_BY_MAPPING_INDEX[index]):
            # Replicate k of a mapping pairs with replicate k of both repositories'
            # main runs: the source supplies the capability bundle, the target
            # supplies the fresh-from-scratch reference.
            reference = _cell(
                phase="main_organization_sweep",
                repository_id=target_id,
                model_slot=CANONICAL_MODEL_SLOT,
                condition=FULL_ORGANIZATION_CONDITION,
                arm_id=FULL_ORGANIZATION_CONDITION,
                seed=MAIN_SEEDS_BY_REPOSITORY[target_id][replicate],
            ).cell_id
            source_formation = _cell(
                phase="main_organization_sweep",
                repository_id=source_id,
                model_slot=CANONICAL_MODEL_SLOT,
                condition=FULL_ORGANIZATION_CONDITION,
                arm_id=FULL_ORGANIZATION_CONDITION,
                seed=MAIN_SEEDS_BY_REPOSITORY[source_id][replicate],
            ).cell_id
            for arm_id, arm in TRANSFER_ARMS.items():
                cells.append(
                    _cell(
                        phase="capability_transfer",
                        repository_id=target_id,
                        model_slot=CANONICAL_MODEL_SLOT,
                        condition=FULL_ORGANIZATION_CONDITION,
                        arm_id=arm_id,
                        seed=seed,
                        source_repository_id=source_id,
                        roster_origin=arm["roster_origin"],
                        capability_form=arm["capability_form"],
                        reference_cell_id=reference,
                        source_formation_cell_id=source_formation,
                        capability_compilation_frozen_episodes=(
                            TRANSFER_EVALUATION_WINDOW_EPISODES
                        ),
                    )
                )
    return tuple(cells)


def expected_phase_counts() -> dict[str, int]:
    return {
        "main_organization_sweep": (
            len(MAIN_REPOSITORIES)
            * len(MODEL_SLOTS)
            * SEEDS_PER_CONDITION
            * len(PAPER_CONDITIONS)
        ),
        "capability_transfer": (
            len(transfer_mappings()) * SEEDS_PER_CONDITION * len(TRANSFER_ARMS)
        ),
    }


def validate_experiment_matrix(
    cells: Sequence[ExperimentCell],
    *,
    model_bindings: Mapping[str, str] | None = None,
    require_model_bindings: bool = False,
) -> ProtocolAudit:
    reasons: list[str] = []
    if len(MAIN_REPOSITORIES) != PAPER_REPOSITORY_COUNT:
        reasons.append(
            "repository_corpus_incomplete:"
            f"declared={len(MAIN_REPOSITORIES)}:required={PAPER_REPOSITORY_COUNT}"
        )
    expected_total = sum(expected_phase_counts().values())
    if len(cells) != expected_total:
        reasons.append(f"expected_{expected_total}_cells:observed={len(cells)}")
    if len({cell.cell_id for cell in cells}) != len(cells):
        reasons.append("duplicate_cell_id")
    canonical = build_experiment_matrix()
    canonical_by_id = {cell.cell_id: cell.to_dict() for cell in canonical}
    observed_by_id = {cell.cell_id: cell.to_dict() for cell in cells}
    if set(observed_by_id) != set(canonical_by_id):
        missing = sorted(set(canonical_by_id) - set(observed_by_id))
        extra = sorted(set(observed_by_id) - set(canonical_by_id))
        reasons.append(
            f"preregistered_cell_identity_mismatch:missing={missing}:extra={extra}"
        )
    else:
        mutated = sorted(
            cell_id
            for cell_id in canonical_by_id
            if observed_by_id[cell_id] != canonical_by_id[cell_id]
        )
        if mutated:
            reasons.append(
                "preregistered_cell_definition_mismatch:" + ",".join(mutated)
            )
    expected_counts = expected_phase_counts()
    phase_counts = {
        phase: sum(cell.phase == phase for cell in cells) for phase in expected_counts
    }
    if phase_counts != expected_counts:
        reasons.append(f"phase_count_mismatch:{phase_counts}")
    unknown_phases = {cell.phase for cell in cells} - set(expected_counts)
    if unknown_phases:
        reasons.append(f"unknown_phase:{sorted(unknown_phases)}")

    by_block: dict[str, list[ExperimentCell]] = {}
    for cell in cells:
        by_block.setdefault(cell.paired_block, []).append(cell)
        if cell.future_vault_access != "evaluator_only":
            reasons.append(f"future_vault_exposed:{cell.cell_id}")
        if cell.agent_visible_future_cards:
            reasons.append(f"future_cards_exposed:{cell.cell_id}")
        if cell.evaluation_perturbation != NO_EVALUATION_PERTURBATION:
            reasons.append(f"unknown_evaluation_perturbation:{cell.cell_id}")
        # Roster size and the decision kernel are treatment factors on this
        # ladder, not confounds to be excluded: B0 is a single agent and the
        # lower rungs deliberately use a traditional LLM action selector. What
        # must hold is that the cell names a real condition.
        resolve_condition(cell.condition)
    for block, block_cells in by_block.items():
        for field in (
            "resource_budget_id",
            "tool_surface_id",
            "information_budget_id",
            "external_signal_panel_id",
            "future_vault_access",
        ):
            if len({getattr(cell, field) for cell in block_cells}) != 1:
                reasons.append(f"paired_block_mismatch:{block}:{field}")

    main_arms = {
        cell.condition for cell in cells if cell.phase == "main_organization_sweep"
    }
    if main_arms != set(PAPER_CONDITIONS):
        reasons.append(f"main_condition_mismatch:{sorted(main_arms)}")
    transfer_cells = [cell for cell in cells if cell.phase == "capability_transfer"]
    main_cell_ids = {
        cell.cell_id for cell in cells if cell.phase == "main_organization_sweep"
    }
    if transfer_cells:
        arms = {cell.arm_id for cell in transfer_cells}
        if arms != set(TRANSFER_ARMS):
            reasons.append(f"transfer_arm_mismatch:{sorted(arms)}")
        for cell in transfer_cells:
            if not cell.source_repository_id:
                reasons.append(f"transfer_source_missing:{cell.cell_id}")
            elif cell.source_repository_id == cell.repository_id:
                reasons.append(f"transfer_source_equals_target:{cell.cell_id}")
            if cell.reference_cell_id not in main_cell_ids:
                reasons.append(f"transfer_reference_unresolved:{cell.cell_id}")
            if cell.source_formation_cell_id not in main_cell_ids:
                reasons.append(f"transfer_source_formation_unresolved:{cell.cell_id}")
            elif cell.source_repository_id not in cell.source_formation_cell_id:
                reasons.append(f"transfer_source_formation_mismatch:{cell.cell_id}")
            if cell.model_slot != CANONICAL_MODEL_SLOT:
                reasons.append(f"transfer_model_not_canonical:{cell.cell_id}")
            if cell.capability_compilation_frozen_episodes != (
                TRANSFER_EVALUATION_WINDOW_EPISODES
            ):
                reasons.append(f"transfer_evaluation_window_mismatch:{cell.cell_id}")
            expected = TRANSFER_ARMS.get(cell.arm_id)
            if expected is not None and (
                cell.roster_origin != expected["roster_origin"]
                or cell.capability_form != expected["capability_form"]
            ):
                reasons.append(f"transfer_arm_definition_mismatch:{cell.cell_id}")
        # Every arm of a mapping-seed must sit in the same paired block, or the
        # contrasts are not paired on the target run they share.
        by_mapping: dict[tuple[str, str, int], set[str]] = {}
        for cell in transfer_cells:
            key = (cell.source_repository_id, cell.repository_id, cell.seed)
            by_mapping.setdefault(key, set()).add(cell.arm_id)
        incomplete = sorted(
            f"{source}->{target}:s{seed}"
            for (source, target, seed), present in by_mapping.items()
            if present != set(TRANSFER_ARMS)
        )
        if incomplete:
            reasons.append(f"transfer_mapping_incomplete:{','.join(incomplete)}")
        sources = [source for source, _ in transfer_mappings()]
        targets = [target for _, target in transfer_mappings()]
        if len(set(sources)) != len(sources) or len(set(targets)) != len(targets):
            reasons.append("transfer_mapping_not_balanced")

    non_agent = [
        cell.cell_id for cell in cells if cell.execution_kind != "agent_execution"
    ]
    if non_agent:
        reasons.append("matrix_must_contain_only_complete_runs:" + ",".join(non_agent))

    bindings = dict(model_bindings or {})
    if require_model_bindings:
        missing = [
            slot for slot in MODEL_SLOTS if not str(bindings.get(slot, "")).strip()
        ]
        if missing:
            reasons.append(f"unresolved_model_slots:{','.join(missing)}")
        resolved_models = [
            str(bindings[slot]).strip()
            for slot in MODEL_SLOTS
            if str(bindings.get(slot, "")).strip()
        ]
        if len(resolved_models) == len(MODEL_SLOTS) and len(
            set(resolved_models)
        ) != len(resolved_models):
            reasons.append("model_family_slots_must_be_distinct")

    payload = {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "estimands": estimand_manifest(),
        "cells": [cell.to_dict() for cell in cells],
    }
    return ProtocolAudit(
        valid=not reasons,
        cell_count=len(cells),
        agent_execution_count=sum(
            cell.execution_kind == "agent_execution" for cell in cells
        ),
        phase_counts=phase_counts,
        protocol_hash=stable_hash(payload),
        blocking_reasons=tuple(dict.fromkeys(reasons)),
    )


def protocol_manifest(
    *,
    model_bindings: Mapping[str, str] | None = None,
    require_model_bindings: bool = False,
) -> dict[str, Any]:
    cells = build_experiment_matrix()
    audit = validate_experiment_matrix(
        cells,
        model_bindings=model_bindings,
        require_model_bindings=require_model_bindings,
    )
    budget = FrozenResourceBudget()
    resolution_audit = audit_preregistered_resolution()
    prospective_power_audit = audit_preregistered_prospective_power()
    return {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "central_estimand": (
            "paired difference in product and event-grounded organizational "
            "capability outcomes under a fixed repo, model slot, seed, tool "
            "surface, information budget, external panel, and resource budget"
        ),
        "claim_boundary": {
            "hidden_test_secrecy": (
                "agent-private during development and immutable during "
                "evaluation; not adversarially unreadable to candidate code "
                "executed by white-box tests"
            ),
            "run_budget": (
                "the frozen matrix is exactly the preregistered complete runs: "
                "the organization ladder on every repository and model, plus "
                "the post-formation transfer arms; nothing else executes"
            ),
            "prompt_visibility": (
                "every provider prompt is checked before dispatch against "
                "evaluator-private identifiers, paths, and high-specificity "
                "text; only hashes, counts, chain state, and denial codes "
                "enter run records"
            ),
            "inferential_resolution": (
                "confirmatory claims use preregistered pooled families with "
                "exact-test resolution"
            ),
            "statistical_population": (
                "all estimates are conditional on the frozen repositories, "
                "model bindings, and seed schedules enumerated here; no "
                "repository-population or model-population inference is made"
            ),
            "product_endpoint": (
                "the primary product composite averages hidden pass rate and "
                "causal fix rate; thresholded release readiness is excluded "
                "from that endpoint and retained only as a secondary "
                "non-independent sensitivity diagnostic"
            ),
            "prospective_power": (
                "sensitivity values are deterministic planning consequences "
                "under explicit assumptions, not observed power, estimated "
                "effects, scientific results, or changes to inference gates"
            ),
            "model_training_contamination": (
                "current models may have latent exposure to the frozen "
                "repository snapshots; every model-by-repository pair carries "
                "a preflight contamination assessment"
            ),
        },
        "repositories": {
            "declared": list(MAIN_REPOSITORIES),
            "required_count": PAPER_REPOSITORY_COUNT,
        },
        "model_slots": list(MODEL_SLOTS),
        "model_bindings": dict(model_bindings or {}),
        "seed_schedule": {
            "main": {
                repository_id: list(seeds)
                for repository_id, seeds in MAIN_SEEDS_BY_REPOSITORY.items()
            },
            "transfer": {
                str(index): list(seeds)
                for index, seeds in TRANSFER_SEEDS_BY_MAPPING_INDEX.items()
            },
        },
        "estimands": estimand_manifest(),
        "resource_budget": asdict(budget),
        "formal_run_prerequisites": (
            "distinct frozen model-family bindings",
            "content-addressed repository preparation",
            "model-by-repository contamination probes",
            "agent-visible snapshot receipt",
            "pre-dispatch prompt-visibility audit",
            "read-only hidden evaluator execution",
            "pre-registered exclusion and retry rules",
        ),
        "transfer_arms": list(TRANSFER_ARMS),
        "transfer_mappings": [
            {"source_repository_id": source, "target_repository_id": target}
            for source, target in transfer_mappings()
        ],
        "inferential_resolution_audit": asdict(resolution_audit),
        "prospective_sensitivity_power_audit": asdict(prospective_power_audit),
        "audit": audit.to_dict(),
        "cells": [cell.to_dict() for cell in cells],
    }


__all__ = [
    "ANCHOR_MODEL_SLOT",
    "CANONICAL_MODEL_SLOT",
    "ExperimentCell",
    "FrozenResourceBudget",
    "MAIN_REPOSITORIES",
    "MAIN_SEEDS_BY_REPOSITORY",
    "MODEL_SLOTS",
    "NO_EVALUATION_PERTURBATION",
    "PAPER_CONDITIONS",
    "PAPER_REPOSITORY_COUNT",
    "PROTOCOL_ID",
    "PROTOCOL_SCHEMA_VERSION",
    "ProtocolAudit",
    "SEEDS_PER_CONDITION",
    "TRANSFER_ARMS",
    "TRANSFER_EVALUATION_WINDOW_EPISODES",
    "TRANSFER_SEEDS_BY_MAPPING_INDEX",
    "build_experiment_matrix",
    "expected_phase_counts",
    "protocol_manifest",
    "transfer_mappings",
    "validate_experiment_matrix",
]
