"""Frozen estimands and multiplicity family for the organization experiment."""

from __future__ import annotations

from typing import Any


ESTIMAND_SCHEMA_VERSION = "organization_experiment_estimands_v4"

# Design constants shared by the protocol and the planning-power audit. They
# live here so the two cannot drift.
PREREGISTERED_REPOSITORY_COUNT = 10
PREREGISTERED_MODEL_COUNT = 3
SEEDS_PER_CONDITION = 3
# A main paired block is one (repository, model, seed) and carries the whole
# organization ladder: 10 x 3 x 3 = 90 blocks of four conditions = 360 runs.
MAIN_PAIRED_BLOCK_COUNT = (
    PREREGISTERED_REPOSITORY_COUNT * PREREGISTERED_MODEL_COUNT * SEEDS_PER_CONDITION
)
# A transfer paired block is one (source->target mapping, seed) and carries the
# four post-formation arms: 10 x 3 = 30 blocks of four arms = 120 runs.
TRANSFER_PAIRED_BLOCK_COUNT = PREREGISTERED_REPOSITORY_COUNT * SEEDS_PER_CONDITION
GLOBAL_CONFIRMATORY_FAMILY_ID = "global_confirmatory_primary_v1"
GLOBAL_CONFIRMATORY_ALPHA = 0.05

PRIMARY_METRIC_SPECS: dict[str, dict[str, Any]] = {
    "product_outcome_composite": {
        "family": "product_effectiveness",
        "direction": "higher_is_better",
        "range": [0.0, 1.0],
        "aggregation": "equal_weight_arithmetic_mean",
        "components": [
            "metrics.oss_hidden_pass_rate",
            "final_evaluation.causal_fix_rate",
        ],
        "interpretation": (
            "frozen acceptance-oracle performance and issue-level causal "
            "repair; the thresholded release-ready flag is excluded to avoid "
            "counting the same evaluator outcome twice"
        ),
    },
    "organizational_capability_composite": {
        "family": "organizational_capability",
        "direction": "higher_is_better",
        "range": [0.0, 1.0],
        "aggregation": "equal_weight_arithmetic_mean",
        "components": [
            "metrics.strong_protocol_emergence_rate",
            "metrics.cross_context_protocol_reuse_rate",
            "metrics.protocol_persistence_rate",
            "metrics.measurable_impact",
        ],
        "interpretation": (
            "event-grounded emergence, reuse, persistence, and measurable "
            "organizational impact"
        ),
    },
}

SECONDARY_SENSITIVITY_METRIC_SPECS: dict[str, dict[str, Any]] = {
    "product_outcome_with_release_readiness_sensitivity": {
        "family": "product_effectiveness_sensitivity",
        "direction": "higher_is_better",
        "range": [0.0, 1.0],
        "aggregation": "equal_weight_arithmetic_mean",
        "components": [
            "metrics.oss_hidden_pass_rate",
            "final_evaluation.causal_fix_rate",
            "final_evaluation.formal_release_ready",
        ],
        "interpretation": (
            "secondary non-confirmatory sensitivity endpoint retaining the "
            "thresholded release decision; it is not independent evidence and "
            "cannot replace the primary product endpoint"
        ),
    },
}

# The three ladder steps: temporary coordination, organizational persistence,
# and capability formation. B3-B2 is a bundled step (kernel, profile
# conditioning and capability lifecycle move together) and is reported as such.
PRIMARY_CONTRASTS_BY_PHASE: dict[str, tuple[tuple[str, str], ...]] = {
    "main_organization_sweep": (
        ("B1", "B0"),
        ("B2", "B1"),
        ("B3", "B2"),
    ),
}

SECONDARY_CONTRASTS_BY_PHASE: dict[str, tuple[tuple[str, str], ...]] = {}

# The reference arm is not a transfer run: it is the target repository's own B3
# cell from the main sweep, reached through each transfer cell's
# reference_cell_id. Naming it here keeps the contrast table honest about the
# fact that two of the five comparisons cross phases.
TRANSFER_REFERENCE_ARM = "B3_target_reference"

TRANSFER_CONTRASTS: tuple[tuple[str, str], ...] = (
    ("R_Exec", "R_Removed"),
    ("F_Exec", TRANSFER_REFERENCE_ARM),
    ("R_Exec", "F_Exec"),
    ("F_Exec", "F_Text"),
    ("F_Text", TRANSFER_REFERENCE_ARM),
)

TRANSFER_CONTRAST_QUESTIONS: dict[tuple[str, str], str] = {
    ("R_Exec", "R_Removed"): (
        "does the learned capability itself carry causal value, holding members, "
        "member memory and target repository fixed"
    ),
    ("F_Exec", TRANSFER_REFERENCE_ARM): (
        "does the capability survive its original members - the core "
        "cross-member, cross-repository transfer claim"
    ),
    ("R_Exec", "F_Exec"): (
        "how much of the benefit is carried by the original members rather than "
        "by the organizational state"
    ),
    ("F_Exec", "F_Text"): (
        "does the executable form matter, holding capability content constant"
    ),
    ("F_Text", TRANSFER_REFERENCE_ARM): (
        "does the organizational knowledge help as text alone, without entering "
        "the runtime"
    ),
}

GLOBAL_CONFIRMATORY_HYPOTHESIS_COUNT = len(PRIMARY_METRIC_SPECS) * sum(
    len(contrasts) for contrasts in PRIMARY_CONTRASTS_BY_PHASE.values()
)


def estimand_manifest() -> dict[str, Any]:
    return {
        "schema_version": ESTIMAND_SCHEMA_VERSION,
        "primary_metrics": PRIMARY_METRIC_SPECS,
        "primary_contrasts_by_phase": {
            phase: [list(contrast) for contrast in contrasts]
            for phase, contrasts in PRIMARY_CONTRASTS_BY_PHASE.items()
        },
        "secondary_contrasts_by_phase": {
            phase: [list(contrast) for contrast in contrasts]
            for phase, contrasts in SECONDARY_CONTRASTS_BY_PHASE.items()
        },
        "multiplicity": {
            "family_id": GLOBAL_CONFIRMATORY_FAMILY_ID,
            "alpha": GLOBAL_CONFIRMATORY_ALPHA,
            "method": "holm_two_sided",
            "hypothesis_count": GLOBAL_CONFIRMATORY_HYPOTHESIS_COUNT,
            "scope": (
                "all pooled primary metric by primary contrast tests on the "
                "organization ladder"
            ),
        },
        "transfer_contrasts": {
            "paired_block": "source_repository, target_repository, seed",
            "paired_block_count": TRANSFER_PAIRED_BLOCK_COUNT,
            "reference_arm": TRANSFER_REFERENCE_ARM,
            "reference_source": (
                "the target repository's own B3 cell on the canonical model, "
                "reached through each transfer cell's reference_cell_id; it is "
                "not a separate run"
            ),
            "evaluation_window_semantics": (
                "capability compilation is frozen for the opening episodes so "
                "the inherited state is measured before any arm can relearn an "
                "equivalent capability"
            ),
            "contrasts": [
                {
                    "treatment": treatment,
                    "control": control,
                    "question": TRANSFER_CONTRAST_QUESTIONS[(treatment, control)],
                }
                for treatment, control in TRANSFER_CONTRASTS
            ],
        },
        "confidence_interval": {
            "method": "paired_block_bootstrap_percentile",
            "confidence_level": 0.95,
        },
        "randomization_test": {
            "method": "paired_two_sided_sign_flip",
        },
        "secondary_sensitivity_metrics": SECONDARY_SENSITIVITY_METRIC_SPECS,
        "construct_boundaries": {
            "organization_ladder_step": (
                "B3-B2 moves the decision kernel, profile conditioning, and "
                "the capability lifecycle together; it estimates the bundled "
                "SocioGenesis step, not an isolated mechanism"
            ),
            "product_release_readiness": (
                "formal_release_ready is a thresholded evaluator diagnostic, "
                "excluded from the primary product composite and retained only "
                "in the declared secondary sensitivity endpoint"
            ),
        },
        "inference_population": {
            "scope": (
                "conditional on the preregistered frozen repositories, model "
                "bindings, and seed schedule"
            ),
            "repository_population_inference": False,
            "reason": (
                "repositories and models are fixed design points rather than "
                "probability samples from target populations"
            ),
        },
        "evidence_kind_policy": {
            "pooled_estimand": "frozen_acceptance_oracle_effect",
            "runtime_behavior_claim_requires": (
                "all main repositories use behavior evidence"
            ),
            "source_semantic_contract_is_not_runtime_behavior": True,
        },
    }


__all__ = [
    "ESTIMAND_SCHEMA_VERSION",
    "GLOBAL_CONFIRMATORY_ALPHA",
    "GLOBAL_CONFIRMATORY_FAMILY_ID",
    "GLOBAL_CONFIRMATORY_HYPOTHESIS_COUNT",
    "MAIN_PAIRED_BLOCK_COUNT",
    "PREREGISTERED_MODEL_COUNT",
    "PREREGISTERED_REPOSITORY_COUNT",
    "PRIMARY_CONTRASTS_BY_PHASE",
    "PRIMARY_METRIC_SPECS",
    "SEEDS_PER_CONDITION",
    "SECONDARY_SENSITIVITY_METRIC_SPECS",
    "SECONDARY_CONTRASTS_BY_PHASE",
    "TRANSFER_CONTRASTS",
    "TRANSFER_CONTRAST_QUESTIONS",
    "TRANSFER_PAIRED_BLOCK_COUNT",
    "TRANSFER_REFERENCE_ARM",
    "estimand_manifest",
]
