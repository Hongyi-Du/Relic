"""Inferential-resolution gates for the organizational-capability protocol."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import comb, isfinite
from typing import Sequence

from environments.org_env.experiments.statistics import (
    minimum_attainable_two_sided_sign_flip_p,
)

from .hashing import stable_hash
from .organization_experiment_estimands import (
    GLOBAL_CONFIRMATORY_ALPHA,
    GLOBAL_CONFIRMATORY_HYPOTHESIS_COUNT,
    MAIN_PAIRED_BLOCK_COUNT,
    TRANSFER_CONTRASTS,
    TRANSFER_PAIRED_BLOCK_COUNT,
)


RESOLUTION_AUDIT_SCHEMA_VERSION = "organization_inferential_resolution_v1"
PROSPECTIVE_POWER_AUDIT_SCHEMA_VERSION = "organization_prospective_sensitivity_power_v1"
PROSPECTIVE_POWER_MODEL_ID = "equal_magnitude_independent_pair_signs_v1"
DEFAULT_FAVORABLE_PAIR_PROBABILITIES = (
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
    0.95,
    0.99,
)
DEFAULT_TARGET_POWER = 0.80


@dataclass(frozen=True)
class ContrastFamilyResolution:
    family_id: str
    pair_count: int
    hypothesis_count: int
    alpha: float
    minimum_unadjusted_p: float
    minimum_holm_p: float
    exact_rejection_resolvable: bool
    intended_scope: str
    claim_boundary: str


@dataclass(frozen=True)
class InferentialResolutionAudit:
    schema_version: str
    families: tuple[ContrastFamilyResolution, ...]
    all_confirmatory_families_resolvable: bool
    blocking_reasons: tuple[str, ...]
    audit_hash: str


@dataclass(frozen=True)
class ProspectivePlanningAssumptions:
    model_id: str
    calculation_method: str
    effect_parameter: str
    paired_difference_absolute_magnitude: float
    independent_pair_signs: bool
    homogeneous_favorable_probability: bool
    null_sign_exchangeability: bool
    no_ties_or_missing_pairs: bool
    two_sided_test: bool
    rejection_rule: str
    multiplicity_planning_rule: str
    empirical_outcomes_used: bool


@dataclass(frozen=True)
class ProspectivePowerPoint:
    favorable_pair_probability: float
    rejection_probability: float


@dataclass(frozen=True)
class ProspectivePowerFamily:
    family_id: str
    pair_count: int
    hypothesis_count: int
    family_alpha: float
    focal_test_alpha_bound: float
    rejection_pair_counts: tuple[int, ...]
    null_rejection_probability: float
    sensitivity_points: tuple[ProspectivePowerPoint, ...]
    target_power: float
    minimum_grid_probability_meeting_target: float | None
    intended_scope: str
    claim_boundary: str


@dataclass(frozen=True)
class ProspectiveSensitivityPowerAudit:
    schema_version: str
    assumptions: ProspectivePlanningAssumptions
    families: tuple[ProspectivePowerFamily, ...]
    resolution_audit_hash: str
    deterministic: bool
    planning_only: bool
    scientific_result: bool
    changes_existing_inference_gate: bool
    claim_boundary: str
    audit_hash: str


def contrast_family_resolution(
    *,
    family_id: str,
    pair_count: int,
    hypothesis_count: int,
    alpha: float = 0.05,
    intended_scope: str,
    claim_boundary: str,
) -> ContrastFamilyResolution:
    if not family_id or not intended_scope or not claim_boundary:
        raise ValueError("resolution family labels must be non-empty")
    if (
        isinstance(hypothesis_count, bool)
        or not isinstance(hypothesis_count, int)
        or hypothesis_count <= 0
    ):
        raise ValueError("hypothesis_count must be a positive integer")
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must be between zero and one")
    minimum = minimum_attainable_two_sided_sign_flip_p(pair_count)
    minimum_holm = min(1.0, minimum * hypothesis_count)
    return ContrastFamilyResolution(
        family_id=family_id,
        pair_count=pair_count,
        hypothesis_count=hypothesis_count,
        alpha=float(alpha),
        minimum_unadjusted_p=minimum,
        minimum_holm_p=minimum_holm,
        exact_rejection_resolvable=minimum_holm < float(alpha),
        intended_scope=intended_scope,
        claim_boundary=claim_boundary,
    )


def audit_preregistered_resolution(
    families: Sequence[ContrastFamilyResolution] | None = None,
) -> InferentialResolutionAudit:
    """Audit whether exact tests can reject at the preregistered family alpha.

    This is a resolution audit, not a prospective power calculation. It only
    detects designs where no possible observed effect can cross the exact-test
    threshold after multiplicity correction.
    """

    rows = tuple(
        families
        or (
            contrast_family_resolution(
                family_id="global_primary_main_resolution",
                pair_count=MAIN_PAIRED_BLOCK_COUNT,
                hypothesis_count=GLOBAL_CONFIRMATORY_HYPOTHESIS_COUNT,
                intended_scope="confirmatory",
                claim_boundary=(
                    "one paired block per repository and seed; adjusted inside "
                    "one global primary family"
                ),
            ),
            contrast_family_resolution(
                family_id="capability_transfer_resolution",
                pair_count=TRANSFER_PAIRED_BLOCK_COUNT,
                hypothesis_count=len(TRANSFER_CONTRASTS),
                intended_scope="mechanism_check",
                claim_boundary=(
                    "one paired block per source-to-target mapping and seed; "
                    "reported as effect sizes and intervals outside the global "
                    "confirmatory family"
                ),
            ),
        )
    )
    blocking = tuple(
        f"exact_rejection_not_resolvable:{row.family_id}"
        for row in rows
        if row.intended_scope == "confirmatory" and not row.exact_rejection_resolvable
    )
    payload = {
        "schema_version": RESOLUTION_AUDIT_SCHEMA_VERSION,
        "families": [asdict(row) for row in rows],
        "all_confirmatory_families_resolvable": not blocking,
        "blocking_reasons": blocking,
    }
    return InferentialResolutionAudit(
        schema_version=RESOLUTION_AUDIT_SCHEMA_VERSION,
        families=rows,
        all_confirmatory_families_resolvable=not blocking,
        blocking_reasons=blocking,
        audit_hash=stable_hash(payload),
    )


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _probability(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite probability")
    number = float(value)
    if not isfinite(number) or not 0.0 < number < 1.0:
        raise ValueError(f"{name} must be a finite probability")
    return number


def _validated_effect_grid(
    probabilities: Sequence[float],
) -> tuple[float, ...]:
    rows = tuple(probabilities)
    if not rows:
        raise ValueError("favorable_pair_probabilities must not be empty")
    validated: list[float] = []
    for value in rows:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("favorable_pair_probabilities must contain finite numbers")
        probability = float(value)
        if not isfinite(probability) or not 0.5 <= probability <= 1.0:
            raise ValueError("favorable_pair_probabilities must be between 0.5 and 1.0")
        validated.append(probability)
    if any(current <= previous for previous, current in zip(validated, validated[1:])):
        raise ValueError("favorable_pair_probabilities must be strictly increasing")
    return tuple(validated)


def _equal_magnitude_two_sided_p(*, pair_count: int, favorable_count: int) -> float:
    observed_distance = abs(2 * favorable_count - pair_count)
    extreme_count = sum(
        comb(pair_count, count)
        for count in range(pair_count + 1)
        if abs(2 * count - pair_count) >= observed_distance
    )
    return extreme_count / (2**pair_count)


def _binomial_rejection_probability(
    *, pair_count: int, favorable_probability: float, rejection_counts: set[int]
) -> float:
    return sum(
        comb(pair_count, count)
        * (favorable_probability**count)
        * ((1.0 - favorable_probability) ** (pair_count - count))
        for count in rejection_counts
    )


def prospective_power_family(
    *,
    family_id: str,
    pair_count: int,
    hypothesis_count: int,
    family_alpha: float = 0.05,
    favorable_pair_probabilities: Sequence[float] = (
        DEFAULT_FAVORABLE_PAIR_PROBABILITIES
    ),
    target_power: float = DEFAULT_TARGET_POWER,
    intended_scope: str,
    claim_boundary: str,
) -> ProspectivePowerFamily:
    """Compute exact planning sensitivity under an explicit sign model.

    Each paired difference is assumed to have absolute magnitude one and an
    independent favorable sign with the supplied probability. Under this
    planning model, the preregistered paired sign-flip statistic has an exact
    binomial rejection probability. For a Holm family, ``alpha / m`` is used
    as a conservative focal-test bound; this is not joint Holm-family power.
    """

    labels = (family_id, intended_scope, claim_boundary)
    if any(not isinstance(label, str) or not label.strip() for label in labels):
        raise ValueError("prospective power family labels must be non-empty")
    pairs = _positive_integer(pair_count, name="pair_count")
    hypotheses = _positive_integer(
        hypothesis_count,
        name="hypothesis_count",
    )
    alpha = _probability(family_alpha, name="family_alpha")
    target = _probability(target_power, name="target_power")
    grid = _validated_effect_grid(favorable_pair_probabilities)
    focal_alpha = alpha / hypotheses
    rejection_counts = {
        count
        for count in range(pairs + 1)
        if _equal_magnitude_two_sided_p(
            pair_count=pairs,
            favorable_count=count,
        )
        < focal_alpha
    }
    points = tuple(
        ProspectivePowerPoint(
            favorable_pair_probability=probability,
            rejection_probability=_binomial_rejection_probability(
                pair_count=pairs,
                favorable_probability=probability,
                rejection_counts=rejection_counts,
            ),
        )
        for probability in grid
    )
    minimum_grid_probability = next(
        (
            point.favorable_pair_probability
            for point in points
            if point.rejection_probability >= target
        ),
        None,
    )
    return ProspectivePowerFamily(
        family_id=family_id,
        pair_count=pairs,
        hypothesis_count=hypotheses,
        family_alpha=alpha,
        focal_test_alpha_bound=focal_alpha,
        rejection_pair_counts=tuple(sorted(rejection_counts)),
        null_rejection_probability=_binomial_rejection_probability(
            pair_count=pairs,
            favorable_probability=0.5,
            rejection_counts=rejection_counts,
        ),
        sensitivity_points=points,
        target_power=target,
        minimum_grid_probability_meeting_target=minimum_grid_probability,
        intended_scope=intended_scope,
        claim_boundary=claim_boundary,
    )


def prospective_sensitivity_power_payload(
    audit: ProspectiveSensitivityPowerAudit,
) -> dict[str, object]:
    payload = asdict(audit)
    payload.pop("audit_hash", None)
    return payload


def verify_prospective_sensitivity_power_audit(
    audit: ProspectiveSensitivityPowerAudit,
) -> bool:
    return stable_hash(prospective_sensitivity_power_payload(audit)) == audit.audit_hash


def audit_preregistered_prospective_power(
    *,
    favorable_pair_probabilities: Sequence[float] = (
        DEFAULT_FAVORABLE_PAIR_PROBABILITIES
    ),
    target_power: float = DEFAULT_TARGET_POWER,
) -> ProspectiveSensitivityPowerAudit:
    """Create a content-addressed prospective planning artifact.

    The artifact consumes no experiment outcomes. Its rejection probabilities
    are sensitivity values conditional on the stated planning model, not
    estimated scientific effects and not evidence that any hypothesis holds.
    """

    grid = _validated_effect_grid(favorable_pair_probabilities)
    target = _probability(target_power, name="target_power")
    families = (
        prospective_power_family(
            family_id="global_primary_main_planning",
            pair_count=MAIN_PAIRED_BLOCK_COUNT,
            hypothesis_count=GLOBAL_CONFIRMATORY_HYPOTHESIS_COUNT,
            family_alpha=GLOBAL_CONFIRMATORY_ALPHA,
            favorable_pair_probabilities=grid,
            target_power=target,
            intended_scope="confirmatory_design_planning",
            claim_boundary=(
                "conservative focal-test sensitivity inside the locked global "
                "primary Holm family; not joint family power"
            ),
        ),
        prospective_power_family(
            family_id="transfer_planning",
            pair_count=TRANSFER_PAIRED_BLOCK_COUNT,
            hypothesis_count=1,
            family_alpha=GLOBAL_CONFIRMATORY_ALPHA,
            favorable_pair_probabilities=grid,
            target_power=target,
            intended_scope="transfer_design_planning",
            claim_boundary=(
                "separate transfer sensitivity scenario; it does not add a "
                "hypothesis to or alter the locked global primary family"
            ),
        ),
    )
    resolution_audit = audit_preregistered_resolution()
    assumptions = ProspectivePlanningAssumptions(
        model_id=PROSPECTIVE_POWER_MODEL_ID,
        calculation_method="exact_binomial_enumeration",
        effect_parameter="Pr(paired_difference > 0)",
        paired_difference_absolute_magnitude=1.0,
        independent_pair_signs=True,
        homogeneous_favorable_probability=True,
        null_sign_exchangeability=True,
        no_ties_or_missing_pairs=True,
        two_sided_test=True,
        rejection_rule="two-sided sign-flip p < family_alpha / hypothesis_count",
        multiplicity_planning_rule=(
            "alpha divided by hypothesis count as a conservative "
            "focal-test bound for Holm"
        ),
        empirical_outcomes_used=False,
    )
    boundary = (
        "Prospective sensitivity values are deterministic consequences of "
        "planning assumptions. They are not observed power, estimated effects, "
        "scientific results, or permission to change an inference gate."
    )
    payload = {
        "schema_version": PROSPECTIVE_POWER_AUDIT_SCHEMA_VERSION,
        "assumptions": asdict(assumptions),
        "families": [asdict(family) for family in families],
        "resolution_audit_hash": resolution_audit.audit_hash,
        "deterministic": True,
        "planning_only": True,
        "scientific_result": False,
        "changes_existing_inference_gate": False,
        "claim_boundary": boundary,
    }
    return ProspectiveSensitivityPowerAudit(
        schema_version=PROSPECTIVE_POWER_AUDIT_SCHEMA_VERSION,
        assumptions=assumptions,
        families=families,
        resolution_audit_hash=resolution_audit.audit_hash,
        deterministic=True,
        planning_only=True,
        scientific_result=False,
        changes_existing_inference_gate=False,
        claim_boundary=boundary,
        audit_hash=stable_hash(payload),
    )


__all__ = [
    "DEFAULT_FAVORABLE_PAIR_PROBABILITIES",
    "DEFAULT_TARGET_POWER",
    "PROSPECTIVE_POWER_AUDIT_SCHEMA_VERSION",
    "PROSPECTIVE_POWER_MODEL_ID",
    "RESOLUTION_AUDIT_SCHEMA_VERSION",
    "ContrastFamilyResolution",
    "InferentialResolutionAudit",
    "ProspectivePlanningAssumptions",
    "ProspectivePowerFamily",
    "ProspectivePowerPoint",
    "ProspectiveSensitivityPowerAudit",
    "audit_preregistered_prospective_power",
    "audit_preregistered_resolution",
    "contrast_family_resolution",
    "prospective_power_family",
    "prospective_sensitivity_power_payload",
    "verify_prospective_sensitivity_power_audit",
]
