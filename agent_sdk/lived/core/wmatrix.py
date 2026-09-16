"""Trait → Action-Feature weight matrix (PCBSP §8).

The fixed, pre-registered, auditable matrix ``W[trait, feature]`` that, together
with the per-feature transforms (`transforms.py`, §6), turns the 12-dim
:class:`~agent_sdk.lived.core.contracts.ProfileVector` into the **TraitScore** term of
the utility (§12.1)::

    TraitScore_i(a) = Σ_t Σ_f  trait_i[t] · W[t][f] · transformed_feature_a[f]

Design rules (§8 / §17.4):
  * **The LLM never decides W.** It is this static table, version-controlled.
    Cells use the discrete level set :data:`ALLOWED_W_LEVELS` =
    {0, ±0.5, ±1.0, ±1.5, ±2.0}.
  * **Raw (un-centered) trait values** are used, per the §12.1 formula
    ``trait_i[t]`` ∈ [0,1]. (This differs from the earlier centered scaffold;
    PCBSP keeps it raw so SurvivalScore/CostScore carry the profile-independent
    baseline and TraitScore is purely the personality contribution.)
  * Cost features (risk_cost, reputation_risk, …) may appear with NEGATIVE cells:
    a negative cell means the trait *amplifies* aversion to that cost (the cell
    multiplies the positive loss-aversion-transformed cost, yielding a negative
    contribution). CostScore (§12.2) carries the profile-independent baseline;
    W adds the personality modulation on top (base + modulation, monotone, no
    double-count of the *baseline*).

The matrix is the literal §8.1 sparse table. Each trait's columns mirror its
``TraitSpec.linked_action_features`` (enforced by :func:`validate_matrix`).
Sensitivity tooling (§17.5): :func:`weight_sensitivity` and
:func:`validate_monotonicity`.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from agent_sdk.lived.core.contracts import ActionFeatures, ProfileVector, ProfileWeights

# §8 header suggests a 0.5 grid, but the concrete §8.1 table uses a finer 0.25
# grid (0.75/0.25/-0.25/-0.75). The concrete table is authoritative, so the
# legal level set is the 0.25-step grid in [-2, 2] (a superset of the §8 grid).
ALLOWED_W_LEVELS: Tuple[float, ...] = tuple(round(-2.0 + 0.25 * i, 2) for i in range(17))


# The literal §8.1 sparse weight table. Only non-zero cells listed.
W_MATRIX: Dict[str, Dict[str, float]] = {
    "risk_aversion": {
        "future_security_gain": 0.5, "storage_value": 0.5, "rule_compliance": 0.5,
        "risk_cost": -2.0, "uncertainty_cost": -1.5, "conflict_risk": -1.0,
        "reputation_risk": -0.75, "experiment_value": -0.5,
    },
    "curiosity": {
        "information_gain": 2.0, "experiment_value": 2.0, "novelty_value": 1.5,
        "skill_gain": 1.0, "future_security_gain": 0.5,
        "time_cost": -0.25, "uncertainty_cost": -0.25,
    },
    "long_termism": {
        "future_security_gain": 2.0, "storage_value": 2.0, "skill_gain": 0.75,
        "institution_gain": 0.75, "public_good_gain": 0.5, "time_cost": -0.25,
    },
    "altruism": {
        "altruistic_gain": 2.0, "trust_gain": 1.0, "reputation_gain": 0.75,
        "public_good_gain": 0.75, "group_loyalty_gain": 0.5,
        "private_gain": -0.5, "opportunistic_gain": -1.0, "public_harm": -1.5,
    },
    "reciprocity": {
        "reciprocity_gain": 2.0, "trust_gain": 1.0, "reputation_gain": 0.5,
        "trust_cost": -1.5, "public_harm": -1.0,
    },
    "fairness": {
        "fairness_gain": 2.0, "violation_detection": 1.5, "rule_compliance": 0.75,
        "institution_gain": 0.75, "public_good_gain": 0.75,
        "fairness_cost": -2.0, "opportunistic_gain": -1.0, "public_harm": -1.5,
    },
    "opportunism": {
        "private_gain": 2.0, "opportunistic_gain": 2.0, "autonomy_gain": 0.75,
        "dominance_gain": 0.5,
        "fairness_gain": -0.75, "rule_compliance": -0.75, "altruistic_gain": -0.75,
        "reputation_risk": -0.5,
    },
    "conformity": {
        "rule_compliance": 2.0, "institution_gain": 1.0, "reputation_gain": 0.75,
        "public_good_gain": 0.25,
        "conformity_cost": -2.0, "violation_gain": -2.0, "opportunistic_gain": -1.0,
    },
    "dominance": {
        "dominance_gain": 2.0, "institution_gain": 0.75, "coordination_gain": 1.0,
        "reputation_risk": -0.25,
    },
    "reputation_concern": {
        "reputation_gain": 2.0, "rule_compliance": 1.0, "altruistic_gain": 0.5,
        "reputation_risk": -2.0, "trust_cost": -1.0, "violation_gain": -2.0,
    },
    "distrust_sensitivity": {
        "verification_value": 2.0, "information_gain": 1.0, "autonomy_gain": 0.5,
        "trust_gain": -1.0,
    },
    "group_loyalty": {
        "group_loyalty_gain": 2.0, "public_good_gain": 0.75, "trust_gain": 0.5,
        "public_harm": -2.0, "opportunistic_gain": -1.5,
    },
}


# --------------------------------------------------------------------------- #
# Compile: ProfileVector -> per-feature TraitScore weights (raw traits, §12.1)
# --------------------------------------------------------------------------- #
def profile_to_weights(profile: ProfileVector) -> ProfileWeights:
    """Compile the per-feature TraitScore weight ``Σ_t trait[t]·W[t][f]`` (raw
    traits). The PCBSP scorer dots this with the *transformed* features; cost
    features thus get their negative contribution from negative W cells."""
    w: Dict[str, float] = {}
    for trait_name in ProfileVector.trait_names():
        tv = float(getattr(profile, trait_name))
        if tv == 0.0:
            continue
        for feat, cell in W_MATRIX.get(trait_name, {}).items():
            w[feat] = w.get(feat, 0.0) + tv * cell
    for f in ActionFeatures.feature_names():
        w.setdefault(f, 0.0)
    return ProfileWeights(weights=w)


# --------------------------------------------------------------------------- #
# Validation + sensitivity analysis (§8 / §17.5)
# --------------------------------------------------------------------------- #
def validate_matrix() -> List[str]:
    """Problems with the matrix (empty == OK): legal discrete levels, real
    features/traits, and each trait's columns == its TraitSpec mirror."""
    from agent_sdk.lived.core.schema import TRAIT_SPECS  # lazy: avoid import cycle

    problems: List[str] = []
    feats = set(ActionFeatures.feature_names())
    for trait, row in W_MATRIX.items():
        if trait not in ProfileVector.trait_names():
            problems.append(f"W_MATRIX references unknown trait {trait!r}")
            continue
        for f, cell in row.items():
            if f not in feats:
                problems.append(f"W[{trait}] references unknown feature {f!r}")
            if cell not in ALLOWED_W_LEVELS:
                problems.append(f"W[{trait}][{f}]={cell} not a legal level")
        spec = TRAIT_SPECS.get(trait)
        if spec is not None and set(row) != set(spec.linked_action_features):
            problems.append(
                f"W[{trait}] columns {sorted(row)} != linked_action_features "
                f"{sorted(spec.linked_action_features)}"
            )
    return problems


def weight_sensitivity(feature: str) -> Dict[str, float]:
    """``∂TraitScore_weight[feature]/∂trait`` for every trait touching it = the
    raw W cell. The auditable "which trait drives this feature, how hard" view."""
    return {t: row[feature] for t, row in W_MATRIX.items() if feature in row}


def validate_monotonicity() -> List[str]:
    """Sign discipline (§17.4): raising a trait 0→1 moves each coupled feature's
    compiled weight in the W cell's sign direction. Returns violations."""
    problems: List[str] = []
    base = ProfileVector().to_dict()
    for trait, row in W_MATRIX.items():
        for feat, cell in row.items():
            if cell == 0.0:
                continue
            lo = profile_to_weights(ProfileVector(**{**base, trait: 0.0})).get(feat)
            hi = profile_to_weights(ProfileVector(**{**base, trait: 1.0})).get(feat)
            moved = hi - lo
            if cell > 0 and moved <= 0:
                problems.append(f"{trait}->{feat}: W>0 but weight did not rise")
            if cell < 0 and moved >= 0:
                problems.append(f"{trait}->{feat}: W<0 but weight did not fall")
    return problems
