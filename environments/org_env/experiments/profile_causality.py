"""Shared-candidate counterfactual evidence for profile-conditioned policy."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from environments.org_env.experiments.provenance import stable_fingerprint


PROFILE_CAUSALITY_SCHEMA_VERSION = "orgenv_profile_causality_v1"
TRAIT_SCAN_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)


def _softmax(values: Sequence[float], temperature: float) -> tuple[float, ...]:
    if not values:
        return ()
    maximum = max(values)
    weights = [math.exp((value - maximum) / temperature) for value in values]
    total = sum(weights) or 1.0
    return tuple(weight / total for weight in weights)


def _action_distribution(
    actions: Sequence[str],
    probabilities: Sequence[float],
) -> dict[str, float]:
    result: dict[str, float] = {}
    for action, probability in zip(actions, probabilities):
        result[action] = result.get(action, 0.0) + float(probability)
    return {key: result[key] for key in sorted(result)}


def _js_divergence(
    first: Mapping[str, float],
    second: Mapping[str, float],
) -> float:
    keys = set(first) | set(second)
    midpoint = {
        key: (float(first.get(key, 0.0)) + float(second.get(key, 0.0))) / 2
        for key in keys
    }

    def divergence(source: Mapping[str, float]) -> float:
        return sum(
            value * math.log(value / midpoint[key])
            for key, value in source.items()
            if value > 0 and midpoint[key] > 0
        )

    return 0.5 * divergence(first) + 0.5 * divergence(second)


def _feature_rows(scored_candidates: Sequence[tuple[Any, Any]]) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for _, features in scored_candidates:
        raw = features.to_dict() if hasattr(features, "to_dict") else dict(features)
        rows.append(
            {
                str(key): float(value)
                for key, value in raw.items()
                if isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
            }
        )
    return rows


def record_profile_counterfactual(
    *,
    world: Any,
    agent: Any,
    scored_candidates: Sequence[tuple[Any, Any]],
    conditioned_utilities: Sequence[float],
    penalties: Sequence[Mapping[str, float]],
    chosen_index: int,
    policy: Any,
    temperature: float,
) -> dict[str, Any] | None:
    """Record a non-interventional profile-off replay on the identical pool."""

    if not scored_candidates:
        return None
    from environments.org_env.runtime_adapter.policy import (
        PROFILE_COEFFS,
        OrgPolicy,
    )

    actions = [
        str(getattr(candidate, "action_type", ""))
        for candidate, _ in scored_candidates
    ]
    feature_rows = _feature_rows(scored_candidates)
    off_policy = OrgPolicy(mode="argmax", use_profile_conditioning=False)
    off_utilities = [
        off_policy.score(agent, features, world) + sum(penalty.values())
        for (_, features), penalty in zip(scored_candidates, penalties)
    ]
    on_probabilities = _softmax(conditioned_utilities, temperature)
    off_probabilities = _softmax(off_utilities, temperature)
    on_distribution = _action_distribution(actions, on_probabilities)
    off_distribution = _action_distribution(actions, off_probabilities)
    chosen_action = actions[chosen_index]
    conditioned_feature_scores = [
        policy.score(agent, features, world)
        for _, features in scored_candidates
    ]
    trait_scans: dict[str, dict[str, Any]] = {}
    traits = sorted({trait for trait, _, _ in PROFILE_COEFFS})
    for trait in traits:
        slopes = [
            sum(
                coefficient * row.get(feature, 0.0)
                for coefficient_trait, feature, coefficient in PROFILE_COEFFS
                if coefficient_trait == trait
            )
            for row in feature_rows
        ]
        if not any(abs(slope) > 1e-12 for slope in slopes):
            continue
        current = float(policy._lookup(agent, trait))
        utility_grid = [
            [
                score - current * slope + value * slope
                for score, slope in zip(conditioned_feature_scores, slopes)
            ]
            for value in TRAIT_SCAN_GRID
        ]
        # Monotonicity is checked on the ACTION PROBABILITY, not the utility.
        # The utility grid is `score - current*slope + value*slope`, which is
        # affine in the grid value by construction, so a utility-level check can
        # never fail and verifies only that the implementation is linear.
        # Under softmax, a candidate's probability is not monotone in the trait
        # whenever competing candidates carry different slopes — that is the
        # falsifiable property the preregistration asks for.
        grid_distributions = [
            _action_distribution(actions, _softmax(row, temperature))
            for row in utility_grid
        ]
        violations = 0
        for index, slope in enumerate(slopes):
            if abs(slope) <= 1e-12:
                continue
            action = actions[index]
            series = [dist.get(action, 0.0) for dist in grid_distributions]
            rising = all(
                later + 1e-12 >= earlier
                for earlier, later in zip(series, series[1:])
            )
            falling = all(
                later - 1e-12 <= earlier
                for earlier, later in zip(series, series[1:])
            )
            if slope >= 0 and not rising:
                violations += 1
            if slope < 0 and not falling:
                violations += 1
        low_distribution = grid_distributions[0]
        high_distribution = grid_distributions[-1]
        trait_scans[trait] = {
            "current_value": current,
            "candidate_slope_count": sum(
                abs(slope) > 1e-12 for slope in slopes
            ),
            "max_action_probability_delta": max(
                (
                    abs(
                        high_distribution.get(action, 0.0)
                        - low_distribution.get(action, 0.0)
                    )
                    for action in set(low_distribution) | set(high_distribution)
                ),
                default=0.0,
            ),
            "monotonicity_violation_count": violations,
        }

    record = {
        "tick": int(getattr(world, "world_tick", 0) or 0),
        "agent_id": str(getattr(agent, "id", "")),
        "profile_conditioning_enabled": bool(policy.use_profile_conditioning),
        "candidate_pool_hash": stable_fingerprint(
            [
                {
                    "action": action,
                    "features": features,
                }
                for action, features in zip(actions, feature_rows)
            ]
        ),
        "candidate_count": len(actions),
        "chosen_action": chosen_action,
        "conditioned_action_probabilities": {
            key: round(value, 12) for key, value in on_distribution.items()
        },
        "profile_off_action_probabilities": {
            key: round(value, 12) for key, value in off_distribution.items()
        },
        "js_divergence": round(
            _js_divergence(on_distribution, off_distribution),
            12,
        ),
        "argmax_changed": (
            max(on_distribution, key=on_distribution.get)
            != max(off_distribution, key=off_distribution.get)
        ),
        "chosen_action_probability_delta": round(
            on_distribution.get(chosen_action, 0.0)
            - off_distribution.get(chosen_action, 0.0),
            12,
        ),
        "trait_scans": trait_scans,
        "counterfactual_scope": (
            "same candidate pool and world state; structured profile-policy "
            "terms disabled; no action is executed by the counterfactual"
        ),
    }
    record["record_hash"] = stable_fingerprint(record)
    sink = getattr(world, "profile_causality_records", None)
    if sink is None:
        sink = []
        world.profile_causality_records = sink
    sink.append(record)
    return record


# The seven dimensions the study preregisters, mapped to the quantity that
# actually drives the policy. Two of them (cost_sensitivity, ownership_drive)
# had no channel at all until they were given PROFILE_COEFFS rows; the rest are
# carried by an existing trait or by a persona SKILL. Skills qualify because the
# shuffled-profile arm permutes the whole persona payload, skills included, so a
# skill-driven effect still moves when role-profile alignment is broken.
#
# Declaring the map lets the report state which of the seven it scanned instead
# of leaving a reader to assume all seven exist under those exact names.
PREREGISTERED_PROFILE_DIMENSIONS: dict[str, tuple[str, ...]] = {
    "evidence_threshold": ("claim_evidence_review", "quality_bar"),
    "quality_bar": ("quality_bar",),
    "cost_sensitivity": ("cost_sensitivity",),
    "customer_sensitivity": ("customer_sense",),
    "speed_bias": ("speed_bias",),
    "ownership_drive": ("ownership_drive",),
    "review_strictness": ("review_quality", "clarity_review"),
}


def preregistered_dimension_coverage(scanned: Sequence[str]) -> dict[str, Any]:
    """Which preregistered dimensions the trait scan actually reached."""
    seen = set(scanned)
    covered = {
        dimension: sorted(set(sources) & seen)
        for dimension, sources in PREREGISTERED_PROFILE_DIMENSIONS.items()
    }
    reached = sorted(name for name, hits in covered.items() if hits)
    return {
        "preregistered_dimension_count": len(PREREGISTERED_PROFILE_DIMENSIONS),
        "dimensions_scanned": reached,
        "dimensions_scanned_count": len(reached),
        "dimensions_not_scanned": sorted(
            set(PREREGISTERED_PROFILE_DIMENSIONS) - set(reached)
        ),
        "driving_quantity_by_dimension": {k: v for k, v in sorted(covered.items())},
    }


def build_profile_causality_evidence(world: Any) -> dict[str, Any]:
    records = list(getattr(world, "profile_causality_records", ()) or ())
    scan_count = sum(len(record.get("trait_scans") or {}) for record in records)
    violations = sum(
        int(scan.get("monotonicity_violation_count", 0) or 0)
        for record in records
        for scan in (record.get("trait_scans") or {}).values()
    )
    scanned_traits = sorted({
        trait
        for record in records
        for trait in (record.get("trait_scans") or {})
    })
    coverage = preregistered_dimension_coverage(scanned_traits)
    count = len(records)
    metrics = {
        "profile_policy_mean_js_divergence": (
            sum(float(record.get("js_divergence", 0.0)) for record in records)
            / count
            if count
            else 0.0
        ),
        "profile_policy_argmax_shift_rate": (
            sum(bool(record.get("argmax_changed")) for record in records) / count
            if count
            else 0.0
        ),
        "profile_policy_mean_chosen_probability_delta": (
            sum(
                float(record.get("chosen_action_probability_delta", 0.0))
                for record in records
            )
            / count
            if count
            else 0.0
        ),
        "profile_policy_monotonicity_violation_rate": (
            violations / scan_count if scan_count else 0.0
        ),
    }
    payload: dict[str, Any] = {
        "schema_version": PROFILE_CAUSALITY_SCHEMA_VERSION,
        "run_id": str(getattr(world, "run_id", "")),
        "decision_count": count,
        "trait_scan_count": scan_count,
        "monotonicity_violation_count": violations,
        "preregistered_dimension_coverage": coverage,
        "metrics": {
            key: round(float(value), 12)
            for key, value in metrics.items()
        },
        "records": records,
        "claim_boundary": (
            "mechanistic within-state profile-policy sensitivity; population "
            "outcomes still require preregistered P2/P3/P4/A1 contrasts"
        ),
    }
    payload["evidence_hash"] = stable_fingerprint(payload)
    validate_profile_causality_evidence(payload)
    return payload


def validate_profile_causality_evidence(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != PROFILE_CAUSALITY_SCHEMA_VERSION:
        raise ValueError("unknown_profile_causality_schema")
    declared = payload.get("evidence_hash")
    unhashed = dict(payload)
    unhashed.pop("evidence_hash", None)
    if not isinstance(declared, str) or stable_fingerprint(unhashed) != declared:
        raise ValueError("profile_causality_evidence_hash_mismatch")
    records = payload.get("records")
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise ValueError("profile_causality_records_must_be_a_list")
    if int(payload.get("decision_count", -1)) != len(records):
        raise ValueError("profile_causality_decision_count_mismatch")
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("profile_causality_record_must_be_an_object")
        unhashed_record = dict(record)
        record_hash = unhashed_record.pop("record_hash", None)
        if stable_fingerprint(unhashed_record) != record_hash:
            raise ValueError("profile_causality_record_hash_mismatch")


__all__ = [
    "PROFILE_CAUSALITY_SCHEMA_VERSION",
    "TRAIT_SCAN_GRID",
    "build_profile_causality_evidence",
    "record_profile_counterfactual",
    "validate_profile_causality_evidence",
]
