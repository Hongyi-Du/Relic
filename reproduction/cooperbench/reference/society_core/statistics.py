"""Statistical evidence helpers for randomized Society-Core contrasts."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import sqrt
from random import Random


@dataclass(frozen=True)
class RandomizationInferenceReport:
    report_id: str
    treated_count: int
    control_count: int
    treated_success_count: int
    control_success_count: int
    observed_lift: float
    p_value_one_sided: float
    method: str
    confidence_low: float
    confidence_high: float
    power_proxy: float
    caveats: tuple[str, ...] = ()


def _rate(successes: int, total: int) -> float:
    return successes / total if total else 0.0


def _wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return (0.0, 1.0)
    phat = successes / total
    denom = 1.0 + z * z / total
    center = (phat + z * z / (2 * total)) / denom
    margin = z * sqrt((phat * (1 - phat) + z * z / (4 * total)) / total) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def _sample_assignment_p_value(
    *,
    outcomes: tuple[int, ...],
    treated_count: int,
    observed_lift: float,
    max_samples: int,
    seed: int,
) -> tuple[float, str]:
    n = len(outcomes)
    if n <= 20:
        total = 0
        extreme = 0
        indexes = range(n)
        for treated_indexes in combinations(indexes, treated_count):
            treated_set = set(treated_indexes)
            treated_success = sum(outcomes[index] for index in treated_set)
            control_success = sum(outcomes[index] for index in indexes if index not in treated_set)
            lift = _rate(treated_success, treated_count) - _rate(control_success, n - treated_count)
            total += 1
            if lift >= observed_lift - 1e-12:
                extreme += 1
        return (extreme / max(1, total), "exact_enumeration")
    rng = Random(seed)
    extreme = 0
    indexes = tuple(range(n))
    for _ in range(max_samples):
        treated_set = set(rng.sample(indexes, treated_count))
        treated_success = sum(outcomes[index] for index in treated_set)
        control_success = sum(outcomes[index] for index in indexes if index not in treated_set)
        lift = _rate(treated_success, treated_count) - _rate(control_success, n - treated_count)
        if lift >= observed_lift - 1e-12:
            extreme += 1
    return ((extreme + 1) / (max_samples + 1), f"deterministic_monte_carlo_{max_samples}")


def peer_signal_randomization_inference(
    *,
    peer_signal_successes: int,
    peer_signal_total: int,
    access_only_successes: int,
    access_only_total: int,
    max_samples: int = 5000,
    seed: int = 1729,
) -> RandomizationInferenceReport:
    treated_rate = _rate(peer_signal_successes, peer_signal_total)
    control_rate = _rate(access_only_successes, access_only_total)
    observed_lift = treated_rate - control_rate
    outcomes = (1,) * peer_signal_successes + (0,) * (peer_signal_total - peer_signal_successes)
    outcomes += (1,) * access_only_successes + (0,) * (access_only_total - access_only_successes)
    caveats: list[str] = []
    if peer_signal_total <= 0 or access_only_total <= 0:
        caveats.append("empty_randomized_group")
    if peer_signal_successes < 5:
        caveats.append("few_treated_successes")
    if len(outcomes) <= 1 or caveats:
        p_value = 1.0
        method = "not_testable"
    else:
        p_value, method = _sample_assignment_p_value(
            outcomes=outcomes,
            treated_count=peer_signal_total,
            observed_lift=observed_lift,
            max_samples=max_samples,
            seed=seed,
        )
    treated_low, treated_high = _wilson_interval(peer_signal_successes, peer_signal_total)
    control_low, control_high = _wilson_interval(access_only_successes, access_only_total)
    confidence_low = treated_low - control_high
    confidence_high = treated_high - control_low
    pooled = _rate(peer_signal_successes + access_only_successes, peer_signal_total + access_only_total)
    standard_error = sqrt(
        pooled
        * (1 - pooled)
        * (1 / max(1, peer_signal_total) + 1 / max(1, access_only_total))
    )
    power_proxy = abs(observed_lift) / max(standard_error, 1e-9)
    if power_proxy < 1.96:
        caveats.append("underpowered_peer_signal_contrast")
    return RandomizationInferenceReport(
        report_id="peer_signal_randomization_inference_v15",
        treated_count=peer_signal_total,
        control_count=access_only_total,
        treated_success_count=peer_signal_successes,
        control_success_count=access_only_successes,
        observed_lift=observed_lift,
        p_value_one_sided=p_value,
        method=method,
        confidence_low=confidence_low,
        confidence_high=confidence_high,
        power_proxy=power_proxy,
        caveats=tuple(caveats),
    )
