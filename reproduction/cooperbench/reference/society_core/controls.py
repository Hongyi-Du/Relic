"""Post-hoc control diagnostics for artifact diffusion claims."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

from .hashing import stable_hash
from .network import artifact_exposure_diagnostic
from .schemas import ExternalEvent, SocietyState
from .statistics import peer_signal_randomization_inference


PEER_EARLY_WINDOW_TICKS = 25
TRIAL_ACTION_TYPES = frozenset({"try_artifact_on_task", "reuse_artifact"})
PAYMENT_ACTION_TYPES = frozenset({"pay_for_artifact"})

COVARIATE_NAMES = (
    "curiosity",
    "attention_budget",
    "hunger",
    "fatigue",
    "stress",
    "uncertainty",
    "food_or_budget_token",
    "physiological_energy",
)

NETWORK_DIFFUSION_FATAL_CAVEATS = {
    "no_randomized_network_holdout_design",
    "randomized_group_imbalance",
    "weak_network_intent_to_treat_lift",
    "no_network_exposure_observed",
    "insufficient_network_exposure",
    "insufficient_network_exposed_trial_actors",
}

PEER_INFLUENCE_FATAL_CAVEATS = NETWORK_DIFFUSION_FATAL_CAVEATS.union(
    {
        "no_peer_signal_randomization_design",
        "peer_signal_group_imbalance",
    }
)

PEER_TRIAL_OUTCOME_CAVEATS = {
    "weak_peer_signal_intent_to_treat_lift",
    "insufficient_peer_signal_trial_actors",
    "peer_signal_randomization_not_significant",
    "underpowered_peer_signal_contrast",
}

PEER_PAYMENT_OUTCOME_CAVEATS = {
    "weak_peer_signal_payment_intent_to_treat_lift",
    "insufficient_peer_signal_payment_actors",
    "peer_signal_payment_randomization_not_significant",
    "underpowered_peer_signal_payment_contrast",
}

PEER_EARLY_TRIAL_OUTCOME_CAVEATS = {
    "weak_peer_signal_early_trial_intent_to_treat_lift",
    "insufficient_peer_signal_early_trial_actors",
    "peer_signal_early_trial_randomization_not_significant",
    "underpowered_peer_signal_early_trial_contrast",
}

PEER_EARLY_PAYMENT_OUTCOME_CAVEATS = {
    "weak_peer_signal_early_payment_intent_to_treat_lift",
    "insufficient_peer_signal_early_payment_actors",
    "peer_signal_early_payment_randomization_not_significant",
    "underpowered_peer_signal_early_payment_contrast",
}


@dataclass(frozen=True)
class ArtifactCovariateControlReport:
    artifact_id: str
    exposure_count: int
    trial_actor_count: int
    payment_actor_count: int
    matched_control_count: int
    trial_rate: float
    payment_rate: float
    matched_mean_distance: float
    max_abs_standardized_mean_difference: float
    standardized_mean_differences: dict[str, float]
    release_exposure_count: int
    network_exposure_count: int
    unknown_exposure_count: int
    network_trial_actor_count: int
    network_payment_actor_count: int
    mean_network_distance: float
    exposure_source_counts: dict[str, int]
    randomized_control_design: bool
    network_eligible_count: int
    network_holdout_count: int
    network_eligible_trial_rate: float
    network_holdout_trial_rate: float
    network_intent_to_treat_lift: float
    randomized_group_max_abs_smd: float
    network_diffusion_claim_allowed: bool
    peer_signal_design: bool
    peer_signal_eligible_count: int
    access_only_control_count: int
    peer_signal_trial_rate: float
    access_only_trial_rate: float
    peer_signal_intent_to_treat_lift: float
    peer_signal_payment_rate: float
    access_only_payment_rate: float
    peer_signal_payment_intent_to_treat_lift: float
    peer_signal_group_max_abs_smd: float
    peer_signal_trial_actor_count: int
    peer_signal_payment_actor_count: int
    peer_signal_randomization_p_value: float
    peer_signal_confidence_low: float
    peer_signal_confidence_high: float
    peer_signal_power_proxy: float
    peer_signal_payment_randomization_p_value: float
    peer_signal_payment_confidence_low: float
    peer_signal_payment_confidence_high: float
    peer_signal_payment_power_proxy: float
    early_window_ticks: int
    peer_signal_early_trial_rate: float
    access_only_early_trial_rate: float
    peer_signal_early_trial_intent_to_treat_lift: float
    peer_signal_early_payment_rate: float
    access_only_early_payment_rate: float
    peer_signal_early_payment_intent_to_treat_lift: float
    peer_signal_early_trial_actor_count: int
    peer_signal_early_payment_actor_count: int
    peer_signal_early_trial_randomization_p_value: float
    peer_signal_early_trial_confidence_low: float
    peer_signal_early_trial_confidence_high: float
    peer_signal_early_trial_power_proxy: float
    peer_signal_early_payment_randomization_p_value: float
    peer_signal_early_payment_confidence_low: float
    peer_signal_early_payment_confidence_high: float
    peer_signal_early_payment_power_proxy: float
    peer_signal_mean_first_trial_tick: float | None
    access_only_mean_first_trial_tick: float | None
    peer_signal_mean_first_payment_tick: float | None
    access_only_mean_first_payment_tick: float | None
    peer_early_trial_claim_allowed: bool
    peer_early_payment_claim_allowed: bool
    peer_trial_claim_allowed: bool
    peer_payment_claim_allowed: bool
    peer_influence_claim_allowed: bool
    influence_claim_allowed: bool
    caveats: tuple[str, ...] = ()


@dataclass(frozen=True)
class ArmControlReport:
    report_id: str
    arm_id: str
    artifact_reports: dict[str, ArtifactCovariateControlReport]
    network_diffusion_claim_allowed: bool
    peer_influence_claim_allowed: bool
    network_influence_claim_allowed: bool
    network_control_caveats: tuple[str, ...]

    def hash(self) -> str:
        return stable_hash(self)


def extract_agent_covariates(state: SocietyState) -> dict[str, dict[str, float]]:
    covariates: dict[str, dict[str, float]] = {}
    for agent_id, agent in sorted(state.agents.items()):
        covariates[agent_id] = {
            "curiosity": float(agent.affect.curiosity),
            "attention_budget": float(agent.action_budgets.attention_budget),
            "hunger": float(agent.body.hunger),
            "fatigue": float(agent.body.fatigue),
            "stress": float(agent.body.stress),
            "uncertainty": float(agent.cognition.uncertainty),
            "food_or_budget_token": float(agent.material_resources.food_or_budget_token),
            "physiological_energy": float(agent.body.physiological_energy),
        }
    return covariates


def actors_for_artifact_actions(
    event_log: list[ExternalEvent],
    artifact_id: str,
    action_types: frozenset[str],
    *,
    max_tick: int | None = None,
) -> set[str]:
    return {
        event.actor_id
        for event in event_log
        if event.artifact_id == artifact_id
        and event.actor_id
        and event.action_type in action_types
        and (max_tick is None or event.tick <= max_tick)
    }


def first_action_ticks_for_artifact(
    event_log: list[ExternalEvent],
    artifact_id: str,
    action_types: frozenset[str],
) -> dict[str, int]:
    first_ticks: dict[str, int] = {}
    for event in event_log:
        if (
            event.artifact_id == artifact_id
            and event.actor_id
            and event.action_type in action_types
        ):
            current = first_ticks.get(event.actor_id)
            if current is None or event.tick < current:
                first_ticks[event.actor_id] = event.tick
    return first_ticks


def trial_actors_for_artifact(event_log: list[ExternalEvent], artifact_id: str) -> set[str]:
    return actors_for_artifact_actions(event_log, artifact_id, TRIAL_ACTION_TYPES)


def payment_actors_for_artifact(event_log: list[ExternalEvent], artifact_id: str) -> set[str]:
    return actors_for_artifact_actions(event_log, artifact_id, PAYMENT_ACTION_TYPES)


def _mean_optional(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return _mean(present) if present else None


def _mean_first_action_tick(actor_ids: tuple[str, ...], first_ticks: dict[str, int]) -> float | None:
    values = [float(first_ticks[agent_id]) for agent_id in actor_ids if agent_id in first_ticks]
    return _mean(values) if values else None


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _sample_sd(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = _mean(values)
    return sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def _distance(a: dict[str, float], b: dict[str, float]) -> float:
    return sqrt(sum((a[name] - b[name]) ** 2 for name in COVARIATE_NAMES))


def _matched_controls(
    *,
    trial_ids: tuple[str, ...],
    nontrial_ids: tuple[str, ...],
    covariates: dict[str, dict[str, float]],
) -> tuple[tuple[str, ...], float]:
    remaining = list(nontrial_ids)
    matches: list[str] = []
    distances: list[float] = []
    for trial_id in trial_ids:
        if not remaining:
            break
        best_index = min(
            range(len(remaining)),
            key=lambda index: _distance(covariates[trial_id], covariates[remaining[index]]),
        )
        matched_id = remaining.pop(best_index)
        matches.append(matched_id)
        distances.append(_distance(covariates[trial_id], covariates[matched_id]))
    return tuple(matches), _mean(distances)


def _standardized_mean_differences(
    *,
    trial_ids: tuple[str, ...],
    control_ids: tuple[str, ...],
    covariates: dict[str, dict[str, float]],
) -> dict[str, float]:
    if not trial_ids or not control_ids:
        return {name: 0.0 for name in COVARIATE_NAMES}
    result: dict[str, float] = {}
    for name in COVARIATE_NAMES:
        trial_values = [covariates[agent_id][name] for agent_id in trial_ids]
        control_values = [covariates[agent_id][name] for agent_id in control_ids]
        pooled = sqrt((_sample_sd(trial_values) ** 2 + _sample_sd(control_values) ** 2) / 2)
        mean_difference = _mean(trial_values) - _mean(control_values)
        if pooled == 0:
            result[name] = 0.0 if mean_difference == 0 else 1_000_000.0
        else:
            result[name] = mean_difference / pooled
    return result


def _group_standardized_mean_differences(
    *,
    group_a_ids: tuple[str, ...],
    group_b_ids: tuple[str, ...],
    covariates: dict[str, dict[str, float]],
) -> dict[str, float]:
    return _standardized_mean_differences(
        trial_ids=group_a_ids,
        control_ids=group_b_ids,
        covariates=covariates,
    )


def build_artifact_covariate_control_report(
    *,
    state: SocietyState,
    initial_covariates: dict[str, dict[str, float]],
    artifact_id: str,
) -> ArtifactCovariateControlReport:
    exposed = tuple(sorted(agent_id for agent_id, agent in state.agents.items() if artifact_id in agent.artifact_access))
    event_log = list(state.event_log)
    trial_ids = tuple(sorted(trial_actors_for_artifact(event_log, artifact_id)))
    payment_ids = tuple(sorted(payment_actors_for_artifact(event_log, artifact_id)))
    early_trial_ids = tuple(sorted(actors_for_artifact_actions(
        event_log,
        artifact_id,
        TRIAL_ACTION_TYPES,
        max_tick=PEER_EARLY_WINDOW_TICKS,
    )))
    early_payment_ids = tuple(sorted(actors_for_artifact_actions(
        event_log,
        artifact_id,
        PAYMENT_ACTION_TYPES,
        max_tick=PEER_EARLY_WINDOW_TICKS,
    )))
    first_trial_ticks = first_action_ticks_for_artifact(event_log, artifact_id, TRIAL_ACTION_TYPES)
    first_payment_ticks = first_action_ticks_for_artifact(event_log, artifact_id, PAYMENT_ACTION_TYPES)
    nontrial_ids = tuple(agent_id for agent_id in exposed if agent_id not in set(trial_ids))
    matched_ids, matched_distance = _matched_controls(
        trial_ids=trial_ids,
        nontrial_ids=nontrial_ids,
        covariates=initial_covariates,
    )
    smds = _standardized_mean_differences(
        trial_ids=trial_ids,
        control_ids=matched_ids,
        covariates=initial_covariates,
    )
    max_abs_smd = max((abs(value) for value in smds.values()), default=0.0)
    network_diagnostic = artifact_exposure_diagnostic(
        state=state,
        artifact_id=artifact_id,
        trial_actor_ids=set(trial_ids),
    )
    eligible_ids = tuple(sorted(
        agent_id
        for agent_id, agent in state.agents.items()
        if agent.artifact_exposure_groups.get(artifact_id) == "network_eligible"
    ))
    holdout_ids = tuple(sorted(
        agent_id
        for agent_id, agent in state.agents.items()
        if agent.artifact_exposure_groups.get(artifact_id) == "network_holdout"
    ))
    randomized_control_design = bool(eligible_ids and holdout_ids)
    eligible_trial_count = len(set(eligible_ids).intersection(trial_ids))
    holdout_trial_count = len(set(holdout_ids).intersection(trial_ids))
    network_payment_count = len({
        agent_id for agent_id in payment_ids
        if state.agents[agent_id].artifact_exposure_sources.get(artifact_id) == "network"
    })
    eligible_trial_rate = eligible_trial_count / max(1, len(eligible_ids))
    holdout_trial_rate = holdout_trial_count / max(1, len(holdout_ids))
    randomized_smds = _group_standardized_mean_differences(
        group_a_ids=eligible_ids,
        group_b_ids=holdout_ids,
        covariates=initial_covariates,
    )
    randomized_group_max_abs_smd = max((abs(value) for value in randomized_smds.values()), default=0.0)
    network_itt_lift = eligible_trial_rate - holdout_trial_rate
    network_diffusion_claim_allowed = (
        randomized_control_design
        and len(eligible_ids) >= 5
        and len(holdout_ids) >= 5
        and network_diagnostic.network_exposure_count >= 5
        and network_diagnostic.network_trial_actor_count >= 5
        and randomized_group_max_abs_smd <= 0.25
        and network_itt_lift > 0.02
    )
    peer_signal_ids = tuple(sorted(
        agent_id
        for agent_id, agent in state.agents.items()
        if agent.artifact_peer_signal_groups.get(artifact_id) == "peer_signal_eligible"
    ))
    access_only_ids = tuple(sorted(
        agent_id
        for agent_id, agent in state.agents.items()
        if agent.artifact_peer_signal_groups.get(artifact_id) == "access_only_control"
    ))
    peer_signal_design = bool(peer_signal_ids and access_only_ids)
    peer_signal_trial_count = len(set(peer_signal_ids).intersection(trial_ids))
    access_only_trial_count = len(set(access_only_ids).intersection(trial_ids))
    peer_signal_payment_count = len(set(peer_signal_ids).intersection(payment_ids))
    access_only_payment_count = len(set(access_only_ids).intersection(payment_ids))
    peer_signal_early_trial_count = len(set(peer_signal_ids).intersection(early_trial_ids))
    access_only_early_trial_count = len(set(access_only_ids).intersection(early_trial_ids))
    peer_signal_early_payment_count = len(set(peer_signal_ids).intersection(early_payment_ids))
    access_only_early_payment_count = len(set(access_only_ids).intersection(early_payment_ids))
    peer_signal_trial_rate = peer_signal_trial_count / max(1, len(peer_signal_ids))
    access_only_trial_rate = access_only_trial_count / max(1, len(access_only_ids))
    peer_signal_itt_lift = peer_signal_trial_rate - access_only_trial_rate
    peer_signal_payment_rate = peer_signal_payment_count / max(1, len(peer_signal_ids))
    access_only_payment_rate = access_only_payment_count / max(1, len(access_only_ids))
    peer_signal_payment_itt_lift = peer_signal_payment_rate - access_only_payment_rate
    peer_signal_early_trial_rate = peer_signal_early_trial_count / max(1, len(peer_signal_ids))
    access_only_early_trial_rate = access_only_early_trial_count / max(1, len(access_only_ids))
    peer_signal_early_trial_itt_lift = peer_signal_early_trial_rate - access_only_early_trial_rate
    peer_signal_early_payment_rate = peer_signal_early_payment_count / max(1, len(peer_signal_ids))
    access_only_early_payment_rate = access_only_early_payment_count / max(1, len(access_only_ids))
    peer_signal_early_payment_itt_lift = (
        peer_signal_early_payment_rate - access_only_early_payment_rate
    )
    peer_signal_mean_first_trial_tick = _mean_first_action_tick(peer_signal_ids, first_trial_ticks)
    access_only_mean_first_trial_tick = _mean_first_action_tick(access_only_ids, first_trial_ticks)
    peer_signal_mean_first_payment_tick = _mean_first_action_tick(peer_signal_ids, first_payment_ticks)
    access_only_mean_first_payment_tick = _mean_first_action_tick(access_only_ids, first_payment_ticks)
    peer_signal_smds = _group_standardized_mean_differences(
        group_a_ids=peer_signal_ids,
        group_b_ids=access_only_ids,
        covariates=initial_covariates,
    )
    peer_signal_group_max_abs_smd = max((abs(value) for value in peer_signal_smds.values()), default=0.0)
    inference = peer_signal_randomization_inference(
        peer_signal_successes=peer_signal_trial_count,
        peer_signal_total=len(peer_signal_ids),
        access_only_successes=access_only_trial_count,
        access_only_total=len(access_only_ids),
    )
    payment_inference = peer_signal_randomization_inference(
        peer_signal_successes=peer_signal_payment_count,
        peer_signal_total=len(peer_signal_ids),
        access_only_successes=access_only_payment_count,
        access_only_total=len(access_only_ids),
    )
    early_trial_inference = peer_signal_randomization_inference(
        peer_signal_successes=peer_signal_early_trial_count,
        peer_signal_total=len(peer_signal_ids),
        access_only_successes=access_only_early_trial_count,
        access_only_total=len(access_only_ids),
    )
    early_payment_inference = peer_signal_randomization_inference(
        peer_signal_successes=peer_signal_early_payment_count,
        peer_signal_total=len(peer_signal_ids),
        access_only_successes=access_only_early_payment_count,
        access_only_total=len(access_only_ids),
    )
    peer_trial_claim_allowed = (
        network_diffusion_claim_allowed
        and peer_signal_design
        and len(peer_signal_ids) >= 5
        and len(access_only_ids) >= 5
        and peer_signal_trial_count >= 5
        and peer_signal_group_max_abs_smd <= 0.25
        and peer_signal_itt_lift > 0.02
        and inference.p_value_one_sided <= 0.05
        and "underpowered_peer_signal_contrast" not in inference.caveats
    )
    peer_payment_claim_allowed = (
        network_diffusion_claim_allowed
        and peer_signal_design
        and len(peer_signal_ids) >= 5
        and len(access_only_ids) >= 5
        and peer_signal_payment_count >= 5
        and peer_signal_group_max_abs_smd <= 0.25
        and peer_signal_payment_itt_lift > 0.02
        and payment_inference.p_value_one_sided <= 0.05
        and "underpowered_peer_signal_contrast" not in payment_inference.caveats
    )
    peer_early_trial_claim_allowed = (
        network_diffusion_claim_allowed
        and peer_signal_design
        and len(peer_signal_ids) >= 5
        and len(access_only_ids) >= 5
        and peer_signal_early_trial_count >= 5
        and peer_signal_group_max_abs_smd <= 0.25
        and peer_signal_early_trial_itt_lift > 0.02
        and early_trial_inference.p_value_one_sided <= 0.05
        and "underpowered_peer_signal_contrast" not in early_trial_inference.caveats
    )
    peer_early_payment_claim_allowed = (
        network_diffusion_claim_allowed
        and peer_signal_design
        and len(peer_signal_ids) >= 5
        and len(access_only_ids) >= 5
        and peer_signal_early_payment_count >= 5
        and peer_signal_group_max_abs_smd <= 0.25
        and peer_signal_early_payment_itt_lift > 0.02
        and early_payment_inference.p_value_one_sided <= 0.05
        and "underpowered_peer_signal_contrast" not in early_payment_inference.caveats
    )
    peer_influence_claim_allowed = (
        peer_trial_claim_allowed
        or peer_payment_claim_allowed
        or peer_early_trial_claim_allowed
        or peer_early_payment_claim_allowed
    )
    influence_claim_allowed = peer_influence_claim_allowed
    caveats: list[str] = []
    if not randomized_control_design:
        caveats.append("no_randomized_network_holdout_design")
    if randomized_control_design and randomized_group_max_abs_smd > 0.25:
        caveats.append("randomized_group_imbalance")
    if randomized_control_design and network_itt_lift <= 0.02:
        caveats.append("weak_network_intent_to_treat_lift")
    if not peer_signal_design:
        caveats.append("no_peer_signal_randomization_design")
    if peer_signal_design and peer_signal_group_max_abs_smd > 0.25:
        caveats.append("peer_signal_group_imbalance")
    if peer_signal_design and peer_signal_itt_lift <= 0.02:
        caveats.append("weak_peer_signal_intent_to_treat_lift")
    if peer_signal_design and peer_signal_trial_count < 5:
        caveats.append("insufficient_peer_signal_trial_actors")
    if peer_signal_design and inference.p_value_one_sided > 0.05:
        caveats.append("peer_signal_randomization_not_significant")
    caveats.extend(inference.caveats)
    if peer_signal_design and peer_signal_payment_itt_lift <= 0.02:
        caveats.append("weak_peer_signal_payment_intent_to_treat_lift")
    if peer_signal_design and peer_signal_payment_count < 5:
        caveats.append("insufficient_peer_signal_payment_actors")
    if peer_signal_design and payment_inference.p_value_one_sided > 0.05:
        caveats.append("peer_signal_payment_randomization_not_significant")
    if "underpowered_peer_signal_contrast" in payment_inference.caveats:
        caveats.append("underpowered_peer_signal_payment_contrast")
    if peer_signal_design and peer_signal_early_trial_itt_lift <= 0.02:
        caveats.append("weak_peer_signal_early_trial_intent_to_treat_lift")
    if peer_signal_design and peer_signal_early_trial_count < 5:
        caveats.append("insufficient_peer_signal_early_trial_actors")
    if peer_signal_design and early_trial_inference.p_value_one_sided > 0.05:
        caveats.append("peer_signal_early_trial_randomization_not_significant")
    if "underpowered_peer_signal_contrast" in early_trial_inference.caveats:
        caveats.append("underpowered_peer_signal_early_trial_contrast")
    if peer_signal_design and peer_signal_early_payment_itt_lift <= 0.02:
        caveats.append("weak_peer_signal_early_payment_intent_to_treat_lift")
    if peer_signal_design and peer_signal_early_payment_count < 5:
        caveats.append("insufficient_peer_signal_early_payment_actors")
    if peer_signal_design and early_payment_inference.p_value_one_sided > 0.05:
        caveats.append("peer_signal_early_payment_randomization_not_significant")
    if "underpowered_peer_signal_contrast" in early_payment_inference.caveats:
        caveats.append("underpowered_peer_signal_early_payment_contrast")
    if network_diagnostic.network_exposure_count == 0:
        caveats.append("no_network_exposure_observed")
    if network_diagnostic.network_exposure_count < 5:
        caveats.append("insufficient_network_exposure")
    if network_diagnostic.network_trial_actor_count < 5:
        caveats.append("insufficient_network_exposed_trial_actors")
    if len(trial_ids) < 5:
        caveats.append("few_trial_actors")
    if max_abs_smd > 0.25:
        caveats.append("covariate_imbalance_after_matching")
    if len(matched_ids) < len(trial_ids):
        caveats.append("insufficient_nontrial_matches")
    return ArtifactCovariateControlReport(
        artifact_id=artifact_id,
        exposure_count=len(exposed),
        trial_actor_count=len(trial_ids),
        payment_actor_count=len(payment_ids),
        matched_control_count=len(matched_ids),
        trial_rate=len(trial_ids) / max(1, len(exposed)),
        payment_rate=len(payment_ids) / max(1, len(exposed)),
        matched_mean_distance=matched_distance,
        max_abs_standardized_mean_difference=max_abs_smd,
        standardized_mean_differences=smds,
        release_exposure_count=network_diagnostic.release_exposure_count,
        network_exposure_count=network_diagnostic.network_exposure_count,
        unknown_exposure_count=network_diagnostic.unknown_exposure_count,
        network_trial_actor_count=network_diagnostic.network_trial_actor_count,
        network_payment_actor_count=network_payment_count,
        mean_network_distance=network_diagnostic.mean_network_distance,
        exposure_source_counts=network_diagnostic.source_counts,
        randomized_control_design=randomized_control_design,
        network_eligible_count=len(eligible_ids),
        network_holdout_count=len(holdout_ids),
        network_eligible_trial_rate=eligible_trial_rate,
        network_holdout_trial_rate=holdout_trial_rate,
        network_intent_to_treat_lift=network_itt_lift,
        randomized_group_max_abs_smd=randomized_group_max_abs_smd,
        network_diffusion_claim_allowed=network_diffusion_claim_allowed,
        peer_signal_design=peer_signal_design,
        peer_signal_eligible_count=len(peer_signal_ids),
        access_only_control_count=len(access_only_ids),
        peer_signal_trial_rate=peer_signal_trial_rate,
        access_only_trial_rate=access_only_trial_rate,
        peer_signal_intent_to_treat_lift=peer_signal_itt_lift,
        peer_signal_payment_rate=peer_signal_payment_rate,
        access_only_payment_rate=access_only_payment_rate,
        peer_signal_payment_intent_to_treat_lift=peer_signal_payment_itt_lift,
        peer_signal_group_max_abs_smd=peer_signal_group_max_abs_smd,
        peer_signal_trial_actor_count=peer_signal_trial_count,
        peer_signal_payment_actor_count=peer_signal_payment_count,
        peer_signal_randomization_p_value=inference.p_value_one_sided,
        peer_signal_confidence_low=inference.confidence_low,
        peer_signal_confidence_high=inference.confidence_high,
        peer_signal_power_proxy=inference.power_proxy,
        peer_signal_payment_randomization_p_value=payment_inference.p_value_one_sided,
        peer_signal_payment_confidence_low=payment_inference.confidence_low,
        peer_signal_payment_confidence_high=payment_inference.confidence_high,
        peer_signal_payment_power_proxy=payment_inference.power_proxy,
        early_window_ticks=PEER_EARLY_WINDOW_TICKS,
        peer_signal_early_trial_rate=peer_signal_early_trial_rate,
        access_only_early_trial_rate=access_only_early_trial_rate,
        peer_signal_early_trial_intent_to_treat_lift=peer_signal_early_trial_itt_lift,
        peer_signal_early_payment_rate=peer_signal_early_payment_rate,
        access_only_early_payment_rate=access_only_early_payment_rate,
        peer_signal_early_payment_intent_to_treat_lift=peer_signal_early_payment_itt_lift,
        peer_signal_early_trial_actor_count=peer_signal_early_trial_count,
        peer_signal_early_payment_actor_count=peer_signal_early_payment_count,
        peer_signal_early_trial_randomization_p_value=early_trial_inference.p_value_one_sided,
        peer_signal_early_trial_confidence_low=early_trial_inference.confidence_low,
        peer_signal_early_trial_confidence_high=early_trial_inference.confidence_high,
        peer_signal_early_trial_power_proxy=early_trial_inference.power_proxy,
        peer_signal_early_payment_randomization_p_value=early_payment_inference.p_value_one_sided,
        peer_signal_early_payment_confidence_low=early_payment_inference.confidence_low,
        peer_signal_early_payment_confidence_high=early_payment_inference.confidence_high,
        peer_signal_early_payment_power_proxy=early_payment_inference.power_proxy,
        peer_signal_mean_first_trial_tick=peer_signal_mean_first_trial_tick,
        access_only_mean_first_trial_tick=access_only_mean_first_trial_tick,
        peer_signal_mean_first_payment_tick=peer_signal_mean_first_payment_tick,
        access_only_mean_first_payment_tick=access_only_mean_first_payment_tick,
        peer_early_trial_claim_allowed=peer_early_trial_claim_allowed,
        peer_early_payment_claim_allowed=peer_early_payment_claim_allowed,
        peer_trial_claim_allowed=peer_trial_claim_allowed,
        peer_payment_claim_allowed=peer_payment_claim_allowed,
        peer_influence_claim_allowed=peer_influence_claim_allowed,
        influence_claim_allowed=influence_claim_allowed,
        caveats=tuple(caveats),
    )


def build_arm_control_report(
    *,
    arm_id: str,
    runtimes,
    initial_covariates_by_seed: dict[int, dict[str, dict[str, float]]],
) -> ArmControlReport:
    artifact_reports: dict[str, list[ArtifactCovariateControlReport]] = {}
    for runtime in runtimes:
        covariates = initial_covariates_by_seed[runtime.config.seed]
        for artifact_id in sorted(runtime.state.artifacts):
            report = build_artifact_covariate_control_report(
                state=runtime.state,
                initial_covariates=covariates,
                artifact_id=artifact_id,
            )
            artifact_reports.setdefault(artifact_id, []).append(report)
    return build_arm_control_report_from_seed_reports(
        arm_id=arm_id,
        seed_artifact_reports=tuple(artifact_reports.values()),
    )


def build_arm_control_report_from_seed_reports(
    *,
    arm_id: str,
    seed_artifact_reports: tuple[dict[str, ArtifactCovariateControlReport], ...] | tuple[list[ArtifactCovariateControlReport], ...],
) -> ArmControlReport:
    artifact_reports: dict[str, list[ArtifactCovariateControlReport]] = {}
    for reports_by_artifact in seed_artifact_reports:
        if isinstance(reports_by_artifact, dict):
            reports_iterable = reports_by_artifact.values()
        else:
            reports_iterable = reports_by_artifact
        for report in reports_iterable:
            artifact_reports.setdefault(report.artifact_id, []).append(report)
    summarized: dict[str, ArtifactCovariateControlReport] = {}
    for artifact_id, reports in artifact_reports.items():
        exposure_count = round(_mean([report.exposure_count for report in reports]))
        trial_actor_count = round(_mean([report.trial_actor_count for report in reports]))
        payment_actor_count = round(_mean([report.payment_actor_count for report in reports]))
        matched_control_count = round(_mean([report.matched_control_count for report in reports]))
        max_abs_smd = _mean([
            report.max_abs_standardized_mean_difference for report in reports
        ])
        release_exposure_count = round(_mean([report.release_exposure_count for report in reports]))
        network_exposure_count = round(_mean([report.network_exposure_count for report in reports]))
        unknown_exposure_count = round(_mean([report.unknown_exposure_count for report in reports]))
        network_trial_actor_count = round(_mean([report.network_trial_actor_count for report in reports]))
        network_payment_actor_count = round(_mean([report.network_payment_actor_count for report in reports]))
        randomized_control_design = all(report.randomized_control_design for report in reports)
        network_eligible_count = round(_mean([report.network_eligible_count for report in reports]))
        network_holdout_count = round(_mean([report.network_holdout_count for report in reports]))
        eligible_trial_rate = _mean([report.network_eligible_trial_rate for report in reports])
        holdout_trial_rate = _mean([report.network_holdout_trial_rate for report in reports])
        itt_lift = eligible_trial_rate - holdout_trial_rate
        randomized_group_max_abs_smd = _mean([
            report.randomized_group_max_abs_smd for report in reports
        ])
        caveats = tuple(sorted({caveat for report in reports for caveat in report.caveats}))
        network_diffusion_claim_allowed = (
            randomized_control_design
            and network_eligible_count >= 5
            and network_holdout_count >= 5
            and network_exposure_count >= 5
            and network_trial_actor_count >= 5
            and randomized_group_max_abs_smd <= 0.25
            and itt_lift > 0.02
            and not set(caveats).intersection(NETWORK_DIFFUSION_FATAL_CAVEATS)
        )
        peer_signal_design = all(report.peer_signal_design for report in reports)
        peer_signal_eligible_count = round(_mean([report.peer_signal_eligible_count for report in reports]))
        access_only_control_count = round(_mean([report.access_only_control_count for report in reports]))
        peer_signal_trial_rate = _mean([report.peer_signal_trial_rate for report in reports])
        access_only_trial_rate = _mean([report.access_only_trial_rate for report in reports])
        peer_signal_itt_lift = peer_signal_trial_rate - access_only_trial_rate
        peer_signal_payment_rate = _mean([report.peer_signal_payment_rate for report in reports])
        access_only_payment_rate = _mean([report.access_only_payment_rate for report in reports])
        peer_signal_payment_itt_lift = peer_signal_payment_rate - access_only_payment_rate
        peer_signal_early_trial_rate = _mean([
            report.peer_signal_early_trial_rate for report in reports
        ])
        access_only_early_trial_rate = _mean([
            report.access_only_early_trial_rate for report in reports
        ])
        peer_signal_early_trial_itt_lift = (
            peer_signal_early_trial_rate - access_only_early_trial_rate
        )
        peer_signal_early_payment_rate = _mean([
            report.peer_signal_early_payment_rate for report in reports
        ])
        access_only_early_payment_rate = _mean([
            report.access_only_early_payment_rate for report in reports
        ])
        peer_signal_early_payment_itt_lift = (
            peer_signal_early_payment_rate - access_only_early_payment_rate
        )
        peer_signal_group_max_abs_smd = _mean([
            report.peer_signal_group_max_abs_smd for report in reports
        ])
        peer_signal_trial_actor_count = round(_mean([
            report.peer_signal_trial_actor_count for report in reports
        ]))
        peer_signal_payment_actor_count = round(_mean([
            report.peer_signal_payment_actor_count for report in reports
        ]))
        peer_signal_early_trial_actor_count = round(_mean([
            report.peer_signal_early_trial_actor_count for report in reports
        ]))
        peer_signal_early_payment_actor_count = round(_mean([
            report.peer_signal_early_payment_actor_count for report in reports
        ]))
        peer_signal_randomization_p_value = max(
            report.peer_signal_randomization_p_value for report in reports
        )
        peer_signal_payment_randomization_p_value = max(
            report.peer_signal_payment_randomization_p_value for report in reports
        )
        peer_signal_early_trial_randomization_p_value = max(
            report.peer_signal_early_trial_randomization_p_value for report in reports
        )
        peer_signal_early_payment_randomization_p_value = max(
            report.peer_signal_early_payment_randomization_p_value for report in reports
        )
        peer_signal_confidence_low = _mean([
            report.peer_signal_confidence_low for report in reports
        ])
        peer_signal_confidence_high = _mean([
            report.peer_signal_confidence_high for report in reports
        ])
        peer_signal_power_proxy = _mean([
            report.peer_signal_power_proxy for report in reports
        ])
        peer_signal_payment_confidence_low = _mean([
            report.peer_signal_payment_confidence_low for report in reports
        ])
        peer_signal_payment_confidence_high = _mean([
            report.peer_signal_payment_confidence_high for report in reports
        ])
        peer_signal_payment_power_proxy = _mean([
            report.peer_signal_payment_power_proxy for report in reports
        ])
        peer_signal_early_trial_confidence_low = _mean([
            report.peer_signal_early_trial_confidence_low for report in reports
        ])
        peer_signal_early_trial_confidence_high = _mean([
            report.peer_signal_early_trial_confidence_high for report in reports
        ])
        peer_signal_early_trial_power_proxy = _mean([
            report.peer_signal_early_trial_power_proxy for report in reports
        ])
        peer_signal_early_payment_confidence_low = _mean([
            report.peer_signal_early_payment_confidence_low for report in reports
        ])
        peer_signal_early_payment_confidence_high = _mean([
            report.peer_signal_early_payment_confidence_high for report in reports
        ])
        peer_signal_early_payment_power_proxy = _mean([
            report.peer_signal_early_payment_power_proxy for report in reports
        ])
        peer_signal_mean_first_trial_tick = _mean_optional([
            report.peer_signal_mean_first_trial_tick for report in reports
        ])
        access_only_mean_first_trial_tick = _mean_optional([
            report.access_only_mean_first_trial_tick for report in reports
        ])
        peer_signal_mean_first_payment_tick = _mean_optional([
            report.peer_signal_mean_first_payment_tick for report in reports
        ])
        access_only_mean_first_payment_tick = _mean_optional([
            report.access_only_mean_first_payment_tick for report in reports
        ])
        peer_trial_claim_allowed = (
            network_diffusion_claim_allowed
            and peer_signal_design
            and peer_signal_eligible_count >= 5
            and access_only_control_count >= 5
            and peer_signal_trial_actor_count >= 5
            and peer_signal_group_max_abs_smd <= 0.25
            and peer_signal_itt_lift > 0.02
            and peer_signal_randomization_p_value <= 0.05
            and peer_signal_power_proxy >= 1.96
            and not set(caveats).intersection(PEER_INFLUENCE_FATAL_CAVEATS)
        )
        peer_payment_claim_allowed = (
            network_diffusion_claim_allowed
            and peer_signal_design
            and peer_signal_eligible_count >= 5
            and access_only_control_count >= 5
            and peer_signal_payment_actor_count >= 5
            and peer_signal_group_max_abs_smd <= 0.25
            and peer_signal_payment_itt_lift > 0.02
            and peer_signal_payment_randomization_p_value <= 0.05
            and peer_signal_payment_power_proxy >= 1.96
            and not set(caveats).intersection(PEER_INFLUENCE_FATAL_CAVEATS)
        )
        peer_early_trial_claim_allowed = (
            network_diffusion_claim_allowed
            and peer_signal_design
            and peer_signal_eligible_count >= 5
            and access_only_control_count >= 5
            and peer_signal_early_trial_actor_count >= 5
            and peer_signal_group_max_abs_smd <= 0.25
            and peer_signal_early_trial_itt_lift > 0.02
            and peer_signal_early_trial_randomization_p_value <= 0.05
            and peer_signal_early_trial_power_proxy >= 1.96
            and not set(caveats).intersection(PEER_INFLUENCE_FATAL_CAVEATS)
        )
        peer_early_payment_claim_allowed = (
            network_diffusion_claim_allowed
            and peer_signal_design
            and peer_signal_eligible_count >= 5
            and access_only_control_count >= 5
            and peer_signal_early_payment_actor_count >= 5
            and peer_signal_group_max_abs_smd <= 0.25
            and peer_signal_early_payment_itt_lift > 0.02
            and peer_signal_early_payment_randomization_p_value <= 0.05
            and peer_signal_early_payment_power_proxy >= 1.96
            and not set(caveats).intersection(PEER_INFLUENCE_FATAL_CAVEATS)
        )
        peer_influence_claim_allowed = (
            peer_trial_claim_allowed
            or peer_payment_claim_allowed
            or peer_early_trial_claim_allowed
            or peer_early_payment_claim_allowed
        )
        influence_claim_allowed = peer_influence_claim_allowed
        summarized[artifact_id] = ArtifactCovariateControlReport(
            artifact_id=artifact_id,
            exposure_count=exposure_count,
            trial_actor_count=trial_actor_count,
            payment_actor_count=payment_actor_count,
            matched_control_count=matched_control_count,
            trial_rate=_mean([report.trial_rate for report in reports]),
            payment_rate=_mean([report.payment_rate for report in reports]),
            matched_mean_distance=_mean([report.matched_mean_distance for report in reports]),
            max_abs_standardized_mean_difference=max_abs_smd,
            standardized_mean_differences={
                name: _mean([report.standardized_mean_differences[name] for report in reports])
                for name in COVARIATE_NAMES
            },
            release_exposure_count=release_exposure_count,
            network_exposure_count=network_exposure_count,
            unknown_exposure_count=unknown_exposure_count,
            network_trial_actor_count=network_trial_actor_count,
            network_payment_actor_count=network_payment_actor_count,
            mean_network_distance=_mean([report.mean_network_distance for report in reports]),
            exposure_source_counts={
                source: round(_mean([
                    report.exposure_source_counts.get(source, 0)
                    for report in reports
                ]))
                for source in sorted({
                    source for report in reports for source in report.exposure_source_counts
                })
            },
            randomized_control_design=randomized_control_design,
            network_eligible_count=network_eligible_count,
            network_holdout_count=network_holdout_count,
            network_eligible_trial_rate=eligible_trial_rate,
            network_holdout_trial_rate=holdout_trial_rate,
            network_intent_to_treat_lift=itt_lift,
            randomized_group_max_abs_smd=randomized_group_max_abs_smd,
            network_diffusion_claim_allowed=network_diffusion_claim_allowed,
            peer_signal_design=peer_signal_design,
            peer_signal_eligible_count=peer_signal_eligible_count,
            access_only_control_count=access_only_control_count,
            peer_signal_trial_rate=peer_signal_trial_rate,
            access_only_trial_rate=access_only_trial_rate,
            peer_signal_intent_to_treat_lift=peer_signal_itt_lift,
            peer_signal_payment_rate=peer_signal_payment_rate,
            access_only_payment_rate=access_only_payment_rate,
            peer_signal_payment_intent_to_treat_lift=peer_signal_payment_itt_lift,
            peer_signal_group_max_abs_smd=peer_signal_group_max_abs_smd,
            peer_signal_trial_actor_count=peer_signal_trial_actor_count,
            peer_signal_payment_actor_count=peer_signal_payment_actor_count,
            peer_signal_randomization_p_value=peer_signal_randomization_p_value,
            peer_signal_confidence_low=peer_signal_confidence_low,
            peer_signal_confidence_high=peer_signal_confidence_high,
            peer_signal_power_proxy=peer_signal_power_proxy,
            peer_signal_payment_randomization_p_value=peer_signal_payment_randomization_p_value,
            peer_signal_payment_confidence_low=peer_signal_payment_confidence_low,
            peer_signal_payment_confidence_high=peer_signal_payment_confidence_high,
            peer_signal_payment_power_proxy=peer_signal_payment_power_proxy,
            early_window_ticks=PEER_EARLY_WINDOW_TICKS,
            peer_signal_early_trial_rate=peer_signal_early_trial_rate,
            access_only_early_trial_rate=access_only_early_trial_rate,
            peer_signal_early_trial_intent_to_treat_lift=peer_signal_early_trial_itt_lift,
            peer_signal_early_payment_rate=peer_signal_early_payment_rate,
            access_only_early_payment_rate=access_only_early_payment_rate,
            peer_signal_early_payment_intent_to_treat_lift=peer_signal_early_payment_itt_lift,
            peer_signal_early_trial_actor_count=peer_signal_early_trial_actor_count,
            peer_signal_early_payment_actor_count=peer_signal_early_payment_actor_count,
            peer_signal_early_trial_randomization_p_value=peer_signal_early_trial_randomization_p_value,
            peer_signal_early_trial_confidence_low=peer_signal_early_trial_confidence_low,
            peer_signal_early_trial_confidence_high=peer_signal_early_trial_confidence_high,
            peer_signal_early_trial_power_proxy=peer_signal_early_trial_power_proxy,
            peer_signal_early_payment_randomization_p_value=peer_signal_early_payment_randomization_p_value,
            peer_signal_early_payment_confidence_low=peer_signal_early_payment_confidence_low,
            peer_signal_early_payment_confidence_high=peer_signal_early_payment_confidence_high,
            peer_signal_early_payment_power_proxy=peer_signal_early_payment_power_proxy,
            peer_signal_mean_first_trial_tick=peer_signal_mean_first_trial_tick,
            access_only_mean_first_trial_tick=access_only_mean_first_trial_tick,
            peer_signal_mean_first_payment_tick=peer_signal_mean_first_payment_tick,
            access_only_mean_first_payment_tick=access_only_mean_first_payment_tick,
            peer_early_trial_claim_allowed=peer_early_trial_claim_allowed,
            peer_early_payment_claim_allowed=peer_early_payment_claim_allowed,
            peer_trial_claim_allowed=peer_trial_claim_allowed,
            peer_payment_claim_allowed=peer_payment_claim_allowed,
            peer_influence_claim_allowed=peer_influence_claim_allowed,
            influence_claim_allowed=influence_claim_allowed,
            caveats=caveats,
        )

    network_diffusion_claim_allowed = any(report.network_diffusion_claim_allowed for report in summarized.values())
    peer_influence_claim_allowed = any(report.peer_influence_claim_allowed for report in summarized.values())
    network_caveats = tuple(sorted({
        caveat
        for report in summarized.values()
        for caveat in report.caveats
    })) or ("no_artifact_exposure",)
    return ArmControlReport(
        report_id="society_core_arm_control_report_v15",
        arm_id=arm_id,
        artifact_reports=summarized,
        network_diffusion_claim_allowed=network_diffusion_claim_allowed,
        peer_influence_claim_allowed=peer_influence_claim_allowed,
        network_influence_claim_allowed=peer_influence_claim_allowed,
        network_control_caveats=network_caveats,
    )
