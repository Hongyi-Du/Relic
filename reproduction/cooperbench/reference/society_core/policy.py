"""Primitive action availability and deterministic utility scoring."""

from __future__ import annotations

from math import exp
from typing import Mapping

from .product_experience import artifact_price
from .randomness import SeededRandom
from .schemas import Action, ActionKind, AgentState, Artifact


def available_actions(agent: AgentState) -> list[ActionKind]:
    actions = [
        ActionKind.EAT,
        ActionKind.REST,
        ActionKind.SLEEP,
        ActionKind.WORK_FOR_RESOURCE,
        ActionKind.READ,
        ActionKind.SEARCH,
        ActionKind.POST,
        ActionKind.IGNORE,
    ]
    if agent.artifact_access:
        actions.extend([
            ActionKind.INSPECT_ARTIFACT,
            ActionKind.TRY_ARTIFACT_ON_TASK,
            ActionKind.REUSE_ARTIFACT,
            ActionKind.PAY_FOR_ARTIFACT,
            ActionKind.ABANDON_ARTIFACT,
            ActionKind.SHARE_FAILURE,
            ActionKind.WRITE_TUTORIAL,
        ])
        if any(agent.artifact_usage_counts.get(artifact_id, 0) > 0 for artifact_id in agent.artifact_access):
            actions.append(ActionKind.SHARE_SUCCESS)
    if agent.cognition.working_memory_items:
        actions.extend([
            ActionKind.REPLY,
            ActionKind.REPOST,
            ActionKind.ASK_QUESTION,
            ActionKind.ASK_FOR_EXPLANATION,
            ActionKind.VERIFY,
            ActionKind.COMPARE_SOURCES,
            ActionKind.CHALLENGE_CLAIM,
            ActionKind.HELP_PEER,
            ActionKind.WARN_PEER,
            ActionKind.MEDIATE_CONFLICT,
            ActionKind.REDUCE_TRUST,
        ])
    return actions


def score_action(
    agent: AgentState,
    kind: ActionKind,
    artifacts: Mapping[str, Artifact] | None = None,
) -> float:
    body = agent.body
    budgets = agent.action_budgets
    resources = agent.material_resources
    affect = agent.affect
    cognition = agent.cognition
    read_drive = agent.profile.reading_propensity * agent.profile.daily_active_probability
    post_drive = agent.profile.posting_propensity * agent.profile.daily_active_probability
    comment_drive = agent.profile.commenting_propensity * agent.profile.daily_active_probability
    institution = agent.cognition.institution_feedback

    homeostatic_urgency = (
        2.0 * body.hunger
        + 1.5 * body.sleep_pressure
        + 1.2 * body.fatigue
        + 1.0 * body.stress
        + (1.0 - body.health_reserve)
    )
    low_attention_penalty = 1.0 - budgets.attention_budget

    if kind == ActionKind.EAT:
        return 3.0 * body.hunger + 0.8 * resources.food_or_budget_token - 0.1
    if kind == ActionKind.SLEEP:
        return 2.5 * body.sleep_pressure + 1.2 * body.fatigue - 0.2
    if kind == ActionKind.REST:
        return 1.4 * body.fatigue + 0.8 * body.stress + 0.4 * low_attention_penalty
    if kind == ActionKind.WORK_FOR_RESOURCE:
        return 1.2 * (1.0 - resources.food_or_budget_token) - 0.8 * homeostatic_urgency
    if kind == ActionKind.READ:
        return 0.6 * budgets.attention_budget + 0.4 * cognition.uncertainty + 0.45 * read_drive - 0.8 * body.fatigue
    if kind == ActionKind.SEARCH:
        return 0.5 * budgets.attention_budget + 0.6 * cognition.uncertainty + 0.18 * read_drive - 0.5 * body.stress
    if kind == ActionKind.POST:
        return (
            0.4 * budgets.attention_budget
            + 0.3 * affect.curiosity
            + 1.15 * post_drive
            + 0.22 * agent.profile.strategic_boldness
            - 0.7 * body.stress
        )
    if kind == ActionKind.REPLY:
        return (
            0.45 * budgets.attention_budget
            + 0.35 * affect.curiosity
            + 0.2 * affect.patience
            + 1.05 * comment_drive
            - 0.5 * body.stress
        )
    if kind == ActionKind.REPOST:
        return 0.35 * budgets.attention_budget + 0.45 * affect.curiosity + 0.85 * post_drive - 0.35 * body.stress
    if kind == ActionKind.ASK_QUESTION:
        return (
            0.45 * budgets.attention_budget
            + 0.75 * cognition.uncertainty
            + 0.35 * affect.curiosity
            + 0.65 * comment_drive
            - 0.35 * body.fatigue
        )
    if kind == ActionKind.ASK_FOR_EXPLANATION:
        return (
            0.4 * budgets.attention_budget
            + 0.65 * cognition.uncertainty
            + 0.25 * affect.patience
            + 0.15 * affect.anger
            + 0.28 * institution.get("explanation_expectation", 0.0)
        )
    if kind == ActionKind.VERIFY:
        return (
            0.45 * budgets.attention_budget
            + 0.7 * cognition.uncertainty
            + 0.35 * affect.patience
            + 0.38 * institution.get("verify_expectation", 0.0)
            - 0.25 * body.fatigue
        )
    if kind == ActionKind.COMPARE_SOURCES:
        return 0.5 * budgets.attention_budget + 0.55 * cognition.uncertainty + 0.25 * affect.curiosity - 0.25 * body.stress
    if kind == ActionKind.CHALLENGE_CLAIM:
        return (
            0.35 * budgets.attention_budget
            + 0.55 * cognition.uncertainty
            + 0.35 * affect.anger
            + 0.20 * agent.profile.strategic_boldness
            + 0.18 * institution.get("callout_social_cost", 0.0)
            - 0.2 * affect.fear
        )
    if kind == ActionKind.HELP_PEER:
        return (
            0.35 * budgets.attention_budget
            + 0.35 * affect.patience
            + 0.25 * affect.curiosity
            + 0.34 * institution.get("help_role_salience", 0.0)
            - 0.25 * body.fatigue
        )
    if kind == ActionKind.WARN_PEER:
        return 0.25 * budgets.attention_budget + 0.45 * affect.fear + 0.3 * affect.anger + 0.25 * cognition.uncertainty
    if kind == ActionKind.MEDIATE_CONFLICT:
        return 0.3 * budgets.attention_budget + 0.55 * affect.patience + 0.25 * cognition.uncertainty - 0.35 * affect.anger
    if kind == ActionKind.REDUCE_TRUST:
        return (
            0.2 * budgets.attention_budget
            + 0.55 * affect.anger
            + 0.35 * affect.fear
            + 0.30 * institution.get("trust_sanction_salience", 0.0)
            - 0.35 * affect.patience
        )
    artifact_pull = best_accessible_artifact_score(agent, artifacts) - 0.55
    artifact_positive_signal = best_accessible_artifact_peer_signal(agent)
    artifact_negative_signal = best_accessible_artifact_negative_signal(agent)
    artifact_social_proof = best_accessible_artifact_social_proof(agent)
    artifact_untried_peer_pressure = best_accessible_artifact_untried_peer_pressure(agent)
    artifact_try_intent = best_accessible_artifact_intent(agent, "intent_to_try")
    artifact_pay_intent = best_accessible_artifact_intent(agent, "intent_to_pay")
    artifact_recommend_intent = best_accessible_artifact_intent(agent, "intent_to_recommend")
    artifact_satisfaction = best_accessible_artifact_intent(agent, "subjective_satisfaction")
    artifact_wtp = best_accessible_artifact_intent(agent, "max_willingness_to_pay")
    artifact_experience_success = best_accessible_artifact_experience(agent, "objective_success")
    artifact_has_experience = best_accessible_artifact_has_experience(agent)
    payment_pressure = max(0.0, 0.35 - resources.food_or_budget_token)
    if kind == ActionKind.INSPECT_ARTIFACT:
        return (
            0.8 * affect.curiosity
            + 0.4 * budgets.attention_budget
            + 0.65 * artifact_pull
            + 0.18 * agent.profile.domain_need
            + 0.14 * agent.profile.novelty_seeking
            + 0.3 * artifact_positive_signal
            + 0.22 * artifact_social_proof
            - 0.35 * artifact_negative_signal
            - 0.6 * homeostatic_urgency
        )
    if kind == ActionKind.TRY_ARTIFACT_ON_TASK:
        return (
            0.7 * resources.artifact_access
            + 0.6 * affect.curiosity
            + 0.9 * artifact_pull
            + 0.78 * artifact_positive_signal
            + 0.42 * artifact_untried_peer_pressure
            + 0.55 * artifact_try_intent
            + 0.28 * agent.profile.domain_need
            + 0.16 * agent.profile.risk_tolerance
            + 0.12 * agent.profile.strategic_boldness
            - 0.5 * artifact_negative_signal
            - 0.7 * homeostatic_urgency
        )
    if kind == ActionKind.REUSE_ARTIFACT:
        return (
            0.5 * resources.artifact_access
            + 0.95 * artifact_pull
            + 0.82 * artifact_positive_signal
            + 0.28 * artifact_social_proof
            + 0.38 * artifact_satisfaction
            + 0.28 * artifact_experience_success
            - 0.55 * artifact_negative_signal
            - 0.55 * homeostatic_urgency
        )
    if kind == ActionKind.PAY_FOR_ARTIFACT:
        return (
            0.55 * resources.food_or_budget_token
            + 1.1 * artifact_pull
            + 1.08 * artifact_positive_signal
            + 0.48 * artifact_social_proof
            + 1.25 * artifact_pay_intent
            + 0.72 * artifact_satisfaction
            + 0.66 * artifact_wtp
            + 0.22 * artifact_experience_success
            + 0.16 * agent.profile.risk_tolerance
            - 1.0 * artifact_negative_signal
            - 1.1 * payment_pressure
            - 0.72 * (1.0 - artifact_has_experience)
            - 0.32 * agent.profile.budget_sensitivity
            - 0.55 * homeostatic_urgency
        )
    if kind == ActionKind.ABANDON_ARTIFACT:
        return (
            0.15
            + 0.85 * artifact_negative_signal
            + 0.35 * body.stress
            + 0.25 * body.fatigue
            + 0.22 * (1.0 - artifact_satisfaction)
            - 0.75 * artifact_pull
            - 0.35 * artifact_positive_signal
        )
    if kind == ActionKind.SHARE_FAILURE:
        return (
            0.1
            + 0.75 * artifact_negative_signal
            + 0.35 * affect.anger
            + 0.25 * budgets.attention_budget
            + 0.85 * post_drive
            - 0.3 * affect.patience
        )
    if kind == ActionKind.SHARE_SUCCESS:
        usage = max((agent.artifact_usage_counts.get(artifact_id, 0) for artifact_id in agent.artifact_access), default=0)
        return (
            0.08
            + 0.35 * artifact_positive_signal
            + 0.22 * min(5, usage)
            + 0.25 * budgets.attention_budget
            + 0.18 * affect.curiosity
            + 0.15 * affect.mood
            + 0.25 * artifact_recommend_intent
            + 0.14 * agent.profile.social_activity
            + 0.12 * agent.profile.strategic_boldness
            + 0.75 * post_drive
            - 0.2 * artifact_negative_signal
            - 0.25 * body.fatigue
        )
    if kind == ActionKind.WRITE_TUTORIAL:
        usage = max((agent.artifact_usage_counts.get(artifact_id, 0) for artifact_id in agent.artifact_access), default=0)
        return (
            0.2
            + 0.25 * budgets.attention_budget
            + 0.3 * affect.curiosity
            + 0.12 * min(5, usage)
            + 0.35 * artifact_positive_signal
            + 0.55 * post_drive
            + 0.18 * agent.profile.strategic_boldness
            - 0.25 * artifact_negative_signal
            - 0.35 * body.fatigue
        )
    if kind == ActionKind.IGNORE:
        return 0.2 + 0.6 * low_attention_penalty + 0.2 * homeostatic_urgency
    return 0.0


def _mean(values) -> float:
    vals = [float(value) for value in values]
    return sum(vals) / len(vals) if vals else 0.0


def score_artifact(agent: AgentState, artifact: Artifact | None) -> float:
    if artifact is None:
        return 1.0
    capability = _mean(artifact.capability_profile.values()) or 0.5
    reliability = _mean(artifact.reliability_profile.values()) or capability
    task_fit = _mean(artifact.task_fit_distribution.values()) or capability
    cost = _mean(artifact.cost_profile.values()) or 0.1
    belief = agent.cognition.beliefs_about_artifacts.get(artifact.id, 0.5)
    peer_signal = agent.cognition.artifact_peer_signal_strength.get(artifact.id, 0.0)
    negative_signal = agent.cognition.artifact_negative_signal_strength.get(artifact.id, 0.0)
    signal_count = agent.cognition.artifact_peer_signal_count.get(artifact.id, 0)
    repetition = min(1.0, signal_count / 5.0)
    social_proof = peer_signal * (0.7 + 0.3 * repetition)
    usage_count = agent.artifact_usage_counts.get(artifact.id, 0)
    habit = min(0.2, 0.04 * usage_count)
    intent = agent.latest_llm_intents.get(artifact.id)
    subjective_satisfaction = intent.subjective_satisfaction if intent else 0.0
    max_willingness_to_pay = intent.max_willingness_to_pay if intent else 0.0
    experience = agent.latest_artifact_experience.get(artifact.id)
    objective_success = experience.objective_success if experience else 0.0
    privacy_cost = artifact.access_constraints.get("data", 0.0)
    score = (
        0.15
        + 0.35 * capability
        + 0.25 * reliability
        + 0.25 * task_fit
        + 0.2 * belief
        + 0.24 * peer_signal
        + 0.16 * social_proof
        + habit
        + 0.14 * agent.profile.domain_need
        + 0.08 * agent.profile.novelty_seeking
        + 0.10 * subjective_satisfaction
        + 0.08 * max_willingness_to_pay
        + 0.06 * objective_success
        + 0.15 * agent.affect.curiosity
        - 0.35 * negative_signal
        - 0.35 * cost
        - 0.12 * agent.profile.budget_sensitivity * artifact_price(artifact)
        - 0.08 * agent.profile.privacy_sensitivity * privacy_cost
    )
    return max(0.001, score)


def best_accessible_artifact_peer_signal(agent: AgentState) -> float:
    if not agent.artifact_access:
        return 0.0
    return max(
        (
            agent.cognition.artifact_peer_signal_strength.get(artifact_id, 0.0)
            for artifact_id in agent.artifact_access
        ),
        default=0.0,
    )


def best_accessible_artifact_negative_signal(agent: AgentState) -> float:
    if not agent.artifact_access:
        return 0.0
    return max(
        (
            agent.cognition.artifact_negative_signal_strength.get(artifact_id, 0.0)
            for artifact_id in agent.artifact_access
        ),
        default=0.0,
    )


def best_accessible_artifact_social_proof(agent: AgentState) -> float:
    if not agent.artifact_access:
        return 0.0
    return max(
        (
            agent.cognition.artifact_peer_signal_strength.get(artifact_id, 0.0)
            * (0.7 + 0.3 * min(1.0, agent.cognition.artifact_peer_signal_count.get(artifact_id, 0) / 5.0))
            for artifact_id in agent.artifact_access
        ),
        default=0.0,
    )


def best_accessible_artifact_untried_peer_pressure(agent: AgentState) -> float:
    if not agent.artifact_access:
        return 0.0
    return max(
        (
            agent.cognition.artifact_peer_signal_strength.get(artifact_id, 0.0)
            * (1.0 - min(1.0, agent.artifact_usage_counts.get(artifact_id, 0)))
            * (0.65 + 0.35 * min(1.0, agent.cognition.artifact_peer_signal_count.get(artifact_id, 0) / 3.0))
            for artifact_id in agent.artifact_access
        ),
        default=0.0,
    )


def best_accessible_artifact_intent(agent: AgentState, field_name: str) -> float:
    if not agent.artifact_access:
        return 0.0
    return max(
        (
            float(getattr(agent.latest_llm_intents[artifact_id], field_name))
            for artifact_id in agent.artifact_access
            if artifact_id in agent.latest_llm_intents
        ),
        default=0.0,
    )


def best_accessible_artifact_experience(agent: AgentState, field_name: str) -> float:
    if not agent.artifact_access:
        return 0.0
    return max(
        (
            float(getattr(agent.latest_artifact_experience[artifact_id], field_name))
            for artifact_id in agent.artifact_access
            if artifact_id in agent.latest_artifact_experience
        ),
        default=0.0,
    )


def best_accessible_artifact_has_experience(agent: AgentState) -> float:
    if not agent.artifact_access:
        return 0.0
    return 1.0 if any(artifact_id in agent.latest_artifact_experience for artifact_id in agent.artifact_access) else 0.0


def best_accessible_artifact_score(
    agent: AgentState,
    artifacts: Mapping[str, Artifact] | None = None,
) -> float:
    if not agent.artifact_access:
        return 0.0
    if artifacts is None:
        return 1.0
    scores = [
        score_artifact(agent, artifacts[artifact_id])
        for artifact_id in agent.artifact_access
        if artifact_id in artifacts
    ]
    return max(scores, default=0.0)


def select_artifact_id(
    agent: AgentState,
    rng: SeededRandom,
    artifacts: Mapping[str, Artifact] | None = None,
) -> str | None:
    if not agent.artifact_access:
        return None
    artifact_ids = tuple(sorted(agent.artifact_access))
    weights = [score_artifact(agent, artifacts.get(artifact_id) if artifacts else None) for artifact_id in artifact_ids]
    return artifact_ids[rng.choice_index("action", weights, kind="artifact_choice")]


def choose_action(
    agent: AgentState,
    rng: SeededRandom,
    artifacts: Mapping[str, Artifact] | None = None,
) -> Action:
    candidates = available_actions(agent)
    utilities = [score_action(agent, kind, artifacts) for kind in candidates]
    max_u = max(utilities)
    # Fatigue/stress increase randomness, habit can later reduce it.
    temperature = 0.25 + 0.6 * agent.body.stress + 0.4 * agent.body.fatigue + 0.2 * agent.cognition.uncertainty
    weights = [exp((utility - max_u) / max(temperature, 0.05)) for utility in utilities]
    index = rng.choice_index("action", weights, kind="action_softmax")
    kind = candidates[index]
    artifact_id = None
    if kind in {
        ActionKind.INSPECT_ARTIFACT,
        ActionKind.TRY_ARTIFACT_ON_TASK,
        ActionKind.REUSE_ARTIFACT,
        ActionKind.PAY_FOR_ARTIFACT,
        ActionKind.ABANDON_ARTIFACT,
        ActionKind.SHARE_FAILURE,
        ActionKind.SHARE_SUCCESS,
        ActionKind.WRITE_TUTORIAL,
    }:
        artifact_id = select_artifact_id(agent, rng, artifacts)
    channel_id = "public_forum" if kind in {
        ActionKind.POST,
        ActionKind.REPLY,
        ActionKind.REPOST,
        ActionKind.ASK_QUESTION,
        ActionKind.ASK_FOR_EXPLANATION,
        ActionKind.VERIFY,
        ActionKind.COMPARE_SOURCES,
        ActionKind.CHALLENGE_CLAIM,
        ActionKind.HELP_PEER,
        ActionKind.WARN_PEER,
        ActionKind.MEDIATE_CONFLICT,
        ActionKind.REDUCE_TRUST,
        ActionKind.SHARE_FAILURE,
        ActionKind.SHARE_SUCCESS,
        ActionKind.WRITE_TUTORIAL,
    } else None
    return Action(actor_id=agent.id, kind=kind, artifact_id=artifact_id, channel_id=channel_id)
