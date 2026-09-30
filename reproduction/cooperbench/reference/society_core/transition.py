"""Typed transition functions for primitive actions."""

from __future__ import annotations

from typing import Mapping

from .llm_intent import feedback_intent_record, payment_block_reason, store_intent_feedback
from .product_experience import apply_product_experience, artifact_price
from .psychology import apply_body_affect_coupling
from .schemas import Action, ActionKind, AgentState, Artifact, BodyState, EventKind, ExternalEvent, ProductExperiencePacket, clamp01


def drift_body(agent: AgentState, hunger_drift: float = 0.03, *, coupling_enabled: bool = True) -> None:
    body = agent.body
    agent.body = BodyState(
        hunger=clamp01(body.hunger + hunger_drift),
        sleep_pressure=clamp01(body.sleep_pressure + 0.015),
        fatigue=clamp01(body.fatigue + 0.01),
        stress=clamp01(body.stress * 0.995),
        health_reserve=clamp01(body.health_reserve - 0.002 * body.stress),
        physiological_energy=clamp01(body.physiological_energy - 0.01 - 0.01 * body.hunger),
    )
    if coupling_enabled:
        apply_body_affect_coupling(agent)


def apply_action_effect(
    agent: AgentState,
    action: Action,
    *,
    artifacts: Mapping[str, Artifact] | None = None,
    coupling_enabled: bool = True,
) -> dict[str, float]:
    body = agent.body
    resources = agent.material_resources
    budgets = agent.action_budgets
    deltas: dict[str, float] = {}

    if action.kind == ActionKind.EAT:
        spend = min(0.2, resources.food_or_budget_token)
        resources.food_or_budget_token = max(0.0, resources.food_or_budget_token - spend)
        body.hunger = clamp01(body.hunger - 0.45 * (spend / 0.2 if spend else 0.0))
        body.physiological_energy = clamp01(body.physiological_energy + 0.12)
        budgets.time_budget = clamp01(budgets.time_budget - 0.04)
        deltas["hunger"] = -0.45
    elif action.kind == ActionKind.SLEEP:
        body.sleep_pressure = clamp01(body.sleep_pressure - 0.55)
        body.fatigue = clamp01(body.fatigue - 0.35)
        body.physiological_energy = clamp01(body.physiological_energy + 0.35)
        budgets.attention_budget = clamp01(budgets.attention_budget + 0.25)
        budgets.time_budget = clamp01(budgets.time_budget - 0.25)
        deltas["sleep_pressure"] = -0.55
    elif action.kind == ActionKind.REST:
        body.fatigue = clamp01(body.fatigue - 0.18)
        body.stress = clamp01(body.stress - 0.12)
        budgets.attention_budget = clamp01(budgets.attention_budget + 0.14)
        budgets.time_budget = clamp01(budgets.time_budget - 0.08)
    elif action.kind == ActionKind.WORK_FOR_RESOURCE:
        resources.food_or_budget_token = min(1.0, resources.food_or_budget_token + 0.12)
        budgets.time_budget = clamp01(budgets.time_budget - 0.18)
        budgets.attention_budget = clamp01(budgets.attention_budget - 0.08)
        body.fatigue = clamp01(body.fatigue + 0.08)
        body.stress = clamp01(body.stress + 0.04)
    elif action.kind in {ActionKind.READ, ActionKind.SEARCH, ActionKind.INSPECT_ARTIFACT}:
        budgets.attention_budget = clamp01(budgets.attention_budget - 0.04)
        budgets.time_budget = clamp01(budgets.time_budget - 0.04)
    elif action.kind == ActionKind.TRY_ARTIFACT_ON_TASK:
        packet = _experience_packet_from_action(action)
        if action.artifact_id:
            agent.artifact_usage_counts[action.artifact_id] = agent.artifact_usage_counts.get(action.artifact_id, 0) + 1
        attention_cost = packet.attention_cost if packet else 0.10
        time_cost = 0.08 + 0.08 * (packet.friction if packet else 0.25)
        budgets.attention_budget = clamp01(budgets.attention_budget - attention_cost)
        budgets.time_budget = clamp01(budgets.time_budget - time_cost)
        body.fatigue = clamp01(body.fatigue + 0.02 + 0.04 * (packet.friction if packet else 0.25))
        if packet:
            deltas.update(apply_product_experience(agent, packet))
    elif action.kind == ActionKind.REUSE_ARTIFACT:
        packet = _experience_packet_from_action(action)
        if action.artifact_id:
            agent.artifact_usage_counts[action.artifact_id] = agent.artifact_usage_counts.get(action.artifact_id, 0) + 1
            belief = agent.cognition.beliefs_about_artifacts.get(action.artifact_id, 0.5)
            agent.cognition.beliefs_about_artifacts[action.artifact_id] = clamp01(belief + 0.03 * (1.0 - belief))
        attention_cost = 0.75 * packet.attention_cost if packet else 0.08
        time_cost = 0.06 + 0.06 * (packet.friction if packet else 0.25)
        budgets.attention_budget = clamp01(budgets.attention_budget - attention_cost)
        budgets.time_budget = clamp01(budgets.time_budget - time_cost)
        body.fatigue = clamp01(body.fatigue + 0.015 + 0.03 * (packet.friction if packet else 0.25))
        if packet:
            deltas.update(apply_product_experience(agent, packet))
    elif action.kind == ActionKind.PAY_FOR_ARTIFACT:
        if action.artifact_id:
            artifact = artifacts.get(action.artifact_id) if artifacts else None
            tick = int(action.typed_payload.get("tick", 0))
            blocked_reason = payment_block_reason(agent, artifact)
            feedback = feedback_intent_record(
                agent=agent,
                artifact_id=action.artifact_id,
                tick=tick,
                blocked_reason=blocked_reason,
            )
            store_intent_feedback(agent, feedback)
            if blocked_reason:
                agent.artifact_payment_state[action.artifact_id] = f"payment_blocked:{blocked_reason}"
                deltas["payment_blocked"] = 1.0
            else:
                spend = artifact_price(artifact)
                resources.food_or_budget_token = clamp01(resources.food_or_budget_token - spend)
                agent.artifact_payment_state[action.artifact_id] = "paid"
                agent.artifact_payment_ticks[action.artifact_id] = tick
                deltas["food_or_budget_token"] = -spend
                deltas["payment_executed"] = 1.0
        budgets.attention_budget = clamp01(budgets.attention_budget - 0.04)
        budgets.time_budget = clamp01(budgets.time_budget - 0.04)
    elif action.kind == ActionKind.ABANDON_ARTIFACT:
        if action.artifact_id:
            agent.artifact_payment_state[action.artifact_id] = "abandoned"
            agent.cognition.artifact_negative_signal_strength[action.artifact_id] = clamp01(
                agent.cognition.artifact_negative_signal_strength.get(action.artifact_id, 0.0) + 0.12
            )
        budgets.attention_budget = clamp01(budgets.attention_budget - 0.03)
    elif action.kind in {
        ActionKind.ASK_QUESTION,
        ActionKind.ASK_FOR_EXPLANATION,
        ActionKind.VERIFY,
        ActionKind.COMPARE_SOURCES,
    }:
        budgets.attention_budget = clamp01(budgets.attention_budget - 0.07)
        budgets.time_budget = clamp01(budgets.time_budget - 0.06)
        agent.cognition.uncertainty = clamp01(agent.cognition.uncertainty - 0.015)
    elif action.kind == ActionKind.CHALLENGE_CLAIM:
        budgets.attention_budget = clamp01(budgets.attention_budget - 0.08)
        budgets.time_budget = clamp01(budgets.time_budget - 0.06)
        body.stress = clamp01(body.stress + 0.015)
        agent.affect.anger = clamp01(agent.affect.anger + 0.025)
    elif action.kind == ActionKind.HELP_PEER:
        budgets.attention_budget = clamp01(budgets.attention_budget - 0.06)
        budgets.time_budget = clamp01(budgets.time_budget - 0.07)
        agent.affect.mood = clamp01(agent.affect.mood + 0.02)
    elif action.kind == ActionKind.MEDIATE_CONFLICT:
        budgets.attention_budget = clamp01(budgets.attention_budget - 0.09)
        budgets.time_budget = clamp01(budgets.time_budget - 0.08)
        body.stress = clamp01(body.stress + 0.01)
        agent.affect.patience = clamp01(agent.affect.patience - 0.015)
    elif action.kind == ActionKind.REDUCE_TRUST:
        budgets.attention_budget = clamp01(budgets.attention_budget - 0.04)
        body.stress = clamp01(body.stress + 0.01)
        if action.target_ids:
            target_id = action.target_ids[0]
            current = agent.social.trust.get(target_id, 0.5)
            agent.social.trust[target_id] = clamp01(current - 0.08)
    elif action.kind in {ActionKind.WRITE_TUTORIAL, ActionKind.SUGGEST_UPGRADE}:
        budgets.attention_budget = clamp01(budgets.attention_budget - 0.10)
        budgets.time_budget = clamp01(budgets.time_budget - 0.10)
        body.fatigue = clamp01(body.fatigue + 0.02)
    elif action.kind == ActionKind.POST:
        budgets.attention_budget = clamp01(budgets.attention_budget - 0.06)
        budgets.time_budget = clamp01(budgets.time_budget - 0.05)
    elif action.kind in {ActionKind.SHARE_FAILURE, ActionKind.WARN_PEER}:
        budgets.attention_budget = clamp01(budgets.attention_budget - 0.06)
        budgets.time_budget = clamp01(budgets.time_budget - 0.05)
        body.stress = clamp01(body.stress + 0.02)
    elif action.kind == ActionKind.SHARE_SUCCESS:
        budgets.attention_budget = clamp01(budgets.attention_budget - 0.06)
        budgets.time_budget = clamp01(budgets.time_budget - 0.05)
        agent.affect.mood = clamp01(agent.affect.mood + 0.015)
    elif action.kind == ActionKind.IGNORE:
        budgets.attention_budget = clamp01(budgets.attention_budget + 0.02)
    if coupling_enabled:
        deltas.update({
            f"coupling_{name}": value
            for name, value in apply_body_affect_coupling(agent).items()
            if value
        })
    return deltas


def _experience_packet_from_action(action: Action) -> ProductExperiencePacket | None:
    packet = action.typed_payload.get("product_experience")
    return packet if isinstance(packet, ProductExperiencePacket) else None


def make_action_event(
    *,
    event_id: str,
    tick: int,
    action: Action,
    pre_state_hash: str,
    post_state_hash: str,
    random_draw_refs: tuple[str, ...],
    typed_payload: dict,
    content_id: str | None = None,
    observation_refs: tuple[str, ...] = (),
) -> ExternalEvent:
    return ExternalEvent(
        event_id=event_id,
        tick=tick,
        kind=EventKind.ACTION,
        actor_id=action.actor_id,
        action_type=action.kind.value,
        target_ids=action.target_ids,
        channel_id=action.channel_id,
        artifact_id=action.artifact_id,
        content_id=content_id or action.content_id,
        pre_state_hash=pre_state_hash,
        post_state_hash=post_state_hash,
        observation_refs=observation_refs,
        random_draw_refs=random_draw_refs,
        typed_payload=typed_payload,
        public_visibility="public"
        if action.kind in {
            ActionKind.POST,
            ActionKind.REPLY,
            ActionKind.REPOST,
            ActionKind.SHARE_FAILURE,
            ActionKind.SHARE_SUCCESS,
            ActionKind.SUGGEST_UPGRADE,
            ActionKind.WRITE_TUTORIAL,
            ActionKind.CHALLENGE_CLAIM,
            ActionKind.ASK_QUESTION,
            ActionKind.ASK_FOR_EXPLANATION,
            ActionKind.VERIFY,
            ActionKind.COMPARE_SOURCES,
            ActionKind.HELP_PEER,
            ActionKind.WARN_PEER,
            ActionKind.MEDIATE_CONFLICT,
            ActionKind.REDUCE_TRUST,
        }
        else "private",
        company_visible_flag=bool(
            action.artifact_id
            and action.kind in {
                ActionKind.SHARE_FAILURE,
                ActionKind.SHARE_SUCCESS,
                ActionKind.SUGGEST_UPGRADE,
                ActionKind.WRITE_TUTORIAL,
            }
        ),
        detector_visible_only_after_run=True,
    )
