"""Body-affect-cognition coupling for Society-Core agents."""

from __future__ import annotations

from .schemas import AgentState, clamp01


def apply_body_affect_coupling(agent: AgentState) -> dict[str, float]:
    """Update affect, attention, and uncertainty from current body state.

    This is a typed endogenous coupling, not a detector: the resulting affect and
    cognition values can influence later actions through the policy utility.
    """

    body = agent.body
    affect = agent.affect
    budgets = agent.action_budgets
    cognition = agent.cognition

    hunger_pressure = body.hunger
    exhaustion = max(body.fatigue, body.sleep_pressure)
    threat_pressure = max(body.stress, 1.0 - body.health_reserve)
    resource_buffer = agent.material_resources.food_or_budget_token

    old_patience = affect.patience
    old_anger = affect.anger
    old_fear = affect.fear
    old_mood = affect.mood
    old_curiosity = affect.curiosity
    old_attention = budgets.attention_budget
    old_uncertainty = cognition.uncertainty

    affect.patience = clamp01(0.82 * affect.patience + 0.18 * (1.0 - 0.72 * hunger_pressure - 0.35 * exhaustion))
    affect.anger = clamp01(0.86 * affect.anger + 0.14 * (0.65 * hunger_pressure + 0.45 * body.stress))
    affect.fear = clamp01(0.88 * affect.fear + 0.12 * (0.8 * threat_pressure + 0.2 * (1.0 - resource_buffer)))
    affect.curiosity = clamp01(0.9 * affect.curiosity + 0.1 * (1.0 - 0.55 * exhaustion - 0.25 * hunger_pressure))
    affect.mood = max(-1.0, min(1.0, 0.86 * affect.mood + 0.14 * (
        0.55 * body.physiological_energy
        + 0.25 * resource_buffer
        - 0.55 * hunger_pressure
        - 0.35 * exhaustion
        - 0.3 * body.stress
    )))
    budgets.attention_budget = clamp01(
        0.86 * budgets.attention_budget
        + 0.14 * (1.0 - 0.7 * exhaustion - 0.35 * hunger_pressure - 0.25 * body.stress)
    )
    cognition.uncertainty = clamp01(
        0.9 * cognition.uncertainty
        + 0.1 * (0.55 * threat_pressure + 0.35 * exhaustion + 0.25 * hunger_pressure)
    )

    return {
        "patience": affect.patience - old_patience,
        "anger": affect.anger - old_anger,
        "fear": affect.fear - old_fear,
        "mood": affect.mood - old_mood,
        "curiosity": affect.curiosity - old_curiosity,
        "attention_budget": budgets.attention_budget - old_attention,
        "uncertainty": cognition.uncertainty - old_uncertainty,
    }
