"""Scenario perturbations for cross-event validation."""

from __future__ import annotations

from .schemas import ContentItem, ContentKind, EventKind, ExternalEvent, SocietyState, clamp01


def inject_resource_shock(state: SocietyState, severity: float = 0.25) -> ExternalEvent:
    for agent in state.agents.values():
        agent.material_resources.food_or_budget_token = max(
            0.0, agent.material_resources.food_or_budget_token - severity
        )
        agent.body.stress = clamp01(agent.body.stress + severity / 2)
    event = ExternalEvent(
        event_id=f"scenario_{state.tick:06d}_resource_shock",
        tick=state.tick,
        kind=EventKind.RESOURCE_SHOCK,
        typed_payload={"severity": severity},
        public_visibility="public",
    )
    state.event_log.append(event)
    return event


def inject_rumor(state: SocietyState, author_id: str | None = None, claim_id: str = "rumor_claim") -> ExternalEvent:
    author = author_id or sorted(state.agents)[0]
    content_id = f"rumor_{state.tick:06d}_{len(state.content):06d}"
    state.content[content_id] = ContentItem(
        id=content_id,
        author_id=author,
        channel="public_forum",
        kind=ContentKind.RUMOR,
        created_at=state.tick,
        claim_refs=(claim_id,),
        text_surface=f"unverified claim {claim_id}",
        visibility_scope="public",
        topic_vector={"rumor": 1.0},
        indexed_at=state.tick,
    )
    event = ExternalEvent(
        event_id=f"scenario_{state.tick:06d}_rumor",
        tick=state.tick,
        kind=EventKind.RUMOR,
        actor_id=author,
        content_id=content_id,
        typed_payload={"claim_id": claim_id, "uncertainty": 0.8},
        public_visibility="public",
    )
    state.event_log.append(event)
    return event


def inject_public_conflict(state: SocietyState, actor_a: str | None = None, actor_b: str | None = None) -> ExternalEvent:
    agents = sorted(state.agents)
    a = actor_a or agents[0]
    b = actor_b or agents[min(1, len(agents) - 1)]
    event = ExternalEvent(
        event_id=f"scenario_{state.tick:06d}_public_conflict",
        tick=state.tick,
        kind=EventKind.PUBLIC_CONFLICT,
        actor_id=a,
        target_ids=(b,),
        channel_id="public_forum",
        typed_payload={"conflict": "visible_disagreement"},
        public_visibility="public",
    )
    state.event_log.append(event)
    return event


def inject_health_or_fatigue_shock(state: SocietyState, severity: float = 0.2) -> ExternalEvent:
    for agent in state.agents.values():
        agent.body.fatigue = clamp01(agent.body.fatigue + severity)
        agent.body.health_reserve = clamp01(agent.body.health_reserve - severity / 2)
    event = ExternalEvent(
        event_id=f"scenario_{state.tick:06d}_health_fatigue_shock",
        tick=state.tick,
        kind=EventKind.HEALTH_OR_FATIGUE_SHOCK,
        typed_payload={"severity": severity},
        public_visibility="private",
    )
    state.event_log.append(event)
    return event
