"""Communication affordances: forum, feed, DM, search, and local observation."""

from __future__ import annotations

from .communities import community_affinity
from .network import network_proximity_score
from .schemas import AgentState, ContentItem, Observation, SocietyState, clamp01


ALLOWED_FEED_FEATURES: tuple[str, ...] = (
    "recency",
    "topic_match",
    "tie_strength",
    "network_proximity",
    "reply_activity",
    "local_engagement",
    "mention_or_dm_relation",
    "search_relevance",
    "exploration_noise",
    "feed_saturation",
    "channel_capacity_pressure",
    "bounded_candidate_generation",
)

FORBIDDEN_FEED_TERMS: tuple[str, ...] = (
    "rule_violation_penalty",
    "norm_compliance_boost",
    "truthfulness_judgment_by_platform",
    "civility_policy_penalty",
    "evidence_norm_penalty",
    "moderator_downrank",
    "trust_and_safety_enforcement",
    "official_truth_label",
    "institutional_rejection",
)

FEED_SCAN_WINDOW = 120
DM_SCAN_WINDOW = 120
SEARCH_SCAN_WINDOW = 250


def assert_feed_affordance_only(features: tuple[str, ...] = ALLOWED_FEED_FEATURES) -> None:
    forbidden = set(features).intersection(FORBIDDEN_FEED_TERMS)
    if forbidden:
        raise ValueError(f"Forbidden feed terms present: {sorted(forbidden)}")


def topic_match(agent: AgentState, content: ContentItem) -> float:
    if not content.topic_vector:
        return 0.1
    curiosity = agent.affect.curiosity
    # Minimal MVR: topic vectors are affordance tags, not truth/norm labels.
    return clamp01(0.2 + curiosity * max(content.topic_vector.values()))


def feed_score(
    agent: AgentState,
    content: ContentItem,
    state: SocietyState,
    proximity_cache: dict[str, float] | None = None,
) -> float:
    tick = state.tick
    recency = 1.0 / (1.0 + max(0, tick - content.created_at))
    tie = agent.social.tie_strength.get(content.author_id, 0.0)
    trust = agent.social.trust.get(content.author_id, 0.0)
    if proximity_cache is None:
        network_proximity = network_proximity_score(agent, content.author_id, state)
    else:
        if content.author_id not in proximity_cache:
            proximity_cache[content.author_id] = network_proximity_score(agent, content.author_id, state)
        network_proximity = proximity_cache[content.author_id]
    engagement = sum(content.engagement_counts.values()) / 10.0
    mention = 1.0 if agent.id in content.text_surface else 0.0
    saturation = max(0.0, len(agent.cognition.working_memory_items) - 5) / 10.0
    capacity_pressure = 1.0 - agent.action_budgets.attention_budget
    affinity = community_affinity(agent, content.channel, state)
    thread_depth_penalty = 0.03 * _thread_depth(content, state.content)
    return (
        0.9 * recency
        + 0.7 * topic_match(agent, content)
        + 0.45 * affinity
        + 0.35 * tie
        + 0.25 * trust
        + 0.5 * network_proximity
        + 0.3 * engagement
        + 0.5 * mention
        - 0.4 * saturation
        - 0.4 * capacity_pressure
        - thread_depth_penalty
    )


def recent_content_values(state: SocietyState, *, max_scan: int) -> tuple[ContentItem, ...]:
    """Return a bounded recency window without scanning the full content history."""
    items: list[ContentItem] = []
    for index, content in enumerate(reversed(state.content.values())):
        if index >= max_scan:
            break
        items.append(content)
    return tuple(items)


def visible_feed_content(agent: AgentState, state: SocietyState, limit: int = 20) -> tuple[str, ...]:
    assert_feed_affordance_only()
    accessible_channels = agent.channel_access.union(agent.subscriptions)
    candidates = [
        content for content in recent_content_values(state, max_scan=FEED_SCAN_WINDOW)
        if content.visibility_scope == "public"
        and content.channel in accessible_channels
    ]
    proximity_cache: dict[str, float] = {}
    ranked = sorted(
        candidates,
        key=lambda item: feed_score(agent, item, state, proximity_cache),
        reverse=True,
    )
    attention_limit = max(1, int(agent.action_budgets.attention_budget * limit))
    return tuple(content.id for content in ranked[:attention_limit])


def dm_content(agent: AgentState, state: SocietyState, limit: int = 20) -> tuple[str, ...]:
    matches = [
        content.id for content in recent_content_values(state, max_scan=DM_SCAN_WINDOW)
        if content.channel == "dm"
        and _content_visible_to_agent(agent, content)
    ]
    return tuple(matches[:limit])


def search_content(agent: AgentState, state: SocietyState, limit: int = 10) -> tuple[str, ...]:
    if not agent.search_queries:
        return ()
    query_terms = {term.lower() for query in agent.search_queries for term in query.split()}
    scored: list[tuple[int, str]] = []
    for content in recent_content_values(state, max_scan=SEARCH_SCAN_WINDOW):
        if not _content_visible_to_agent(agent, content):
            continue
        indexed_terms = set(state.search_index_terms.get(content.id, ()))
        if not indexed_terms:
            indexed_terms = set(content.text_surface.lower().split())
        score = sum(1 for term in query_terms if term and term in indexed_terms)
        if score:
            scored.append((score, content.id))
    return tuple(content_id for _, content_id in sorted(scored, reverse=True)[:limit])


def _content_visible_to_agent(agent: AgentState, content: ContentItem) -> bool:
    if content.visibility_scope == "public":
        return True
    if content.author_id == agent.id:
        return True
    if content.recipient_ids:
        return agent.id in content.recipient_ids
    if content.visibility_scope == f"dm:{agent.id}":
        return True
    return (
        content.channel == "dm"
        and content.visibility_scope == "private"
        and agent.id in content.text_surface
    )


def observe_local_slice(agent: AgentState, state: SocietyState) -> Observation:
    local_event_ids = tuple(
        event.event_id
        for event in state.event_log[-50:]
        if event.actor_id == agent.id
        or event.public_visibility == "public"
        or event.channel_id in agent.channel_access
    )
    own_memory_refs = tuple(memory.event_ref for memory in agent.memory[-10:])
    return Observation(
        agent_id=agent.id,
        tick=state.tick,
        visible_content_ids=visible_feed_content(agent, state),
        dm_ids=dm_content(agent, state),
        search_result_ids=search_content(agent, state),
        local_event_ids=local_event_ids,
        own_memory_refs=own_memory_refs,
    )


def _thread_depth(content: ContentItem, all_content: dict[str, ContentItem]) -> int:
    depth = 0
    current = content
    seen = {content.id}
    while current.parent_id and current.parent_id not in seen:
        parent = all_content.get(current.parent_id)
        if parent is None:
            break
        depth += 1
        seen.add(parent.id)
        current = parent
    return depth
