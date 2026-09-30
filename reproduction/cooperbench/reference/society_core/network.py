"""Pre-institutional social graph and network exposure mechanics."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import sqrt

from .communities import community_affinity
from .randomness import SeededRandom
from .schemas import AgentState, Observation, SocietyState, clamp01


NETWORK_SCORE_FEATURES: tuple[str, ...] = (
    "homophily",
    "tie_strength",
    "private_trust",
    "familiarity",
    "second_hop_proximity",
    "seeded_exploration_noise",
)

FORBIDDEN_NETWORK_TERMS: tuple[str, ...] = (
    "moderator",
    "administrator",
    "rule_enforcement",
    "formal_reputation",
    "official_status",
    "institutional_authority",
    "platform_safety_label",
)


@dataclass(frozen=True)
class NetworkArtifactExposureDiagnostic:
    artifact_id: str
    exposure_count: int
    release_exposure_count: int
    network_exposure_count: int
    unknown_exposure_count: int
    network_trial_actor_count: int
    mean_network_distance: float
    source_counts: dict[str, int]


def assert_pre_institutional_network_features(features: tuple[str, ...] = NETWORK_SCORE_FEATURES) -> None:
    forbidden = set(features).intersection(FORBIDDEN_NETWORK_TERMS)
    if forbidden:
        raise ValueError(f"Forbidden network terms present: {sorted(forbidden)}")


def homophily_score(a: AgentState, b: AgentState) -> float:
    differences = (
        abs(a.affect.curiosity - b.affect.curiosity),
        abs(a.cognition.uncertainty - b.cognition.uncertainty),
        abs(a.action_budgets.attention_budget - b.action_budgets.attention_budget),
        abs(a.body.stress - b.body.stress),
    )
    distance = sqrt(sum(value * value for value in differences) / len(differences))
    return clamp01(1.0 - distance)


def initialize_social_graph(
    state: SocietyState,
    rng: SeededRandom,
    *,
    max_out_degree: int = 8,
) -> None:
    assert_pre_institutional_network_features()
    agent_ids = sorted(state.agents)
    if len(agent_ids) <= 1:
        return
    out_degree = max(1, min(max_out_degree, len(agent_ids) - 1))
    for agent_id in agent_ids:
        agent = state.agents[agent_id]
        scored: list[tuple[float, str]] = []
        for other_id in agent_ids:
            if other_id == agent_id:
                continue
            other = state.agents[other_id]
            noise = rng.random("network", kind="init_social_edge_noise")
            score = 0.82 * homophily_score(agent, other) + 0.18 * noise
            scored.append((score, other_id))
        selected = sorted(scored, reverse=True)[:out_degree]
        agent.social.tie_strength.clear()
        agent.social.trust.clear()
        agent.social.familiarity.clear()
        agent.social.social_trust_as_private_belief.clear()
        for score, other_id in selected:
            tie = clamp01(0.15 + 0.75 * score)
            trust = clamp01(0.2 + 0.55 * score)
            agent.social.tie_strength[other_id] = tie
            agent.social.trust[other_id] = trust
            agent.social.familiarity[other_id] = clamp01(0.1 + 0.65 * score)
            agent.social.social_trust_as_private_belief[other_id] = trust


def network_distance(agent: AgentState, author_id: str, state: SocietyState) -> int | None:
    if author_id == agent.id:
        return 0
    if author_id in agent.social.tie_strength:
        return 1
    for neighbor_id, tie in agent.social.tie_strength.items():
        if tie <= 0:
            continue
        neighbor = state.agents.get(neighbor_id)
        if neighbor and author_id in neighbor.social.tie_strength:
            return 2
    return None


def network_proximity_score(agent: AgentState, author_id: str, state: SocietyState) -> float:
    distance = network_distance(agent, author_id, state)
    if distance is None:
        return 0.0
    if distance == 0:
        return 0.75
    if distance == 1:
        tie = agent.social.tie_strength.get(author_id, 0.0)
        trust = agent.social.trust.get(author_id, 0.0)
        familiarity = agent.social.familiarity.get(author_id, 0.0)
        return clamp01(0.45 * tie + 0.35 * trust + 0.2 * familiarity)
    best_second_hop = 0.0
    for neighbor_id, tie in agent.social.tie_strength.items():
        neighbor = state.agents.get(neighbor_id)
        if not neighbor or author_id not in neighbor.social.tie_strength:
            continue
        best_second_hop = max(best_second_hop, sqrt(tie * neighbor.social.tie_strength[author_id]))
    return clamp01(0.35 * best_second_hop)


def grant_artifact_access(
    agent: AgentState,
    artifact_id: str,
    *,
    source: str,
    tick: int,
    exposed_by: str | None = None,
    content_id: str | None = None,
    distance: int = 0,
) -> bool:
    if artifact_id in agent.artifact_access:
        return False
    agent.artifact_access.add(artifact_id)
    agent.material_resources.artifact_access = max(agent.material_resources.artifact_access, 1.0)
    if artifact_id not in agent.artifact_exposure_groups:
        agent.artifact_exposure_groups[artifact_id] = "release_seed" if source == "release" else "network_eligible"
    agent.artifact_exposure_sources[artifact_id] = source
    agent.artifact_exposure_ticks[artifact_id] = tick
    if exposed_by:
        agent.artifact_exposed_by[artifact_id] = exposed_by
    if content_id:
        agent.artifact_exposure_content[artifact_id] = content_id
    agent.artifact_exposure_distance[artifact_id] = distance
    return True


def apply_peer_artifact_signal(
    *,
    state: SocietyState,
    agent: AgentState,
    artifact_id: str,
    author_id: str,
    content_kind: object,
    content_channel: str | None = None,
) -> bool:
    if agent.artifact_peer_signal_groups.get(artifact_id) != "peer_signal_eligible":
        return False
    proximity = network_proximity_score(agent, author_id, state)
    if proximity <= 0:
        return False
    author = state.agents.get(author_id)
    tie = agent.social.tie_strength.get(author_id, 0.0)
    trust = agent.social.trust.get(author_id, 0.0)
    familiarity = agent.social.familiarity.get(author_id, 0.0)
    perceived_status = agent.social.perceived_status.get(author_id, 0.0)
    similarity = homophily_score(agent, author) if author else 0.0
    affinity = community_affinity(agent, content_channel or "public_forum", state)
    count = agent.cognition.artifact_peer_signal_count.get(artifact_id, 0) + 1
    repetition = min(1.0, count / 5.0)
    kind_value = getattr(content_kind, "value", str(content_kind))
    positive_content = kind_value in {"success_report", "tutorial", "repost", "reply"}
    negative_content = kind_value in {"failure_report", "callout", "rumor"}
    content_bonus = 0.16 if kind_value == "tutorial" else 0.12 if kind_value == "success_report" else 0.04
    relationship_signal = (
        0.32 * proximity
        + 0.22 * trust
        + 0.18 * tie
        + 0.12 * familiarity
        + 0.10 * similarity
        + 0.08 * affinity
        + 0.06 * perceived_status
        + 0.06 * repetition
    )
    signal_strength = clamp01(0.04 + relationship_signal + content_bonus)
    agent.cognition.artifact_peer_signal_count[artifact_id] = count
    if negative_content:
        previous_negative = agent.cognition.artifact_negative_signal_strength.get(artifact_id, 0.0)
        agent.cognition.artifact_negative_signal_strength[artifact_id] = clamp01(
            previous_negative + signal_strength * (1.0 - previous_negative)
        )
        previous_belief = agent.cognition.beliefs_about_artifacts.get(artifact_id, 0.5)
        agent.cognition.beliefs_about_artifacts[artifact_id] = clamp01(
            previous_belief - 0.35 * signal_strength * previous_belief
        )
        agent.cognition.uncertainty = clamp01(agent.cognition.uncertainty + 0.05 * signal_strength)
        agent.affect.fear = clamp01(agent.affect.fear + 0.03 * signal_strength)
        return True
    if not positive_content:
        signal_strength *= 0.65
    previous_signal = agent.cognition.artifact_peer_signal_strength.get(artifact_id, 0.0)
    agent.cognition.artifact_peer_signal_strength[artifact_id] = clamp01(
        previous_signal + signal_strength * (1.0 - previous_signal)
    )
    previous_belief = agent.cognition.beliefs_about_artifacts.get(artifact_id, 0.5)
    agent.cognition.beliefs_about_artifacts[artifact_id] = clamp01(
        previous_belief + signal_strength * (1.0 - previous_belief)
    )
    agent.cognition.uncertainty = clamp01(agent.cognition.uncertainty - 0.06 * signal_strength)
    agent.affect.mood = clamp01(agent.affect.mood + 0.025 * signal_strength)
    return True


def update_network_exposures_from_observation(
    *,
    state: SocietyState,
    agent: AgentState,
    observation: Observation,
    peer_signal_enabled: bool = True,
) -> tuple[str, ...]:
    visible_authors: list[str] = []
    for content_id in observation.visible_content_ids:
        content = state.content.get(content_id)
        if content is None:
            continue
        distance = network_distance(agent, content.author_id, state)
        if distance is None:
            continue
        visible_authors.append(content.author_id)
        for artifact_id in content.artifact_refs:
            if artifact_id not in state.artifacts:
                continue
            if agent.artifact_exposure_groups.get(artifact_id) == "network_holdout":
                continue
            grant_artifact_access(
                agent,
                artifact_id,
                source="network",
                tick=state.tick,
                exposed_by=content.author_id,
                content_id=content_id,
                distance=distance,
            )
            if peer_signal_enabled:
                apply_peer_artifact_signal(
                    state=state,
                    agent=agent,
                    artifact_id=artifact_id,
                    author_id=content.author_id,
                    content_kind=content.kind,
                    content_channel=content.channel,
                )
    return tuple(sorted(set(visible_authors)))


def artifact_exposure_diagnostic(
    *,
    state: SocietyState,
    artifact_id: str,
    trial_actor_ids: set[str],
) -> NetworkArtifactExposureDiagnostic:
    exposed_agents = [
        agent for agent in state.agents.values()
        if artifact_id in agent.artifact_access
    ]
    source_counts = Counter(
        agent.artifact_exposure_sources.get(artifact_id, "unknown")
        for agent in exposed_agents
    )
    network_agents = [
        agent for agent in exposed_agents
        if agent.artifact_exposure_sources.get(artifact_id) == "network"
    ]
    distances = [
        float(agent.artifact_exposure_distance.get(artifact_id, 0))
        for agent in network_agents
    ]
    network_trial_actors = {
        agent.id for agent in network_agents
        if agent.id in trial_actor_ids
    }
    return NetworkArtifactExposureDiagnostic(
        artifact_id=artifact_id,
        exposure_count=len(exposed_agents),
        release_exposure_count=source_counts.get("release", 0),
        network_exposure_count=source_counts.get("network", 0),
        unknown_exposure_count=source_counts.get("unknown", 0),
        network_trial_actor_count=len(network_trial_actors),
        mean_network_distance=sum(distances) / len(distances) if distances else 0.0,
        source_counts=dict(sorted(source_counts.items())),
    )
