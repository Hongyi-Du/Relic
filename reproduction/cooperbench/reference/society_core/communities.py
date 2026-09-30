"""Pre-institutional community affordances for the external forum.

The four curated DEFAULT_COMMUNITIES are archetype slots (product builders,
extension authors, ops maintainers, learners) whose surface semantics were
hand-tuned for one frontend-tooling product. When the loaded substrate is a
different product, ``apply_substrate_communities`` replaces the curated set
with communities derived from the artifact's declared semantics and records
which derived community fills each archetype slot in
``state.community_archetype_map`` (archetype id -> derived id), so that
membership, affinity, and posting routing keep their meaning. The curated set
remains the fallback for the repo-digest substrate and for artifacts that
declare no usable semantics.
"""

from __future__ import annotations

import re

from .schemas import AgentState, Artifact, Community, SocietyState, clamp01


DEFAULT_COMMUNITIES: tuple[Community, ...] = (
    Community(
        id="frontend_build",
        label="Frontend Build",
        channel_id="community_frontend_build",
        topic_vector={"frontend": 1.0, "build": 0.9, "tooling": 0.7},
        activity_multiplier=1.12,
        technical_depth=0.68,
        debate_intensity=0.55,
    ),
    Community(
        id="framework_authors",
        label="Framework Authors",
        channel_id="community_framework_authors",
        topic_vector={"framework": 1.0, "plugins": 0.8, "architecture": 0.9},
        activity_multiplier=0.92,
        technical_depth=0.88,
        debate_intensity=0.72,
    ),
    Community(
        id="ops_maintenance",
        label="Ops Maintenance",
        channel_id="community_ops_maintenance",
        topic_vector={"production": 1.0, "dependencies": 0.8, "reliability": 0.9},
        activity_multiplier=0.78,
        technical_depth=0.76,
        debate_intensity=0.48,
    ),
    Community(
        id="learners",
        label="Learners",
        channel_id="community_learners",
        topic_vector={"learning": 1.0, "documentation": 0.8, "examples": 0.8},
        activity_multiplier=1.24,
        technical_depth=0.36,
        debate_intensity=0.38,
    ),
)

ARCHETYPE_SLOT_ORDER: tuple[str, ...] = tuple(
    community.id for community in DEFAULT_COMMUNITIES
)


DOMAIN_COMMUNITIES: dict[str, tuple[str, ...]] = {
    "frontend_product": ("frontend_build", "learners"),
    "library_author": ("framework_authors", "frontend_build"),
    "ops_maintainer": ("ops_maintenance", "frontend_build"),
    "learner_builder": ("learners", "frontend_build"),
    "general_software": ("frontend_build", "learners"),
}


def _theme_slug(theme: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", theme.lower()).strip("_")[:48]


def derive_substrate_communities(artifact: Artifact | None) -> tuple[Community, ...]:
    """Communities anchored in the artifact's declared semantics.

    Returns () when the artifact is the curated repo-digest substrate or
    declares nothing to derive from, meaning the caller must keep the
    curated fallback.
    """

    from .product_experience import (
        _looks_like_repo_digest_artifact,
        derive_substrate_theme_lexicon,
    )

    if artifact is None or _looks_like_repo_digest_artifact(artifact):
        return ()
    derived: list[Community] = []
    seen: set[str] = set()
    for theme, keywords in derive_substrate_theme_lexicon(artifact):
        slug = _theme_slug(theme)
        if not slug or slug in seen:
            continue
        seen.add(slug)
        topic_vector = {slug: 1.0}
        for token in sorted(keywords):
            topic_vector.setdefault(token, 0.8)
        derived.append(
            Community(
                id=f"substrate_{slug}",
                label=slug.replace("_", " ").title(),
                channel_id=f"community_substrate_{slug}",
                topic_vector=topic_vector,
            )
        )
        if len(derived) >= len(ARCHETYPE_SLOT_ORDER):
            break
    return tuple(derived)


def apply_substrate_communities(state: SocietyState, artifact: Artifact | None) -> bool:
    """Anchor the forum's community structure to the t0 substrate.

    Only the first injected artifact may reshape the forum; later releases
    (upgraded versions, placebos, competitors) must never rescramble it.
    Deterministic and rng-free, so random draw sequences are unaffected.
    """

    if state.artifacts or state.community_archetype_map:
        return False
    derived = derive_substrate_communities(artifact)
    if not derived:
        return False
    curated_channels = {
        community.channel_id for community in state.communities.values()
    }
    state.communities = {community.id: community for community in derived}
    state.community_archetype_map = {
        archetype_id: derived[index % len(derived)].id
        for index, archetype_id in enumerate(ARCHETYPE_SLOT_ORDER)
    }
    for agent_id in sorted(state.agents):
        agent = state.agents[agent_id]
        agent.channel_access -= curated_channels
        agent.subscriptions -= curated_channels
        assign_agent_communities(agent, state)
    return True


def _resolved_membership_ids(
    memberships: tuple[str, ...], state: SocietyState
) -> tuple[str, ...]:
    if not state.community_archetype_map:
        return tuple(memberships)
    resolved: list[str] = []
    for community_id in memberships:
        mapped = state.community_archetype_map.get(community_id, community_id)
        if mapped not in resolved:
            resolved.append(mapped)
    return tuple(resolved)


def _archetype_aliases(community_id: str, state: SocietyState) -> frozenset[str]:
    aliases = {community_id}
    aliases.update(
        archetype_id
        for archetype_id, derived_id in state.community_archetype_map.items()
        if derived_id == community_id
    )
    return frozenset(aliases)


def initialize_communities(state: SocietyState) -> None:
    for community in DEFAULT_COMMUNITIES:
        state.communities[community.id] = community


def assign_agent_communities(agent: AgentState, state: SocietyState) -> None:
    memberships = agent.profile.community_memberships
    if not memberships:
        memberships = DOMAIN_COMMUNITIES.get(agent.profile.primary_domain, ("frontend_build",))
    for community_id in _resolved_membership_ids(tuple(memberships), state):
        community = state.communities.get(community_id)
        if community is None:
            continue
        agent.channel_access.add(community.channel_id)
        agent.subscriptions.add(community.channel_id)


def community_affinity(agent: AgentState, channel_id: str, state: SocietyState) -> float:
    if channel_id in {"public_forum", "feed", "search", "dm"}:
        return 0.35
    for community in state.communities.values():
        if community.channel_id == channel_id:
            aliases = _archetype_aliases(community.id, state)
            member = (
                bool(aliases & set(agent.profile.community_memberships))
                or channel_id in agent.subscriptions
            )
            domain_match = bool(
                aliases & set(DOMAIN_COMMUNITIES.get(agent.profile.primary_domain, ()))
            )
            return clamp01(0.25 + 0.45 * float(member) + 0.20 * float(domain_match))
    return 0.05


def preferred_public_channel(agent: AgentState, state: SocietyState) -> str:
    memberships = agent.profile.community_memberships or DOMAIN_COMMUNITIES.get(
        agent.profile.primary_domain,
        (),
    )
    candidates = [
        state.communities[community_id].channel_id
        for community_id in _resolved_membership_ids(tuple(memberships), state)
        if community_id in state.communities
    ]
    if not candidates:
        return "public_forum"
    if agent.profile.strategic_boldness >= 0.72 and len(candidates) > 1:
        return candidates[1]
    return candidates[0]
