"""Profile Graph (design doc §3.3).

A per-agent graph of value / motivation / memory / skill / goal / relationship-
reference / identity / fear nodes connected by typed edges (supports,
conflicts_with, strengthens, weakens, caused_by, motivates, suppresses).

This sits alongside the flat :class:`agent_sdk.lived.persona.profile.ProfileVector`:
the vector is the fast scorer input; this graph captures the *structure* behind
it (e.g. ``last_winter_starvation strengthens long_termism``) for explanation,
case replay, and richer profile-update rules later.
"""
from __future__ import annotations

from typing import List

from agent_sdk.lived.graphs.base import GraphStore

# Node types (§3.3).
VALUE = "value"
MOTIVATION = "motivation"
MEMORY = "memory"
SKILL = "skill"
GOAL = "goal"
RELATION_REF = "relationship_ref"
IDENTITY = "identity"
FEAR = "fear"

# Edge types (§3.3).
SUPPORTS = "supports"
CONFLICTS_WITH = "conflicts_with"
STRENGTHENS = "strengthens"
WEAKENS = "weakens"
CAUSED_BY = "caused_by"
MOTIVATES = "motivates"
SUPPRESSES = "suppresses"


class ProfileGraph:
    """One profile graph per agent."""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.store = GraphStore(name=f"profile:{agent_id}")

    def add(self, node_id: str, ntype: str, **attrs) -> None:
        self.store.add_node(node_id, ntype, **attrs)

    def link(self, src: str, dst: str, etype: str, **attrs) -> None:
        self.store.add_edge(src, dst, etype, **attrs)

    # Convenience for the canonical §3.3 examples.
    def memory_strengthens_trait(self, memory_id: str, trait: str, summary: str = "") -> None:
        self.store.add_node(memory_id, MEMORY, summary=summary)
        self.store.add_node(trait, VALUE)
        self.store.add_edge(memory_id, trait, STRENGTHENS)

    def value_conflicts_with(self, value_a: str, behaviour: str) -> None:
        self.store.add_node(value_a, VALUE)
        self.store.add_edge(value_a, behaviour, CONFLICTS_WITH)

    def supporting_memories(self, trait: str) -> List[str]:
        return [e.src for e in self.store.edges(STRENGTHENS) if e.dst == trait]
