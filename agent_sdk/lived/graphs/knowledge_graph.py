"""Knowledge / Technology Graph (design doc §6).

Tracks cumulative culture: who discovered/taught/learned what, recipe
dependencies, and which knowledge is archived publicly (§6.2/§6.3). Enables
the knowledge-diffusion and skill-retention metrics (§18.4).
"""
from __future__ import annotations

from typing import List

from agent_sdk.lived.graphs.base import GraphStore

# Node types (§6.2): skill / recipe / tool / resource_knowledge /
# survival_strategy / infrastructure_technique / social_rule_knowledge.
# Edge types (§6.3):
REQUIRES = "requires"
ENABLES = "enables"
DISCOVERED_BY = "discovered_by"
TAUGHT_BY = "taught_by"
LEARNED_BY = "learned_by"
ARCHIVED_IN = "stored_in_public_archive"
IMPROVES = "improves"


class KnowledgeGraph:
    def __init__(self):
        self.store = GraphStore(name="knowledge")

    def add_knowledge(self, knowledge_id: str, ntype: str = "skill", **attrs) -> None:
        self.store.add_node(knowledge_id, ntype, **attrs)

    def discover(self, knowledge_id: str, discoverer: str, turn: int = 0) -> None:
        self.store.add_node(knowledge_id, "skill")
        self.store.add_node(discoverer, "agent")
        self.store.add_edge(knowledge_id, discoverer, DISCOVERED_BY, turn=turn)

    def teach(self, knowledge_id: str, teacher: str, student: str, turn: int = 0) -> None:
        self.store.add_node(student, "agent")
        self.store.add_edge(knowledge_id, teacher, TAUGHT_BY, turn=turn)
        self.store.add_edge(knowledge_id, student, LEARNED_BY, turn=turn)

    def add_dependency(self, recipe: str, requires: str) -> None:
        self.store.add_edge(recipe, requires, REQUIRES)

    def archive(self, knowledge_id: str, archive_id: str) -> None:
        self.store.add_edge(knowledge_id, archive_id, ARCHIVED_IN)

    # -- queries ------------------------------------------------------------
    def holders(self, knowledge_id: str) -> List[str]:
        """Agents who learned or discovered this knowledge."""
        out = set(self.store.neighbors(knowledge_id, LEARNED_BY))
        out.update(self.store.neighbors(knowledge_id, DISCOVERED_BY))
        return sorted(out)

    def diffusion_chain(self, knowledge_id: str) -> List[str]:
        """Ordered learner uids (proxy for a diffusion chain; refine with
        per-edge turn sort when populated)."""
        learners = [(self.store.edge_attr(knowledge_id, s, LEARNED_BY, "turn", 0), s)
                    for s in self.store.neighbors(knowledge_id, LEARNED_BY)]
        learners.sort()
        return [s for _, s in learners]
