"""Social Relation Graph (design doc §5).

Nodes are agents; edges carry the social relations that gate sharing,
disclosure, help, accusation belief, rule/leader support and faction
membership (§5.2). Edge attributes accumulate (trust goes up/down over many
events) via the base store's additive numeric merge.

The update helpers (§5.3) are the canonical nudges; richer dynamics (decay,
reputation propagation) slot in later behind the same methods.
"""
from __future__ import annotations

from typing import Dict, List

from agent_sdk.lived.graphs.base import GraphStore

# Edge types (§5.1).
TRUST = "trust"
DEBT = "debt"
PROMISE = "promise"
FRIENDSHIP = "friendship"
RIVALRY = "rivalry"
BETRAYAL = "betrayal"
TEACHING = "teaching"
HELP = "help"
SUPPORT = "support"
OPPOSITION = "opposition"
GRIEVANCE = "grievance"
FACTION = "faction_member"


class SocialGraph:
    def __init__(self):
        self.store = GraphStore(name="social")

    def ensure_agent(self, uid: str) -> None:
        self.store.add_node(uid, "agent")

    def trust(self, a: str, b: str) -> float:
        return float(self.store.edge_attr(a, b, TRUST, "strength", 0.0))

    # -- §5.3 update rules --------------------------------------------------
    def keep_promise(self, a: str, b: str, amount: float = 0.1) -> None:
        self.store.add_edge(a, b, TRUST, strength=amount)

    def break_promise(self, a: str, b: str, amount: float = 0.2) -> None:
        self.store.add_edge(a, b, TRUST, strength=-amount)
        self.store.add_edge(b, a, GRIEVANCE, strength=amount)

    def teach_skill(self, teacher: str, student: str) -> None:
        self.store.add_edge(teacher, student, TEACHING, strength=1.0)
        self.store.add_edge(student, teacher, TRUST, strength=0.1)

    def help_during_crisis(self, helper: str, helped: str, amount: float = 0.2) -> None:
        self.store.add_edge(helped, helper, DEBT, strength=amount)
        self.store.add_edge(helped, helper, TRUST, strength=amount)

    def false_accusation_refuted(self, accuser: str, accused: str, amount: float = 0.2) -> None:
        # accused's trust toward accuser drops (§5.3)
        self.store.add_edge(accused, accuser, TRUST, strength=-amount)

    def set_faction(self, uid: str, faction_id: str) -> None:
        self.store.add_edge(uid, faction_id, FACTION, strength=1.0)

    # -- queries ------------------------------------------------------------
    def allies(self, uid: str, min_trust: float = 0.3) -> List[str]:
        return [b for b in self.store.neighbors(uid, TRUST)
                if self.trust(uid, b) >= min_trust]

    def debts_owed_by(self, uid: str) -> Dict[str, float]:
        return {e.dst: e.attrs.get("strength", 0.0)
                for e in self.store.edges(DEBT) if e.src == uid}

    def has_betrayed(self, suspect: str, victim: str) -> bool:
        return self.store.get_edge(suspect, victim, BETRAYAL) is not None or \
               self.store.edge_attr(victim, suspect, TRUST, "strength", 0.0) < -0.3
