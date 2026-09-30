"""Institution / Governance Graph (design doc §7).

Turns one-off cooperation into persistent structure. Crucially (§7.4) an
institution is NOT a natural-language label — it must affect the environment
(rationing limits public-storage takes, contribution records drive reputation,
punishment changes resources/permissions, leader permission gates role
assignment). This graph stores the *state* the action handlers must consult to
enforce those effects, plus the lifecycle data the emergence detector reads
(§17.2 criteria).

The rule lifecycle state on each rule node is the contract the EmergenceDetector
keys on: ``proposed_by / supporters / opposers / created_turn / last_enforced_turn
/ violation_count / state_impact``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from agent_sdk.lived.graphs.base import GraphStore

# Node types (§7.2): rule / role / public_storage / contribution_ledger /
# punishment_record / appeal_case / leader / council / warehouse_keeper / scribe.
# Edge types (§7.3):
CREATED_BY = "created_by"
SUPPORTED_BY = "supported_by"
OPPOSED_BY = "opposed_by"
ENFORCED_BY = "enforced_by"
VIOLATED_BY = "violated_by"
AMENDED_BY = "amended_by"
APPLIES_TO = "applies_to"
CONTROLS_RESOURCE = "controls_resource"
GRANTS_PERMISSION = "grants_permission"


@dataclass
class RuleState:
    """Lifecycle bookkeeping for one rule node (mirrors §17.2 criteria)."""
    rule_id: str
    description: str = ""
    proposed_by: str = ""
    created_turn: int = -1
    supporters: Set[str] = field(default_factory=set)
    opposers: Set[str] = field(default_factory=set)
    adopted: bool = False
    adopted_turn: int = -1
    last_active_turn: int = -1
    enforcement_count: int = 0
    violation_count: int = 0
    # Did enforcing/violating this rule measurably change world/social state?
    state_impact: bool = False

    def to_dict(self) -> Dict[str, Any]:
        d = self.__dict__.copy()
        d["supporters"] = sorted(self.supporters)
        d["opposers"] = sorted(self.opposers)
        return d


class InstitutionGraph:
    def __init__(self):
        self.store = GraphStore(name="institution")
        self.rules: Dict[str, RuleState] = {}

    # -- rule lifecycle (§7.4) ---------------------------------------------
    def propose_rule(self, rule_id: str, proposer: str, turn: int, description: str = "") -> RuleState:
        rs = self.rules.get(rule_id) or RuleState(rule_id=rule_id)
        rs.description = description or rs.description
        rs.proposed_by = proposer
        rs.created_turn = turn if rs.created_turn < 0 else rs.created_turn
        rs.last_active_turn = turn
        self.rules[rule_id] = rs
        self.store.add_node(rule_id, "rule", **rs.to_dict())
        self.store.add_edge(rule_id, proposer, CREATED_BY, turn=turn)
        return rs

    def support_rule(self, rule_id: str, uid: str, turn: int) -> None:
        rs = self.rules.get(rule_id)
        if rs is None:
            return
        rs.supporters.add(uid)
        rs.opposers.discard(uid)
        rs.last_active_turn = turn
        self.store.add_edge(rule_id, uid, SUPPORTED_BY, turn=turn)
        self._refresh(rule_id)

    def oppose_rule(self, rule_id: str, uid: str, turn: int) -> None:
        rs = self.rules.get(rule_id)
        if rs is None:
            return
        rs.opposers.add(uid)
        rs.supporters.discard(uid)
        rs.last_active_turn = turn
        self.store.add_edge(rule_id, uid, OPPOSED_BY, turn=turn)
        self._refresh(rule_id)

    def adopt(self, rule_id: str, turn: int) -> None:
        rs = self.rules.get(rule_id)
        if rs is None:
            return
        rs.adopted = True
        rs.adopted_turn = turn if rs.adopted_turn < 0 else rs.adopted_turn
        rs.last_active_turn = turn
        self._refresh(rule_id)

    def enforce(self, rule_id: str, enforcer: str, turn: int, *, had_impact: bool = True) -> None:
        rs = self.rules.get(rule_id)
        if rs is None:
            return
        rs.enforcement_count += 1
        rs.last_active_turn = turn
        rs.state_impact = rs.state_impact or had_impact
        self.store.add_edge(rule_id, enforcer, ENFORCED_BY, turn=turn)
        self._refresh(rule_id)

    def violate(self, rule_id: str, violator: str, turn: int) -> None:
        rs = self.rules.get(rule_id)
        if rs is None:
            return
        rs.violation_count += 1
        rs.last_active_turn = turn
        self.store.add_edge(rule_id, violator, VIOLATED_BY, turn=turn)
        self._refresh(rule_id)

    def _refresh(self, rule_id: str) -> None:
        rs = self.rules.get(rule_id)
        if rs is not None:
            self.store.add_node(rule_id, "rule", **rs.to_dict())

    # -- roles / permissions (§7.4) ----------------------------------------
    def assign_role(self, uid: str, role_id: str, turn: int = 0) -> None:
        self.store.add_node(role_id, "role")
        self.store.add_edge(role_id, uid, APPLIES_TO, turn=turn)

    def grant_permission(self, role_id: str, resource_id: str) -> None:
        self.store.add_edge(role_id, resource_id, GRANTS_PERMISSION)

    # -- queries ------------------------------------------------------------
    def active_rules(self) -> List[RuleState]:
        return [rs for rs in self.rules.values() if rs.adopted]

    def get_rule(self, rule_id: str) -> Optional[RuleState]:
        return self.rules.get(rule_id)
