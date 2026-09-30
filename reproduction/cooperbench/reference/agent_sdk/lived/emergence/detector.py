"""Emergence Detector (design doc §17).

Core principle (§17.1, §22.2): a high-level social phenomenon does NOT exist
because an agent said so. It must meet *structural* criteria measured over the
event + institution graphs. This keeps emergence from being "scripted" — the
detector reads accumulated state, it never writes it.

Institution emergence (§17.2) requires all five:
  1. Proposal    — proposed by >=1 agent
  2. Adoption    — accepted / voted / repeatedly used by multiple agents
  3. Persistence — active for >= min_persistence turns
  4. Enforcement — violations carry consequences
  5. State impact — measurably changed env state or a social graph

``detect_institutions`` returns one :class:`DetectionResult` per rule with the
per-criterion booleans + an ``emerged`` flag (AND of all five). Other §17.3
phenomena (teaching chains, trade networks, factions, ...) get their own
``detect_*`` methods as the social-systems layer lands.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from agent_sdk.lived.graphs.event_graph import EventGraph
from agent_sdk.lived.graphs.institution_graph import InstitutionGraph
from agent_sdk.lived.graphs.knowledge_graph import KnowledgeGraph


@dataclass
class DetectionResult:
    kind: str                       # e.g. "institution", "teaching_chain"
    subject_id: str                 # rule_id / knowledge_id / faction_id
    emerged: bool = False
    criteria: Dict[str, bool] = field(default_factory=dict)
    evidence: Dict[str, Any] = field(default_factory=dict)


class EmergenceDetector:
    def __init__(
        self,
        *,
        min_persistence_turns: int = 20,
        min_adopters: int = 2,
    ):
        self.min_persistence_turns = min_persistence_turns
        self.min_adopters = min_adopters

    # -- §17.2 institutions -------------------------------------------------
    def detect_institutions(
        self, inst: InstitutionGraph, *, current_turn: int
    ) -> List[DetectionResult]:
        results: List[DetectionResult] = []
        for rs in inst.rules.values():
            proposal = bool(rs.proposed_by)
            adoption = rs.adopted and len(rs.supporters) >= self.min_adopters
            persistence = (
                rs.created_turn >= 0
                and (rs.last_active_turn - rs.created_turn) >= self.min_persistence_turns
            )
            enforcement = rs.enforcement_count > 0
            state_impact = rs.state_impact
            crit = {
                "proposal": proposal,
                "adoption": adoption,
                "persistence": persistence,
                "enforcement": enforcement,
                "state_impact": state_impact,
            }
            results.append(DetectionResult(
                kind="institution",
                subject_id=rs.rule_id,
                emerged=all(crit.values()),
                criteria=crit,
                evidence={
                    "supporters": sorted(rs.supporters),
                    "enforcement_count": rs.enforcement_count,
                    "violation_count": rs.violation_count,
                    "lifespan": max(0, rs.last_active_turn - rs.created_turn),
                },
            ))
        return results

    # -- §17.3 knowledge diffusion chain -----------------------------------
    def detect_knowledge_diffusion(
        self, kg: KnowledgeGraph, *, min_chain: int = 3
    ) -> List[DetectionResult]:
        results: List[DetectionResult] = []
        for node in kg.store.nodes("skill"):
            chain = kg.diffusion_chain(node.node_id)
            results.append(DetectionResult(
                kind="knowledge_diffusion",
                subject_id=node.node_id,
                emerged=len(chain) >= min_chain,
                criteria={"chain_length_ok": len(chain) >= min_chain},
                evidence={"chain": chain},
            ))
        return results

    # -- §17.3 teaching / apprenticeship (event-graph based) ---------------
    def detect_teaching(self, eg: EventGraph, *, min_events: int = 2) -> List[DetectionResult]:
        teach_events = eg.by_action("teach")
        by_teacher: Dict[str, int] = {}
        for ev in teach_events:
            by_teacher[ev.actor] = by_teacher.get(ev.actor, 0) + 1
        return [
            DetectionResult(
                kind="teaching",
                subject_id=teacher,
                emerged=count >= min_events,
                criteria={"repeated_teaching": count >= min_events},
                evidence={"teach_count": count},
            )
            for teacher, count in by_teacher.items()
        ]

    def run_all(
        self,
        *,
        current_turn: int,
        institution: InstitutionGraph | None = None,
        knowledge: KnowledgeGraph | None = None,
        event: EventGraph | None = None,
    ) -> Dict[str, List[DetectionResult]]:
        """Convenience: run every detector that has its graph supplied."""
        out: Dict[str, List[DetectionResult]] = {}
        if institution is not None:
            out["institution"] = self.detect_institutions(institution, current_turn=current_turn)
        if knowledge is not None:
            out["knowledge_diffusion"] = self.detect_knowledge_diffusion(knowledge)
        if event is not None:
            out["teaching"] = self.detect_teaching(event)
        return out
