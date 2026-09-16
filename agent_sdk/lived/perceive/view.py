"""Read-only frontend aggregator for the lived society (design doc §17/§18 viz).

Mirrors the (removed) ``EvolutionHost.meta_evolver_state(...)`` pattern: a
single pure function that snapshots the lived graphs + profiles + emergence
detections into a **JSON-serializable dict** the frontend can render. It has
NO side effects and NEVER writes graph state.

The schema is frozen in :data:`EMPTY_LIVED_SNAPSHOT` so the frontend panel can
be built (and shows an "awaiting wiring" placeholder) before the engine
populates real data. When the harness engine wires the lived system in, it
calls ``lived_society_snapshot(...)`` and stuffs the result under
``state["society"]`` (read by ``/api/lived/state`` and the society panel).

Env-agnostic: takes the graph/profile/detector objects, returns plain data.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

from agent_sdk.lived.core.contracts import ProfileVector
from agent_sdk.lived.emergence import EmergenceDetector
from agent_sdk.lived.graphs import (
    EventGraph,
    InstitutionGraph,
    KnowledgeGraph,
    SocialGraph,
)
from agent_sdk.lived.graphs.social_graph import DEBT, TRUST
from agent_sdk.lived.persona.profile import ProfileState


# Frozen schema — the contract the frontend society panel renders against.
EMPTY_LIVED_SNAPSHOT: Dict[str, Any] = {
    "turn": 0,
    "wired": False,
    "profiles": [],       # [{agent_id, role, top_traits:[{name,value}], mood:{...}}]
    "social": {"trust_edges": [], "debt_edges": [], "factions": []},
    "institutions": [],   # [{rule_id, description, adopted, supporters, enforcement_count,
                          #   violation_count, emerged, criteria}]
    "knowledge": {"diffusion": []},   # [{knowledge_id, chain:[uid...]}]
    "emergence": [],      # [{kind, subject_id, emerged, criteria}]
}


def _top_traits(profile: ProfileVector, k: int = 4) -> List[Dict[str, Any]]:
    items = sorted(profile.to_dict().items(), key=lambda kv: kv[1], reverse=True)
    return [{"name": n, "value": round(v, 3)} for n, v in items[:k]]


def lived_society_snapshot(
    *,
    current_turn: int = 0,
    profiles: Optional[Mapping[str, ProfileState]] = None,
    social: Optional[SocialGraph] = None,
    institution: Optional[InstitutionGraph] = None,
    knowledge: Optional[KnowledgeGraph] = None,
    event: Optional[EventGraph] = None,
    detector: Optional[EmergenceDetector] = None,
    trust_edge_cap: int = 200,
) -> Dict[str, Any]:
    """Snapshot the lived society into a JSON-serializable dict (read-only)."""
    snap: Dict[str, Any] = {
        "turn": int(current_turn),
        "wired": True,
        "profiles": [],
        "social": {"trust_edges": [], "debt_edges": [], "factions": []},
        "institutions": [],
        "knowledge": {"diffusion": []},
        "emergence": [],
    }

    # -- profiles -----------------------------------------------------------
    if profiles:
        for uid, ps in profiles.items():
            snap["profiles"].append({
                "agent_id": uid,
                "role": ps.role,
                "top_traits": _top_traits(ps.profile),
                "mood": ps.mood.to_dict(),
            })

    # -- social graph -------------------------------------------------------
    if social is not None:
        for e in social.store.edges(TRUST)[:trust_edge_cap]:
            snap["social"]["trust_edges"].append({
                "src": e.src, "dst": e.dst,
                "strength": round(float(e.attrs.get("strength", 0.0)), 3),
            })
        for e in social.store.edges(DEBT)[:trust_edge_cap]:
            snap["social"]["debt_edges"].append({
                "src": e.src, "dst": e.dst,
                "strength": round(float(e.attrs.get("strength", 0.0)), 3),
            })

    # -- institutions (+ emergence verdict per rule) -----------------------
    inst_results = {}
    if institution is not None and detector is not None:
        for r in detector.detect_institutions(institution, current_turn=current_turn):
            inst_results[r.subject_id] = r
    if institution is not None:
        for rule_id, rs in institution.rules.items():
            res = inst_results.get(rule_id)
            snap["institutions"].append({
                "rule_id": rule_id,
                "description": rs.description,
                "adopted": rs.adopted,
                "supporters": sorted(rs.supporters),
                "opposers": sorted(rs.opposers),
                "enforcement_count": rs.enforcement_count,
                "violation_count": rs.violation_count,
                "emerged": bool(res.emerged) if res else False,
                "criteria": res.criteria if res else {},
            })

    # -- knowledge diffusion ------------------------------------------------
    if knowledge is not None:
        for node in knowledge.store.nodes("skill"):
            chain = knowledge.diffusion_chain(node.node_id)
            if chain:
                snap["knowledge"]["diffusion"].append({
                    "knowledge_id": node.node_id, "chain": chain,
                })

    # -- emergence (all detectors) -----------------------------------------
    if detector is not None:
        results = detector.run_all(
            current_turn=current_turn,
            institution=institution, knowledge=knowledge, event=event,
        )
        for kind, items in results.items():
            for r in items:
                snap["emergence"].append({
                    "kind": r.kind, "subject_id": r.subject_id,
                    "emerged": r.emerged, "criteria": r.criteria,
                })

    return snap
