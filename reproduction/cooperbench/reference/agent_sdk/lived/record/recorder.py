"""Deep snapshot + run recorder for the Lived Inspector frontend (§28).

The shallow :func:`agent_sdk.lived.perceive.view.lived_society_snapshot` is enough for a
summary panel, but the Inspector needs EVERYTHING, per turn, so the user can:
  * open every agent's every graph and click any node for its full attributes,
  * read every action's PolicyTrace (candidates, feature annotations, the
    six-component utility formula + per-feature contributions, shortlist,
    probabilities, the sampled draw, explanation paths),
  * scrub a recorded run back and forth (replay mode).

So this module provides:
  * :func:`lived_full_snapshot` — one DEEP frame: all five graphs in full
    (nodes+edges+attrs), every agent's full profile (incl. TraitNode provenance
    + history + grounding), episodes, emergence, and any decisions/events the
    caller passes for that turn.
  * :class:`RunRecorder` — accumulates frames into a replay file
    (``{meta, frames:[...]}``), JSON-serializable, loadable by the frontend.

Env-agnostic: takes graph/profile/episode/detector objects, returns plain data.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

from agent_sdk.lived.emergence import EmergenceDetector
from agent_sdk.lived.cognition.episode import EpisodeManager
from agent_sdk.lived.graphs import (
    EventGraph,
    InstitutionGraph,
    KnowledgeGraph,
    ProfileGraph,
    SocialGraph,
)
from agent_sdk.lived.persona.profile import ProfileState


# --------------------------------------------------------------------------- #
# Per-object serializers (full detail)
# --------------------------------------------------------------------------- #
def profile_detail(ps: ProfileState) -> Dict[str, Any]:
    """Full profile incl. the rich TraitNode provenance (base/current/plasticity/
    confidence/update_history) so the Inspector can show why a trait drifted."""
    d = ps.to_dict()
    traits_detail: Dict[str, Any] = {}
    nodes = ps.traits or {}
    for name in ps.profile.to_dict():
        node = nodes.get(name)
        if node is not None:
            traits_detail[name] = {
                "base_value": round(node.base_value, 4),
                "current_value": round(node.current_value, 4),
                "stability": node.stability,
                "plasticity": node.plasticity,
                "confidence": round(node.confidence, 4),
                "linked_action_features": list(node.linked_action_features),
                "update_history": list(node.update_history),
            }
        else:
            traits_detail[name] = {
                "current_value": round(float(getattr(ps.profile, name)), 4),
                "update_history": [],
            }
    d["traits_detail"] = traits_detail
    return d


def graph_detail(graph: Any) -> Dict[str, Any]:
    """Serialize any GraphStore-backed graph to {nodes:[...], edges:[...]}.

    Synthesizes a node for any edge endpoint missing from the node list (the
    base store's ``add_edge`` does not auto-create nodes — social/knowledge
    graphs accumulate edges without explicit nodes), so the frontend always has
    something to render and click."""
    store = getattr(graph, "store", graph)
    if not hasattr(store, "to_dict"):
        return {"nodes": [], "edges": []}
    d = store.to_dict()
    known = {n["id"] for n in d.get("nodes", [])}
    for e in d.get("edges", []):
        for end in (e["src"], e["dst"]):
            if end not in known:
                known.add(end)
                d["nodes"].append({"id": end, "type": "agent", "attrs": {}, "synthetic": True})
    return d


def persona_graph_full(pg: Optional[ProfileGraph], ps: ProfileState) -> Dict[str, Any]:
    """The agent's persona graph enriched with the FULL persona: all 12 stable
    traits (with base/current/plasticity/confidence/update_history), active
    transient states, and trait→action_feature ``supports`` ontology edges, on
    top of any seeded ProfileGraph structure. Every node is clickable with its
    full attributes (the §28 "all info, every node clickable" requirement)."""
    base = graph_detail(pg) if pg is not None else {"nodes": [], "edges": []}
    by_id = {n["id"]: n for n in base["nodes"]}

    def put(node_id: str, ntype: str, attrs: Dict[str, Any]) -> None:
        if node_id in by_id:
            by_id[node_id]["attrs"].update(attrs)
            if by_id[node_id].get("type") in (None, "", "agent"):
                by_id[node_id]["type"] = ntype
        else:
            n = {"id": node_id, "type": ntype, "attrs": dict(attrs)}
            by_id[node_id] = n
            base["nodes"].append(n)

    detail = profile_detail(ps).get("traits_detail", {})
    edge_keys = {(e["src"], e["dst"], e["type"]) for e in base["edges"]}
    for trait, td in detail.items():
        put(trait, "stable_trait", td)
        for feat in td.get("linked_action_features", []):
            put(feat, "action_feature", {})
            key = (trait, feat, "supports")
            if key not in edge_keys:
                edge_keys.add(key)
                base["edges"].append({"src": trait, "dst": feat, "type": "supports", "attrs": {}})
    # active transient states
    for name, val in ps.mood.to_dict().items():
        if abs(float(val)) > 1e-9:
            put(f"mood:{name}", "transient_state", {"value": round(float(val), 3)})
    return base


def _institution_detail(inst: InstitutionGraph) -> Dict[str, Any]:
    d = graph_detail(inst)
    d["rules"] = {rid: rs.to_dict() for rid, rs in inst.rules.items()}
    return d


def _event_detail(event: EventGraph) -> Dict[str, Any]:
    d = graph_detail(event)
    d["records"] = [e.to_dict() for e in event.events()]
    return d


# --------------------------------------------------------------------------- #
# Deep single-turn snapshot
# --------------------------------------------------------------------------- #
def lived_full_snapshot(
    *,
    turn: int = 0,
    profiles: Optional[Mapping[str, ProfileState]] = None,
    social: Optional[SocialGraph] = None,
    event: Optional[EventGraph] = None,
    knowledge: Optional[KnowledgeGraph] = None,
    institution: Optional[InstitutionGraph] = None,
    persona_graphs: Optional[Mapping[str, ProfileGraph]] = None,
    episodes: Optional[EpisodeManager] = None,
    detector: Optional[EmergenceDetector] = None,
    decisions: Optional[List[Dict[str, Any]]] = None,
    events: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """One DEEP frame the Inspector can fully render (read-only)."""
    frame: Dict[str, Any] = {
        "turn": int(turn),
        "wired": True,
        "agents": sorted(profiles.keys()) if profiles else [],
        "profiles": {uid: profile_detail(ps) for uid, ps in (profiles or {}).items()},
        "graphs": {
            "social": graph_detail(social) if social is not None else {"nodes": [], "edges": []},
            "event": _event_detail(event) if event is not None else {"nodes": [], "edges": [], "records": []},
            "knowledge": graph_detail(knowledge) if knowledge is not None else {"nodes": [], "edges": []},
            "institution": _institution_detail(institution) if institution is not None else {"nodes": [], "edges": [], "rules": {}},
            "persona": {
                uid: persona_graph_full((persona_graphs or {}).get(uid), ps)
                for uid, ps in (profiles or {}).items()
            },
        },
        "episodes": episodes.snapshot() if episodes is not None else [],
        "decisions": list(decisions or []),
        "events": list(events or []),
        "emergence": [],
    }
    if detector is not None:
        results = detector.run_all(current_turn=turn, institution=institution,
                                   knowledge=knowledge, event=event)
        for kind, items in results.items():
            for r in items:
                frame["emergence"].append({
                    "kind": r.kind, "subject_id": r.subject_id,
                    "emerged": r.emerged, "criteria": r.criteria,
                })
    return frame


# --------------------------------------------------------------------------- #
# Run recorder -> replay file
# --------------------------------------------------------------------------- #
@dataclass
class RunRecorder:
    """Accumulates per-turn deep frames into a replay file."""
    name: str = "run"
    scenario: str = ""
    frames: List[Dict[str, Any]] = field(default_factory=list)
    created: float = field(default_factory=time.time)

    def capture(self, **snapshot_kwargs: Any) -> Dict[str, Any]:
        """Build + append a deep frame (args == :func:`lived_full_snapshot`)."""
        frame = lived_full_snapshot(**snapshot_kwargs)
        self.frames.append(frame)
        return frame

    def add_frame(self, frame: Dict[str, Any]) -> None:
        self.frames.append(frame)

    def to_dict(self) -> Dict[str, Any]:
        agents = sorted({a for f in self.frames for a in f.get("agents", [])})
        return {
            "meta": {
                "name": self.name,
                "scenario": self.scenario,
                "created": self.created,
                "turns": len(self.frames),
                "agents": agents,
            },
            "frames": self.frames,
        }

    def save(self, path: str) -> str:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, ensure_ascii=False, indent=2)
        return path
