"""Event / Memory Graph (design doc §4) — the system's fact base.

Every important behaviour is structured here (§4.2 event schema). Downstream
consumers: memory retrieval, profile updates, the other graphs, the emergence
detector, metrics and case replay (§4.3).

This is the spine the whole platform stands on, so the event schema is frozen
in :class:`EventRecord`. The store is append-only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agent_sdk.lived.graphs.base import GraphStore


@dataclass
class EventRecord:
    """One structured event (§4.2). ``graph_updates`` records which edges this
    event caused in other graphs so replay can reconstruct state."""
    event_id: str
    timestamp: int
    actor: str
    action_type: str
    target: Optional[str] = None
    location: Optional[Any] = None
    resources: Dict[str, int] = field(default_factory=dict)
    precondition_state: Dict[str, Any] = field(default_factory=dict)
    result: str = ""
    success: bool = True
    social_consequence: str = ""
    graph_updates: List[str] = field(default_factory=list)
    summary: str = ""  # natural-language summary

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id, "timestamp": self.timestamp, "actor": self.actor,
            "action_type": self.action_type, "target": self.target, "location": self.location,
            "resources": dict(self.resources), "precondition_state": dict(self.precondition_state),
            "result": self.result, "success": self.success,
            "social_consequence": self.social_consequence,
            "graph_updates": list(self.graph_updates), "summary": self.summary,
        }


class EventGraph:
    """Append-only event log + actor/target adjacency (§4).

    Nodes: agents + event nodes. Edges: actor --did--> event --on--> target.
    Provides the queries §4.3 lists (who helped whom, who broke promises, ...).
    """

    def __init__(self):
        self.store = GraphStore(name="event")
        self._events: List[EventRecord] = []
        self._seq = 0

    def record(self, ev: EventRecord) -> EventRecord:
        if not ev.event_id:
            ev.event_id = f"ev_{self._seq}"
        self._seq += 1
        self._events.append(ev)
        self.store.add_node(ev.event_id, "event", **ev.to_dict())
        self.store.add_node(ev.actor, "agent")
        self.store.add_edge(ev.actor, ev.event_id, "did", t=ev.timestamp)
        if ev.target:
            self.store.add_node(ev.target, "agent")
            self.store.add_edge(ev.event_id, ev.target, "on")
        return ev

    def new_event(self, **kwargs: Any) -> EventRecord:
        """Convenience: build + record in one call (event_id auto-assigned)."""
        ev = EventRecord(event_id="", timestamp=int(kwargs.pop("timestamp", 0)),
                         actor=str(kwargs.pop("actor", "")),
                         action_type=str(kwargs.pop("action_type", "")), **kwargs)
        return self.record(ev)

    # -- §4.3 queries -------------------------------------------------------
    def events(self) -> List[EventRecord]:
        return list(self._events)

    def by_actor(self, actor: str) -> List[EventRecord]:
        return [e for e in self._events if e.actor == actor]

    def by_action(self, action_type: str) -> List[EventRecord]:
        return [e for e in self._events if e.action_type == action_type]

    def between(self, actor: str, target: str) -> List[EventRecord]:
        return [e for e in self._events if e.actor == actor and e.target == target]
