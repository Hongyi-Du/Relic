"""Minimal in-memory graph store backing all five lived graphs (§4-7).

A deliberately tiny adjacency-map store — enough to satisfy
``agent_sdk.lived.core.ports.GraphStorePort`` and let the typed graphs (social /
event / knowledge / institution / profile) and the emergence detector query
nodes and edges. Swap for a real graph DB later without touching callers.

Design intent: graphs are the *persistent fact base* (§4.1). Edges carry
typed attributes (e.g. trust strength, timestamp) so updates accumulate
rather than overwrite.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass
class Node:
    node_id: str
    ntype: str
    attrs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Edge:
    src: str
    dst: str
    etype: str
    attrs: Dict[str, Any] = field(default_factory=dict)


class GraphStore:
    """In-memory typed multigraph. Edges keyed by (src, dst, etype)."""

    def __init__(self, name: str = ""):
        self.name = name
        self._nodes: Dict[str, Node] = {}
        self._edges: Dict[Tuple[str, str, str], Edge] = {}

    # -- nodes --------------------------------------------------------------
    def add_node(self, node_id: str, ntype: str, **attrs: Any) -> None:
        n = self._nodes.get(node_id)
        if n is None:
            self._nodes[node_id] = Node(node_id=node_id, ntype=ntype, attrs=dict(attrs))
        else:
            n.attrs.update(attrs)

    def get_node(self, node_id: str) -> Optional[Node]:
        return self._nodes.get(node_id)

    def nodes(self, ntype: str | None = None) -> List[Node]:
        if ntype is None:
            return list(self._nodes.values())
        return [n for n in self._nodes.values() if n.ntype == ntype]

    # -- edges --------------------------------------------------------------
    def add_edge(self, src: str, dst: str, etype: str, **attrs: Any) -> None:
        """Create or merge an edge. Numeric attrs that already exist are
        ADDED (accumulating trust/debt); others overwrite."""
        key = (src, dst, etype)
        e = self._edges.get(key)
        if e is None:
            self._edges[key] = Edge(src=src, dst=dst, etype=etype, attrs=dict(attrs))
        else:
            for k, v in attrs.items():
                if isinstance(v, (int, float)) and isinstance(e.attrs.get(k), (int, float)):
                    e.attrs[k] = e.attrs[k] + v
                else:
                    e.attrs[k] = v

    def get_edge(self, src: str, dst: str, etype: str) -> Optional[Edge]:
        return self._edges.get((src, dst, etype))

    def edges(self, etype: str | None = None) -> List[Edge]:
        if etype is None:
            return list(self._edges.values())
        return [e for e in self._edges.values() if e.etype == etype]

    def neighbors(self, node_id: str, etype: str | None = None) -> List[str]:
        out: List[str] = []
        for (src, dst, et), _ in self._edges.items():
            if src == node_id and (etype is None or et == etype):
                out.append(dst)
        return out

    def edge_attr(self, src: str, dst: str, etype: str, key: str, default: Any = None) -> Any:
        e = self._edges.get((src, dst, etype))
        return default if e is None else e.attrs.get(key, default)

    # -- serialization ------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "nodes": [{"id": n.node_id, "type": n.ntype, "attrs": n.attrs}
                      for n in self._nodes.values()],
            "edges": [{"src": e.src, "dst": e.dst, "type": e.etype, "attrs": e.attrs}
                      for e in self._edges.values()],
        }
