"""OrgEventGraph (DESIGN env_org O1 §12) — typed nodes + edges for every action.

Downstream graph-prediction / graph-intervention experiments (O5) depend on
these node + edge types, so they are emitted from the start. Nodes are typed by
their id prefix or an explicit type; edges come from ExecutionResult.graph_edges.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

NODE_TYPES = {
    "agent", "task", "doc", "file", "message", "meeting", "decision", "action_item",
    "repo_branch", "commit", "pr", "experiment", "run", "result", "sandbox_job",
    "external_post", "external_profile", "protocol", "artifact", "cost_event",
    "payroll_event", "offer", "search", "action",
    # O1.7 policy-grounded text objects
    "commitment", "dispute", "request",
    # reflection layer
    "episode", "reflection", "wish", "tool",
}
EDGE_TYPES = {
    "sent", "received", "read", "mentioned", "shared", "opened", "edited", "created",
    "assigned", "owned", "worked_on", "reviewed", "approved", "requested_changes",
    "merged", "ran", "produced", "exported_to_tracker", "cited", "attended", "decided",
    "supported", "opposed", "used_protocol", "violated_protocol", "enforced_protocol",
    "paid", "delayed_payment", "helped", "blocked", "conflicted_with", "searched",
    "retrieved", "performed",
    # O1.7 text-action edges
    "challenged", "warned_about", "promised", "committed_to", "requested",
    "requested_review_from", "disputed", "violated", "proposed", "reminded",
    # reflection layer edges
    "reflected", "produced_wish", "involved", "triggered_by", "produced", "related_to",
    "governed_by", "enforced_via",
    # product/patch/release/governance edges emitted by world.py long before the
    # strict_edges whitelist existed (42eca02 omitted them; a formal run crashed on
    # the first patch_applied event with unknown_formal_event_graph_edge:patched).
    # tests/org_env/test_event_graph_edge_parity.py keeps emitters and this
    # whitelist in sync from now on.
    "patched", "merged_into", "released", "trialed_by", "review_requested",
    "proposed_amendment",
}

# id-prefix -> node type (best-effort typing). NOTE order: commitment_ before commit_.
_PREFIX = {
    "task_": "task", "doc_": "doc", "msg_": "message", "meeting_": "meeting",
    "branch_": "repo_branch", "commitment_": "commitment", "commit_": "commit",
    "pr_": "pr", "exp": "experiment",
    "run_": "run", "result_": "result", "job_": "sandbox_job", "post_": "external_post",
    "ext_": "external_profile", "proto_": "protocol", "cost_": "cost_event",
    "issue_": "task", "search_": "search", "act_": "action", "mnote_": "doc",
    "ledger": "artifact", "att_": "file", "dispute_": "dispute", "request_": "request",
    "refl_": "reflection", "wish_": "wish", "ep_": "episode",
}


def _node_type(node_id: str, agent_ids: Set[str]) -> str:
    if node_id in agent_ids:
        return "agent"
    for pre, t in _PREFIX.items():
        if str(node_id).startswith(pre):
            return t
    return "action"


@dataclass
class OrgEventGraph:
    agent_ids: Set[str] = field(default_factory=set)
    nodes: Dict[str, str] = field(default_factory=dict)        # node_id -> type
    edges: List[Tuple[str, str, str]] = field(default_factory=list)  # (src, edge_type, dst)
    edge_set: Set[Tuple[str, str, str]] = field(default_factory=set)
    strict_edges: bool = False

    def add_node(self, node_id: str, node_type: str = "") -> None:
        if node_id is None:
            return
        self.nodes[node_id] = node_type or self.nodes.get(node_id) or _node_type(node_id, self.agent_ids)

    def add_edge(self, src: str, edge_type: str, dst: str) -> None:
        if src is None or dst is None:
            return
        self.add_node(src); self.add_node(dst)
        key = (src, edge_type, dst)
        if key not in self.edge_set:
            self.edge_set.add(key)
            self.edges.append(key)

    def ingest(self, execution_result) -> None:
        self.add_node(execution_result.agent_id, "agent")
        for oid in execution_result.created_objects:
            self.add_node(oid)
        for oid in execution_result.modified_objects:
            self.add_node(oid)
        for (src, etype, dst) in execution_result.graph_edges:
            if etype not in EDGE_TYPES and self.strict_edges:
                raise ValueError(f"unknown_formal_event_graph_edge:{etype}")
            self.add_edge(
                src,
                etype if etype in EDGE_TYPES else "performed",
                dst,
            )

    def summary(self) -> dict:
        from collections import Counter
        ntypes = Counter(self.nodes.values())
        etypes = Counter(e[1] for e in self.edges)
        return {"num_nodes": len(self.nodes), "num_edges": len(self.edges),
                "node_types": dict(ntypes), "edge_types": dict(etypes)}


__all__ = ["OrgEventGraph", "NODE_TYPES", "EDGE_TYPES"]
