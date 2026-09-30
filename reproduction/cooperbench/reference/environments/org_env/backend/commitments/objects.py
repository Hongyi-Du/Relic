"""Commitment / dispute / requested-action objects (OrgEnv O1.7, spec §27).

These are the structured world-state objects that policy-grounded TEXT creates:
``promise_work`` -> :class:`CommitmentObject`, ``challenge_result`` ->
:class:`ClaimDispute`, ``ask_for_review`` / ``request_reproduction`` ->
:class:`RequestedAction`. They live in the world (not the message text) so the
event graph + detectors + retention model can reason over them, and an unmet
commitment can later raise a PromiseViolationEvent + trust hit.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class CommitmentObject:
    commitment_id: str
    agent_id: str
    description: str = ""
    target_object_id: Optional[str] = None
    linked_task_id: Optional[str] = None
    audience: List[str] = field(default_factory=list)
    created_tick: int = 0
    due_tick: Optional[int] = None
    status: str = "open"             # open | fulfilled | violated
    source_message_id: Optional[str] = None


@dataclass
class ClaimDispute:
    dispute_id: str
    challenger_id: str
    target_object_id: str
    target_object_type: str = "result"
    owner_id: Optional[str] = None   # v8d P1c: who owns the disputed result (should respond)
    reason: str = ""                 # v8d P1c: human-readable problem (from issue_tags)
    linked_issue_id: Optional[str] = None  # v8d P1c: the escalation issue, so it can be closed
    claim_summary: str = ""
    issue_tags: List[str] = field(default_factory=list)
    evidence_requested: bool = True
    created_tick: int = 0
    status: str = "open"             # open | resolved | escalated | withdrawn
    source_message_id: Optional[str] = None
    # #4: one dispute per (target, problem); repeat challengers become supporters, and the
    # dispute reaches a closure (resolved on evidence, or escalated after latency).
    supporter_ids: List[str] = field(default_factory=list)
    support_count: int = 1
    evidence_request_count: int = 1
    last_activity_tick: int = 0
    resolution: str = ""             # "" | reproduced | evidence_provided | escalated
    resolved_tick: Optional[int] = None
    resolved_by: Optional[str] = None


@dataclass
class RequestedAction:
    request_id: str
    requester_id: str
    target_agent_id: Optional[str] = None
    action_requested: str = "review"   # review | reproduction | evidence | changes
    target_object_id: Optional[str] = None
    created_tick: int = 0
    due_tick: Optional[int] = None
    status: str = "open"             # open | fulfilled | expired
    source_message_id: Optional[str] = None


class CommitmentRegistry:
    """Tracks commitments / disputes / requests + detects overdue promises."""

    def __init__(self) -> None:
        self.commitments: Dict[str, CommitmentObject] = {}
        self.disputes: Dict[str, ClaimDispute] = {}
        self.requests: Dict[str, RequestedAction] = {}
        self._seq = 0

    def _id(self, p: str) -> str:
        self._seq += 1
        return f"{p}_{self._seq}"

    def add_commitment(self, **kw) -> CommitmentObject:
        c = CommitmentObject(commitment_id=self._id("commitment"), **kw)
        self.commitments[c.commitment_id] = c
        return c

    def add_dispute(self, **kw) -> ClaimDispute:
        d = ClaimDispute(dispute_id=self._id("dispute"), **kw)
        self.disputes[d.dispute_id] = d
        return d

    def find_open_dispute(self, target_object_id: str, issue_tags=None):
        """#4: an existing open/escalated dispute on the same target (and overlapping
        problem tag) — so a repeat challenge supports it instead of forking a new one."""
        tags = set(issue_tags or [])
        for d in self.disputes.values():
            if d.status in ("open", "escalated") and d.target_object_id == target_object_id:
                if not tags or not d.issue_tags or (set(d.issue_tags) & tags):
                    return d
        return None

    def find_dispute_any_status(self, target_object_id: str, issue_tags=None):
        """v8d P1c: ANY prior dispute on the same (target, problem), regardless of status —
        so a settled concern is not re-litigated into a brand-new dispute every time the
        result is seen again."""
        tags = set(issue_tags or [])
        for d in self.disputes.values():
            if d.target_object_id != target_object_id:
                continue
            if not tags or not d.issue_tags or (set(d.issue_tags) & tags):
                return d
        return None

    def add_request(self, **kw) -> RequestedAction:
        r = RequestedAction(request_id=self._id("request"), **kw)
        self.requests[r.request_id] = r
        return r

    def overdue_commitments(self, tick: int) -> List[CommitmentObject]:
        return [c for c in self.commitments.values()
                if c.status == "open" and c.due_tick is not None and tick > c.due_tick]


__all__ = ["CommitmentObject", "ClaimDispute", "RequestedAction", "CommitmentRegistry"]
