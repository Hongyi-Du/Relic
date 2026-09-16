"""Protocol objects + events (DESIGN env_org §57/§58)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional


# §58 protocol event types
PROTOCOL_EVENT_TYPES = (
    "proposal", "support", "oppose", "adoption", "use",
    "violation", "enforcement", "amendment", "obsolete", "impact",
)

GOVERNED_OBJECT_TYPES = frozenset(
    {
        "pull_request",
        "release_candidate",
        "release",
        "issue",
        "workflow",
    }
)

GOVERNED_OBJECT_LIFECYCLE_STATES = {
    "pull_request": frozenset(
        {
            "open",
            "review_requested",
            "changes_requested",
            "approved",
            "merged",
            "closed",
            "stale",
        }
    ),
    "release_candidate": frozenset(
        {"draft", "under_review", "approved", "blocked", "released"}
    ),
    "release": frozenset({"staged", "published", "withdrawn"}),
    "issue": frozenset({"open", "in_progress", "resolved", "closed"}),
    "workflow": frozenset(
        {
            "candidate",
            "experimental",
            "known",
            "adopted",
            "widely_adopted",
            "active",
            "blocked",
            "completed",
        }
    ),
}

INDEPENDENT_OUTCOME_ORACLE_TYPES = frozenset(
    {
        "acceptance_test",
        "external_evaluator",
        "external_measurement",
    }
)


@dataclass(frozen=True)
class GovernedObjectSnapshot:
    """A point-in-time state of a concrete object governed by a protocol."""

    object_type: str
    object_id: str
    lifecycle_state: str
    observed_tick: int
    provenance_ref: str
    attributes: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class IndependentOutcomeOracle:
    """Outcome evidence produced outside the protocol event ledger."""

    oracle_id: str
    oracle_type: str
    evaluator_id: str
    governed_object_type: str
    governed_object_id: str
    enforcement_event_id: str
    governed_state_after_hash: str
    metric_name: str
    baseline_value: float
    observed_value: float
    favorable_direction: str
    observed_tick: int
    evidence_ref: str
    evidence_hash: str


@dataclass
class ProtocolEvent:
    event_id: str
    event_type: str
    protocol_id: str
    actor_id: str = ""
    tick: int = 0
    data: dict = field(default_factory=dict)


@dataclass
class Protocol:
    protocol_id: str
    protocol_type: str                       # task_ownership|experiment_logging|review_before_merge|meeting_notes_required|daily_sync_cadence|...
    proposer_id: str = ""
    proposal_event_id: str = ""
    rule_summary: str = ""
    scope: str = "review"                    # working_hours|after_hours|weekend|meeting|review|experiment|customer|budget|repo|external_community|payroll|hiring
    target_process: str = ""
    supporters: List[str] = field(default_factory=list)
    opposers: List[str] = field(default_factory=list)
    adoption_status: str = "proposed"        # proposed|adopted|contested|obsolete
    usage_events: List[str] = field(default_factory=list)
    violation_events: List[str] = field(default_factory=list)
    enforcement_events: List[str] = field(default_factory=list)
    first_tick: int = 0
    last_active_tick: int = 0
    persistence_ticks: int = 0
    impact_metrics: Dict[str, float] = field(default_factory=dict)
    independent_outcome_oracles: List[IndependentOutcomeOracle] = field(
        default_factory=list
    )
    emergence_level: str = "none"            # none|weak|strong
    status: str = "active"                   # active|obsolete


__all__ = [
    "GOVERNED_OBJECT_LIFECYCLE_STATES",
    "GOVERNED_OBJECT_TYPES",
    "INDEPENDENT_OUTCOME_ORACLE_TYPES",
    "PROTOCOL_EVENT_TYPES",
    "GovernedObjectSnapshot",
    "IndependentOutcomeOracle",
    "Protocol",
    "ProtocolEvent",
]
