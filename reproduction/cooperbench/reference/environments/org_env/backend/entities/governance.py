"""OrgEnv governance objects — SharedBoard / WorkflowArtifact / Protocol
(DESIGN env_org §33.1).

These are the emergence targets: WorkflowArtifact is the org analogue of nature's
material prototype (created via create_workflow_artifact), and Protocol is the
org institution detected by the Core civic mechanism (§19/§24).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SharedBoard:
    board_id: str
    tasks: List[str] = field(default_factory=list)
    issues: List[str] = field(default_factory=list)
    owners: Dict[str, str] = field(default_factory=dict)
    tags: Dict[str, List[str]] = field(default_factory=dict)
    priority_labels: Dict[str, int] = field(default_factory=dict)
    protocol_links: List[str] = field(default_factory=list)
    update_history: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class WorkflowArtifact:
    """Org-domain analogue of nature's material prototype (mechanism: core §18)."""
    artifact_id: str
    artifact_type: str = ""   # experiment_tracker|cost_ledger|task_board|review_checklist|handoff_template|customer_triage_sheet|SOP|automation_script|claim_evidence_table
    title: str = ""
    creator_id: Optional[str] = None
    intended_function: str = ""
    target_bottleneck: str = ""
    content_schema: Dict[str, Any] = field(default_factory=dict)
    users: List[str] = field(default_factory=list)
    usage_count: int = 0
    adoption_status: str = "candidate"   # candidate|experimental|known|adopted|widely_adopted
    linked_protocol_id: Optional[str] = None
    effectiveness_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class Protocol:
    """Org institution — composed/detected via the Core civic mechanism (§19/§24)."""
    protocol_id: str
    protocol_type: str = ""   # budget_approval|task_ownership|review_protocol|handoff_protocol|experiment_logging_rule|customer_escalation|meeting_cadence|documentation_norm
    proposal_event_id: Optional[str] = None
    proposer_id: Optional[str] = None
    supporters: List[str] = field(default_factory=list)
    opposers: List[str] = field(default_factory=list)
    rule_summary: str = ""
    target_resource_or_process: str = ""
    adoption_status: str = "proposed"
    usage_events: List[Dict[str, Any]] = field(default_factory=list)
    violation_events: List[Dict[str, Any]] = field(default_factory=list)
    enforcement_events: List[Dict[str, Any]] = field(default_factory=list)
    persistence_ticks: int = 0
    impact_metrics: Dict[str, float] = field(default_factory=dict)


__all__ = ["SharedBoard", "WorkflowArtifact", "Protocol"]
