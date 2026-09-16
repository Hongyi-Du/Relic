"""OrgEventAppraisal (DESIGN env_org O1 §11) — ExecutionResult -> AppraisedEvents.

Classifies each raw event a chosen action produced into one of the §11 event
types, assigns a salience (for memory ranking), and routes it to logs / memory /
replay / event graph / protocol detectors (the OrgWorld.step loop does the
routing; this module produces the appraised list).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

# §11 event taxonomy (raw event "type" already uses these names where possible).
EVENT_TYPES = {
    "task_progress_event", "communication_event", "file_share_event", "meeting_event",
    "repo_event", "sandbox_event", "experiment_event", "search_event",
    "external_signal_event", "budget_event", "payroll_event", "retention_event",
    "protocol_proposal_event", "protocol_use_event", "protocol_violation_event",
    "protocol_enforcement_event", "protocol_support_event", "conflict_event",
    "help_event", "handoff_event", "overtime_event", "weekend_work_event", "action_event",
    # O1.6 work-state events
    "recovery_event", "background_job_event",
    # O1.7 policy-grounded text events
    "speech_act_event", "commitment_event", "claim_dispute_event", "requested_action_event",
    "object_appraisal_event", "feedback_decision_event", "text_generation_event",
    # governance (proposal approval/rejection by designated approvers)
    "governance_event", "proposal_event",
    # v5 product release lifecycle
    "release_event",
}

# salience weight per event family (memory ranking).
SALIENCE = {
    "protocol_proposal_event": 0.9, "protocol_violation_event": 0.9,
    "protocol_enforcement_event": 0.85, "protocol_use_event": 0.6,
    "experiment_event": 0.7, "payroll_event": 0.8, "retention_event": 0.85,
    "external_signal_event": 0.6, "meeting_event": 0.6, "repo_event": 0.55,
    "file_share_event": 0.5, "task_progress_event": 0.5, "communication_event": 0.4,
    "search_event": 0.3, "overtime_event": 0.4, "action_event": 0.2,
    "recovery_event": 0.3, "background_job_event": 0.4,
    "claim_dispute_event": 0.8, "commitment_event": 0.7, "requested_action_event": 0.55,
    "speech_act_event": 0.45, "object_appraisal_event": 0.5, "feedback_decision_event": 0.45,
    "text_generation_event": 0.3,
    "governance_event": 0.7, "proposal_event": 0.7, "release_event": 0.85,
}


@dataclass
class AppraisedEvent:
    event_type: str
    agent_id: str
    tick: int
    salience: float = 0.4
    payload: Dict[str, Any] = field(default_factory=dict)


class OrgEventAppraisalImpl:
    def appraise(self, execution_result: Any, org_world: Any) -> List[AppraisedEvent]:
        out: List[AppraisedEvent] = []
        for ev in execution_result.events:
            etype = ev.get("type", "action_event")
            if etype not in EVENT_TYPES:
                etype = "action_event"
            out.append(AppraisedEvent(
                event_type=etype, agent_id=ev.get("agent_id", execution_result.agent_id),
                tick=ev.get("tick", 0), salience=SALIENCE.get(etype, 0.3), payload=dict(ev)))
        if not execution_result.success:
            out.append(AppraisedEvent(event_type="action_event",
                                      agent_id=execution_result.agent_id, tick=0, salience=0.2,
                                      payload={"failed": execution_result.action_type,
                                               "reason": execution_result.failure_reason}))
        return out


__all__ = ["AppraisedEvent", "OrgEventAppraisalImpl", "EVENT_TYPES", "SALIENCE"]
