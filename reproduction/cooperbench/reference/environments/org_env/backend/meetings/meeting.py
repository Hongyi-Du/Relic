"""Meeting objects (DESIGN env_org §11-§13/§37)."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class MeetingStatus(str, Enum):
    SCHEDULED = "scheduled"
    ACTIVE = "active"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"


@dataclass
class Meeting:
    meeting_id: str
    meeting_type: str = "ad_hoc_discussion"   # daily_sync|planning_meeting|design_review|experiment_review|customer_triage|incident_meeting|protocol_discussion|one_on_one|postmortem|ad_hoc_discussion|launch_readiness_check
    title: str = ""
    scheduled_tick: int = 0
    start_tick: Optional[int] = None
    end_tick: Optional[int] = None
    duration: int = 1
    participants: List[str] = field(default_factory=list)
    required_participants: List[str] = field(default_factory=list)
    optional_participants: List[str] = field(default_factory=list)
    attendees: List[str] = field(default_factory=list)
    skipped_by: List[str] = field(default_factory=list)
    agenda: List[str] = field(default_factory=list)
    created_by: Optional[str] = None
    room_channel_id: Optional[str] = None
    shared_artifacts: List[str] = field(default_factory=list)
    notes_doc_id: Optional[str] = None
    decision_ids: List[str] = field(default_factory=list)
    action_item_ids: List[str] = field(default_factory=list)
    follow_up_tasks: List[str] = field(default_factory=list)
    fatigue_cost: float = 0.1
    coordination_gain: float = 0.0
    interruption_cost: float = 0.05
    notes_missing: bool = False
    status: MeetingStatus = MeetingStatus.SCHEDULED


@dataclass
class MeetingRoom:
    room_id: str
    meeting_id: str
    participants: List[str] = field(default_factory=list)
    message_ids: List[str] = field(default_factory=list)
    shared_object_ids: List[str] = field(default_factory=list)
    pinned_items: List[str] = field(default_factory=list)


@dataclass
class MeetingNote:
    doc_id: str
    meeting_id: str
    author_id: str = ""
    summary: str = ""
    decisions: List[str] = field(default_factory=list)
    action_items: List[str] = field(default_factory=list)
    unresolved_questions: List[str] = field(default_factory=list)
    linked_objects: List[str] = field(default_factory=list)
    participants: List[str] = field(default_factory=list)
    created_tick: int = 0


@dataclass
class DecisionRecord:
    decision_id: str
    meeting_id: str
    decision_summary: str = ""
    decided_by: List[str] = field(default_factory=list)
    affected_objects: List[str] = field(default_factory=list)
    rationale: str = ""
    dissenting_agents: List[str] = field(default_factory=list)
    review_tick: Optional[int] = None


@dataclass
class ActionItem:
    action_item_id: str
    meeting_id: str
    description: str = ""
    assignee_id: Optional[str] = None
    due_tick: Optional[int] = None
    linked_task_id: Optional[str] = None
    status: str = "open"             # open | done | overdue
    reassigned_count: int = 0        # v8d P2a: bumped when an overdue item is reassigned
    # Gates this item exists to TRACK (auto blocker-tracking items only). Empty for
    # agent-created work items — auto-close on gate clear keys on this tag, never on
    # the gate name merely appearing in the free-text description.
    linked_gates: List[str] = field(default_factory=list)


__all__ = ["MeetingStatus", "Meeting", "MeetingRoom", "MeetingNote",
           "DecisionRecord", "ActionItem"]
