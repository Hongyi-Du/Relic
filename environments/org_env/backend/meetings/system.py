"""MeetingSystem (DESIGN env_org §11-§15/§37).

A meeting is a synchronous collaboration space: scheduling creates a MeetingRoom
backed by a comm channel; objects are *shared by reference* into the room; on
close it must produce MeetingNote / DecisionRecord / ActionItem — and if no notes
were recorded it flags ``meeting_note_missing`` (a protocol-wish trigger, §15).
"""
from __future__ import annotations

from typing import Dict, List, Optional

from environments.org_env.backend.comm.system import CommunicationSystem
from environments.org_env.backend.meetings.meeting import (
    ActionItem,
    DecisionRecord,
    Meeting,
    MeetingNote,
    MeetingRoom,
    MeetingStatus,
)

# Meeting duration in ticks (O1.6 §13). attend_meeting blocks a participant for it.
MEETING_DURATION = {
    "daily_sync": 1, "experiment_review": 1, "planning_meeting": 2, "customer_triage": 1,
    "incident_meeting": 1, "protocol_discussion": 1, "postmortem": 2,
    "launch_readiness_check": 1, "sync": 1,
}

# sub-actions only valid inside an active meeting (O1.6 §13).
MEETING_SUB_ACTIONS = {
    "share_object_in_meeting", "ask_question", "present_report", "propose_decision",
    "assign_action_item", "record_meeting_notes", "summarize_decision", "discuss_issue",
}


def meeting_duration(meeting_type: str) -> int:
    return int(MEETING_DURATION.get(meeting_type, 1))


class MeetingSystem:
    def __init__(self, comm: Optional[CommunicationSystem] = None):
        self.comm = comm
        self.meetings: Dict[str, Meeting] = {}
        self.rooms: Dict[str, MeetingRoom] = {}
        self.notes: Dict[str, MeetingNote] = {}
        self.decisions: Dict[str, DecisionRecord] = {}
        self.action_items: Dict[str, ActionItem] = {}
        self.events: List[dict] = []
        self._seq = 0

    def _id(self, prefix: str) -> str:
        self._seq += 1
        return f"{prefix}_{self._seq}"

    # -- lifecycle ----------------------------------------------------------
    def schedule_meeting(self, *, created_by: str, meeting_type: str, title: str,
                         participants: List[str], scheduled_tick: int = 0,
                         agenda: Optional[List[str]] = None,
                         required_participants: Optional[List[str]] = None) -> Meeting:
        mid = self._id("meeting")
        room_channel = f"meeting_{mid}"
        m = Meeting(meeting_id=mid, meeting_type=meeting_type, title=title,
                    scheduled_tick=scheduled_tick, participants=list(participants),
                    required_participants=list(required_participants or participants),
                    agenda=list(agenda or []), created_by=created_by,
                    room_channel_id=room_channel, status=MeetingStatus.SCHEDULED)
        self.meetings[mid] = m
        room = MeetingRoom(room_id=room_channel, meeting_id=mid, participants=list(participants))
        self.rooms[mid] = room
        if self.comm is not None:
            self.comm.create_channel(room_channel, channel_type="meeting_room_channel",
                                     members=set(participants))
        return m

    def start_meeting(self, meeting_id: str, tick: int = 0) -> None:
        m = self.meetings.get(meeting_id)
        if m:
            m.status = MeetingStatus.ACTIVE
            m.start_tick = tick

    def attend(self, agent_id: str, meeting_id: str) -> bool:
        m = self.meetings.get(meeting_id)
        if not m or agent_id not in m.participants:
            return False
        if agent_id not in m.attendees:
            m.attendees.append(agent_id)
        if self.comm is not None and m.room_channel_id:
            self.comm.join(m.room_channel_id, agent_id)
        return True

    def skip_meeting(self, agent_id: str, meeting_id: str) -> None:
        m = self.meetings.get(meeting_id)
        if m and agent_id not in m.skipped_by:
            m.skipped_by.append(agent_id)
            self.events.append({"type": "meeting_skipped", "meeting_id": meeting_id,
                                "agent_id": agent_id})

    def share_object_in_meeting(self, *, agent_id: str, meeting_id: str, object_id: str,
                                attachment_type: str, title: str = "", content_hash: str = "",
                                tick: int = 0) -> bool:
        m = self.meetings.get(meeting_id)
        room = self.rooms.get(meeting_id)
        if not m or not room:
            return False
        room.shared_object_ids.append(object_id)
        m.shared_artifacts.append(object_id)
        if self.comm is not None and m.room_channel_id:
            msg = self.comm.share_object(sender_id=agent_id, channel_id=m.room_channel_id,
                                         object_id=object_id, attachment_type=attachment_type,
                                         title=title, content_hash=content_hash, tick=tick)
            room.message_ids.append(msg.message_id)
        return True

    # -- outputs ------------------------------------------------------------
    def record_meeting_notes(self, *, agent_id: str, meeting_id: str, summary: str,
                             decisions: Optional[List[str]] = None,
                             unresolved: Optional[List[str]] = None, tick: int = 0) -> MeetingNote:
        m = self.meetings[meeting_id]
        note_id = self._id("mnote")
        note = MeetingNote(doc_id=note_id, meeting_id=meeting_id, author_id=agent_id,
                           summary=summary, decisions=list(decisions or []),
                           unresolved_questions=list(unresolved or []),
                           participants=list(m.attendees or m.participants), created_tick=tick)
        self.notes[note_id] = note
        m.notes_doc_id = note_id
        m.notes_missing = False
        for d in (decisions or []):
            did = self._id("decision")
            self.decisions[did] = DecisionRecord(decision_id=did, meeting_id=meeting_id,
                                                 decision_summary=d, decided_by=list(m.attendees))
            m.decision_ids.append(did)
            note.decisions.append(did) if did not in note.decisions else None
            # spec #7 + v8d P2a: a meeting decision must produce a concrete action item, and
            # the owner rotates across attendees (so it isn't always the chair / Paul).
            roster = list(m.attendees or m.participants) or [agent_id]
            assignee = roster[len(m.action_item_ids) % len(roster)]
            self.assign_action_item(meeting_id=meeting_id, description=f"follow up: {str(d)[:60]}",
                                    assignee_id=assignee, due_tick=tick + 24)
            note.action_items.append(m.action_item_ids[-1])
        return note

    def assign_action_item(self, *, meeting_id: str, description: str,
                           assignee_id: Optional[str] = None, due_tick: Optional[int] = None,
                           linked_task_id: Optional[str] = None,
                           linked_gates: Optional[List[str]] = None) -> ActionItem:
        aid = self._id("aitem")
        ai = ActionItem(action_item_id=aid, meeting_id=meeting_id, description=description,
                        assignee_id=assignee_id, due_tick=due_tick, linked_task_id=linked_task_id,
                        linked_gates=list(linked_gates or []))
        self.action_items[aid] = ai
        self.meetings[meeting_id].action_item_ids.append(aid)
        return ai

    def close_meeting(self, meeting_id: str, tick: int = 0) -> Meeting:
        m = self.meetings[meeting_id]
        m.status = MeetingStatus.COMPLETED
        m.end_tick = tick
        if m.notes_doc_id is None:
            m.notes_missing = True
            self.events.append({"type": "meeting_note_missing", "meeting_id": meeting_id,
                                "tick": tick})   # protocol-wish trigger (§15/§47)
        return m

    # -- O1.6 lifecycle helpers (driven by OrgWorld.step) ------------------
    def due_to_start(self, tick: int) -> List[Meeting]:
        return [m for m in self.meetings.values()
                if m.status == MeetingStatus.SCHEDULED and m.scheduled_tick <= tick]

    def due_to_close(self, tick: int) -> List[Meeting]:
        out = []
        for m in self.meetings.values():
            if m.status != MeetingStatus.ACTIVE:
                continue
            dur = meeting_duration(m.meeting_type)
            if m.start_tick is not None and tick >= m.start_tick + dur:
                out.append(m)
        return out

    def active_meeting_for(self, agent_id: str):
        for m in self.meetings.values():
            if m.status == MeetingStatus.ACTIVE and agent_id in m.participants:
                return m
        return None


__all__ = ["MeetingSystem", "MEETING_DURATION", "MEETING_SUB_ACTIONS", "meeting_duration"]
