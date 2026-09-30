"""CalendarSystem (DESIGN env_org §7/§36)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class CalendarEvent:
    calendar_event_id: str
    event_type: str                  # meeting|deadline|payroll|funding_tranche|launch|demo|planned_work_block|personal_unavailable|external_event|incident_window
    title: str = ""
    start_tick: int = 0
    end_tick: int = 0
    participants: List[str] = field(default_factory=list)
    required_participants: List[str] = field(default_factory=list)
    optional_participants: List[str] = field(default_factory=list)
    location_or_channel: Optional[str] = None
    linked_objects: List[str] = field(default_factory=list)
    created_by: Optional[str] = None
    status: str = "scheduled"
    recurrence: Optional[int] = None     # tick interval, None = one-off


class CalendarSystem:
    def __init__(self):
        self.events: dict = {}
        self._seq = 0

    def add_event(self, *, event_type: str, title: str, start_tick: int, end_tick: int,
                  participants: Optional[List[str]] = None, created_by: Optional[str] = None,
                  recurrence: Optional[int] = None) -> CalendarEvent:
        self._seq += 1
        eid = f"cal_{self._seq}"
        ev = CalendarEvent(calendar_event_id=eid, event_type=event_type, title=title,
                           start_tick=start_tick, end_tick=end_tick,
                           participants=list(participants or []), created_by=created_by,
                           recurrence=recurrence)
        self.events[eid] = ev
        return ev

    def due_events(self, tick: int) -> List[CalendarEvent]:
        out = []
        for ev in self.events.values():
            if ev.status != "scheduled":
                continue
            if ev.start_tick == tick:
                out.append(ev)
            elif ev.recurrence and tick >= ev.start_tick and (tick - ev.start_tick) % ev.recurrence == 0:
                out.append(ev)
        return out


__all__ = ["CalendarEvent", "CalendarSystem"]
