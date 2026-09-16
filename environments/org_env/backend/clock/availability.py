"""AgentAvailability + work/overtime/weekend logs (DESIGN env_org §8-§10/§36).

Agents are NOT online 24h. Availability gates whether an agent responds to a
message (offline agents ignore normal messages but can be pulled in for
incidents). Overtime/weekend work has progress benefit AND fatigue/stress/
burnout cost (§10).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class AgentAvailability:
    agent_id: str
    timezone: str = "UTC"
    preferred_work_hours: tuple = (9, 18)
    deep_work_preference: float = 0.5
    meeting_tolerance: float = 0.5
    after_hours_responsiveness: float = 0.3
    weekend_work_tendency: float = 0.3
    sleep_need: float = 0.6
    burnout_threshold: float = 0.8
    current_availability_status: str = "available"   # available|focused|in_meeting|offline|resting|asleep|overtime|weekend_working|burned_out|forced_rest
    current_focus_object_id: Optional[str] = None
    offline_until_tick: Optional[int] = None

    def go_offline(self, until_tick: Optional[int] = None, status: str = "offline") -> None:
        self.current_availability_status = status
        self.offline_until_tick = until_tick

    def is_online(self) -> bool:
        return self.current_availability_status not in ("offline", "asleep", "resting",
                                                        "forced_rest", "burned_out")

    def responds_to(self, urgency: str) -> bool:
        """Does this agent respond *now* to a message of the given urgency?
        Offline/asleep agents ignore normal traffic but answer incidents
        (and high-responsiveness agents answer urgent pings)."""
        if urgency == "incident":
            return True
        if self.is_online():
            return True
        if urgency == "urgent" and self.after_hours_responsiveness >= 0.7:
            return True
        return False


@dataclass
class WorkSessionLog:
    work_session_id: str
    agent_id: str
    start_tick: int
    end_tick: int
    action_type: str = ""
    target_object_id: Optional[str] = None
    is_overtime: bool = False
    is_weekend: bool = False
    is_deep_work: bool = False
    attention_cost: float = 0.0
    fatigue_delta: float = 0.0
    stress_delta: float = 0.0
    progress_delta: float = 0.0
    output_object_ids: List[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class OvertimeLog:
    overtime_event_id: str
    agent_id: str
    start_tick: int
    end_tick: int
    reason: str = ""
    linked_task_id: Optional[str] = None
    voluntary: bool = True
    requested_by: Optional[str] = None
    progress_gain: float = 0.0
    fatigue_delta: float = 0.0
    stress_delta: float = 0.0
    next_day_penalty: float = 0.0
    burnout_risk_delta: float = 0.0


@dataclass
class WeekendWorkLog:
    weekend_work_id: str
    agent_id: str
    linked_task_id: Optional[str] = None
    reason: str = ""
    duration: int = 1
    progress_gain: float = 0.0
    recovery_loss: float = 0.0
    morale_effect: float = 0.0
    burnout_risk_delta: float = 0.0


@dataclass
class RecoveryLog:
    """Sleep / rest-offline / daily-reset recovery (O1.6 §8)."""
    recovery_id: str
    agent_id: str
    kind: str                       # sleep | rest_offline | daily_reset
    start_tick: int
    end_tick: int
    duration: int = 1
    sleep_quality: float = 1.0
    attention_delta: float = 0.0
    fatigue_delta: float = 0.0
    stress_delta: float = 0.0
    burnout_risk_delta: float = 0.0


@dataclass
class BackgroundJob:
    """A submitted-now / completes-later job (O1.6 §5.2)."""
    job_id: str
    agent_id: str
    action_type: str
    submit_tick: int
    expected_completion_tick: int
    status: str = "running"         # running | completed
    result_id: Optional[str] = None
    notified: bool = False


__all__ = ["AgentAvailability", "WorkSessionLog", "OvertimeLog", "WeekendWorkLog",
           "RecoveryLog", "BackgroundJob"]
