"""Agent work state + org psychological state (OrgEnv O1.6, spec Part I §7).

Company agents have a "body state" too — not hunger/hp but **work capacity +
psychological state**: attention, fatigue, stress, burnout, morale, plus
organizational feelings (trust in company, compensation stress, retention risk).

To stay non-breaking, the continuous vitals remain stored in the agent's existing
``VitalState.variables`` bag (read by ``can_agent_act`` / ``OrgPolicy.weight`` /
perception). :class:`AgentWorkState` and :class:`AgentOrgState` are structured
*views* over that bag (property accessors) plus the new scheduling fields
(``next_available_tick`` / current activity / aux-speech slots / daily counters /
background jobs) that have no vitals home.
"""
from __future__ import annotations

from typing import List, Optional

# Continuous vitals (spec §7.3 ORG_VITALS). Names here are the underlying vital
# KEYS; AgentWorkState/AgentOrgState expose them under the spec's property names.
ORG_VITALS = (
    "attention", "fatigue", "stress", "burnout_risk", "morale", "context_switch_cost",
    "deadline_pressure", "workload", "reputation_pressure",
    "compensation_stress", "trust_in_company", "retention_risk",
    "perceived_recognition", "role_clarity",
)

_VITAL_DEFAULTS = {
    "attention": 1.0, "morale": 0.6, "trust_in_company": 0.7,
    "role_clarity": 0.4, "perceived_recognition": 0.5,
}


def org_vital_defaults() -> dict:
    return {k: _VITAL_DEFAULTS.get(k, 0.0) for k in ORG_VITALS}


class _VitalsView:
    """Mixin: property accessors backed by a shared ``variables`` dict."""

    __slots__ = ()

    def _g(self, key: str, default: float = 0.0) -> float:
        return float(self._v.get(key, default))

    def _s(self, key: str, value: float) -> None:
        self._v[key] = max(0.0, min(1.0, float(value)))


class AgentWorkState(_VitalsView):
    """Short-term work capacity + accumulated body/cognitive state + scheduling.
    Continuous fields mirror the agent vitals bag; scheduling fields are local."""

    __slots__ = ("_v", "current_activity_id", "current_activity_type", "activity_started_tick",
                 "activity_ends_tick", "next_available_tick", "current_meeting_id",
                 "aux_speech_slots_remaining", "default_aux_speech_slots", "daily_message_count",
                 "daily_meeting_count", "deep_work_blocks_used_today", "background_jobs")

    def __init__(self, variables: dict, default_aux_slots: int = 1):
        self._v = variables
        self.current_activity_id: Optional[str] = None
        self.current_activity_type: Optional[str] = None
        self.activity_started_tick: Optional[int] = None
        self.activity_ends_tick: Optional[int] = None
        self.next_available_tick: int = 0
        self.current_meeting_id: Optional[str] = None
        self.default_aux_speech_slots = int(default_aux_slots)
        self.aux_speech_slots_remaining: int = int(default_aux_slots)
        self.daily_message_count: int = 0
        self.daily_meeting_count: int = 0
        self.deep_work_blocks_used_today: int = 0
        self.background_jobs: List[str] = []

    # -- continuous fields mirror vitals (spec §7.1) -----------------------
    @property
    def attention_remaining_today(self) -> float: return self._g("attention", 1.0)
    @attention_remaining_today.setter
    def attention_remaining_today(self, x: float) -> None: self._s("attention", x)

    @property
    def fatigue(self) -> float: return self._g("fatigue")
    @fatigue.setter
    def fatigue(self, x: float) -> None: self._s("fatigue", x)

    @property
    def stress(self) -> float: return self._g("stress")
    @stress.setter
    def stress(self, x: float) -> None: self._s("stress", x)

    @property
    def burnout_risk(self) -> float: return self._g("burnout_risk")
    @burnout_risk.setter
    def burnout_risk(self, x: float) -> None: self._s("burnout_risk", x)

    @property
    def morale(self) -> float: return self._g("morale", 0.6)
    @morale.setter
    def morale(self, x: float) -> None: self._s("morale", x)

    @property
    def context_switch_load(self) -> float: return self._g("context_switch_cost")
    @context_switch_load.setter
    def context_switch_load(self, x: float) -> None: self._s("context_switch_cost", x)

    # -- scheduling helpers ------------------------------------------------
    def is_busy(self, tick: int) -> bool:
        return tick < self.next_available_tick

    def start_activity(self, *, action_id: str, action_type: str, tick: int, duration: int) -> None:
        self.current_activity_id = action_id
        self.current_activity_type = action_type
        self.activity_started_tick = tick
        self.activity_ends_tick = tick + duration
        self.next_available_tick = tick + duration

    def clear_activity(self) -> None:
        self.current_activity_id = None
        self.current_activity_type = None
        self.activity_started_tick = None
        self.activity_ends_tick = None

    def snapshot(self) -> dict:
        return {"attention_remaining_today": round(self.attention_remaining_today, 3),
                "fatigue": round(self.fatigue, 3), "stress": round(self.stress, 3),
                "burnout_risk": round(self.burnout_risk, 3), "morale": round(self.morale, 3),
                "context_switch_load": round(self.context_switch_load, 3),
                "next_available_tick": self.next_available_tick,
                "current_activity_type": self.current_activity_type,
                "aux_speech_slots_remaining": self.aux_speech_slots_remaining,
                "daily_message_count": self.daily_message_count,
                "background_jobs": list(self.background_jobs)}


class AgentOrgState(_VitalsView):
    """Organizational psychological state (spec §7.2), mirrored on the vitals bag
    so the policy/perception already see it."""

    __slots__ = ("_v",)

    def __init__(self, variables: dict):
        self._v = variables

    @property
    def trust_in_company(self) -> float: return self._g("trust_in_company", 0.7)
    @trust_in_company.setter
    def trust_in_company(self, x: float) -> None: self._s("trust_in_company", x)

    @property
    def compensation_stress(self) -> float: return self._g("compensation_stress")
    @compensation_stress.setter
    def compensation_stress(self, x: float) -> None: self._s("compensation_stress", x)

    @property
    def perceived_recognition(self) -> float: return self._g("perceived_recognition", 0.5)
    @perceived_recognition.setter
    def perceived_recognition(self, x: float) -> None: self._s("perceived_recognition", x)

    @property
    def role_clarity(self) -> float: return self._g("role_clarity", 0.4)
    @role_clarity.setter
    def role_clarity(self, x: float) -> None: self._s("role_clarity", x)

    @property
    def workload_pressure(self) -> float: return self._g("workload")
    @workload_pressure.setter
    def workload_pressure(self, x: float) -> None: self._s("workload", x)

    @property
    def retention_risk(self) -> float: return self._g("retention_risk")
    @retention_risk.setter
    def retention_risk(self, x: float) -> None: self._s("retention_risk", x)


__all__ = ["ORG_VITALS", "org_vital_defaults", "AgentWorkState", "AgentOrgState"]
