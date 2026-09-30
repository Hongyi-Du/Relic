"""Per-agent routine profiles (OrgEnv O1.6, spec Part I §10).

A RoutineProfile is a daily-rhythm prior (preferred hours, deep-work hours,
after-hours/weekend tendency, sleep need). It does NOT force behavior — the
RoutineScheduler (runtime_adapter/routine.py) turns it + the clock phase into a
utility prior over candidate actions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class RoutineProfile:
    agent_id: str = ""
    preferred_start_hour: int = 9
    preferred_end_hour: int = 18
    deep_work_hours: List[int] = field(default_factory=list)
    meeting_preference_hours: List[int] = field(default_factory=lambda: [9, 16])
    after_hours_responsiveness: str = "medium"   # low | medium | high
    weekend_work_tendency: str = "medium"
    sleep_need_ticks: int = 7
    max_meetings_per_day: int = 4
    max_deep_work_blocks_per_day: int = 4
    update_frequency: str = "medium"
    routine_stability: str = "medium"

    def is_preferred_hour(self, hour: int) -> bool:
        return self.preferred_start_hour <= hour < self.preferred_end_hour

    def is_deep_work_hour(self, hour: int) -> bool:
        return hour in self.deep_work_hours


# §10 suggested profiles (keyed by agent_id). Missing fields fall back to defaults.
ROUTINES: Dict[str, RoutineProfile] = {
    "paul": RoutineProfile(agent_id="paul", preferred_start_hour=10, preferred_end_hour=21,
                           after_hours_responsiveness="high", weekend_work_tendency="medium",
                           meeting_preference_hours=[9, 16, 20], deep_work_hours=[11, 15],
                           routine_stability="low", update_frequency="high"),
    "victor": RoutineProfile(agent_id="victor", preferred_start_hour=9, preferred_end_hour=22,
                             after_hours_responsiveness="high", weekend_work_tendency="high",
                             deep_work_hours=[10, 11, 13, 14, 15, 21],
                             routine_stability="medium", update_frequency="medium",
                             sleep_need_ticks=6),
    "calvin": RoutineProfile(agent_id="calvin", preferred_start_hour=9, preferred_end_hour=18,
                             after_hours_responsiveness="low", weekend_work_tendency="low",
                             deep_work_hours=[10, 11, 13, 14, 15], meeting_preference_hours=[9, 16],
                             routine_stability="high", update_frequency="low", sleep_need_ticks=8),
    "scarlett": RoutineProfile(agent_id="scarlett", preferred_start_hour=9, preferred_end_hour=19,
                               after_hours_responsiveness="medium", weekend_work_tendency="medium",
                               meeting_preference_hours=[9, 15, 16], deep_work_hours=[10, 13, 14],
                               routine_stability="medium", update_frequency="high"),
    "will": RoutineProfile(agent_id="will", preferred_start_hour=10, preferred_end_hour=19,
                           after_hours_responsiveness="medium", weekend_work_tendency="medium",
                           deep_work_hours=[11, 14, 15, 16], update_frequency="medium"),
    "skitty": RoutineProfile(agent_id="skitty", preferred_start_hour=10, preferred_end_hour=20,
                             after_hours_responsiveness="medium", weekend_work_tendency="medium",
                             deep_work_hours=[13, 14, 18], routine_stability="medium",
                             update_frequency="medium"),
    "iris": RoutineProfile(agent_id="iris", preferred_start_hour=10, preferred_end_hour=20,
                           after_hours_responsiveness="medium", weekend_work_tendency="medium",
                           deep_work_hours=[13, 14, 15, 19], update_frequency="medium"),
    "sean": RoutineProfile(agent_id="sean", preferred_start_hour=10, preferred_end_hour=23,
                           after_hours_responsiveness="high", weekend_work_tendency="high",
                           deep_work_hours=[14, 15, 16, 20, 21, 22],
                           routine_stability="low", update_frequency="low"),
}


def get_routine(agent_id: str) -> RoutineProfile:
    return ROUTINES.get(agent_id, RoutineProfile(agent_id=agent_id))


__all__ = ["RoutineProfile", "ROUTINES", "get_routine"]
