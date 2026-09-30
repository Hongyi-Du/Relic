"""OrgClock (DESIGN env_org §6/§36). 1 day = 24 ticks, 1 week = 7 days.

A startup runs nights/weekends, so phase_of_day + weekday/weekend + work-hours
flags are derived from the tick. These are *defaults*, not hard rules — agents
may overtime, rest, meet, or work weekends.
"""
from __future__ import annotations

# (start_hour_inclusive, end_hour_exclusive, phase) — default weekday rhythm.
PHASE_SCHEDULE = (
    (0, 8, "night_sleep"),
    (8, 9, "morning_catchup"),
    (9, 10, "daily_sync"),
    (10, 12, "deep_work_morning"),
    (12, 13, "lunch_low_activity"),
    (13, 16, "deep_work_afternoon"),
    (16, 17, "collaboration_review"),
    (17, 18, "wrap_up"),
    (18, 22, "evening_overtime"),
    (22, 24, "late_night"),
)
_DOW = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


class OrgClock:
    def __init__(self, start_tick: int = 0):
        self.current_tick = start_tick

    def advance(self, n: int = 1) -> None:
        self.current_tick += n

    @property
    def hour_in_day(self) -> int:
        return self.current_tick % 24

    @property
    def day_index(self) -> int:
        return self.current_tick // 24

    @property
    def week_index(self) -> int:
        return self.day_index // 7

    @property
    def day_of_week(self) -> int:
        return self.day_index % 7      # 0=Mon .. 6=Sun

    @property
    def day_of_week_name(self) -> str:
        return _DOW[self.day_of_week]

    @property
    def is_weekend(self) -> bool:
        return self.day_of_week >= 5

    @property
    def is_weekday(self) -> bool:
        return not self.is_weekend

    @property
    def phase_of_day(self) -> str:
        h = self.hour_in_day
        for lo, hi, name in PHASE_SCHEDULE:
            if lo <= h < hi:
                return name
        return "night_sleep"

    @property
    def is_work_hours(self) -> bool:
        return self.is_weekday and 9 <= self.hour_in_day < 18

    @property
    def is_after_hours(self) -> bool:
        return (not self.is_work_hours) and (self.hour_in_day >= 18 or self.hour_in_day < 8) and not self.is_weekend or \
               (self.is_weekday and self.hour_in_day >= 18)

    @property
    def is_late_night(self) -> bool:
        return self.hour_in_day >= 22 or self.hour_in_day < 6

    @property
    def is_common_meeting_time(self) -> bool:
        return self.is_weekday and self.hour_in_day in (9, 10, 16)

    def snapshot(self) -> dict:
        return {
            "current_tick": self.current_tick, "day_index": self.day_index,
            "week_index": self.week_index, "hour_in_day": self.hour_in_day,
            "day_of_week": self.day_of_week_name, "is_weekday": self.is_weekday,
            "is_weekend": self.is_weekend, "phase_of_day": self.phase_of_day,
            "is_work_hours": self.is_work_hours, "is_after_hours": self.is_after_hours,
            "is_late_night": self.is_late_night,
            "is_common_meeting_time": self.is_common_meeting_time,
        }


__all__ = ["OrgClock", "PHASE_SCHEDULE"]
