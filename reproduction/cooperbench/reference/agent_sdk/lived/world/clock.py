"""World clock + day/night cycle (async infra §2, §3).

A lightweight discrete-tick clock with a four-season daylight table. v1 keeps it
deliberately simple (no astronomy / weather / ecology, §25): day/night only
influences visibility, movement cost, gather yield, passive energy decay and
sleep recovery — all as deterministic functions of the clock that the
FeatureExtractor / handlers / perception read.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

# §2 four-season daylight (ticks of daylight per ticks_per_day=12). "fall" alias.
SEASON_DAYLIGHT: Dict[str, int] = {
    "spring": 7, "summer": 8, "autumn": 7, "fall": 7, "winter": 5,
}
DEFAULT_TICKS_PER_DAY = 12

# §3.1 visibility
DAY_VISUAL_RADIUS = 6
NIGHT_VISUAL_RADIUS = 3
CAMPFIRE_VISUAL_BONUS = 3
# §3.2 movement cost multipliers
NIGHT_MOVE_MULT = 1.5
WINTER_NIGHT_MOVE_MULT = 1.8
# §3.3 gather yield
NIGHT_GATHER_MULT = 0.5
# §7 passive energy decay (per tick, awake)
DAY_PASSIVE_DECAY = 0.5
NIGHT_PASSIVE_DECAY = 1.0
WINTER_NIGHT_PASSIVE_DECAY = 1.5


@dataclass
class WorldClock:
    world_tick: int = 0
    ticks_per_day: int = DEFAULT_TICKS_PER_DAY
    season: str = "summer"

    @property
    def day_index(self) -> int:
        return self.world_tick // self.ticks_per_day

    @property
    def tick_in_day(self) -> int:
        return self.world_tick % self.ticks_per_day

    @property
    def daylight_ticks(self) -> int:
        return SEASON_DAYLIGHT.get(self.season, 8)

    @property
    def night_ticks(self) -> int:
        return self.ticks_per_day - self.daylight_ticks

    @property
    def is_daytime(self) -> bool:
        return self.tick_in_day < self.daylight_ticks

    @property
    def is_nighttime(self) -> bool:
        return not self.is_daytime

    @property
    def is_winter(self) -> bool:
        return self.season == "winter"

    @property
    def time_of_day(self) -> str:
        if self.is_nighttime:
            return "night"
        # last daylight tick reads as "evening" (planning pressure, §3.5)
        return "evening" if self.tick_in_day == self.daylight_ticks - 1 else "day"

    @property
    def ticks_until_daytime(self) -> int:
        """Ticks until the next daytime begins (0 if already daytime)."""
        if self.is_daytime:
            return 0
        return self.ticks_per_day - self.tick_in_day

    def tick(self, n: int = 1) -> "WorldClock":
        self.world_tick += n
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {
            "world_tick": self.world_tick, "day_index": self.day_index,
            "tick_in_day": self.tick_in_day, "ticks_per_day": self.ticks_per_day,
            "season": self.season, "time_of_day": self.time_of_day,
            "daylight_ticks": self.daylight_ticks, "night_ticks": self.night_ticks,
            "is_daytime": self.is_daytime, "is_nighttime": self.is_nighttime,
        }


# --------------------------------------------------------------------------- #
# Day/night effect helpers (deterministic functions of the clock)
# --------------------------------------------------------------------------- #
# §3.1 night reduces the agent's own vision to this fraction of its daytime value.
NIGHT_VISION_FACTOR = 0.5


def visual_radius(clock: WorldClock, *, near_campfire: bool = False, base: float = 0.0) -> float:
    """§3.1 effective vision radius. When ``base`` (the agent's own vision_radius)
    is given (>0), it IS the daytime radius and night halves it (floored at the
    night default) — so individual / env vision is respected instead of being
    overwritten. With no ``base`` the fixed day/night defaults are used. Campfire
    or home proximity grants a night bonus."""
    if base and base > 0:
        r = float(base) if clock.is_daytime else max(
            float(NIGHT_VISUAL_RADIUS), float(base) * NIGHT_VISION_FACTOR)
    else:
        r = DAY_VISUAL_RADIUS if clock.is_daytime else NIGHT_VISUAL_RADIUS
    if near_campfire:
        r += CAMPFIRE_VISUAL_BONUS
    return float(r)


def move_cost_multiplier(clock: WorldClock) -> float:
    if clock.is_daytime:
        return 1.0
    return WINTER_NIGHT_MOVE_MULT if clock.is_winter else NIGHT_MOVE_MULT


def gather_yield_multiplier(clock: WorldClock) -> float:
    return 1.0 if clock.is_daytime else NIGHT_GATHER_MULT


def passive_energy_decay(clock: WorldClock) -> float:
    if clock.is_daytime:
        return DAY_PASSIVE_DECAY
    return WINTER_NIGHT_PASSIVE_DECAY if clock.is_winter else NIGHT_PASSIVE_DECAY


def sleep_recovery(clock: WorldClock, *, at_home: bool = False, near_campfire: bool = False
                   ) -> Dict[str, Any]:
    """§6.2 per-tick sleep recovery + risk by location/time."""
    if at_home and near_campfire:
        return {"energy": 6.0, "fatigue": -0.06, "risk": "very_low"}
    if near_campfire or at_home:
        return {"energy": 5.0, "fatigue": -0.05, "risk": "low"}
    if clock.is_daytime:
        return {"energy": 3.0, "fatigue": -0.04, "risk": "medium"}
    return {"energy": 2.0, "fatigue": -0.03, "risk": "high"}
