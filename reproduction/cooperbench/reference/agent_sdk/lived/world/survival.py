"""SocioGenesis survival model — energy / fatigue / satiety / hp + overexertion.

Four orthogonal survival variables (the whole point is to STOP `energy` from
secretly meaning hunger, so rest/sleep can no longer substitute for eating):

  * ``energy``   — short-term stamina; spent by actions, restored by rest/sleep.
  * ``fatigue``  — tiredness on a 0–150+ scale (NOT the [0,1] mood scalar); rises
    with action, falls with rest/sleep, may exceed the 100 soft cap into the
    overexertion zone and the 150 hard cap = collapse.
  * ``satiety``  — fullness 0–100; decays every tick, restored ONLY by eating.
  * ``hp``       — health; drained by critical starvation, overexertion, overwork.

Design rules (§16): hunger/fatigue/low-energy NEVER hard-pick the action — they
flow into PCBSP utility/cost/risk. The agent may push through while hungry/tired
at extra energy + fatigue + failure-risk + hp cost (overexertion). Only a true
**collapse** (fatigue ≥ hard cap / forced-rest window) overrides selection.

Everything here is env-agnostic + pure: functions read/write a duck-typed object
(the real nature_env agent, an :class:`~agent_sdk.lived.perceive.perception.AgentGT`, or
the standalone :class:`SurvivalState` used in tests) via getattr/setattr. All
constants live on :class:`SurvivalConfig` so they are tunable + seed-free.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, fields
from typing import Any, Dict, Optional, Tuple


# --------------------------------------------------------------------------- #
# Config (all tunables, defaults from the spec)
# --------------------------------------------------------------------------- #
@dataclass
class SurvivalConfig:
    # satiety
    max_satiety: float = 100.0
    satiety_start: float = 75.0
    satiety_decay_day: float = 4.0
    satiety_decay_night: float = 3.0
    satiety_decay_sleep: float = 2.0
    winter_satiety_multiplier: float = 1.2
    satiety_decay_scale: float = 1.0
    hungry_at: float = 60.0
    starving_at: float = 30.0
    critical_at: float = 10.0
    critical_starvation_hp_loss: float = 3.0
    zero_satiety_hp_loss: float = 5.0
    # fatigue
    fatigue_start: float = 20.0
    fatigue_soft_cap: float = 100.0
    fatigue_hard_cap: float = 150.0
    fatigue_high_at: float = 60.0
    # overexertion
    overexertion_energy_multiplier: float = 1.5
    overexertion_fatigue_multiplier: float = 1.5
    overexertion_hp_min: float = 1.0
    overexertion_hp_max: float = 3.0
    high_cost_threshold: float = 4.0          # base_energy_cost considered "high"
    low_energy_threshold: float = 20.0
    low_satiety_threshold: float = 20.0
    # forced rest / collapse
    base_forced_rest: int = 2
    consecutive_overexertion_collapse: int = 4
    # overwork death
    overwork_strain_per_severity: float = 10.0
    overwork_death_strain: float = 100.0
    overwork_death_hp: float = 20.0
    sleep_home_strain_recovery: float = 15.0
    wild_rest_strain_recovery: float = 5.0
    forced_rest_strain_recovery: float = 10.0
    # fatigue delta scale: OUR ActionSpec.fatigue_delta is on a 0..~0.05 scale;
    # multiply by this to land on the 0–150 fatigue scale.
    fatigue_delta_scale: float = 100.0


DEFAULT = SurvivalConfig()

# fields ensure_fields installs (name -> default factory from cfg or constant)
_SURVIVAL_DEFAULTS = {
    "satiety": lambda c: c.satiety_start,
    "max_satiety": lambda c: c.max_satiety,
    "satiety_decay_multiplier": lambda c: 1.0,
    "fatigue": lambda c: c.fatigue_start,
    "fatigue_soft_cap": lambda c: c.fatigue_soft_cap,
    "fatigue_hard_cap": lambda c: c.fatigue_hard_cap,
    "overexertion_debt": lambda c: 0.0,
    "overexertion_count": lambda c: 0,
    "consecutive_overexertion_ticks": lambda c: 0,
    "forced_rest_until_tick": lambda c: -1,
    "collapse_count": lambda c: 0,
    "overwork_strain": lambda c: 0.0,
    "death_reason": lambda c: None,
}


@dataclass
class SurvivalState:
    """Standalone survival bundle (tests + optional embedding)."""
    energy: float = 100.0
    max_energy: float = 100.0
    hp: float = 10.0
    max_hp: float = 10.0
    satiety: float = 75.0
    max_satiety: float = 100.0
    satiety_decay_multiplier: float = 1.0
    fatigue: float = 20.0
    fatigue_soft_cap: float = 100.0
    fatigue_hard_cap: float = 150.0
    overexertion_debt: float = 0.0
    overexertion_count: int = 0
    consecutive_overexertion_ticks: int = 0
    forced_rest_until_tick: int = -1
    collapse_count: int = 0
    overwork_strain: float = 0.0
    death_reason: Optional[str] = None


def ensure_fields(obj: Any, cfg: SurvivalConfig = DEFAULT) -> None:
    """Install any missing survival attributes on a duck-typed object."""
    for name, default in _SURVIVAL_DEFAULTS.items():
        if getattr(obj, name, None) is None and not hasattr(obj, name):
            setattr(obj, name, default(cfg))
        elif getattr(obj, name, None) is None and name == "death_reason":
            pass  # None is a valid value for death_reason


def _g(obj: Any, name: str, default: float) -> float:
    v = getattr(obj, name, default)
    try:
        return float(v) if v is not None else default
    except (TypeError, ValueError):
        return default


# --------------------------------------------------------------------------- #
# Satiety / hunger
# --------------------------------------------------------------------------- #
def satiety_decay_amount(cfg: SurvivalConfig, *, is_daytime: bool, sleeping: bool,
                         winter: bool, mult: float = 1.0) -> float:
    base = (cfg.satiety_decay_sleep if sleeping
            else (cfg.satiety_decay_day if is_daytime else cfg.satiety_decay_night))
    out = base * cfg.satiety_decay_scale * float(mult)
    if winter:
        out *= cfg.winter_satiety_multiplier
    return out


def tick_satiety(obj: Any, cfg: SurvivalConfig = DEFAULT, *, is_daytime: bool = True,
                 sleeping: bool = False, winter: bool = False) -> float:
    """Apply one tick of satiety decay. Returns the (negative) delta."""
    mult = _g(obj, "satiety_decay_multiplier", 1.0)
    dec = satiety_decay_amount(cfg, is_daytime=is_daytime, sleeping=sleeping, winter=winter, mult=mult)
    before = _g(obj, "satiety", cfg.satiety_start)
    obj.satiety = max(0.0, before - dec)
    return obj.satiety - before


def hunger_stage(satiety: float, cfg: SurvivalConfig = DEFAULT) -> str:
    if satiety < cfg.critical_at:
        return "critical"
    if satiety < cfg.starving_at:
        return "starving"
    if satiety < cfg.hungry_at:
        return "hungry"
    return "normal"


def hunger_pressure(satiety: float, cfg: SurvivalConfig = DEFAULT) -> float:
    """[0,1], rises as satiety falls below the hungry threshold (§4, linear v1)."""
    return max(0.0, min(1.0, (cfg.hungry_at - satiety) / cfg.hungry_at))


def update_hunger_flags(obj: Any, cfg: SurvivalConfig = DEFAULT) -> str:
    s = _g(obj, "satiety", cfg.satiety_start)
    stage = hunger_stage(s, cfg)
    obj.is_hungry = s < cfg.hungry_at
    obj.is_starving = s < cfg.starving_at
    obj.is_critically_starving = s < cfg.critical_at
    return stage


def apply_starvation_hp(obj: Any, cfg: SurvivalConfig = DEFAULT) -> float:
    """Critical-starvation HP drain (§3). Returns hp lost (>=0)."""
    s = _g(obj, "satiety", cfg.satiety_start)
    loss = 0.0
    if s <= 0.0:
        loss = cfg.zero_satiety_hp_loss
    elif s < cfg.critical_at:
        loss = cfg.critical_starvation_hp_loss
    if loss > 0:
        obj.hp = max(0.0, _g(obj, "hp", 0.0) - loss)
    return loss


# --------------------------------------------------------------------------- #
# Fatigue / exhaustion
# --------------------------------------------------------------------------- #
def fatigue_zone(fatigue: float, cfg: SurvivalConfig = DEFAULT) -> str:
    if fatigue >= cfg.fatigue_hard_cap:
        return "collapse"
    if fatigue >= cfg.fatigue_soft_cap:
        return "overexertion"
    if fatigue >= cfg.fatigue_high_at:
        return "high"
    return "normal"


def exhaustion_pressure(obj: Any, cfg: SurvivalConfig = DEFAULT) -> float:
    """[0,1] from low energy + high fatigue (drives rest/sleep utility, §19)."""
    e = _g(obj, "energy", 100.0)
    me = _g(obj, "max_energy", 100.0) or 100.0
    f = _g(obj, "fatigue", 0.0)
    energy_term = max(0.0, 1.0 - e / me)
    fatigue_term = max(0.0, min(1.0, f / cfg.fatigue_soft_cap))
    return max(0.0, min(1.0, 0.5 * energy_term + 0.5 * fatigue_term))


# --------------------------------------------------------------------------- #
# Overexertion (§7-§8): applied per executed non-recovery action
# --------------------------------------------------------------------------- #
def overexertion_severity(fatigue: float, action_energy_cost: float,
                          cfg: SurvivalConfig = DEFAULT) -> float:
    return max(0.0, (fatigue - cfg.fatigue_soft_cap) / 50.0) + action_energy_cost / 10.0


def overexertion_triggered(obj: Any, *, action_type: str, base_energy_cost: float,
                           cfg: SurvivalConfig = DEFAULT) -> Tuple[bool, str]:
    """§7 triggers (no hard-block — just signals extra cost)."""
    f = _g(obj, "fatigue", 0.0)
    e = _g(obj, "energy", 100.0)
    s = _g(obj, "satiety", 100.0)
    hp = _g(obj, "hp", 10.0)
    max_hp = _g(obj, "max_hp", 10.0) or 10.0
    high = base_energy_cost >= cfg.high_cost_threshold
    if f >= cfg.fatigue_soft_cap:
        return True, "fatigue>=soft_cap"
    if f >= 80.0 and high:
        return True, "high_fatigue+high_cost"
    if e < cfg.low_energy_threshold and high:
        return True, "low_energy+high_cost"
    if s < cfg.low_satiety_threshold and action_type not in (
            "eat_food", "rest", "sleep", "seek_safety", "return_home", "wait_or_continue"):
        return True, "low_satiety+work"
    if hp < 0.3 * max_hp and high:
        return True, "low_hp+high_cost"
    return False, ""


def apply_action_cost(obj: Any, *, action_type: str, base_energy_cost: float,
                      base_fatigue_delta: float, allow_push_through: bool = True,
                      cfg: SurvivalConfig = DEFAULT) -> Dict[str, Any]:
    """Apply the survival cost of ONE executed (non-recovery) action.

    NOTE: the *base* energy cost is charged by the env's own action handler — this
    only adds the SocioGenesis layer: always the fatigue delta; and when the agent
    is pushing through (overexertion), the EXTRA energy, extra fatigue, hp damage,
    debt + strain. Returns a result dict for the ActionExecutionLog (§32)."""
    fatigue = _g(obj, "fatigue", cfg.fatigue_start)
    over, reason = overexertion_triggered(obj, action_type=action_type,
                                          base_energy_cost=base_energy_cost, cfg=cfg)
    res: Dict[str, Any] = {"overexertion": False, "overexertion_reason": "",
                           "overexertion_severity": 0.0, "extra_energy_cost": 0.0,
                           "extra_fatigue_delta": 0.0, "hp_damage_from_overexertion": 0.0,
                           "fatigue_before": fatigue}
    fat_delta = base_fatigue_delta
    if over and allow_push_through:
        sev = overexertion_severity(fatigue, base_energy_cost, cfg)
        extra_energy = base_energy_cost * (cfg.overexertion_energy_multiplier - 1.0)
        fat_delta = base_fatigue_delta * cfg.overexertion_fatigue_multiplier
        hp_damage = min(cfg.overexertion_hp_max, cfg.overexertion_hp_min + sev)
        obj.energy = max(0.0, _g(obj, "energy", 0.0) - extra_energy)
        obj.hp = max(0.0, _g(obj, "hp", 0.0) - hp_damage)
        obj.overexertion_debt = _g(obj, "overexertion_debt", 0.0) + sev
        obj.overexertion_count = int(_g(obj, "overexertion_count", 0)) + 1
        obj.consecutive_overexertion_ticks = int(_g(obj, "consecutive_overexertion_ticks", 0)) + 1
        obj.overwork_strain = _g(obj, "overwork_strain", 0.0) + sev * cfg.overwork_strain_per_severity
        res.update(overexertion=True, overexertion_reason=reason, overexertion_severity=round(sev, 3),
                   extra_energy_cost=round(extra_energy, 3),
                   extra_fatigue_delta=round(fat_delta - base_fatigue_delta, 3),
                   hp_damage_from_overexertion=round(hp_damage, 3))
    else:
        obj.consecutive_overexertion_ticks = 0
    obj.fatigue = max(0.0, min(cfg.fatigue_hard_cap + 50.0, fatigue + fat_delta))
    res["fatigue_after"] = obj.fatigue
    res["fatigue_delta"] = round(obj.fatigue - fatigue, 3)
    return res


# --------------------------------------------------------------------------- #
# Forced rest / collapse (§9) + recovery (§13/§14) + overwork death (§10)
# --------------------------------------------------------------------------- #
def forced_rest_triggered(obj: Any, cfg: SurvivalConfig = DEFAULT) -> Tuple[bool, str]:
    f = _g(obj, "fatigue", 0.0)
    e = _g(obj, "energy", 100.0)
    consec = int(_g(obj, "consecutive_overexertion_ticks", 0))
    if f >= cfg.fatigue_hard_cap:
        return True, "fatigue>=hard_cap"
    if e <= 0.0 and f > 120.0:
        return True, "no_energy+high_fatigue"
    if consec >= cfg.consecutive_overexertion_collapse:
        return True, "consecutive_overexertion"
    return False, ""


def enter_forced_rest(obj: Any, current_tick: int, cfg: SurvivalConfig = DEFAULT) -> int:
    debt = _g(obj, "overexertion_debt", 0.0)
    dur = cfg.base_forced_rest + math.ceil(debt / 20.0)
    obj.forced_rest_until_tick = current_tick + dur
    obj.collapse_count = int(_g(obj, "collapse_count", 0)) + 1
    obj.consecutive_overexertion_ticks = 0
    return dur


def is_forced_resting(obj: Any, current_tick: int) -> bool:
    return current_tick < int(_g(obj, "forced_rest_until_tick", -1))


def apply_recovery(obj: Any, *, energy_delta: float, fatigue_delta: float,
                   strain_recovery: float = 0.0, cfg: SurvivalConfig = DEFAULT) -> None:
    """rest / sleep / forced-rest recovery: restores energy, lowers fatigue + overwork
    strain. Does NOT touch satiety (§13/§14: you wake up still hungry)."""
    me = _g(obj, "max_energy", 100.0)
    obj.energy = max(0.0, min(me, _g(obj, "energy", 0.0) + energy_delta))
    obj.fatigue = max(0.0, _g(obj, "fatigue", 0.0) + fatigue_delta)   # fatigue_delta negative
    if strain_recovery:
        obj.overwork_strain = max(0.0, _g(obj, "overwork_strain", 0.0) - strain_recovery)


def overwork_death_check(obj: Any, cfg: SurvivalConfig = DEFAULT) -> Optional[str]:
    """§10 deterministic v1: high accumulated strain + low hp = overwork death."""
    if (_g(obj, "overwork_strain", 0.0) >= cfg.overwork_death_strain
            and _g(obj, "hp", 10.0) < cfg.overwork_death_hp):
        obj.death_reason = "overwork"
        return "overwork"
    return None


# --------------------------------------------------------------------------- #
# One-call self-state snapshot (for SelfStatePercept / body_state / logs)
# --------------------------------------------------------------------------- #
def snapshot(obj: Any, current_tick: int = 0, cfg: SurvivalConfig = DEFAULT) -> Dict[str, Any]:
    s = _g(obj, "satiety", cfg.satiety_start)
    f = _g(obj, "fatigue", cfg.fatigue_start)
    return {
        "energy": _g(obj, "energy", 0.0), "max_energy": _g(obj, "max_energy", 100.0),
        "hp": _g(obj, "hp", 0.0), "max_hp": _g(obj, "max_hp", 10.0),
        "satiety": s, "max_satiety": _g(obj, "max_satiety", cfg.max_satiety),
        "hunger_pressure": round(hunger_pressure(s, cfg), 3),
        "satiety_stage": hunger_stage(s, cfg),
        "is_hungry": s < cfg.hungry_at, "is_starving": s < cfg.starving_at,
        "is_critically_starving": s < cfg.critical_at,
        "fatigue": f, "fatigue_zone": fatigue_zone(f, cfg),
        "fatigue_soft_cap": cfg.fatigue_soft_cap, "fatigue_hard_cap": cfg.fatigue_hard_cap,
        "exhaustion_pressure": round(exhaustion_pressure(obj, cfg), 3),
        "is_exhausted": exhaustion_pressure(obj, cfg) > 0.6,
        "is_collapsed": is_forced_resting(obj, current_tick) or f >= cfg.fatigue_hard_cap,
        "overexertion_debt": round(_g(obj, "overexertion_debt", 0.0), 3),
        "overexertion_count": int(_g(obj, "overexertion_count", 0)),
        "consecutive_overexertion_ticks": int(_g(obj, "consecutive_overexertion_ticks", 0)),
        "forced_rest_until_tick": int(_g(obj, "forced_rest_until_tick", -1)),
        "collapse_count": int(_g(obj, "collapse_count", 0)),
        "overwork_strain": round(_g(obj, "overwork_strain", 0.0), 3),
        "death_reason": getattr(obj, "death_reason", None),
    }
