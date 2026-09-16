"""TimeSystem — clock + calendar + availability + work/overtime/weekend logs
(DESIGN env_org §6-§10/§36).

``log_work`` is the single place work effects are applied: it classifies the
session (overtime / weekend / deep-work) from the clock and mutates the agent's
vitals (attention down, fatigue/stress up), emitting the right logs. Overtime
and weekend work carry extra cost (§10).
"""
from __future__ import annotations

from typing import Dict, List, Optional

from environments.org_env.backend.clock.availability import (
    AgentAvailability,
    BackgroundJob,
    OvertimeLog,
    RecoveryLog,
    WeekendWorkLog,
    WorkSessionLog,
)
from environments.org_env.backend.clock.calendar import CalendarSystem
from environments.org_env.backend.clock.clock import OrgClock


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


class TimeSystem:
    def __init__(self, start_tick: int = 0, *, rhythm_enabled: bool = True):
        # When the work_rhythm mechanism is ablated the sessions are still
        # logged - what work happened, when, and for how long is organizational
        # evidence - but the vitals they would have moved stay at their neutral
        # defaults, so nothing downstream reads a fatigue signal that the design
        # is no longer making a claim about.
        self.rhythm_enabled = rhythm_enabled
        self.clock = OrgClock(start_tick=start_tick)
        self.calendar = CalendarSystem()
        self.availability: Dict[str, AgentAvailability] = {}
        self.work_sessions: List[WorkSessionLog] = []
        self.overtime_logs: List[OvertimeLog] = []
        self.weekend_logs: List[WeekendWorkLog] = []
        self.recovery_logs: List[RecoveryLog] = []
        self.background_jobs: List[BackgroundJob] = []
        self._seq = 0

    def register_agent(self, av: AgentAvailability) -> None:
        self.availability[av.agent_id] = av

    def _id(self, p: str) -> str:
        self._seq += 1
        return f"{p}_{self._seq}"

    # -- work effects -------------------------------------------------------
    def log_work(self, agent, *, action_type: str = "work_on_task",
                 is_deep_work: bool = False, voluntary: bool = True,
                 linked_task_id: Optional[str] = None, duration: int = 1,
                 requested_by: Optional[str] = None) -> WorkSessionLog:
        clk = self.clock
        is_overtime = clk.is_after_hours or clk.is_late_night
        is_weekend = clk.is_weekend

        attention_cost = 0.10 * duration + (0.03 if is_deep_work else 0.0)
        fatigue_delta = 0.03 * duration + (0.02 if is_deep_work else 0.0)
        stress_delta = 0.01 * duration
        recovery_loss = 0.0
        burnout_delta = 0.0
        if is_overtime:
            fatigue_delta += 0.08 * duration
            stress_delta += 0.05 * duration
            burnout_delta += 0.05 * duration
        if is_weekend:
            recovery_loss += 0.06 * duration       # weekends should restore — working loses that
            fatigue_delta += recovery_loss
            stress_delta += 0.03 * duration
            burnout_delta += 0.04 * duration

        if not self.rhythm_enabled:
            attention_cost = fatigue_delta = stress_delta = 0.0
            recovery_loss = burnout_delta = 0.0
        else:
            v = agent.vitals.variables
            v["attention"] = _clamp(v.get("attention", 1.0) - attention_cost)
            v["fatigue"] = _clamp(v.get("fatigue", 0.0) + fatigue_delta)
            v["stress"] = _clamp(v.get("stress", 0.0) + stress_delta)
            if burnout_delta:
                v["burnout_risk"] = _clamp(v.get("burnout_risk", 0.0) + burnout_delta)
            if is_weekend:
                v["morale"] = _clamp(v.get("morale", 0.6) - 0.02 * duration)

        wsid = self._id("ws")
        log = WorkSessionLog(work_session_id=wsid, agent_id=agent.id,
                             start_tick=clk.current_tick, end_tick=clk.current_tick + duration,
                             action_type=action_type, target_object_id=linked_task_id,
                             is_overtime=is_overtime, is_weekend=is_weekend,
                             is_deep_work=is_deep_work, attention_cost=attention_cost,
                             fatigue_delta=fatigue_delta, stress_delta=stress_delta,
                             progress_delta=0.1 * duration)
        self.work_sessions.append(log)

        if is_overtime:
            self.overtime_logs.append(OvertimeLog(
                overtime_event_id=self._id("ot"), agent_id=agent.id,
                start_tick=clk.current_tick, end_tick=clk.current_tick + duration,
                reason=action_type, linked_task_id=linked_task_id, voluntary=voluntary,
                requested_by=requested_by, progress_gain=0.1 * duration,
                fatigue_delta=fatigue_delta, stress_delta=stress_delta,
                next_day_penalty=0.05, burnout_risk_delta=burnout_delta))
        if is_weekend:
            self.weekend_logs.append(WeekendWorkLog(
                weekend_work_id=self._id("we"), agent_id=agent.id, linked_task_id=linked_task_id,
                reason=action_type, duration=duration, progress_gain=0.1 * duration,
                recovery_loss=recovery_loss, morale_effect=-0.02 * duration,
                burnout_risk_delta=burnout_delta))

        av = self.availability.get(agent.id)
        if av is not None and self.rhythm_enabled:
            av.current_availability_status = "weekend_working" if is_weekend else (
                "overtime" if is_overtime else "focused")
        return log

    def rest(self, agent, *, duration: int = 1) -> RecoveryLog:
        """Rest-offline: partial recovery (O1.6 §8.3)."""
        d_attn = 0.08 * duration
        d_fat = -0.04 * duration
        d_str = -0.03 * duration
        if not self.rhythm_enabled:
            d_attn = d_fat = d_str = 0.0
        else:
            v = agent.vitals.variables
            v["attention"] = _clamp(v.get("attention", 0.0) + d_attn)
            v["fatigue"] = _clamp(v.get("fatigue", 0.0) + d_fat)
            v["stress"] = _clamp(v.get("stress", 0.0) + d_str)
        log = RecoveryLog(recovery_id=self._id("rec"), agent_id=agent.id, kind="rest_offline",
                          start_tick=self.clock.current_tick, end_tick=self.clock.current_tick + duration,
                          duration=duration, attention_delta=d_attn, fatigue_delta=d_fat, stress_delta=d_str)
        self.recovery_logs.append(log)
        return log

    def sleep(self, agent, *, duration: int = 7) -> RecoveryLog:
        """Sleep: strong recovery (O1.6 §8.2). Bigger attention/fatigue/stress/
        burnout recovery than rest; spans several ticks (handled by caller)."""
        deltas = (0.4, -0.25, -0.15, -0.05)
        if not self.rhythm_enabled:
            deltas = (0.0, 0.0, 0.0, 0.0)
        else:
            v = agent.vitals.variables
            v["attention"] = _clamp(v.get("attention", 0.0) + deltas[0])
            v["fatigue"] = _clamp(v.get("fatigue", 0.0) + deltas[1])
            v["stress"] = _clamp(v.get("stress", 0.0) + deltas[2])
            v["burnout_risk"] = _clamp(v.get("burnout_risk", 0.0) + deltas[3])
        log = RecoveryLog(recovery_id=self._id("rec"), agent_id=agent.id, kind="sleep",
                          start_tick=self.clock.current_tick, end_tick=self.clock.current_tick + duration,
                          duration=duration, attention_delta=deltas[0], fatigue_delta=deltas[1],
                          stress_delta=deltas[2], burnout_risk_delta=deltas[3])
        self.recovery_logs.append(log)
        av = self.availability.get(agent.id)
        if av is not None and self.rhythm_enabled:
            av.current_availability_status = "asleep"
        return log

    # -- daily reset (§8.1) -------------------------------------------------
    def compute_sleep_quality(self, agent) -> float:
        """Sleep quality degrades with carried fatigue/stress/burnout (e.g. from
        late-night/weekend work the prior day)."""
        v = agent.vitals.variables
        q = 1.0 - 0.30 * v.get("fatigue", 0.0) - 0.20 * v.get("stress", 0.0) \
            - 0.20 * v.get("burnout_risk", 0.0)
        return _clamp(q, 0.4, 1.0)

    def start_new_day(self, agent, *, default_aux_slots: int = 1) -> RecoveryLog:
        """Begin-of-day partial reset — NOT a full reset (§8.1). Recovered
        attention depends on sleep quality minus carried fatigue/stress."""
        v = agent.vitals.variables
        before = v.get("attention", 0.0)
        sleep_quality = self.compute_sleep_quality(agent)
        if self.rhythm_enabled:
            fatigue_penalty = _clamp(v.get("fatigue", 0.0) * 0.25, 0.0, 0.4)
            stress_penalty = _clamp(v.get("stress", 0.0) * 0.15, 0.0, 0.3)
            recovered = 1.0 * sleep_quality - fatigue_penalty - stress_penalty
            v["attention"] = _clamp(recovered, 0.3, 1.0)
            # overnight: fatigue/stress ease somewhat (good sleep eases more)
            v["fatigue"] = _clamp(v.get("fatigue", 0.0) - 0.20 * sleep_quality)
            v["stress"] = _clamp(v.get("stress", 0.0) - 0.12 * sleep_quality)
            v["burnout_risk"] = _clamp(v.get("burnout_risk", 0.0) - 0.03 * sleep_quality)
        # The per-day quotas below are throughput budgets rather than
        # physiology, so they reset either way - otherwise an ablated run would
        # exhaust its speech slots on day one and never get them back.
        ws = getattr(agent, "work_state", None)
        if ws is not None:
            ws.aux_speech_slots_remaining = default_aux_slots
            ws.daily_message_count = 0
            ws.daily_meeting_count = 0
            ws.deep_work_blocks_used_today = 0
        log = RecoveryLog(recovery_id=self._id("rec"), agent_id=agent.id, kind="daily_reset",
                          start_tick=self.clock.current_tick, end_tick=self.clock.current_tick,
                          sleep_quality=sleep_quality, attention_delta=v["attention"] - before)
        self.recovery_logs.append(log)
        av = self.availability.get(agent.id)
        if av is not None and (
            not self.rhythm_enabled
            or av.current_availability_status in ("asleep", "resting", "offline")
        ):
            av.current_availability_status = "available"
        return log

    # -- background jobs (§5.2) --------------------------------------------
    def submit_background_job(self, *, agent_id: str, action_type: str, submit_tick: int,
                              background_duration: int, result_id=None) -> BackgroundJob:
        job = BackgroundJob(job_id=self._id("bgjob"), agent_id=agent_id, action_type=action_type,
                            submit_tick=submit_tick,
                            expected_completion_tick=submit_tick + max(1, background_duration),
                            result_id=result_id)
        self.background_jobs.append(job)
        return job

    def due_background_jobs(self, tick: int) -> List[BackgroundJob]:
        return [j for j in self.background_jobs
                if j.status == "running" and j.expected_completion_tick <= tick]


__all__ = ["TimeSystem"]
