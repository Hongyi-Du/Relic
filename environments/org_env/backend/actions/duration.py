"""Action duration + cost specs (OrgEnv O1.6, spec Part I §4-§6).

Every org action carries time + resource cost. Three action shapes:

* **auxiliary speech** (``is_auxiliary_speech=True``, ``blocking_duration_ticks=0``):
  short messages/replies/promises — can happen in the same tick as a primary
  action, but consume an aux-speech slot + a little attention.
* **blocking** (``blocking_duration_ticks>=1``): occupies the agent's main time;
  the agent is busy until ``tick + duration`` and cannot start another primary.
* **background** (``can_run_in_background=True``): submit now (``submit_duration``),
  result lands later (``background_duration``); agent is only busy for the submit.

The registry is keyed by ``action_type`` with a per-category fallback, and
``apply_time_modifiers`` scales fatigue/stress/error/burnout for overtime /
late-night / weekend work (spec §9).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional

from environments.org_env.backend.actions.registry import action_category


@dataclass
class ActionDurationSpec:
    action_type: str
    # Time
    blocking_duration_ticks: int = 0
    submit_duration_ticks: int = 0
    background_duration_ticks: int = 0
    can_run_in_background: bool = False
    is_auxiliary_speech: bool = False
    is_recovery: bool = False
    # Overlap rules
    can_overlap_with_aux_speech: bool = True
    can_execute_while_busy: bool = False
    # Schedule constraints
    can_execute_after_hours: bool = True
    can_execute_weekend: bool = True
    default_phase_preference: List[str] = field(default_factory=list)
    # Resource cost
    min_attention_required: float = 0.0
    attention_cost: float = 0.0
    fatigue_delta: float = 0.0
    stress_delta: float = 0.0
    context_switch_delta: float = 0.0
    interruption_cost: float = 0.0
    # Long-term risk
    error_risk_delta: float = 0.0
    burnout_risk_delta: float = 0.0

    @property
    def is_blocking(self) -> bool:
        return self.blocking_duration_ticks > 0 and not self.can_run_in_background

    @property
    def total_busy_ticks(self) -> int:
        """Ticks the agent is occupied (and cannot start a new primary)."""
        if self.can_run_in_background:
            return max(0, self.submit_duration_ticks)
        return max(0, self.blocking_duration_ticks)


# --------------------------------------------------------------------------- #
# v1 duration table (spec §6). Ranges collapsed to a representative value;
# estimate() refines per-context. Helper builders keep the table compact.
# --------------------------------------------------------------------------- #
def _aux(at: str, attn: float = 0.04, fatigue: float = 0.0, stress: float = 0.0) -> ActionDurationSpec:
    return ActionDurationSpec(at, blocking_duration_ticks=0, is_auxiliary_speech=True,
                              attention_cost=attn, fatigue_delta=fatigue, stress_delta=stress,
                              can_execute_while_busy=True)


def _block(at: str, ticks: int = 1, attn: float = 0.10, fatigue: float = 0.04,
           stress: float = 0.02, ctx: float = 0.0, min_attn: float = 0.0,
           phases: Optional[List[str]] = None) -> ActionDurationSpec:
    return ActionDurationSpec(at, blocking_duration_ticks=ticks, attention_cost=attn,
                              fatigue_delta=fatigue, stress_delta=stress, context_switch_delta=ctx,
                              min_attention_required=min_attn, default_phase_preference=phases or [])


def _bg(at: str, submit: int = 1, background: int = 2, attn: float = 0.06,
        fatigue: float = 0.02) -> ActionDurationSpec:
    return ActionDurationSpec(at, submit_duration_ticks=submit, background_duration_ticks=background,
                              can_run_in_background=True, attention_cost=attn, fatigue_delta=fatigue)


_AUX_SPEECH = (
    "quick_check_messages", "read_mention", "acknowledge_message", "reply_thread_short",
    "reply_thread", "send_message", "send_async_update", "ask_for_review",
    "ask_for_clarification", "ask_for_help", "promise_work", "challenge_result",
    "warn_about_risk", "ask_about_payroll", "mention_agent", "defer_until_work_hours",
    "share_signal_to_team", "request_after_hours_help",
)

_BLOCKING_COMM = {
    "inbox_triage": 1, "read_long_thread": 1, "review_thread_history": 1,
    "process_customer_feedback_queue": 2, "prepare_customer_feedback_summary": 2,
    "summarize_external_discussion": 1,
}

_DURATION_TABLE: Dict[str, ActionDurationSpec] = {}

for _at in _AUX_SPEECH:
    _DURATION_TABLE[_at] = _aux(_at, attn=0.03, stress=0.01)
# challenge / warn carry a touch more stress; payroll asks too
_DURATION_TABLE["challenge_result"] = _aux("challenge_result", attn=0.05, stress=0.03)
_DURATION_TABLE["warn_about_risk"] = _aux("warn_about_risk", attn=0.05, stress=0.02)
_DURATION_TABLE["ask_about_payroll"] = _aux("ask_about_payroll", attn=0.04, stress=0.03)

for _at, _t in _BLOCKING_COMM.items():
    _DURATION_TABLE[_at] = _block(_at, ticks=_t, attn=0.08 + 0.04 * (_t - 1),
                                  fatigue=0.03, stress=0.02 + 0.02 * (_t - 1), ctx=0.05)

# work / repo / doc / artifact — blocking deep work
_DURATION_TABLE.update({
    "work_on_task": _block("work_on_task", 2, attn=0.14, fatigue=0.06, stress=0.03,
                           phases=["deep_work_morning", "deep_work_afternoon"]),
    "edit_file": _block("edit_file", 1, attn=0.12, fatigue=0.05),
    "commit_changes": _block("commit_changes", 1, attn=0.08, fatigue=0.03),
    "open_pr": _block("open_pr", 1, attn=0.08, fatigue=0.03),
    "review_pr": _block("review_pr", 1, attn=0.10, fatigue=0.04,
                        phases=["collaboration_review"]),
    "formal_pr_review": _block("formal_pr_review", 2, attn=0.14, fatigue=0.05, stress=0.03,
                               phases=["collaboration_review"]),
    "approve_pr": _block("approve_pr", 1, attn=0.06),
    "request_changes": _block("request_changes", 1, attn=0.08, fatigue=0.03),
    "debug_failure": _block("debug_failure", 2, attn=0.16, fatigue=0.07, stress=0.05),
    "debug_code": _block("debug_code", 2, attn=0.16, fatigue=0.07, stress=0.05),
    "create_doc": _block("create_doc", 2, attn=0.12, fatigue=0.05),
    "edit_doc": _block("edit_doc", 1, attn=0.10, fatigue=0.04),
    "review_doc": _block("review_doc", 1, attn=0.10, fatigue=0.04,
                         phases=["collaboration_review"]),
    "create_experiment_tracker": _block("create_experiment_tracker", 2, attn=0.12, fatigue=0.05),
    "update_experiment_tracker": _block("update_experiment_tracker", 1, attn=0.08),
    "update_tracker": _block("update_tracker", 1, attn=0.08),
    "create_cost_ledger": _block("create_cost_ledger", 2, attn=0.12, fatigue=0.05),
    "create_review_checklist": _block("create_review_checklist", 2, attn=0.12, fatigue=0.05),
    "create_customer_triage_sheet": _block("create_customer_triage_sheet", 2, attn=0.12),
    "export_result_to_tracker": _block("export_result_to_tracker", 1, attn=0.06),
    # search — blocking 1 tick
    "internal_search": _block("internal_search", 1, attn=0.06, fatigue=0.02),
    "repo_search": _block("repo_search", 1, attn=0.06, fatigue=0.02),
    "sandbox_search": _block("sandbox_search", 1, attn=0.06, fatigue=0.02),
    "external_community_search": _block("external_community_search", 1, attn=0.06),
    "frozen_web_search": _block("frozen_web_search", 1, attn=0.06),
    "read_external_feed": _block("read_external_feed", 1, attn=0.06),
    "read_feed": _block("read_feed", 1, attn=0.05),
    # primary text actions (O1.7 §28) — occupy main slot
    "post_company_update": _block("post_company_update", 2, attn=0.12, fatigue=0.04, stress=0.03),
    "respond_to_public_comment": _block("respond_to_public_comment", 1, attn=0.10, stress=0.03),
    "record_meeting_notes": _block("record_meeting_notes", 1, attn=0.08),
    # payroll / admin
    "run_payroll": _block("run_payroll", 1, attn=0.08, stress=0.05),
    "consider_external_offer": _block("consider_external_offer", 1, attn=0.10, stress=0.06),
    # background jobs (§5.2)
    "run_cheap_pilot": _bg("run_cheap_pilot", submit=1, background=2),
    "run_experiment": _bg("run_experiment", submit=1, background=4),
    "run_paper_baseline": _bg("run_paper_baseline", submit=1, background=8),
    "run_script": _bg("run_script", submit=0, background=1),
    "ci_test": _bg("ci_test", submit=0, background=1),
    "install_package": _bg("install_package", submit=0, background=1),
    # time / recovery
    "rest_offline": ActionDurationSpec("rest_offline", blocking_duration_ticks=2, is_recovery=True,
                                       attention_cost=0.0),
    "sleep": ActionDurationSpec("sleep", blocking_duration_ticks=7, is_recovery=True,
                                attention_cost=0.0, can_execute_while_busy=False),
    "work_overtime": _block("work_overtime", 2, attn=0.14, fatigue=0.10, stress=0.05),
    "weekend_work": _block("weekend_work", 2, attn=0.14, fatigue=0.10, stress=0.05),
    # admin / scheduling — instantaneous
    "schedule_meeting": _aux("schedule_meeting", attn=0.02),
    "set_availability_status": _aux("set_availability_status", attn=0.0),
    "handoff_before_offline": _aux("handoff_before_offline", attn=0.03),
})

# per-category default for anything not explicitly listed
_CATEGORY_DEFAULT: Dict[str, ActionDurationSpec] = {
    "work": _block("_work", 1, attn=0.12, fatigue=0.05, stress=0.02),
    "comm": _aux("_comm", attn=0.05, stress=0.01),
    "meeting": _block("_meeting", 1, attn=0.10, fatigue=0.06, stress=0.03),
    "repo": _block("_repo", 1, attn=0.12, fatigue=0.05, stress=0.03),
    "sandbox": _bg("_sandbox", submit=1, background=2),
    "search": _block("_search", 1, attn=0.06, fatigue=0.02),
    "doc": _block("_doc", 1, attn=0.10, fatigue=0.04),
    "artifact": _block("_artifact", 1, attn=0.12, fatigue=0.05),
    "protocol": _aux("_protocol", attn=0.06, stress=0.02),
    "time": _aux("_time", attn=0.0),
    "payroll": _block("_payroll", 1, attn=0.08, stress=0.03),
    "bridge": _block("_bridge", 1, attn=0.06, fatigue=0.02),
}


class ActionDurationRegistry:
    """Lookup + per-context refinement of action durations (spec §4)."""

    def get(self, action_type: str) -> ActionDurationSpec:
        spec = _DURATION_TABLE.get(action_type)
        if spec is not None:
            return spec
        cat = action_category(action_type)
        base = _CATEGORY_DEFAULT.get(cat)
        if base is not None:
            return replace(base, action_type=action_type)
        return ActionDurationSpec(action_type)

    def estimate(self, action: Any, agent: Any, context: Any = None) -> ActionDurationSpec:
        """Refine duration from action params + agent skill (longer if low-skill on
        a hard task; meeting duration from the meeting type)."""
        at = getattr(action, "action_type", str(action))
        spec = self.get(at)
        params = dict(getattr(action, "parameters", {}) or {})
        if at == "attend_meeting":
            from environments.org_env.backend.meetings.system import MEETING_DURATION
            dur = int(params.get("duration") or MEETING_DURATION.get(
                params.get("meeting_type", "daily_sync"), 1))
            return replace(spec, blocking_duration_ticks=max(1, dur))
        # low skill on a blocking work action -> a little longer + more error risk
        if spec.is_blocking and agent is not None and at in ("debug_failure", "debug_code",
                                                             "work_on_task", "formal_pr_review"):
            skill = float(getattr(agent, "skill", lambda *_: 0.5)("core_coding", 0.5))
            if skill < 0.35:
                return replace(spec, blocking_duration_ticks=spec.blocking_duration_ticks + 1,
                               error_risk_delta=spec.error_risk_delta + 0.1)
        return spec

    def apply_time_modifiers(self, spec: ActionDurationSpec, agent: Any, org_clock: Any,
                             context: Any = None) -> ActionDurationSpec:
        """Scale fatigue/stress/error/burnout for overtime / late-night / weekend
        + low-attention error risk (spec §9)."""
        fatigue = spec.fatigue_delta
        stress = spec.stress_delta
        error = spec.error_risk_delta
        burnout = spec.burnout_risk_delta
        if org_clock is not None and getattr(org_clock, "is_after_hours", False):
            fatigue *= 1.2
            stress *= 1.1
            burnout += 0.02
        if org_clock is not None and getattr(org_clock, "is_late_night", False):
            fatigue *= 1.5
            stress *= 1.3
            error += 0.10
            burnout += 0.04
        if org_clock is not None and getattr(org_clock, "is_weekend", False) and not spec.is_recovery:
            fatigue *= 1.2
            burnout += 0.03
        # low remaining attention -> more mistakes
        if agent is not None:
            attn = float(getattr(agent, "work_state", None).attention_remaining_today
                         if getattr(agent, "work_state", None) else
                         agent.vitals.get("attention", 1.0))
            if attn < spec.min_attention_required or attn < 0.25:
                error += 0.15
        return replace(spec, fatigue_delta=fatigue, stress_delta=stress,
                       error_risk_delta=error, burnout_risk_delta=burnout)


DURATION_REGISTRY = ActionDurationRegistry()

__all__ = ["ActionDurationSpec", "ActionDurationRegistry", "DURATION_REGISTRY"]
