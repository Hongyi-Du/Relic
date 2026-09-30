"""RoutineScheduler (OrgEnv O1.6, spec Part I §11).

Turns the clock phase + the agent's :class:`RoutineProfile` into a utility *prior*
over candidate actions — it never executes or hard-blocks anything; it only writes
``OrgFeatures.routine_prior`` (a positive/negative scalar the OrgPolicy scorer
weights), so a deep-work hour boosts coding and penalizes meetings, late night
boosts sleep and penalizes nonurgent work, etc.
"""
from __future__ import annotations

from typing import Any, Dict, Set

from environments.org_env.backend.actions import action_category
from environments.org_env.backend.clock.routine import get_routine

_BOOST = 0.6
_PEN = -0.6

# phase -> (boosted action types, penalized action types). Category names also
# match (so "work" boosts all work actions). Concrete types override categories.
PHASE_PRIORS: Dict[str, Dict[str, Set[str]]] = {
    "morning_catchup": {
        "boost": {"quick_check_messages", "inbox_triage", "read_mention", "send_async_update",
                  "update_task_status", "acknowledge_message", "reply_thread_short"},
        "penalize": {"post_company_update", "run_paper_baseline", "schedule_meeting"},
    },
    "daily_sync": {
        "boost": {"attend_meeting", "schedule_meeting", "send_async_update", "present_report"},
        "penalize": {"debug_failure", "debug_code", "run_paper_baseline"},
    },
    "deep_work_morning": {
        "boost": {"work_on_task", "edit_file", "debug_failure", "debug_code", "run_experiment",
                  "run_cheap_pilot", "create_doc", "commit_changes"},
        "penalize": {"schedule_meeting", "inbox_triage", "quick_check_messages"},
    },
    "deep_work_afternoon": {
        "boost": {"work_on_task", "edit_file", "debug_failure", "debug_code", "run_experiment",
                  "run_cheap_pilot", "create_doc", "commit_changes"},
        "penalize": {"schedule_meeting", "inbox_triage", "quick_check_messages"},
    },
    "collaboration_review": {
        "boost": {"review_pr", "formal_pr_review", "review_doc", "reply_thread_short",
                  "ask_for_review", "update_experiment_tracker", "approve_pr", "request_changes"},
        "penalize": set(),
    },
    "wrap_up": {
        "boost": {"send_async_update", "update_task_status", "handoff_before_offline",
                  "update_experiment_tracker"},
        "penalize": {"run_paper_baseline", "schedule_meeting"},
    },
    "evening_overtime": {
        "boost": {"work_overtime", "edit_file", "debug_failure", "send_async_update"},
        "penalize": {"schedule_meeting", "inbox_triage"},
    },
    "late_night": {
        "boost": {"sleep", "rest_offline"},
        "penalize": {"schedule_meeting", "post_company_update", "work_on_task", "create_doc",
                     "run_paper_baseline", "inbox_triage"},
    },
    "lunch_low_activity": {"boost": {"rest_offline", "quick_check_messages"}, "penalize": set()},
    "night_sleep": {"boost": {"sleep", "rest_offline"},
                    "penalize": {"work_on_task", "schedule_meeting", "post_company_update"}},
}


class RoutineScheduler:
    """Writes ``features.routine_prior`` from phase + persona routine (§11)."""

    def modify_features(self, agent: Any, candidate: Any, features: Any, org_clock: Any,
                        context: Any = None) -> Any:
        time_system = getattr(context, "time", None)
        from environments.org_env.cooperbench.work_schedule import compressed_schedule_enabled
        if time_system is not None and (
            not getattr(time_system, "rhythm_enabled", True)
            or compressed_schedule_enabled(time_system)
        ):
            features.routine_prior = 0.0
            return features
        at = candidate.action_type
        cat = action_category(at)
        phase = getattr(org_clock, "phase_of_day", "")
        prior = 0.0
        spec = PHASE_PRIORS.get(phase)
        if spec:
            if at in spec["boost"] or cat in spec["boost"]:
                prior += _BOOST
            if at in spec["penalize"] or cat in spec["penalize"]:
                prior += _PEN
        # persona deep-work hours: extra boost for focused work in *their* hours
        routine = get_routine(getattr(agent, "id", ""))
        hour = int(getattr(org_clock, "hour_in_day", 0))
        if routine.is_deep_work_hour(hour) and cat in ("work", "repo", "sandbox"):
            prior += 0.3
        # deadline pressure flips evening/late-night overtime positive
        deadline = float(getattr(features, "deadline_urgency", 0.0))
        if phase in ("evening_overtime", "late_night") and deadline >= 0.6 \
                and at in ("work_overtime", "edit_file", "debug_failure", "weekend_work"):
            prior += 0.5
        # weekend: boost recovery, allow urgent work only
        if getattr(org_clock, "is_weekend", False):
            if at in ("rest_offline", "sleep", "send_async_update"):
                prior += 0.4
            elif cat in ("work", "repo", "sandbox", "meeting") and deadline < 0.6:
                prior += _PEN
        features.routine_prior = round(prior, 3)
        return features


__all__ = ["RoutineScheduler", "PHASE_PRIORS"]
