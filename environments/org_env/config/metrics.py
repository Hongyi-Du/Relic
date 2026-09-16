"""Condition-neutral OrgEnv metrics for controlled experiments."""
from __future__ import annotations

from dataclasses import dataclass, field
from collections import Counter
from typing import Any, Dict

BEHAVIORAL_METRICS = (
    "expensive_experiment_count", "cheap_pilot_count", "duplicated_experiment_count",
    "unlogged_experiment_count", "budget_use_per_completed_task",
    "task_completion_rate", "deadline_miss_rate",
    # Completion over the pack's own seeded issues. task_completion_rate divides
    # by whatever landed on the board, which differs per arm; this denominator
    # is the same twelve issues on the same clock in every run.
    "seeded_task_completion_rate", "seeded_task_count", "seeded_tasks_completed",
)
SOCIAL_METRICS = (
    "artifact_creation_rate", "artifact_adoption_rate", "protocol_proposal_rate",
    "protocol_use_count", "violation_count", "enforcement_count",
    "ownership_clarity", "review_backlog",
)
EXTERNAL_METRICS = (
    "signal_exposure_count", "external_post_reach", "expert_advice_retrieved",
    "customer_complaint_pressure", "reputation_change",
)
DETECTOR_METRICS = (
    "weak_protocol_emergence_rate", "strong_protocol_emergence_rate",
    "time_to_emergence", "persistence", "repeated_protocol_use_rate",
    "cross_context_protocol_reuse_rate", "protocol_persistence_rate",
    "valid_third_party_enforcement_rate", "protocol_amendment_rate",
    "protocol_repair_rate", "measurable_impact",
    # The preregistration's MIDDLE capability tier — used beyond the episode it
    # was adopted in. It was never evaluated because nothing stamps an episode
    # id on an event, so the published cross-episode figure came from a key that
    # is always unset and actually reported cross-CONTEXT reuse. Episode
    # start/end ticks recover the real quantity without inventing an identifier.
    "episode_transfer_rate",
    # Reported beside it so a 0.0 reads as "did not transfer", never as "could
    # not be measured" — the same ambiguity strong_emergence_oracle_attached
    # exists to remove one line below.
    "episode_transfer_measurable_rate",
    # Stage flag, not an outcome: 0.0 until the final-evaluation oracle is
    # attached, at which point strong_protocol_emergence_rate/measurable_impact
    # first become interpretable. Guards against reading a mid-run 0.0 strong
    # rate as a broken formation chain.
    "strong_emergence_oracle_attached",
    # The denominator all eleven rates above divide by. Without it a run with no
    # protocols and a run whose protocols all failed both read 0.0 everywhere,
    # and three of the four conditions disable institutionalization by
    # definition — so "not applicable" and "tried and did not form" were the
    # same reading on four components of the capability composite.
    "protocol_carrier_count",
)
PROFILE_METRICS = (
    "profile_policy_mean_js_divergence",
    "profile_policy_argmax_shift_rate",
    "profile_policy_mean_chosen_probability_delta",
    "profile_policy_monotonicity_violation_rate",
)
ALL_METRICS = (
    BEHAVIORAL_METRICS
    + SOCIAL_METRICS
    + EXTERNAL_METRICS
    + DETECTOR_METRICS
    + PROFILE_METRICS
)


SEEDED_TASK_PREFIX = "task_oss_issue_"


def _seeded_task_metrics(tasks: Any) -> Dict[str, Any]:
    """Completion over the pack's own issues, the one denominator every arm shares.

    ``tasks`` is the task collection as the caller holds it, values or mapping.
    Returns an empty dict when the substrate seeds none, so a non-OSS run does
    not report a rate over nothing.
    """
    rows = list(tasks.values()) if hasattr(tasks, "values") else list(tasks)
    seeded = [t for t in rows
              if str(getattr(t, "task_id", "")).startswith(SEEDED_TASK_PREFIX)]
    if not seeded:
        return {}
    done = sum(
        1 for t in seeded
        if str(getattr(getattr(t, "status", None), "value",
                       getattr(t, "status", ""))) in ("done", "merged", "released")
    )
    return {
        "seeded_task_completion_rate": round(done / len(seeded), 4),
        "seeded_task_count": len(seeded),
        "seeded_tasks_completed": done,
    }


@dataclass
class OrgMetrics:
    """Read-only collector over world state and its append-only event stream."""
    values: Dict[str, Any] = field(default_factory=lambda: {k: 0 for k in ALL_METRICS})

    def compute(self, *, state: Any, events: Any = None) -> Dict[str, Any]:
        world = state
        event_rows = list(events if events is not None else getattr(world, "events", []) or [])
        action_rows = list(
            getattr(world, "baseline_archived_action_log", []) or []
        ) + list(getattr(world, "action_log", []) or [])
        tasks = list((getattr(world, "tasks", {}) or {}).values())
        tick = max(1, int(getattr(world, "world_tick", 0) or 0))

        from environments.org_env.runtime_adapter.org_metrics import (
            compute_delivery_funnel,
            compute_org_metrics,
        )
        org = compute_org_metrics(world)
        delivery_funnel = compute_delivery_funnel(world)

        def task_done(task: Any) -> bool:
            status = getattr(getattr(task, "status", None), "value", getattr(task, "status", ""))
            return str(status) in ("done", "merged", "released")

        done_count = sum(1 for task in tasks if task_done(task))
        deadline_tasks = [task for task in tasks if getattr(task, "deadline_tick", None) is not None]
        missed_deadlines = sum(
            1
            for task in deadline_tasks
            if int(getattr(task, "deadline_tick", tick)) < tick and not task_done(task)
        )

        sandbox = getattr(world, "sandbox_system", None)
        jobs = list(
            getattr(world, "baseline_archived_sandbox_jobs", []) or []
        )
        results = list(
            getattr(world, "baseline_archived_sandbox_results", []) or []
        )
        if sandbox is not None:
            jobs.extend(getattr(sandbox, "jobs", {}).values())
            results.extend(getattr(sandbox, "results", {}).values())
        expensive_jobs = [
            job
            for job in jobs
            if getattr(job, "job_type", "") in (
                "run_experiment",
                "run_paper_baseline",
                "evaluate_model",
            )
            or float(getattr(job, "cost", 0.0) or 0.0) >= 5.0
        ]
        cheap_jobs = [
            job for job in jobs if getattr(job, "job_type", "") == "run_cheap_pilot"
        ]
        config_counts = Counter(
            str(getattr(job, "config_hash", ""))
            for job in jobs
            if getattr(job, "config_hash", "")
        )
        duplicate_count = sum(max(0, count - 1) for count in config_counts.values())
        unlogged_count = sum(
            1 for result in results if not bool(getattr(result, "logged_to_tracker", False))
        )

        proposal_manager = getattr(world, "proposal_manager", None)
        proposals = list(getattr(proposal_manager, "proposals", {}).values()) \
            if proposal_manager is not None else []
        tools = list(getattr(proposal_manager, "tools", {}).values()) \
            if proposal_manager is not None else []
        protocol_specs = list(getattr(proposal_manager, "protocol_specs", {}).values()) \
            if proposal_manager is not None else []
        adopted_mechanisms = sum(
            1 for tool in tools if getattr(tool, "status", "") == "active"
        ) + sum(
            1 for protocol in protocol_specs if getattr(protocol, "status", "") == "adopted"
        )

        from environments.org_env.experiments.capability_evidence import (
            build_organizational_capability_evidence,
        )

        capability_evidence = build_organizational_capability_evidence(world)
        capability_metrics = dict(capability_evidence["metrics"])
        # Strong emergence needs the final-evaluation oracle; surface the stage
        # next to the rates so a mid-run 0.0 cannot be misread as breakage.
        capability_metrics["strong_emergence_oracle_attached"] = (
            1.0
            if capability_evidence.get("oracle_attachment") == "attached"
            else 0.0
        )
        from environments.org_env.experiments.profile_causality import (
            build_profile_causality_evidence,
        )

        profile_causality_evidence = build_profile_causality_evidence(world)
        profile_metrics = dict(profile_causality_evidence["metrics"])

        external_events = [
            event for event in event_rows if event.get("type") == "external_signal_event"
        ]
        community = getattr(world, "community", None)
        posts = list(getattr(community, "posts", {}).values()) if community is not None else []
        expert_actions = sum(
            1
            for action in action_rows
            if action.get("action_type") in (
                "ask_external_expert",
                "read_external_doc",
                "search_external_posts",
            )
        )
        tickets = list((getattr(world, "tickets", {}) or {}).values())
        complaint_pressure = sum(
            1
            for ticket in tickets
            if getattr(ticket, "status", "open") not in ("closed", "resolved")
        )
        reputation_total = sum(
            float(value or 0.0)
            for agent in (getattr(world, "agents", {}) or {}).values()
            for value in (getattr(agent, "reputation", {}) or {}).values()
        )
        repo = getattr(getattr(world, "repo_system", None), "repo", None)
        pull_requests = list(getattr(repo, "pull_requests", {}).values()) if repo else []
        review_backlog = sum(
            1
            for pull_request in pull_requests
            if str(
                getattr(
                    getattr(pull_request, "status", None),
                    "value",
                    getattr(pull_request, "status", ""),
                )
            )
            not in ("merged", "closed")
        )

        self.values.update(
            {
                "expensive_experiment_count": len(expensive_jobs),
                "cheap_pilot_count": len(cheap_jobs),
                "duplicated_experiment_count": duplicate_count,
                "unlogged_experiment_count": unlogged_count,
                "budget_use_per_completed_task": round(
                    float(org.get("token_burn_total", 0.0)) / max(1, done_count), 4
                ),
                "task_completion_rate": round(done_count / max(1, len(tasks)), 4),
                # The denominator above is whatever ended up on the board, and
                # that is not the same board across arms. The seeded coding
                # issues arrive on a fixed clock -- twelve of them at ticks 12
                # through 144, identical in every run -- while the feedback
                # tasks the world generates alongside them ranged from 4 to 21
                # across four runs of the same pack and seed. A rate over the
                # whole board therefore moves with how much endogenous work
                # happened to spawn: 0.333 / 0.139 / 0.531 / 0.447 over all
                # tasks, against 0.500 / 0.417 / 0.833 / 0.833 over the twelve.
                # Only the second is a comparison between organizations.
                **_seeded_task_metrics(tasks),
                "deadline_miss_rate": round(
                    missed_deadlines / max(1, len(deadline_tasks)), 4
                ),
                "artifact_creation_rate": round(
                    (
                        len(getattr(world, "artifacts", {}) or {})
                        + len(getattr(world, "documents", {}) or {})
                    )
                    / tick,
                    4,
                ),
                "artifact_adoption_rate": round(
                    min(1.0, adopted_mechanisms / max(1, len(proposals))), 4
                ),
                "protocol_proposal_rate": round(
                    sum(
                        1
                        for proposal in proposals
                        if getattr(proposal, "proposal_type", "") in (
                            "protocol_proposal",
                            "policy_repair_proposal",
                        )
                    )
                    / tick,
                    4,
                ),
                "protocol_use_count": int(org.get("protocol_use_count", 0)),
                "violation_count": sum(
                    event.get("type") == "protocol_violation_event"
                    for event in event_rows
                ),
                "enforcement_count": sum(
                    event.get("type") == "protocol_enforcement_event"
                    for event in event_rows
                ),
                "ownership_clarity": round(
                    sum(bool(getattr(task, "owner_id", None)) for task in tasks)
                    / max(1, len(tasks)),
                    4,
                ),
                "review_backlog": review_backlog,
                "signal_exposure_count": len(external_events),
                "external_post_reach": sum(
                    int(getattr(post, "reach", 0) or 0) for post in posts
                ),
                "expert_advice_retrieved": expert_actions,
                "customer_complaint_pressure": complaint_pressure,
                "reputation_change": round(reputation_total, 4),
                # None passes through: it is the reading for "nothing to
                # average", and the live metrics panel showing 0.0 for a run
                # with no protocols is the same misreading the record fixed.
                **{
                    name: (
                        None
                        if capability_metrics[name] is None
                        else round(float(capability_metrics[name]), 4)
                    )
                    for name in DETECTOR_METRICS
                },
                **{
                    name: round(float(profile_metrics[name]), 4)
                    for name in PROFILE_METRICS
                },
            }
        )

        try:
            from environments.org_env.backend.market.validation import product_quality
            quality = product_quality(world)
        except Exception:
            quality = {}
        from environments.org_env.experiments.resources import experiment_resource_snapshot
        return {
            **self.values,
            "experiment_condition": getattr(world, "experiment_condition", None),
            "mechanism_ablations": getattr(
                getattr(world, "mechanism_ablations", None), "arm_id", "full"
            ),
            "resource_control": experiment_resource_snapshot(world),
            "product_quality": quality,
            "org_metrics": org,
            "delivery_funnel": delivery_funnel,
            "organizational_capability_evidence": capability_evidence,
            "profile_causality_evidence": profile_causality_evidence,
        }


__all__ = [
    "BEHAVIORAL_METRICS", "SOCIAL_METRICS", "EXTERNAL_METRICS", "DETECTOR_METRICS",
    "PROFILE_METRICS", "ALL_METRICS", "OrgMetrics",
]
