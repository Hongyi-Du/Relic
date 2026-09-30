"""Minimal organization-level metrics collector (v8g P3 / Stage O5 first cut).

Aggregates raw world state into the handful of organization metrics a paper / org-review
needs: cycle time, review latency, blocker duration, resolution rates, protocol activity,
communication quality, rework, specialization, and token economics. Pure read-only; safe
defaults so it never breaks a snapshot.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


def _avg(xs: List[float]) -> Optional[float]:
    """Mean of the observations, or None when there were none.

    Returning 0.0 for an empty sample is not a safe default here: every average
    in this module is a latency or a duration, where lower is better, so the
    condition that completed no tasks and merged no pull requests scored best
    on all of them. Measured on one pilot pair: the arm with zero merged PRs
    read 0.0 against 88.4 for the arm with seventeen, and the same inversion
    appeared in cycle time and both review latencies.

    `records.py` copies only int/float metrics into the run record, so None
    lands as "not measured" rather than as a number — which is what an empty
    sample means. `coding/metrics.py::rate` already does this.
    """
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 2) if xs else None


def compute_delivery_funnel(world: Any) -> Dict[str, Any]:
    """The stage-by-stage survival of work, kept out of the flat metric map.

    The flat map is copied into the run record one scalar at a time, and a
    funnel read one scalar at a time is a list of counts, not a funnel: the
    thing a reader needs is where the drop is. It travels whole.
    """
    from environments.org_env.runtime_adapter.delivery_funnel import delivery_funnel

    return delivery_funnel(world)


def compute_org_metrics(world: Any) -> Dict[str, Any]:
    tick = int(getattr(world, "world_tick", 0) or 0)
    tasks = list((getattr(world, "tasks", {}) or {}).values())
    arts = list((getattr(world, "product_artifacts", {}) or {}).values())
    gaps = list((getattr(world, "known_gaps", {}) or {}).values())
    repo = getattr(getattr(world, "repo_system", None), "repo", None)
    prs = list(getattr(repo, "pull_requests", {}).values()) if repo is not None else []
    bs = getattr(world, "budget_system", None)
    comm = getattr(world, "comm", None)
    msgs = list(getattr(comm, "messages", {}).values()) if comm is not None else []

    def _done(t):
        return getattr(t.status, "value", str(t.status)) in ("done", "merged", "released")

    # task cycle time: first -> last progress-evidence tick on a completed task
    cycle = []
    for t in tasks:
        if not _done(t):
            continue
        ev = [int(e.get("tick", 0) or 0) for e in (getattr(t, "progress_evidence", []) or [])]
        if ev:
            cycle.append(max(ev) - min(ev))

    # PR review latency (opened -> approved) + merge latency (opened -> merged)
    review_lat, merge_lat = [], []
    for pr in prs:
        op = getattr(pr, "opened_tick", None)
        ap = getattr(pr, "approved_tick", None)
        mg = getattr(pr, "merged_tick", None)
        if op is not None and ap is not None:
            review_lat.append(int(ap) - int(op))
        if op is not None and mg is not None:
            merge_lat.append(int(mg) - int(op))
    merged_prs = sum(1 for pr in prs if getattr(pr.status, "value", str(pr.status)) == "merged")

    # release blocker duration (created -> resolved on rel_blocker_* issues)
    blocker_dur = []
    for a in arts:
        if str(getattr(a, "artifact_id", "")).startswith("rel_blocker_"):
            c = int(getattr(a, "created_at_tick", 0) or 0)
            if getattr(a, "status", "") == "resolved":
                blocker_dur.append(int(getattr(a, "updated_at_tick", c) or c) - c)

    # gap / issue resolution
    gap_resolved = sum(1 for g in gaps if getattr(g, "status", "") in ("resolved", "mitigated"))
    issues = [a for a in arts if getattr(a, "artifact_type", "") == "issue"]
    issues_open = sum(1 for a in issues if getattr(a, "status", "") not in ("resolved", "closed"))

    # protocol activity
    pm = getattr(world, "proposal_manager", None)
    specs = list(getattr(pm, "protocol_specs", {}).values()) if pm is not None else []
    proto_use = sum(int(getattr(s, "use_count", 0) or 0) for s in specs)
    proto_enf = sum(int(getattr(s, "enforcement_count", 0) or 0) for s in specs)
    adopted_protocols = sum(1 for s in specs if getattr(s, "status", "") == "adopted")

    # communication quality
    # A ratio over nothing is not zero. An organization that sent no messages
    # has no linked-message ratio, and one that wrote no patches has no rework
    # rate — reporting 0.0 credits it with perfect behaviour it never showed.
    linked = sum(1 for m in msgs if getattr(m, "linked_objects", None))
    msg_linked_ratio = round(linked / len(msgs), 3) if msgs else None

    # rework rate (rejected patches / all patches)
    patches = list((getattr(world, "patches", {}) or {}).values())
    rejected = sum(1 for p in patches if getattr(p, "validation_status", "") == "rejected")
    rework_rate = round(rejected / len(patches), 3) if patches else None

    # growth: specialization + go-to count
    go_to = sum(len(getattr(a, "go_to_tags", []) or []) for a in (getattr(world, "agents", {}) or {}).values())
    spec_idx = 0.0
    gr = getattr(world, "_growth_reconciler", None) or getattr(world, "growth_reconciler", None)
    try:
        from environments.org_env.growth.objects import REPUTATION_DOMAINS
        ags = list((getattr(world, "agents", {}) or {}).values())
        var = 0.0
        for dmn in REPUTATION_DOMAINS:
            vals = [float((getattr(a, "authority", {}) or {}).get(dmn, 0.0)) for a in ags]
            if vals:
                mu = sum(vals) / len(vals)
                var += sum((v - mu) ** 2 for v in vals) / len(vals)
        spec_idx = round(var / max(1, len(REPUTATION_DOMAINS)), 4)
    except Exception:
        pass

    # token economics
    ledger = list(getattr(bs, "token_ledger", []) or []) if bs is not None else []
    burn = round(sum(e["amount"] for e in ledger if e.get("kind") == "debit"), 1)
    releases = len(getattr(repo, "releases", {}) or {}) if repo is not None else 0

    return {
        "tick": tick,
        "task_cycle_time_avg": _avg(cycle),
        "pr_review_latency_avg": _avg(review_lat),
        "pr_merge_latency_avg": _avg(merge_lat),
        "release_blocker_duration_avg": _avg(blocker_dur),
        # A repository with no recorded gaps has no resolution rate. The old
        # `gap_total = len(gaps) or 1` read that as "none of the one gap was
        # resolved", i.e. a measured zero.
        "gap_resolution_rate": (
            round(gap_resolved / len(gaps), 3) if gaps else None
        ),
        "gap_count": len(gaps),
        "open_issue_count": issues_open,
        "protocol_use_count": proto_use,
        "protocol_enforcement_count": proto_enf,
        "adopted_protocol_count": adopted_protocols,
        "message_linked_ratio": msg_linked_ratio,
        "rework_rate": rework_rate,
        "go_to_count": go_to,
        "specialization_index": spec_idx,
        "token_burn_total": burn,
        # This is the in-world treasury currency, not API tokens: a flat charge
        # per action category plus 0.4 per LLM call however large that call is,
        # on top of a constant daily burn and payroll. Under the old name it
        # reached a paper panel labelled "cost per merged PR / tokens" reading
        # 29.5 where the run's real cost was 43,254 tokens per merged PR. The
        # two do not even agree on ratios: across B0 and B3 the same pair of
        # runs is 11x apart in this unit and 21x apart in tokens, because the
        # constant daily burn is 86% of a quiet arm's ledger and 18% of a busy
        # one's. Publish llm_tokens_per_merged_pr; read this one for treasury
        # pressure, which is what the agents themselves see.
        #
        # Cost per merged PR is undefined without a merged PR. Reporting 0.0
        # put the arm that merged nothing at the cheap end of the cost panel.
        "budget_burn_per_merged_pr": (
            round(burn / merged_prs, 1) if merged_prs else None
        ),
        "merged_pr_count": merged_prs,
        "release_count": releases,
        # The datum for time-to-release existed on ProductRelease and nothing
        # read it, so "how long until the organization shipped" was
        # unmeasurable while a terminal count was reported.
        "first_release_tick": (
            min(
                (
                    int(getattr(release, "released_at_tick", 0) or 0)
                    for release in (getattr(repo, "releases", {}) or {}).values()
                ),
                default=None,
            )
            if repo is not None
            else None
        ),
        "released_at_all": bool(getattr(repo, "releases", {}) or {}),
    }


__all__ = ["compute_org_metrics"]
