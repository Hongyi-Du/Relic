"""The delivery chain as an auditable object: what was reachable, what happened.

A condition that ships nothing is only informative if the record shows it could
have shipped. Two things are needed for that and neither existed: at each
decision, which delivery steps were on the menu and why the absent ones were
absent; and over the run, how far work travelled from an edit to a release.

Both are read-only views over state the runtime already keeps. ``availability``
reads the candidate pool the agent was actually handed plus the preconditions
each missing stage failed, so "commit_patch was offered 41 times and taken
twice" and "merge_pr was never offered because no PR ever passed CI" are
different sentences in the log rather than the same silence. ``delivery_funnel``
counts the stage-to-stage survival of the work itself, and splits each stage by
whether an agent did it or the deterministic backstop did.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

# The chain a change travels, in order. Several stages accept more than one
# action: reviewing is `review_pr`/`approve_pr` in the repo vocabulary and
# `formal_pr_review` in the speech-act one, and they reach the same transition.
DELIVERY_STAGES: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("implement", ("edit_repo_file", "edit_file")),
    ("verify", ("run_public_tests",)),
    ("commit", ("commit_patch", "commit_changes")),
    ("open_pr", ("open_pr",)),
    ("ci", ("run_ci",)),
    ("review", ("formal_pr_review", "review_pr", "approve_pr", "request_changes")),
    ("merge", ("merge_pr",)),
    ("release_candidate", ("create_release_candidate",)),
    ("release_gate", ("run_launch_readiness_check",)),
    ("release_approve", ("approve_release_candidate",)),
    ("release_publish", ("publish_product_release",)),
)

STAGE_BY_ACTION: Dict[str, str] = {
    action: stage for stage, actions in DELIVERY_STAGES for action in actions
}

DELIVERY_ACTIONS: frozenset = frozenset(STAGE_BY_ACTION)

_RECENT_LOG_CAP = 400


def _pr_status(pr) -> str:
    return str(getattr(getattr(pr, "status", None), "value", getattr(pr, "status", "")))


def _blocked_reason(world: Any, agent_id: str, stage: str) -> str:
    """Why ``stage`` has no candidate: the precondition it failed, named.

    Empty means the stage's precondition holds and its absence from the pool is
    a masking or role decision rather than a state one - which the caller
    distinguishes, because those are different accusations.
    """
    repo = getattr(getattr(world, "repo_system", None), "repo", None)
    if repo is None:
        return "no_repo_system"
    prs = list(getattr(repo, "pull_requests", {}).values())
    if stage == "commit":
        from environments.org_env.backend.repo.workflow import pending_for_agent
        if not pending_for_agent(world, agent_id):
            return "no_accepted_patch_awaiting_commit"
        return ""
    if stage == "open_pr":
        mine = [b for b in getattr(repo, "branches", {}).values()
                if getattr(b, "owner_id", None) == agent_id]
        if not any(str(getattr(getattr(b, "status", None), "value", b.status)) == "ready_for_pr"
                   and getattr(b, "commit_ids", None) for b in mine):
            return "no_committed_branch_ready_for_pr"
        return ""
    if stage == "ci":
        if not any(_pr_status(pr) in ("open", "review_requested", "approved", "changes_requested")
                   and not getattr(pr, "ci_passed", False) for pr in prs):
            return "no_open_pr_awaiting_ci"
        return ""
    if stage == "review":
        if not any(agent_id in (getattr(pr, "reviewers", []) or [])
                   and _pr_status(pr) in ("open", "review_requested", "changes_requested")
                   for pr in prs):
            return "no_pr_assigned_to_me_for_review"
        return ""
    if stage == "merge":
        if not any(_pr_status(pr) == "approved" and getattr(pr, "ci_passed", False)
                   for pr in prs):
            return "no_approved_pr_with_passing_ci"
        return ""
    if stage.startswith("release"):
        return _release_blocked_reason(world, repo, prs, stage)
    if stage == "verify":
        return _verify_blocked_reason(world)
    return ""


def _verify_blocked_reason(world: Any) -> str:
    """Why the public suite is not on offer - and whether that is the substrate.

    A substrate that ships no runnable suite denies verification to every
    condition equally, which is a different sentence from "this agent was not
    allowed to verify" and has to read differently in the log.
    """
    from environments.org_env.product.materialize import (
        _repo_hash,
        declared_public_test_command,
    )
    from environments.org_env.product.substrates.issue_stream import (
        unpatched_coding_issues,
    )

    try:
        if not declared_public_test_command(world):
            return "substrate_declares_no_public_test_suite"
        if not unpatched_coding_issues(world):
            return "no_unresolved_coding_work"
        if world.__dict__.get("_public_tests_last_hash") == _repo_hash(
            world, prefer_mainline=False
        ):
            return "tree_unchanged_since_last_run"
    except Exception:
        return "public_test_state_unavailable"
    return ""


def _release_blocked_reason(world: Any, repo, prs, stage: str) -> str:
    candidates = list(getattr(repo, "release_candidates", {}).values())
    in_flight = next((rc for rc in candidates
                      if getattr(rc, "status", "") in
                      ("draft", "under_review", "approved", "blocked")), None)
    if stage == "release_candidate":
        if in_flight is not None:
            return f"release_candidate_in_flight:{in_flight.status}"
        if not any(getattr(pr, "patch_ids", None) and _pr_status(pr) == "merged"
                   for pr in prs):
            return "nothing_merged_to_release"
        return ""
    if in_flight is None:
        return "no_release_candidate"
    status = str(getattr(in_flight, "status", ""))
    if stage == "release_gate":
        return "" if status in ("draft", "blocked") else f"candidate_is_{status}"
    if stage == "release_approve":
        if status not in ("draft", "under_review"):
            return f"candidate_is_{status}"
        blockers = list(getattr(in_flight, "blockers", []) or [])
        return f"gates_failing:{','.join(blockers)}" if blockers else ""
    if stage == "release_publish":
        if status != "approved":
            blockers = list(getattr(in_flight, "blockers", []) or [])
            if blockers:
                return f"gates_failing:{','.join(blockers)}"
            return f"candidate_is_{status}"
        return ""
    return ""


def availability(
    world: Any,
    agent_id: str,
    pool: Iterable[Any],
    masked: Optional[Mapping[str, str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Per stage: offered on the menu, masked out, or unreachable from state."""
    offered: Dict[str, List[str]] = {}
    for candidate in pool:
        stage = STAGE_BY_ACTION.get(getattr(candidate, "action_type", ""))
        if stage:
            offered.setdefault(stage, []).append(candidate.action_type)
    masked = dict(masked or {})
    out: Dict[str, Dict[str, Any]] = {}
    for stage, actions in DELIVERY_STAGES:
        if stage in offered:
            out[stage] = {"offered": True, "actions": sorted(set(offered[stage]))}
            continue
        mask_reason = next((masked[a] for a in actions if a in masked), "")
        out[stage] = {
            "offered": False,
            "blocked_by": _blocked_reason(world, agent_id, stage),
            "masked_reason": mask_reason,
        }
    return out


def record_availability(world: Any, agent_id: str, tick: int, pool: Iterable[Any]) -> Dict[str, Any]:
    """Append one decision-point record and return it so the caller can fill in
    the choice once the policy or the LLM has made one."""
    masked = {
        str(entry.get("action")): str(entry.get("reason") or "")
        for entry in (getattr(world, "_attractor_masked", {}) or {}).get(agent_id, [])
    }
    stages = availability(world, agent_id, pool, masked)
    record: Dict[str, Any] = {
        "tick": int(tick),
        "agent_id": agent_id,
        "pool_size": len(list(pool)),
        "stages": stages,
        "chosen_action": None,
        "chosen_stage": None,
    }
    tally = world.__dict__.setdefault("delivery_availability_tally", {})
    for stage, info in stages.items():
        row = tally.setdefault(
            stage, {"offered": 0, "chosen": 0, "masked": 0, "unreachable": 0}
        )
        if info["offered"]:
            row["offered"] += 1
        elif info.get("masked_reason"):
            row["masked"] += 1
        else:
            row["unreachable"] += 1
    log = world.__dict__.setdefault("delivery_availability_log", [])
    log.append(record)
    if len(log) > _RECENT_LOG_CAP:
        del log[: len(log) - _RECENT_LOG_CAP]
    return record


def note_choice(world: Any, record: Optional[Dict[str, Any]], action_type: str) -> None:
    if record is None:
        return
    record["chosen_action"] = action_type
    stage = STAGE_BY_ACTION.get(action_type)
    record["chosen_stage"] = stage
    if stage:
        tally = world.__dict__.setdefault("delivery_availability_tally", {})
        tally.setdefault(
            stage, {"offered": 0, "chosen": 0, "masked": 0, "unreachable": 0}
        )["chosen"] += 1


def _auto_split(world: Any, event_type: str, subtype: str) -> Tuple[int, int]:
    """(agent_initiated, automatic) counts for one world event subtype.

    The deterministic backstops that advance a stalled PR or cut an overdue
    release tag their events ``auto``. Reporting a merge count without that
    split credits an organization for transitions the runtime performed on its
    behalf, which is exactly the number a reader of a one-person condition needs
    to be able to discount.
    """
    agent = auto = 0
    for event in getattr(world, "events", []) or []:
        if event.get("type") != event_type or event.get("subtype") != subtype:
            continue
        if event.get("auto"):
            auto += 1
        else:
            agent += 1
    return agent, auto


def _rate(numerator: int, denominator: int) -> Optional[float]:
    """Conversion between two stages, or None when the upstream stage is empty.

    A ratio over nothing is not zero: an organization that wrote no patches has
    no patch-to-merge conversion, and 0.0 would read as "converted none of
    them" - the same failure ``org_metrics._avg`` documents for latencies.
    """
    return round(numerator / denominator, 3) if denominator else None


def delivery_funnel(world: Any) -> Dict[str, Any]:
    """Stage-to-stage survival of the work, from an edit to a shipped release."""
    repo = getattr(getattr(world, "repo_system", None), "repo", None)
    patches = list((getattr(world, "patches", {}) or {}).values())
    accepted = [p for p in patches
                if str(getattr(p, "validation_status", "")) in ("accepted", "applied", "merged")]
    edits = sum(1 for entry in (getattr(world, "action_log", []) or [])
                if entry.get("action_type") in ("edit_repo_file", "edit_file"))
    commits = list(getattr(repo, "commits", {}).values()) if repo is not None else []
    prs = list(getattr(repo, "pull_requests", {}).values()) if repo is not None else []
    merged = [pr for pr in prs if _pr_status(pr) == "merged"]
    merged_with_patches = [pr for pr in merged if getattr(pr, "patch_ids", None)]
    releases = list(getattr(repo, "releases", {}).values()) if repo is not None else []
    candidates = list(getattr(repo, "release_candidates", {}).values()) if repo is not None else []

    merge_agent, merge_auto = _auto_split(world, "repo_event", "pr_merged")
    pr_agent, pr_auto = _auto_split(world, "repo_event", "pr_opened")
    review_agent, review_auto = _auto_split(world, "repo_event", "pr_reviewed")
    release_agent, release_auto = _auto_split(world, "release_event", "published")
    _internal_agent, internal_auto = _auto_split(
        world, "release_event", "published_internal"
    )

    return {
        "edits_attempted": edits,
        "patches_written": len(patches),
        "patches_accepted": len(accepted),
        "commits": len(commits),
        "prs_opened": len(prs),
        "prs_merged": len(merged),
        "merged_prs_carrying_patches": len(merged_with_patches),
        "release_candidates": len(candidates),
        "releases_published": len(releases),
        # Where the chain leaks. Each is None when its upstream stage is empty.
        "edit_to_patch": _rate(len(accepted), edits),
        "patch_to_commit": _rate(len(commits), len(accepted)),
        "commit_to_pr": _rate(len(prs), len(commits)),
        "pr_to_merge": _rate(len(merged), len(prs)),
        "merge_to_release": _rate(len(releases), len(merged_with_patches)),
        # Who performed the transition: the organization, or the backstop.
        "agent_initiated": {
            "prs_opened": pr_agent,
            "reviews": review_agent,
            "merges": merge_agent,
            "releases": release_agent,
        },
        "automatic": {
            "prs_opened": pr_auto,
            "reviews": review_auto,
            "merges": merge_auto,
            "releases": release_auto + internal_auto,
        },
        "availability": dict(getattr(world, "delivery_availability_tally", {}) or {}),
    }


__all__ = [
    "DELIVERY_ACTIONS",
    "DELIVERY_STAGES",
    "STAGE_BY_ACTION",
    "availability",
    "delivery_funnel",
    "note_choice",
    "record_availability",
]
