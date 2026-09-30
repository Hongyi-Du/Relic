"""Explicit member actions for integration evidence; never a hidden approval.

Enabled separately from private source desks so older checkpoints do not gain
trusted plan receipts by migration. Normal PR-head review remains independent.
"""
from __future__ import annotations

import copy
from typing import Any


SCHEMA = "cooperbench_integrated_probe_workflow_v1"
_MARKER = "_cooperbench_integrated_probe_workflow"
ADJUDICATION_RETRY_TICKS = 2
MAX_UNRESOLVED_ADJUDICATIONS = 3


def workflow_enabled(world: Any) -> bool:
    if _MARKER not in getattr(world, "__dict__", {}):
        return False
    from .source_views import SourceViewError, actor_desks_enabled
    marker = world.__dict__[_MARKER]
    if not isinstance(marker, dict) or marker.get("schema_version") != SCHEMA or not actor_desks_enabled(world):
        raise SourceViewError("integrated_probe_workflow_invalid")
    return True


def _status(obj: Any) -> str:
    value = getattr(obj, "status", "")
    return str(getattr(value, "value", value) or "")


def _peer(world: Any, pr: Any) -> str | None:
    members = set((world._cooperbench_sdl_state.get("feature_owners") or {}).values())
    candidates = members - {str(getattr(pr, "author_id", ""))}
    if len(candidates) != 1:
        return None
    return str(next(iter(candidates)))


def integration_pass_matches(world: Any, pr: Any) -> bool:
    if not workflow_enabled(world):
        return True
    from .joint_probe_replay import current_accepted_probe_replay_matches
    receipt = (world.__dict__.get("_cooperbench_pr_integrated_replays") or {}).get(pr.pr_id)
    peer = _peer(world, pr)
    return bool(peer and isinstance(receipt, dict) and current_accepted_probe_replay_matches(
        world, actor_id=peer, receipt=receipt, pull_request=pr,
    ))


def require_coordination(world: Any, evidence: dict, tick: int) -> None:
    """Persist an actionable diagnostic instead of replaying an unowned failure."""
    exhausted = bool(evidence.get("unresolved_probe_ids")) and int(
        evidence.get("adjudication_attempts") or 0
    ) >= MAX_UNRESOLVED_ADJUDICATIONS
    if evidence.get("status") != "needs_coordination" and not exhausted:
        return
    world.__dict__["_cooperbench_coordination_required"] = {
        "kind": "integrated_probe_failure_requires_coordination",
        "tick": int(tick), "failure_id": evidence.get("failure_id"),
        "reason": ("integrated_probe_validity_unresolved_after_bounded_retries"
                   if exhausted else evidence.get("reason")),
        "stage": evidence.get("stage"),
        "source_snapshot": copy.deepcopy(evidence.get("source_snapshot")),
        "plan_set_digest": evidence.get("plan_set_digest"),
        "failure_observations": copy.deepcopy(evidence.get("failure_observations") or []),
        "adjudication_attempts": int(evidence.get("adjudication_attempts") or 0),
        "unresolved_probe_ids": list(evidence.get("unresolved_probe_ids") or []),
        "last_adjudication_error": evidence.get("last_adjudication_error"),
        "repair_paths": [],
    }


def adjudication_retry_block_reason(evidence: dict, tick: int) -> str | None:
    """The scheduler and actual handler share one same-evidence retry budget."""
    attempts = int(evidence.get("adjudication_attempts") or 0)
    if attempts >= MAX_UNRESOLVED_ADJUDICATIONS:
        return "integrated_probe_validity_retry_budget_exhausted"
    if not attempts:
        return None
    # Unknown and transport outcomes retain an explicit retry, but cannot
    # monopolize every tick while source, definitions and observations are
    # unchanged. A different replay/debt identity starts its own budget.
    last_tick = int(evidence.get("adjudicated_tick") or 0)
    if int(tick) < last_tick + ADJUDICATION_RETRY_TICKS:
        return "integrated_probe_validity_retry_cooldown"
    return None


def _active_pr_failure(world: Any, pr_id: str) -> bool:
    from .replay_adjudication import active_replay_failures
    return any(row.get("integration_pr_id") == pr_id and row.get("current")
               for row in active_replay_failures(world))


def action_requests(world: Any, actor_id: str) -> list[tuple[str, dict]]:
    """Read-only candidate descriptions. No tests, calls or source sync here."""
    if not workflow_enabled(world):
        return []
    from .replay_adjudication import (
        pending_replay_adjudications, pending_probe_corrections,
    )
    corrections = pending_probe_corrections(world, actor_id)
    if corrections:
        return [("repair_accepted_probe_plan", {"failure_id": item["failure_id"], "feature_id": item["feature_id"]})
                for item in corrections]
    adjudications = [item for item in pending_replay_adjudications(world, actor_id)
                    if adjudication_retry_block_reason(item, int(getattr(world, "world_tick", 0))) is None]
    if adjudications:
        return [("review_integration_failure", {"failure_id": item["failure_id"]}) for item in adjudications]
    from .replay_adjudication import confirmed_replay_repair_debts
    if any(item.get("repair_phase") == "needs_sync" for item in confirmed_replay_repair_debts(world, actor_id)):
        from .source_views import actor_desk_snapshot, mainline_snapshot
        return [("sync_actor_workspace", {
            "expected_snapshot_id": actor_desk_snapshot(world, actor_id).snapshot_id,
            "mainline_snapshot_id": mainline_snapshot(world).snapshot_id,
        })]
    from .source_ci import current_source_ci_matches
    from .semantic_review import current_semantic_approval
    repo = world.repo_system.repo
    requests = []
    for pr in repo.pull_requests.values():
        if _status(pr) != "approved" or _peer(world, pr) != actor_id:
            continue
        features = set(getattr(pr, "linked_issue_ids", []) or [])
        features &= set(world._cooperbench_sdl_state.get("feature_owners") or {})
        if (len(features) != 1 or not current_source_ci_matches(world, pr)
                or not current_semantic_approval(world, next(iter(features)), pr)):
            continue
        if _active_pr_failure(world, pr.pr_id):
            # Waiting for an adjudication, correction by the other actor or
            # committed owner repair must never execute the old failed plan
            # again. A ready repair on a NEW head has current=False and passes.
            continue
        if not integration_pass_matches(world, pr):
            requests.append(("verify_pr_integration", {"pr_id": pr.pr_id}))
    return requests


def verify_pr_integration(world: Any, actor_id: str, pr_id: str, tick: int) -> dict:
    from .joint_probe_replay import replay_accepted_probe_plans
    from .replay_adjudication import record_replay_failure, resolve_replay_failures
    from .source_views import SourceViewError
    if not workflow_enabled(world):
        raise SourceViewError("integrated_probe_workflow_not_enabled")
    pr = world.repo_system.repo.pull_requests.get(pr_id)
    if pr is None or _status(pr) != "approved" or _peer(world, pr) != actor_id:
        raise SourceViewError("integrated_probe_reviewer_not_peer")
    if _active_pr_failure(world, pr_id):
        raise SourceViewError("integrated_probe_failure_requires_resolution")
    replay = replay_accepted_probe_plans(world, actor_id=actor_id, pull_request=pr)
    world.__dict__.setdefault("_cooperbench_pr_integrated_replays", {})[pr_id] = copy.deepcopy(replay)
    if replay.get("available") and replay.get("ok"):
        resolve_replay_failures(world, actor_id=actor_id, replay=replay, tick=tick)
    elif replay.get("classification") == "unadjudicated_probe_failure":
        debt = record_replay_failure(world, actor_id=actor_id, replay=replay, tick=tick)
        replay = {**replay, "failure_id": debt["failure_id"]}
    return replay


def joint_replay(world: Any, actor_id: str, tick: int) -> dict:
    from .joint_probe_replay import replay_accepted_probe_plans
    from .replay_adjudication import record_replay_failure, resolve_replay_failures
    replay = replay_accepted_probe_plans(world, actor_id=actor_id, require_all_features=True)
    if replay.get("available") and replay.get("ok"):
        resolve_replay_failures(world, actor_id=actor_id, replay=replay, tick=tick)
    elif replay.get("classification") == "unadjudicated_probe_failure":
        debt = record_replay_failure(world, actor_id=actor_id, replay=replay, tick=tick)
        require_coordination(world, debt, tick)
        replay = {**replay, "failure_id": debt["failure_id"]}
    return replay
