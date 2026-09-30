"""Repair the packaging, not just the rule, when a gate walls off finished work.

The policy-repair path can only ease the rule a merge-evidence gate enforces,
and easing it does not move the work. On a tg_automation B3 arm the desk held
correct fixes for four contracts while the mainline carried one: each fix rode
in a branch or a fat pull request whose other, unfinished surfaces kept the
whole request red, so nothing merged after t47 while the same rule was flagged
harmful and relaxed six times over three hundred ticks. Relaxing it enough to
let those requests through would have landed the unfinished surfaces too.

The repair the situation needs is to the packaging. A member who found correct
work stuck behind unrelated stubs would split it out: open a clean request
carrying only the desk's version of one issue's file, check it against that
issue's own gate, and land it when it is green. That is what this does, and it
only fires when the organization's own adopted rule is what is blocking
delivery -- so it is the delivery half of policy repair, not a way around the
gate. Work is landed only when its issue-scoped CI passes; a red repackage is
left as a red request, never forced.
"""
from __future__ import annotations

import hashlib
from typing import Any, Callable, List, Optional, Tuple


def _desk_ahead_of_mainline(art: Any) -> bool:
    """The members wrote something the mainline does not carry yet."""
    content = getattr(art, "content", "") or ""
    mainline = getattr(art, "mainline_content", "") or ""
    return bool(content) and content != mainline


def landable_desk_deliverables(world: Any) -> List[Tuple[str, List[str]]]:
    """Issues whose desk is ahead of the mainline, newest-blocking first.

    An issue qualifies when at least one of its component artifacts has desk
    content the mainline has not promoted. The pack's own scoped CI decides
    whether that desk content is actually correct; this only finds where work is
    waiting to be checked, so a wrong desk is caught by the gate rather than
    landed by this.
    """
    from environments.org_env.product.substrates.issue_stream import (
        _component_artifact_ids)

    stream = world.__dict__.get("_oss_issue_stream") or []
    arts = getattr(world, "product_artifacts", {}) or {}
    out: List[Tuple[str, List[str]]] = []
    for entry in stream:
        issue_id = entry.get("issue_id")
        if not issue_id:
            continue
        ahead = [aid for aid in _component_artifact_ids(world, issue_id)
                 if aid in arts and _desk_ahead_of_mainline(arts[aid])]
        if ahead:
            out.append((str(issue_id), ahead))
    return out


def _repackage_payload_digest(
    world: Any,
    issue_id: str,
    artifact_ids: List[str],
) -> str:
    """Identity of the exact desk payload a repair request carries."""
    arts = getattr(world, "product_artifacts", {}) or {}
    rows = []
    for artifact_id in sorted(set(artifact_ids)):
        artifact = arts.get(artifact_id)
        if artifact is None:
            continue
        rows.append((artifact_id, getattr(artifact, "content", "") or ""))
    if not rows:
        return ""
    digest = hashlib.sha256()
    digest.update(str(issue_id).encode("utf-8"))
    for artifact_id, content in rows:
        for value in (artifact_id, content):
            encoded = value.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
    return digest.hexdigest()


def _repackage_pr_digest(world: Any, pr: Any) -> str:
    """Recover payload identity for new and checkpoint-restored requests."""
    recorded = str(
        getattr(pr, "_delivery_repackage_source_digest", "") or ""
    )
    if recorded:
        return recorded
    patches = getattr(world, "patches", {}) or {}
    rows = []
    for patch_id in (getattr(pr, "patch_ids", []) or []):
        patch = patches.get(patch_id)
        if patch is None:
            continue
        rows.append((
            str(getattr(patch, "target_object_id", "") or ""),
            str(getattr(patch, "new_content", "") or ""),
        ))
    if not rows:
        return ""
    issue_ids = list(getattr(pr, "linked_issue_ids", []) or [])
    issue_id = str(issue_ids[0]) if len(issue_ids) == 1 else ""
    digest = hashlib.sha256()
    digest.update(issue_id.encode("utf-8"))
    for artifact_id, content in sorted(set(rows)):
        for value in (artifact_id, content):
            encoded = value.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
    return digest.hexdigest()


def _clean_repackage_exists(world: Any, issue_id: str) -> bool:
    """Whether a live request carries the desk's *current* issue payload.

    A red request is not permanent evidence that later desk work was packaged.
    If the desk changed after that request was opened, the old head is stale and
    must not suppress the next issue-scoped repair.
    """
    from environments.org_env.product.substrates.issue_stream import (
        _component_artifact_ids)

    arts = getattr(world, "product_artifacts", {}) or {}
    current_ids = [
        artifact_id
        for artifact_id in _component_artifact_ids(world, issue_id)
        if artifact_id in arts and _desk_ahead_of_mainline(arts[artifact_id])
    ]
    current_digest = _repackage_payload_digest(world, issue_id, current_ids)
    for pr in getattr(world.repo_system.repo, "pull_requests", {}).values():
        if getattr(pr, "merged_tick", None) is not None:
            continue
        if not getattr(pr, "_delivery_repackage", False):
            continue
        if list(getattr(pr, "linked_issue_ids", []) or []) == [issue_id]:
            existing_digest = _repackage_pr_digest(world, pr)
            # Unknown legacy payloads stay conservative. Checkpoint-restored
            # repair requests normally reconstruct their digest from patches.
            if not existing_digest or existing_digest == current_digest:
                return True
    return False


def _actor_and_reviewer(
    world: Any,
    issue_id: Optional[str] = None,
) -> Tuple[str, str]:
    """Use the assigned feature owner as author when one is known."""
    assigned = world.__dict__.get("_cooperbench_feature_issue_by_agent") or {}
    owner = next((
        str(agent_id)
        for agent_id, assigned_issue in assigned.items()
        if str(assigned_issue) == str(issue_id)
        and str(agent_id) in world.agents
    ), None)
    founder = next((a for a, ag in world.agents.items()
                    if getattr(ag, "is_founder", False)), None)
    agents = list(world.agents)
    actor = owner or founder or (agents[0] if agents else "system")
    reviewer = next((a for a in agents if a != actor), actor)
    return actor, reviewer


def repackage_and_land(
    world: Any,
    issue_id: str,
    artifact_ids: List[str],
    tick: int,
    *,
    ci_runner: Optional[Callable[[Any, Any], dict]] = None,
) -> dict:
    """Open a single-issue request from the desk, check it, and land it if green.

    Returns a small record: whether CI passed and whether the mainline moved.
    Nothing is forced -- a red repackage stays a red request, and the caller can
    read the brief the gate wrote on it like any other.
    """
    # This generic issue-scoped shortcut is not the ProgramBench public
    # differential gate.  It also writes repo and mainline state directly,
    # outside the profile's action authorization.  The adapted profile must
    # repair its one integration candidate through VERIFY_REPAIR instead.
    if "programbench_profile_state" in getattr(world, "__dict__", {}):
        return {
            "issue_id": issue_id,
            "landed": False,
            "reason": "programbench_requires_verified_integration_candidate",
        }
    from environments.org_env.product.patch_objects import CodePatch

    arts = getattr(world, "product_artifacts", {}) or {}
    targets = [aid for aid in artifact_ids
               if aid in arts and _desk_ahead_of_mainline(arts[aid])]
    if not targets:
        return {"issue_id": issue_id, "landed": False, "reason": "nothing_ahead"}

    payload_digest = _repackage_payload_digest(
        world, issue_id, targets
    )
    # A prior red request may carry an older desk snapshot. Retire only those
    # obsolete repair envelopes; their patches and review evidence remain in
    # the world for audit, while the updated desk can be checked independently.
    from environments.org_env.backend.repo.repo import BranchStatus, PRStatus
    for old_pr in getattr(world.repo_system.repo, "pull_requests", {}).values():
        if getattr(old_pr, "merged_tick", None) is not None:
            continue
        if not getattr(old_pr, "_delivery_repackage", False):
            continue
        if list(getattr(old_pr, "linked_issue_ids", []) or []) != [issue_id]:
            continue
        old_digest = _repackage_pr_digest(world, old_pr)
        if not old_digest or old_digest == payload_digest:
            continue
        old_pr.status = PRStatus.STALE
        old_branch = world.repo_system.repo.branches.get(
            getattr(old_pr, "source_branch", "")
        )
        if old_branch is not None:
            old_branch.status = BranchStatus.ABANDONED
        world.events.append({
            "type": "governance_event",
            "subtype": "delivery_repackage_superseded",
            "issue_id": issue_id,
            "pr_id": getattr(old_pr, "pr_id", ""),
            "old_payload_digest": old_digest,
            "new_payload_digest": payload_digest,
            "tick": int(tick),
            "auto": True,
        })

    actor, reviewer = _actor_and_reviewer(world, issue_id)
    repo = world.repo_system
    branch = repo.create_branch(owner_id=actor, base="main", tick=tick)
    # A repackage does not spend the organization's 16-branch coordination
    # budget (workflow.MAX_ACTIVE_BRANCHES): it is the repair path, not a member
    # carrying work, and counting it would let the congestion it clears keep it
    # from clearing it. The flag also marks the branch's request as a repackage.
    setattr(branch, "_delivery_repackage", True)

    patch_ids: List[str] = []
    for aid in targets:
        art = arts[aid]
        pid = f"deliver_{issue_id}_{aid}_{tick}"
        patch = CodePatch(
            patch_id=pid, target_object_id=aid, actor_id=actor, tick=tick,
            new_content=getattr(art, "content", "") or "",
            edit_goal=f"land desk fix for {issue_id}",
            change_summary=f"repackage {issue_id} from desk into a single-issue request",
            related_issue_ids=[issue_id],
            related_task_ids=list(getattr(art, "linked_task_ids", []) or []),
            validation_status="accepted", applied_tick=tick)
        world.patches[pid] = patch
        patch_ids.append(pid)

    commit = repo.commit_changes(
        agent_id=actor, branch_id=branch.branch_id,
        message=f"deliver {issue_id} from desk",
        changed_files=[getattr(arts[a], "linked_file_path", "") or a for a in targets],
        tick=tick, test_status="passed", risk_level="low",
        patch_ids=patch_ids, artifact_ids=list(targets))
    if commit is None:
        return {"issue_id": issue_id, "landed": False, "reason": "commit_failed"}

    pr = repo.open_pr(agent_id=actor, source_branch=branch.branch_id,
                      reviewers=[reviewer])
    pr.linked_issue_ids = [issue_id]
    pr.linked_task_ids = sorted({t for a in targets
                                 for t in (getattr(arts[a], "linked_task_ids", []) or [])})
    pr.__dict__["_delivery_repackage"] = True
    pr.__dict__["_delivery_repackage_source_digest"] = payload_digest

    # The issue-scoped gate on the mainline plus exactly this request's patches.
    ci = repo.run_ci(pr_id=pr.pr_id, tick=tick)
    run_ci = ci_runner or _default_ci
    verdict = run_ci(world, pr)
    from environments.org_env.product.contracts import record_integration_verdict
    record_integration_verdict(ci, pr, verdict, world=world)
    if verdict.get("ok"):
        pr.ci_brief = ""

    landed = False
    if getattr(pr, "ci_passed", False):
        repo.approve_pr(reviewer_id=reviewer, pr_id=pr.pr_id, tick=tick)
        if repo.merge_pr(pr_id=pr.pr_id, tick=tick):
            world.apply_merged_pr(pr, actor, tick)
            landed = True

    world.events.append({
        "type": "governance_event", "subtype": "delivery_repackaged",
        "issue_id": issue_id, "pr_id": pr.pr_id, "artifact_ids": list(targets),
        "ci_passed": bool(getattr(pr, "ci_passed", False)), "landed": landed,
        "brief": str(getattr(pr, "ci_brief", "") or "")[:200],
        "tick": int(tick), "auto": True})
    return {"issue_id": issue_id, "pr_id": pr.pr_id,
            "ci_passed": bool(getattr(pr, "ci_passed", False)), "landed": landed}


def _default_ci(world: Any, pr: Any) -> dict:
    from environments.org_env.product.contracts import run_integration_ci
    return run_integration_ci(world, pr)
