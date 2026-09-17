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


def _clean_repackage_exists(world: Any, issue_id: str) -> bool:
    """A repackage request for this issue is already open and unmerged."""
    for pr in getattr(world.repo_system.repo, "pull_requests", {}).values():
        if getattr(pr, "merged_tick", None) is not None:
            continue
        if not getattr(pr, "_delivery_repackage", False):
            continue
        if list(getattr(pr, "linked_issue_ids", []) or []) == [issue_id]:
            return True
    # A compiled admission guard can stop repair after branch/commit but before
    # PR creation.  Keep that staged branch as the in-flight repair rather than
    # creating one blocked branch per deterministic sweep.
    for branch in getattr(world.repo_system.repo, "branches", {}).values():
        if not getattr(branch, "_delivery_repackage", False):
            continue
        if str(getattr(branch, "linked_issue", "") or "") != str(issue_id):
            continue
        if str(getattr(getattr(branch, "status", ""), "value", getattr(branch, "status", ""))) \
                not in {"merged", "abandoned"}:
            return True
    return False


def _actor_and_reviewer(world: Any) -> Tuple[str, str]:
    founder = next((a for a, ag in world.agents.items()
                    if getattr(ag, "is_founder", False)), None)
    agents = list(world.agents)
    actor = founder or (agents[0] if agents else "system")
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
    from environments.org_env.policy.compiled_protocols import (
        authorize_before_action,
    )

    arts = getattr(world, "product_artifacts", {}) or {}
    targets = [aid for aid in artifact_ids
               if aid in arts and _desk_ahead_of_mainline(arts[aid])]
    if not targets:
        return {"issue_id": issue_id, "landed": False, "reason": "nothing_ahead"}

    actor, reviewer = _actor_and_reviewer(world)
    linked_task_ids = sorted({
        str(task_id)
        for artifact_id in targets
        for task_id in (getattr(arts[artifact_id], "linked_task_ids", []) or [])
        if str(task_id)
    })
    repo = world.repo_system
    branch = repo.create_branch(
        owner_id=actor,
        base="main",
        linked_task=(linked_task_ids[0] if linked_task_ids else None),
        tick=tick,
    )
    # A repackage does not spend the organization's 16-branch coordination
    # budget (workflow.MAX_ACTIVE_BRANCHES): it is the repair path, not a member
    # carrying work, and counting it would let the congestion it clears keep it
    # from clearing it. The flag also marks the branch's request as a repackage.
    setattr(branch, "_delivery_repackage", True)
    branch.linked_issue = issue_id

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
        linked_task_id=(linked_task_ids[0] if linked_task_ids else None),
        patch_ids=patch_ids, artifact_ids=list(targets))
    if commit is None:
        return {"issue_id": issue_id, "landed": False, "reason": "commit_failed"}
    commit.linked_task_ids = list(linked_task_ids)
    commit.linked_issue_ids = [issue_id]

    authorization = authorize_before_action(
        world,
        actor,
        "open_pr",
        {"source_branch": branch.branch_id},
        tick=tick,
    )
    if not authorization.allowed:
        return {
            "issue_id": issue_id,
            "branch_id": branch.branch_id,
            "landed": False,
            "reason": "compiled_protocol_blocked:open_pr",
        }
    pr = repo.open_pr(agent_id=actor, source_branch=branch.branch_id,
                      reviewers=[reviewer])
    pr.linked_issue_ids = [issue_id]
    pr.linked_task_ids = list(linked_task_ids)
    pr.__dict__["_delivery_repackage"] = True

    # The issue-scoped gate on the mainline plus exactly this request's patches.
    authorization = authorize_before_action(
        world,
        actor,
        "run_ci",
        {"pr_id": pr.pr_id},
        tick=tick,
    )
    if not authorization.allowed:
        return {
            "issue_id": issue_id,
            "pr_id": pr.pr_id,
            "ci_passed": False,
            "landed": False,
            "reason": "compiled_protocol_blocked:run_ci",
        }
    ci = repo.run_ci(pr_id=pr.pr_id, tick=tick)
    run_ci = ci_runner or _default_ci
    verdict = run_ci(world, pr)
    from environments.org_env.product.contracts import record_integration_verdict
    record_integration_verdict(ci, pr, verdict, world=world)
    if verdict.get("ok"):
        pr.ci_brief = ""

    landed = False
    if getattr(pr, "ci_passed", False):
        review_authorization = authorize_before_action(
            world,
            reviewer,
            "approve_pr",
            {"pr_id": pr.pr_id},
            tick=tick,
        )
        if review_authorization.allowed and repo.approve_pr(
            reviewer_id=reviewer,
            pr_id=pr.pr_id,
            tick=tick,
        ):
            merge_authorization = authorize_before_action(
                world,
                actor,
                "merge_pr",
                {"pr_id": pr.pr_id},
                tick=tick,
            )
            if merge_authorization.allowed and repo.merge_pr(
                pr_id=pr.pr_id,
                tick=tick,
            ):
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
