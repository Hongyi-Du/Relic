"""Seat-scoped, read-only resource details.

The normal P2 view is deliberately compact.  This module supplies the lazy
detail seam used by the P3 Resource Inspector without turning a browser request
into an omniscient world lookup.  Every invocation starts from the current
seat-visible object projection; PR children are then admitted only through the
visible PR's recorded commit/patch/CI lineage.

It intentionally does *not* implement actions.  Approve, edit, run CI, merge,
and every other consequential operation continue through the normal P1/P2
gateway.
"""
from __future__ import annotations

import copy
import hashlib
from typing import Any, Dict, Iterable, List, Optional, Tuple

from environments.org_env.human import visibility as vis


_PR_SECTIONS = ("overview", "review", "ci", "commits", "patches", "files", "diff", "provenance")
_TASK_SECTIONS = ("overview", "review", "ci", "files", "diff", "provenance")
_PROPOSAL_SECTIONS = ("overview", "review", "provenance")
RESOURCE_SECTIONS_BY_KIND = {
    "pull_request": _PR_SECTIONS,
    "task": _TASK_SECTIONS,
    "proposal": _PROPOSAL_SECTIONS,
}


def _enum(value: Any) -> str:
    return str(getattr(value, "value", value) or "")


def _plain_list(value: Any) -> List[Any]:
    return list(value or [])


def _content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _artifact_visible(world: Any, artifact: Any, agent_id: str) -> bool:
    """Mirror the existing seat tool's product-artifact visibility contract.

    ProductArtifact is older than the generic workspace object and many
    snapshots do not carry a visibility attribute.  Its historical default is
    team-visible; explicit private/channel visibility still requires ownership.
    This is deliberately narrower than dumping ``world.product_artifacts``.
    """
    if artifact is None:
        return False
    if getattr(artifact, "owner_agent_id", None) == agent_id:
        return True
    return _enum(getattr(artifact, "visibility", "team")).lower() in vis.TEAM_VISIBLE


def _availability(*, available: bool, reason: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {"available": bool(available)}
    if reason:
        out["reason"] = reason
    return out


def _unavailable(resource_kind: str, resource_id: str, reason: str = "resource_not_visible") -> Dict[str, Any]:
    """Do not distinguish unknown from non-visible resources to callers."""
    return {
        "error": reason,
        "resource": {"kind": str(resource_kind), "id": str(resource_id)},
        "availability": {"overview": _availability(available=False, reason=reason)},
    }


def _visible_row(view: Dict[str, Any], resource_kind: str, resource_id: str) -> Optional[Dict[str, Any]]:
    for rows in (view.get("objects") or {}).values():
        for row in rows or []:
            if (str(row.get("kind") or "") == resource_kind
                    and str(row.get("id") or "") == resource_id):
                return row
    return None


def _generic_resource(view: Dict[str, Any], resource_kind: str, resource_id: str,
                      section: str) -> Dict[str, Any]:
    row = _visible_row(view, resource_kind, resource_id)
    if row is None:
        return _unavailable(resource_kind, resource_id)
    normalized_section = section or "overview"
    available = normalized_section in ("overview", "provenance")
    content = (copy.deepcopy(row) if normalized_section == "overview" else {
        "record": copy.deepcopy(row),
        "recording_note": (
            "This is the complete current seat-visible projection for this resource; "
            "no unrestricted backend fields were added."
        ),
    }) if available else {}
    return {
        "resource": {
            "kind": resource_kind,
            "id": resource_id,
            "title": str(row.get("title") or resource_id),
            "status": str(row.get("status") or ""),
        },
        "section": normalized_section,
        "availability": {
            "overview": _availability(available=True),
            "provenance": _availability(available=True),
            normalized_section: _availability(
                available=available,
                reason="detail_section_not_recorded_for_this_resource" if not available else "",
            ),
        },
        # This is a deep copy of the already allowlisted P2 projection, never
        # a vars() dump of the omniscient domain object.
        "content": content,
        "links": [],
        "provenance": {
            "source": "current_seat_view",
            "view_version": view.get("version"),
        },
    }


def _proposal_resource(world: Any, resource_id: str, section: str,
                       view: Dict[str, Any]) -> Dict[str, Any]:
    """Expose the complete decision packet for a currently visible proposal.

    Proposal review is the second audited rich adapter after PR review.  It is
    deliberately explicit: source reflections/episodes and arbitrary future
    fields are not serialized just because the proposal object has them.
    """
    row = _visible_row(view, "proposal", resource_id)
    manager = getattr(world, "proposal_manager", None)
    proposal = (getattr(manager, "proposals", {}) or {}).get(resource_id) if manager else None
    if row is None or proposal is None:
        return _unavailable("proposal", resource_id)
    normalized_section = section or "overview"
    availability = {
        "overview": _availability(available=True),
        "review": _availability(available=True),
        "provenance": _availability(available=True),
    }
    if normalized_section == "overview":
        content = copy.deepcopy(row)
    elif normalized_section == "review":
        content = {
            "summary": str(getattr(proposal, "summary", "") or ""),
            "target_problem": str(getattr(proposal, "target_problem", "") or ""),
            "proposed_solution": str(getattr(proposal, "proposed_solution", "") or ""),
            "required_actions": _plain_list(getattr(proposal, "required_actions", [])),
            "required_capabilities": _plain_list(getattr(proposal, "required_capabilities", [])),
            "required_artifacts": _plain_list(getattr(proposal, "required_artifacts", [])),
            "required_participants": _plain_list(getattr(proposal, "required_participants", [])),
            "affected_agents": _plain_list(getattr(proposal, "affected_agents", [])),
            "affected_objects": _plain_list(getattr(proposal, "affected_objects", [])),
            "affected_protocols": _plain_list(getattr(proposal, "affected_protocols", [])),
            "expected_benefits": _plain_list(getattr(proposal, "expected_benefits", [])),
            "expected_costs": _plain_list(getattr(proposal, "expected_costs", [])),
            "risks": _plain_list(getattr(proposal, "risks", [])),
            "failure_modes": _plain_list(getattr(proposal, "failure_modes", [])),
            "feasibility_score": getattr(proposal, "feasibility_score", None),
            "usefulness_score": getattr(proposal, "usefulness_score", None),
            "risk_score": getattr(proposal, "risk_score", None),
            "adoption_score": getattr(proposal, "adoption_score", None),
            "suggested_revision": str(getattr(proposal, "suggested_revision", "") or ""),
            "approval_required_from": _plain_list(getattr(proposal, "approval_required_from", [])),
            "approved_by": _plain_list(getattr(proposal, "approved_by", [])),
            "rejected_by": _plain_list(getattr(proposal, "rejected_by", [])),
            "rejection_reason": str(getattr(proposal, "rejection_reason", "") or ""),
        }
    elif normalized_section == "provenance":
        content = {
            "parent": {"kind": "proposal", "id": resource_id},
            "proposal_type": str(getattr(proposal, "proposal_type", "") or ""),
            "proposer": str(getattr(proposal, "proposer_agent_id", "") or ""),
            "created_at_tick": getattr(proposal, "created_at_tick", None),
            "updated_at_tick": getattr(proposal, "updated_at_tick", None),
            "status": _enum(getattr(proposal, "status", "")),
            "recording_note": "Decision material is read from the current domain proposal record.",
        }
    else:
        availability[normalized_section] = _availability(
            available=False, reason="detail_section_not_recorded_for_this_resource")
        content = {}
    return {
        "resource": {
            "kind": "proposal", "id": resource_id,
            "title": str(row.get("title") or resource_id),
            "status": str(row.get("status") or ""),
        },
        "section": normalized_section,
        "availability": availability,
        "content": content,
        "links": [],
        "provenance": {
            "source": "domain_proposal_record",
            "view_version": view.get("version"),
            "seat_visibility_revalidated": True,
        },
    }


def _visible_pr(world: Any, agent_id: str, resource_id: str) -> Optional[Any]:
    pr = (getattr(getattr(getattr(world, "repo_system", None), "repo", None),
                  "pull_requests", {}) or {}).get(resource_id)
    return pr if pr is not None and vis.pull_request_visible_to(pr, agent_id) else None


def _pr_overview(pr: Any, view: Dict[str, Any]) -> Dict[str, Any]:
    row = _visible_row(view, "pull_request", str(getattr(pr, "pr_id", ""))) or {}
    return {
        "id": str(getattr(pr, "pr_id", "")),
        "title": str(row.get("title") or getattr(pr, "title", "") or getattr(pr, "pr_id", "")),
        "author": str(getattr(pr, "author_id", "")),
        "source_branch": str(getattr(pr, "source_branch", "")),
        "target_branch": str(getattr(pr, "target_branch", "main")),
        "status": _enum(getattr(pr, "status", "")),
        "reviewers": _plain_list(getattr(pr, "reviewers", [])),
        "linked_task": getattr(pr, "linked_task", None),
        "linked_task_ids": _plain_list(getattr(pr, "linked_task_ids", [])),
        "linked_issue": getattr(pr, "linked_issue", None),
        "linked_issue_ids": _plain_list(getattr(pr, "linked_issue_ids", [])),
        "opened_tick": getattr(pr, "opened_tick", None),
        "approved_tick": getattr(pr, "approved_tick", None),
        "merged_tick": getattr(pr, "merged_tick", None),
        "reviewed": bool(getattr(pr, "reviewed", False)),
        "ci_passed": getattr(pr, "ci_passed", None),
        "ci_base_main_commit_ids": _plain_list(getattr(pr, "ci_base_main_commit_ids", None)),
    }


def _pr_review(pr: Any) -> Dict[str, Any]:
    comments = []
    for comment in _plain_list(getattr(pr, "review_comments", [])):
        if not isinstance(comment, dict):
            continue
        # Keep the domain record exact, but never turn arbitrary future fields
        # into a disclosure channel.
        comments.append({
            "reviewer": str(comment.get("reviewer") or ""),
            "approve": bool(comment.get("approve", False)),
            "comment": str(comment.get("comment") or ""),
            **({"review_id": str(comment["review_id"])} if comment.get("review_id") else {}),
            **({"tick": comment["tick"]} if comment.get("tick") is not None else {}),
        })
    return {
        "reviewed": bool(getattr(pr, "reviewed", False)),
        "comments": comments,
        "approved_by": _plain_list(getattr(pr, "approved_by", [])),
        "requested_changes": _plain_list(getattr(pr, "requested_changes", [])),
        "recording_note": (
            "No review comments are recorded." if not comments else
            "Each entry is the domain-recorded reviewer verdict and comment."
        ),
    }


def _pr_ci(world: Any, pr: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    repo = world.repo_system.repo
    runs = []
    missing: List[str] = []
    invalid: List[str] = []
    for ci_id in _plain_list(getattr(pr, "ci_run_ids", [])):
        ci = (getattr(repo, "ci_runs", {}) or {}).get(ci_id)
        if ci is None:
            missing.append(str(ci_id))
            continue
        # A corrupted/stale child id cannot make an unrelated run inspectable.
        if str(getattr(ci, "ci_id", "")) != str(ci_id) or str(getattr(ci, "pr_id", "")) != str(pr.pr_id):
            invalid.append(str(ci_id))
            continue
        runs.append({
            "id": str(ci.ci_id),
            "pr_id": str(ci.pr_id),
            "commit_id": getattr(ci, "commit_id", None),
            "created_at_tick": getattr(ci, "created_at_tick", None),
            "status": str(getattr(ci, "status", "pending")),
            "checks": copy.deepcopy(_plain_list(getattr(ci, "checks", []))),
            "failure_reasons": _plain_list(getattr(ci, "failure_reasons", [])),
            # Runtime integration may attach these dynamic, public attestation
            # fields.  They are included only when the domain record has them.
            **({"tree_hash": str(getattr(ci, "tree_hash"))} if getattr(ci, "tree_hash", None) else {}),
            **({"brief": str(getattr(ci, "brief"))} if getattr(ci, "brief", None) else {}),
        })
    return ({
        "runs": runs,
        "pr_ci_passed": getattr(pr, "ci_passed", None),
        "ci_base_main_commit_ids": _plain_list(getattr(pr, "ci_base_main_commit_ids", None)),
    }, _availability(
        available=True,
        reason=("no_ci_runs_recorded" if not runs and not missing and not invalid else ""),
    ) | ({"missing_run_ids": missing, "invalid_run_ids": invalid} if missing or invalid else {}))


def _pr_commits(world: Any, pr: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    repo = world.repo_system.repo
    commits = []
    missing = []
    for commit_id in _plain_list(getattr(pr, "commit_ids", [])):
        commit = (getattr(repo, "commits", {}) or {}).get(commit_id)
        if commit is None:
            missing.append(str(commit_id))
            continue
        commits.append({
            "id": str(getattr(commit, "commit_id", commit_id)),
            "author": str(getattr(commit, "author_id", "")),
            "branch": str(getattr(commit, "branch_id", "")),
            "message": str(getattr(commit, "message", "")),
            "changed_files": _plain_list(getattr(commit, "changed_files", [])),
            "linked_task_id": getattr(commit, "linked_task_id", None),
            "linked_task_ids": _plain_list(getattr(commit, "linked_task_ids", [])),
            "linked_issue_ids": _plain_list(getattr(commit, "linked_issue_ids", [])),
            "timestamp": getattr(commit, "timestamp", None),
            "test_status": str(getattr(commit, "test_status", "unknown")),
            "risk_level": str(getattr(commit, "risk_level", "")),
            "status": str(getattr(commit, "status", "")),
            "patch_ids": _plain_list(getattr(commit, "patch_ids", [])),
        })
    return ({"commits": commits}, _availability(
        available=True,
        reason="no_commits_recorded" if not commits and not missing else "",
    ) | ({"missing_commit_ids": missing} if missing else {}))


def _pr_visible_patches(world: Any, pr: Any, agent_id: str) -> Tuple[List[Tuple[Any, Any]], Dict[str, Any]]:
    """Return only PR-linked patches whose linked artifact is currently visible."""
    repo = world.repo_system.repo
    allowed = {str(pid) for pid in _plain_list(getattr(pr, "patch_ids", []))}
    commit_allowed = {
        str(pid)
        for commit_id in _plain_list(getattr(pr, "commit_ids", []))
        for pid in _plain_list(getattr((getattr(repo, "commits", {}) or {}).get(commit_id), "patch_ids", []))
    }
    # A patch must be in the PR's own chain; a commit is supporting
    # provenance, never an alternative way to inject an unrelated patch.
    # Patch content is code-level material, so the parent chain is strict:
    # every disclosed patch must be linked from both the PR and one of its
    # recorded commits.  A missing/dangling commit therefore fails closed.
    allowed &= commit_allowed
    output: List[Tuple[Any, Any]] = []
    hidden_or_missing = 0
    for patch_id in sorted(allowed):
        patch = (getattr(world, "patches", {}) or {}).get(patch_id)
        if patch is None:
            hidden_or_missing += 1
            continue
        artifact = (getattr(world, "product_artifacts", {}) or {}).get(
            getattr(patch, "target_object_id", None))
        if not _artifact_visible(world, artifact, agent_id):
            hidden_or_missing += 1
            continue
        output.append((patch, artifact))
    return output, _availability(
        available=bool(output),
        reason=("no_visible_pr_linked_patches" if not output else ""),
    ) | ({"unavailable_patch_count": hidden_or_missing} if hidden_or_missing else {})


def _patch_metadata(patch: Any, artifact: Any) -> Dict[str, Any]:
    return {
        "id": str(getattr(patch, "patch_id", "")),
        "patch_type": str(getattr(patch, "patch_type", "")),
        "actor": str(getattr(patch, "actor_id", "")),
        "tick": getattr(patch, "tick", None),
        "edit_goal": str(getattr(patch, "edit_goal", "")),
        "status": str(getattr(patch, "validation_status", "pending")),
        "rejection_reason": getattr(patch, "rejection_reason", None),
        "applied_tick": getattr(patch, "applied_tick", None),
        "creates_file": bool(getattr(patch, "creates_file", False)),
        "base_mainline_revision": getattr(patch, "base_mainline_revision", None),
        "change_summary": str(getattr(patch, "change_summary", "")),
        "artifact": {
            "id": str(getattr(artifact, "artifact_id", "")),
            "title": str(getattr(artifact, "title", "")),
            "path": str(getattr(artifact, "linked_file_path", "") or ""),
            "revision": getattr(artifact, "revision", None),
            "mainline_revision": getattr(artifact, "mainline_revision", None),
        },
        "related_task_ids": _plain_list(getattr(patch, "related_task_ids", [])),
        "related_issue_ids": _plain_list(getattr(patch, "related_issue_ids", [])),
        "related_proposal_ids": _plain_list(getattr(patch, "related_proposal_ids", [])),
    }


def _pr_patches(world: Any, pr: Any, agent_id: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    pairs, availability = _pr_visible_patches(world, pr, agent_id)
    return {"patches": [_patch_metadata(patch, artifact) for patch, artifact in pairs]}, availability


def _pr_files(world: Any, pr: Any, agent_id: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    pairs, availability = _pr_visible_patches(world, pr, agent_id)
    files = []
    for patch, artifact in pairs:
        patch_content = str(getattr(patch, "new_content", "") or "")
        artifact_content = str(getattr(artifact, "content", "") or "")
        content = patch_content or artifact_content
        source = "patch_new_content" if patch_content else "visible_artifact_snapshot"
        if not content:
            continue
        files.append({
            "patch_id": str(getattr(patch, "patch_id", "")),
            "artifact_id": str(getattr(artifact, "artifact_id", "")),
            "path": str(getattr(artifact, "linked_file_path", "") or ""),
            "source": source,
            "content": content,
            "content_hash": _content_hash(content),
        })
    return ({"files": files}, _availability(
        available=bool(files),
        reason="no_visible_pr_linked_file_content" if not files else "",
    ) | ({"unavailable_patch_count": availability["unavailable_patch_count"]}
         if availability.get("unavailable_patch_count") else {}))


def _pr_diff(world: Any, pr: Any, agent_id: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    pairs, availability = _pr_visible_patches(world, pr, agent_id)
    diffs = []
    for patch, artifact in pairs:
        diff = str(getattr(patch, "unified_diff", "") or "")
        if not diff:
            continue
        diffs.append({
            "patch_id": str(getattr(patch, "patch_id", "")),
            "artifact_id": str(getattr(artifact, "artifact_id", "")),
            "path": str(getattr(artifact, "linked_file_path", "") or ""),
            "unified_diff": diff,
            "content_hash": _content_hash(diff),
        })
    return ({"diffs": diffs, "recording_note": "Only stored unified_diff values are shown; pseudo_diff is not exact evidence."},
            _availability(
                available=bool(diffs),
                reason="no_domain_recorded_unified_diff" if not diffs else "",
            ) | ({"unavailable_patch_count": availability["unavailable_patch_count"]}
                 if availability.get("unavailable_patch_count") else {}))


def _pr_resource(world: Any, agent_id: str, resource_id: str, section: str,
                 view: Dict[str, Any]) -> Dict[str, Any]:
    pr = _visible_pr(world, agent_id, resource_id)
    if pr is None:
        return _unavailable("pull_request", resource_id)
    section = section or "overview"
    overview = _pr_overview(pr, view)
    availability: Dict[str, Dict[str, Any]] = {
        key: _availability(available=True) for key in _PR_SECTIONS
    }
    if section not in _PR_SECTIONS:
        availability[section] = _availability(available=False, reason="unsupported_pr_section")
        content: Dict[str, Any] = {}
    elif section == "overview":
        content = overview
    elif section == "review":
        content = _pr_review(pr)
    elif section == "ci":
        content, availability["ci"] = _pr_ci(world, pr)
    elif section == "commits":
        content, availability["commits"] = _pr_commits(world, pr)
    elif section == "patches":
        content, availability["patches"] = _pr_patches(world, pr, agent_id)
    elif section == "files":
        content, availability["files"] = _pr_files(world, pr, agent_id)
    elif section == "diff":
        content, availability["diff"] = _pr_diff(world, pr, agent_id)
    else:  # provenance
        content = {
            "parent": {"kind": "pull_request", "id": str(pr.pr_id)},
            "commit_ids": _plain_list(getattr(pr, "commit_ids", [])),
            "patch_ids": _plain_list(getattr(pr, "patch_ids", [])),
            "ci_run_ids": _plain_list(getattr(pr, "ci_run_ids", [])),
            "ci_base_main_commit_ids": _plain_list(getattr(pr, "ci_base_main_commit_ids", None)),
            "recording_note": "Child records are revalidated against this PR on every read.",
        }
    return {
        "resource": {"kind": "pull_request", "id": str(pr.pr_id),
                     "title": overview["title"], "status": overview["status"]},
        "section": section,
        "availability": availability,
        "content": content,
        "links": [
            {"kind": "commit", "id": str(item)} for item in getattr(pr, "commit_ids", [])
        ] + [
            {"kind": "patch", "id": str(item)} for item in getattr(pr, "patch_ids", [])
        ] + [
            {"kind": "ci_run", "id": str(item)} for item in getattr(pr, "ci_run_ids", [])
        ],
        "provenance": {
            "source": "domain_pull_request_record",
            "seat_visibility": "author_or_reviewer",
            "view_version": view.get("version"),
            "parent_chain_revalidated": True,
        },
    }


def _task_related_prs(world: Any, agent_id: str, task_id: str) -> List[Any]:
    repo = world.repo_system.repo
    return [
        pr for pr in repo.pull_requests.values()
        if vis.pull_request_visible_to(pr, agent_id)
        and task_id in {
            str(getattr(pr, "linked_task", "") or ""),
            *[str(item) for item in _plain_list(getattr(pr, "linked_task_ids", []))],
        }
    ]


def _task_related_rows(view: Dict[str, Any], task: Any) -> List[Dict[str, Any]]:
    linked_ids = {
        *[str(item) for item in _plain_list(getattr(task, "linked_docs", []))],
        *[str(item) for item in _plain_list(getattr(task, "linked_issues", []))],
        *[str(item) for item in _plain_list(getattr(task, "linked_experiments", []))],
    }
    return [
        copy.deepcopy(row)
        for rows in (view.get("objects") or {}).values()
        for row in rows or []
        if str(row.get("id") or "") in linked_ids
    ]


def _task_related_events(view: Dict[str, Any], task_id: str) -> List[Dict[str, Any]]:
    return [
        copy.deepcopy(event)
        for event in ((view.get("feed") or {}).get("events") or [])
        if str(event.get("task_id") or "") == task_id
        or task_id in [str(item) for item in _plain_list(event.get("related_task_ids"))]
    ]


def _task_visible_artifacts(world: Any, task: Any, agent_id: str) -> List[Any]:
    artifacts = getattr(world, "product_artifacts", {}) or {}
    return [
        artifacts[artifact_id]
        for artifact_id in _plain_list(getattr(task, "linked_artifacts", []))
        if artifact_id in artifacts and _artifact_visible(world, artifacts[artifact_id], agent_id)
    ]


def _task_files(world: Any, task: Any, agent_id: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    files = []
    for artifact in _task_visible_artifacts(world, task, agent_id):
        working = str(getattr(artifact, "content", "") or "")
        mainline = str(getattr(artifact, "mainline_content", "") or "")
        content = working or mainline
        if not content:
            continue
        files.append({
            "artifact_id": str(getattr(artifact, "artifact_id", "")),
            "path": str(getattr(artifact, "linked_file_path", "") or ""),
            "title": str(getattr(artifact, "title", "") or ""),
            "status": str(getattr(artifact, "status", "") or ""),
            "revision": getattr(artifact, "revision", None),
            "source": "working_artifact" if working else "mainline_artifact",
            "content": content,
            "content_hash": _content_hash(content),
        })
    return {"files": files}, _availability(
        available=bool(files),
        reason="no_visible_task_linked_file_content" if not files else "",
    )


def _task_diffs(world: Any, task: Any, agent_id: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    task_id = str(getattr(task, "task_id", ""))
    linked_artifact_ids = {
        str(item) for item in _plain_list(getattr(task, "linked_artifacts", []))
    }
    visible_artifact_ids = {
        str(getattr(artifact, "artifact_id", ""))
        for artifact in _task_visible_artifacts(world, task, agent_id)
    }
    diffs = []
    for patch in (getattr(world, "patches", {}) or {}).values():
        artifact_id = str(getattr(patch, "target_object_id", "") or "")
        related_task_ids = {
            str(item) for item in _plain_list(getattr(patch, "related_task_ids", []))
        }
        if artifact_id not in visible_artifact_ids:
            continue
        if task_id not in related_task_ids and artifact_id not in linked_artifact_ids:
            continue
        unified_diff = str(getattr(patch, "unified_diff", "") or "")
        if not unified_diff:
            continue
        artifact = (getattr(world, "product_artifacts", {}) or {}).get(artifact_id)
        diffs.append({
            "patch_id": str(getattr(patch, "patch_id", "")),
            "artifact_id": artifact_id,
            "path": str(getattr(artifact, "linked_file_path", "") or ""),
            "unified_diff": unified_diff,
            "content_hash": _content_hash(unified_diff),
        })
    return {
        "diffs": diffs,
        "recording_note": "Only stored unified_diff values are shown; pseudo_diff is not exact evidence.",
    }, _availability(
        available=bool(diffs),
        reason="no_domain_recorded_task_unified_diff" if not diffs else "",
    )


def _task_review(prs: List[Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    comments = []
    for pr in prs:
        overview = {"id": str(pr.pr_id), "title": str(getattr(pr, "title", "") or pr.pr_id)}
        for comment in _pr_review(pr)["comments"]:
            comments.append({**overview, **comment})
    return {
        "reviewed": any(bool(getattr(pr, "reviewed", False)) for pr in prs),
        "comments": comments,
        "pull_requests": [str(pr.pr_id) for pr in prs],
        "approved_by": {
            str(pr.pr_id): _plain_list(getattr(pr, "approved_by", [])) for pr in prs
        },
        "requested_changes": {
            str(pr.pr_id): _plain_list(getattr(pr, "requested_changes", [])) for pr in prs
        },
        "recording_note": (
            "Review records are aggregated from the task's seat-visible pull requests."
            if prs else "No seat-visible pull request is linked to this task."
        ),
    }, _availability(
        available=bool(prs),
        reason="no_visible_task_linked_pull_request" if not prs else "",
    )


def _task_ci(world: Any, prs: List[Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    runs = []
    for pr in prs:
        ci, _ = _pr_ci(world, pr)
        runs.extend({**run, "pull_request_id": str(pr.pr_id)} for run in ci["runs"])
    return {
        "runs": runs,
        "pull_requests": [str(pr.pr_id) for pr in prs],
    }, _availability(
        available=bool(runs),
        reason="no_visible_task_linked_ci_run" if not runs else "",
    )


def _task_resource(world: Any, agent_id: str, resource_id: str, section: str,
                   view: Dict[str, Any]) -> Dict[str, Any]:
    row = _visible_row(view, "task", resource_id)
    task = (getattr(world, "tasks", {}) or {}).get(resource_id)
    if row is None or task is None:
        return _unavailable("task", resource_id)

    section = section or "overview"
    prs = _task_related_prs(world, agent_id, resource_id)
    files, files_available = _task_files(world, task, agent_id)
    diffs, diffs_available = _task_diffs(world, task, agent_id)
    review, review_available = _task_review(prs)
    ci, ci_available = _task_ci(world, prs)
    related_rows = _task_related_rows(view, task)
    related_events = _task_related_events(view, resource_id)
    overview = {
        "task": copy.deepcopy(row),
        "related_pull_requests": [_pr_overview(pr, view) for pr in prs],
        "related_objects": related_rows,
        "related_events": related_events,
        "linked_files": [
            {key: item.get(key) for key in ("artifact_id", "path", "title", "status", "revision", "source")}
            for item in files["files"]
        ],
    }
    availability = {
        "overview": _availability(available=True),
        "review": review_available,
        "ci": ci_available,
        "files": files_available,
        "diff": diffs_available,
        "provenance": _availability(available=True),
    }
    if section not in _TASK_SECTIONS:
        availability[section] = _availability(available=False, reason="unsupported_task_section")
        content: Dict[str, Any] = {}
    elif section == "overview":
        content = overview
    elif section == "review":
        content = review
    elif section == "ci":
        content = ci
    elif section == "files":
        content = files
    elif section == "diff":
        content = diffs
    else:
        content = {
            "parent": {"kind": "task", "id": resource_id},
            "related_pull_request_ids": [str(pr.pr_id) for pr in prs],
            "linked_artifact_ids": _plain_list(getattr(task, "linked_artifacts", [])),
            "related_events": related_events,
            "recording_note": "Relations are rebuilt from the current seat-visible world snapshot.",
        }
    artifact_links = [
        {
            "kind": "artifact",
            "id": str(getattr(artifact, "artifact_id", "")),
            "title": str(getattr(artifact, "title", "") or ""),
        }
        for artifact in _task_visible_artifacts(world, task, agent_id)
    ]
    return {
        "resource": {
            "kind": "task", "id": resource_id,
            "title": str(row.get("title") or resource_id),
            "status": str(row.get("status") or ""),
        },
        "section": section,
        "availability": availability,
        "content": content,
        "links": [
            {"kind": "pull_request", "id": str(pr.pr_id),
             "title": str(getattr(pr, "title", "") or pr.pr_id)}
            for pr in prs
        ] + [
            {"kind": str(item.get("kind") or "object"),
             "id": str(item.get("id") or ""),
             "title": str(item.get("title") or item.get("id") or "")}
            for item in related_rows
        ] + artifact_links,
        "provenance": {
            "source": "domain_task_relations",
            "view_version": view.get("version"),
            "seat_visibility_revalidated": True,
        },
    }
def build_resource(world: Any, agent_id: str, resource_kind: str, resource_id: str,
                   *, section: str = "overview", view: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Build one current seat-scoped detail payload.

    ``view`` must be a freshly built filtered P2 view from the same locked world
    snapshot.  Generic resources expose that existing allowlisted projection;
    PR and task resources have audited rich adapters; other kinds expose only
    their current allowlisted projection.
    """
    if not resource_kind or not resource_id:
        return _unavailable(resource_kind, resource_id, "missing_resource_identity")
    if view is None:
        # Keeping this lazy import avoids a module cycle with ``seat_view``.
        from environments.org_env.human.seat_view import build_seat_view
        view = build_seat_view(world, agent_id)
    if resource_kind == "pull_request":
        return _pr_resource(world, agent_id, resource_id, section, view)
    if resource_kind == "task":
        return _task_resource(world, agent_id, resource_id, section, view)
    if resource_kind == "proposal":
        return _proposal_resource(world, resource_id, section, view)
    return _generic_resource(view, resource_kind, resource_id, section)


__all__ = ["RESOURCE_SECTIONS_BY_KIND", "build_resource"]
