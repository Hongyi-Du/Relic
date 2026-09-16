"""Human-readable, provenance-bearing organization briefs.

This module deliberately accepts a completed seat view rather than a world.
The inspector has a much richer, omniscient snapshot nearby, but a liaison
must be able to explain only what the seat is already allowed to see.  Building
the brief as a projection of ``build_seat_view`` makes that boundary structural
instead of relying on each summary branch to remember it.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List


_DONE = {"done", "completed", "closed", "merged", "released", "cancelled"}
_BLOCKED = {"blocked", "changes_requested", "failed"}


def _ref(row: Dict[str, Any]) -> Dict[str, str]:
    """A stable, UI-resolvable pointer to one already-visible object."""
    return {"kind": str(row.get("kind", "")), "id": str(row.get("id", "")),
            "title": str(row.get("title", "") or row.get("id", ""))}


def _join_refs(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, str]]:
    seen = set()
    refs = []
    for row in rows:
        ref = _ref(row)
        key = (ref["kind"], ref["id"])
        if ref["id"] and key not in seen:
            refs.append(ref)
            seen.add(key)
    return refs


def _task_item(task: Dict[str, Any]) -> Dict[str, Any]:
    status = str(task.get("status", ""))
    if status == "blocked":
        why = "This work is blocked."
    elif status == "review_pending":
        why = "The work is waiting for review."
    elif status == "in_progress":
        why = "The owner is actively working on it."
    else:
        why = "This is an active item on the organization board."
    if task.get("description"):
        why += " " + str(task["description"])
    return {"id": task.get("id"), "kind": task.get("kind", "task"),
            "title": task.get("title") or task.get("id"),
            "what": task.get("title") or task.get("id"), "status": status,
            "why": why, "owner": task.get("owner"), "refs": [_ref(task)]}


def _pr_item(pr: Dict[str, Any]) -> Dict[str, Any]:
    status = str(pr.get("status", ""))
    if pr.get("ci_passed") is False:
        why = "The pull request's visible CI result is failing."
    elif status == "changes_requested":
        why = "Changes were requested during review."
    elif status in {"open", "review_requested"}:
        why = "The pull request is awaiting the visible review and CI workflow."
    else:
        why = "This is a visible pull request in the current workflow."
    return {"id": pr.get("id"), "kind": pr.get("kind", "pull_request"),
            "title": pr.get("title") or pr.get("id"),
            "what": pr.get("title") or pr.get("id"), "status": status,
            "why": why, "owner": pr.get("author"), "refs": [_ref(pr)]}


def _experiment_item(experiment: Dict[str, Any]) -> Dict[str, Any]:
    return {"id": experiment.get("id"), "kind": experiment.get("kind", "experiment"),
            "title": experiment.get("title") or experiment.get("id"),
            "what": experiment.get("title") or experiment.get("id"),
            "status": str(experiment.get("status", "")),
            "why": "This visible experiment is still active.", "owner": None,
            "refs": [_ref(experiment)]}


def _protocol_item(protocol: Dict[str, Any]) -> Dict[str, Any]:
    status = str(protocol.get("status", ""))
    return {"id": protocol.get("id"), "kind": protocol.get("kind", "protocol"),
            "title": protocol.get("title") or protocol.get("id"),
            "what": protocol.get("title") or protocol.get("id"), "status": status,
            "why": f"This organization-wide protocol is currently {status or 'visible'}.",
            "owner": None, "refs": [_ref(protocol)]}


def build_organization_brief(seat_view: Dict[str, Any], *, limit: int = 5) -> Dict[str, Any]:
    """Summarize *only* an already-filtered seat view.

    The result intentionally contains structured ``refs`` for every claim so a
    caller can open the supporting task, PR, proposal, or protocol.  It does
    not make recommendations, mutate state, assign work, or reinterpret a
    person's pending approval as a decision made for them.
    """
    objects = seat_view.get("objects") or {}
    tasks = list(objects.get("tasks") or [])
    prs = list(objects.get("pull_requests") or [])
    experiments = list(objects.get("experiments") or [])
    protocols = list(objects.get("protocols") or [])

    happening: List[Dict[str, Any]] = []
    # Put visible blockers first, then the highest-priority active work.  This
    # is presentation order only; it never changes the world's priority policy.
    blocked_tasks = [t for t in tasks if str(t.get("status", "")) == "blocked"]
    blocked_prs = [p for p in prs if str(p.get("status", "")) in _BLOCKED
                   or p.get("ci_passed") is False]
    happening.extend(_task_item(t) for t in blocked_tasks)
    happening.extend(_pr_item(p) for p in blocked_prs)

    active_tasks = [t for t in tasks if str(t.get("status", "")) not in _DONE
                    and t not in blocked_tasks]
    active_tasks.sort(key=lambda t: (-int(t.get("priority", 0) or 0),
                                     str(t.get("id", ""))))
    happening.extend(_task_item(t) for t in active_tasks)
    happening.extend(_pr_item(p) for p in prs
                     if str(p.get("status", "")) not in _DONE and p not in blocked_prs)
    happening.extend(_experiment_item(e) for e in experiments
                     if str(e.get("status", "")) not in _DONE)
    happening.extend(_protocol_item(p) for p in protocols
                     if str(p.get("status", "")) not in _DONE)

    # A row can surface through more than one category above; the public ref is
    # its identity, so use it to make a concise, non-duplicative brief.
    unique_happening = []
    seen_refs = set()
    for item in happening:
        ref = item["refs"][0]
        key = (ref["kind"], ref["id"])
        if key not in seen_refs:
            unique_happening.append(item)
            seen_refs.add(key)
        if len(unique_happening) >= max(1, limit):
            break

    awaiting = (seat_view.get("member") or {}).get("awaiting_me") or {}
    decisions: List[Dict[str, Any]] = []
    for proposal in awaiting.get("proposals") or []:
        decisions.append({"id": proposal.get("id"), "kind": proposal.get("kind", "proposal"),
                          "title": proposal.get("title") or proposal.get("id"),
                          "what": f"Decide proposal: {proposal.get('title') or proposal.get('id')}",
                          "why": "You are a designated approver; the liaison cannot approve it for you.",
                          "owner": proposal.get("author"), "refs": [_ref(proposal)],
                          "decision_type": "proposal_approval"})
    for pr in awaiting.get("reviews") or []:
        decisions.append({"id": pr.get("id"), "kind": pr.get("kind", "pull_request"),
                          "title": pr.get("title") or pr.get("id"),
                          "what": f"Review pull request: {pr.get('title') or pr.get('id')}",
                          "why": "This pull request is awaiting your review; the liaison cannot review it for you.",
                          "owner": pr.get("author"), "refs": [_ref(pr)],
                          "decision_type": "pull_request_review"})
    for meeting in awaiting.get("meetings") or []:
        decisions.append({"id": meeting.get("id"), "kind": meeting.get("kind", "meeting"),
                          "title": meeting.get("title") or meeting.get("id"),
                          "what": f"Attend meeting: {meeting.get('title') or meeting.get('id')}",
                          "why": "You are an invited participant in this visible meeting.",
                          "owner": None, "refs": [_ref(meeting)],
                          "decision_type": "meeting_attendance"})

    company = (seat_view.get("member") or {}).get("company") or {}
    summary = ("The organization has visible active work" if unique_happening
               else "No active organization work is visible to this seat")
    if decisions:
        summary += f" and {len(decisions)} item(s) awaiting your decision or review"
    summary += "."
    return {
        "kind": "organization_brief",
        "title": "Organization brief",
        "visibility": "seat_visible_state",
        "summary": summary,
        "organization": {"product_name": company.get("product_name", ""),
                         "company_name": company.get("company_name", ""),
                         "product_stage": company.get("product_stage", ""),
                         "runway_days": company.get("runway_days")},
        "what_is_happening": unique_happening,
        "needs_your_decision": decisions,
        "source_refs": _join_refs(ref for item in unique_happening + decisions
                                    for ref in item["refs"]),
        # Presentation aliases keep the view usable by a deliberately thin UI;
        # they are aliases of the canonical fields above, never a second state
        # query or a policy recommendation.
        "workstreams": unique_happening,
        "blockers": [item for item in unique_happening
                     if item.get("status") in _BLOCKED],
        "pending_decisions": decisions,
        "agents": list((seat_view.get("member") or {}).get("members") or []),
    }


def render_organization_brief(brief: Dict[str, Any]) -> str:
    """Compact plain text for the liaison transcript; refs remain explicit."""
    lines = ["Organization brief", brief.get("summary", "")]
    happening = brief.get("what_is_happening") or []
    if happening:
        lines.append("What is happening:")
        for item in happening:
            owner = f" Owner: {item['owner']}." if item.get("owner") else ""
            refs = ", ".join(f"{r['kind']}:{r['id']}" for r in item.get("refs") or [])
            lines.append(f"- {item['what']} ({item.get('status') or 'active'}). "
                         f"{item['why']}{owner} Sources: {refs}.")
    decisions = brief.get("needs_your_decision") or []
    if decisions:
        lines.append("Needs your decision:")
        for item in decisions:
            refs = ", ".join(f"{r['kind']}:{r['id']}" for r in item.get("refs") or [])
            lines.append(f"- {item['what']}. {item['why']} Sources: {refs}.")
    return "\n".join(lines)


__all__ = ["build_organization_brief", "render_organization_brief"]
