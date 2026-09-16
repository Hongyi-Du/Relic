"""The organization as one member sees it.

This is the read side of a human seat: everything the three panes of the HCI
need, and nothing the member has no right to. It is built by naming fields
explicitly — never by dumping objects — because the omniscient inspector
snapshot (``org_lived_full_snapshot``) exists right next door and would happily
serialise every other member's private workspace if it were reused here.

Time is reported in wall-clock terms. The world still runs on ticks internally
(1 tick = 1 organizational hour), but a human is never shown one: objects carry
``age_hours`` and, when the runtime supplies a mapping, an absolute timestamp.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from environments.org_env.human import visibility as vis

#: How much history a seat gets in one view. The feed is a working surface, not
#: an archive; the object drawers page separately.
FEED_EVENT_LIMIT = 120
FEED_MESSAGE_LIMIT = 200


def _enum(value: Any) -> str:
    return getattr(value, "value", str(value)) if value is not None else ""


def _age_hours(world: Any, tick: Any) -> Optional[int]:
    """Organizational hours since something happened (1 tick = 1 hour)."""
    try:
        return max(0, int(world.world_tick) - int(tick))
    except (TypeError, ValueError):
        return None


def _stamp(world: Any, tick: Any, wall_clock: Optional[Callable[[int], float]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"age_hours": _age_hours(world, tick)}
    if wall_clock is not None:
        try:
            out["at"] = wall_clock(int(tick))
        except (TypeError, ValueError):
            pass
    return out


# --------------------------------------------------------------------------- #
# Objects
# --------------------------------------------------------------------------- #
def _task(world, t, wall_clock) -> Dict[str, Any]:
    return {"id": t.task_id, "kind": "task", "title": t.title,
            "status": _enum(t.status), "priority": getattr(t, "priority", 3),
            "owner": t.owner_id,
            "description": getattr(t, "description", "") or "",
            "deadline_tick": getattr(t, "deadline_tick", None),
            "progress": getattr(t, "progress_score", 0.0),
            "linked": {"docs": list(getattr(t, "linked_docs", []) or []),
                       "issues": list(getattr(t, "linked_issues", []) or []),
                       "experiments": list(getattr(t, "linked_experiments", []) or []),
                       "artifacts": list(getattr(t, "linked_artifacts", []) or [])},
            **_stamp(world, getattr(t, "created_tick", 0), wall_clock)}


def _document(world, d, wall_clock) -> Dict[str, Any]:
    return {"id": getattr(d, "doc_id", ""), "kind": "document", "title": d.title,
            "doc_type": getattr(d, "doc_type", "doc"),
            "owner": getattr(d, "owner_id", None) or getattr(d, "author_id", None),
            "version": getattr(d, "version", 1),
            "linked_tasks": list(getattr(d, "linked_tasks", []) or []),
            **_stamp(world, getattr(d, "created_tick", 0), wall_clock)}


def _pull_request(world, pr, agent_id, wall_clock) -> Dict[str, Any]:
    return {"id": pr.pr_id, "kind": "pull_request", "title": getattr(pr, "title", "") or pr.pr_id,
            "author": pr.author_id, "status": _enum(pr.status),
            "reviewers": list(pr.reviewers), "approved_by": list(getattr(pr, "approved_by", []) or []),
            "reviewed": getattr(pr, "reviewed", False),
            "ci_passed": getattr(pr, "ci_passed", None),
            "source_branch": getattr(pr, "source_branch", ""),
            "awaiting_my_review": (agent_id in pr.reviewers
                                   and _enum(pr.status) in ("open", "review_requested",
                                                            "changes_requested")),
            # PullRequest records its actual creation time as ``opened_tick``;
            # it has no ``created_tick`` field.
            **_stamp(world, getattr(pr, "opened_tick", 0), wall_clock)}


def _experiment(world, e, wall_clock) -> Dict[str, Any]:
    return {"id": getattr(e, "experiment_id", ""), "kind": "experiment",
            "title": getattr(e, "title", ""), "status": _enum(getattr(e, "status", "")),
            "reproducibility": getattr(e, "reproducibility_status", ""),
            **_stamp(world, getattr(e, "created_tick", 0), wall_clock)}


def _meeting(world, m, agent_id, wall_clock) -> Dict[str, Any]:
    meeting_system = world.meeting_system
    note = meeting_system.notes.get(getattr(m, "notes_doc_id", None))
    decisions = [meeting_system.decisions[did] for did in (getattr(m, "decision_ids", []) or [])
                 if did in meeting_system.decisions]
    action_items = [meeting_system.action_items[aid]
                    for aid in (getattr(m, "action_item_ids", []) or [])
                    if aid in meeting_system.action_items]
    return {"id": m.meeting_id, "kind": "meeting", "title": getattr(m, "title", ""),
            "meeting_type": getattr(m, "meeting_type", ""), "status": _enum(m.status),
            "participants": sorted(m.participants),
            "attended": agent_id in getattr(m, "attendees", ()),
            "declined": agent_id in getattr(m, "skipped_by", ()),
            "scheduled_tick": getattr(m, "scheduled_tick", None),
            "agenda": list(getattr(m, "agenda", []) or []),
            "room_channel": getattr(m, "room_channel_id", None),
            "notes_summary": getattr(note, "summary", "") if note is not None else "",
            "unresolved_questions": list(getattr(note, "unresolved_questions", []) or [])
            if note is not None else [],
            "decision_summaries": [getattr(decision, "decision_summary", "")
                                   for decision in decisions],
            "action_items": [{
                "description": getattr(item, "description", ""),
                "assignee_id": getattr(item, "assignee_id", None),
                "due_tick": getattr(item, "due_tick", None),
                "status": getattr(item, "status", "open"),
                "linked_task_id": getattr(item, "linked_task_id", None),
            } for item in action_items],
            **_stamp(world, getattr(m, "scheduled_tick", 0), wall_clock)}


def _proposal(world, p, agent_id, wall_clock) -> Dict[str, Any]:
    approvers = list(getattr(p, "approval_required_from", []) or [])
    return {"id": p.proposal_id, "kind": "proposal",
            "title": getattr(p, "title", "") or getattr(p, "summary", ""),
            "proposal_type": getattr(p, "proposal_type", ""),
            "status": _enum(getattr(p, "status", "")),
            "author": (getattr(p, "proposer_agent_id", None)
                       or getattr(p, "author_id", None)
                       or getattr(p, "proposer_id", None)),
            "approvers": approvers,
            "approved_by": list(getattr(p, "approved_by", []) or []),
            "awaiting_my_approval": (agent_id in approvers
                                     and agent_id not in (getattr(p, "approved_by", []) or [])
                                     and _enum(getattr(p, "status", "")) == "under_review"),
            **_stamp(world, getattr(p, "created_at_tick", 0), wall_clock)}


def _protocol(world, p, wall_clock) -> Dict[str, Any]:
    return {"id": getattr(p, "protocol_id", ""), "kind": "protocol",
            "title": getattr(p, "protocol_type", ""),
            "protocol_type": getattr(p, "protocol_type", ""),
            "status": getattr(p, "adoption_status", ""),
            "supporters": list(getattr(p, "supporters", []) or []),
            "opposers": list(getattr(p, "opposers", []) or []),
            "emergence": getattr(p, "emergence_level", "none"),
            "use_count": len(getattr(p, "usage_events", []) or []),
            "violation_count": len(getattr(p, "violation_events", []) or []),
            **_stamp(world, getattr(p, "created_at_tick", 0), wall_clock)}


def _issue(world, issue, wall_clock) -> Dict[str, Any]:
    return {"id": getattr(issue, "issue_id", ""), "kind": "issue",
            "title": getattr(issue, "title", ""),
            "description": getattr(issue, "description", "") or "",
            "status": _enum(getattr(issue, "status", "")),
            "severity": getattr(issue, "severity", ""),
            "owner": getattr(issue, "owner_id", None),
            "related_task": getattr(issue, "related_task_id", None),
            **_stamp(world, getattr(issue, "created_tick", 0), wall_clock)}


def _result(world, result, wall_clock) -> Dict[str, Any]:
    return {"id": getattr(result, "result_id", ""), "kind": "result",
            "title": f"Result {getattr(result, 'result_id', '')}",
            "experiment_id": getattr(result, "experiment_id", ""),
            "reproducibility": getattr(result, "reproducibility_status", ""),
            "confidence": getattr(result, "confidence", None),
            "metrics": dict(getattr(result, "metrics", {}) or {}),
            "logged_to_tracker": bool(getattr(result, "logged_to_tracker", False))}


def _release_candidate(world, candidate, wall_clock) -> Dict[str, Any]:
    return {"id": getattr(candidate, "candidate_id", ""), "kind": "release_candidate",
            "title": f"Release candidate {getattr(candidate, 'version', '')}",
            "version": getattr(candidate, "version", ""),
            "status": _enum(getattr(candidate, "status", "")),
            "created_by": getattr(candidate, "created_by", None),
            "blockers": list(getattr(candidate, "blockers", []) or []),
            "approvals": list(getattr(candidate, "approvals", []) or []),
            **_stamp(world, getattr(candidate, "created_at_tick", 0), wall_clock)}


def _message(world, m, agent_id, wall_clock) -> Dict[str, Any]:
    return {"id": m.message_id, "kind": "message", "sender": m.sender_id,
            "channel": m.channel_id, "thread_id": m.thread_id or m.message_id,
            "recipients": list(getattr(m, "recipients", []) or []),
            "dm_id": getattr(m, "dm_id", None),
            "reply_to": m.reply_to_message_id,
            "text": m.full_text or m.text_summary,
            "mentions": list(m.mentions), "attachments": [
                {"object_id": a.object_id, "type": a.attachment_type, "title": a.title}
                for a in m.attachments],
            "importance": m.importance, "urgency": m.urgency,
            "unread": agent_id not in m.read_by,
            "mentions_me": agent_id in m.mentions,
            **_stamp(world, m.created_tick, wall_clock)}


# --------------------------------------------------------------------------- #
# The view
# --------------------------------------------------------------------------- #
def build_seat_view(world: Any, agent_id: str, *,
                    wall_clock: Optional[Callable[[int], float]] = None,
                    seat_presence: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Everything the member ``agent_id`` may see, shaped for the three panes.

    ``wall_clock`` maps a tick to a POSIX timestamp when a live runtime is
    driving the world; without it objects still carry ``age_hours``.
    ``seat_presence`` is the online/offline map for members, which says nothing
    about who controls them.
    """
    if agent_id not in world.agents:
        raise KeyError(f"unknown_member:{agent_id}")
    agent = world.agents[agent_id]

    tasks = [_task(world, t, wall_clock) for t in world.tasks.values()
             if vis.task_visible_to(t, agent_id)]
    docs = [_document(world, d, wall_clock) for d in world.documents.values()
            if vis.document_visible_to(d, agent_id)]
    repo = world.repo_system.repo
    prs = [_pull_request(world, pr, agent_id, wall_clock) for pr in repo.pull_requests.values()
           if vis.pull_request_visible_to(pr, agent_id)]
    experiments = [_experiment(world, e, wall_clock) for e in world.experiments.values()
                   if vis.experiment_visible_to(e, agent_id)]
    meetings = [_meeting(world, m, agent_id, wall_clock)
                for m in world.meeting_system.meetings.values()
                if vis.meeting_visible_to(m, agent_id)]
    protocols = [_protocol(world, p, wall_clock)
                 for p in world.protocol_registry.protocols.values()]
    issues = [_issue(world, issue, wall_clock)
              for issue in getattr(world, "issues", {}).values()
              if vis.issue_visible_to(issue, agent_id)]
    results = [_result(world, result, wall_clock)
               for result in vis.visible_results(world, agent_id)]
    pm = getattr(world, "proposal_manager", None)
    proposals = [_proposal(world, p, agent_id, wall_clock)
                 for p in (getattr(pm, "proposals", {}) or {}).values()] if pm else []
    messages = [_message(world, m, agent_id, wall_clock)
                for m in vis.visible_messages(world, agent_id)]
    messages.sort(key=lambda m: (m["age_hours"] is None, -(m["age_hours"] or 0)))
    messages = messages[-FEED_MESSAGE_LIMIT:]

    branches = [{"id": b.branch_id, "kind": "branch", "status": _enum(b.status),
                 "commits": len(b.commit_ids), "uncommitted": b.uncommitted_changes,
                 "linked_task": b.linked_task}
                for b in repo.branches.values() if vis.branch_visible_to(b, agent_id)]
    releases = [{"id": r.release_id, "kind": "release", "version": getattr(r, "version", ""),
                 "released_by": getattr(r, "released_by", None),
                 "known_limitations": list(getattr(r, "known_limitations", []) or []),
                 **_stamp(world, getattr(r, "released_tick", 0), wall_clock)}
                for r in getattr(repo, "releases", {}).values()]
    release_candidates = [_release_candidate(world, candidate, wall_clock)
                          for candidate in getattr(repo, "release_candidates", {}).values()
                          if vis.release_visible_to(candidate, agent_id)]
    decisions = _decisions(world, agent_id, wall_clock)

    my_tasks = [t for t in tasks if t["owner"] == agent_id]
    events = [e for e in world.events[-FEED_EVENT_LIMIT * 3:]
              if vis.event_visible_to(e, agent_id)][-FEED_EVENT_LIMIT:]

    view = {
        "seat": {
            "agent_id": agent_id, "name": agent.name, "codename": agent.codename,
            "role": agent.role, "is_founder": agent.is_founder,
            "identity": agent.initial_identity,
            "permissions": list(agent.permissions),
            "status": agent.current_status,
        },
        "clock": _clock_block(world, wall_clock),
        "member": {
            "my_tasks": my_tasks,
            "blocked_tasks": [t for t in my_tasks if t["status"] == "blocked"],
            "awaiting_me": {
                "mentions": [m for m in messages if m["mentions_me"] and m["unread"]],
                "reviews": [p for p in prs if p["awaiting_my_review"]],
                "proposals": [p for p in proposals if p["awaiting_my_approval"]],
                "meetings": [m for m in meetings if m["status"] in ("scheduled", "active")
                             and not m["attended"]],
            },
            "unread_count": sum(1 for m in messages if m["unread"]),
            "my_work": {"branches": branches,
                        "experiments": experiments,
                        "docs": [d for d in docs if d["owner"] == agent_id]},
            "company": _company_block(world, agent),
            "members": _roster(world, seat_presence),
        },
        "feed": {
            "channels": [{"id": ch.channel_id, "type": ch.channel_type,
                          "members": sorted(ch.members)}
                         for ch in vis.visible_channels(world, agent_id)],
            "threads": _threads(messages),
            "events": events,
        },
        "objects": {
            "tasks": tasks, "documents": docs, "pull_requests": prs,
            "experiments": experiments, "meetings": meetings,
            "proposals": proposals, "protocols": protocols,
            "issues": issues, "results": results,
            "release_candidates": release_candidates, "releases": releases,
            "decisions": decisions, "branches": branches, "messages": messages,
        },
    }
    # The brief must be built *after* the fully filtered view exists.  It is a
    # presentation projection, never an additional read from the world.
    from environments.org_env.human.organization_brief import build_organization_brief
    view["organization_brief"] = build_organization_brief(view)
    return view


def build_seat_resource(world: Any, agent_id: str, resource_kind: str,
                        resource_id: str, *, section: str = "overview",
                        wall_clock: Optional[Callable[[int], float]] = None,
                        seat_presence: Optional[Dict[str, Any]] = None,
                        view_version: Optional[int] = None) -> Dict[str, Any]:
    """Build one lazy read-only detail from the *current* P2 visibility seam.

    The caller must hold the runtime world lock while invoking this function.
    It intentionally rebuilds the filtered view before resolving an object, so
    a stale browser reference cannot use this read path to inspect an object
    which has since become invisible.
    """
    view = build_seat_view(world, agent_id, wall_clock=wall_clock,
                           seat_presence=seat_presence)
    if view_version is not None:
        view["version"] = view_version
    # Local import keeps the normal P2 projection independent from the rich
    # inspector adapter and avoids a seat_view <-> resource_inspector cycle.
    from environments.org_env.human.resource_inspector import build_resource
    return build_resource(world, agent_id, resource_kind, resource_id,
                          section=section, view=view)


def _clock_block(world, wall_clock) -> Dict[str, Any]:
    """Wall-clock only. A human seat never sees the tick counter (HCI V0 §5)."""
    out: Dict[str, Any] = {}
    if wall_clock is not None:
        out["now"] = wall_clock(int(world.world_tick))
    return out


def _company_block(world, agent) -> Dict[str, Any]:
    """The same budget split perception applies: everyone feels the pressure,
    founders see the balance. Project-workspace HCI has no investor economy,
    so it exposes product identity without inventing runway or cash settings."""
    b = world.budget_system.budget
    config = getattr(world, "company_config", {}) or {}
    product = getattr(world, "product", None)
    out = {"product_stage": getattr(product, "milestone_stage", ""),
           "product_name": str(config.get("product_name")
                                or getattr(product, "name", "") or "Organization"),
           "company_name": str(getattr(getattr(world, "company", None),
                                        "company_name", "") or "")}
    if getattr(world, "funding_simulation_enabled", True):
        out.update({"budget_pressure": b.budget_pressure,
                    "runway_days": b.runway_days})
        if agent.is_founder:
            out["cash_balance"] = round(b.cash_balance, 1)
    return out


def _roster(world, seat_presence) -> List[Dict[str, Any]]:
    """Who is in the organization. Deliberately identical in shape for every
    member: presence is shown, controller is not (HCI V0 §6)."""
    presence = seat_presence or {}
    return [{"agent_id": aid, "name": a.name, "role": a.role,
             "codename": a.codename,
             "online": bool(presence.get(aid, {}).get("online", True))}
            for aid, a in world.agents.items()]


def _threads(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Group the flat message list into threads, newest activity first."""
    by_thread: Dict[str, List[Dict[str, Any]]] = {}
    for m in messages:
        by_thread.setdefault(m["thread_id"], []).append(m)
    threads = []
    for thread_id, msgs in by_thread.items():
        threads.append({"thread_id": thread_id, "channel": msgs[0]["channel"],
                        "messages": msgs, "reply_count": len(msgs) - 1,
                        "unread": any(m["unread"] for m in msgs),
                        "age_hours": min((m["age_hours"] for m in msgs
                                          if m["age_hours"] is not None), default=None)})
    threads.sort(key=lambda t: (t["age_hours"] is None, t["age_hours"] or 0))
    return threads


def _decisions(world, agent_id, wall_clock) -> List[Dict[str, Any]]:
    """Meeting decisions. Reachable only through a meeting the member attended,
    which is where their visibility comes from."""
    out = []
    ms = world.meeting_system
    for m in ms.meetings.values():
        if not vis.meeting_visible_to(m, agent_id):
            continue
        for did in getattr(m, "decision_ids", []) or []:
            d = getattr(ms, "decisions", {}).get(did)
            if d is None:
                continue
            out.append({"id": did, "kind": "decision", "meeting_id": m.meeting_id,
                        "summary": getattr(d, "decision_summary", ""),
                        "decided_by": list(getattr(d, "decided_by", []) or []),
                        "dissenting": list(getattr(d, "dissenting_agents", []) or []),
                        **_stamp(world, getattr(d, "created_tick", 0), wall_clock)})
    return out


__all__ = ["build_seat_view", "build_seat_resource", "FEED_EVENT_LIMIT", "FEED_MESSAGE_LIMIT"]
