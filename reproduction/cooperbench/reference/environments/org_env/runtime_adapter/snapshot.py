"""org_lived_full_snapshot — one deep, information-complete frame of an OrgWorld
for the OrgEnv Live Inspector (frontend spec §10/§11).

This is a debugger/replay contract, not an in-world view: it exposes EVERYTHING
(including private workspaces + agent internal state) but tags each object with
visibility metadata (``debug_visible`` / ``visible_to_agents`` / ``read_by`` /
``private_owner``) so the UI can also render an "agent-visible" view and debug
information asymmetry.

Pure + JSON-able + read-only (never mutates the world). Mirrors the env-agnostic
recorder pattern (agent_sdk/lived/recorder.py) but emits the org-specific shape.
"""
from __future__ import annotations

import dataclasses
import enum
from typing import Any, Dict, List, Optional

from environments.org_env.backend.clock.routine import get_routine


# --------------------------------------------------------------------------- #
# JSON-safe serialization
# --------------------------------------------------------------------------- #
def _json(obj: Any, _depth: int = 0) -> Any:
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, enum.Enum):
        return obj.value
    if isinstance(obj, (set, frozenset)):
        return sorted((_json(x, _depth + 1) for x in obj), key=lambda v: str(v))
    if isinstance(obj, (list, tuple)):
        return [_json(x, _depth + 1) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _json(v, _depth + 1) for k, v in obj.items()}
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        if _depth > 6:
            return str(obj)
        return {f.name: _json(getattr(obj, f.name), _depth + 1) for f in dataclasses.fields(obj)}
    return str(obj)


def _fields(obj: Any) -> Dict[str, Any]:
    """Dump a dataclass/instance's public attributes (JSON-safe)."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _json(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    return {k: _json(v) for k, v in vars(obj).items() if not k.startswith("_")}


# --------------------------------------------------------------------------- #
# Visibility helpers (debug shows all; tag what each agent could perceive)
# --------------------------------------------------------------------------- #
def _doc_visible_to(world, doc) -> List[str]:
    vis = getattr(doc, "visibility", "team")
    owner = getattr(doc, "owner_id", None) or getattr(doc, "author_id", None)
    if vis in ("team", "public"):
        return sorted(world.agents.keys())
    return [a for a in world.agents if a == owner]


def _message_visible_to(world, msg) -> List[str]:
    ch = world.comm.channels.get(getattr(msg, "channel_id", None))
    if ch is not None:
        return sorted(ch.members)
    return sorted(getattr(msg, "recipients", []) or [])


def _result_seen_by(world, result_id: str) -> List[str]:
    seen = set()
    r = world.sandbox_system.results.get(result_id)
    if r is None:
        return []
    # owner (whoever's sandbox cached it)
    for aid, sb in getattr(world.sandbox_system, "sandboxes", {}).items():
        if result_id in getattr(sb, "cached_result_ids", []):
            seen.add(getattr(sb, "owner_id", aid))
    if getattr(r, "logged_to_tracker", False):           # on shared tracker -> everyone
        seen.update(world.agents.keys())
    # shared in a message attachment that an agent has read
    for m in world.comm.messages.values():
        for a in getattr(m, "attachments", []) or []:
            if getattr(a, "object_id", None) == result_id:
                seen.update(getattr(m, "read_by", set()) or set())
    return sorted(seen)


# --------------------------------------------------------------------------- #
# Agents
# --------------------------------------------------------------------------- #
def _agent_snapshot(world, aid: str) -> Dict[str, Any]:
    a = world.agents[aid]
    ws = a.work_state
    comp = world.budget_system.comp.get(aid)
    av = world.time.availability.get(aid)
    rt = get_routine(aid)
    inv_unread = sum(1 for m in world.comm.perceivable_messages(aid)
                     if aid not in getattr(m, "read_by", set()))
    open_commitments = sum(1 for c in world.commitment_registry.commitments.values()
                           if c.agent_id == aid and c.status == "open")
    sandbox_jobs = sum(1 for j in world.sandbox_system.jobs.values() if j.agent_id == aid)
    return {
        "id": aid, "name": a.name, "codename": a.codename, "role": a.role,
        "is_founder": a.is_founder, "initial_identity": a.initial_identity,
        "current_status": getattr(av, "current_availability_status", a.current_status) if av else a.current_status,
        "work_state": ws.snapshot(),
        "org_state": {
            "trust_in_company": round(a.org_state.trust_in_company, 3),
            "compensation_stress": round(a.org_state.compensation_stress, 3),
            "perceived_recognition": round(a.org_state.perceived_recognition, 3),
            "role_clarity": round(a.org_state.role_clarity, 3),
            "workload_pressure": round(a.org_state.workload_pressure, 3),
            "retention_risk": round(a.org_state.retention_risk, 3),
        },
        "profile": _json(a.profile), "skills": _json(a.skills),
        "failure_modes": list(a.failure_modes),
        "communication_style": _json(a.communication_style),
        "work_rhythm": _json(a.work_rhythm),
        "routine_profile": _fields(rt),
        "compensation": _fields(comp) if comp else {},
        "availability": _fields(av) if av else {},
        "assigned_tasks": list(a.active_tasks),
        "assigned_tasks_count": len(a.active_tasks),
        "unread_messages_count": inv_unread,
        "open_commitments_count": open_commitments,
        "sandbox_jobs_count": sandbox_jobs,
        "memory": _agent_memory(world, aid),
        "workspace": _agent_workspace(world, aid),
    }


def _agent_workspace(world, aid: str) -> Dict[str, Any]:
    """The agent's PRIVATE workspace (frontend spec — agent perspective): notes, todos,
    current focus, drafts, and local files WITH their contents (content_summary +
    raw_payload), which other agents cannot see until shared."""
    pw = (getattr(world, "personal", {}) or {}).get(aid)
    if pw is None:
        return {"agent_id": aid, "local_files": [], "local_file_count": 0}
    files = [_file(world, fo) for fo in (getattr(pw, "local_files", {}) or {}).values()]
    return {
        "agent_id": aid,
        "personal_workspace_id": getattr(pw, "personal_workspace_id", f"pw_{aid}"),
        "current_focus": getattr(pw, "current_focus", None),
        "personal_notes": list(getattr(pw, "personal_notes", []) or []),
        "private_todos": list(getattr(pw, "private_todos", []) or []),
        "open_questions": list(getattr(pw, "open_questions", []) or []),
        "draft_docs": list(getattr(pw, "draft_docs", []) or []),
        "saved_post_ids": list(getattr(pw, "saved_post_ids", []) or []),
        "downloaded_doc_ids": list(getattr(pw, "downloaded_doc_ids", []) or []),
        "local_branch_ids": list(getattr(pw, "local_branch_ids", []) or []),
        "local_experiment_ids": list(getattr(pw, "local_experiment_ids", []) or []),
        "sandbox_id": getattr(pw, "sandbox_id", None),
        "external_contacts": list(getattr(pw, "external_contacts", []) or []),
        "local_files": files,
        "local_file_count": len(files),
    }


def _agent_memory(world, aid: str) -> Dict[str, Any]:
    rm = getattr(world, "reflection_manager", None)
    mem = (getattr(world, "agent_memories", {}) or {}).get(aid)
    out = mem.to_dict() if mem is not None else {"agent_id": aid, "reflections": [],
            "lessons_learned": [], "unresolved_needs": [], "repeated_blockers": [],
            "commitments": [], "concerns": [], "last_reflection_tick": None}
    if rm is not None:
        out["open_wishes"] = [w.wish_id for w in rm.wishes.values()
                              if w.agent_id == aid and w.status in ("open", "interpreted")]
        out["reflection_count"] = sum(1 for r in rm.reflections.values() if r.agent_id == aid)
    return out


# --------------------------------------------------------------------------- #
# Internal subsystems
# --------------------------------------------------------------------------- #
def _internal(world) -> Dict[str, Any]:
    cr = world.commitment_registry
    repo = world.repo_system.repo
    return {
        "tasks": [_task(world, t) for t in world.tasks.values()],
        "docs": [_doc(world, d) for d in world.documents.values()],
        "files": [_file(world, f) for f in _all_files(world)],
        "messages": [_message(world, m) for m in world.comm.messages.values()],
        "channels": [_channel(c) for c in world.comm.channels.values()],
        "meetings": [_meeting(world, m) for m in world.meeting_system.meetings.values()],
        "repo": {
            "repo_id": getattr(repo, "repo_id", "repo"), "name": getattr(repo, "name", ""),
            "modules": list(getattr(repo, "modules", [])),
            "build_status": getattr(repo, "build_status", "unknown"),
            "technical_debt": getattr(repo, "technical_debt", 0.0),
            "branches": [_fields(b) for b in repo.branches.values()],
            "commits": [_fields(c) for c in repo.commits.values()],
            "pull_requests": [_pr(p) for p in repo.pull_requests.values()],
            "ci_runs": [_fields(c) for c in getattr(repo, "ci_runs", {}).values()],
            "release_candidates": [_fields(rc) for rc in getattr(repo, "release_candidates", {}).values()],
            "releases": [_fields(r) for r in getattr(repo, "releases", {}).values()],
            "release_tags": list(getattr(repo, "release_tags", [])),
            "current_version": getattr(repo, "current_version", "0.0.1"),
        },
        "sandbox": {
            "sandboxes": [_fields(s) for s in world.sandbox_system.sandboxes.values()],
            "jobs": [_fields(j) for j in world.sandbox_system.jobs.values()],
        },
        "experiments": [_fields(e) for e in world.experiments.values()],
        "results": [_result(world, r) for r in world.sandbox_system.results.values()],
        "budget": _fields(world.budget_system.budget),
        "payroll": _fields(world.budget_system.payroll),
        "funding": _fields(world.budget_system.funding),
        "cost_events": [_fields(c) for c in world.budget_system.cost_events],
        "protocols": [_protocol(world, p) for p in world.protocol_registry.protocols.values()],
        "detectors": _json(world.detector_summary),
        "commitments": [_fields(c) for c in cr.commitments.values()],
        "disputes": [_fields(d) for d in cr.disputes.values()],
        "requests": [_fields(r) for r in cr.requests.values()],
        "tickets": [_fields(t) for t in world.tickets.values()],
        "artifacts": [_fields(a) for a in world.artifacts.values()],
        "issues": [_fields(i) for i in world.issues.values()],
        "searches": [_fields(s) for s in getattr(world.search_system, "logs", [])],
    }


def _task(world, t) -> Dict[str, Any]:
    d = _fields(t)
    d["debug_visible"] = True
    d["visible_to_agents"] = (sorted(world.agents.keys())
                              if getattr(t, "visibility", "team") in ("team", "public")
                              else [getattr(t, "owner_id", None)])
    return d


def _doc(world, doc) -> Dict[str, Any]:
    d = _fields(doc)
    d.update(_visibility_block(world, _doc_visible_to(world, doc),
                              private_owner=getattr(doc, "owner_id", None)))
    return d


def _file(world, f) -> Dict[str, Any]:
    d = _fields(f)
    owner = getattr(f, "owner_id", None)
    vis = getattr(f, "visibility", "private")
    vis = vis.value if isinstance(vis, enum.Enum) else vis
    visible = sorted(world.agents.keys()) if vis in ("team", "public") else [owner]
    visible = sorted(set([x for x in visible if x] + list(getattr(f, "granted_to", set()) or [])))
    d.update(_visibility_block(world, visible, private_owner=owner))
    return d


def _all_files(world):
    out = list(getattr(world.company, "files", {}).values())
    for pw in world.personal.values():
        out.extend(getattr(pw, "local_files", {}).values())
    return out


def _message(world, m) -> Dict[str, Any]:
    d = _fields(m)
    d["read_by"] = sorted(getattr(m, "read_by", set()) or set())
    d["acknowledged_by"] = sorted(getattr(m, "acknowledged_by", set()) or set())
    d.update(_visibility_block(world, _message_visible_to(world, m),
                              read_by=d["read_by"], private_owner=getattr(m, "sender_id", None)))
    return d


def _channel(c) -> Dict[str, Any]:
    return {"channel_id": c.channel_id, "channel_type": getattr(c, "channel_type", ""),
            "members": sorted(getattr(c, "members", set()) or set()),
            "message_count": len(getattr(c, "message_ids", []))}


def _meeting(world, m) -> Dict[str, Any]:
    d = _fields(m)
    note = world.meeting_system.notes.get(getattr(m, "notes_doc_id", None) or "")
    d["notes"] = _fields(note) if note else None
    d["action_items"] = [_fields(world.meeting_system.action_items[a])
                         for a in getattr(m, "action_item_ids", []) if a in world.meeting_system.action_items]
    d["decisions"] = [_fields(world.meeting_system.decisions[x])
                      for x in getattr(m, "decision_ids", []) if x in world.meeting_system.decisions]
    d["visible_to_agents"] = list(getattr(m, "participants", []))
    return d


def _pr(p) -> Dict[str, Any]:
    return _fields(p)


def _result(world, r) -> Dict[str, Any]:
    d = _fields(r)
    seen = _result_seen_by(world, r.result_id)
    d.update(_visibility_block(world, seen, private_owner=None))
    d["lifecycle"] = {
        "exists_privately": not getattr(r, "logged_to_tracker", False),
        "logged_to_tracker": getattr(r, "logged_to_tracker", False),
        "shared_in_message": any(getattr(a, "object_id", None) == r.result_id
                                 for m in world.comm.messages.values()
                                 for a in getattr(m, "attachments", []) or []),
        "disputed": bool(getattr(r, "disputed_by", [])),
    }
    return d


def _protocol(world, p) -> Dict[str, Any]:
    d = _fields(p)
    ds = world.detector_summary.get(getattr(p, "protocol_type", ""), {})
    d["detector"] = _json(ds)
    d["event_chain"] = [_fields(e) for e in world.protocol_registry.events
                        if getattr(e, "protocol_id", None) == p.protocol_id]
    return d


def _visibility_block(world, visible_to, *, read_by=None, private_owner=None) -> Dict[str, Any]:
    return {"debug_visible": True,
            "visible_to_agents": sorted([v for v in (visible_to or []) if v]),
            "read_by": list(read_by or []),
            "private_owner": private_owner}


# --------------------------------------------------------------------------- #
# External community
# --------------------------------------------------------------------------- #
def _external(world) -> Dict[str, Any]:
    comm = world.community
    posts = [_post(world, p) for p in comm.posts.values()]
    try:
        from environments.org_env.backend.market import market_summary
        market = market_summary(world)
    except Exception:
        market = {}
    return {
        "profiles": [_fields(pr) for pr in comm.profiles.values()],
        "posts": posts,
        "comments": [],   # community has only an integer comment count per post (no objects)
        "docs": [_fields(d) for d in getattr(comm, "docs", {}).values()],
        "signals": _signals(world),
        "offers": [_fields(o) for o in getattr(world.budget_system, "external_offers", {}).values()],
        # v14 P5: post-release market-validation loop (trials + conversions + WTP)
        "trials": [_fields(t) for t in getattr(world, "trials", {}).values()],
        "market": market,
    }


def _post(world, p) -> Dict[str, Any]:
    d = _fields(p)
    pid = p.post_id
    read_by, shared_to = set(), set()
    for aid, pw in world.personal.items():
        if pid in getattr(pw, "saved_post_ids", []):
            read_by.add(aid)
    cited_by = []
    for m in world.comm.messages.values():
        for a in getattr(m, "attachments", []) or []:
            if getattr(a, "object_id", None) == pid:
                shared_to.add(getattr(m, "channel_id", None))
                read_by.update(getattr(m, "read_by", set()) or set())
    d["internal_exposure"] = {
        "read_by_internal_agents": sorted(read_by),
        "shared_to_internal_channels": sorted([c for c in shared_to if c]),
        "cited_by_docs": cited_by,
    }
    return d


def _signals(world) -> List[Dict[str, Any]]:
    """No CommunitySignal objects are populated yet — synthesize from external
    signal events + market-shock events so the External tab has a signals view."""
    out = []
    for ev in world.events:
        if ev.get("type") == "external_signal_event" or ev.get("subtype") == "api_price_shock":
            out.append({"signal_id": f"sig_{len(out)}", "type": ev.get("subtype", ev.get("type")),
                        "tick": ev.get("tick"), "topic": ev.get("topic", ev.get("subtype", "")),
                        "source": ev.get("post_id") or ev.get("agent_id"), "raw": _json(ev)})
    for s in getattr(world.community, "signals", []) or []:
        out.append(_fields(s))
    return out


# --------------------------------------------------------------------------- #
# Graphs (persona per-agent, social, event, external_network)
# --------------------------------------------------------------------------- #
def _node(nid, ntype, **attrs):
    return {"id": str(nid), "type": ntype, "attrs": attrs}


def _persona_modes(world, aid: str) -> Dict[str, Any]:
    """All four persona views (summary default + policy_debug / evidence / raw),
    generated by the generic PersonaGraphBuilder (no agent hardcoding)."""
    from environments.org_env.graphs.persona_graph_builder import build_all_modes
    return build_all_modes(world.agents[aid], world)


def _persona_graph(world, aid: str) -> Dict[str, Any]:
    """Layered persona graph (NOT a pure star): traits/skills connect to SHARED
    action-feature nodes via the real policy coefficients, those features drive the
    agent; states→policy bias, style→speech, failure→risk. So e.g. speed_bias and
    urgency_bias both link to progress_gain, giving real cross-structure."""
    from environments.org_env.runtime_adapter.policy import PROFILE_COEFFS
    a = world.agents[aid]
    prof, skills = a.profile or {}, a.skills or {}
    nodes: Dict[str, Dict[str, Any]] = {}
    edges: List[Dict[str, Any]] = []

    def add(nid, ntype, **attrs):
        if nid not in nodes:
            nodes[nid] = _node(nid, ntype, **attrs)

    add(aid, "agent", name=a.name, role=a.role)
    feats_used = set()
    # trait/skill -> action_feature (policy coefficients), feature -> agent
    for key, feat, coeff in PROFILE_COEFFS:
        val = prof.get(key, skills.get(key))
        if val is None:
            continue
        is_trait = key in prof
        src = (f"trait:{key}" if is_trait else f"skill:{key}")
        add(src, "trait" if is_trait else "skill", value=round(float(val), 3))
        add(f"feat:{feat}", "action_feature", feature=feat)
        edges.append({"src": src, "type": "boosts" if coeff >= 0 else "suppresses",
                      "dst": f"feat:{feat}", "attrs": {"w": round(coeff * float(val), 3)}})
        feats_used.add(feat)
    for feat in feats_used:
        edges.append({"src": f"feat:{feat}", "type": "drives", "dst": aid})
    # traits/skills not in the coeff table still shown, linked to a "policy" hub
    add("bias:disposition", "policy_bias", label="disposition")
    edges.append({"src": "bias:disposition", "type": "shapes", "dst": aid})
    for t, v in prof.items():
        if f"trait:{t}" not in nodes:
            add(f"trait:{t}", "trait", value=round(float(v), 3))
            edges.append({"src": f"trait:{t}", "type": "tendency", "dst": "bias:disposition"})
    add("aff:skillset", "policy_bias", label="skillset")
    edges.append({"src": "aff:skillset", "type": "enables", "dst": aid})
    for s, v in skills.items():
        if f"skill:{s}" not in nodes:
            add(f"skill:{s}", "skill", value=round(float(v), 3))
            edges.append({"src": f"skill:{s}", "type": "affordance", "dst": "aff:skillset"})
    # failure modes -> shared risk hub -> agent
    if a.failure_modes:
        add("risk:profile", "policy_bias", label="risk")
        edges.append({"src": "risk:profile", "type": "threatens", "dst": aid})
        for fm in a.failure_modes:
            add(f"fail:{fm}", "failure_mode")
            edges.append({"src": f"fail:{fm}", "type": "risk", "dst": "risk:profile"})
    # communication style -> speech-realization hub -> agent
    if a.communication_style:
        add("speech:voice", "policy_bias", label="speech")
        edges.append({"src": "speech:voice", "type": "voice_of", "dst": aid})
        for k, v in a.communication_style.items():
            add(f"style:{k}", "communication_style", value=_json(v))
            edges.append({"src": f"style:{k}", "type": "shapes", "dst": "speech:voice"})
    # work state -> policy-modifier hub -> agent
    add("policy:state", "policy_bias", label="work-state")
    edges.append({"src": "policy:state", "type": "modifies", "dst": aid})
    for k, v in a.work_state.snapshot().items():
        if isinstance(v, (int, float)):
            add(f"state:{k}", "state", value=_json(v))
            edges.append({"src": f"state:{k}", "type": "modulates", "dst": "policy:state"})
    # routine -> agent
    add(f"routine:{aid}", "routine", **_fields(get_routine(aid)))
    edges.append({"src": f"routine:{aid}", "type": "time_preference", "dst": aid})
    return {"nodes": list(nodes.values()), "edges": edges}


def _social_graph(world) -> Dict[str, Any]:
    nodes = [_node(aid, "agent", name=world.agents[aid].name) for aid in world.agents]
    agg: Dict[tuple, int] = {}
    # agent->agent edges from the event graph (requested_review_from / promised / ...)
    for (src, etype, dst) in world.event_graph.edges:
        if src in world.agents and dst in world.agents and src != dst:
            agg[(src, etype, dst)] = agg.get((src, etype, dst), 0) + 1
    # communication edges: sender -> each recipient (the chat network)
    for m in world.comm.messages.values():
        s = getattr(m, "sender_id", None)
        if s not in world.agents:
            continue
        for r in getattr(m, "recipients", []) or []:
            if r in world.agents and r != s:
                agg[(s, "messaged", r)] = agg.get((s, "messaged", r), 0) + 1
    edges = [{"src": s, "type": t, "dst": d, "attrs": {"weight": w}} for (s, t, d), w in agg.items()]
    return {"nodes": nodes, "edges": edges}


def _event_graph(world) -> Dict[str, Any]:
    eg = world.event_graph
    nodes = [_node(nid, ntype) for nid, ntype in eg.nodes.items()]
    edges = [{"src": s, "type": t, "dst": d} for (s, t, d) in eg.edges]
    # overlay episode structure (episode -> involved agent / produced object / related)
    mgr = getattr(world, "episode_manager", None)
    if mgr is not None:
        existing = {n["id"] for n in nodes}
        for ep in mgr.episodes.values():
            nodes.append(_node(ep.episode_id, "episode", episode_type=ep.episode_type,
                               status=ep.status, title=ep.title))
            for a in ep.participants:
                edges.append({"src": ep.episode_id, "type": "involved", "dst": a})
            if ep.trigger_object_id and ep.trigger_object_id in existing:
                edges.append({"src": ep.episode_id, "type": "triggered_by", "dst": ep.trigger_object_id})
            for o in (ep.produced_artifacts + ep.produced_protocols + ep.produced_tasks
                      + ep.produced_product_changes):
                if o in existing:
                    edges.append({"src": ep.episode_id, "type": "produced", "dst": o})
            for rid in ep.related_episode_ids:
                edges.append({"src": ep.episode_id, "type": "related_to", "dst": rid})
    return {"nodes": nodes, "edges": edges}


def _external_network_graph(world) -> Dict[str, Any]:
    comm = world.community
    nodes, edges = [], []
    for pr in comm.profiles.values():
        nodes.append(_node(pr.external_agent_id, "external_profile", name=pr.name, role=pr.role))
        for nb in getattr(pr, "network_neighbors", []) or []:
            edges.append({"src": pr.external_agent_id, "type": "connected", "dst": nb})
    for p in comm.posts.values():
        nodes.append(_node(p.post_id, "external_post", topic=p.topic))
        edges.append({"src": p.author_id, "type": "posted", "dst": p.post_id})
        for aid, pw in world.personal.items():
            if p.post_id in getattr(pw, "saved_post_ids", []):
                if not any(n["id"] == aid for n in nodes):
                    nodes.append(_node(aid, "internal_agent", name=world.agents[aid].name))
                edges.append({"src": aid, "type": "read_by", "dst": p.post_id})
    return {"nodes": nodes, "edges": edges}


# --------------------------------------------------------------------------- #
# Company dashboard summary
# --------------------------------------------------------------------------- #
def _governance_block(world) -> Dict[str, Any]:
    """Approval mode + who-must-approve-what, so the inspector shows institution
    formation as an explicit, agent-driven (not auto) process."""
    pm = getattr(world, "proposal_manager", None)
    pending = []
    if pm is not None:
        for p in pm.proposals.values():
            if p.status != "under_review":
                continue
            pending.append({"proposal_id": p.proposal_id, "title": p.title, "type": p.proposal_type,
                            "adoption_score": p.adoption_score,
                            "approval_required_from": list(p.approval_required_from),
                            "approved_by": list(p.approved_by), "rejected_by": list(p.rejected_by),
                            "suggested_revision": p.suggested_revision})
    return {"approval_mode": getattr(world, "approval_mode", "auto"),
            "auto_approve": bool(getattr(world, "auto_approve", True)),
            "pending_approvals": pending,
            "adopted": sum(1 for p in (pm.proposals.values() if pm else []) if p.status == "adopted"),
            "rejected": sum(1 for p in (pm.proposals.values() if pm else []) if p.status == "rejected")}


def _org_metrics_safe(world) -> Dict[str, Any]:
    try:
        from environments.org_env.runtime_adapter.org_metrics import compute_org_metrics
        return compute_org_metrics(world)
    except Exception:
        return {}


def _coding_metrics_safe(world) -> Dict[str, Any]:
    try:
        from environments.org_env.coding.metrics import coding_metrics
        return coding_metrics(world)
    except Exception:
        return {}


def _company(world) -> Dict[str, Any]:
    b = world.budget_system.budget
    pay = world.budget_system.payroll
    clk = world.time.clock
    tasks = list(world.tasks.values())
    # P0-7: recompute from the task table (the enum's str() is "TaskStatus.DONE", so the
    # old endswith("done") always returned 0).
    def _ts(t):
        return getattr(getattr(t, "status", None), "value", str(getattr(t, "status", ""))).lower()
    tasks_by_status: Dict[str, int] = {}
    for t in tasks:
        tasks_by_status[_ts(t)] = tasks_by_status.get(_ts(t), 0) + 1
    # spec #2: a product task counts as done only once merged/released (or plain done).
    done = sum(1 for t in tasks if _ts(t) in ("done", "merged", "released"))
    in_progress = sum(1 for t in tasks if _ts(t) in ("in_progress", "review",
                                                     "implementation_done", "review_pending"))
    open_tasks = sum(1 for t in tasks if _ts(t) == "open")
    blockers = sum(1 for t in tasks if _ts(t) == "blocked")
    open_prs = sum(1 for p in world.repo_system.repo.pull_requests.values()
                   if getattr(getattr(p, "status", ""), "value", str(getattr(p, "status", "")))
                   not in ("merged", "closed"))
    running_jobs = sum(1 for j in world.sandbox_system.jobs.values() if getattr(j, "status", "") == "running")
    untracked = sum(1 for r in world.sandbox_system.results.values() if not getattr(r, "logged_to_tracker", False))
    today = clk.day_index
    meetings_today = sum(1 for m in world.meeting_system.meetings.values()
                         if getattr(m, "scheduled_tick", 0) // 24 == today)
    overtime = len(world.time.overtime_logs)
    burnout = [round(a.work_state.burnout_risk, 2) for a in world.agents.values()]
    return {
        "name": "LanternForge",
        "cash_balance": getattr(b, "cash_balance", 0.0),
        "remaining_budget": getattr(b, "remaining_budget", 0.0),
        "runway_days": getattr(b, "runway_days", 0.0),
        "budget_pressure": getattr(b, "budget_pressure", 0.0),
        "cost_multiplier": getattr(b, "cost_multiplier", 1.0),
        "payroll_status": getattr(pay, "payroll_status", "normal"),
        "missed_payroll_count": getattr(pay, "missed_payroll_count", 0),
        # v8g P1: unified token currency — treasury, REAL rolling daily burn, runway derived
        # from them, runway_pressure for policy, and a tail of the token ledger.
        "treasury_tokens": round(getattr(b, "cash_balance", 0.0), 1),
        "daily_burn_tokens": world.budget_system.daily_burn_tokens(world.world_tick),
        "runway_days_tokens": round(getattr(b, "cash_balance", 0.0)
                                    / max(1.0, world.budget_system.daily_burn_tokens(world.world_tick)), 1),
        "runway_pressure": round(min(1.0, world.budget_system.daily_burn_tokens(world.world_tick) * 7.0
                                     / max(1.0, getattr(b, "cash_balance", 0.0))), 3),
        "token_ledger_tail": list(getattr(world.budget_system, "token_ledger", []) or [])[-40:],
        "token_ledger_count": len(getattr(world.budget_system, "token_ledger", []) or []),
        "token_debits_total": round(sum(e["amount"] for e in getattr(world.budget_system, "token_ledger", [])
                                        if e.get("kind") == "debit"), 1),
        "token_credits_total": round(sum(e["amount"] for e in getattr(world.budget_system, "token_ledger", [])
                                         if e.get("kind") == "credit"), 1),
        # spec #5: funding checkpoint state + decisions
        "next_tranche_tick": min([t.scheduled_tick for t in world.budget_system.funding.tranches
                                  if t.status in ("scheduled", "delayed")], default=None),
        "milestone_status": dict(getattr(world.budget_system.funding, "milestone_status", {}) or {}),
        "funding_history": list(getattr(world.budget_system.funding, "funding_history", []) or []),
        "task_count": len(tasks), "tasks_done": done, "tasks_in_progress": in_progress,
        "tasks_open": open_tasks, "critical_blockers": blockers, "tasks_by_status": tasks_by_status,
        "institution_context": dict(getattr(world, "institution_context", {}) or {}),
        # v8g P3: minimal organization-level metrics (cycle time / latency / rates / tokens)
        "org_metrics": _org_metrics_safe(world),
        # spec #8: institutionalised mechanisms (adopted + actually used)
        "company_skills": [dict(s) for s in getattr(world, "company_skills", []) or []],
        "company_skill_count": len(getattr(world, "company_skills", []) or []),
        # v11 §8: coding capability metrics (merged PRs / test pass / blocker resolution / ...)
        "coding_metrics": _coding_metrics_safe(world),
        # spec #6: pending split work units + average tick cost per action type
        "pending_jobs": [dict(j) for j in getattr(world, "pending_jobs", []) or []],
        "pending_job_count": sum(1 for j in getattr(world, "pending_jobs", []) or []
                                 if j.get("status") == "pending"),
        "avg_tick_cost_by_action": {a: round(sum(v) / len(v), 2)
                                    for a, v in (getattr(world, "_action_tick_cost", {}) or {}).items() if v},
        "open_prs": open_prs, "running_sandbox_jobs": running_jobs, "untracked_results": untracked,
        "meetings_today": meetings_today, "overtime_count": overtime,
        "burnout_risk_max": max(burnout) if burnout else 0.0,
        "burnout_risk_mean": round(sum(burnout) / len(burnout), 3) if burnout else 0.0,
        "demo_ready": any(getattr(d, "doc_type", "") == "experiment_tracker" for d in world.documents.values()),
        "protocol_count": len(world.protocol_registry.protocols),
        "weak_protocols": [p.protocol_type for p in world.protocol_registry.protocols.values()
                           if getattr(p, "emergence_level", "none") in ("weak", "strong")],
    }


def _timeline(world, limit: int = 60) -> List[Dict[str, Any]]:
    return [_json(e) for e in world.events[-limit:]]


def _logs(world, limit: int = 200) -> Dict[str, Any]:
    return {
        "actions": _json(world.action_log[-limit:]),
        "appraised": _json(world.appraised_log[-limit:]),
        "work_sessions": [_fields(w) for w in world.time.work_sessions[-limit:]],
        "appraisals": _json(world.object_appraisal_log[-limit:]),
        "feedback_decisions": _json(world.feedback_decision_log[-limit:]),
        "text_generation": _json(world.text_generation_log[-limit:]),
        "protocol_events": [_fields(e) for e in world.protocol_registry.events[-limit:]],
        "recovery": [_fields(r) for r in world.time.recovery_logs[-limit:]],
        "agent_log": [e.to_dict() for e in getattr(world, "agent_log", [])[-limit:]],
        "policy_trace": _json(getattr(world, "policy_trace", [])[-limit:]),
    }


# --------------------------------------------------------------------------- #
# The full snapshot
# --------------------------------------------------------------------------- #
def _product_block(world) -> Dict[str, Any]:
    ps = getattr(world, "product", None)
    arts = getattr(world, "product_artifacts", {}) or {}
    if ps is None:
        return {"stage": None, "repo_files": [], "open_issues": [], "known_gaps": [],
                "artifacts": [], "recent_changes": []}
    items = [a.to_dict() for a in arts.values()]
    files = [a for a in items if a["artifact_type"] != "issue"]
    issues = [a for a in items if a["artifact_type"] == "issue" and a["status"] == "open"]
    recent = sorted([a for a in items if a["updated_at_tick"] > 0],
                    key=lambda a: -a["updated_at_tick"])[:12]
    # v6 P0.5: persist the full execution-layer patch objects (applied AND rejected),
    # so a trajectory can be evaluated for HOW the product changed — pseudo_diff,
    # added_fields/checks, removed_overclaims, resolved_gaps, validation_status,
    # rejection_reason, related task/issue ids — not just patch ids + change summaries.
    patches = [p.to_dict() for p in (getattr(world, "patches", {}) or {}).values()]
    patches.sort(key=lambda d: int(d.get("tick", 0) or 0))
    # spec #1: status-bearing gaps + derived readiness + reconciliation warnings
    gaps = [g.to_dict() for g in (getattr(world, "known_gaps", {}) or {}).values()]
    gaps_by_status: Dict[str, int] = {}
    for g in gaps:
        gaps_by_status[g["status"]] = gaps_by_status.get(g["status"], 0) + 1
    return {
        "product_id": ps.product_id, "name": ps.name, "stage": ps.stage, "summary": ps.summary,
        "known_gaps": list(ps.known_systemic_issues), "repo_files": files,
        "open_issues": issues, "artifacts": items, "recent_changes": recent,
        "open_issue_count": len(issues), "artifact_count": len(items),
        "patches": patches, "patch_count": len(patches),
        "patch_applied": sum(1 for d in patches if d.get("validation_status") == "accepted"),
        "patch_rejected": sum(1 for d in patches if d.get("validation_status") == "rejected"),
        "known_gaps_v2": gaps, "known_gaps_by_status": gaps_by_status,
        "readiness": dict(getattr(world, "product_readiness", {}) or {}),
        "reconcile_warnings": list(getattr(world, "_reconcile_warnings", []) or []),
        "company_config": _json(getattr(world, "company_config", {})),
        "substrate_type": getattr(ps, "substrate_type", "synthetic_lanternscout"),
        # OSS time-machine: counts-only summary — never reference code or hidden-test content (§7.2)
        "oss_eval": _oss_eval_summary(world),
    }


def _oss_eval_summary(world) -> Dict[str, Any]:
    try:
        from environments.org_env.product.substrates.eval_assets import oss_eval_public_summary
        return oss_eval_public_summary(world) or {}
    except Exception:
        return {}


def _growth_block(world) -> Dict[str, Any]:
    """Internal Growth Module (§16 panels + §17 metrics): per-agent skill/reputation/
    authority/go-to, recent GrowthEvents, domain authority leaderboard, specialization
    index + authority shift from the t0 baseline."""
    import math
    from environments.org_env.growth.objects import REPUTATION_DOMAINS, SKILL_DOMAINS
    agents = getattr(world, "agents", {}) or {}
    per_agent = {}
    for aid, a in agents.items():
        skills = {k: round(float(a.skills.get(k, 0.0)), 4) for k in SKILL_DOMAINS if k in a.skills}
        per_agent[aid] = {
            "name": getattr(a, "name", aid),
            "skills": skills,
            "reputation": {d: round(float(getattr(a, "reputation", {}).get(d, 0.5)), 4) for d in REPUTATION_DOMAINS},
            "authority": {d: round(float(getattr(a, "authority", {}).get(d, 0.0)), 4) for d in REPUTATION_DOMAINS},
            "go_to_tags": list(getattr(a, "go_to_tags", []) or []),
        }
    # domain -> ranked agents by authority (the informal authority graph)
    leaderboard = {}
    for d in REPUTATION_DOMAINS:
        ranked = sorted(agents.values(), key=lambda a: -float(getattr(a, "authority", {}).get(d, 0.0)))
        leaderboard[d] = [(a.id, round(float(getattr(a, "authority", {}).get(d, 0.0)), 3))
                          for a in ranked if float(getattr(a, "authority", {}).get(d, 0.0)) > 0][:3]
    # §17.1 specialization index (1 - normalized entropy of authority per domain)
    n = max(1, len(agents))
    spec_d = {}
    for d in REPUTATION_DOMAINS:
        vals = [float(getattr(a, "authority", {}).get(d, 0.0)) for a in agents.values()]
        tot = sum(vals)
        if tot <= 0:
            spec_d[d] = 0.0
            continue
        ps = [v / tot for v in vals if v > 0]
        H = -sum(p * math.log(p) for p in ps)
        spec_d[d] = round(1 - H / math.log(n), 4) if n > 1 else 0.0
    specialization = round(sum(spec_d.values()) / len(REPUTATION_DOMAINS), 4)
    # §17.2 authority shift from t0
    t0 = getattr(world, "_authority_t0", {}) or {}
    shift = 0.0
    if t0:
        for d in REPUTATION_DOMAINS:
            shift += sum(abs(float(getattr(a, "authority", {}).get(d, 0.0)) - float(t0.get(aid, {}).get(d, 0.0)))
                         for aid, a in agents.items())
        shift = round(shift / len(REPUTATION_DOMAINS), 4)
    events = [e.to_dict() for e in (getattr(world, "growth_events", []) or [])]
    return {
        "agents": per_agent,
        "authority_leaderboard": leaderboard,
        "go_to": {aid: pa["go_to_tags"] for aid, pa in per_agent.items() if pa["go_to_tags"]},
        "specialization_index": specialization,
        "specialization_by_domain": spec_d,
        "authority_shift": shift,
        "growth_events": events[-40:],
        "growth_event_count": len(events),
    }


def _llm_block(world) -> Dict[str, Any]:
    c = getattr(world, "llm_client", None)
    st = c.stats() if c is not None else {"provider": None, "calls": 0, "failures": 0}
    return {"enabled": c is not None, "provider": st.get("provider"),
            "calls": st.get("calls", 0), "failures": st.get("failures", 0),
            "fallbacks": int(getattr(world, "_llm_fallbacks", 0))}


def _action_decisions_block(world) -> Dict[str, Any]:
    # Telemetry definitions (preflight §15.2):
    #   rejected_count  = LLM candidate generated but the validator rejected it
    #   fallback_count  = historical counter name for failed/rejected LLM attempts;
    #                     no profile-policy action is executed in their place
    #   accepted_count  = validated LLM action executed
    decs = getattr(world, "action_decisions", []) or []
    items = [d.to_dict() for d in decs]
    # #9: DISAMBIGUATE the counters. `total` here counts LLM ACTION decisions only (0 by design when
    # the policy drives actions) — it is NOT "did the org act". Surface the world/policy/cognitive
    # counts alongside so `total=0` can't be misread as "no decisions/actions".
    extra = {
        "counter_semantics": "LLM action-decisions only; policy actions are world_actions_total",
        "llm_drives_actions": bool(getattr(world, "llm_decides_actions", False)),
        "action_selection_mode": getattr(
            world, "action_selection_mode", "profile_policy"
        ),
        "world_actions_total": len(getattr(world, "action_log", []) or []),
    }
    # v8 #4: prefer the uncapped running tally; fall back to the (capped) list for old worlds.
    tally = getattr(world, "action_decision_tally", None)
    if tally:
        block = {"recent": items[-40:],
                 "accepted_count": tally["accepted"], "rejected_count": tally["rejected"],
                 "fallback_count": tally["fallback"], "total": tally["total"]}
    else:
        block = {"recent": items[-40:],
                 "accepted_count": sum(1 for d in decs if d.validation_status == "accepted"),
                 "rejected_count": sum(1 for d in decs if d.validation_status == "rejected"),
                 "fallback_count": sum(1 for d in decs
                                       if d.validation_status in ("rejected", "fallback_used")),
                 "total": len(decs)}
    block["llm_action_decisions_total"] = block["total"]
    block.update(extra)
    return block


def org_lived_full_snapshot(world, *, mode: str = "live") -> Dict[str, Any]:
    clk = world.time.clock
    return {
        "wired": True, "mode": mode,
        "tick": world.world_tick, "day": clk.day_index, "hour": clk.hour_in_day,
        "phase": clk.phase_of_day, "day_of_week": clk.day_of_week_name,
        "is_weekday": clk.is_weekday, "is_weekend": clk.is_weekend,
        "is_work_hours": clk.is_work_hours,
        "scenario": {
            "name": world.scenario.name,
            "seed": world.scenario.seed,
            "policy_mode": getattr(world, "policy_mode", "mock"),
            "experiment_condition": getattr(
                world, "experiment_condition", "b3_full_sociogenesis"
            ),
            "action_selection_mode": getattr(
                world, "action_selection_mode", "profile_policy"
            ),
            "condition_explicit": bool(
                getattr(world, "experiment_condition_explicit", False)
            ),
        },
        "baseline": {
            "epoch": int(getattr(world, "baseline_epoch", 0)),
            "sprint_ticks": int(getattr(world, "baseline_sprint_ticks", 0)),
            "reset_ticks": list(getattr(world, "baseline_reset_ticks", []) or []),
            "archived_action_count": len(
                getattr(world, "baseline_archived_action_log", []) or []
            ),
            "archived_policy_trace_count": len(
                getattr(world, "baseline_archived_policy_trace", []) or []
            ),
            "profile_conditioning_enabled": bool(
                getattr(world, "profile_conditioning_enabled", True)
            ),
            "action_selection_mode": getattr(
                world, "action_selection_mode", "profile_policy"
            ),
            "configured_action_selection_mode": getattr(
                getattr(world, "condition_spec", None),
                "action_selection_mode",
                "profile_policy",
            ),
            "profile_assignment": getattr(
                getattr(world, "condition_spec", None),
                "profile_assignment",
                "aligned",
            ),
            "institutionalization_enabled": bool(
                getattr(world, "institutionalization_enabled", True)
            ),
            "capability_learning_enabled": bool(
                getattr(world, "capability_learning_enabled", True)
            ),
        },
        "company": _company(world),
        "agents": {aid: _agent_snapshot(world, aid) for aid in world.agents},
        "internal": _internal(world),
        "external": _external(world),
        "episodes": world.episode_manager.snapshot() if getattr(world, "episode_manager", None) else
        {"items": [], "open_count": 0, "closed_count": 0, "by_type": {}, "total": 0},
        "reflections": world.reflection_manager.snapshot(world) if getattr(world, "reflection_manager", None)
        else {"items": [], "recent": [], "by_agent": {}, "total": 0},
        "wishes": world.reflection_manager.wishes_snapshot() if getattr(world, "reflection_manager", None)
        else {"items": [], "by_type": {}, "total": 0},
        "agent_memories": world.reflection_manager.memories_snapshot(world)
        if getattr(world, "reflection_manager", None) else {},
        "llm": _llm_block(world),
        "action_decisions": _action_decisions_block(world),
        "proposals": world.proposal_manager.proposals_snapshot() if getattr(world, "proposal_manager", None)
        else {"items": [], "by_status": {}, "by_type": {}, "total": 0},
        "tools": world.proposal_manager.tools_snapshot() if getattr(world, "proposal_manager", None)
        else {"items": [], "active_count": 0, "total": 0},
        "protocol_specs": world.proposal_manager.protocol_specs_snapshot() if getattr(world, "proposal_manager", None)
        else {"items": [], "active_count": 0, "total": 0},
        "governance": _governance_block(world),
        "episode_summaries": _json(getattr(world, "episode_summaries", {})),
        "product": _product_block(world),
        "growth": _growth_block(world),
        "graphs": {
            "persona": {aid: _persona_modes(world, aid) for aid in world.agents},
            "social": _social_graph(world),
            "event": _event_graph(world),
            "external_network": _external_network_graph(world),
        },
        "timeline": _timeline(world),
        "logs": _logs(world),
    }


__all__ = ["org_lived_full_snapshot"]
