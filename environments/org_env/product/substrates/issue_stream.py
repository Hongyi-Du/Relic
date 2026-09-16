"""Historical issue stream for OSS time-machine substrates (brief §9).

Real historical issues are the experiment backbone. They are seeded as product issues at t0 (so the
org has concrete work) AND released over time as EXTERNAL community pressure that mirrors the
project's real timeline:

    historical issue  ->  external post / customer complaint  ->  internal feed  ->  ticket

Held-out issues are NEVER placed on the stream (they live only in evaluator assets), so they can't be
released during the capability-formation phase. The ticket-closure rule (§9.3) forbids closing an
issue/ticket on a stale patch — it requires post-issue work AND passing test evidence.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from environments.org_env.product.substrates.eval_assets import (
    is_oss_substrate,
    oss_issue_has_hidden_test,
)

# Backlog retry loop (review §6 follow-up): an issue is NOT retired just because a patch touched its
# file — it stays in the actionable backlog until genuinely resolved. To avoid re-editing the same
# file every tick (which would starve everything else via the attractor guard), each landed patch
# opens a cooldown window; after it, an unresolved issue re-surfaces for another attempt.
_OSS_RETRY_COOLDOWN = int(os.environ.get("ORG_OSS_RETRY_COOLDOWN", "8") or 8)

# When to stop forcing an issue. Counting attempts alone ended work that was
# getting somewhere: on a five-step greenfield Pack every issue retired at
# exactly four attempts, so the whole coding budget was twenty tries for five
# modules written from nothing, and by the end the backlog was empty while the
# tree was at 16 of 35 contracts. The refusals over that stretch had been moving
# — a missing symbol, then a parameter name, then a KeyError from real logic —
# which is the shape of converging, and the cap could not tell that from spinning.
#
# So what retires an issue is a stretch of attempts after which the gate's
# complaint has not changed at all. An absolute ceiling stays as a backstop
# against unbounded churn, high enough not to interrupt an organization that is
# still learning something.
_OSS_STALL_LIMIT = int(os.environ.get("ORG_OSS_STALL_LIMIT", "4") or 4)
_OSS_MAX_ATTEMPTS = int(os.environ.get("ORG_OSS_MAX_ATTEMPTS", "24") or 24)


def _community(world: Any):
    return getattr(world, "community", None)


def _severity_strength(severity: str) -> float:
    return {"critical": 1.0, "major": 0.8, "medium": 0.5, "minor": 0.3}.get(severity, 0.5)


def release_due_issues(world: Any, tick: int) -> List[str]:
    """Release every not-yet-released public issue whose ``release_tick <= tick`` as an external
    community complaint (+ a CustomerTicket + an external_signal_event). Returns released issue ids.
    No-op for non-OSS worlds. Idempotent per issue (``released`` flag)."""
    if not is_oss_substrate(world):
        return []
    stream: List[Dict[str, Any]] = world.__dict__.get("_oss_issue_stream") or []
    if not stream:
        return []
    community = _community(world)
    if getattr(world, "events", None) is None:
        world.events = []
    tickets = world.__dict__.setdefault("tickets", {})

    released: List[str] = []
    for entry in stream:
        if entry.get("released") or int(entry.get("release_tick", 0) or 0) > int(tick):
            continue
        iid = entry["issue_id"]
        component = entry.get("component", "") or "product"
        severity = entry.get("severity", "medium")
        # The user report in full: this post is the agent-facing copy of the
        # issue, so a 240-character cut removed the reproduction steps and left
        # a headline nobody could act on.
        text = f"{entry.get('title', '')}: {entry.get('body', '')}".strip()

        if community is not None:
            from environments.org_env.backend.community.objects import MarketSignal, Post
            pid = f"post_oss_issue_{iid}"
            community.add_post(Post(
                post_id=pid, author_id=f"ext_user_{iid}", topic=component,
                content_summary=f"[user report] {text}", stance=-0.6,
                credibility=0.6, reach=20, created_tick=int(tick), visibility="public"))
            if severity in ("major", "critical"):
                community.add_signal(MarketSignal(
                    signal_id=f"sig_oss_issue_{iid}", signal_type="customer_demand_shift",
                    topic=component, severity=severity, start_tick=int(tick),
                    source_posts=[pid], strength=_severity_strength(severity)))

        _write_ticket(world, tickets, iid, component, severity, text)
        world.events.append({"type": "external_signal_event", "subtype": "historical_issue_released",
                             "issue_id": iid, "component": component, "severity": severity,
                             "tick": int(tick), "source": "oss_time_machine"})
        entry["released"] = True
        released.append(iid)
    return released


def _write_ticket(world: Any, tickets: Dict[str, Any], issue_id: str, component: str,
                  severity: str, text: str) -> None:
    tid = f"ticket_oss_{issue_id}"
    if tid in tickets:
        return
    try:
        from environments.org_env.backend.entities.economy import CustomerTicket
        tickets[tid] = CustomerTicket(
            ticket_id=tid, customer_type="oss_user", complaint_or_request=text,
            severity=severity, topic=component, status="open", response_status="pending")
    except Exception:
        tickets[tid] = {"ticket_id": tid, "topic": component, "severity": severity,
                        "status": "open", "complaint_or_request": text, "issue_id": issue_id}


def _component_artifact_ids(world: Any, issue_id: str) -> list:
    """Artifact ids for an issue's component — exact path or logical name via component_map (§7)."""
    stream = world.__dict__.get("_oss_issue_stream") or []
    comp = ""
    for e in stream:
        if e.get("issue_id") == issue_id:
            comp = e.get("component", "") or ""
            break
    if not comp:
        return []
    arts = getattr(world, "product_artifacts", {}) or {}
    path_to_art = {getattr(a, "linked_file_path", ""): getattr(a, "artifact_id", "")
                   for a in arts.values() if getattr(a, "linked_file_path", "")}
    cmap = world.__dict__.get("_oss_component_map") or {}
    if comp in path_to_art:
        return [path_to_art[comp]]
    return [path_to_art[p] for p in (cmap.get(comp) or []) if p in path_to_art]


def _has_post_issue_work(world: Any, issue_id: str) -> bool:
    """True iff a task/patch/PR linked to the issue (or its component) advanced AFTER the issue was
    created (brief §9.3 conditions 1+2). Scans world.events defensively across field spellings."""
    arts = getattr(world, "product_artifacts", {}) or {}
    iss = arts.get(issue_id)
    created = int(getattr(iss, "created_at_tick", 0) or 0) if iss is not None else 0
    comp_arts = set(_component_artifact_ids(world, issue_id))
    for ev in (getattr(world, "events", []) or []):
        if int(ev.get("tick", 0) or 0) < created:
            continue
        if ev.get("type") not in ("repo_event", "patch_event", "product_event", "task_event"):
            continue
        if ev.get("issue_id") == issue_id or issue_id in (ev.get("linked_issues") or []):
            return True
        if comp_arts and (ev.get("artifact_id") in comp_arts
                          or comp_arts & set(ev.get("changed_artifacts") or [])
                          or comp_arts & set(ev.get("artifact_ids") or [])):
            return True
    return False


def _oss_attempts_that_changed_nothing(world: Any, comp_arts: List[str]) -> int:
    """How long the build gate has been saying the same thing about these modules.

    A stretch of identical refusals is the one thing that distinguishes an
    organization spinning from one converging: while its work is landing somewhere
    the refusal moves, or more checks pass behind it, even though it stays a
    refusal. The bookkeeping is kept where the verdict is decided, against a stable
    signature (step, module, error kind) rather than the message text, because a
    temporary path or a tick changes the text while the situation is identical.
    """
    arts = getattr(world, "product_artifacts", {}) or {}
    book = world.__dict__.get("_gate_stall") or {}
    if not book:
        return 0
    stalls = [int((book.get(str(getattr(arts.get(a), "linked_file_path", "") or ""))
                   or {}).get("stalled", 0) or 0)
              for a in (comp_arts or ())]
    return max(stalls) if stalls else 0


def _oss_patch_attempts(world: Any, issue_id: str, comp_arts: List[str], created: int) -> tuple:
    """(count, last_tick) of LANDED patches attributable to this issue — used for the retry cadence.

    Patches dealt by the OSS/agent issue candidate carry ``related_issue_ids`` (set in
    _patch_artifact), so counting is per-issue and avoids component cross-talk (e.g. a submodules
    patch on cloning.py no longer counts as an http attempt). Falls back to the component file for
    untagged patches.

    ``created`` bounds only that fallback. An OSS issue is seeded onto the board at t0 but its
    artifact records the tick it was RELEASED as community pressure, which for a late issue is well
    past the work: on vite, twenty-six patches explicitly tagged issue_concurrent_denied_request
    landed between t1 and t115 while the artifact said it was created at t132, so every one of them
    was filtered out, the count stayed at zero, the attempt cap never engaged, and the organization
    rewrote one file twenty-five times while eleven other issues went untouched. An explicit tag is
    unambiguous and needs no window; an inference from the file alone still does."""
    comp = set(comp_arts or [])
    count, last = 0, -1
    for p in (getattr(world, "patches", {}) or {}).values():
        if getattr(p, "validation_status", "") not in ("accepted", "applied", "merged"):
            continue
        t = int(getattr(p, "applied_tick", 0) or getattr(p, "tick", 0) or 0)
        rel = set(getattr(p, "related_issue_ids", []) or [])
        if issue_id in rel:
            attributed = True
        elif not rel and getattr(p, "target_object_id", None) in comp:
            # Inferred from the file alone, so the window still applies: work that
            # predates the issue says nothing about how hard this issue is.
            attributed = t >= int(created or 0)
        else:
            attributed = False
        if attributed:
            count += 1
            if t > last:
                last = t
    return count, last


def _oss_ticket_resolved(world: Any, issue_id: str) -> bool:
    t = (world.__dict__.get("tickets") or {}).get(f"ticket_oss_{issue_id}")
    if t is None:
        return False
    st = t.get("status") if isinstance(t, dict) else getattr(t, "status", None)
    rst = t.get("response_status") if isinstance(t, dict) else getattr(t, "response_status", None)
    return st == "resolved" or rst == "resolved"


def _oss_issue_done(world: Any, issue_id: str) -> bool:
    """An OSS stream issue is retired from the backlog ONLY when genuinely resolved. Genuine
    resolution = the EVALUATOR's ticket closure (hidden test + post-issue work). A bare ``close_issue``
    on the artifact does NOT retire it (the "declared victory" gap). Issues that have no hidden test
    have no evaluator signal, so there the org's own ``close_issue`` (artifact closed) is honored."""
    if _oss_ticket_resolved(world, issue_id):
        return True
    if not oss_issue_has_hidden_test(world, issue_id):
        a = (getattr(world, "product_artifacts", {}) or {}).get(issue_id)
        if a is not None and getattr(a, "status", "open") == "closed":
            # "Declared victory" is only honoured when the org has verification
            # evidence to declare it on. Where the substrate ships a public test
            # suite, a close with the suite last seen FAILING is not a
            # resolution — that is exactly how a 336-tick run closed four issues
            # while fixing none of them. Substrates without a suite keep the
            # previous behaviour (no signal exists to demand).
            return _public_tests_support_closure(world)
    return False


def _public_tests_support_closure(world: Any) -> bool:
    """Whether public-test evidence permits honouring the org's own close_issue."""
    try:
        from environments.org_env.product.materialize import declared_public_test_command
    except Exception:
        return True
    if not declared_public_test_command(world):
        return True                      # nothing to verify against
    last = world.__dict__.get("_public_tests_last")
    if not isinstance(last, dict):
        return False                     # a suite exists but was never run
    return bool(last.get("passed"))


def _agent_issue_component_artifacts(world: Any, iss: Any) -> List[str]:
    """Map an AGENT-CREATED issue (from ``create_issue``) to code artifact ids by matching its
    topic/title/description against the component_map keys or a real file name. Returns [] for
    non-coding issues (customer/docs), which therefore never enter the coding backlog."""
    cmap = world.__dict__.get("_oss_component_map") or {}
    arts = getattr(world, "product_artifacts", {}) or {}
    path_to_art = {getattr(a, "linked_file_path", ""): getattr(a, "artifact_id", "")
                   for a in arts.values() if getattr(a, "linked_file_path", "")}
    text = " ".join(str(getattr(iss, k, "") or "") for k in ("topic", "title", "description")).lower()
    if not text.strip():
        return []
    hits: List[str] = []
    for comp, paths in cmap.items():
        if comp and comp.lower() in text:
            hits += [path_to_art[p] for p in paths if p in path_to_art]
    for p, aid in path_to_art.items():
        base = p.split("/")[-1].lower()
        if p.lower() in text or (len(base) > 4 and base in text):
            hits.append(aid)
    seen, out = set(), []
    for h in hits:
        if h and h not in seen:
            seen.add(h)
            out.append(h)
    return out


def _agent_created_coding_issues(world: Any, tick: int) -> List[Dict[str, Any]]:
    """Fix #2: OPEN issues the org filed itself (``create_issue``) that map to a code module feed the
    same coding backlog, so agents can ADD new work (e.g. turning customer feedback into a fix)."""
    issues = getattr(world, "issues", {}) or {}
    if not issues:
        return []
    out: List[Dict[str, Any]] = []
    for iid, iss in issues.items():
        if getattr(iss, "status", "open") not in ("open", "in_progress", "reopened", "triaged"):
            continue
        comp_arts = _agent_issue_component_artifacts(world, iss)
        if not comp_arts:
            continue
        created = int(getattr(iss, "created_tick", 0) or 0)
        count, last = _oss_patch_attempts(world, iid, comp_arts, created)
        if count >= _OSS_MAX_ATTEMPTS:
            continue
        if count > 0 and (int(tick) - last) < _OSS_RETRY_COOLDOWN:
            continue
        title = (getattr(iss, "title", "") or "").strip()
        desc = (getattr(iss, "description", "") or "").strip()
        acc = (f"{title}. {desc}".strip(". ") if desc else title)
        out.append({"issue_id": iid, "artifact_ids": comp_arts, "title": title,
                    "component": getattr(iss, "topic", ""),
                    "severity": getattr(iss, "severity", "medium"), "attempts": count,
                    "source": "agent_created", "acceptance": acc[:400]})
        if len(out) >= 5:
            break
    return out


def unpatched_coding_issues(world: Any) -> List[Dict[str, Any]]:
    """Coding backlog the org still needs to fix — the *visible* signal a candidate generator / guard
    may act on (never references hidden tests or the future fix). Includes:
      • OSS public historical issues that map to a code module and are NOT yet genuinely resolved, and
      • agent-created issues (``create_issue``) that map to a code module (Fix #2).
    An issue stays here across attempts (Fix #1: a bare patch no longer retires it) with a cooldown +
    attempt cap so the org keeps working the fix without hammering one file every tick."""
    if not is_oss_substrate(world):
        return []
    arts = getattr(world, "product_artifacts", {}) or {}
    tick = int(getattr(world, "world_tick", 0) or 0)
    out: List[Dict[str, Any]] = []
    for entry in (world.__dict__.get("_oss_issue_stream") or []):
        iid = entry.get("issue_id")
        a = arts.get(iid)
        if a is None or getattr(a, "artifact_type", "") != "issue":
            continue
        if _oss_issue_done(world, iid):            # Fix #1: retire only on genuine resolution
            continue
        comp_arts = _component_artifact_ids(world, iid)
        if not comp_arts:
            continue
        created = int(getattr(a, "created_at_tick", 0) or 0)
        count, last = _oss_patch_attempts(world, iid, comp_arts, created)
        stalled = _oss_attempts_that_changed_nothing(world, comp_arts)
        # Retire what has stopped moving, not what is taking a while. An issue is
        # dropped from the active backlog when the gate has repeated itself for a
        # stretch, or on an absolute ceiling that only unbounded churn reaches.
        # It is also closed on the artifact, because an issue left open at high
        # severity blocks the release gate forever, and one unfixable bug must not
        # halt every release — a "wontfix" is a decision an organization can live
        # with, an indefinite block is not.
        stuck = stalled >= _OSS_STALL_LIMIT or count >= _OSS_MAX_ATTEMPTS
        if stuck and getattr(a, "status", "open") != "closed":
            # Closing is about the release gate: one unfixable bug must not block
            # every release forever. It is not a statement that the work is over,
            # and it used to be read as one -- the issue left the backlog with it.
            a.status = "closed"
            a.issue_status = "deferred"
            a.close_reason = (
                f"auto-deferred: {stalled} attempts left the build saying the same thing"
                if stalled >= _OSS_STALL_LIMIT else
                f"auto-deferred: {count} attempts without passing tests")
        if count > 0 and (tick - last) < _OSS_RETRY_COOLDOWN:   # waiting for the repo/eval cycle
            continue
        # A seeded issue is the work this pack exists to have done, and it stays
        # on offer however long it takes. Dropping it when the gate repeated
        # itself removed the work definition exactly where the work was hardest:
        # on a five-step greenfield pack all five left the backlog by t144 with
        # step 3 never once correct, and the members spent the remaining two
        # hundred ticks spreading edits over four modules nobody had asked them
        # to touch, while governance grew from a tenth of their actions to a
        # fifth. Being stuck is a reason to rank it below work that is moving,
        # not a reason to stop mentioning it.
        out.append({"issue_id": iid, "artifact_ids": comp_arts,
                    "title": getattr(a, "title", ""), "component": entry.get("component", ""),
                    "severity": entry.get("severity", "medium"), "attempts": count,
                    "source": "oss_stream", "stalled": stuck,
                    # agent-visible behavioral acceptance (the issue's problem text incl the
                    # acceptance hint) — concrete EDIT GOAL for the code editor. NOT the hidden test.
                    "acceptance": (getattr(a, "problem", "") or getattr(a, "summary", "") or "")})
    out.extend(_agent_created_coding_issues(world, tick))
    # high-severity real bugs first (visible info; no hidden-test leakage) — review fix §4,
    # and work that is still moving ahead of work that has stopped, so a hard
    # issue neither hogs every tick nor disappears.
    _rank = {"critical": 0, "major": 1, "medium": 2, "minor": 3}
    out.sort(key=lambda it: (bool(it.get("stalled")), _rank.get(it.get("severity"), 2)))
    return out


def oss_issue_resolved(world: Any, issue_id: str, *, hidden_result: Optional[Dict[str, Any]] = None) -> bool:
    """Ticket/issue closure rule (brief §9.3): an issue is resolved ONLY when BOTH hold —
    (a) test evidence: the issue's hidden test passes now (``hidden_result.issue_fix[issue_id]``);
    (b) post-issue work: a task/patch/PR linked to the issue/component advanced after it was created.
    A stale patch alone (no post-issue work, no passing test) can NEVER resolve it."""
    if not hidden_result:
        return False
    if (hidden_result.get("issue_fix") or {}).get(issue_id) is not True:
        return False
    return _has_post_issue_work(world, issue_id)


def close_resolved_oss_tickets(world: Any, *, hidden_result: Optional[Dict[str, Any]] = None) -> List[str]:
    """Close only the OSS tickets whose issue is genuinely resolved (§9.3). Returns closed ids."""
    tickets = world.__dict__.get("tickets") or {}
    closed: List[str] = []
    for tid, t in tickets.items():
        if not str(tid).startswith("ticket_oss_"):
            continue
        status = getattr(t, "status", None) if not isinstance(t, dict) else t.get("status")
        if status not in ("open", "in_progress", None):
            continue
        issue_id = str(tid)[len("ticket_oss_"):]
        if oss_issue_resolved(world, issue_id, hidden_result=hidden_result):
            if isinstance(t, dict):
                t["status"] = "resolved"
            else:
                setattr(t, "status", "resolved")
                if hasattr(t, "response_status"):
                    setattr(t, "response_status", "resolved")
            closed.append(tid)
    return closed


__all__ = ["release_due_issues", "oss_issue_resolved", "close_resolved_oss_tickets",
           "unpatched_coding_issues"]
