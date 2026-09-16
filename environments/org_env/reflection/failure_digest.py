"""What has actually been failing, in the words the gate used, with a shape.

An organization that cannot institutionalize is not the problem this answers. One
that institutionalizes the wrong thing is. Measured over a run: 46 reflections and
35 wishes, and the four things blocking every remaining step of the pack --
UploadConflictError, GCPlan, ReplicaStatus, Manifest.__init__ -- were named zero
times, while "evidence" appeared 46 times and "matrix" 27. The organization then
adopted four rules about recording evidence and one about code.

The complaint existed. It sat in the gate's own verdict and in each pull request's
brief, spelled out to the parameter -- and nothing carried it to the agent doing
the diagnosing. What reflection saw of a failure was the verb and the target: an
edit that failed, a CI run that failed. It diagnosed from that, rationally, and
got paperwork.

So the verdicts travel, verbatim, with a count of each kind beside them. No
summary, no ranking, no advice about what to do: an organization that is told what
its problem is has not diagnosed anything. What is added is the evidence it was
already producing and could not see.

"""
from __future__ import annotations

import collections
from typing import Any, Dict, List

# How much of the recent past one reflection is shown. Long enough for a repeat
# to look like a repeat, short enough that a fixed failure stops being news.
_WINDOW_TICKS = 48
# Room for every step a stacked pack can fail at once, plus the other kinds.
_MAX_VERBATIM = 12
_MAX_CHARS = 320


def _recent(world: Any, tick: int) -> int:
    return max(0, int(tick) - _WINDOW_TICKS)


def _clean(text: Any) -> str:
    return " ".join(str(text or "").split())[:_MAX_CHARS]


def _action_failures(world: Any, since: int) -> List[Dict[str, str]]:
    rows = [r for r in (list(getattr(world, "baseline_archived_action_log", []) or [])
                        + list(getattr(world, "action_log", []) or []))
            if isinstance(r, dict) and int(r.get("tick") or 0) >= since
            and not r.get("success", True)]
    out = []
    for r in rows:
        why = _clean(r.get("failure_reason") or r.get("reason"))
        if why:
            out.append({"kind": "action refused", "tick": int(r.get("tick") or 0),
                        "what": f"{r.get('action_type')} on {r.get('target') or '-'}",
                        "said": why})
    return out


def _failing_steps(world: Any) -> List[str]:
    """Every step the gate reported, not only the one the brief kept.

    A pack whose steps stack fails several at once and says so a line at a time.
    `_build_error` is one of those lines, so an organization reading it saw one
    of four blockers: it heard that Manifest.__init__ was wrong and never that
    UploadConflictError, ReplicaStatus and GCPlan were missing too.
    """
    detail = str(getattr(world, "_build_error_detail", "") or "")
    lines = [_clean(ln) for ln in detail.splitlines() if ln.strip()]
    # Keep the lines that name a step's own verdict; a traceback frame is noise
    # to someone deciding what rule the organization needs.
    steps = [ln for ln in lines if "step " in ln.lower() and "fail" in ln.lower()]
    if steps:
        return list(dict.fromkeys(steps))
    single = _clean(getattr(world, "_build_error", ""))
    return [single] if single else []


def _gate_verdicts(world: Any, since: int) -> List[Dict[str, str]]:
    """The build gate's own words, which name the symbol and the parameters."""
    out = []
    for said in _failing_steps(world):
        out.append({"kind": "contract check", "tick": int(getattr(world, "world_tick", 0)),
                    "what": "the product's own smoke gate", "said": said})
    repo = getattr(getattr(world, "repo_system", None), "repo", None)
    for pr in (getattr(repo, "pull_requests", {}) or {}).values():
        if getattr(pr, "merged_tick", None) is not None:
            continue
        said = _clean(getattr(pr, "ci_brief", "") or getattr(pr, "ci_failure_reason", ""))
        if said:
            out.append({"kind": "pull request not merged",
                        "tick": int(getattr(pr, "_last_ci_tick", 0) or 0),
                        "what": f"{getattr(pr, 'pr_id', 'a pull request')} "
                                f"({'reviewed' if getattr(pr, 'reviewed', False) else 'unreviewed'}, "
                                f"ci {'passed' if getattr(pr, 'ci_passed', False) else 'failing'})",
                        "said": said})
    return out


def _patch_rejections(world: Any, since: int) -> List[Dict[str, str]]:
    out = []
    for patch in (getattr(world, "patches", {}) or {}).values():
        if str(getattr(patch, "validation_status", "")) != "rejected":
            continue
        if int(getattr(patch, "tick", 0) or 0) < since:
            continue
        said = _clean(getattr(patch, "rejection_reason", ""))
        if said:
            out.append({"kind": "change rejected", "tick": int(getattr(patch, "tick", 0) or 0),
                        "what": str(getattr(patch, "target_object_id", "") or "-"),
                        "said": said})
    return out


def _refused_by_its_own_rules(world: Any, since: int) -> List[Dict[str, str]]:
    """Work the organization's own adopted rules turned away.

    A rule can be the thing that is failing. One run adopted a rule whose text
    opened "Propose a protocol backed by one reusable review checklist", and the
    reviewer then held code patches to it: fifteen refusals telling an edit of
    blobstore/manifests.py to add a review checklist, which no edit of that file
    can do. Each author rewrote three times against an unanswerable demand and
    the change was dropped.

    Nothing repairs that here. policy_repair_need already exists for a rule that
    has become the problem, and the organization is the one that has to decide
    this rule is. It could not decide it while its own refusals were the one kind
    of failure the digest did not carry.
    """
    out = []
    for e in (getattr(world, "events", []) or []):
        if not isinstance(e, dict) or e.get("subtype") != "patch_refused_by_protocol":
            continue
        if int(e.get("tick") or 0) < since:
            continue
        said = _clean(e.get("reason"))
        if not said:
            continue
        out.append({"kind": "refused by our own rule",
                    "tick": int(e.get("tick") or 0),
                    "what": f"{e.get('artifact_id') or '-'} under "
                            f"{e.get('protocol_id') or 'an adopted rule'} "
                            f"(rewrite {e.get('attempt') or 1})",
                    "said": said})
    return out


def recent_failure_digest(world: Any, agent_id: str = "") -> Dict[str, Any]:
    """Recent failures verbatim, and how the recent ones divide by kind."""
    tick = int(getattr(world, "world_tick", 0) or 0)
    since = _recent(world, tick)
    items = (_gate_verdicts(world, since) + _action_failures(world, since)
             + _patch_rejections(world, since)
             + _refused_by_its_own_rules(world, since))
    if not items:
        return {}
    items.sort(key=lambda i: -i["tick"])

    counts = collections.Counter(i["kind"] for i in items)
    total = sum(counts.values())
    # The share matters as much as the text: one loud failure and a hundred quiet
    # ones of another kind read very differently, and a count alone hides that.
    shape = [f"{kind}: {n} of {total} ({round(100 * n / total)}%)"
             for kind, n in counts.most_common()]

    # Round-robin the kinds rather than taking the most recent outright. A single
    # loud kind fills every slot otherwise: a run whose own rules refused a change
    # forty-five times would have shown twelve refusals and none of the contract
    # errors that were the reason to edit anything, which is the diagnosis this
    # exists to prevent, arriving from the other direction.
    by_kind: Dict[str, List[Dict[str, str]]] = collections.defaultdict(list)
    seen = set()
    for item in items:
        if item["said"] in seen:
            continue
        seen.add(item["said"])
        by_kind[item["kind"]].append(item)

    verbatim: List[str] = []
    queues = [by_kind[k] for k, _ in counts.most_common()]
    while len(verbatim) < _MAX_VERBATIM and any(queues):
        for q in queues:
            if not q:
                continue
            item = q.pop(0)
            verbatim.append(
                f"t{item['tick']} [{item['kind']}] {item['what']}: {item['said']}")
            if len(verbatim) >= _MAX_VERBATIM:
                break
    return {"window_ticks": _WINDOW_TICKS, "failures_by_kind": shape,
            "what_they_said": verbatim}


__all__ = ["recent_failure_digest"]
