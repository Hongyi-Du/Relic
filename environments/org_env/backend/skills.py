"""Company-skill detector (Internal Pipeline Completion spec #8).

A "company skill" is a mechanism the org actually institutionalised — NOT a protocol
or tool that was merely created. It counts only when the mechanism is adopted/active
AND has been used (or enforced) enough to shape behaviour, with inspectable evidence.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

MIN_USE_COUNT = 2


@dataclass
class CompanySkill:
    skill_id: str
    skill_type: str                                  # protocol | tool
    reason: str = ""
    source_protocol_id: Optional[str] = None
    source_tool_id: Optional[str] = None
    source_artifact_id: Optional[str] = None
    created_tick: int = 0
    adopted_tick: Optional[int] = None
    use_count: int = 0
    enforcement_count: int = 0
    violation_count: int = 0
    affected_task_ids: List[str] = field(default_factory=list)
    affected_issue_ids: List[str] = field(default_factory=list)
    affected_artifact_ids: List[str] = field(default_factory=list)
    evidence_event_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {k: (list(v) if isinstance(v, list) else v) for k, v in self.__dict__.items()}


def _has_lineage(s) -> bool:
    """v11 hard invariant: a company skill must be event-grounded — the protocol/tool it is
    built from MUST trace to a wish or an episode (source_wish_id / source_episode_id(s)).
    An adopted protocol/tool with no provenance never counts as institutionalised."""
    return bool(getattr(s, "source_wish_id", None) or getattr(s, "source_episode_id", None)
                or getattr(s, "source_episode_ids", None))


def _distinct_outputs(art_ids, arts) -> int:
    """How many DISTINCT (normalized) contents the affected artifacts collapse to. v14c: a tool
    that produced N near-identical files (e.g. 14 copies of the same report template, each just
    'Clarified the document around: the workflow is underspecified.') is artifact SPAM, not an
    institutionalised capability — `use_count` high but real output ~1."""
    seen = set()
    for aid in art_ids:
        a = arts.get(aid)
        if a is None:
            continue
        c = getattr(a, "content", "") or getattr(a, "summary", "") or ""
        norm = " ".join(str(c).lower().split())[:400]
        if norm:
            seen.add(norm)
    return len(seen)


def _classify_obj(obj, art_set, issue_set, task_set, arts) -> None:
    if not isinstance(obj, str):
        return
    a = arts.get(obj)
    if a is not None and getattr(a, "artifact_type", "") == "issue":
        issue_set.add(obj)
    elif obj.startswith("issue") or obj.startswith("rel_blocker") or obj.startswith("proto_violation"):
        issue_set.add(obj)
    elif obj.startswith("task"):
        task_set.add(obj)
    elif obj.startswith("art_") or a is not None:
        art_set.add(obj)


def detect_company_skills(world: Any) -> List[Dict[str, Any]]:
    """Adopted protocols (used/enforced >= threshold) + active tools (used >= threshold),
    each with an inspectable evidence chain: which tasks/issues/artifacts it touched and
    which events prove it (spec #8 + #3 evidence-chain follow-up)."""
    from collections import Counter
    pm = getattr(world, "proposal_manager", None)
    if pm is None:
        return []
    arts = getattr(world, "product_artifacts", {}) or {}
    events = getattr(world, "events", []) or []
    tool_uses = Counter(e.get("tool_id") for e in events
                        if e.get("type") == "tool_use_event" and e.get("tool_id"))
    # group protocol/tool lifecycle events by id (for the evidence chain)
    proto_ev: Dict[str, list] = {}
    tool_ev: Dict[str, list] = {}
    for i, e in enumerate(events):
        et = e.get("type")
        if et in ("protocol_use_event", "protocol_enforcement_event", "protocol_violation_event",
                  "protocol_revision_event") and e.get("protocol_id"):
            proto_ev.setdefault(e["protocol_id"], []).append((i, e))
        elif et == "tool_use_event" and e.get("tool_id"):
            tool_ev.setdefault(e["tool_id"], []).append((i, e))

    skills: List[Dict[str, Any]] = []
    for s in pm.protocol_specs.values():
        if s.status != "adopted":
            continue
        if not _has_lineage(s):                          # v11: no wish/episode provenance -> not a skill
            continue
        adopted_tick = getattr(s, "adopted_at_tick", None)
        # v8g P2: counts are derived from the canonical post-adoption event stream, so the
        # skill's "used Nx / enforced Mx" ALWAYS matches its evidence_event_ids — one
        # consistent number across protocol_specs / internal.protocols / company_skills.
        pev = [(i, e) for i, e in proto_ev.get(s.protocol_id, [])
               if adopted_tick is None or int(e.get("tick", 0) or 0) >= int(adopted_tick)]
        uses = sum(1 for _, e in pev if e.get("type") == "protocol_use_event")
        enf = sum(1 for _, e in pev if e.get("type") == "protocol_enforcement_event")
        if not ((uses + enf) >= MIN_USE_COUNT or (uses >= 1 and enf >= 1)):
            continue
        art_set, issue_set, task_set, ev_ids = set(getattr(s, "affected_artifacts", []) or []), set(), \
            set(getattr(s, "affected_task_ids", []) or []), []
        for i, e in pev:
            ev_ids.append(f"{e['type']}@t{e.get('tick')}")
            for o in (e.get("object_id"), e.get("follow_up_issue"), e.get("action_id")):
                _classify_obj(o, art_set, issue_set, task_set, arts)
        for aid in list(art_set):                       # an affected artifact's tasks/issues count too
            a = arts.get(aid)
            if a is not None:
                task_set.update(getattr(a, "linked_task_ids", []) or [])
        skills.append(CompanySkill(
            skill_id=f"skill_{s.protocol_id}", skill_type="protocol",
            reason=f"adopted protocol used {uses}x, enforced {enf}x (rev {getattr(s, 'revision', 0)})",
            source_protocol_id=s.protocol_id, created_tick=int(getattr(s, "created_at_tick", 0) or 0),
            adopted_tick=adopted_tick,
            use_count=uses, enforcement_count=enf, violation_count=int(getattr(s, "violation_count", 0) or 0),
            affected_task_ids=sorted(task_set), affected_issue_ids=sorted(issue_set),
            affected_artifact_ids=sorted(art_set), evidence_event_ids=ev_ids[:10]).to_dict())
    for t in pm.tools.values():
        if getattr(t, "status", "") != "active":
            continue
        if not _has_lineage(t):                          # v11: no wish/episode provenance -> not a skill
            continue
        uses = int(tool_uses.get(t.tool_id, 0))
        if uses < MIN_USE_COUNT:
            continue
        art_set, issue_set, task_set, ev_ids = set(), set(), set(), []
        for i, e in tool_ev.get(t.tool_id, []):
            ev_ids.append(f"tool_use_event@t{e.get('tick')}")
            for o in [e.get("object_id"), e.get("artifact_id")] + list(e.get("affected", []) or []):
                _classify_obj(o, art_set, issue_set, task_set, arts)
        # v14c: a tool whose many uses only spawned near-identical artifacts is artifact SPAM, not a
        # company skill — `use_count=14` over 14 copies of the same template is not a capability.
        distinct = _distinct_outputs(art_set, arts)
        if len(art_set) >= 3 and distinct <= 1:
            continue
        # #6: capability = adoption + repeated use + real DOWNSTREAM EFFECT. A tool invoked N times
        # that touched NO artifacts/issues/tasks (0 distinct outputs) is an adopted-tool-with-repeated-
        # invocation, not an organization-level capability — exclude it from company_skills.
        if not art_set and not issue_set and not task_set:
            continue
        for aid_ in list(art_set):                      # an affected artifact's tasks count too
            a = arts.get(aid_)
            if a is not None:
                task_set.update(getattr(a, "linked_task_ids", []) or [])
        skills.append(CompanySkill(
            skill_id=f"skill_{t.tool_id}", skill_type="tool",
            reason=f"active tool used {uses}x ({distinct} distinct outputs)",
            source_tool_id=t.tool_id, use_count=uses, affected_task_ids=sorted(task_set),
            affected_issue_ids=sorted(issue_set), affected_artifact_ids=sorted(art_set),
            evidence_event_ids=ev_ids[:10]).to_dict())
    return skills


__all__ = ["CompanySkill", "detect_company_skills", "MIN_USE_COUNT"]
