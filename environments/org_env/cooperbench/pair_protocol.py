"""Peer-authored process rules over the pair adapter's explicit binding surface."""
from __future__ import annotations

import json
import hashlib
import re
from typing import Mapping


BINDING = "fresh_attestations_after_adoption"


def _pair_members(world):
    private = getattr(world, "_cooperbench_private_briefs", {}) or {}
    return {row.get("owner_id") for row in (private.get("briefs") or {}).values()
            if isinstance(row, Mapping) and row.get("owner_id")}


def _status(value):
    return str(getattr(value, "value", value) or "")


def _published_pr(world, pr_id):
    repo = getattr(getattr(world, "repo_system", None), "repo", None)
    pr = (getattr(repo, "pull_requests", {}) or {}).get(pr_id)
    if (pr is not None and getattr(pr, "author_id", None) in _pair_members(world)
            and _status(getattr(pr, "status", "")) in {
                "open", "review_requested", "changes_requested", "approved", "merged", "closed"}):
        return pr
    return None


def _friction_visibility(world, ref):
    """Return (published_to_pair, private_owner) from actual referenced objects."""
    repo = getattr(getattr(world, "repo_system", None), "repo", None)
    kind, _, object_id = ref.partition(":")
    if kind == "patch_rejected":
        patch = (getattr(world, "patches", {}) or {}).get(object_id)
        if patch is None or _status(getattr(patch, "validation_status", "")) != "rejected":
            return False, None
        published = any(
            object_id in (getattr(pr, "patch_ids", []) or []) and _published_pr(world, pr_id) is not None
            for pr_id, pr in (getattr(repo, "pull_requests", {}) or {}).items())
        return published, getattr(patch, "actor_id", None)
    if kind == "ci_failed":
        ci = (getattr(repo, "ci_runs", {}) or {}).get(object_id)
        if ci is None or _status(getattr(ci, "status", "")) != "failed":
            return False, None
        pr_id = getattr(ci, "pr_id", None)
        pr = (getattr(repo, "pull_requests", {}) or {}).get(pr_id)
        return _published_pr(world, pr_id) is not None, getattr(pr, "author_id", None)
    if kind == "review_changes":
        pr_id, _, number = object_id.rpartition(":")
        pr = (getattr(repo, "pull_requests", {}) or {}).get(pr_id)
        comments = list(getattr(pr, "requested_changes", []) or [])
        if not number.isdigit() or not 1 <= int(number) <= len(comments):
            return False, None
        return _published_pr(world, pr_id) is not None, getattr(pr, "author_id", None)
    if kind == "joint_verification_failed":
        number, _, digest = object_id.partition(":")
        failures = (getattr(world, "_cooperbench_sdl_state", {}) or {}).get("verification_failures", [])
        if number.isdigit() and 1 <= int(number) <= len(failures):
            failure = failures[int(number) - 1]
            if (isinstance(failure, Mapping) and failure.get("agent_id") in _pair_members(world)
                    and str(failure.get("digest") or "")[:12] == digest):
                return True, None
    if kind == "peer_reviewed_delivery":
        state = getattr(world, "_cooperbench_sdl_state", {}) or {}
        owner = str((state.get("feature_owners") or {}).get(object_id) or "")
        members = _pair_members(world)
        for pr_id, pr in (getattr(repo, "pull_requests", {}) or {}).items():
            linked = {
                str(item)
                for item in (getattr(pr, "linked_issue_ids", []) or [])
            }
            linked_issue = str(getattr(pr, "linked_issue", "") or "")
            if linked_issue:
                linked.add(linked_issue)
            approvers = {
                str(item) for item in (getattr(pr, "approved_by", []) or [])
            }
            if (
                object_id in linked
                and str(getattr(pr, "author_id", "") or "") == owner
                and _status(getattr(pr, "status", "")) == "merged"
                and approvers & (members - {owner})
            ):
                return _published_pr(world, pr_id) is not None, owner
    return False, None


def visible_protocol_friction(world, actor_id, evidence):
    """Resolve only actual published friction or the actor's own private failure."""
    from .lifecycle import protocol_friction_details
    from .visibility import is_strict_coop
    refs = list(dict.fromkeys(ref for ref in evidence if isinstance(ref, str)))
    if not is_strict_coop(world):
        return protocol_friction_details(world, refs)
    if actor_id not in _pair_members(world):
        return {}
    visible = []
    for ref in refs:
        published, owner = _friction_visibility(world, ref)
        if published or owner == actor_id:
            visible.append(ref)
    return protocol_friction_details(world, visible)


def common_public_protocol_friction(world, evidence, members):
    """The pair gate must not be triggered by private, uncommunicated failures."""
    from .visibility import is_strict_coop
    refs = list(dict.fromkeys(ref for ref in evidence if isinstance(ref, str)))
    if not is_strict_coop(world):
        return refs
    if set(members) != _pair_members(world) or len(set(members)) != 2:
        return []
    return [ref for ref in refs if _friction_visibility(world, ref)[0]]


def visible_protocol_events(world, actor_id, limit=24):
    """Keep own event text; other events carry only verified structural metadata.

    A raw event with no actor is not public merely because it lacks an owner.
    Public PR/CI details are resolved from their objects in friction_details,
    never from an arbitrary nested payload attached to a recent event.
    """
    from .visibility import is_strict_coop
    events = list(getattr(world, "events", []) or [])
    if not is_strict_coop(world):
        return events[-limit:]
    if actor_id not in _pair_members(world):
        return []
    result = []
    for event in events:
        if not isinstance(event, Mapping):
            continue
        if event.get("agent_id") == actor_id:
            result.append(dict(event))
            continue
        # Known structural names/numbers only. Do not carry free-form reason,
        # description, summary, edit_goal, source code, receipt or tool output.
        metadata = {key: event[key] for key in ("type", "subtype")
                    if isinstance(event.get(key), str)
                    and re.fullmatch(r"[a-z][a-z0-9_]{0,79}", event[key])}
        if not metadata.get("type"):
            continue
        if event.get("agent_id") in _pair_members(world):
            metadata["agent_id"] = event["agent_id"]
        for key in ("tick", "success", "blocked"):
            if type(event.get(key)) in (int, bool):
                metadata[key] = event[key]
        pr_id = event.get("pr_id")
        if _published_pr(world, pr_id) is not None:
            metadata["pr_id"] = pr_id
        metadata["visibility"] = "structural_metadata_only"
        result.append(metadata)
    return result[-limit:]


def draft_pair_protocol(world, agent_id, evidence, members):
    """Author content only after propose_protocol; never fall back to a template."""
    if agent_id not in members:
        raise ValueError("cooperbench_protocol_proposer_not_in_pair")
    client = getattr(world, "llm_client", None)
    if client is None:
        raise ValueError("cooperbench_protocol_author_unavailable")
    from .lifecycle import _redact
    from .visibility import is_strict_coop
    details = visible_protocol_friction(world, agent_id, evidence)
    if is_strict_coop(world):
        evidence = list(details)
        if len(evidence) < 2:
            raise ValueError("cooperbench_protocol_requires_two_visible_friction_refs")
    system = (
        "You are a member of a two-person organization who selected propose_protocol after repeated "
        "public delivery evidence. Write your own concise rule and rationale grounded in the supplied "
        "evidence. The adapter already requires owner delivery, current-head CI, reciprocal review and "
        "same-digest public verification; do not claim to invent those baseline gates. The currently "
        "supported additional binding is fresh_attestations_after_adoption: both members must renew "
        "their verification after adoption, so earlier attestations cannot freeze the delivery. "
        "Evidence may be repeated workflow failures or two completed peer-reviewed feature cycles. "
        "Explain why this binding addresses the observed evidence, or set applicable=false and give "
        "a concrete target_problem and summary explaining why it does not. A reasoned non-applicability "
        "decision is valid; do not invent a rule just to complete the task. Do not invent evidence, "
        "a benchmark solution, hidden tests, or unsupported executable "
        "actions. Your partner will explicitly review the proposal. Return JSON only."
    )
    answer = client.generate_json(system, json.dumps({
        "agent_id": agent_id, "members": members, "evidence_ids": evidence,
        "friction_details": _redact(details),
        "recent_public_events": _redact(visible_protocol_events(world, agent_id)),
        "supported_binding": BINDING,
    }, ensure_ascii=False), {
        "applicable": "boolean", "binding": BINDING, "title": "string",
        "summary": "string", "target_problem": "string", "proposed_solution": "string",
        "source_event_ids": ["at least two distinct supplied evidence ids"],
    }, max_tokens=1800)
    if not isinstance(answer, Mapping) or type(answer.get("applicable")) is not bool:
        raise ValueError("cooperbench_protocol_applicability_invalid")
    refs = answer.get("source_event_ids")
    if (not isinstance(refs, list)
            or any(not isinstance(i, str) or i not in evidence for i in refs)
            or len(set(refs)) < 2):
        raise ValueError("cooperbench_protocol_binding_or_evidence_invalid")
    if answer["applicable"] and answer.get("binding") != BINDING:
        raise ValueError("cooperbench_protocol_binding_or_evidence_invalid")
    required = ("title", "summary", "target_problem", "proposed_solution") if answer["applicable"] else ("summary", "target_problem")
    for key in required:
        if not isinstance(answer.get(key), str) or not 1 <= len(answer[key].strip()) <= 1600:
            raise ValueError("cooperbench_protocol_authored_content_invalid")
    return dict(answer)


def authored_bindings(world, specs):
    declarations = (getattr(world, "_cooperbench_sdl_state", {}) or {}).get("protocol_authorship", {})
    matched = []
    for spec in specs:
        declaration = declarations.get(str(getattr(spec, "created_from_proposal_id", "") or ""), {})
        digest = hashlib.sha256(str(getattr(spec, "enforcement_rule", "") or "").encode()).hexdigest()
        if (declaration.get("binding") == BINDING
                and declaration.get("origin") == "agent_authored_bounded_binding"
                and declaration.get("solution_digest") == digest):
            matched.append(spec)
    return matched
