"""Make an adopted protocol bind the action space instead of annotating it.

Today a protocol is a ledger. `ProtocolRegistry.enforce` appends an event;
`world.note_protocol_enforcement` increments a counter and is called with
`blocked=False`. The clearest case is an unreviewed merge: `_h_merge_pr` takes
the `force = (not pr.reviewed)` branch, the repository **is merged**, and only
afterwards is a violation and a paired "enforcement" recorded. So a run
reporting `enforcements: 2` is reporting two labels applied to things that
already happened, not two actions prevented.

Under `protocol_masking_enabled` an adopted protocol instead removes the
candidates it forbids, before either the LLM or the profile policy sees the
menu. The agent keeps its choice; the choice set is what the institution
constrains.

WHY A REGISTRY AND NOT A GENERAL RULE. Protocols here are emergent: they are
written at runtime by agents, carry ids like `proto_spec_1`, and state their
rule in `enforcement_rule` / `violation_condition` as free text. No general
function can decide whether an arbitrary candidate violates arbitrary English.
What *is* machine-readable is which actions a protocol claims to govern
(`ProtocolSpec.declared_action_ids`), and the keyword families the codebase
already uses to credit uses and enforcements. So a protocol becomes binding
only when it matches a family whose precondition can actually be checked
against world state. Everything else keeps today's advisory behaviour, which
means adding a family is the only way to make more of the institution binding —
deliberately, so that "the protocol changed what agents could do" is never
claimed on the strength of text nobody verified.

WHAT THIS COSTS. A hard mask makes violations of a bound family impossible, so
its violation count is 0 by construction and can no longer be read as evidence
of anything. The prevented-action ledger below replaces it: an institution that
binds is one whose gates fire, and each firing names the protocol, the agent,
the action and the reason.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

# Keyword families, matched against a spec's name/trigger/enforcement blob.
# These mirror the tuples the merge path already passes to
# note_protocol_use / note_protocol_enforcement, so a protocol that is credited
# with governing merges is the same protocol that binds them.
REVIEW_BEFORE_MERGE_KEYWORDS = (
    "review before merge",
    "code review",
    "pr review",
    "peer review",
    "unreviewed",
    "force merge",
    "force-merge",
    "merge approval",
)


def _pull_request(world: Any, candidate: Any) -> Any:
    pr_id = (getattr(candidate, "parameters", None) or {}).get("pr_id")
    if not pr_id:
        return None
    repo_system = getattr(world, "repo_system", None)
    repo = getattr(repo_system, "repo", None)
    return (getattr(repo, "pull_requests", {}) or {}).get(pr_id)


def _unreviewed_merge(candidate: Any, agent: Any, world: Any) -> Optional[str]:
    """Forbid merging a pull request nobody has reviewed.

    A missing pull request is not this rule's business: the candidate is either
    malformed or refers to something already gone, and both are the repository
    layer's to refuse. Only a real, unreviewed pull request is a violation.
    """
    pull_request = _pull_request(world, candidate)
    if pull_request is None:
        return None
    if bool(getattr(pull_request, "reviewed", False)):
        return None
    return f"{getattr(pull_request, 'pr_id', 'the pull request')} has had no review"


@dataclass(frozen=True)
class StructuralGate:
    """A protocol family whose precondition can be checked against the world.

    ``violates`` returns why this candidate breaks the rule, or None. It is
    consulted only for actions in ``governed_actions``, and only once a
    protocol matching ``keywords`` (or declaring one of those actions) has been
    adopted and is live.
    """

    family: str
    keywords: tuple[str, ...]
    governed_actions: frozenset[str]
    violates: Callable[[Any, Any, Any], Optional[str]]


STRUCTURAL_GATES: tuple[StructuralGate, ...] = (
    StructuralGate(
        family="review_before_merge",
        keywords=REVIEW_BEFORE_MERGE_KEYWORDS,
        governed_actions=frozenset({"merge_pr"}),
        violates=_unreviewed_merge,
    ),
)


def _spec_blob(spec: Any) -> str:
    return " ".join(
        str(getattr(spec, field, "") or "")
        for field in ("name", "trigger_condition", "enforcement_rule",
                      "violation_condition")
    ).lower()


def _declared_actions(spec: Any) -> set:
    declared = getattr(spec, "declared_action_ids", None)
    if callable(declared):
        try:
            return set(declared())
        except Exception:  # noqa: BLE001 - a malformed spec must not stop the tick
            return set()
    return set()


def _live_adopted_specs(world: Any):
    """Adopted protocol specs whose registry mirror has not gone terminal.

    Reuses world's own liveness check so a spec the executable registry has
    already rejected cannot bind the action space either — the same fail-closed
    rule that governs whether it may accrue a use.
    """
    manager = getattr(world, "proposal_manager", None)
    specs = getattr(manager, "protocol_specs", None) or {}
    is_live = getattr(world, "_protocol_mirror_is_live_or_absent", None)
    for spec in specs.values():
        if getattr(spec, "status", "") != "adopted":
            continue
        if callable(is_live):
            try:
                if not is_live(spec):
                    continue
            except Exception:  # noqa: BLE001
                continue
        yield spec


def protocol_masking_enabled(world: Any) -> bool:
    """Whether this run's condition binds the action space to its protocols."""
    condition = getattr(world, "condition_spec", None)
    return bool(getattr(condition, "protocol_masking_enabled", False))


def mask_reason(candidate: Any, agent: Any, world: Any) -> Optional[str]:
    """Why an adopted protocol forbids this candidate this tick, or None.

    The returned string names the protocol, so the prevented-action ledger can
    attribute the block to the institution that caused it rather than to the
    guard that carried it out.
    """
    if not protocol_masking_enabled(world):
        return None
    action_type = str(getattr(candidate, "action_type", "") or "")
    if not action_type:
        return None
    gates = [g for g in STRUCTURAL_GATES if action_type in g.governed_actions]
    if not gates:
        return None
    for spec in _live_adopted_specs(world):
        blob = _spec_blob(spec)
        declared = _declared_actions(spec)
        for gate in gates:
            matches = any(keyword in blob for keyword in gate.keywords) or (
                action_type in declared
            )
            if not matches:
                continue
            reason = gate.violates(candidate, agent, world)
            if reason:
                return (
                    f"{getattr(spec, 'protocol_id', gate.family)} forbids "
                    f"{action_type}: {reason}"
                )
    return None


def filter_candidates(candidates, agent_id: str, world: Any):
    """Split a pool into what the institution permits and what it forbids.

    Never empties the pool on the institution's account alone: a rule that
    leaves an agent with nothing to do has stopped being a constraint on
    behaviour and started being a halt. In that case the blocks are recorded
    but the pool is returned intact, so the run reports an institution it could
    not honour instead of silently idling every agent it governs.
    """
    kept, blocked = [], []
    for candidate in candidates:
        reason = mask_reason(candidate, world.agents.get(agent_id), world)
        if reason:
            blocked.append((candidate, reason))
        else:
            kept.append(candidate)
    if blocked and not kept:
        return list(candidates), blocked
    return kept, blocked


def record_prevented(world: Any, agent_id: str, blocked, tick: int) -> None:
    """Log each prevented action, because the violation counter can no longer.

    Under a hard mask the forbidden action never happens, so violation_count
    stays 0 for every bound family. These events are what remains as evidence
    that the institution was load-bearing.
    """
    if not blocked:
        return
    ledger = world.__dict__.setdefault("_protocol_prevented", [])
    for candidate, reason in blocked:
        entry = {
            "tick": int(tick),
            "agent_id": str(agent_id),
            "action": str(getattr(candidate, "action_type", "") or ""),
            "reason": str(reason),
        }
        ledger.append(entry)
        events = getattr(world, "events", None)
        if isinstance(events, list):
            events.append({"type": "protocol_prevented_action_event", **entry})
    del ledger[:-500]


__all__ = [
    "STRUCTURAL_GATES",
    "StructuralGate",
    "filter_candidates",
    "mask_reason",
    "protocol_masking_enabled",
    "record_prevented",
]
