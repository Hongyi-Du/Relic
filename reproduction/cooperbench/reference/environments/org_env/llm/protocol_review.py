"""Check a finished change against the rules the organization adopted for itself.

An institution that only exists as a registry row and a counter cannot be told
apart from one that governs the work. Measured on a five-step Pack: an arm adopted
"any change to a module other code depends on must preserve its callable surface —
names and each function's parameter names in order — exactly as the published
interface declares it", carried it for a hundred ticks with seven of nine members
supporting it and 118 recorded enforcements, and wrote code that dropped a
required class, renamed a public method to a private one and left out a declared
parameter. The rule was never in front of anyone writing code: the editor's prompt
had no notion of a protocol, and the decision context named the protocol by id
without its text.

So the rules travel to the editor before it writes, and one review reads the
finished change against them. This is the organization's own quality control
rather than the environment's: with nothing adopted there is nothing to check,
which is what makes an arm that institutionalizes differ from one that does not.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from environments.org_env.llm.client import LLMError, OrgLLMClient

REVIEW_SYSTEM = (
    "You check one finished code change against rules this organization adopted "
    "for itself. Judge only against the rules given. A change that does not touch "
    "what a rule governs complies with it. "
    "You are reading the change alone: no review, signoff, approval, test run or "
    "merge has happened yet, and none of them can be judged from here. A rule that "
    "also demands one of those is met, for your purposes, by whatever it says the "
    "code must be. Never refuse a change for lacking a review, a signoff, an "
    "approval or a passing test run. "
    "`rule_number` is the number of the rule you refused under, or 0 when the "
    "change complies. `offending_symbol` is the function, class, parameter or file "
    "in the change that breaks the rule, and is empty when it complies. In `reason` "
    "say what that symbol must become — the author will revise and resubmit against "
    "your words and can change nothing else. "
    "Reply JSON only."
)

REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "complies": {"type": "boolean"},
        # The rule is asked for as a number the schema requires, not as prose to
        # be parsed back. Reading it out of a quoted line failed on the shapes
        # that actually arrived — "3. A change to a dependency module…" and
        # 'Rule 3: "A change to…"' — and 170 of 199 refusals were credited to no
        # rule at all, leaving the enforcement count of the rule doing the work
        # wrong. Any parser here is a guess about wording; a required field is not.
        "rule_number": {"type": "integer",
                        "description": "the rule refused under, 0 when it complies"},
        # A refusal has to point at something the author can edit. Without this,
        # 111 of 208 refusals in one run read "obtain reviewer signoff before
        # merge", which no rewrite of a patch can ever produce: the loop spent
        # its three revisions and dropped the change, 35 times over, with the
        # code unchanged at the end of 312 ticks.
        "offending_symbol": {
            "type": "string",
            "description": "the function, class, parameter or file in the change "
                           "that breaks the rule; empty when it complies"},
        "reason": {"type": "string"},
    },
    "required": ["complies", "rule_number", "offending_symbol"],
}

# How much of a change one review reads. A whole large module would crowd out the
# rules themselves, and what a rule about a public surface needs to see is the
# surface, which is what a diff shows.
_MAX_DIFF_CHARS = 12000


# Protocols the environment seeds and runs a gate for itself. The merge gate
# already refuses an unreviewed merge; the tracker already records an experiment.
# Handing them to this review as well was 78 of 103 refusals in one run: a patch
# on a branch merges nothing, so the refusal named a rule the change could not
# have broken, and every one of those changes was dropped after exhausting its
# rewrites against a rule no rewrite could satisfy. It also counted one act of
# enforcement twice, in a number the capability claim rests on.
#
# Only ids this codebase itself writes belong here, each checkable by grep. A
# protocol the agents invent is deliberately absent: nothing else enforces it,
# so this review is the whole of its bite, which is what makes an arm that
# institutionalizes differ from one that does not.
_GATED_BY_THE_ENVIRONMENT = frozenset({
    "proto_review_before_merge",
    "proto_experiment_logging",
})


def adopted_rules(world: Any, *, action_type: Optional[str] = None) -> List[Tuple[str, str, str]]:
    """(protocol_id, protocol_type, rule) for each adopted rule this review governs.

    Adopted, carrying a rule to judge by, and not already enforced by a gate of
    the environment's own.
    """
    from environments.org_env.experiments.capability_transfer import inherited_base_id
    from environments.org_env.backend.protocol.registry import protocol_is_live

    registry = getattr(world, "protocol_registry", None)
    specs_by_id = {}
    if action_type is not None:
        manager = getattr(world, "proposal_manager", None)
        for spec in (getattr(manager, "protocol_specs", {}) or {}).values():
            spec_id = str(getattr(spec, "protocol_id", ""))
            specs_by_id[spec_id] = spec
            specs_by_id[f"proto_spec_{spec_id.split('_')[-1]}"] = spec
    out: List[Tuple[str, str, str]] = []
    for pid, protocol in (getattr(registry, "protocols", {}) or {}).items():
        if (
            not protocol_is_live(protocol)
            or str(getattr(protocol, "adoption_status", "")) != "adopted"
        ):
            continue
        # A transferred copy of a rule this codebase already gates is the same
        # rule wearing a namespace, and reviewing it here would enforce it a
        # second time -- in exactly the arms whose enforcement counts the
        # transfer contrast is read from.
        if inherited_base_id(pid) in _GATED_BY_THE_ENVIRONMENT:
            continue
        # Local specs and their registry mirrors share identity. An inherited
        # rule can have the same numeric suffix but is a different institution.
        spec = specs_by_id.get(str(pid))
        if spec is not None and getattr(spec, "affected_actions", None):
            from environments.org_env.proposals.objects import ProtocolSpec

            declared = ProtocolSpec.declared_action_ids(spec)
            if declared and action_type not in declared:
                continue
        rule = str(getattr(protocol, "rule_summary", "") or "").strip()
        if rule:
            out.append((str(pid), str(getattr(protocol, "protocol_type", "") or ""), rule))
    return out


def rules_block(world: Any) -> str:
    """How this organization works, as its members read it, or "" if it has none.

    One block, one heading, one position, whether the rules are executable or
    inherited as prose. The transfer design turns on a contrast between a rule
    that binds and a rule that is only known, and that contrast survives only
    if the two arms differ in the binding and in nothing else -- not in whether
    the rule is in front of them, and not in what the prompt says about it.

    The heading used to add "your change is reviewed against these before it
    lands", which is true where a registry enforces them and false where the
    organization has only been told about them. Keeping it would have handed
    one arm a statement the other could not be given, so it is in neither.
    """
    entries = [rule for _pid, _type, rule in adopted_rules(world)]
    inherited: list[str] = []
    try:
        from environments.org_env.experiments.capability_transfer import (
            inherited_capability_texts,
            note_inherited_prose_shown,
        )

        inherited = inherited_capability_texts(world)
        if inherited:
            note_inherited_prose_shown(world)
    except Exception:
        inherited = []
    entries.extend(inherited)
    if not entries:
        return ""
    lines = "\n".join(f"- {entry}" for entry in entries)
    return f"\nHOW THIS ORGANIZATION WORKS:\n{lines}\n"


def enforcement_keywords(protocol_type: str, rule: str) -> tuple:
    """Words that identify this protocol to the world's enforcement bookkeeping.

    Taken from the protocol's own type, which is derived from its name, so an
    enforcement is credited to the rule that actually refused rather than to
    whichever adopted protocol happens to share a common word.
    """
    words = [w for w in str(protocol_type or "").split("_")
             if len(w) > 3 and w not in ("protocol", "spec")]
    return tuple(words) or ("contract",)


def _which_rule(rules: List[Tuple[str, str, str]], answer: Dict[str, Any]):
    """The rule a refusal was made under, taken from the field that carries it.

    The number is what the schema requires, so nothing here interprets prose. An
    organization carrying one rule needs no number to be unambiguous; carrying
    several, a refusal that names none cannot be credited to any of them.
    """
    number = answer.get("rule_number")
    if isinstance(number, int) and 1 <= number <= len(rules):
        return rules[number - 1]
    return rules[0] if len(rules) == 1 else None


def _merge_only_test_rule(rule: str) -> bool:
    """A PR opening/approval gate cannot constrain an intermediate source edit.

    Keep genuinely pre-edit rules in scope, even when they also mention PRs.
    This only classifies explicit tests-before-PR/merge forms; the
    rule remains visible to the author and to actions it actually governs.
    """
    text = str(rule)
    return bool(
        re.search(r"\b(?:pull request|PR)\b", text, re.IGNORECASE)
        and re.search(r"\b(?:tests?|CI)\b", text, re.IGNORECASE)
        and re.search(r"\b(?:before merge approval|approved for merge|before merging|"
                      r"before merge|merge approval|"
                      r"before (?:a |the |any )?(?:pull request|PR) "
                      r"(?:may |can |must )?be (?:opened|submitted)|"
                      r"before (?:opening|submitting) (?:a |the |any )?(?:pull request|PR))\b",
                      text, re.IGNORECASE)
        and not re.search(r"\b(?:edit_file|edit_repo_file|implementation work|"
                          r"before (?:coding|editing)|no edits?)\b", text, re.IGNORECASE)
    )


def review_change(
    world: Any,
    *,
    file_path: str,
    diff: str,
    candidate_content: str = "",
    action_type: str = "edit_repo_file",
    client: Optional[OrgLLMClient] = None,
) -> Optional[Dict[str, Any]]:
    """Judge one change against the adopted rules.

    Returns None when there is nothing to judge — no rules adopted, no change to
    read, or no model to ask. A refusal that cannot be reasoned about is worse
    than none, so an unreachable or unparseable reviewer lets the change through
    rather than blocking work on an outage.
    """
    # PR obligations cannot veto an intermediate edit of one source file.
    # Keep them in rules_block (author context); match enforcement to the
    # actual action, using the spec's typed declaration rather than its prose.
    rules = adopted_rules(world, action_type=action_type)
    if action_type in {"edit_file", "edit_repo_file"}:
        rules = [entry for entry in rules if not _merge_only_test_rule(entry[2])]
    if getattr(world, "_cooperbench_delivery_focus", False):
        # These adopted pair rules govern repository actions and joint receipts,
        # which the SDL enforces. A source diff cannot violate branch ownership
        # by using a context variable, lock, or shared cache. Keep the rule in
        # the editor's context, but never send it to the source-rewrite judge.
        manager = getattr(world, "proposal_manager", None)
        workflow_ids = set()
        for spec in (getattr(manager, "protocol_specs", {}) or {}).values():
            if (str(getattr(spec, "family", "")) == "cooperbench_joint_delivery"
                    and str(getattr(getattr(spec, "status", ""), "value", getattr(spec, "status", ""))) == "adopted"):
                spec_id = str(getattr(spec, "protocol_id", ""))
                workflow_ids.update((spec_id, f"proto_spec_{spec_id.split('_')[-1]}"))
        rules = [entry for entry in rules if entry[0] not in workflow_ids]
    if not rules or client is None or not (diff or "").strip():
        return None
    listed = "\n".join(f"{index + 1}. {rule}" for index, (_p, _t, rule) in enumerate(rules))
    user = (f"ACTION: {action_type}\nRULES:\n{listed}\n\n"
            f"CHANGE to {file_path}:\n{diff[:_MAX_DIFF_CHARS]}\n\n")
    if candidate_content:
        # A short diff omits unchanged documentation. Its absence from a hunk
        # does not mean the author failed to provide it in this candidate file.
        user += (
            f"COMPLETE CANDIDATE FILE AFTER THIS CHANGE ({file_path}):\n{candidate_content}\n\n"
            "Use the complete file to check existing declarations and documentation. "
            "Judge the change, not unrelated code. A rule may already be satisfied by "
            "unchanged material outside the diff; do not demand that it be added again.\n\n"
        )
    user += "Does this change comply with every rule above?"
    try:
        answer = client.generate_json(REVIEW_SYSTEM, user, REVIEW_SCHEMA)
    except (LLMError, Exception):  # noqa: BLE001  an outage must not refuse the work
        return None
    if not isinstance(answer, dict) or answer.get("complies") is not False:
        return None
    # A refusal that points at nothing in the change cannot be answered by
    # changing it. Half the refusals in one run asked for a reviewer's signoff,
    # which an author holding a patch has no way to produce: the loop spent its
    # revisions, dropped 42 of 77 changes, and the product ended the run exactly
    # as it began. Letting such a verdict through costs one unreviewed change;
    # honouring it costs every change the rule ever touches.
    symbol = str(answer.get("offending_symbol", "") or "").strip()
    if not symbol:
        return None
    matched = _which_rule(rules, answer)
    return {
        "symbol": symbol,
        "protocol_id": matched[0] if matched else "",
        "keywords": enforcement_keywords(matched[1], matched[2]) if matched else ("contract",),
        # The rule's own text, from the registry rather than from the answer, so
        # what the author is shown is the rule as adopted.
        "rule": matched[2] if matched else "an adopted rule",
        "reason": str(answer.get("reason", "") or "")[:400],
    }


def revision_brief(history: List[Dict[str, Any]], previous_patch: str) -> str:
    """What to tell an author whose change was refused, so the rewrite answers it.

    Everything, in full: which attempt this is, the whole patch that was just
    refused, and every reason given so far in order. A refusal with no memory of
    the earlier ones is how an arm made the same forbidden change over and over —
    213 edits, 199 refused, all but a handful renaming the same public symbol in
    the same file, because each rewrite began from a blank slate.
    """
    if not history:
        return ""
    lines = [f"\n\nATTEMPT {len(history) + 1}. Your previous change was REFUSED by a "
             "rule this organization adopted. Do not send it again unchanged."]
    lines.append("\nTHE PATCH THAT WAS REFUSED, in full:\n" + (previous_patch or "(empty)"))
    lines.append("\nEVERY REFUSAL SO FAR, oldest first:")
    for index, item in enumerate(history, start=1):
        lines.append(f"  refusal {index} — under: {item.get('rule', '')}")
        if item.get("symbol"):
            lines.append(f"      the symbol at issue: {item['symbol']}")
        lines.append(f"      what must change instead: {item.get('reason', '')}")
    lines.append("\nWrite a change that does what the goal asks AND satisfies every "
                 "refusal above. If the goal cannot be met without breaking a rule, "
                 "make the smallest change that keeps the rule and say so in "
                 "`change_summary`.")
    return "\n".join(lines)


__all__ = ["adopted_rules", "rules_block", "review_change", "enforcement_keywords",
           "revision_brief"]
