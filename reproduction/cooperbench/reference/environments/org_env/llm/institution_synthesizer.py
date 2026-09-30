"""InstitutionSynthesizer (Phase 9) — induce candidate protocols from REPEATED
patterns (failures / disputes / repairs / complaints / violations).

Detects a recurring pattern, then drafts a protocol Proposal (LLM or template).
The proposal still goes through approval + adoption — never auto-adopted.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from environments.org_env.llm.client import LLMError, OrgLLMClient
from environments.org_env.llm.prompts import PROTOCOL_SYSTEM, protocol_user
from environments.org_env.llm.schemas import PROTOCOL_SCHEMA, with_action_ids
from environments.org_env.proposals.objects import Proposal, ensure_list

# How much of the backlog one clustering pass reads, and how little it will act
# on. Below the floor there is no recurrence to find, only a queue.
_MIN_WISHES_TO_CLUSTER = 4
_MAX_WISHES_READ = 40


def _late_cooperbench_binding_actions(world: Any) -> List[str]:
    """Return real actions still reachable after every feature was peer-approved.

    A long main experiment can apply a new standing rule to later episodes. A
    finite two-feature Cooper unit cannot: once its last review has happened,
    only the final merge and the two required mainline verifications remain.
    Tell the model about that finite horizon instead of accepting a retrospective
    rule whose every declared action already happened.
    """

    flags = getattr(world, "__dict__", {})
    if not (
        flags.get("_cooperbench_delivery_focus", False)
        and flags.get("_cooperbench_main_b3_lifecycle", False)
    ):
        return []
    from environments.org_env.cooperbench.lifecycle import (
        peer_review_coverage,
        peer_review_ready_coverage,
    )

    coverage = peer_review_ready_coverage(world)
    if not coverage or not all(coverage.values()):
        return []
    merged = peer_review_coverage(world)
    return (["merge_pr"] if not all(merged.values()) else []) + ["verify_joint_delivery"]


def _normalized_action_ids(value: Any) -> set[str]:
    return {
        str(item).strip().casefold().replace("-", "_").replace(" ", "_")
        for item in ensure_list(value)
        if str(item).strip()
    }


def revise_late_cooperbench_proposal(
    proposal: Proposal,
    world: Any,
    remaining_actions: List[str],
    client: Optional[OrgLLMClient],
    *,
    reviewer_id: str = "",
) -> Optional[Dict[str, Any]]:
    """Ask the original proposer to revise a reviewed draft, not adopt it.

    Review can outlast the actions named by an earlier draft. The host supplies
    current reachability; only the member's model supplies the replacement rule.
    The caller must discard old approvals and route the revision through review.
    """
    if client is None:
        return None
    from environments.org_env.llm.prompt_assets import agent_identity_for, system_for

    proposer = world.agents.get(proposal.proposer_agent_id)
    available = _available(world)
    context = {
        "revision_of_proposal": proposal.to_dict(),
        "available_actions": available,
        "review_feedback": {
            "reviewer_id": reviewer_id,
            "requested_revision": proposal.suggested_revision,
        },
    }
    if remaining_actions:
        context["finite_delivery_action_constraint"] = {
            "must_include_at_least_one_in_affected_actions": remaining_actions,
            "reason": proposal.suggested_revision,
        }
    user = agent_identity_for(proposer, world) + "\n\n" + protocol_user(context)
    user += (
        "\nRevise your existing proposal for the current delivery state. Preserve "
        "the recurrent organizational problem and address the review feedback "
        "in the trigger and rule, governing a reachable action. Do not return "
        "the unchanged draft or merely append an action name. "
        "This is a new draft: both members must review it again before adoption."
    )
    schema = with_action_ids(PROTOCOL_SCHEMA, "affected_actions", actions=available)
    try:
        data = client.generate_json(
            system_for(proposer, world, "institution", PROTOCOL_SYSTEM), user, schema
        )
    except LLMError:
        return None
    actions = ensure_list(data.get("affected_actions"))
    if not (
        (not remaining_actions or set(actions) & set(remaining_actions))
        and set(actions) <= set(available)
        and data.get("name")
        and data.get("trigger_condition")
        and data.get("enforcement_rule")
    ):
        return None
    if reviewer_id and (
        data["enforcement_rule"] == proposal.proposed_solution
        and data["trigger_condition"] == proposal.target_problem
        and set(actions) == set(proposal.required_actions)
        and ensure_list(data.get("required_fields")) == proposal.required_artifacts
    ):
        return None
    return data


def _record_late_draft_event(
    world: Any,
    subtype: str,
    *,
    required_actions: List[str],
    attempts: int,
    reason: str,
) -> None:
    row = {
        "type": "governance_event",
        "subtype": subtype,
        "tick": int(getattr(world, "world_tick", 0) or 0),
        "required_future_actions": list(required_actions),
        "attempts": int(attempts),
        "reason": str(reason)[:300],
    }
    getattr(world, "__dict__", {}).setdefault(
        "_cooperbench_protocol_draft_events", []
    ).append(dict(row))
    events = getattr(world, "events", None)
    if isinstance(events, list):
        events.append(row)

CLUSTER_SYSTEM = (
    "You read what the members of one organization have been asking for and name "
    "the few themes that recur. A theme is worth a standing rule when the same "
    "avoidable failure keeps costing the organization and a checkable requirement "
    "on work would have caught it. "
    "Group the wishes: cite in `wish_numbers` the ones each theme rests on. "
    "Normally a theme must cite at least two numbered wishes. A single numbered "
    "wish is also recurrent when its listing explicitly says that at least two "
    "independent members support it, because identical wishes are deduplicated. "
    "Do not restate a rule the organization has already adopted, and do not return "
    "two themes that a single rule would cover — each must stand on its own. "
    "`enforcement_rule` must be checkable against a piece of work by someone who "
    "was not there: name what must be true, not what people should care about. "
    "Return nothing rather than something vague. Reply JSON only."
)

CLUSTER_SCHEMA = {
    "type": "object",
    "required": ["themes"],
    "properties": {
        "themes": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["name", "enforcement_rule", "wish_numbers"],
                "properties": {
                    "name": {"type": "string"},
                    "trigger_condition": {"type": "string"},
                    "enforcement_rule": {"type": "string"},
                    "required_fields": {"type": "array"},
                    "required_actions": {"type": "array"},
                    "wish_numbers": {"type": "array", "items": {"type": "integer"}},
                },
            },
        },
    },
}


def _label_from(rule: str) -> str:
    """A short title taken from the rule's own opening, for when none was given."""
    head = str(rule).strip().split(".")[0].strip()
    words = head.split()
    return (" ".join(words[:9]) + ("…" if len(words) > 9 else "")) or "an adopted rule"


def _traced(world: Any, why: str, out: List[Dict[str, Any]], *, read: int = 0):
    """Say why a clustering pass produced what it did, and return it unchanged.

    The pass used to swallow every outcome: a model that answered nothing, a
    model that could not be reached, and a theme dropped for resting on one wish
    all left the same trace, which is none. A run then showed zero clustered
    rules from a backlog of nine open wishes with no way to tell which had
    happened.
    """
    world.__dict__.setdefault("_wish_clustering", []).append({
        "tick": int(getattr(world, "world_tick", 0) or 0),
        "open_wishes": read, "themes": len(out), "why": why,
    })
    events = getattr(world, "events", None)
    if isinstance(events, list):
        events.append({"type": "governance_event", "subtype": "wish_clustering",
                       "tick": int(getattr(world, "world_tick", 0) or 0),
                       "open_wishes": read, "themes": len(out), "why": why})
    return out


def _adopted_rule_texts(world: Any) -> List[str]:
    """What the organization already requires of itself, as it wrote it."""
    from environments.org_env.backend.protocol.registry import protocol_is_live

    out: List[str] = []
    registry = getattr(world, "protocol_registry", None)
    for protocol in (getattr(registry, "protocols", {}) or {}).values():
        if (
            not protocol_is_live(protocol)
            or str(getattr(protocol, "adoption_status", "")) != "adopted"
        ):
            continue
        rule = str(getattr(protocol, "rule_summary", "") or "").strip()
        if rule:
            out.append(rule[:220])
    return out


# repeated episode_type -> (threshold, candidate protocol seed)
_PATTERN_SEEDS = {
    "claim_dispute_episode": (2, {
        "name": "Result Evidence Protocol",
        "trigger_condition": "any result used in a report or public claim",
        "required_fields": ["seed", "config", "raw_trace", "cost", "owner"],
        "enforcement_rule": "a reviewer may block report inclusion if evidence is missing",
        "required_actions": ["export_result_to_tracker", "request_reproduction"]}),
    "launch_crunch_episode": (3, {
        "name": "Launch Readiness Protocol",
        "trigger_condition": "any launch / public release",
        "required_fields": ["tests_passed", "review_done", "docs_ready"],
        "enforcement_rule": "no public update until the launch checklist passes",
        "required_actions": ["formal_pr_review", "create_doc"]}),
    "customer_triage_episode": (3, {
        "name": "Customer Triage Protocol",
        "trigger_condition": "any inbound customer issue",
        "required_fields": ["severity", "owner", "impact", "status"],
        "enforcement_rule": "every customer issue gets an owner + a triage record",
        "required_actions": ["create_customer_triage_sheet"]}),
    # v11 Coding Capability Layer §8 (Priority 3): repeated smoke/CI debugging episodes
    # institutionalize into a Release Readiness Debug Protocol — the deep protocol that
    # turns "fix one blocker" into "how this org closes technical release blockers".
    "debugging_episode": (2, {
        "name": "Release Readiness Debug Protocol",
        "trigger_condition": "a release smoke/CI gate fails and blocks the release candidate",
        "required_fields": ["failing_gate", "suspected_module", "patch", "smoke_result", "reviewer_signoff"],
        "enforcement_rule": ("on a smoke/CI failure: localize the bug, assign an owner, patch it, "
                             "run targeted + smoke tests, get review, and rerun the release gate "
                             "before re-attempting release"),
        "required_actions": ["edit_repo_file", "commit_patch", "run_ci", "run_launch_readiness_check"]}),
}


# v13 P4: when the repeated debugging cluster is about CROSS-MODULE INTERFACE breaks (not just a
# flaky smoke), the org should institutionalize a deeper protocol than "run smoke" — a contract
# protocol governing the schemas that keep the product's modules agreeing with each other.
_LANTERNSCOUT_CONTRACT_SEED = {
    "name": "Claim-Source Interface Contract Protocol",
    "trigger_condition": ("a patch/PR changes a core interface "
                          "(run_research / ClaimTracker.add / run_eval / write_report / smoke_check)"),
    "required_fields": ["claim_schema", "source_schema", "research_result_schema",
                        "run_eval_signature", "contract_test_updated"],
    "enforcement_rule": ("any change to a contract producer/consumer must keep the "
                         "Claim/Source/ResearchResult schema consistent and pass the contract test "
                         "(end-to-end smoke) before merge"),
    "required_actions": ["edit_repo_file", "run_ci", "review_pr"],
}
_CONTRACT_CLUSTER_KW = ("source_id", "signature", "attribute", "positional argument",
                        "contract", "run_eval", ".all", "takes")


def _interface_contract_seed(world: Any) -> Dict[str, Any]:
    """The interface-contract seed written in THIS product's own vocabulary.

    The seed is both shown to the model as the basis for the rule and used
    verbatim when the model call fails, so a hardcoded one decides what the
    institution ends up being about. With only the LanternScout seed available,
    an organization building a blob store adopted, enforced 87 times, and kept
    for 96 ticks a rule requiring "the Claim, Source and ResearchResult schemas"
    to stay consistent — schemas its product does not contain. The shape was
    right and the subject was another product's, so following the rule could not
    fix what was breaking: it recorded a violation_rate_delta of exactly 0.0.

    Everything here comes from what the organization can see it owns, so the
    rule names the specification and modules its own members are reading.
    """
    artifacts = getattr(world, "product_artifacts", {}) or {}
    path_of = {aid: str(getattr(a, "linked_file_path", "") or "")
               for aid, a in artifacts.items()
               if getattr(a, "artifact_type", "") != "issue"}
    paths = set(path_of.values())
    # Only where the product really is the one this rule describes. Asking
    # whether ANY file of CONTRACT_FILES is present matched on smoke_check.py,
    # which any product may ship — a blob store was handed the research
    # product's contract because both happen to have a smoke.
    if {"research_loop.py", "tools/claim_tracker.py"} <= paths:
        return dict(_LANTERNSCOUT_CONTRACT_SEED)

    # A README explains how to run the suite; it does not declare the surface, and
    # citing it as the specification sends a reader to the wrong file.
    spec = sorted(p for p in paths if p.endswith((".md", ".json"))
                  and "readme" not in p.lower()
                  and any(k in p.lower() for k in ("public_api", "architecture", "contract")))
    checks = sorted((p for p in paths if p.endswith(".py")
                     and ("contract_test" in p or "smoke" in p or p.startswith("tests/"))),
                    key=lambda p: ("/" in p, p))   # the whole-product gate before one step's
    try:
        from environments.org_env.product.substrates.issue_stream import unpatched_coding_issues
        modules = sorted({path_of.get(aid, "") for item in unpatched_coding_issues(world)
                          for aid in (item.get("artifact_ids") or ())} - {""})
    except Exception:  # noqa: BLE001
        modules = []
    modules = [m for m in modules if m.endswith(".py")] or sorted(
        p for p in paths if p.endswith(".py") and p not in checks and "/" in p)

    where = ", ".join(spec) if spec else "the product's published interface"
    what = ", ".join(modules[:6]) if modules else "any module other modules import"
    gate = f" and pass {checks[0]}" if checks else " and pass the contract test"
    return {
        "name": "Published Interface Contract Protocol",
        "trigger_condition": f"a patch/PR changes a module other code depends on ({what})",
        "required_fields": ["changed_module", "published_surface_checked",
                            "contract_test_result", "reviewer_signoff"],
        "enforcement_rule": (
            f"any change to a module other code depends on must keep its callable "
            f"surface — names, and each function's parameter names in order — "
            f"exactly as {where} declares it{gate} before merge"),
        "required_actions": ["edit_repo_file", "run_ci", "review_pr"],
    }


def _is_contract_cluster(mgr: Any, episode_ids: List[str]) -> bool:
    """True if the debugging cluster involves cross-module interface breaks (contract files /
    schema-mismatch errors) — warranting the interface contract protocol over a generic debug one."""
    try:
        from environments.org_env.product.contracts import CONTRACT_FILES
        cfiles = {f.split("/")[-1] for f in CONTRACT_FILES}
    except Exception:
        cfiles = set()
    for eid in episode_ids:
        ep = mgr.episodes.get(eid)
        if ep is None:
            continue
        mod = (getattr(ep, "suspected_module", "") or "").replace("\\", "/").split("/")[-1]
        log = (getattr(ep, "failure_log", "") or "").lower()
        if (mod and mod in cfiles) or any(k in log for k in _CONTRACT_CLUSTER_KW):
            return True
    return False


class InstitutionSynthesizer:
    def detect(self, world: Any) -> List[Dict[str, Any]]:
        """Return recurring patterns that warrant a candidate institution."""
        mgr = getattr(world, "episode_manager", None)
        if mgr is None:
            return []
        counts: Dict[str, List[str]] = {}
        for ep in mgr.episodes.values():       # count ALL occurrences of the pattern
            counts.setdefault(ep.episode_type, []).append(ep.episode_id)
        out = []
        existing = {s.name for s in world.proposal_manager.protocol_specs.values()}
        proposed = {p.title for p in world.proposal_manager.proposals.values()
                    if p.proposal_type == "protocol_proposal"}
        for etype, (thresh, seed) in _PATTERN_SEEDS.items():
            eps = counts.get(etype, [])
            if len(eps) < thresh:
                continue
            use_seed = seed
            # a contract-flavored debugging cluster -> the deeper interface contract protocol
            if etype == "debugging_episode" and _is_contract_cluster(mgr, eps):
                use_seed = _interface_contract_seed(world)
            if use_seed["name"] in existing or use_seed["name"] in proposed:
                continue
            out.append({"episode_type": etype, "episode_ids": eps, "seed": use_seed})
        return out

    def cluster_wishes(self, world: Any, client: Optional[OrgLLMClient] = None,
                       *, limit: int = 2) -> List[Dict[str, Any]]:
        """Themes the organization keeps asking for that no rule of its own covers.

        `detect` reads a fixed catalogue: four entries, thresholds of two or three
        episodes, one rule each and never again. It is met within the first few
        dozen ticks and then returns nothing for the rest of the run, so what an
        organization institutionalizes is decided in this file rather than by what
        it ran into. Measured across three runs of the same arm: every rule came
        from that catalogue, the last one arrived before the run was a third done,
        and the agents' own 52 wishes never became a rule.

        This reads the wishes instead. One theme is not a rule — a single wish is
        one bad afternoon — so a theme has to rest on several, and the model is
        given the rules already adopted and asked for themes none of them covers
        and that do not overlap each other. Nothing here adopts anything: a theme
        becomes a proposal like any other, and the review, the approval, the
        cross-episode depth gate and the semantic merge against adopted rules all
        still stand between it and the registry.
        """
        rm = getattr(world, "reflection_manager", None)
        if rm is None or client is None:
            return _traced(world, "no reflection manager or no model", [])
        wishes = [w for w in getattr(rm, "wishes", {}).values()
                  if str(getattr(w, "status", "")) == "open"]
        try:
            minimum = int(
                getattr(world, "__dict__", {}).get(
                    "_institution_wish_cluster_minimum",
                    _MIN_WISHES_TO_CLUSTER,
                )
            )
        except (TypeError, ValueError):
            minimum = _MIN_WISHES_TO_CLUSTER
        # One wish remains one bad afternoon.  An explicit small-organization
        # treatment may lower the eight-person backlog floor, but never below
        # the two independent wishes the theme validator already requires.
        minimum = max(2, minimum)
        independent_evidence = max(
            [len(wishes)]
            + [
                len(
                    {
                        str(agent_id)
                        for agent_id in (
                            getattr(wish, "supporting_agent_ids", None) or []
                        )
                        if str(agent_id)
                    }
                )
                for wish in wishes
            ]
        )
        if independent_evidence < minimum:
            return _traced(
                world,
                f"only {len(wishes)} open wishes / {independent_evidence} "
                f"independent evidence units, {minimum} needed",
                [],
            )
        wishes.sort(key=lambda w: -float(getattr(w, "urgency", 0.0) or 0.0))
        listed = "\n".join(
            f"{i + 1}. [{getattr(w, 'wish_type', '')}] "
            f"{str(getattr(w, 'interpreted_need', ''))[:180]}"
            f"  (independent supporters: "
            f"{len(set(getattr(w, 'supporting_agent_ids', None) or []))}; "
            f"because: {str(getattr(w, 'target_problem', ''))[:120]})"
            for i, w in enumerate(wishes[:_MAX_WISHES_READ]))

        adopted = _adopted_rule_texts(world)
        rules = "\n".join(f"- {r}" for r in adopted) or "(none yet)"
        user = (f"RULES THIS ORGANIZATION HAS ALREADY ADOPTED:\n{rules}\n\n"
                f"WHAT ITS MEMBERS HAVE BEEN ASKING FOR:\n{listed}\n\n"
                f"Name at most {limit} themes worth a standing rule.")
        try:
            answer = client.generate_json(CLUSTER_SYSTEM, user, CLUSTER_SCHEMA)
        except (LLMError, Exception) as exc:  # noqa: BLE001  no rule is better than a bad one
            return _traced(world, f"the model could not be read: "
                                  f"{type(exc).__name__}: {exc}"[:200], [],
                           read=len(wishes))
        if not isinstance(answer, dict):
            return _traced(world, f"the answer was {type(answer).__name__}, not an "
                                  f"object", [], read=len(wishes))

        out: List[Dict[str, Any]] = []
        dropped: List[str] = []
        for theme in (answer.get("themes") or [])[:limit]:
            if not isinstance(theme, dict):
                dropped.append("a theme was not an object")
                continue
            rule = str(theme.get("enforcement_rule", "") or "").strip()
            # A title is a label; the rule is the thing. Requiring a separate one
            # threw away a whole clustering pass: the schema asks for `name`, the
            # model answered with `theme`, and a usable rule about a cross-step
            # evidence matrix was dropped for being unnamed. Reading the other key
            # would be a guess about wording, so the label is taken from the rule's
            # own first clause when none is given.
            name = str(theme.get("name", "") or "").strip() or _label_from(rule)
            # A theme resting on one wish is one member's bad afternoon. The
            # indices are the model's own account of what it grouped, so a theme
            # that cannot name two is not a recurrence.
            picked = [wishes[i - 1] for i in (theme.get("wish_numbers") or [])
                      if isinstance(i, int) and 1 <= i <= len(wishes[:_MAX_WISHES_READ])]
            supporter_ids = list(
                dict.fromkeys(
                    str(agent_id)
                    for wish in picked
                    for agent_id in (
                        getattr(wish, "supporting_agent_ids", None) or []
                    )
                    if str(agent_id)
                )
            )
            recurrence = max(len(picked), len(supporter_ids))
            if not rule or recurrence < 2:
                # Name the condition that actually failed. The first version
                # reported a wish count whenever a rule was present, so a theme
                # dropped for having no title was reported as "rests on 3
                # wishes, 2 needed" — a complaint about a count that was fine.
                dropped.append(f"{name or 'an unnamed theme'}: "
                               + ("no rule to check" if not rule
                                   else (
                                       f"rests on {len(picked)} wish(es) from "
                                       f"{len(supporter_ids)} independent supporter(s), 2 needed"
                                   )))
                continue
            episodes: List[str] = []
            for w in picked:
                for eid in (getattr(w, "related_episode_ids", None) or []):
                    if eid not in episodes:
                        episodes.append(str(eid))
            if not episodes:
                episodes = [f"wish:{getattr(w, 'wish_id', '')}" for w in picked]
            out.append({
                "episode_type": "recurring_wish_cluster",
                "episode_ids": episodes,
                "wish_ids": [str(getattr(w, "wish_id", "")) for w in picked],
                "supporter_ids": supporter_ids,
                "seed": {
                    "name": name[:90],
                    "trigger_condition": str(theme.get("trigger_condition", ""))[:200],
                    "required_fields": ensure_list(theme.get("required_fields")),
                    "enforcement_rule": rule[:400],
                    "required_actions": ensure_list(theme.get("required_actions")),
                },
            })
        note = f"{len(out)} theme(s) from {len(wishes)} open wishes"
        if dropped:
            note += "; dropped " + "; ".join(dropped[:4])
        return _traced(world, note, out, read=len(wishes))

    def synthesize(self, pattern: Dict[str, Any], world: Any,
                   client: Optional[OrgLLMClient] = None) -> Optional[Proposal]:
        seed = pattern["seed"]
        data = None
        late_binding_actions = _late_cooperbench_binding_actions(world)
        if client is not None:
            try:
                from environments.org_env.llm.prompt_assets import agent_identity_for, system_for
                proposer_agent = next((world.agents.get(aid) for aid, a in world.agents.items()
                                       if getattr(a, "role", "") in ("founder", "cofounder")), None)
                ctx = {"pattern": pattern["episode_type"], "occurrences": len(pattern["episode_ids"]),
                       "seed": seed, "available_actions": _available(world)}
                if late_binding_actions:
                    ctx["finite_delivery_action_constraint"] = {
                        "must_include_at_least_one_in_affected_actions": late_binding_actions,
                        "reason": (
                            "Both feature heads have already completed peer review. "
                            "A retrospective rule cannot become operative in this finite "
                            "two-person unit; choose a real remaining merge or joint-mainline "
                            "verification action that this rule will govern."
                        ),
                    }
                system = system_for(proposer_agent, world, "institution", PROTOCOL_SYSTEM)
                # Prefix Cache Rule: per-agent identity travels in the USER message so the
                # system prompt stays byte-identical across agents (see prompt_assets).
                _identity = agent_identity_for(proposer_agent, world)
                _identity = (_identity + "\n\n") if _identity else ""
                schema = with_action_ids(PROTOCOL_SCHEMA, "affected_actions",
                                         actions=ctx["available_actions"])
                user = _identity + protocol_user(ctx)
                data = client.generate_json(system, user, schema)
                if late_binding_actions and not (
                    _normalized_action_ids(data.get("affected_actions", []))
                    & set(late_binding_actions)
                ):
                    correction = (
                        user
                        + "\n\nTHE PREVIOUS DRAFT IS INVALID FOR THIS FINITE DELIVERY. "
                        + "Its affected_actions contain no action that will still be "
                        + "reachable after adoption. Return one corrected JSON object. "
                        + "Keep the organization-authored rule, but affected_actions MUST "
                        + "include at least one of: "
                        + ", ".join(late_binding_actions)
                        + ".\nPREVIOUS DRAFT:\n"
                        + json.dumps(data, sort_keys=True, ensure_ascii=False)
                    )
                    data = client.generate_json(system, correction, schema)
                    if late_binding_actions and not (
                        _normalized_action_ids(data.get("affected_actions", []))
                        & set(late_binding_actions)
                    ):
                        _record_late_draft_event(
                            world,
                            "late_protocol_draft_rejected",
                            required_actions=late_binding_actions,
                            attempts=2,
                            reason="corrected_draft_still_declares_only_completed_actions",
                        )
                        return None
                    _record_late_draft_event(
                        world,
                        "late_protocol_draft_repaired",
                        required_actions=late_binding_actions,
                        attempts=2,
                        reason="bounded_action_reachability_correction_used",
                    )
            except (LLMError, Exception):
                if late_binding_actions:
                    _record_late_draft_event(
                        world,
                        "late_protocol_draft_rejected",
                        required_actions=late_binding_actions,
                        attempts=1,
                        reason="model_draft_or_bounded_correction_unavailable",
                    )
                    return None
                data = None
        data = data or seed
        mgr = world.proposal_manager
        # founder/cofounder proposes the institution
        proposer = next((aid for aid, a in world.agents.items()
                         if getattr(a, "role", "") in ("founder", "cofounder")), None)
        # Which of the two roads this came down. `cluster_wishes` names the wishes
        # it grouped and this dropped every one of them, so a rule the members
        # reached for themselves and a rule lifted from the fixed catalogue in
        # this file were recorded the same way -- both as a cluster of episodes,
        # since a wish carries its episodes too. Whether an organization is
        # forming its own institutions is the question the arm exists to answer,
        # and the answer was being thrown away one line before it was written down.
        wish_ids = [str(w) for w in (pattern.get("wish_ids") or []) if w]
        supporter_ids = [
            str(agent_id)
            for agent_id in (pattern.get("supporter_ids") or [])
            if agent_id
        ]
        return Proposal(
            proposal_id=mgr.next_id("proposal"), proposal_type="protocol_proposal",
            title=data.get("name", seed["name"]),
            summary=data.get("enforcement_rule", seed["enforcement_rule"]),
            proposer_agent_id=proposer, source_episode_id=pattern["episode_ids"][-1],
            source_episode_ids=list(pattern["episode_ids"]),   # v11: full cluster (depth gate + structure)
            source_event_ids=list(pattern["episode_ids"]),
            source_wish_id=(wish_ids[-1] if wish_ids else None),
            source_wish_ids=wish_ids,
            source_wish_supporter_ids=list(dict.fromkeys(supporter_ids)),
            target_problem=data.get("trigger_condition", seed["trigger_condition"]),
            proposed_solution=data.get("enforcement_rule", seed["enforcement_rule"]),
            required_actions=[a for a in data.get("affected_actions", seed.get("required_actions", []))
                              if not _available(world) or a in _available(world)],
            required_artifacts=ensure_list(data.get("required_fields", seed.get("required_fields", []))),
            expected_benefits=ensure_list(data.get("benefits", ["reproducibility", "institutional memory"])),
            risks=ensure_list(data.get("risks", ["slower iteration"])),
            created_at_tick=int(getattr(world, "world_tick", 0)),
            updated_at_tick=int(getattr(world, "world_tick", 0)))


def _available(world):
    try:
        from environments.org_env.backend.actions import registered_action_types
        return sorted(registered_action_types())
    except Exception:
        return []


__all__ = ["InstitutionSynthesizer"]
