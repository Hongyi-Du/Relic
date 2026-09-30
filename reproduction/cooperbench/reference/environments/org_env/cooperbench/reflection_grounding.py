"""Host-grounded current-state receipts for CooperBench reflection.

Reflection sees long-lived memories, episode summaries, and conversation. Those
are useful evidence about *why* the organization reached its current state, but
they are not an authoritative description of that state. In particular, a
member can remember a protocol-formation episode from before the corresponding
proposal was adopted. This module projects only shared, public workflow state
so the reflection model can distinguish that history from the current facts.

No candidate source, private desk contents, hidden tests, gold patches, or
evaluator output enters this receipt.
"""
from __future__ import annotations

from typing import Any, Mapping


RECEIPT_SCHEMA = "cooperbench_reflection_current_state_v2"


def _status(value: Any) -> str:
    return str(getattr(value, "value", value) or "").strip().casefold()


def _linked_features(pull_request: Any) -> set[str]:
    linked = list(getattr(pull_request, "linked_issue_ids", []) or [])
    single = getattr(pull_request, "linked_issue", None)
    if single:
        linked.append(single)
    return {
        str(item)
        for item in linked
        if str(item).startswith("cooper_feature_")
    }


def _repository_delivery(world: Any, agent_id: str) -> dict[str, Any]:
    """Return shared PR/mainline facts without exposing either private desk."""

    state = getattr(world, "_cooperbench_sdl_state", {}) or {}
    owners = state.get("feature_owners") or {}
    repo = getattr(getattr(world, "repo_system", None), "repo", None)
    pull_requests = getattr(repo, "pull_requests", {}) or {}
    features: dict[str, dict[str, Any]] = {}
    for feature_id, owner_id in sorted(owners.items()):
        prs = []
        for pr_id, pull_request in pull_requests.items():
            if str(feature_id) not in _linked_features(pull_request):
                continue
            prs.append(
                {
                    "pr_id": str(pr_id),
                    "status": _status(getattr(pull_request, "status", "")),
                    "ci_passed": bool(getattr(pull_request, "ci_passed", False)),
                    "approved_by": sorted(
                        str(item)
                        for item in (getattr(pull_request, "approved_by", []) or [])
                        if str(item)
                    ),
                    "merged_tick": getattr(pull_request, "merged_tick", None),
                }
            )
        statuses = {row["status"] for row in prs}
        if "merged" in statuses:
            shared_status = "merged"
        elif statuses & {
            "open",
            "review_requested",
            "changes_requested",
            "approved",
        }:
            shared_status = "published_for_review"
        elif statuses:
            shared_status = "closed_or_stale"
        else:
            shared_status = "not_yet_published"
        features[str(feature_id)] = {
            "owner_relation": "self" if str(owner_id) == str(agent_id) else "peer",
            "shared_status": shared_status,
            "pull_requests": prs,
        }
    return {
        "phase": str(state.get("phase") or ""),
        "features": features,
        "joint_attestation_member_ids": sorted(
            str(item) for item in (state.get("attestations") or {}) if str(item)
        ),
        "frozen": bool(state.get("frozen_digest")),
        "frozen_tick": state.get("frozen_tick"),
        "visibility_note": (
            "not_yet_published describes shared repository state only; it does not "
            "claim that the owner's private desk has no work"
        ),
    }


def _governance_state(world: Any) -> dict[str, Any]:
    manager = getattr(world, "proposal_manager", None)
    specs = getattr(manager, "protocol_specs", {}) or {}
    proposals = getattr(manager, "proposals", {}) or {}

    realization: Mapping[str, Any] = {}
    try:
        from .lifecycle import protocol_realization

        realization = protocol_realization(world)
    except Exception:  # noqa: BLE001 - reflection must remain available
        realization = {}
    formed_ids = {
        str(protocol_id)
        for protocol_id in (
            realization.get("formed_main_b3_protocol_ids") or []
        )
    }

    protocol_rows = []
    for protocol_id, spec in sorted(specs.items()):
        protocol_rows.append(
            {
                "protocol_id": str(protocol_id),
                "name": str(getattr(spec, "name", "") or "")[:300],
                "status": _status(getattr(spec, "status", "")),
                "adopted_at_tick": getattr(spec, "adopted_at_tick", None),
                "adopted_by": sorted(
                    str(item)
                    for item in (getattr(spec, "adopted_by", []) or [])
                    if str(item)
                ),
                "affected_actions": sorted(
                    str(item)
                    for item in (getattr(spec, "affected_actions", []) or [])
                    if str(item)
                ),
                "use_count": int(getattr(spec, "use_count", 0) or 0),
                "enforcement_count": int(
                    getattr(spec, "enforcement_count", 0) or 0
                ),
                "formation_qualified": str(protocol_id) in formed_ids,
                "source_episode_ids": sorted(
                    str(item)
                    for item in (getattr(spec, "source_episode_ids", []) or [])
                    if str(item)
                ),
                "source_wish_ids": sorted(
                    str(item)
                    for item in (getattr(spec, "source_wish_ids", []) or [])
                    if str(item)
                ),
                "source_wish_supporter_ids": sorted(
                    str(item)
                    for item in (
                        getattr(spec, "source_wish_supporter_ids", []) or []
                    )
                    if str(item)
                ),
            }
        )

    open_proposals = []
    for proposal_id, proposal in sorted(proposals.items()):
        status = _status(getattr(proposal, "status", ""))
        if status not in {"draft", "under_review", "approved"}:
            continue
        open_proposals.append(
            {
                "proposal_id": str(proposal_id),
                "proposal_type": str(
                    getattr(proposal, "proposal_type", "") or ""
                ),
                "title": str(getattr(proposal, "title", "") or "")[:300],
                "status": status,
                "required_actions": list(
                    getattr(proposal, "required_actions", []) or []
                ),
                "suggested_revision": str(
                    getattr(proposal, "suggested_revision", "") or ""
                ),
                "target_problem": str(getattr(proposal, "target_problem", "") or ""),
                "proposed_solution": str(getattr(proposal, "proposed_solution", "") or ""),
                "required_artifacts": list(
                    getattr(proposal, "required_artifacts", []) or []
                ),
                "approved_by": sorted(
                    str(item)
                    for item in (getattr(proposal, "approved_by", []) or [])
                    if str(item)
                ),
                "approval_required_from": sorted(
                    str(item)
                    for item in (
                        getattr(proposal, "approval_required_from", []) or []
                    )
                    if str(item)
                ),
            }
        )

    adopted_ids = [
        row["protocol_id"] for row in protocol_rows if row["status"] == "adopted"
    ]
    return {
        "adopted_protocol_ids": adopted_ids,
        "formed_protocol_ids": sorted(formed_ids),
        "protocols": protocol_rows,
        "open_proposals": open_proposals,
        "protocol_formation_satisfied": bool(
            realization.get("protocol_formation_satisfied")
        ),
        "protocol_gate_satisfied": bool(
            realization.get("protocol_gate_satisfied")
        ),
        "action_bound_protocol_ids": list(
            realization.get("main_b3_action_bound_protocol_ids") or []
        ),
        "realized_b3": bool(realization.get("realized_b3")),
        "state_distinction": (
            "adopted means the organization formally accepted the rule; "
            "formation_qualified separately requires recurrent member/episode provenance; "
            "protocol_gate_satisfied and realized_b3 then require a successful action-bound "
            "use receipt"
        ),
    }


def cooperbench_reflection_current_state(
    world: Any, agent_id: str
) -> dict[str, Any] | None:
    """Project the authoritative current shared state for the B3-2 treatment."""

    state = getattr(world, "__dict__", {}).get("_cooperbench_sdl_state")
    if not isinstance(state, dict) or not getattr(
        world, "_cooperbench_delivery_focus", False
    ):
        return None
    return {
        "schema_version": RECEIPT_SCHEMA,
        "tick": int(getattr(world, "world_tick", 0) or 0),
        "authority": (
            "Host-maintained current public workflow state. Treat these fields as "
            "authoritative for current delivery and governance status. Memories, "
            "episodes, and conversation may explain earlier states but cannot "
            "contradict this receipt."
        ),
        "delivery": _repository_delivery(world, agent_id),
        "governance": _governance_state(world),
    }


def stale_cooperbench_governance_need(
    world: Any,
    wish_type: str,
    description: str,
    *,
    reflection_text: str = "",
) -> tuple[bool, list[str]]:
    """Reject a reflected need whose claimed missing adoption already exists.

    Only an unqualified standalone absence claim is contradicted by an existing
    adoption. Mentioning adoption, or asking to improve its process, does not
    establish that a new organizational need has already been satisfied.
    """

    if wish_type == "policy_repair_need":
        return False, []
    state = getattr(world, "__dict__", {}).get("_cooperbench_sdl_state")
    if not isinstance(state, dict) or not getattr(
        world, "_cooperbench_delivery_focus", False
    ):
        return False, []
    manager = getattr(world, "proposal_manager", None)
    adopted = sorted(
        str(protocol_id)
        for protocol_id, spec in (
            getattr(manager, "protocol_specs", {}) or {}
        ).items()
        if _status(getattr(spec, "status", "")) == "adopted"
    )
    if getattr(world, "__dict__", {}).get(
        "_cooperbench_main_b3_lifecycle", False
    ):
        # A generic, shallow ProtocolSpec can be formally adopted while still
        # failing the main B3 recurrence/provenance gate.  Such a rule must not
        # suppress the very reflected need that can produce a qualifying rule.
        # r86 demonstrated the otherwise permanent state: adopted=True while
        # formed/action-bound/realized all remain false.
        from .lifecycle import formed_main_b3_protocols

        formed_ids = {
            str(getattr(spec, "protocol_id", "") or "")
            for spec in formed_main_b3_protocols(world)
        }
        adopted = [protocol_id for protocol_id in adopted if protocol_id in formed_ids]
    if not adopted:
        return False, []
    description_text = " ".join(str(description or "").casefold().split())
    absence_statements = {
        "no adopted protocol",
        "there is no adopted protocol",
        "we have no adopted protocol",
    }
    stale = (
        wish_type in {"protocol_need", "workflow_need"}
        and description_text.rstrip(" .!?") in absence_statements
    )
    return stale, adopted


__all__ = [
    "RECEIPT_SCHEMA",
    "cooperbench_reflection_current_state",
    "stale_cooperbench_governance_need",
]
