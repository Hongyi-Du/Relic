"""ProposalGenerator (Phase 7) — turn a Wish into a structured Proposal DRAFT.

LLM-drafted (template fallback). The draft is NOT a world object; it enters
ProposalManager.create_proposal -> evaluate -> approval -> adoption.
"""
from __future__ import annotations

from typing import Any, Optional

from environments.org_env.llm.client import LLMError, OrgLLMClient
from environments.org_env.llm.prompts import PROPOSAL_SYSTEM, proposal_user
from environments.org_env.llm.schemas import PROPOSAL_SCHEMA, with_action_ids
from environments.org_env.proposals.objects import Proposal, ensure_list

# missing_support_type -> proposal_type
_SUPPORT_TO_PROPOSAL = {
    "tool": "tool_proposal", "workflow": "workflow_proposal", "protocol": "protocol_proposal",
    "artifact": "artifact_template_proposal", "role_clarity": "role_proposal",
    "resource": "task_proposal", "coordination": "workflow_proposal",
    "information": "artifact_template_proposal", "policy_repair": "policy_repair_proposal",
}


def _repair_fields(wish, data):
    """For a policy-repair wish, resolve WHICH adopted protocol to repair and HOW (relax|deprecate)
    — from the wish's related objects + language (or explicit LLM fields)."""
    target = data.get("repair_target_protocol_id") if isinstance(data, dict) else None
    if not target:
        for oid in (getattr(wish, "related_object_ids", []) or []):
            if str(oid).startswith(("protospec", "proto_spec", "protocol")):
                target = oid
                break
    kind = (data.get("repair_kind") if isinstance(data, dict) else None) or ""
    if kind not in ("relax", "deprecate"):
        blob = " ".join(str(getattr(wish, k, "") or "") for k in
                        ("interpreted_need", "target_problem", "raw_reflection_excerpt")).lower()
        kind = "deprecate" if any(w in blob for w in
                                  ("deprecat", "repeal", "retire", "remove the", "abolish")) else "relax"
    return target, kind
# the LLM emits natural-language proposal_type ('workflow_improvement', 'artifact_need',
# 'protocol', ...) -> canonical enum (preflight review #3, else validator rejects).
_PTYPE_ALIAS = {
    "workflow": "workflow_proposal", "workflow_improvement": "workflow_proposal",
    "process": "workflow_proposal", "process_improvement": "workflow_proposal",
    "tool": "tool_proposal", "tool_need": "tool_proposal", "technical": "tool_proposal",
    "protocol": "protocol_proposal", "protocol_need": "protocol_proposal",
    "norm": "protocol_proposal", "policy": "policy_repair_proposal",
    "policy_repair": "policy_repair_proposal", "quality_gate": "protocol_proposal",
    "artifact": "artifact_template_proposal", "artifact_need": "artifact_template_proposal",
    "artifact_template": "artifact_template_proposal", "template": "artifact_template_proposal",
    "doc": "artifact_template_proposal", "documentation": "artifact_template_proposal",
    "role": "role_proposal", "role_clarity": "role_proposal", "role_clarity_need": "role_proposal",
    "ownership": "role_proposal", "task": "task_proposal", "resource": "task_proposal",
}


def alias_proposal_type(raw: str):
    """Map a (possibly natural-language) proposal_type to the canonical enum, or
    None if it can't be mapped (so a truly-illegal type still gets rejected)."""
    from environments.org_env.proposals.objects import PROPOSAL_TYPES
    key = str(raw or "").strip().lower().replace(" ", "_")
    if key in PROPOSAL_TYPES:
        return key
    if key in _PTYPE_ALIAS:
        return _PTYPE_ALIAS[key]
    base = key.replace("_need", "").replace("_proposal", "")
    return _PTYPE_ALIAS.get(base)


def canon_proposal_type(raw: str, fallback: str) -> str:
    """As above but always returns a valid type (used by the generator: an
    unmappable LLM type falls back to the wish-derived canonical type)."""
    mapped = alias_proposal_type(raw)
    if mapped:
        return mapped
    from environments.org_env.proposals.objects import PROPOSAL_TYPES
    return fallback if fallback in PROPOSAL_TYPES else "workflow_proposal"
# light keyword -> existing actions (so a draft references real actions)
_NEED_ACTIONS = {
    "tracker": ["export_result_to_tracker", "create_experiment_tracker"],
    "evidence": ["request_reproduction", "create_claim_evidence_table"],
    "review": ["formal_pr_review", "review_pr"],
    "triage": ["create_customer_triage_sheet"],
    "launch": ["post_company_update", "create_doc"],
}


class ProposalGenerator:
    def generate(self, wish: Any, world: Any, client: Optional[OrgLLMClient] = None) -> Proposal:
        from environments.org_env.reflection.objects import support_type_for
        mgr = world.proposal_manager
        support = (getattr(wish, "missing_support_type", "")
                   or support_type_for(getattr(wish, "wish_type", "")))
        ptype = _SUPPORT_TO_PROPOSAL.get(support, "workflow_proposal")
        data = None
        if client is not None:
            data = self._llm(client, wish, world)
        if data is None:
            data = self._template(wish, ptype)
        p = Proposal(
            proposal_id=mgr.next_id("proposal"),
            proposal_type=canon_proposal_type(data.get("proposal_type"), ptype),
            title=data.get("title", "")[:80] or f"support: {getattr(wish, 'interpreted_need', '')}"[:80],
            summary=data.get("summary", "") or getattr(wish, "suggested_improvement", ""),
            proposer_agent_id=getattr(wish, "agent_id", None),
            source_wish_id=getattr(wish, "wish_id", None),
            source_wish_ids=(
                [str(getattr(wish, "wish_id"))]
                if getattr(wish, "wish_id", None)
                else []
            ),
            source_wish_supporter_ids=list(
                dict.fromkeys(
                    str(agent_id)
                    for agent_id in (
                        getattr(wish, "supporting_agent_ids", None) or []
                    )
                    if str(agent_id)
                )
            ),
            source_reflection_id=getattr(wish, "source_reflection_id", None),
            source_episode_id=getattr(wish, "source_episode_id", None),
            source_episode_ids=list(
                dict.fromkeys(
                    str(episode_id)
                    for episode_id in (
                        [getattr(wish, "source_episode_id", None)]
                        + list(getattr(wish, "related_episode_ids", None) or [])
                    )
                    if str(episode_id or "")
                )
            ),
            source_event_ids=list(getattr(wish, "source_event_ids", [])),
            target_problem=data.get("target_problem", "") or getattr(wish, "target_problem", ""),
            proposed_solution=data.get("proposed_solution", "") or getattr(wish, "suggested_improvement", ""),
            required_actions=self._actions(data, wish, world),
            required_capabilities=ensure_list(data.get("required_capabilities")),
            required_artifacts=ensure_list(data.get("required_artifacts")),
            required_participants=ensure_list(data.get("required_participants")),
            expected_benefits=ensure_list(data.get("expected_benefits")) or [getattr(wish, "expected_benefit", "")],
            expected_costs=ensure_list(data.get("expected_costs")),
            risks=ensure_list(data.get("risks")) or [getattr(wish, "risk_if_unaddressed", "")],
            created_at_tick=int(getattr(world, "world_tick", 0)),
            updated_at_tick=int(getattr(world, "world_tick", 0)))
        if p.proposal_type == "policy_repair_proposal":
            p.repair_target_protocol_id, p.repair_kind = _repair_fields(wish, data)
        return p

    def _actions(self, data, wish, world=None):
        """The actions this proposal declares, keeping only ones that exist.

        The validator rejects a whole proposal over one action it cannot find,
        and this handed it whatever the model wrote. The synthesizer's own path
        has always filtered against the registered set; a proposal reached from
        a wish had no such guard and lost the argument on a formatting detail
        rather than on its merits. Dropping an unknown name costs the rule
        nothing it can enforce -- nothing enforces an action that does not exist
        -- and keeps the rest of the proposal alive to be judged.
        """
        known = set(_available(world)) if world is not None else set()
        if data.get("required_actions"):
            named = ensure_list(data["required_actions"])
            return [a for a in named if not known or a in known]
        need = (getattr(wish, "interpreted_need", "") + " " + getattr(wish, "target_problem", "")).lower()
        out = []
        for kw, acts in _NEED_ACTIONS.items():
            if kw in need:
                out.extend(acts)
        return out

    def _llm(self, client, wish, world):
        try:
            from environments.org_env.llm.prompt_assets import agent_identity_for, render_product_context, system_for
            from environments.org_env.cooperbench.visibility import is_strict_coop, visible_brief_context
            actor_id = getattr(wish, "agent_id", None)
            agent = world.agents.get(actor_id)
            ctx = {"wish": wish.to_dict() if hasattr(wish, "to_dict") else str(wish),
                   "available_actions": _available(world),
                   "product_context": render_product_context(world, actor_id=actor_id)}
            if is_strict_coop(world):
                ctx["cooperbench_visible_briefs"] = visible_brief_context(world, actor_id)
            system = system_for(agent, world, "proposal", PROPOSAL_SYSTEM)
            # Prefix Cache Rule: per-agent identity travels in the USER message so the
            # system prompt stays byte-identical across agents (see prompt_assets).
            _identity = agent_identity_for(agent, world)
            _identity = (_identity + "\n\n") if _identity else ""
            schema = with_action_ids(PROPOSAL_SCHEMA, "required_actions",
                                     actions=ctx["available_actions"])
            return client.generate_json(system, _identity + proposal_user(ctx), schema)
        except LLMError:
            return None
        except Exception:
            return None

    def _template(self, wish, ptype):
        need = getattr(wish, "interpreted_need", "a needed improvement")
        return {"proposal_type": ptype, "title": need[:80].title(),
                "summary": getattr(wish, "suggested_improvement", need),
                "target_problem": getattr(wish, "target_problem", ""),
                "proposed_solution": getattr(wish, "suggested_improvement", need)}


def _available(world):
    try:
        from environments.org_env.backend.actions import registered_action_types
        return sorted(registered_action_types())
    except Exception:
        return []


__all__ = ["ProposalGenerator", "canon_proposal_type", "alias_proposal_type"]
