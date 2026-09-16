"""ToolComposer (Phase 8) — compose a simulation-level ToolSpec draft from existing
actions/capabilities (no arbitrary code). Enriches a tool/workflow proposal; the
ToolSpec is only minted by ProposalManager.adopt after approval.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from environments.org_env.llm.client import LLMError, OrgLLMClient
from environments.org_env.llm.prompts import TOOL_SYSTEM, tool_user
from environments.org_env.llm.schemas import TOOL_SCHEMA


class ToolComposer:
    def compose(self, proposal: Any, world: Any, client: Optional[OrgLLMClient] = None) -> Dict[str, Any]:
        data = None
        if client is not None:
            try:
                from environments.org_env.llm.prompt_assets import agent_identity_for, system_for
                agent = world.agents.get(getattr(proposal, "proposer_agent_id", None))
                ctx = {"proposal": proposal.to_dict(), "available_actions": _available(world)}
                system = system_for(agent, world, "tool", TOOL_SYSTEM)
                # Prefix Cache Rule: per-agent identity travels in the USER message so the
                # system prompt stays byte-identical across agents (see prompt_assets).
                _identity = agent_identity_for(agent, world)
                _identity = (_identity + "\n\n") if _identity else ""
                data = client.generate_json(system, _identity + tool_user(ctx), TOOL_SCHEMA)
            except (LLMError, Exception):
                data = None
        if data is None:
            data = {"name": proposal.title, "description": proposal.summary,
                    "tool_type": "composed_action_tool",
                    "required_actions": list(proposal.required_actions),
                    "required_capabilities": list(proposal.required_capabilities)}
        # only keep actions that actually exist (safety). The real LLM sometimes returns
        # required_actions as dicts ({"action": "edit_doc"}) — coerce to strings first so
        # the membership test never hits an unhashable element.
        known = set(_available(world))
        acts = _coerce_actions(data.get("required_actions"))
        data["required_actions"] = [a for a in acts if not known or a in known]
        # write composed details back onto the proposal draft
        if data["required_actions"]:
            proposal.required_actions = data["required_actions"]
        if data.get("required_capabilities"):
            from environments.org_env.proposals.objects import ensure_list
            proposal.required_capabilities = ensure_list(data["required_capabilities"])
        return data


def _coerce_actions(raw) -> list:
    """Normalize an LLM required_actions list to a list of action-id strings, tolerating
    dict-wrapped items like {"action": "edit_doc"} / {"name": ...} (real-LLM quirk)."""
    out = []
    for a in (raw or []):
        if isinstance(a, str):
            a = a.strip()
        elif isinstance(a, dict):
            v = a.get("action") or a.get("name") or a.get("id") or a.get("type")
            a = v.strip() if isinstance(v, str) else None
        else:
            a = None
        if a:
            out.append(a)
    return out


def _available(world):
    try:
        from environments.org_env.backend.actions import registered_action_types
        return sorted(registered_action_types())
    except Exception:
        return []


__all__ = ["ToolComposer"]
