"""EventAppraisal (spec §23) — optional LLM annotation of an ALREADY-executed event.

Produces structured appraisal fields (reputation/knowledge/compliance/valence/…)
the system MAY use for memory/graph/social signals. The LLM only annotates; it does
not mutate state. NOT auto-run per event (cost) — call explicitly when a salient
event warrants a richer read; falls back to a neutral template.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from environments.org_env.llm.client import LLMError, OrgLLMClient
from environments.org_env.llm.schemas import EVENT_APPRAISAL_SCHEMA

EVENT_APPRAISAL_SYSTEM = (
    "You annotate the consequences of an already-executed event. You do not update "
    "state — you only produce structured appraisal fields. Use only observed event "
    "data; distinguish actual outcome from possible implication.")

# Growth Module §5: reputation_delta is no longer a fixed 0.0 — it follows the event's
# rule-based outcome sign (the Growth Module owns the actual persistent reputation state;
# this annotation just reflects the direction for memory/inspector readability).
_REP_SIGN = {
    "merged_to_mainline": 0.03, "pr_merged": 0.03, "patch_applied": 0.015, "pr_reviewed": 0.02,
    "published": 0.03, "protocol_use_event": 0.015, "protocol_enforcement_event": 0.02,
    "patch_rejected": -0.02, "changes_requested": -0.02, "ci_failed": -0.03,
    "protocol_violation_event": -0.04, "gate_blocked": -0.03,
}


def _event_reputation_delta(event: Dict[str, Any]) -> float:
    key = event.get("subtype") or event.get("type") or ""
    if key in _REP_SIGN:
        return _REP_SIGN[key]
    et = event.get("type", "")
    if et in _REP_SIGN:
        return _REP_SIGN[et]
    if event.get("success") is False:
        return -0.01
    return 0.0


class EventAppraiser:
    def appraise(self, event: Dict[str, Any], world: Any,
                 client: Optional[OrgLLMClient] = None) -> Dict[str, Any]:
        if client is not None:
            try:
                from environments.org_env.llm.prompt_assets import agent_identity_for, system_for
                agent = world.agents.get(event.get("agent_id")) if world else None
                sysp = system_for(agent, world, "event_appraisal", EVENT_APPRAISAL_SYSTEM)
                # Prefix Cache Rule: per-agent identity travels in the USER message so the
                # system prompt stays byte-identical across agents (see prompt_assets).
                _identity = agent_identity_for(agent, world)
                _identity = (_identity + "\n\n") if _identity else ""
                user = ("Annotate this event:\n" + str({k: event.get(k) for k in
                        ("type", "subtype", "agent_id", "tick", "artifact_id", "object_id",
                         "result_id", "pr_id", "protocol_id")}))
                return client.generate_json(sysp, _identity + user, EVENT_APPRAISAL_SCHEMA)
            except (LLMError, Exception):
                pass
        return self._template(event)

    def _template(self, event: Dict[str, Any]) -> Dict[str, Any]:
        et = event.get("type", "")
        compliance = ("violation" if "violation" in et else
                      ("compliant" if "enforcement" in et or "protocol_use" in et else "none"))
        return {
            "success": bool(event.get("success", True)),
            "reputation_delta": _event_reputation_delta(event),
            "knowledge_gain": 0.1 if et in ("experiment_event", "search_event") else 0.0,
            "social_support_received": 0.0, "social_harm_received": 0.0,
            "rule_compliance_result": compliance,
            "promise_kept_or_broken": ("broken" if event.get("subtype") == "violation" else "none"),
            "emotional_valence": 0.0, "intensity": "minor",
            "visibility": "team", "rationale": f"template appraisal of {et}",
        }


__all__ = ["EventAppraiser", "EVENT_APPRAISAL_SYSTEM"]
