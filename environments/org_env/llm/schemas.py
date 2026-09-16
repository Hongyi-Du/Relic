"""JSON schemas for each LLM cognitive module (loose — required keys + object type).

These are intentionally light: the OrgLLMClient checks required keys, and each
module's downstream validator/manager enforces the real-world constraints (legal
action, existing target, approval, etc.).
"""
from __future__ import annotations

from typing import Any, Dict

from environments.org_env.reflection.objects import WISH_TYPES

ACTION_DECISION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["candidate_action"],
    "properties": {
        "candidate_action": {"type": "string"},
        "candidate_speech_act": {"type": ["string", "null"]},
        "target_agent_id": {"type": ["string", "null"]},
        "target_object_id": {"type": ["string", "null"]},
        "channel_id": {"type": ["string", "null"]},
        "params": {"type": "object"},
        "rationale": {"type": "string"},
        "expected_effect": {"type": ["string", "null"]},
        "risk_assessment": {"type": ["string", "null"]},
        "confidence": {"type": "number"},
    },
}

SURFACE_TEXT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["surface_text"],
    "properties": {
        "surface_text": {"type": "string"},
        "style_tags": {"type": "array"},
        "contains_new_facts": {"type": "boolean"},
    },
}

REFLECTION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["self_assessment", "team_assessment", "improvement_ideas"],
    "properties": {
        "self_assessment": {"type": "string"},
        "team_assessment": {"type": "string"},
        "perceived_blockers": {"type": "array"},
        "perceived_self_needs": {"type": "array"},
        "perceived_team_needs": {"type": "array"},
        "perceived_repeated_failures": {"type": "array"},
        # `need_type` used to be an unconstrained string, and a keyword table
        # downstream mapped whatever arrived onto the nine kinds a wish can have,
        # defaulting to workflow_need. Only six spellings reached protocol_need,
        # so an agent that wrote "standard", "process" or "naming convention"
        # asked for a workflow. Across three runs that produced 49 workflow_need
        # wishes and not one protocol_need, and the whole path from a wish to an
        # adopted rule — which is wired and works — was never once entered.
        # The vocabulary is now in the schema, so the agent chooses from it.
        "improvement_ideas": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["need_type", "description"],
                "properties": {
                    "need_type": {"type": "string", "enum": list(WISH_TYPES)},
                    "description": {"type": "string"},
                    "urgency": {"type": "number"},
                    "risk_if_unaddressed": {"type": "string"},
                },
            },
        },
        "raw_text": {"type": "string"},
    },
}

WISH_EXTRACTION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["wishes"],
    "properties": {"wishes": {"type": "array"}},
}

PROPOSAL_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["proposal_type", "title", "summary"],
    "properties": {
        "proposal_type": {"type": "string"}, "title": {"type": "string"},
        "summary": {"type": "string"}, "target_problem": {"type": "string"},
        "proposed_solution": {"type": "string"}, "required_actions": {"type": "array"},
        "required_capabilities": {"type": "array"}, "required_artifacts": {"type": "array"},
        "required_participants": {"type": "array"}, "expected_benefits": {"type": "array"},
        "expected_costs": {"type": "array"}, "risks": {"type": "array"},
        "approval_required_from": {"type": "array"},
    },
}

def with_action_ids(schema: Dict[str, Any], *fields: str,
                    actions: Any) -> Dict[str, Any]:
    """The same schema, with the named array fields bound to real action ids.

    A proposal names the actions its rule governs, and those names have to be
    ids the simulator registers -- the validator rejects the whole proposal over
    a single one it does not know. Asked for an untyped array, a model answers in
    the register of the question: "Create a claim-evidence table or equivalent
    release-evidence record for Steps 1-5." was a required action, and the
    proposal it belonged to -- the first any agent reached from its own wishes --
    died on that sentence at t6 while every proposal the environment seeded, whose
    ids are hardcoded, passed.

    Listing the ids in the prompt was already being done and did not help. An
    enum is the same information where the model cannot answer around it.
    """
    ids = [str(a) for a in (actions or []) if a]
    if not ids:
        return schema
    out = dict(schema)
    out["properties"] = dict(schema.get("properties") or {})
    for field_name in fields:
        out["properties"][field_name] = {
            "type": "array", "items": {"type": "string", "enum": ids}}
    return out


PROPOSAL_EVAL_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["feasibility_score", "usefulness_score", "risk_score", "adoption_score"],
    "properties": {
        "feasibility_score": {"type": "number"}, "usefulness_score": {"type": "number"},
        "risk_score": {"type": "number"}, "adoption_score": {"type": "number"},
        "blocking_issues": {"type": "array"}, "suggested_revision": {"type": "string"},
    },
}

TOOL_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["name", "tool_type"],
    "properties": {
        "name": {"type": "string"}, "description": {"type": "string"},
        "tool_type": {"type": "string"}, "input_schema": {"type": "object"},
        "output_schema": {"type": "object"}, "required_actions": {"type": "array"},
        "required_capabilities": {"type": "array"}, "risk_tags": {"type": "array"},
        "validation_rules": {"type": "array"}, "callable_by_roles": {"type": "array"},
    },
}

PROTOCOL_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["name", "trigger_condition"],
    "properties": {
        "name": {"type": "string"}, "trigger_condition": {"type": "string"},
        "required_steps": {"type": "array"}, "required_fields": {"type": "array"},
        "enforcement_rule": {"type": "string"}, "violation_condition": {"type": "string"},
        "exception_rule": {"type": ["string", "null"]}, "benefits": {"type": "array"},
        "costs": {"type": "array"}, "risks": {"type": "array"}, "affected_actions": {"type": "array"},
    },
}

EPISODE_SUMMARY_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["title", "outcome_summary"],
    "properties": {
        "title": {"type": "string"}, "trigger_summary": {"type": "string"},
        "participant_summary": {"type": "string"}, "conflict_summary": {"type": "string"},
        "decision_summary": {"type": "string"}, "outcome_summary": {"type": "string"},
        "product_change_summary": {"type": "string"}, "open_questions": {"type": "array"},
    },
}

EVENT_APPRAISAL_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["success", "intensity"],
    "properties": {
        "success": {"type": "boolean"}, "reputation_delta": {"type": "number"},
        "knowledge_gain": {"type": "number"}, "social_support_received": {"type": "number"},
        "social_harm_received": {"type": "number"},
        "rule_compliance_result": {"type": "string"},   # compliant|violation|none
        "promise_kept_or_broken": {"type": "string"},    # kept|broken|none
        "emotional_valence": {"type": "number"}, "intensity": {"type": "string"},  # minor|moderate|major
        "visibility": {"type": "string"}, "rationale": {"type": "string"},
    },
}

__all__ = [
    "ACTION_DECISION_SCHEMA", "SURFACE_TEXT_SCHEMA", "REFLECTION_SCHEMA",
    "WISH_EXTRACTION_SCHEMA", "PROPOSAL_SCHEMA", "PROPOSAL_EVAL_SCHEMA",
    "TOOL_SCHEMA", "PROTOCOL_SCHEMA", "EPISODE_SUMMARY_SCHEMA", "EVENT_APPRAISAL_SCHEMA",
    "with_action_ids",
]
