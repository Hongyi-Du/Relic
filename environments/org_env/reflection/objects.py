"""Reflection / Wish / Memory data structures (OrgEnv reflection layer).

The cognitive chain is:

    recent events / episode context / work state
      -> AgentReflection (explicit self/team assessment + improvement ideas)
      -> Wish (structured need EXTRACTED from the reflection, traceable to it)
      -> (later) Proposal -> Tool / Protocol / Workflow

Reflection is the LLM-visible cognitive step; a Wish is never invented out of thin
air — it is pulled from a reflection and keeps ``source_reflection_id`` +
``raw_reflection_excerpt`` so we can trace back to what the agent thought.

Reflections are also written into long-term ``AgentMemory`` + an ``AgentLogEntry``
trace so the agent actually "remembers" and is influenced later.

All generic: no per-agent-id logic anywhere here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# wish_type / missing_support_type vocabulary (spec §6)
WISH_TYPES = ("tool_need", "protocol_need", "workflow_need", "artifact_need",
              "role_clarity_need", "resource_need", "coordination_need", "information_need",
              "policy_repair_need")
SUPPORT_TYPES = ("tool", "workflow", "protocol", "artifact", "role_clarity",
                 "resource", "coordination", "information")

WISH_STATUS = ("open", "interpreted", "converted_to_proposal", "rejected",
               "satisfied", "dormant")

# raw need-category -> canonical wish_type (preflight §9.2). Fixes the "everything
# becomes tool_need" bug; default is workflow_need (neutral), never tool_need.
NEED_TYPE_MAP = {
    "technical": "tool_need", "tool": "tool_need", "tool_need": "tool_need",
    "process": "workflow_need", "workflow": "workflow_need", "workflow_need": "workflow_need",
    "documentation": "artifact_need", "doc": "artifact_need", "docs": "artifact_need",
    "artifact": "artifact_need", "artifact_need": "artifact_need",
    "evaluation": "protocol_need", "eval": "protocol_need", "protocol": "protocol_need",
    "quality_gate": "protocol_need", "quality": "protocol_need", "protocol_need": "protocol_need",
    "collaboration": "coordination_need", "coordination": "coordination_need",
    "coordination_need": "coordination_need",
    "role": "role_clarity_need", "ownership": "role_clarity_need",
    "role_clarity": "role_clarity_need", "role_clarity_need": "role_clarity_need",
    "resource": "resource_need", "capacity": "resource_need", "resource_need": "resource_need",
    "information": "information_need", "info": "information_need", "information_need": "information_need",
    # self-correction: a wish to CHANGE an existing rule (relax / repeal a self-binding protocol)
    "policy": "policy_repair_need", "policy_repair": "policy_repair_need",
    "policy_repair_need": "policy_repair_need", "rule_change": "policy_repair_need",
    "relax_rule": "policy_repair_need", "loosen": "policy_repair_need", "repeal": "policy_repair_need",
    "deprecate_protocol": "policy_repair_need", "amend_protocol": "policy_repair_need",
}


def canon_wish_type(raw: str) -> str:
    key = str(raw or "").strip().lower().replace(" ", "_")
    return NEED_TYPE_MAP.get(key, "workflow_need")


def support_type_for(wish_type: str) -> str:
    return wish_type[:-5] if wish_type.endswith("_need") else "workflow"


def make_wish_fingerprint(wish_type: str, target_problem: str,
                          related_object_ids: Optional[List[str]] = None) -> str:
    """Identity for dedup/merge (preflight §8.1): type + canonical problem + objects."""
    norm = " ".join(str(target_problem or "").lower().split())[:80]
    objs = ",".join(sorted(set(related_object_ids or [])))
    return f"{wish_type}|{norm}|{objs}"


@dataclass
class AgentReflection:
    reflection_id: str
    agent_id: str
    tick: int

    source_episode_ids: List[str] = field(default_factory=list)
    source_event_ids: List[str] = field(default_factory=list)
    source_object_ids: List[str] = field(default_factory=list)

    self_assessment: str = ""
    team_assessment: str = ""

    perceived_blockers: List[str] = field(default_factory=list)
    perceived_repeated_failures: List[str] = field(default_factory=list)
    perceived_team_needs: List[str] = field(default_factory=list)
    perceived_self_needs: List[str] = field(default_factory=list)

    # each: {need_type, description, urgency, risk_if_unaddressed, missing_support_type,
    #        self_related, team_related}
    improvement_ideas: List[Dict[str, Any]] = field(default_factory=list)

    raw_text: str = ""
    llm_model: Optional[str] = None
    trigger_reason: str = ""

    created_wish_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reflection_id": self.reflection_id, "agent_id": self.agent_id, "tick": self.tick,
            "source_episode_ids": list(self.source_episode_ids),
            "source_event_ids": list(self.source_event_ids),
            "source_object_ids": list(self.source_object_ids),
            "self_assessment": self.self_assessment, "team_assessment": self.team_assessment,
            "perceived_blockers": list(self.perceived_blockers),
            "perceived_repeated_failures": list(self.perceived_repeated_failures),
            "perceived_team_needs": list(self.perceived_team_needs),
            "perceived_self_needs": list(self.perceived_self_needs),
            "improvement_ideas": [dict(i) for i in self.improvement_ideas],
            "raw_text": self.raw_text, "llm_model": self.llm_model,
            "trigger_reason": self.trigger_reason,
            "created_wish_ids": list(self.created_wish_ids),
        }


@dataclass
class Wish:
    wish_id: str
    agent_id: str

    source_reflection_id: str = ""
    source_reflection_ids: List[str] = field(default_factory=list)
    supporting_agent_ids: List[str] = field(default_factory=list)
    support_count: int = 1
    source_episode_id: Optional[str] = None
    source_event_ids: List[str] = field(default_factory=list)

    raw_reflection_excerpt: str = ""
    wish_type: str = "workflow_need"
    fingerprint: str = ""

    interpreted_need: str = ""
    target_problem: str = ""

    self_related: bool = False
    team_related: bool = True

    suggested_improvement: str = ""
    missing_support_type: str = "tool"   # tool/workflow/protocol/artifact/role_clarity/resource/coordination/information

    urgency: float = 0.5
    expected_benefit: str = ""
    risk_if_unaddressed: str = ""

    related_object_ids: List[str] = field(default_factory=list)
    related_issue_ids: List[str] = field(default_factory=list)
    related_agent_ids: List[str] = field(default_factory=list)
    related_episode_ids: List[str] = field(default_factory=list)
    related_channel_ids: List[str] = field(default_factory=list)

    status: str = "open"
    generated_proposal_ids: List[str] = field(default_factory=list)

    created_at_tick: int = 0
    updated_at_tick: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "wish_id": self.wish_id, "agent_id": self.agent_id,
            "source_reflection_id": self.source_reflection_id,
            "source_reflection_ids": list(self.source_reflection_ids),
            "supporting_agent_ids": list(self.supporting_agent_ids),
            "support_count": self.support_count,
            "source_episode_id": self.source_episode_id,
            "source_event_ids": list(self.source_event_ids),
            "raw_reflection_excerpt": self.raw_reflection_excerpt, "wish_type": self.wish_type,
            "fingerprint": self.fingerprint,
            "interpreted_need": self.interpreted_need, "target_problem": self.target_problem,
            "self_related": self.self_related, "team_related": self.team_related,
            "suggested_improvement": self.suggested_improvement,
            "missing_support_type": self.missing_support_type, "urgency": round(self.urgency, 3),
            "expected_benefit": self.expected_benefit, "risk_if_unaddressed": self.risk_if_unaddressed,
            "related_object_ids": list(self.related_object_ids),
            "related_issue_ids": list(self.related_issue_ids),
            "related_agent_ids": list(self.related_agent_ids),
            "related_episode_ids": list(self.related_episode_ids),
            "related_channel_ids": list(self.related_channel_ids),
            "status": self.status, "generated_proposal_ids": list(self.generated_proposal_ids),
            "created_at_tick": self.created_at_tick, "updated_at_tick": self.updated_at_tick,
        }


@dataclass
class AgentMemory:
    """Long-term cognitive state — reflections become durable here (not one-shot text)."""
    agent_id: str
    reflections: List[str] = field(default_factory=list)
    lessons_learned: List[str] = field(default_factory=list)
    unresolved_needs: List[str] = field(default_factory=list)
    repeated_blockers: List[str] = field(default_factory=list)
    commitments: List[str] = field(default_factory=list)
    concerns: List[str] = field(default_factory=list)
    # The conversation this agent is part of: what colleagues said and what it
    # said back, in the order it happened. Kept apart from the fields above
    # because those are all conclusions the agent drew, and a transcript is
    # evidence rather than a lesson already taken from it.
    #
    # Stored in full and surfaced as a window: a prompt cannot hold hundreds of
    # lines, but a memory that only kept the last handful could not answer "what
    # did we agree three days ago", which is the question a transcript exists
    # for. The parallel id list is what makes appending idempotent, and it lives
    # here rather than on the world so that B1's sprint reset clears it with the
    # rest of member-local memory instead of leaving the new team unable to
    # record anything.
    conversation: List[str] = field(default_factory=list)
    conversation_message_ids: List[str] = field(default_factory=list)
    last_reflection_tick: Optional[int] = None

    # A transcript is not a set of insights; it grows with the run and is
    # trimmed to keep the tail, not deduplicated. "sounds good" said twice is
    # two turns of a conversation.
    CONVERSATION_CAP = 400

    @staticmethod
    def _push(lst: List[str], item: str, cap: int = 12) -> None:
        if item and item not in lst:
            lst.append(item)
            if len(lst) > cap:
                del lst[:-cap]

    def record_turn(self, message_id: str, line: str) -> bool:
        """Append one turn of the transcript. Returns whether it was new."""
        message_id = str(message_id or "")
        if not message_id or not line or message_id in self.conversation_message_ids:
            return False
        self.conversation.append(line)
        self.conversation_message_ids.append(message_id)
        if len(self.conversation) > self.CONVERSATION_CAP:
            del self.conversation[: -self.CONVERSATION_CAP]
            del self.conversation_message_ids[: -self.CONVERSATION_CAP]
        return True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id, "reflections": list(self.reflections),
            "lessons_learned": list(self.lessons_learned),
            "unresolved_needs": list(self.unresolved_needs),
            "repeated_blockers": list(self.repeated_blockers),
            "commitments": list(self.commitments), "concerns": list(self.concerns),
            "conversation": list(self.conversation),
            "conversation_message_ids": list(self.conversation_message_ids),
            "last_reflection_tick": self.last_reflection_tick,
        }


@dataclass
class AgentLogEntry:
    log_id: str
    agent_id: str
    tick: int
    entry_type: str          # action / speech / reflection / memory_update / wish_created / proposal_created
    summary: str = ""
    related_event_ids: List[str] = field(default_factory=list)
    related_episode_ids: List[str] = field(default_factory=list)
    related_object_ids: List[str] = field(default_factory=list)
    raw_payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "log_id": self.log_id, "agent_id": self.agent_id, "tick": self.tick,
            "entry_type": self.entry_type, "summary": self.summary,
            "related_event_ids": list(self.related_event_ids),
            "related_episode_ids": list(self.related_episode_ids),
            "related_object_ids": list(self.related_object_ids),
            "raw_payload": dict(self.raw_payload),
        }


__all__ = ["AgentReflection", "Wish", "AgentMemory", "AgentLogEntry",
           "WISH_TYPES", "SUPPORT_TYPES", "WISH_STATUS", "NEED_TYPE_MAP",
           "canon_wish_type", "support_type_for", "make_wish_fingerprint"]
