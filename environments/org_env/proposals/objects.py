"""Proposal / ToolSpec / ProtocolSpec — the world objects the LLM cognitive layer
DRAFTS but only the System creates (after validation + approval/adoption).

A Wish -> Proposal (draft) -> evaluated -> approved -> adopted -> ToolSpec /
ProtocolSpec in the world. The LLM never instantiates these directly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

PROPOSAL_TYPES = ("task_proposal", "tool_proposal", "workflow_proposal", "protocol_proposal",
                  "role_proposal", "policy_repair_proposal", "artifact_template_proposal")
PROPOSAL_STATUS = ("draft", "under_review", "approved", "rejected", "adopted", "implemented", "failed")


def ensure_list(x: Any) -> List[str]:
    """Normalize an LLM-provided value into a list of strings WITHOUT splitting a
    string into characters (preflight v3 §2): a bare string becomes [string]."""
    if x is None:
        return []
    if isinstance(x, str):
        s = x.strip()
        return [s] if s else []
    if isinstance(x, (list, tuple, set)):
        out = []
        for v in x:
            if isinstance(v, str):
                v = v.strip()
                if v:
                    out.append(v)
            elif v is not None:
                out.append(str(v))
        return out
    return [str(x)]


@dataclass
class Proposal:
    proposal_id: str
    proposal_type: str
    title: str = ""
    summary: str = ""
    proposer_agent_id: Optional[str] = None

    source_wish_id: Optional[str] = None
    # Every wish a clustered proposal rests on. A rule reached by grouping what
    # members kept asking for and a rule read off the environment's fixed
    # catalogue arrive at the registry identical: the clustering pass names the
    # wishes it grouped, and the proposal it produces recorded none of them, so
    # the run that mattered could not say which of its three adopted rules the
    # organization had reached for itself.
    source_wish_ids: List[str] = field(default_factory=list)
    source_reflection_id: Optional[str] = None
    source_episode_id: Optional[str] = None
    source_episode_ids: List[str] = field(default_factory=list)   # v11: full episode cluster
    source_event_ids: List[str] = field(default_factory=list)

    target_problem: str = ""
    proposed_solution: str = ""

    required_actions: List[str] = field(default_factory=list)
    required_capabilities: List[str] = field(default_factory=list)
    required_artifacts: List[str] = field(default_factory=list)
    required_participants: List[str] = field(default_factory=list)

    affected_agents: List[str] = field(default_factory=list)
    affected_objects: List[str] = field(default_factory=list)
    affected_protocols: List[str] = field(default_factory=list)

    expected_benefits: List[str] = field(default_factory=list)
    expected_costs: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    failure_modes: List[str] = field(default_factory=list)

    feasibility_score: Optional[float] = None
    usefulness_score: Optional[float] = None
    risk_score: Optional[float] = None
    adoption_score: Optional[float] = None
    suggested_revision: str = ""

    family: str = ""                                               # v11 P2: concern family (anti-sprawl)
    status: str = "draft"
    approval_required_from: List[str] = field(default_factory=list)
    approved_by: List[str] = field(default_factory=list)
    supporters: List[str] = field(default_factory=list)            # v11: who backed it (telemetry)
    rejected_by: List[str] = field(default_factory=list)
    rejection_reason: str = ""
    object_created_id: Optional[str] = None
    adopted_tick: Optional[int] = None                              # v11: adoption telemetry
    impact: Dict[str, Any] = field(default_factory=dict)            # v11: post-adoption impact metrics
    # v6 P0.2: when a protocol proposal is semantically covered by an already-adopted
    # protocol, it is routed as an AMENDMENT (revision) of that protocol instead of
    # minting a new spec — keeps governance at "1 adopted protocol per domain + N revisions".
    amends_protocol_id: Optional[str] = None
    # policy repair (self-correction): a proposal can RELAX or DEPRECATE a self-binding / harmful
    # adopted protocol instead of only strengthening it. repair_kind in {"relax","deprecate"};
    # repair_target_protocol_id names the spec to repair (else the covering protocol is used).
    repair_kind: Optional[str] = None
    repair_target_protocol_id: Optional[str] = None

    created_at_tick: int = 0
    updated_at_tick: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {k: (list(v) if isinstance(v, list) else v) for k, v in self.__dict__.items()}


@dataclass
class ToolSpec:
    tool_id: str
    name: str
    description: str = ""
    created_from_proposal_id: Optional[str] = None
    creator_agent_id: Optional[str] = None
    # v11: lineage so a tool can only become a company skill if it traces to a wish/episode
    source_wish_id: Optional[str] = None
    source_episode_id: Optional[str] = None
    source_episode_ids: List[str] = field(default_factory=list)
    tool_type: str = "composed_action_tool"   # composed_action_tool/artifact_tool/checking_tool/workflow_tool/sandbox_tool
    input_schema: Dict[str, Any] = field(default_factory=dict)
    output_schema: Dict[str, Any] = field(default_factory=dict)
    required_actions: List[str] = field(default_factory=list)
    required_capabilities: List[str] = field(default_factory=list)
    required_permissions: List[str] = field(default_factory=list)
    risk_tags: List[str] = field(default_factory=list)
    validation_rules: List[str] = field(default_factory=list)
    callable_by_roles: List[str] = field(default_factory=list)
    callable_by_agents: List[str] = field(default_factory=list)
    # v11 P2: family + folding — a same-family near-duplicate is folded in as support
    # instead of minting another tool (per-family active-tool cap).
    family: str = ""
    supporters: List[str] = field(default_factory=list)
    support_count: int = 0
    folded_proposal_ids: List[str] = field(default_factory=list)
    status: str = "active"                     # proposed/active/deprecated/disabled
    created_at_tick: int = 0
    adopted_at_tick: int = 0                    # v8 #4: parity with ProtocolSpec / company skills
    updated_at_tick: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {k: (list(v) if isinstance(v, list) else (dict(v) if isinstance(v, dict) else v))
                for k, v in self.__dict__.items()}


@dataclass
class ProtocolSpec:
    protocol_id: str
    name: str = ""
    created_from_proposal_id: Optional[str] = None
    source_episode_id: Optional[str] = None
    source_episode_ids: List[str] = field(default_factory=list)   # v11: full episode cluster
    source_wish_id: Optional[str] = None
    source_wish_ids: List[str] = field(default_factory=list)   # the cluster it was reached from
    source_reflection_id: Optional[str] = None
    trigger_condition: str = ""
    required_steps: List[str] = field(default_factory=list)
    required_fields: List[str] = field(default_factory=list)
    enforcement_rule: str = ""
    violation_condition: str = ""
    exception_rule: Optional[str] = None
    # v11 P3: full institution structure (not just a name) — what evidence motivated it,
    # where it applies, who is responsible, how we know it works, when it sunsets.
    family: str = ""
    problem_evidence: List[str] = field(default_factory=list)      # episode / event / failed-gate ids
    scope: str = ""                                                # which workflow / artifacts / roles
    responsible_roles: Dict[str, List[str]] = field(default_factory=dict)  # executor/reviewer/approver
    success_metric: str = ""
    enforcement_action: str = ""
    sunset_rule: str = ""
    affected_agents: List[str] = field(default_factory=list)
    affected_actions: List[str] = field(default_factory=list)
    affected_artifacts: List[str] = field(default_factory=list)
    benefits: List[str] = field(default_factory=list)
    costs: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    status: str = "proposed"                   # proposed/adopted/rejected/violated/enforced/deprecated
    proposed_by: Optional[str] = None
    adopted_by: List[str] = field(default_factory=list)
    enforced_by: List[str] = field(default_factory=list)
    created_at_tick: int = 0
    adopted_at_tick: Optional[int] = None
    # v6 P0.2: subsequent semantically-covered proposals amend this spec rather than
    # spawning a near-duplicate one — each amendment bumps `revision` and appends a row.
    revision: int = 0
    last_revised_tick: Optional[int] = None     # v8d P1a: latest amendment (adopted_at_tick stays first)
    revisions: List[Dict[str, Any]] = field(default_factory=list)
    superseded_proposal_ids: List[str] = field(default_factory=list)
    # Internal Pipeline spec #4: lifecycle depth — an adopted protocol must actually be
    # used/enforced to count as a company skill (not just created).
    use_count: int = 0
    enforcement_count: int = 0
    violation_count: int = 0
    last_used_tick: Optional[int] = None
    affected_action_ids: List[str] = field(default_factory=list)
    affected_task_ids: List[str] = field(default_factory=list)
    # v8h P1: event refs so protocol_specs shows the SAME use/enforcement evidence as the
    # registry + company_skills (no "enforced 2x" with enforcement_events == null elsewhere).
    use_event_ids: List[str] = field(default_factory=list)
    enforcement_event_ids: List[str] = field(default_factory=list)
    violation_event_ids: List[str] = field(default_factory=list)

    def declared_action_ids(self) -> set:
        """Normalize declared affected_actions (validated action-id strings on the live
        path, but historically also dicts / free-case text from LLM output) into a set
        of action-id tokens — the structured channel that use AND enforcement crediting
        match against, so wording never gates whether a protocol can accrue evidence."""
        out = set()
        for a in (self.affected_actions or []):
            if isinstance(a, dict):
                a = a.get("action") or a.get("action_type") or a.get("name") or ""
            a = str(a).strip().lower().replace(" ", "_").replace("-", "_")
            if a:
                out.add(a)
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {k: (list(v) if isinstance(v, list) else v) for k, v in self.__dict__.items()}


__all__ = ["Proposal", "ToolSpec", "ProtocolSpec", "PROPOSAL_TYPES", "PROPOSAL_STATUS"]
