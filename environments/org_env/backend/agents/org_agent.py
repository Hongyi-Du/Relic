"""OrgEnv internal agents — full lived agents (DESIGN env_org §33).

Internal company agents run the complete Relic Core pipeline (driven
through the runtime_adapter). This holds env-side per-agent domain state +
seeding. SKELETON: identity + domain-state container only (Stage O2 adds
ProfileState seeding + skill/role assignment).
"""
from __future__ import annotations

from typing import Dict, List, Optional

from relic.core.domain import DomainAgentState, VitalState
from environments.org_env.growth.objects import effective_skill
from environments.org_env.backend.agents.work_state import (
    ORG_VITALS,
    AgentOrgState,
    AgentWorkState,
    org_vital_defaults,
)


def new_org_vitals() -> VitalState:
    # O1.6: full work + organizational vital set (attention=1.0, morale=0.6,
    # trust_in_company=0.7, role_clarity=0.4, perceived_recognition=0.5, rest 0).
    return VitalState(variables=org_vital_defaults())


class OrgAgent:
    """Internal company agent (full lived). O1: carries a rich persona (profile /
    failure_modes / communication_style / work_rhythm) that the OrgPolicy scorer
    consumes so persona actually changes behaviour."""

    def __init__(self, agent_id: str, name: str = "", role: str = "engineer",
                 codename: str = "", initial_identity: str = "",
                 skills: Optional[Dict[str, float]] = None,
                 profile: Optional[Dict[str, float]] = None,
                 failure_modes: Optional[List[str]] = None,
                 communication_style: Optional[Dict[str, object]] = None,
                 work_rhythm: Optional[Dict[str, object]] = None,
                 permissions: Optional[List[str]] = None,
                 is_founder: bool = False):
        self.id = agent_id
        self.name = name or agent_id
        self.codename = codename
        self.role = role
        self.initial_identity = initial_identity
        self.skills = dict(skills or {})
        self.profile = dict(profile or {})
        self.failure_modes = list(failure_modes or [])
        self.communication_style = dict(communication_style or {})
        self.work_rhythm = dict(work_rhythm or {})
        self.permissions = list(permissions or [])
        self.is_founder = is_founder
        self.active_tasks: List[str] = []
        self.personal_workspace_id = f"pw_{agent_id}"
        self.sandbox_id = f"sandbox_{agent_id}"
        self.current_status = "available"
        # Internal Growth Module (§1): persistent skill (grows in self.skills) + domain
        # reputation + derived informal authority + go-to tags. Core profile stays fixed.
        from environments.org_env.growth.objects import new_authority, new_reputation
        self.reputation: Dict[str, float] = new_reputation()
        self.authority: Dict[str, float] = new_authority()
        self.go_to_tags: List[str] = []
        self._domain_events: Dict[str, list] = {}      # domain -> [(tick, outcome>0)]
        self.vitals = new_org_vitals()   # persistent — accumulates fatigue/stress
        # O1.6 structured views over the vitals bag + scheduling/aux-speech state.
        aux_slots = self._default_aux_slots()
        self.work_state = AgentWorkState(self.vitals.variables, default_aux_slots=aux_slots)
        self.org_state = AgentOrgState(self.vitals.variables)

    def _default_aux_slots(self) -> int:
        """High-communication agents get 2 aux-speech slots/tick, terse ones 1
        (spec §5.3). Communication *volume* ~ warmth / assertiveness / social_tact
        (not directness — a terse agent can be very direct but say little)."""
        def _num(d: dict, key: str) -> float:
            v = d.get(key, 0.0)
            return float(v) if isinstance(v, (int, float)) else 0.0
        vol = max(_num(self.communication_style, "warmth"),
                  _num(self.communication_style, "assertiveness"),
                  _num(self.profile, "social_tact"))
        return 2 if vol >= 0.7 else 1

    @classmethod
    def from_seed(cls, member) -> "OrgAgent":
        return cls(agent_id=member.agent_id, name=member.agent_name, role=member.role,
                   codename=member.codename, initial_identity=member.initial_identity,
                   skills=member.skills, profile=member.profile,
                   failure_modes=member.failure_modes,
                   communication_style=member.communication_style,
                   work_rhythm=member.work_rhythm, is_founder=member.is_founder)

    def trait(self, name: str, default: float = 0.0) -> float:
        return float(self.profile.get(name, default))

    def skill(self, name: str, default: float = 0.0) -> float:
        # Alias-resolved: growth writes canonical names, seeds may carry aliases;
        # a raw dict read here freezes every aliased consumption point at its seed.
        return effective_skill(self.skills, name, default)

    def domain_state(self) -> DomainAgentState:
        return DomainAgentState(
            agent_id=self.id, agent_name=self.name, role=self.role,
            vitals=self.vitals, skills=dict(self.skills),
            active_tasks_or_plans=list(self.active_tasks),
            permissions=list(self.permissions),
            private_state={"profile": dict(self.profile), "codename": self.codename,
                           "failure_modes": list(self.failure_modes),
                           "status": self.current_status},
        )


__all__ = ["ORG_VITALS", "new_org_vitals", "OrgAgent"]
