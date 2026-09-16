"""Organization baselines for the controlled Relic experiment.

The baseline axis is independent from the OSS substrate/control axis:

* B0: one generalist founder with aggregate roster skills and the same company
  budget/tool surface. A traditional LLM chooses WHAT action to take.
* B1: an eight-person persistent role-based organization with flat role/skill
  prompts; profile coefficients, institutionalization and growth are disabled.
  A traditional LLM chooses WHAT action to take.
* B2: the same eight people, now selecting through a policy instead of letting
  the model name the next action — candidates are scored, prioritized and
  routed — with that scoring conditioned on profiles, plus individual growth.
  Still no institutions: they cannot turn a practice into a rule that outlives
  the moment.
* B3: B2 plus institutionalization. The reflection-to-proposal-to-adoption-to-
  persistence chain runs, which is what capability formation means here.

The ladder is arranged so its headline step varies one field. B1 to B2 is the
policy-mediation step and changes three (selection mode, profile conditioning,
growth): a policy that scores candidates needs weights, and a profile is what
those weights are made of, so the parts do not come apart here.

That makes B2 minus B1 an estimate of mediated action selection, not of
professional profiles. Reporting it as a profile effect overstates what the arm
manipulates — the model stops choosing the action at all. Isolating the profile
would need a further arm with B2's architecture, candidate pool and LLM call
path, differing only in role-agnostic versus profile-conditioned weights.

B2 to B3 changes ``institutionalization_enabled`` alone, so the capability
formation contrast is single-factor and can be reported as one.

Reflection sits deliberately on the institutionalization switch rather than on
its own: the chain that produces an institution begins with an agent reflecting
on what went wrong, so reflection is part of the mechanism under test, not a
confound riding alongside it.

The eight-person temporary team, which reset member-local state at each sprint
boundary, was a rung until it produced almost nothing distinguishable from B0
across a full pilot. It is now an overlay any condition can carry
(``temporary_team`` scenario parameter or ``ORG_TEMPORARY_TEAM``), so the
persistence claim stays testable without spending a rung on it.

Profile manipulations (homogeneous, role-only, shuffled, random) are NOT
organization conditions. They are post-formation intervention arms and
counterfactual-analysis transforms, and live in
``environments.org_env.experiments.persona_transforms``.

The module is deliberately independent from OSS datasets so baseline work cannot
mutate or specialize against Gitingest.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

from environments.org_env.backend.agents.seed_team import SeedMember

B0_SINGLE_AGENT_FOUNDER = "b0_single_agent_founder"
B1_PERSISTENT_ROLE_ORG = "b1_persistent_role_org"
B2_POLICY_CONDITIONED_ORG = "b2_policy_conditioned_org"
B3_RELIC_ORGANIZATION = "b3_relic_organization"

# Retired rungs. Kept resolvable so runs recorded under the previous ladder
# still load, and so their records keep reporting the condition they actually
# ran under rather than being silently relabelled as something they were not.
B1_TEMPORARY_SPECIALIST_TEAM = "b1_temporary_specialist_team"
B2_PERSISTENT_ROLE_ORG = "b2_persistent_role_org"

PROFILE_ASSIGNMENT_ALIGNED = "aligned"
PROFILE_ASSIGNMENT_MODES = frozenset(
    {
        PROFILE_ASSIGNMENT_ALIGNED,
    }
)

ACTION_SELECTION_LLM_DIRECT = "llm_direct"
ACTION_SELECTION_PROFILE_POLICY = "profile_policy"
ACTION_SELECTION_FLAT_DETERMINISTIC = "flat_deterministic"
ACTION_SELECTION_MODES = frozenset(
    {
        ACTION_SELECTION_LLM_DIRECT,
        ACTION_SELECTION_PROFILE_POLICY,
        ACTION_SELECTION_FLAT_DETERMINISTIC,
    }
)


@dataclass(frozen=True)
class OrganizationCondition:
    condition_id: str
    short_name: str
    roster_size: int
    action_selection_mode: str
    profile_conditioning_enabled: bool
    institutionalization_enabled: bool
    capability_learning_enabled: bool
    temporary_team: bool = False
    default_sprint_ticks: int = 168
    profile_assignment: str = PROFILE_ASSIGNMENT_ALIGNED
    role_mandates_enabled: bool = True

    def __post_init__(self) -> None:
        if self.action_selection_mode not in ACTION_SELECTION_MODES:
            raise ValueError(
                f"unknown action selection mode: {self.action_selection_mode!r}"
            )
        if self.profile_assignment not in PROFILE_ASSIGNMENT_MODES:
            raise ValueError(
                f"unknown profile assignment mode: {self.profile_assignment!r}"
            )
        if self.temporary_team and self.profile_assignment != PROFILE_ASSIGNMENT_ALIGNED:
            # The sprint reset rebuilds members from the canonical SEED_TEAM, which
            # would silently restore the aligned persona->role mapping mid-run.
            raise ValueError(
                "temporary_team resets restore the canonical roster and cannot "
                f"preserve profile_assignment={self.profile_assignment!r}"
            )

    def as_dict(self) -> dict:
        return {
            "condition_id": self.condition_id,
            "short_name": self.short_name,
            "roster_size": self.roster_size,
            "action_selection_mode": self.action_selection_mode,
            "profile_conditioning_enabled": self.profile_conditioning_enabled,
            "institutionalization_enabled": self.institutionalization_enabled,
            "capability_learning_enabled": self.capability_learning_enabled,
            "temporary_team": self.temporary_team,
            "default_sprint_ticks": self.default_sprint_ticks,
            "profile_assignment": self.profile_assignment,
            "role_mandates_enabled": self.role_mandates_enabled,
        }


CONDITIONS = {
    B0_SINGLE_AGENT_FOUNDER: OrganizationCondition(
        condition_id=B0_SINGLE_AGENT_FOUNDER,
        short_name="b0",
        roster_size=1,
        action_selection_mode=ACTION_SELECTION_LLM_DIRECT,
        profile_conditioning_enabled=False,
        institutionalization_enabled=False,
        capability_learning_enabled=False,
    ),
    B1_PERSISTENT_ROLE_ORG: OrganizationCondition(
        condition_id=B1_PERSISTENT_ROLE_ORG,
        short_name="b1",
        roster_size=8,
        action_selection_mode=ACTION_SELECTION_LLM_DIRECT,
        profile_conditioning_enabled=False,
        institutionalization_enabled=False,
        capability_learning_enabled=False,
    ),
    B2_POLICY_CONDITIONED_ORG: OrganizationCondition(
        condition_id=B2_POLICY_CONDITIONED_ORG,
        short_name="b2",
        roster_size=8,
        action_selection_mode=ACTION_SELECTION_PROFILE_POLICY,
        profile_conditioning_enabled=True,
        institutionalization_enabled=False,
        capability_learning_enabled=True,
    ),
    B3_RELIC_ORGANIZATION: OrganizationCondition(
        condition_id=B3_RELIC_ORGANIZATION,
        short_name="b3",
        roster_size=8,
        action_selection_mode=ACTION_SELECTION_PROFILE_POLICY,
        profile_conditioning_enabled=True,
        institutionalization_enabled=True,
        capability_learning_enabled=True,
    ),
}

# Not rungs. Resolvable so an older run's records and checkpoints still load.
RETIRED_CONDITIONS = {
    B1_TEMPORARY_SPECIALIST_TEAM: OrganizationCondition(
        condition_id=B1_TEMPORARY_SPECIALIST_TEAM,
        short_name="b1_temporary",
        roster_size=8,
        action_selection_mode=ACTION_SELECTION_LLM_DIRECT,
        profile_conditioning_enabled=False,
        institutionalization_enabled=False,
        capability_learning_enabled=False,
        temporary_team=True,
    ),
    B2_PERSISTENT_ROLE_ORG: OrganizationCondition(
        condition_id=B2_PERSISTENT_ROLE_ORG,
        short_name="b2_persistent_llm_direct",
        roster_size=8,
        action_selection_mode=ACTION_SELECTION_LLM_DIRECT,
        profile_conditioning_enabled=False,
        institutionalization_enabled=False,
        capability_learning_enabled=False,
    ),
}

_ALIASES = {
    "b0": B0_SINGLE_AGENT_FOUNDER,
    "single": B0_SINGLE_AGENT_FOUNDER,
    "single_agent": B0_SINGLE_AGENT_FOUNDER,
    "b1": B1_PERSISTENT_ROLE_ORG,
    "persistent": B1_PERSISTENT_ROLE_ORG,
    "persistent_role": B1_PERSISTENT_ROLE_ORG,
    "b2": B2_POLICY_CONDITIONED_ORG,
    "policy": B2_POLICY_CONDITIONED_ORG,
    "policy_conditioned": B2_POLICY_CONDITIONED_ORG,
    "b3": B3_RELIC_ORGANIZATION,
    "full": B3_RELIC_ORGANIZATION,
    "relic": B3_RELIC_ORGANIZATION,
    # Retired rungs keep their own names and resolve to what they actually were.
    "temporary": B1_TEMPORARY_SPECIALIST_TEAM,
    "temporary_team": B1_TEMPORARY_SPECIALIST_TEAM,
}


def resolve_condition(value: str | None) -> OrganizationCondition:
    """Resolve a strict condition id/alias; an omitted value means current B3.

    Retired rungs resolve to their own definition rather than to whichever rung
    now carries their old short name. A run recorded as
    ``b2_persistent_role_org`` was an LLM-direct organization, and reading it
    back as today's policy-conditioned B2 would misreport what was measured.
    """
    raw = (value or B3_RELIC_ORGANIZATION).strip().lower()
    condition_id = _ALIASES.get(raw, raw)
    if condition_id in CONDITIONS:
        return CONDITIONS[condition_id]
    if condition_id in RETIRED_CONDITIONS:
        return RETIRED_CONDITIONS[condition_id]
    allowed = ", ".join(CONDITIONS)
    raise ValueError(f"unknown experiment condition {value!r}; expected one of: {allowed}")


def aggregate_numeric_attribute(
    members: Sequence[SeedMember],
    attribute: str,
    *,
    reducer: str,
) -> dict[str, float]:
    mappings = [getattr(member, attribute) or {} for member in members]
    keys = sorted({key for mapping in mappings for key in mapping})
    out: dict[str, float] = {}
    for key in keys:
        values = [float(mapping.get(key, 0.0)) for mapping in mappings]
        value = max(values) if reducer == "max" else sum(values) / max(1, len(values))
        out[key] = round(value, 4)
    return out


def aggregate_mixed_attribute(
    members: Sequence[SeedMember], attribute: str, label: str
) -> dict:
    mappings = [getattr(member, attribute) or {} for member in members]
    keys = sorted({key for mapping in mappings for key in mapping})
    out: dict = {}
    for key in keys:
        numeric = [
            float(mapping[key])
            for mapping in mappings
            if isinstance(mapping.get(key), (int, float))
        ]
        if numeric:
            out[key] = round(sum(numeric) / len(numeric), 4)
    out["tone"] = label
    return out


def aggregate_single_founder(members: Sequence[SeedMember]) -> SeedMember:
    """Create B0's generalist without changing the canonical seed roster.

    Skill coverage takes the per-domain maximum so the single agent receives the
    team's aggregate tool competence. Personality/work rhythm are averaged to
    avoid selecting one specialist persona as the baseline.
    """
    if not members:
        raise ValueError("cannot build B0 founder from an empty roster")
    founder = next((member for member in members if member.is_founder), members[0])
    return SeedMember(
        agent_id=founder.agent_id,
        agent_name="Single-Agent Founder",
        codename="Generalist",
        role="founder",
        initial_identity=(
            "Single-agent generalist baseline with aggregate team information, "
            "tool competence, and company budget."
        ),
        profile=aggregate_numeric_attribute(members, "profile", reducer="mean"),
        skills=aggregate_numeric_attribute(members, "skills", reducer="max"),
        failure_modes=["single_point_of_failure", "context_overload", "no_independent_review"],
        communication_style=aggregate_mixed_attribute(
            members, "communication_style", "generalist_founder"
        ),
        work_rhythm=aggregate_mixed_attribute(
            members, "work_rhythm", "generalist_founder"
        ),
        is_founder=True,
    )


def roster_for_condition(
    condition: OrganizationCondition,
    members: Sequence[SeedMember],
    *,
    requested_size: int,
    condition_explicit: bool,
    seed: int | None = None,
) -> list[SeedMember]:
    """Return a fresh roster while preserving legacy custom-size B3 sessions."""
    if condition.condition_id == B0_SINGLE_AGENT_FOUNDER:
        return [aggregate_single_founder(members)]
    size = condition.roster_size if condition_explicit else requested_size
    return list(members[: max(1, min(int(size), len(members)))])


def organization_condition_ids() -> tuple[str, ...]:
    """The preregistered organization ladder, in ladder order."""
    return (
        B0_SINGLE_AGENT_FOUNDER,
        B1_PERSISTENT_ROLE_ORG,
        B2_POLICY_CONDITIONED_ORG,
        B3_RELIC_ORGANIZATION,
    )


def baseline_condition_ids() -> tuple[str, ...]:
    return (
        B0_SINGLE_AGENT_FOUNDER,
        B1_PERSISTENT_ROLE_ORG,
        B2_POLICY_CONDITIONED_ORG,
    )


__all__ = [
    "ACTION_SELECTION_FLAT_DETERMINISTIC",
    "ACTION_SELECTION_LLM_DIRECT",
    "ACTION_SELECTION_MODES",
    "ACTION_SELECTION_PROFILE_POLICY",
    "B0_SINGLE_AGENT_FOUNDER",
    "B1_PERSISTENT_ROLE_ORG",
    "B1_TEMPORARY_SPECIALIST_TEAM",
    "B2_PERSISTENT_ROLE_ORG",
    "B2_POLICY_CONDITIONED_ORG",
    "B3_RELIC_ORGANIZATION",
    "CONDITIONS",
    "RETIRED_CONDITIONS",
    "OrganizationCondition",
    "PROFILE_ASSIGNMENT_ALIGNED",
    "PROFILE_ASSIGNMENT_MODES",
    "aggregate_mixed_attribute",
    "aggregate_numeric_attribute",
    "aggregate_single_founder",
    "baseline_condition_ids",
    "organization_condition_ids",
    "resolve_condition",
    "roster_for_condition",
]
