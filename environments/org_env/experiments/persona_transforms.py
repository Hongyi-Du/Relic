"""Persona transforms for post-formation interventions and counterfactuals.

These are deliberately NOT organization conditions. The organization axis is
B0-B3 (``config/baseline_conditions.py``); manipulating who holds which persona
is an intervention applied to an already-formed organization, or a
counterfactual used by profile-causality analysis.

Every transform keeps the governance topology fixed: ``agent_id``, ``role`` and
``is_founder`` stay with the slot, and only the persona payload moves. Role
labels are governance topology rather than persona - at least ten role-literal
gates decide who may approve a release, publish, merge or speak externally, so
rewriting them would amputate the organization instead of altering its people.
"""

from __future__ import annotations

import random
from typing import Sequence

from environments.org_env.backend.agents.seed_team import SeedMember
from environments.org_env.config.baseline_conditions import (
    aggregate_mixed_attribute,
    aggregate_numeric_attribute,
)


def homogenize_roster(members: Sequence[SeedMember]) -> list[SeedMember]:
    """Remove persona AND skill differentiation while keeping the organization.

    Every member receives the same averaged profile, skills, communication
    style, and work rhythm. Stable ids, founder flags and role labels remain
    fixed. Callers that want to also suppress the per-role behavioural mandate
    must set ``role_mandates_enabled=False``; the mandate text is a behavioural
    prior, not governance topology, and leaving it on would hand eight
    supposedly identical agents eight different mandates.
    """
    if not members:
        raise ValueError("cannot homogenize an empty roster")
    profile = aggregate_numeric_attribute(members, "profile", reducer="mean")
    skills = aggregate_numeric_attribute(members, "skills", reducer="mean")
    communication_style = aggregate_mixed_attribute(
        members, "communication_style", "homogeneous_generalist"
    )
    work_rhythm = aggregate_mixed_attribute(
        members, "work_rhythm", "homogeneous_generalist"
    )
    return [
        SeedMember(
            agent_id=member.agent_id,
            agent_name=f"Homogeneous Agent {index + 1}",
            codename=f"Homogeneous{index + 1}",
            role=member.role,
            initial_identity=(
                "Homogeneous condition: identical professional tendencies, "
                "skills, communication style, and work rhythm across the team; "
                "organizational roles and governance topology are unchanged."
            ),
            profile=dict(profile),
            skills=dict(skills),
            failure_modes=["homogeneous_generalist"],
            communication_style=dict(communication_style),
            work_rhythm=dict(work_rhythm),
            is_founder=member.is_founder,
        )
        for index, member in enumerate(members)
    ]


def flatten_profiles_for_role_only(
    members: Sequence[SeedMember],
) -> list[SeedMember]:
    """Keep role/skill specialization but remove the persona decision prior."""
    if not members:
        raise ValueError("cannot flatten an empty roster")
    profile = aggregate_numeric_attribute(members, "profile", reducer="mean")
    communication_style = aggregate_mixed_attribute(
        members, "communication_style", "role_only"
    )
    work_rhythm = aggregate_mixed_attribute(members, "work_rhythm", "role_only")
    return [
        SeedMember(
            agent_id=member.agent_id,
            agent_name=f"Role-only {member.role}",
            codename=f"Role{index + 1}",
            role=member.role,
            initial_identity=(
                f"Role-only condition for {member.role}; role skills are "
                "available but professional persona priors are held constant."
            ),
            profile=dict(profile),
            skills=dict(member.skills),
            failure_modes=["role_only_baseline"],
            communication_style=dict(communication_style),
            work_rhythm=dict(work_rhythm),
            is_founder=member.is_founder,
        )
        for index, member in enumerate(members)
    ]


def shuffle_profile_assignment(
    members: Sequence[SeedMember],
    *,
    seed: int,
) -> list[SeedMember]:
    """Permute persona payloads across role slots, keeping the org structure fixed.

    The persona payload travels as one unit: name, codename, identity text,
    profile, skills, failure modes, communication style, and work rhythm. The
    identity text moves with the persona deliberately - the agent describes
    itself per its persona while occupying a mismatched role slot, which is
    exactly the misalignment this arm measures.

    The permutation is derived from a namespaced RNG so it is seed-reproducible
    and independent of any other consumer of the run seed. The identity
    permutation is rejected so the arm can never silently degenerate into the
    aligned roster.
    """
    if len(members) < 2:
        raise ValueError("shuffled profile assignment needs at least 2 members")
    rng = random.Random(f"profile_assignment:shuffled:{int(seed)}")
    aligned = list(range(len(members)))
    indices = list(aligned)
    while indices == aligned:
        rng.shuffle(indices)
    shuffled: list[SeedMember] = []
    for slot, source_index in zip(members, indices):
        persona = members[source_index]
        shuffled.append(
            SeedMember(
                agent_id=slot.agent_id,
                agent_name=persona.agent_name,
                codename=persona.codename,
                role=slot.role,
                initial_identity=persona.initial_identity,
                profile=dict(persona.profile),
                skills=dict(persona.skills),
                failure_modes=list(persona.failure_modes),
                communication_style=dict(persona.communication_style),
                work_rhythm=dict(persona.work_rhythm),
                is_founder=slot.is_founder,
            )
        )
    return shuffled


def _mix_persona_mapping(
    members: Sequence[SeedMember],
    attribute: str,
    weights: Sequence[float],
    dominant: SeedMember,
) -> dict:
    """Convex-mix one persona mapping under shared mixture weights.

    A key missing from a member counts as 0.0, matching the policy scorer's
    ``profile_or_skill.get(trait, 0.0)`` semantics. Non-numeric values (e.g.
    the communication ``tone`` label) cannot be averaged and are taken from the
    dominant mixture component instead.
    """
    mappings = [getattr(member, attribute) or {} for member in members]
    out: dict = {}
    for key in sorted({key for mapping in mappings for key in mapping}):
        values = [mapping.get(key, 0.0) for mapping in mappings]
        if all(isinstance(value, (int, float)) for value in values):
            out[key] = round(
                sum(weight * float(value) for weight, value in zip(weights, values)), 4
            )
        else:
            dominant_value = (getattr(dominant, attribute) or {}).get(key)
            if dominant_value is not None:
                out[key] = dominant_value
    return out


def synthesize_random_roster(
    members: Sequence[SeedMember],
    *,
    seed: int,
) -> list[SeedMember]:
    """Synthesize a neutral-prior persona for every role slot.

    Each slot receives a synthetic persona built from one draw of Dirichlet(1)
    mixture weights over the seed corpus - uniform on the simplex, i.e. THE
    neutral prior, with no tunable concentration parameter. All numeric
    mappings are convex mixtures under the *same* weights, so the correlation
    structure between traits within a persona is inherited from the seed corpus
    rather than destroyed by per-key independent sampling. Failure modes and
    tone come from the dominant mixture component, and the identity text is
    templated from the sampled numbers themselves, so the self-description can
    never contradict the numeric profile.
    """
    if len(members) < 2:
        raise ValueError("random profile synthesis needs at least 2 members")
    rng = random.Random(f"profile_assignment:random:{int(seed)}")
    synthesized: list[SeedMember] = []
    for slot_index, slot in enumerate(members):
        draws = [rng.expovariate(1.0) for _ in members]
        total = sum(draws)
        weights = [draw / total for draw in draws]
        dominant = members[max(range(len(members)), key=lambda i: weights[i])]
        profile = _mix_persona_mapping(members, "profile", weights, dominant)
        skills = _mix_persona_mapping(members, "skills", weights, dominant)
        top_traits = sorted(profile, key=profile.get, reverse=True)[:3]
        top_skills = sorted(skills, key=skills.get, reverse=True)[:3]
        synthesized.append(
            SeedMember(
                agent_id=slot.agent_id,
                agent_name=f"Synthetic Member {slot_index + 1}",
                codename=f"Synth{slot_index + 1}",
                role=slot.role,
                initial_identity=(
                    "Randomized-profile control persona synthesized from the seed "
                    "corpus under a neutral Dirichlet(1) prior; strongest "
                    "tendencies: " + ", ".join(top_traits) + "; strongest skills: "
                    + ", ".join(top_skills) + "."
                ),
                profile=profile,
                skills=skills,
                failure_modes=list(dominant.failure_modes),
                communication_style=_mix_persona_mapping(
                    members, "communication_style", weights, dominant
                ),
                work_rhythm=_mix_persona_mapping(
                    members, "work_rhythm", weights, dominant
                ),
                is_founder=slot.is_founder,
            )
        )
    return synthesized


__all__ = [
    "flatten_profiles_for_role_only",
    "homogenize_roster",
    "shuffle_profile_assignment",
    "synthesize_random_roster",
]
