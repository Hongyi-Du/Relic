"""Affordance Synthesizer + Verifier (design doc §12).

Core principle (§12.1): **Wish creates affordance, not outcome.** A Need does
not produce a social structure; it (maybe) unlocks an *attemptable* action
that still has preconditions, cost and failure modes.

Design decision (per review): we do NOT do open-vocabulary action synthesis.
Instead a **closed template library** keyed by NeedType (§11.2's 9 need types)
maps each need to one or more pre-declared, always-groundable affordance
templates with slot-filled parameters. This keeps every synthesized affordance
decomposable into registered primitives — exactly what the verifier (§12.3)
requires — while still feeling open-ended to the agent.

To "fill in later": register richer templates in ``DEFAULT_TEMPLATES`` and
tighten ``AffordanceVerifier`` checks against the live primitive registry +
institution graph.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List

from agent_sdk.lived.core.contracts import (
    AffordanceProposal,
    Need,
    NeedType,
    VerifierVerdict,
)


@dataclass
class AffordanceTemplate:
    """A pre-declared affordance recipe (§12.2). ``builder`` turns a concrete
    Need into a populated AffordanceProposal (slot filling)."""
    name: str
    need_type: NeedType
    primitive_decomposition: List[str]
    required_resources: Dict[str, int] = field(default_factory=dict)
    cost: Dict[str, float] = field(default_factory=dict)
    failure_modes: List[str] = field(default_factory=list)
    # When True the verifier routes it to the institution graph as a pending
    # proposal instead of unlocking it directly (e.g. rules need adoption).
    needs_group_adoption: bool = False


# §12.2 example mapping (storage->build_granary, fairness->ledger/ration, ...).
DEFAULT_TEMPLATES: Dict[NeedType, List[AffordanceTemplate]] = {
    NeedType.STORAGE: [
        AffordanceTemplate(
            name="build_granary",
            need_type=NeedType.STORAGE,
            primitive_decomposition=["gather", "build", "store"],
            required_resources={"wood": 10, "stone": 4},
            cost={"energy": 20.0, "time": 5.0},
            failure_modes=["insufficient_materials", "no_valid_site", "interrupted"],
        ),
    ],
    NeedType.FAIRNESS: [
        AffordanceTemplate(
            name="create_contribution_ledger",
            need_type=NeedType.FAIRNESS,
            primitive_decomposition=["record"],
            cost={"energy": 3.0},
            failure_modes=["no_scribe_skill"],
            needs_group_adoption=True,
        ),
        AffordanceTemplate(
            name="propose_rationing_rule",
            need_type=NeedType.FAIRNESS,
            primitive_decomposition=["propose_rule"],
            failure_modes=["rejected_by_vote"],
            needs_group_adoption=True,
        ),
    ],
    NeedType.TEACHING: [
        AffordanceTemplate(
            name="assign_apprentice",
            need_type=NeedType.TEACHING,
            primitive_decomposition=["teach", "assign_role"],
            failure_modes=["no_willing_student"],
        ),
    ],
    NeedType.LEADERSHIP: [
        AffordanceTemplate(
            name="call_council_meeting",
            need_type=NeedType.LEADERSHIP,
            primitive_decomposition=["broadcast", "propose_rule"],
            failure_modes=["no_quorum"],
            needs_group_adoption=True,
        ),
    ],
    NeedType.RECORDING: [
        AffordanceTemplate(
            name="record_keeping_post",
            need_type=NeedType.RECORDING,
            primitive_decomposition=["record"],
            failure_modes=["no_scribe_skill"],
        ),
    ],
    # TRADE / DEFENSE / PUNISHMENT / COORDINATION: register templates as the
    # social-systems layer (design §6-9) lands.
}


class AffordanceSynthesizer:
    """Need -> [AffordanceProposal] via the template library (§12.2).

    Satisfies ``agent_sdk.lived.core.ports.AffordanceSynthPort``.
    """

    def __init__(self, templates: Dict[NeedType, List[AffordanceTemplate]] | None = None):
        self.templates = templates if templates is not None else DEFAULT_TEMPLATES

    def synthesize(self, *, need: Need, state: Any = None) -> List[AffordanceProposal]:
        out: List[AffordanceProposal] = []
        for tpl in self.templates.get(need.need_type, []):
            out.append(AffordanceProposal(
                name=tpl.name,
                need=need,
                primitive_decomposition=list(tpl.primitive_decomposition),
                required_resources={**tpl.required_resources, **need.required_resources},
                cost=dict(tpl.cost),
                failure_modes=list(tpl.failure_modes),
                observable=True,
            ))
        return out


class AffordanceVerifier:
    """Gate proposals against the §12.3 checklist.

    The checklist (must all hold to unlock):
      1. triggered by a real env problem  (delegated to ``problem_predicate``)
      2. decomposable into registered primitives
      3. has cost / required resources
      4. has at least one failure mode
      5. does NOT directly create a social outcome (only an attemptable action)
      6. observable / recordable
      7. (group-adoption affordances) -> PENDING_PROPOSAL, not direct unlock

    SCAFFOLD: ``known_primitives`` defaults permissive; pass the live registry
    to enforce check #2 strictly.
    """

    def __init__(
        self,
        known_primitives: set[str] | None = None,
        problem_predicate: Callable[[AffordanceProposal, Any], bool] | None = None,
    ):
        self.known_primitives = known_primitives
        self.problem_predicate = problem_predicate

    def verify(self, proposal: AffordanceProposal, *, state: Any = None,
               needs_group_adoption: bool = False) -> AffordanceProposal:
        # 1. triggered by a real problem
        if self.problem_predicate is not None and not self.problem_predicate(proposal, state):
            proposal.verdict = VerifierVerdict.REJECTED
            proposal.reject_reason = "not triggered by an active environment problem"
            return proposal
        # 2. decomposable into primitives
        if not proposal.primitive_decomposition:
            proposal.verdict = VerifierVerdict.REJECTED
            proposal.reject_reason = "empty primitive decomposition"
            return proposal
        if self.known_primitives is not None:
            unknown = [p for p in proposal.primitive_decomposition if p not in self.known_primitives]
            if unknown:
                proposal.verdict = VerifierVerdict.REJECTED
                proposal.reject_reason = f"unknown primitives: {unknown}"
                return proposal
        # 3. has cost or required resources
        if not proposal.cost and not proposal.required_resources:
            proposal.verdict = VerifierVerdict.REJECTED
            proposal.reject_reason = "no cost / required resources (free outcome)"
            return proposal
        # 4. has a failure mode
        if not proposal.failure_modes:
            proposal.verdict = VerifierVerdict.REJECTED
            proposal.reject_reason = "no declared failure mode"
            return proposal
        # 6/7. observable + adoption routing
        if not proposal.observable:
            proposal.verdict = VerifierVerdict.REJECTED
            proposal.reject_reason = "not observable / recordable"
            return proposal
        proposal.verdict = (
            VerifierVerdict.PENDING_PROPOSAL if needs_group_adoption else VerifierVerdict.UNLOCKED
        )
        return proposal
