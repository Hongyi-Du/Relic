"""ProposalManager + ProposalValidator + ToolRegistry (spec §13).

The ONLY path by which an LLM-drafted proposal becomes a world object: validate ->
route for approval -> approve (enough approvers) -> adopt -> create ToolSpec /
ProtocolSpec. Nothing is created on a raw LLM draft; adoption is never automatic.
"""
from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from environments.org_env.proposals.families import classify_family
from environments.org_env.proposals.objects import (
    PROPOSAL_TYPES,
    Proposal,
    ProtocolSpec,
    ToolSpec,
    ensure_list,
)

# preflight v3 §7: institutions must not form same-tick — they need review latency.
MIN_PROPOSAL_REVIEW_TICKS = 2
MIN_PROTOCOL_REVIEW_TICKS = 3
MIN_DISTINCT_PROTOCOL_APPROVERS = 2
MAX_ACTIVE_PROTOCOL_PROPOSALS = 1     # v4 §8: at most one protocol under review at a time
MAX_QUEUED_PROTOCOL_PROPOSALS = 3     # ... and a small draft+review queue
# v11 P2: at most this many ACTIVE tools per concern family — a same-family near-duplicate
# is folded into an existing tool (support) rather than minting another.
MAX_ACTIVE_TOOLS_PER_FAMILY = 2


def _tick_value(v, fallback: int) -> int:
    """Tick with a missing-value fallback. 0 is a legitimate tick, NOT a missing
    value — a falsy-zero fallback (``v or tick``) rewrote a tick-0 proposal's
    birth tick to its adoption tick, defeating the sites' own stated intent."""
    return int(v) if v is not None else int(fallback)


def recurrence_behind(p) -> int:
    """How much repetition a protocol proposal rests on, counted in either currency.

    Repetition is what separates a rule from one bad afternoon, and it reaches a
    proposal two ways: the same kind of episode happening again, or several
    members asking for the same thing. Only episodes were ever counted, and the
    two roads do not produce them alike -- the environment's catalogue fires on a
    cluster of two episodes by construction, while a theme grouped from three
    members' wishes carries whichever episodes those wishes happened to name,
    which can be one. So a theme three people asked for was ruled shallow and
    made to queue, while the catalogue's own entry never was.
    """
    cluster = set(getattr(p, "source_episode_ids", []) or [])
    if getattr(p, "source_episode_id", None):
        cluster.add(p.source_episode_id)
    cluster |= set(getattr(p, "source_event_ids", []) or [])   # synthesizer stores the cluster here
    wishes = {w for w in (getattr(p, "source_wish_ids", None) or []) if w}
    return max(len(cluster), len(wishes))


def protocol_depth_ok(p) -> bool:
    """v11 P2 abstraction gate: a PROTOCOL institutionalizes a class of repeated friction,
    so it must trace to a cluster of >=2 episodes, or to >=2 members asking for it. A
    proposal resting on a single occurrence is too shallow — it should stay a wish /
    local fix / tool. Proposals carrying no provenance at all are allowed
    (backward-compatible; depth can't be judged)."""
    if getattr(p, "amends_protocol_id", None):
        return True
    cluster = set(getattr(p, "source_episode_ids", []) or [])
    if getattr(p, "source_episode_id", None):
        cluster.add(p.source_episode_id)
    cluster |= set(getattr(p, "source_event_ids", []) or [])
    if not cluster and not (getattr(p, "source_wish_ids", None) or []):
        return True
    return recurrence_behind(p) >= 2


@dataclass
class ValidationResult:
    passed: bool
    reason: str = ""


class ProposalValidator:
    """Reality constraints before a draft enters review (spec §13.2)."""

    def validate(self, p: Proposal, world: Any) -> ValidationResult:
        if p.proposal_type not in PROPOSAL_TYPES:
            return ValidationResult(False, f"illegal proposal_type '{p.proposal_type}'")
        if not p.title or not p.summary:
            return ValidationResult(False, "missing title/summary")
        registry_target = getattr(
            p, "repair_target_registry_protocol_id", None
        )
        if registry_target is not None:
            if (
                p.proposal_type != "policy_repair_proposal"
                or not isinstance(registry_target, str)
                or not registry_target
                or len(registry_target) > 200
                or getattr(p, "repair_target_protocol_id", None) is not None
                or getattr(p, "amends_protocol_id", None) is not None
                or getattr(p, "repair_kind", None) != "deprecate"
                or getattr(p, "programbench_transition_repair", False) is not True
                or re.fullmatch(
                    r"[0-9a-f]{64}",
                    str(
                        getattr(
                            p,
                            "programbench_friction_evidence_digest",
                            "",
                        )
                        or ""
                    ),
                )
                is None
                or not isinstance(
                    getattr(p, "programbench_transition_phase", None), str
                )
                or not getattr(p, "programbench_transition_phase", "")
                or not isinstance(
                    getattr(
                        p, "programbench_phase_transition_count", None
                    ),
                    int,
                )
                or isinstance(
                    getattr(
                        p, "programbench_phase_transition_count", None
                    ),
                    bool,
                )
                or getattr(p, "programbench_phase_transition_count", -1) < 0
                or not isinstance(
                    getattr(
                        p,
                        "programbench_registry_observation_started_tick",
                        None,
                    ),
                    int,
                )
                or isinstance(
                    getattr(
                        p,
                        "programbench_registry_observation_started_tick",
                        None,
                    ),
                    bool,
                )
                or getattr(
                    p,
                    "programbench_registry_observation_started_tick",
                    -1,
                )
                < 0
                or not isinstance(
                    getattr(p, "programbench_registry_evidence_refs", None),
                    list,
                )
                or not getattr(p, "programbench_registry_evidence_refs", [])
                or len(
                    getattr(p, "programbench_registry_evidence_refs", [])
                )
                > 64
                or not all(
                    isinstance(ref, str) and 0 < len(ref) <= 200
                    for ref in getattr(
                        p, "programbench_registry_evidence_refs", []
                    )
                )
            ):
                return ValidationResult(
                    False, "invalid registry-only policy repair"
                )
        try:
            from environments.org_env.backend.actions import registered_action_types
            known = set(registered_action_types())
        except Exception:
            known = set()
        for a in p.required_actions:
            if known and (not isinstance(a, str) or a not in known):
                return ValidationResult(False, f"required action '{a}' does not exist")
        # not a duplicate of an existing active tool / adopted protocol
        mgr = getattr(world, "proposal_manager", None)
        if mgr is not None:
            if any(t.name == p.title and t.status == "active" for t in mgr.tools.values()):
                return ValidationResult(False, "duplicate of an active tool")
            if any(s.name == p.title and s.status == "adopted" for s in mgr.protocol_specs.values()):
                return ValidationResult(False, "duplicate of an adopted protocol")
        return ValidationResult(True, "")


# which roles must approve which proposal types (generic, role-keyed)
_APPROVERS = {
    "protocol_proposal": ("founder", "cofounder"),
    "policy_repair_proposal": ("founder", "cofounder"),
    "role_proposal": ("founder", "cofounder"),
    "tool_proposal": ("cofounder", "reliability"),
    "workflow_proposal": ("cofounder", "reliability"),
    "artifact_template_proposal": ("cofounder",),
    "task_proposal": ("founder", "cofounder"),
}


class ProposalManager:
    def __init__(self) -> None:
        self.proposals: Dict[str, Proposal] = {}
        self.tools: Dict[str, ToolSpec] = {}
        self.protocol_specs: Dict[str, ProtocolSpec] = {}
        self.validator = ProposalValidator()
        self._seq = 0
        self._tseq = 0
        self._pseq = 0

    # -- lifecycle --------------------------------------------------------- #
    def create_proposal(self, proposal: Proposal, world: Any) -> Proposal:
        # #8 provenance: never leave the creation tick / approver fields empty, so the
        # wish -> proposal -> approvers -> adoption event chain is always complete + inspectable.
        _tick = int(getattr(world, "world_tick", 0) or 0)
        if not getattr(proposal, "created_at_tick", 0):
            proposal.created_at_tick = _tick
        if not getattr(proposal, "updated_at_tick", 0):
            proposal.updated_at_tick = _tick
        if getattr(proposal, "approval_required_from", None) is None:
            proposal.approval_required_from = []
        if getattr(proposal, "approved_by", None) is None:
            proposal.approved_by = []
        # defensive: map a KNOWN natural-language proposal_type to the canonical enum
        # (preflight review #3). A truly-illegal/unmappable type is left as-is so the
        # validator still rejects it.
        if proposal.proposal_type not in PROPOSAL_TYPES:
            try:
                from environments.org_env.llm.proposal_generator import alias_proposal_type
                mapped = alias_proposal_type(proposal.proposal_type)
                if mapped:
                    proposal.proposal_type = mapped
            except Exception:
                pass
        vr = self.validator.validate(proposal, world)
        if not vr.passed:
            proposal.status = "rejected"
            proposal.rejection_reason = vr.reason
            self.proposals[proposal.proposal_id] = proposal
            self._event(world, "proposal_event", "rejected_invalid", proposal)
            return proposal
        # v4 §8: keep protocol formation sparse — at most one active + a small queue.
        # v14 P1: a proposal GROUNDED IN RECURRENCE is EXEMPT from this cap — shallow
        # single-occurrence proposals must not starve a deep, evidence-backed one (v13b:
        # the Claim-Source Interface Contract Protocol got rejected for "too many in flight"
        # while shallow workflow proposals filled capacity). Recurrence counts repeated
        # episodes OR repeated askers, so which road a proposal came down no longer
        # decides whether it has to queue.
        if proposal.proposal_type == "protocol_proposal":
            grounded = recurrence_behind(proposal) >= 2
            if not grounded:
                active = sum(1 for p in self.proposals.values()
                             if p.proposal_type == "protocol_proposal" and p.status == "under_review")
                queued = sum(1 for p in self.proposals.values()
                             if p.proposal_type == "protocol_proposal" and p.status in ("draft", "under_review"))
                if active >= MAX_ACTIVE_PROTOCOL_PROPOSALS or queued >= MAX_QUEUED_PROTOCOL_PROPOSALS:
                    proposal.status = "rejected"
                    proposal.rejection_reason = "too many protocol proposals in flight"
                    self.proposals[proposal.proposal_id] = proposal
                    self._event(world, "proposal_event", "rejected_invalid", proposal)
                    return proposal
        # v11 P2: tag the concern family up front (used for the active-tool cap + telemetry)
        if not proposal.family:
            proposal.family = classify_family(proposal)
        # v6 P0.2: a protocol proposal semantically covered by an already-adopted protocol
        # is routed as an AMENDMENT of that protocol (it still goes through review/approval;
        # on adoption it bumps that protocol's revision instead of minting a near-duplicate).
        if proposal.proposal_type in ("protocol_proposal", "policy_repair_proposal"):
            # a repair targets a SPECIFIC adopted protocol (relax/deprecate); else fall back to the
            # semantically-covering protocol (a normal amendment).
            if proposal.proposal_type == "policy_repair_proposal":
                if not getattr(
                    proposal, "repair_target_registry_protocol_id", None
                ):
                    self._name_the_rule_being_repaired(proposal, world)
            tgt = getattr(proposal, "repair_target_protocol_id", None)
            registry_target = getattr(
                proposal, "repair_target_registry_protocol_id", None
            )
            covering = None if registry_target else (
                (self.protocol_specs.get(tgt) if tgt in self.protocol_specs else None)
                or self._covering_protocol(proposal, world)
            )
            if covering is not None:
                proposal.amends_protocol_id = covering.protocol_id
            # v11 P2 abstraction gate: a single-episode protocol is too shallow — reject so it
            # stays a wish / local fix until the friction recurs across episodes.
            elif proposal.proposal_type == "protocol_proposal" and not protocol_depth_ok(proposal):
                proposal.status = "rejected"
                proposal.rejection_reason = ("protocol needs cross-episode friction "
                                             "(single episode — keep as wish / local fix / tool)")
                self.proposals[proposal.proposal_id] = proposal
                self._event(world, "proposal_event", "rejected_shallow", proposal)
                return proposal
        proposal.status = "draft"
        self.proposals[proposal.proposal_id] = proposal
        self._event(world, "proposal_event", "created", proposal)
        self._link_sources(world, proposal)
        return proposal

    def evaluate_proposal(self, proposal: Proposal, world: Any, evaluator=None) -> None:
        if evaluator is not None:
            scores = evaluator(proposal, world)
            proposal.feasibility_score = scores.get("feasibility_score")
            proposal.usefulness_score = scores.get("usefulness_score")
            proposal.risk_score = scores.get("risk_score")
            proposal.adoption_score = scores.get("adoption_score")
            proposal.suggested_revision = scores.get("suggested_revision", "")
        proposal.updated_at_tick = int(getattr(world, "world_tick", 0))

    def route_for_approval(self, proposal: Proposal, world: Any) -> None:
        roles = _APPROVERS.get(proposal.proposal_type, ("founder", "cofounder"))
        approvers = [aid for aid, a in getattr(world, "agents", {}).items()
                     if getattr(a, "role", "") in roles]
        proposal.approval_required_from = approvers
        proposal.status = "under_review"
        proposal.updated_at_tick = int(getattr(world, "world_tick", 0))   # review-entry tick
        self._event(world, "proposal_event", "under_review", proposal)

    def approve_proposal(self, proposal_id: str, agent_id: str, world: Any) -> Optional[Any]:
        p = self.proposals.get(proposal_id)
        if p is None or p.status not in ("draft", "under_review"):
            return None
        if agent_id not in p.approved_by:
            p.approved_by.append(agent_id)
        self._event(world, "proposal_event", "approved", p, extra={"approver": agent_id})
        # enough approvers AND past the review-latency gate -> adopt; else stay
        # under_review and let the per-tick sweep adopt once latency passes (§7).
        if self._approver_threshold_met(p) and self._can_adopt(p, world):
            p.status = "approved"
            return self.adopt_proposal(proposal_id, world)
        return None

    @staticmethod
    def _approver_threshold_met(p) -> bool:
        need = set(p.approval_required_from)
        return (need and need <= set(p.approved_by)) or (not need and len(set(p.approved_by)) >= 2)

    def _can_adopt(self, p, world: Any) -> bool:
        """Preflight v3 §7: institutions can't form same-tick. Enforce a minimum review
        latency (protocols longer) and a distinct-approver floor for protocols."""
        tick = int(getattr(world, "world_tick", 0))
        review_entry = int(getattr(p, "updated_at_tick", 0) or getattr(p, "created_at_tick", 0) or 0)
        is_protocol = p.proposal_type == "protocol_proposal"
        if is_protocol:
            min_ticks = MIN_PROTOCOL_REVIEW_TICKS
        elif p.proposal_type == "task_proposal":
            min_ticks = 0                       # low-risk task proposals may adopt promptly
        else:
            min_ticks = MIN_PROPOSAL_REVIEW_TICKS
        if tick - review_entry < min_ticks:
            return False
        if is_protocol and len(set(p.approved_by)) < MIN_DISTINCT_PROTOCOL_APPROVERS:
            return False
        return True

    def process_pending_adoptions(self, world: Any) -> List[Any]:
        """Adopt under_review proposals that have enough approvers and have now
        cleared the review-latency gate (called once per tick from world.step)."""
        adopted = []
        for p in list(self.proposals.values()):
            if p.status != "under_review":
                continue
            if self._approver_threshold_met(p) and self._can_adopt(p, world):
                p.status = "approved"
                obj = self.adopt_proposal(p.proposal_id, world)
                if obj is not None:
                    adopted.append(obj)
        return adopted

    def reject_proposal(self, proposal_id: str, agent_id: str, reason: str, world: Any) -> None:
        p = self.proposals.get(proposal_id)
        if p is None:
            return
        p.rejected_by.append(agent_id)
        p.status = "rejected"
        p.rejection_reason = reason
        self._event(world, "proposal_event", "rejected", p, extra={"by": agent_id})

    def adopt_proposal(self, proposal_id: str, world: Any) -> Optional[Any]:
        """Create the real world object — the ONLY place tools/protocols are minted."""
        p = self.proposals.get(proposal_id)
        if p is None or p.status not in ("approved",):
            return None
        from environments.org_env.experiments.ablations import (
            INSTITUTIONALIZATION,
            mechanism_disabled,
        )
        if (
            not getattr(world, "institutionalization_enabled", True)
            or mechanism_disabled(world, INSTITUTIONALIZATION)
        ):
            p.status = "rejected"
            p.rejection_reason = "mechanism_ablation:institutionalization"
            p.updated_at_tick = int(getattr(world, "world_tick", 0))
            self._event(
                world,
                "mechanism_ablation_event",
                "institutionalization_blocked",
                p,
            )
            return None
        tick = int(getattr(world, "world_tick", 0))
        obj = None
        if p.proposal_type in ("tool_proposal", "workflow_proposal", "artifact_template_proposal"):
            # v11 P2: fold a same-family near-duplicate (or an over-cap family) into an
            # existing active tool as SUPPORT, instead of minting yet another tool.
            covering = self._covering_tool(p, world)
            if covering is not None:
                return self._fold_into_tool(covering, p, world, tick)
            self._tseq += 1
            obj = ToolSpec(
                tool_id=f"tool_{self._tseq}", name=p.title, description=p.summary,
                created_from_proposal_id=p.proposal_id, creator_agent_id=p.proposer_agent_id,
                source_wish_id=p.source_wish_id, source_episode_id=p.source_episode_id,
                source_episode_ids=list(getattr(p, "source_episode_ids", []) or []),
                family=p.family or classify_family(p),
                tool_type="workflow_tool" if p.proposal_type == "workflow_proposal" else
                ("artifact_tool" if p.proposal_type == "artifact_template_proposal" else "composed_action_tool"),
                required_actions=ensure_list(p.required_actions),
                required_capabilities=ensure_list(p.required_capabilities),
                callable_by_roles=[], status="active", created_at_tick=tick,
                adopted_at_tick=tick, updated_at_tick=tick)
            self.tools[obj.tool_id] = obj
            p.object_created_id = obj.tool_id
            self._event(world, "tool_event", "created", p, extra={"tool_id": obj.tool_id, "family": obj.family})
        elif p.proposal_type in ("protocol_proposal", "policy_repair_proposal"):
            if (
                p.proposal_type == "policy_repair_proposal"
                and getattr(p, "repair_target_registry_protocol_id", None)
            ):
                return self._repair_registry_protocol(p, world, tick)
            # v6 P0.2: amend an existing adopted protocol if this proposal is covered by
            # one (re-checked at adopt time to also catch a sibling adopted since draft).
            existing = self.protocol_specs.get(getattr(p, "amends_protocol_id", None) or "")
            if existing is None or existing.status != "adopted":
                tgt = getattr(p, "repair_target_protocol_id", None)   # a repair names its target
                existing = (self.protocol_specs.get(tgt) if tgt in self.protocol_specs else None) \
                    or self._covering_protocol(p, world)
            if existing is not None:
                return self._amend_protocol(existing, p, world, tick)
            if p.proposal_type == "policy_repair_proposal":
                p.status = "implemented"          # a repair with no target protocol is a no-op
                return None
            self._pseq += 1
            obj = ProtocolSpec(
                protocol_id=f"protospec_{self._pseq}", name=p.title,
                created_from_proposal_id=p.proposal_id, source_episode_id=p.source_episode_id,
                source_episode_ids=list(getattr(p, "source_episode_ids", []) or []),
                source_wish_id=p.source_wish_id,
                source_wish_ids=list(getattr(p, "source_wish_ids", []) or []),
                source_reflection_id=p.source_reflection_id,
                trigger_condition=p.target_problem, required_steps=ensure_list(p.required_actions),
                required_fields=ensure_list(p.required_artifacts), enforcement_rule=p.proposed_solution,
                affected_actions=ensure_list(p.required_actions), benefits=ensure_list(p.expected_benefits),
                costs=ensure_list(p.expected_costs), risks=ensure_list(p.risks), status="adopted",
                proposed_by=p.proposer_agent_id, adopted_by=list(p.approved_by),
                created_at_tick=_tick_value(getattr(p, "created_at_tick", None), tick),  # proposal birth, not adoption
                adopted_at_tick=tick)
            self._fill_protocol_structure(obj, p, world)   # v11 P3: full institution structure
            self.protocol_specs[obj.protocol_id] = obj
            p.object_created_id = obj.protocol_id
            self._event(world, "protocol_spec_event", "adopted", p, extra={"protocol_id": obj.protocol_id})
            self._register_live_protocol(world, obj)   # make it enforceable via the registry
        if obj is not None:
            p.adopted_tick = tick                       # v11 telemetry: when it became real
            p.supporters = sorted(set(list(p.supporters) + list(p.approved_by)))
        p.status = "adopted" if obj is not None else "implemented"
        return obj

    def _reject_registry_repair(self, proposal, world, reason: str):
        proposal.status = "rejected"
        proposal.rejection_reason = reason
        proposal.updated_at_tick = int(getattr(world, "world_tick", 0) or 0)
        self._event(
            world,
            "proposal_event",
            "rejected_stale_registry_repair",
            proposal,
            extra={"reason": reason},
        )
        return None

    def _repair_registry_protocol(self, proposal, world, tick: int):
        """Adopt an evidence-bound repair without inventing a ProtocolSpec.

        Registry-only rules have no structured fields that can be safely
        relaxed.  The only supported operation is therefore deprecation: it
        changes the live rule's lifecycle/enforcement status, while preserving
        its complete event history.  Every public precondition is re-attested
        after the ordinary review latency so a stale proposal fails closed.
        """

        protocol_id = str(
            getattr(proposal, "repair_target_registry_protocol_id", "") or ""
        )
        if (
            not protocol_id
            or len(protocol_id) > 200
            or getattr(proposal, "repair_target_protocol_id", None) is not None
            or getattr(proposal, "amends_protocol_id", None) is not None
            or getattr(proposal, "repair_kind", None) != "deprecate"
            or getattr(proposal, "programbench_transition_repair", False)
            is not True
        ):
            return self._reject_registry_repair(
                proposal, world, "registry_repair_metadata_invalid"
            )

        try:
            from environments.org_env.programbench import (
                refresh_programbench_protocol_adaptation,
            )

            state = refresh_programbench_protocol_adaptation(world)
        except Exception:
            return self._reject_registry_repair(
                proposal, world, "registry_repair_profile_invalid"
            )

        claimed_digest = getattr(
            proposal, "programbench_friction_evidence_digest", None
        )
        bound_phase = getattr(
            proposal, "programbench_transition_phase", None
        )
        bound_transition = getattr(
            proposal, "programbench_phase_transition_count", None
        )
        bound_observation_start = getattr(
            proposal,
            "programbench_registry_observation_started_tick",
            None,
        )
        bound_refs = getattr(
            proposal, "programbench_registry_evidence_refs", None
        )
        cohort_rows = [
            row
            for row in (state.get("protocol_transition_cohort") or [])
            if isinstance(row, dict)
            and str(row.get("canonical_protocol_id") or "") == protocol_id
            and row.get("protocol_spec_id") is None
            and str(row.get("registry_protocol_id") or "") == protocol_id
        ]
        evidence = state.get("protocol_friction_evidence")
        target_rows = [
            row
            for row in (
                evidence.get("target_metrics", [])
                if isinstance(evidence, dict)
                else []
            )
            if isinstance(row, dict)
            and str(row.get("protocol_id") or "") == protocol_id
        ]
        target_row = target_rows[0] if len(target_rows) == 1 else None
        if (
            state.get("protocol_adaptation_active") is not True
            or protocol_id
            not in set(state.get("protocol_repair_eligible_target_ids") or [])
            or len(cohort_rows) != 1
            or not isinstance(claimed_digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", claimed_digest) is None
            or not isinstance(bound_phase, str)
            or bound_phase != str(state.get("phase") or "")
            or not isinstance(bound_transition, int)
            or isinstance(bound_transition, bool)
            or bound_transition
            != int(state.get("phase_transition_count") or 0)
            or not isinstance(bound_observation_start, int)
            or isinstance(bound_observation_start, bool)
            or bound_observation_start
            != int(
                cohort_rows[0].get("observation_started_tick") or 0
            )
            or target_row is None
            or target_row.get("qualified") is not True
            or target_row.get("live") is not True
            or int(target_row.get("observation_started_tick") or 0)
            != bound_observation_start
            or not isinstance(bound_refs, list)
            or not bound_refs
            or len(bound_refs) > 64
            or not all(
                isinstance(ref, str) and 0 < len(ref) <= 200
                for ref in bound_refs
            )
        ):
            return self._reject_registry_repair(
                proposal, world, "registry_repair_evidence_stale"
            )

        registry = getattr(world, "protocol_registry", None)
        protocol = getattr(registry, "protocols", {}).get(protocol_id)
        mirrored_spec_ids = {
            f"proto_spec_{str(getattr(spec, 'protocol_id', spec_id)).split('_')[-1]}"
            for spec_id, spec in self.protocol_specs.items()
        }
        if (
            protocol is None
            or str(getattr(protocol, "adoption_status", "")) != "adopted"
            or str(getattr(protocol, "status", "active")) != "active"
            or protocol_id in self.protocol_specs
            or protocol_id in mirrored_spec_ids
        ):
            return self._reject_registry_repair(
                proposal, world, "registry_repair_target_not_live_registry_only"
            )

        actor = (
            (getattr(proposal, "approved_by", None) or [None])[0]
            or getattr(proposal, "proposer_agent_id", "")
            or "organizational_gate"
        )
        previous_revisions = sum(
            1
            for event in (getattr(registry, "events", None) or [])
            if str(getattr(event, "protocol_id", "") or "") == protocol_id
            and str(getattr(event, "event_type", "") or "")
            in ("amendment", "obsolete")
        )
        registry_event = registry.obsolete(
            actor,
            protocol_id,
            tick=tick,
            source_proposal_id=proposal.proposal_id,
        )
        proposal.object_created_id = protocol_id
        proposal.adopted_tick = tick
        proposal.supporters = sorted(
            set(
                list(getattr(proposal, "supporters", None) or [])
                + list(getattr(proposal, "approved_by", None) or [])
            )
        )
        proposal.status = "adopted"
        proposal.updated_at_tick = tick
        self._event(
            world,
            "protocol_registry_event",
            "deprecated",
            proposal,
            extra={"protocol_id": protocol_id},
        )
        getattr(world, "events", []).append(
            {
                "type": "protocol_revision_event",
                "protocol_id": protocol_id,
                "revision": previous_revisions + 1,
                "kind": "deprecate",
                "from_proposal_id": proposal.proposal_id,
                "registry_event_id": getattr(
                    registry_event, "event_id", None
                ),
                "agent_id": proposal.proposer_agent_id,
                "tick": tick,
            }
        )
        return protocol

    # -- v11 P2 tool family cap / fold ------------------------------------- #
    _TOOL_MERGE_CRITERION = (
        "two tools automate the SAME workflow step toward the SAME goal (e.g. both check / "
        "assemble evidence for claims, or both run the release smoke gate), making a second "
        "tool redundant — treat them as the same tool even if wording or naming differ, and "
        "only answer false when they automate genuinely different work")

    @staticmethod
    def _tool_proposal_texts(p) -> List[str]:
        return [p.title or "", p.summary or "", p.target_problem or "", p.proposed_solution or "",
                " ".join(ensure_list(p.required_actions))]

    @staticmethod
    def _tool_texts(t) -> List[str]:
        return [t.name or "", t.description or "", " ".join(t.required_actions or [])]

    def _covering_tool(self, p, world):
        """An active tool that should absorb this proposal instead of minting a new one:
        either a same-family semantic duplicate, or (family at cap) the family's anchor tool."""
        fam = p.family or classify_family(p)
        same_fam = [t for t in self.tools.values()
                    if t.status == "active" and (getattr(t, "family", "") or "") == fam and fam != "other"]
        if not same_fam:
            return None
        from environments.org_env.llm.semantic_dedup import equivalent
        client = getattr(world, "llm_client", None)
        a = self._tool_proposal_texts(p)
        for t in same_fam:
            if equivalent(world, a, self._tool_texts(t), kind="workflow tool", client=client,
                          lo=0.0, hi=0.6, criterion=self._TOOL_MERGE_CRITERION):
                return t
        # family active-tool cap: fold into the most-supported anchor tool of the family
        if len(same_fam) >= MAX_ACTIVE_TOOLS_PER_FAMILY:
            return max(same_fam, key=lambda t: (getattr(t, "support_count", 0), getattr(t, "adopted_at_tick", 0)))
        return None

    def _fold_into_tool(self, tool, p, world, tick: int):
        """Record this proposal as SUPPORT for an existing tool (no new tool minted)."""
        tool.support_count += 1
        if p.proposer_agent_id and p.proposer_agent_id not in tool.supporters:
            tool.supporters.append(p.proposer_agent_id)
        if p.proposal_id not in tool.folded_proposal_ids:
            tool.folded_proposal_ids.append(p.proposal_id)
        # carry any new lineage so the anchor tool stays a well-grounded company-skill candidate
        for eid in (list(getattr(p, "source_episode_ids", []) or [])
                    + ([p.source_episode_id] if p.source_episode_id else [])):
            if eid and eid not in tool.source_episode_ids:
                tool.source_episode_ids.append(eid)
        tool.updated_at_tick = tick
        p.object_created_id = tool.tool_id
        p.status = "adopted"
        p.adopted_tick = tick
        self._event(world, "tool_event", "folded", p,
                    extra={"tool_id": tool.tool_id, "family": tool.family,
                           "support_count": tool.support_count})
        return tool

    # -- v11 P3 protocol structure fill ------------------------------------ #
    def _fill_protocol_structure(self, spec, p, world) -> None:
        """Populate the full institution structure from the proposal + its episode cluster, so
        an adopted protocol is a real norm (problem evidence / scope / responsible roles /
        success metric / enforcement / sunset), not just a name."""
        spec.family = p.family or classify_family(p)
        # problem_evidence: the episode cluster + cited events/objects that motivated it
        evidence = list(dict.fromkeys(
            (list(getattr(p, "source_episode_ids", []) or []))
            + ([p.source_episode_id] if p.source_episode_id else [])
            + list(getattr(p, "source_event_ids", []) or [])
            + list(getattr(p, "affected_objects", []) or [])))
        spec.problem_evidence = evidence
        # scope: which workflow / artifacts / roles it governs
        roles_aff = sorted({getattr(world.agents.get(a), "role", "") for a in spec.adopted_by
                            if a in getattr(world, "agents", {})} - {""})
        scope_bits = []
        if spec.affected_actions:
            scope_bits.append("actions: " + ", ".join(spec.affected_actions[:4]))
        if p.required_artifacts:
            scope_bits.append("artifacts: " + ", ".join(ensure_list(p.required_artifacts)[:4]))
        spec.scope = "; ".join(scope_bits) or f"{spec.family} workflow"
        # responsible roles: executor (proposer), reviewers/approvers (adopters)
        proposer_role = getattr(world.agents.get(spec.proposed_by), "role", "") if spec.proposed_by else ""
        spec.responsible_roles = {
            "executor": [proposer_role] if proposer_role else [],
            "reviewer": roles_aff,
            "approver": roles_aff or ["founder", "cofounder"],
        }
        spec.enforcement_action = spec.enforcement_rule or (
            "block the release candidate / request the missing artifact until the protocol is satisfied")
        spec.violation_condition = spec.violation_condition or (
            f"a {spec.family} step is skipped (required artifact / step missing)")
        spec.success_metric = self._DEFAULT_SUCCESS_METRIC.get(
            spec.family, "the targeted failure/friction rate drops on subsequent cycles")
        spec.sunset_rule = ("review after 5 uses or 240 ticks; amend if friction persists, "
                            "retire if unused for 120 ticks or superseded by a broader protocol")

    _DEFAULT_SUCCESS_METRIC = {
        "evidence_governance": "unsupported-claim rate in release candidates drops to ~0",
        "release_engineering": "release candidates pass the smoke/CI gate on first re-check more often",
        "debugging": "release smoke/CI blockers are closed via patch (not abandoned), with fewer CI iterations",
        "review_merge": "PRs reach mainline with review sign-off and no post-merge regressions",
        "customer": "every inbound customer issue gets an owner + triage record within a day",
        "budget_governance": "expensive runs are approved before spend; token cost per merged patch trends down",
        "ownership": "every critical task/artifact has an explicit owner",
        "experiment": "experiment results are reproducible + logged to the shared tracker",
    }

    # -- v6 P0.2 protocol semantic merge / revision ------------------------ #
    @staticmethod
    def _protocol_proposal_texts(p) -> List[str]:
        return [p.title or "", p.summary or "", p.target_problem or "", p.proposed_solution or "",
                " ".join(ensure_list(p.required_artifacts)),
                " ".join(ensure_list(p.required_actions)),
                " ".join(ensure_list(p.expected_benefits))]

    @staticmethod
    def _protocol_spec_texts(s) -> List[str]:
        return [s.name or "", s.trigger_condition or "", s.enforcement_rule or "",
                " ".join(s.required_fields or []), " ".join(s.required_steps or []),
                " ".join(s.benefits or [])]

    # v6 P0.2: governance protocols that gate the SAME activity toward the SAME goal are
    # redundant even when scope/wording/naming differ (e.g. "minimal evidence gate for
    # reports" vs "Claim Evidence Gate Protocol"). A loose, goal-level merge criterion so
    # the adopted set converges to one protocol per concern + revisions, not a stack.
    _PROTOCOL_MERGE_CRITERION = (
        "two governance protocols regulate the SAME activity/trigger toward the SAME goal "
        "(e.g. both require evidence / sources / traceability for claims or results before "
        "they ship), making it redundant to adopt both — treat them as the same protocol "
        "even if their scope, wording, or naming differ, and only answer false when they "
        "govern genuinely different concerns")

    def _covering_protocol(self, p, world):
        """An adopted ProtocolSpec that already covers this proposal's trigger/goal, so a
        new spec would just stack a near-duplicate (semantic_dedup: deterministic overlap
        + a loose, goal-level LLM judgement)."""
        adopted = [s for s in self.protocol_specs.values() if s.status == "adopted"]
        if not adopted:
            return None
        from environments.org_env.llm.semantic_dedup import equivalent
        a = self._protocol_proposal_texts(p)
        client = getattr(world, "llm_client", None)
        for s in adopted:
            # always defer to the LLM (lo=0.0) below a fairly low overlap bar (hi=0.6),
            # with the loose goal-level criterion. Mock (no client) stays "distinct".
            if equivalent(world, a, self._protocol_spec_texts(s),
                          kind="governance protocol", client=client, lo=0.0, hi=0.6,
                          criterion=self._PROTOCOL_MERGE_CRITERION):
                return s
        return None

    def _name_the_rule_being_repaired(self, proposal, world) -> None:
        """Decide which adopted rule a repair is about, and record it on the proposal.

        A repair that came from a member's reflection names its rule in prose --
        "the interface-handoff rule is the primary blocker" -- and carried no
        id, so the target was left to `_covering_protocol`, which matches on
        wording and answered with whichever adopted rule read most like the
        replacement text. On one run every repair arrived with no target and
        the relaxations landed on a rule the proposer had not been complaining
        about: refusals went up rather than down.

        So it is resolved from the same evidence a member reads. The words the
        proposal uses decide it; where they decide nothing, the rule that has
        been turning away the most work does, because that is the one the
        organization is asking about.
        """
        if getattr(proposal, "repair_target_protocol_id", None) in self.protocol_specs:
            return
        adopted = [s for s in self.protocol_specs.values()
                   if getattr(s, "status", "") == "adopted"]
        if not adopted:
            return

        said = " ".join(str(getattr(proposal, f, "") or "") for f in
                        ("title", "summary", "target_problem", "proposed_solution")).lower()
        words = {w for w in re.findall(r"[a-z]{4,}", said)}

        def overlap(spec) -> int:
            text = f"{getattr(spec, 'name', '')} {getattr(spec, 'enforcement_rule', '')} " \
                   f"{getattr(spec, 'trigger_condition', '')}".lower()
            return len(words & {w for w in re.findall(r"[a-z]{4,}", text)})

        best = max(adopted, key=overlap)
        if overlap(best) >= 3:
            proposal.repair_target_protocol_id = best.protocol_id
            return

        try:
            from environments.org_env.backend.protocol.harm import blocked_without_delivery
        except Exception:
            return
        worst = max(adopted, key=lambda s: blocked_without_delivery(world, s)[0])
        if blocked_without_delivery(world, worst)[0] > 0:
            proposal.repair_target_protocol_id = worst.protocol_id

    def _mirror_registry_rule(self, world, spec, rule: str) -> None:
        """Put the eased wording where enforcement reads it.

        Review judges a change against `rule_summary` in the registry. It was
        written once, when the rule was proposed, and no amendment has ever
        touched it -- so a relaxation that dropped a required field left the
        sentence demanding that field still standing, and the reviewer went on
        refusing exactly what it had refused before.
        """
        reg = getattr(world, "protocol_registry", None)
        if reg is None:
            return
        pid = f"proto_spec_{str(spec.protocol_id).split('_')[-1]}"
        protocol = getattr(reg, "protocols", {}).get(pid)
        if protocol is not None:
            try:
                protocol.rule_summary = rule
            except Exception:
                pass

    def _mirror_registry_revision(
        self,
        world,
        spec,
        proposal,
        *,
        tick: int,
        revision_kind: str,
    ):
        reg = getattr(world, "protocol_registry", None)
        if reg is None:
            return None
        pid = f"proto_spec_{str(spec.protocol_id).split('_')[-1]}"
        if pid not in getattr(reg, "protocols", {}):
            return None
        actor = (
            (getattr(proposal, "approved_by", None) or [None])[0]
            or getattr(proposal, "proposer_agent_id", "")
            or "organizational_gate"
        )
        if revision_kind == "deprecate":
            return reg.obsolete(
                actor,
                pid,
                tick=tick,
                source_proposal_id=proposal.proposal_id,
            )
        else:
            return reg.amend(
                actor,
                pid,
                tick=tick,
                revision_kind=revision_kind,
                source_proposal_id=proposal.proposal_id,
            )

    @staticmethod
    def _protocol_revision_snapshot(world, spec, proposal) -> dict:
        """Capture the bounded state one mirrored revision may mutate."""

        registry = getattr(world, "protocol_registry", None)
        mirror_id = f"proto_spec_{str(spec.protocol_id).split('_')[-1]}"
        mirror = (
            getattr(registry, "protocols", {}).get(mirror_id)
            if registry is not None
            else None
        )
        world_events = getattr(world, "events", None)
        agent_log = getattr(world, "agent_log", None)
        return {
            "spec": copy.deepcopy(spec.__dict__),
            "proposal": copy.deepcopy(proposal.__dict__),
            "registry": registry,
            "mirror": mirror,
            "mirror_state": (
                copy.deepcopy(mirror.__dict__)
                if mirror is not None
                else None
            ),
            "registry_events": (
                list(getattr(registry, "events", []))
                if registry is not None
                else None
            ),
            "registry_seq": (
                getattr(registry, "_seq", None)
                if registry is not None
                else None
            ),
            "world_events": world_events,
            "world_event_rows": (
                list(world_events) if isinstance(world_events, list) else None
            ),
            "agent_log": agent_log,
            "agent_log_rows": (
                list(agent_log) if isinstance(agent_log, list) else None
            ),
        }

    @staticmethod
    def _restore_protocol_revision_snapshot(snapshot, spec, proposal) -> None:
        """Roll back a failed mirrored revision without replacing live objects."""

        spec.__dict__.clear()
        spec.__dict__.update(snapshot["spec"])
        proposal.__dict__.clear()
        proposal.__dict__.update(snapshot["proposal"])
        mirror = snapshot["mirror"]
        if mirror is not None:
            mirror.__dict__.clear()
            mirror.__dict__.update(snapshot["mirror_state"])
        registry = snapshot["registry"]
        if registry is not None:
            registry.events[:] = snapshot["registry_events"]
            if snapshot["registry_seq"] is not None:
                registry._seq = snapshot["registry_seq"]
        if snapshot["world_event_rows"] is not None:
            snapshot["world_events"][:] = snapshot["world_event_rows"]
        if snapshot["agent_log_rows"] is not None:
            snapshot["agent_log"][:] = snapshot["agent_log_rows"]

    def _amend_protocol(self, spec, p, world, tick: int):
        snapshot = self._protocol_revision_snapshot(world, spec, p)
        try:
            return self._amend_protocol_transaction(spec, p, world, tick)
        except Exception:
            self._restore_protocol_revision_snapshot(snapshot, spec, p)
            raise

    def _amend_protocol_transaction(self, spec, p, world, tick: int):
        """Record this proposal as a revision of an adopted protocol instead of minting a new spec.
        A normal amendment EXTENDS the protocol; a policy-repair proposal RELAXES (drops the named
        fields/steps, or undoes the last strengthening) or DEPRECATES it — so the org can undo a
        self-binding / harmful rule, not only ever tighten it (self-correction §)."""
        repair = getattr(p, "repair_kind", None) if p.proposal_type == "policy_repair_proposal" else None
        # The registry mirror is the executable half of this rule. Transition
        # it first, before touching the ProtocolSpec, so lifecycle rejection or
        # a registry failure cannot leave the descriptive and executable views
        # half-amended. A spec-only native rule still returns ``None`` here and
        # follows its historical path unchanged.
        registry_revision_event = self._mirror_registry_revision(
            world,
            spec,
            p,
            tick=tick,
            revision_kind=repair or "extend",
        )
        spec.revision += 1
        spec.last_revised_tick = tick           # v8d P1a: keep adopted_at_tick = FIRST adoption
        added_fields, added_steps, removed_fields, removed_steps = [], [], [], []
        if repair == "deprecate":
            spec.status = "deprecated"
        elif repair == "relax":
            drop_f = set(ensure_list(p.required_artifacts))
            drop_s = set(ensure_list(p.required_actions))
            if not drop_f and not drop_s and spec.revisions:      # nothing named -> undo last tightening
                last = spec.revisions[-1]
                drop_f = set(last.get("added_fields", []) or [])
                drop_s = set(last.get("added_steps", []) or [])
            removed_fields = [f for f in spec.required_fields if f in drop_f]
            removed_steps = [s for s in spec.required_steps if s in drop_s]
            spec.required_fields = [f for f in spec.required_fields if f not in drop_f]
            spec.required_steps = [s for s in spec.required_steps if s not in drop_s]
            # And say so in the rule itself. Review judges a change against the
            # rule's text, not against these lists, so dropping a required field
            # while leaving the sentence that demands it relaxed nothing: the
            # reviewer went on asking for exactly what it had asked for before.
            # A repair that names how the rule should read replaces it.
            eased = str(getattr(p, "proposed_solution", "") or "").strip()
            if eased and len(eased) > 20:
                spec.enforcement_rule = eased[:600]
                self._mirror_registry_rule(world, spec, eased[:600])
        else:
            added_fields = [f for f in ensure_list(p.required_artifacts) if f not in spec.required_fields]
            added_steps = [s for s in ensure_list(p.required_actions) if s not in spec.required_steps]
            spec.required_fields.extend(added_fields)
            spec.required_steps.extend(added_steps)
        for a in list(p.approved_by):
            if a not in spec.adopted_by:
                spec.adopted_by.append(a)
        spec.revisions.append({
            "revision": spec.revision, "from_proposal_id": p.proposal_id, "tick": tick,
            "title": p.title, "summary": p.summary, "kind": repair or "extend",
            "added_fields": added_fields, "added_steps": added_steps,
            "removed_fields": removed_fields, "removed_steps": removed_steps,
            "rationale": p.proposed_solution or p.target_problem})
        if p.proposal_id not in spec.superseded_proposal_ids:
            spec.superseded_proposal_ids.append(p.proposal_id)
        p.object_created_id = spec.protocol_id
        p.status = "adopted"
        _sub = ("deprecated" if repair == "deprecate" else "relaxed" if repair == "relax" else "amended")
        self._event(world, "protocol_spec_event", _sub, p,
                    extra={"protocol_id": spec.protocol_id, "revision": spec.revision})
        # spec #4: surface a dedicated revision event in the lifecycle stream
        revision_event = {
            "type": "protocol_revision_event", "protocol_id": spec.protocol_id,
            "revision": spec.revision, "kind": repair or "extend", "from_proposal_id": p.proposal_id,
            "agent_id": p.proposer_agent_id, "tick": int(getattr(world, "world_tick", 0))}
        if repair == "deprecate" and registry_revision_event is not None:
            revision_event["registry_event_id"] = getattr(
                registry_revision_event, "event_id", None
            )
        getattr(world, "events", []).append(revision_event)
        return spec

    def _register_live_protocol(self, world, spec) -> None:
        """Mirror an adopted ProtocolSpec into the live protocol_registry so the
        existing enforcement machinery (use / violate / enforce / detectors) and the
        action-decision context treat it as an adopted, enforceable norm."""
        reg = getattr(world, "protocol_registry", None)
        if reg is None:
            return
        # Cut on a word boundary. A flat [:40] split "…readiness_protocol" into
        # "…readiness_protoco", and the enforcement keywords are the words of this
        # slug: the fragment matched nothing, and it slipped past the filter that
        # drops "protocol" precisely because it was no longer that word.
        _slug = "_".join(w for w in "".join(
            c if c.isalnum() else "_" for c in spec.name.lower()).split("_") if w)
        ptype = _slug[:40].rsplit("_", 1)[0] if len(_slug) > 40 else _slug
        ptype = ptype or "spec_protocol"
        pid = f"proto_spec_{spec.protocol_id.split('_')[-1]}"
        if pid in getattr(reg, "protocols", {}):
            return
        tick = int(getattr(world, "world_tick", 0))
        try:
            # v8 #2: mirror with the spec's REAL proposal + adoption ticks and approver so
            # the registry telemetry shows the actual review latency, not a same-tick adopt.
            created = _tick_value(getattr(spec, "created_at_tick", None), tick)
            adopted = _tick_value(getattr(spec, "adopted_at_tick", None), tick)
            approver = (spec.adopted_by[0] if getattr(spec, "adopted_by", None) else spec.proposed_by)
            reg.propose(proposer_id=spec.proposed_by, protocol_type=ptype,
                        rule_summary=spec.enforcement_rule or spec.name, scope="org",
                        # When the rule bites, which the spec already says. Left
                        # empty, every adopted rule reached the registry with no
                        # record of the situation it governs.
                        target_process=str(getattr(spec, "trigger_condition", ""))[:160],
                        tick=created, protocol_id=pid)
            rp = reg.protocols.get(pid)
            if rp is not None:
                for a in (getattr(spec, "adopted_by", None) or []):
                    if a not in rp.supporters:
                        reg.support(a, pid, tick=adopted)
            if rp is not None and rp.adoption_status != "adopted":
                reg.adopt(pid, adopted, approver_id=approver, force=True)
            spec.affected_agents = list(getattr(spec, "affected_agents", []))
        except Exception:
            pass

    # -- helpers ----------------------------------------------------------- #
    def next_id(self, prefix: str = "proposal") -> str:
        self._seq += 1
        return f"{prefix}_{self._seq}"

    def _link_sources(self, world, p: Proposal) -> None:
        # Episode linkage is optional; wish closure is not. Guarding both behind
        # `source_episode_id` meant a proposal raised from a wish with no episode
        # never marked its wish converted. The auto harm-flagger is exactly that
        # path -- it raises a policy_repair wish from a system reflection with no
        # episode -- so its wish stayed `open` after the repair adopted, and the
        # flagger, which skips any protocol that already has an open repair wish,
        # was muzzled for the rest of the run: on one B3 arm protospec_1 degraded
        # to 468 refusals against 39 uses, flagged once at t25 and never again.
        rm = getattr(world, "reflection_manager", None)
        w = rm.wishes.get(p.source_wish_id) if (rm and p.source_wish_id) else None
        if w is not None:
            if p.proposal_id not in w.generated_proposal_ids:
                w.generated_proposal_ids.append(p.proposal_id)
            w.status = "converted_to_proposal"
        if not p.source_episode_id:
            return
        ep = getattr(world, "episode_manager", None)
        ep = ep.episodes.get(p.source_episode_id) if ep else None
        if ep is not None and p.proposal_id not in ep.linked_proposal_ids:
            ep.linked_proposal_ids.append(p.proposal_id)

    def _event(self, world, etype, subtype, p: Proposal, extra=None) -> None:
        ev = {"type": etype, "subtype": subtype, "agent_id": p.proposer_agent_id,
              "tick": int(getattr(world, "world_tick", 0)), "object_id": p.proposal_id,
              "proposal_type": p.proposal_type}
        if extra:
            ev.update(extra)
        getattr(world, "events", []).append(ev)
        log = getattr(world, "agent_log", None)
        if log is not None and p.proposer_agent_id:
            from environments.org_env.reflection.objects import AgentLogEntry
            log.append(AgentLogEntry(
                log_id=f"log_{len(log)}", agent_id=p.proposer_agent_id,
                tick=int(getattr(world, "world_tick", 0)), entry_type="proposal_created",
                summary=f"{subtype} proposal: {p.title}",
                related_episode_ids=[p.source_episode_id] if p.source_episode_id else [],
                raw_payload={"proposal_id": p.proposal_id, "status": p.status}))

    # -- snapshot ---------------------------------------------------------- #
    def proposals_snapshot(self) -> Dict[str, Any]:
        items = [p.to_dict() for p in self.proposals.values()]
        items.sort(key=lambda x: x.get("created_at_tick", 0))
        by_status: Dict[str, int] = {}
        by_type: Dict[str, int] = {}
        for p in self.proposals.values():
            by_status[p.status] = by_status.get(p.status, 0) + 1
            by_type[p.proposal_type] = by_type.get(p.proposal_type, 0) + 1
        return {"items": items, "by_status": by_status, "by_type": by_type, "total": len(self.proposals)}

    def tools_snapshot(self) -> Dict[str, Any]:
        items = [t.to_dict() for t in self.tools.values()]
        return {"items": items, "active_count": sum(1 for t in self.tools.values() if t.status == "active"),
                "total": len(self.tools)}

    def protocol_specs_snapshot(self) -> Dict[str, Any]:
        items = [s.to_dict() for s in self.protocol_specs.values()]
        return {"items": items, "active_count": sum(1 for s in self.protocol_specs.values()
                                                    if s.status == "adopted"), "total": len(self.protocol_specs)}


__all__ = ["ProposalManager", "ProposalValidator", "ValidationResult"]
