"""OrgPolicy — B3 profile selection plus the explicit no-LLM test fallback.

utility(candidate) = Σ_feature  weight(agent, feature) · feature_value
weight(agent, f)   = BASE_WEIGHT[f] + Σ_trait profile_or_skill[trait]·COEFF[trait,f]
                     (with global budget-pressure + self-state modulation)

In B3 the persona therefore *actually* changes the chosen action (O1 §2 #10): Calvin
up-weights review/reproducibility/protocol-use; Sean up-weights progress/demo and
discounts review; Los Xi up-weights external/customer; Will up-weights
review/claim-evidence; etc. B0/B1/B2 do not call ``select`` when an LLM client is
available. Their client-free test path sets ``use_profile_conditioning=False``
and uses deterministic argmax. Scoring modes:
  * mock  — deterministic seeded softmax over utilities (no LLM; for CI/smoke)
  * pcbsp — same scorer; hook left for an LLM candidate/rationale layer (O2)
"""
from __future__ import annotations

import math
import random
from typing import Any, List, Tuple

from environments.org_env.growth.objects import effective_skill

# Base per-feature weights: gains positive, costs/risks negative.
BASE_WEIGHTS = {
    # task/progress
    "progress_gain": 0.55, "deadline_urgency": 0.30, "task_priority": 0.30,
    "blocker_resolution": 0.35, "dependency_unlock": 0.20, "demo_relevance": 0.30,
    "customer_relevance": 0.30,
    # skill
    "skill_match": 0.40, "role_affinity": 0.30, "learning_gain": 0.10,
    "failure_risk_from_low_skill": -0.30,
    # social
    "coordination_gain": 0.30, "clarity_gain": 0.20, "trust_gain": 0.25,
    "trust_risk": -0.25, "conflict_risk": -0.25, "visibility_gain": 0.20,
    "reputation_gain": 0.20, "reputation_risk": -0.20,
    # cost
    "attention_cost": -0.30, "fatigue_delta": -0.20, "stress_delta": -0.20,
    "context_switch_cost": -0.20, "overtime_penalty": -0.30, "weekend_penalty": -0.30,
    "burnout_risk_delta": -0.30,
    # repo/experiment
    "repo_health_gain": 0.30, "technical_debt_risk": -0.30, "review_quality_gain": 0.25,
    "reproducibility_gain": 0.25, "untracked_result_risk": -0.20, "claim_evidence_gain": 0.25,
    # budget/payroll
    "api_cost": -0.15, "compute_cost": -0.12, "budget_cost": -0.15,
    "runway_risk_delta": -0.25, "payroll_trust_gain": 0.35, "retention_risk_delta": -0.25,
    # protocol
    "protocol_creation_potential": 0.20, "protocol_use_potential": 0.20,
    "protocol_violation_risk": -0.40, "norm_enforcement_gain": 0.20,
    "institutional_memory_gain": 0.20,
    # governance (proposal approval) — both sides modest so persona breaks the tie
    "proposal_endorsement": 0.45, "proposal_skepticism": 0.45,
    # external
    "external_signal_value": 0.20, "customer_pressure": 0.20, "expert_advice_value": 0.20,
    "competitor_pressure": 0.15, "recruiting_pressure": 0.10, "public_reputation_effect": 0.20,
    # O1.6 work-state + routine
    "routine_prior": 1.0, "recovery_value": 0.9, "capacity_strain": -0.9,
    "org_grievance_pull": 0.9,
}

# (trait_or_skill, feature, coeff) — persona modulation of feature weights (§9.2).
PROFILE_COEFFS: List[Tuple[str, str, float]] = [
    # Calvin — reliability / reproducibility / process
    ("process_commitment", "review_quality_gain", 0.6),
    ("process_commitment", "reproducibility_gain", 0.7),
    ("process_commitment", "protocol_use_potential", 0.5),
    ("process_commitment", "untracked_result_risk", -0.7),
    ("reproducibility_tracking", "reproducibility_gain", 0.5),
    ("experiment_logging", "institutional_memory_gain", 0.4),
    ("conformity", "protocol_violation_risk", -0.4),
    ("conformity", "protocol_use_potential", 0.4),
    # Sean — speed / demo / process-resistant
    ("speed_bias", "progress_gain", 0.6),
    ("speed_bias", "demo_relevance", 0.6),
    ("speed_bias", "review_quality_gain", -0.4),
    ("speed_bias", "reproducibility_gain", -0.35),
    ("process_resistance", "protocol_use_potential", -0.4),
    ("process_resistance", "protocol_violation_risk", 0.5),   # tolerate violations
    ("process_resistance", "review_quality_gain", -0.25),
    ("demo_building", "demo_relevance", 0.4),
    # Will — editorial / claim clarity
    ("quality_bar", "review_quality_gain", 0.5),
    ("quality_bar", "claim_evidence_gain", 0.5),
    ("quality_bar", "technical_debt_risk", -0.4),
    ("claim_wording", "claim_evidence_gain", 0.5),
    ("clarity_review", "review_quality_gain", 0.4),
    ("communication_clarity", "clarity_gain", 0.4),
    # Los Xi — community / customer
    ("external_community_sensing", "external_signal_value", 0.6),
    ("customer_sense", "customer_pressure", 0.5),
    ("customer_sense", "customer_relevance", 0.5),
    ("team_morale", "coordination_gain", 0.3),
    # Victor — architecture / institution / protocol
    ("protocol_design", "protocol_creation_potential", 0.5),
    ("institutional_memory", "institutional_memory_gain", 0.5),
    ("claim_evidence_review", "claim_evidence_gain", 0.4),
    # Paul — vision / dominance / urgency
    ("dominance", "visibility_gain", 0.4),
    ("dominance", "protocol_creation_potential", 0.25),
    ("urgency_bias", "deadline_urgency", 0.5),
    ("urgency_bias", "progress_gain", 0.3),
    ("public_narrative", "public_reputation_effect", 0.5),
    # Skitty / Iris — artifacts / external docs
    ("artifact_bias", "protocol_creation_potential", 0.4),
    ("artifact_design", "institutional_memory_gain", 0.3),
    # governance friction — who rubber-stamps vs who scrutinizes a proposal.
    # Endorsers: conformity / group loyalty / long-termism / process commitment.
    ("conformity", "proposal_endorsement", 0.5),
    ("group_loyalty", "proposal_endorsement", 0.35),
    ("long_termism", "proposal_endorsement", 0.3),
    ("process_commitment", "proposal_endorsement", 0.3),
    # Skeptics: risk aversion / high quality bar / claim-evidence review / dominance.
    ("risk_aversion", "proposal_skepticism", 0.55),
    ("quality_bar", "proposal_skepticism", 0.4),
    ("claim_evidence_review", "proposal_skepticism", 0.3),
    ("dominance", "proposal_skepticism", 0.25),         # founders push back on others' proposals
    ("process_resistance", "proposal_endorsement", -0.3),
    # generic traits
    ("reputation_concern", "reputation_gain", 0.4),
    ("reputation_concern", "reputation_risk", -0.4),
    ("reputation_concern", "public_reputation_effect", 0.3),
    ("long_termism", "institutional_memory_gain", 0.3),
    ("long_termism", "technical_debt_risk", -0.3),
    ("long_termism", "runway_risk_delta", -0.3),
    # Two dimensions the study preregisters that had no persona channel at all.
    # Budget pressure was applied globally (see `weight`), so every agent felt
    # cost identically no matter what persona it carried — which makes a
    # profile-causality claim about cost undecidable, and leaves shuffling
    # profiles unable to move it. Ownership had no trait either: taking and
    # assigning work was driven only by role slots.
    # Magnitudes are deliberately below the base cost weights (api_cost -0.15,
    # budget_cost -0.15, runway_risk_delta -0.25) so the trait MODULATES the
    # cost rather than dominating it. A first pass at -0.30 roughly tripled the
    # penalty at the high end of the roster and suppressed the repo chain
    # entirely: a three-day smoke produced zero repo events.
    ("cost_sensitivity", "api_cost", -0.10),
    ("cost_sensitivity", "compute_cost", -0.08),
    ("cost_sensitivity", "budget_cost", -0.10),
    ("cost_sensitivity", "runway_risk_delta", -0.12),
    # Ownership means TAKING work, so it loads dependency_unlock, which is what
    # pick_task scores. Two neighbouring channels were tried and rejected on
    # evidence: coordination_gain pulled agents into meetings and assignment
    # talk, and progress_gain is dominated by work_on_task (0.7), which pinned
    # them to the task they already held instead of advancing the repo chain.
    # Either one drove a three-day smoke to zero repo events; this row alone
    # leaves it green.
    ("ownership_drive", "dependency_unlock", 0.30),
    ("risk_aversion", "technical_debt_risk", -0.3),
    ("risk_aversion", "failure_risk_from_low_skill", -0.3),
    ("risk_aversion", "protocol_violation_risk", -0.2),
    ("curiosity", "learning_gain", 0.4),
    ("curiosity", "expert_advice_value", 0.3),
    ("social_tact", "coordination_gain", 0.4),
    ("social_tact", "conflict_risk", 0.2),       # tactful -> less conflict penalty weight toward 0
    ("group_loyalty", "coordination_gain", 0.25),
    ("fairness", "norm_enforcement_gain", 0.3),
    ("overcommitment_risk", "progress_gain", 0.2),
]

TAU = 0.6   # softmax temperature (mock mode)


class OrgPolicy:
    def __init__(self, mode: str = "mock", *, use_profile_conditioning: bool = True):
        self.mode = mode
        self.use_profile_conditioning = bool(use_profile_conditioning)

    def _lookup(self, agent, name: str) -> float:
        if name in agent.profile:
            return float(agent.profile[name])
        return effective_skill(agent.skills, name, 0.0)

    def weight(self, agent, feature: str, world: Any) -> float:
        w = BASE_WEIGHTS.get(feature, 0.0)
        if self.use_profile_conditioning:
            for trait, feat, coeff in PROFILE_COEFFS:
                if feat == feature:
                    w += self._lookup(agent, trait) * coeff
        # global budget pressure makes monetary costs hurt more
        if feature in ("api_cost", "compute_cost", "budget_cost", "runway_risk_delta"):
            w *= (1.0 + world.budget_system.budget.budget_pressure)
        # night owls discount overtime; weekend workers discount weekend penalty
        if feature == "overtime_penalty":
            w *= (1.0 - 0.7 * float(agent.work_rhythm.get("late_night_bias", 0.3)))
        if feature == "weekend_penalty":
            w *= (1.0 - 0.7 * float(agent.work_rhythm.get("weekend_work_tendency", 0.3)))
        # tired/stressed agents weight fatigue/stress costs more
        if feature in ("fatigue_delta", "burnout_risk_delta"):
            w *= (1.0 + agent.vitals.get("fatigue", 0.0))
        if feature == "stress_delta":
            w *= (1.0 + agent.vitals.get("stress", 0.0))
        return w

    def score(self, agent, features, world) -> float:
        """Deterministic utility of one candidate (no noise/sampling)."""
        fd = features.to_dict() if hasattr(features, "to_dict") else dict(features)
        return sum(self.weight(agent, k, world) * v for k, v in fd.items())

    def select(self, agent, perception, scored_candidates: List[Tuple[Any, Any]],
               world: Any, *, rng: random.Random):
        """scored_candidates: list of (candidate, features). Returns the chosen
        ActionCandidate via seeded softmax over utilities (deterministic per rng).
        Utilities are shaped by the v4 attractor guard's marginal-utility penalties, and a
        top-k policy trace is recorded so action choices are explainable."""
        if not scored_candidates:
            return None
        tick = int(getattr(world, "world_tick", 0))
        profile_decisions = self._programbench_candidate_decisions(
            world, scored_candidates
        )
        raw = [self.score(agent, feats, world) for _, feats in scored_candidates]
        # §11.3 / v8 #3.3: small soft priors for acting in a domain the agent has standing in
        if self.use_profile_conditioning:
            abonus = [self._authority_bonus(agent, c, world) for c, _ in scored_candidates]
            rbonus = [self._reputation_bonus(agent, c, world) for c, _ in scored_candidates]
            cbonus = [self._coding_bonus(agent, c, world) for c, _ in scored_candidates]   # v11 §8
        else:
            abonus = [0.0] * len(scored_candidates)
            rbonus = [0.0] * len(scored_candidates)
            cbonus = [0.0] * len(scored_candidates)
        base = [r + ab + rb + cb for r, ab, rb, cb in zip(raw, abonus, rbonus, cbonus)]
        pens = [self._attractor_penalties(c, agent, world, tick) for c, _ in scored_candidates]
        utils = [b + sum(p.values()) for b, p in zip(base, pens)]
        if profile_decisions is not None:
            utils = [
                utility + decision.bonus
                for utility, decision in zip(utils, profile_decisions)
            ]
            selectable = [
                index
                for index, decision in enumerate(profile_decisions)
                if decision.allowed
            ]
            if not selectable:
                self._record_trace(
                    agent,
                    world,
                    tick,
                    scored_candidates,
                    base,
                    pens,
                    utils,
                    None,
                    abonus=abonus,
                    rbonus=rbonus,
                    decision_id=None,
                    source="rule",
                    profile_decisions=profile_decisions,
                )
                return None
        else:
            selectable = list(range(len(scored_candidates)))
        # Keep the native sampler byte-for-byte equivalent; the adapted path
        # draws jitter only for candidates that survived the hard guard.
        if profile_decisions is None:
            jit = [u + rng.uniform(-0.05, 0.05) for u in utils]
            if self.mode == "argmax":
                chosen = max(range(len(jit)), key=lambda i: jit[i])
            else:
                m = max(jit)
                exps = [math.exp((u - m) / TAU) for u in jit]
                total = sum(exps) or 1.0
                r = rng.random() * total
                acc, chosen = 0.0, len(jit) - 1
                for i, e in enumerate(exps):
                    acc += e
                    if r <= acc:
                        chosen = i
                        break
        else:
            jit = {
                index: utils[index] + rng.uniform(-0.05, 0.05)
                for index in selectable
            }
            if self.mode == "argmax":
                chosen = max(selectable, key=lambda i: jit[i])
            else:
                m = max(jit.values())
                exps = {
                    index: math.exp((jit[index] - m) / TAU)
                    for index in selectable
                }
                total = sum(exps.values()) or 1.0
                r = rng.random() * total
                acc, chosen = 0.0, selectable[-1]
                for i in selectable:
                    e = exps[i]
                    acc += e
                    if r <= acc:
                        chosen = i
                        break
        dec = self._decision_for(world, agent.id, tick)
        self._record_trace(agent, world, tick, scored_candidates, base, pens, utils, chosen,
                           abonus=abonus, rbonus=rbonus, decision_id=getattr(dec, "decision_id", None),
                           source=getattr(dec, "decision_source", "rule"),
                           profile_decisions=profile_decisions)
        return scored_candidates[chosen][0]

    def trace_choice(self, agent, world, tick, scored_candidates, chosen_candidate, *,
                     decision_id=None, source="llm") -> None:
        """Record a policy trace for an externally-chosen action (e.g. an accepted LLM
        decision) so every executed action has an explainable, decision-joined trace
        showing what the policy WOULD have scored vs what was actually chosen (§4)."""
        if not scored_candidates:
            return
        profile_decisions = self._programbench_candidate_decisions(
            world, scored_candidates
        )
        raw = [self.score(agent, feats, world) for _, feats in scored_candidates]
        if self.use_profile_conditioning:
            abonus = [self._authority_bonus(agent, c, world) for c, _ in scored_candidates]
            rbonus = [self._reputation_bonus(agent, c, world) for c, _ in scored_candidates]
            cbonus = [self._coding_bonus(agent, c, world) for c, _ in scored_candidates]   # v11 §8
        else:
            abonus = [0.0] * len(scored_candidates)
            rbonus = [0.0] * len(scored_candidates)
            cbonus = [0.0] * len(scored_candidates)
        base = [r + ab + rb + cb for r, ab, rb, cb in zip(raw, abonus, rbonus, cbonus)]
        pens = [self._attractor_penalties(c, agent, world, tick) for c, _ in scored_candidates]
        utils = [b + sum(p.values()) for b, p in zip(base, pens)]
        if profile_decisions is not None:
            utils = [
                utility + decision.bonus
                for utility, decision in zip(utils, profile_decisions)
            ]
        chosen = next((i for i, (c, _) in enumerate(scored_candidates) if c is chosen_candidate), 0)
        self._record_trace(agent, world, tick, scored_candidates, base, pens, utils, chosen,
                           abonus=abonus, rbonus=rbonus, decision_id=decision_id, source=source,
                           profile_decisions=profile_decisions)

    @staticmethod
    def _programbench_candidate_decisions(world, scored_candidates):
        """Return the explicit task-family overlay, or ``None`` in native mode.

        Keeping the import and all candidate inspection behind the active-state
        check leaves the native policy path (including RNG consumption and trace
        schema) unchanged.  Candidate parameters are passed through verbatim so
        probe mode and integration-candidate ownership remain auditable inputs.
        """

        if "programbench_profile_state" not in getattr(world, "__dict__", {}):
            return None

        from environments.org_env.programbench.leaderboard_profile import (
            EXPLORE_PROTOCOL_FORMATION_TICKS,
            candidate_decision,
            refresh_programbench_protocol_adaptation,
        )

        state = refresh_programbench_protocol_adaptation(world)
        phase = state.get("phase")
        tick = int(getattr(world, "world_tick", 0) or 0)
        observation_until = state.get("protocol_observation_until_tick")
        observation_active = bool(
            observation_until is not None
            and tick < int(observation_until)
        )
        adaptation_active = bool(state.get("protocol_adaptation_active"))
        friction_target_ids = list(
            state.get("protocol_repair_eligible_target_ids") or []
        )
        exploration_eligible = int(
            state.get("exploration_transition_eligible_tick") or 0
        )
        late_explore_formation = bool(
            str(phase) == "explore"
            and tick
            >= exploration_eligible - EXPLORE_PROTOCOL_FORMATION_TICKS
            and tick < exploration_eligible
        )
        return [
            candidate_decision(
                phase,
                candidate.action_type,
                parameters={
                    **(candidate.parameters or {}),
                    "programbench_protocol_observation_active": observation_active,
                    "programbench_protocol_adaptation_active": adaptation_active,
                    "programbench_protocol_friction_target_ids": friction_target_ids,
                    "programbench_exploration_protocol_formation_active": (
                        late_explore_formation
                    ),
                },
            )
            for candidate, _ in scored_candidates
        ]

    def _decision_for(self, world, agent_id, tick):
        for d in reversed(getattr(world, "action_decisions", []) or []):
            if getattr(d, "agent_id", None) == agent_id and int(getattr(d, "tick", -1)) == int(tick):
                return d
        return None

    @staticmethod
    def _authority_bonus(agent, candidate, world) -> float:
        try:
            from environments.org_env.growth.authority import policy_authority_bonus
            return policy_authority_bonus(agent, candidate.action_type, world)
        except Exception:
            return 0.0

    @staticmethod
    def _reputation_bonus(agent, candidate, world) -> float:
        try:
            from environments.org_env.growth.authority import policy_reputation_bonus
            return policy_reputation_bonus(agent, candidate.action_type, world)
        except Exception:
            return 0.0

    @staticmethod
    def _coding_bonus(agent, candidate, world) -> float:
        """v11 §8: nudge an agent toward coding actions matching its coding profile."""
        try:
            from environments.org_env.coding.profile import coding_policy_bonus
            return coding_policy_bonus(agent, candidate.action_type, world)
        except Exception:
            return 0.0

    def _attractor_penalties(self, candidate, agent, world, tick) -> dict:
        guard = getattr(world, "_attractor_guard", None)
        if guard is None:
            return {}
        try:
            return guard.penalties(candidate, agent.id, world, tick)
        except Exception:
            return {}

    def _record_trace(self, agent, world, tick, scored, base, pens, utils, chosen,
                      *, abonus=None, rbonus=None, decision_id=None, source="rule",
                      profile_decisions=None) -> None:
        from environments.org_env.experiments.profile_causality import (
            record_profile_counterfactual,
        )

        if profile_decisions is None:
            record_profile_counterfactual(
                world=world,
                agent=agent,
                scored_candidates=scored,
                conditioned_utilities=utils,
                penalties=pens,
                chosen_index=chosen,
                policy=self,
                temperature=TAU,
            )
        elif chosen is not None:
            allowed = [
                index
                for index, decision in enumerate(profile_decisions)
                if decision.allowed
            ]
            if chosen in allowed:
                record_profile_counterfactual(
                    world=world,
                    agent=agent,
                    scored_candidates=[scored[index] for index in allowed],
                    conditioned_utilities=[utils[index] for index in allowed],
                    penalties=[pens[index] for index in allowed],
                    chosen_index=allowed.index(chosen),
                    policy=self,
                    temperature=TAU,
                )
        sink = getattr(world, "policy_trace", None)
        if sink is None:
            return
        order = sorted(range(len(utils)), key=lambda i: -utils[i])[:4]
        rows = []
        for i in order:
            c = scored[i][0]
            ab = round(abonus[i], 3) if abonus else 0.0
            rb = round(rbonus[i], 3) if rbonus else 0.0
            target = ((c.parameters or {}).get("artifact_id")
                      or (c.parameters or {}).get("task_id"))
            if profile_decisions is not None:
                target = target or (c.parameters or {}).get("protocol_id")
            row = {"action": c.action_type,
                   "target": target,
                   # v8 #3.4: surface growth's influence so it's explainable, not just
                   # in the inspector — feature score vs authority_bonus vs reputation_bonus.
                   "raw_score": round(base[i] - ab - rb, 3), "authority_bonus": ab,
                   "reputation_bonus": rb, "final_score": round(utils[i], 3)}
            row.update({k: round(v, 3) for k, v in pens[i].items()})
            if profile_decisions is not None:
                profile_decision = profile_decisions[i]
                row.update({
                    "profile_bonus": round(profile_decision.bonus, 3),
                    "profile_reason": profile_decision.reason,
                    "profile_allowed": profile_decision.allowed,
                })
            rows.append(row)
        trace = {
            "agent_id": agent.id, "tick": tick,
            "decision_id": decision_id, "decision_source": source,
            "action_selection_mode": getattr(
                world, "action_selection_mode", "profile_policy"
            ),
            "profile_conditioning_enabled": self.use_profile_conditioning,
            "candidate_actions": rows,
            "masked_actions": getattr(world, "_attractor_masked", {}).get(agent.id, []),
            "chosen_action": scored[chosen][0].action_type if chosen is not None else None,
        }
        if profile_decisions is not None:
            state = getattr(world, "__dict__", {}).get(
                "programbench_profile_state", {}
            )
            if isinstance(state, dict):
                evidence = state.get("protocol_friction_evidence")
                trace["programbench_profile"] = {
                    "phase": state.get("phase"),
                    "previous_phase": state.get("previous_phase"),
                    "phase_started_tick": state.get("phase_started_tick"),
                    "phase_transition_count": state.get(
                        "phase_transition_count"
                    ),
                    "transition_protocol_ids": list(
                        state.get("transition_protocol_ids") or []
                    ),
                    "protocol_registry_event_cursor": state.get(
                        "protocol_registry_event_cursor"
                    ),
                    "protocol_world_event_cursor": state.get(
                        "protocol_world_event_cursor"
                    ),
                    "protocol_transition_mode": state.get(
                        "protocol_transition_mode"
                    ),
                    "protocol_observation_until_tick": state.get(
                        "protocol_observation_until_tick"
                    ),
                    "protocol_friction_target_ids": list(
                        state.get("protocol_friction_target_ids") or []
                    ),
                    "protocol_live_friction_target_ids": list(
                        state.get("protocol_live_friction_target_ids") or []
                    ),
                    "protocol_friction_evidence_digest": (
                        evidence.get("evidence_digest")
                        if isinstance(evidence, dict)
                        else None
                    ),
                    "protocol_adaptation_started_tick": state.get(
                        "protocol_adaptation_started_tick"
                    ),
                    "protocol_adaptation_min_until_tick": state.get(
                        "protocol_adaptation_min_until_tick"
                    ),
                    "protocol_adaptation_active": bool(
                        state.get("protocol_adaptation_active")
                    ),
                }
        sink.append(trace)
        if len(sink) > 400:
            del sink[:len(sink) - 400]


__all__ = ["OrgPolicy", "BASE_WEIGHTS", "PROFILE_COEFFS"]
