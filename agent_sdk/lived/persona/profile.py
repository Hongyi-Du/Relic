"""Lived persona state + experience-driven update (design doc §3, §14.8).

``ProfileState`` is the per-agent lived state carried across turns (and, via
the existing reproduction path, optionally across generations). It bundles:

  * ``profile``  — :class:`ProfileVector` long-term tendencies (§3.2)
  * ``mood``     — :class:`MoodState` short-term state (§3.2)
  * identity / role / self_image / goals / commitments (§3.2 identity block)

``ProfileUpdater`` applies *rule-based* updates from events (§14.8). The
update table here is a SCAFFOLD: each rule is a small, interpretable nudge.
Bandit / preference-learning / RL variants slot in behind the same
``apply_event`` entry point later.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agent_sdk.lived.core.contracts import IdentityGrounding, MoodState, ProfileVector
from agent_sdk.lived.core.schema import (
    INTENSITY_SCALE,
    STATE_SPECS,
    TRAIT_SPECS,
    Intensity,
    TraitNode,
    default_trait_nodes,
)


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


@dataclass
class ProfileState:
    """Full lived persona state for one agent."""
    agent_id: str
    profile: ProfileVector = field(default_factory=ProfileVector)
    mood: MoodState = field(default_factory=MoodState)

    # Identity & experience (§3.2). ``important_memories`` are memory-node ids
    # into the event/profile graph, not free text — kept as ids so they stay
    # grounded (§15.7 "memory referenced is real").
    role: str = "forager"
    self_image: str = ""
    important_memories: List[str] = field(default_factory=list)
    skills: Dict[str, float] = field(default_factory=dict)   # skill -> proficiency [0,1]
    long_term_goals: List[str] = field(default_factory=list)
    commitments: List[str] = field(default_factory=list)

    # §2.4 spatial / group grounding (home, terrain skill identity, group...).
    grounding: IdentityGrounding = field(default_factory=IdentityGrounding)
    # §8.1 rich per-trait nodes (base/current/plasticity/confidence/history).
    # Optional: the flat ``profile`` vector stays the scorer's source of truth;
    # ``traits`` mirrors it with provenance. Built lazily via ensure_trait_nodes.
    traits: Dict[str, TraitNode] = field(default_factory=dict)

    def ensure_trait_nodes(self) -> Dict[str, TraitNode]:
        """Build (once) the rich trait-node map, seeded from the flat vector."""
        if not self.traits:
            self.traits = default_trait_nodes()
            for name, node in self.traits.items():
                node.current_value = float(getattr(self.profile, name))
        return self.traits

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "profile": self.profile.to_dict(),
            "mood": self.mood.to_dict(),
            "role": self.role,
            "self_image": self.self_image,
            "important_memories": list(self.important_memories),
            "skills": dict(self.skills),
            "long_term_goals": list(self.long_term_goals),
            "commitments": list(self.commitments),
            "grounding": self.grounding.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ProfileState":
        return cls(
            agent_id=str(d.get("agent_id", "")),
            profile=ProfileVector.from_dict(d.get("profile", {})),
            mood=MoodState.from_dict(d.get("mood", {})),
            role=str(d.get("role", "forager")),
            self_image=str(d.get("self_image", "")),
            important_memories=list(d.get("important_memories", [])),
            skills=dict(d.get("skills", {})),
            long_term_goals=list(d.get("long_term_goals", [])),
            commitments=list(d.get("commitments", [])),
            grounding=IdentityGrounding.from_dict(d.get("grounding", {})),
        )


class ProfileUpdater:
    """Rule-based experience-driven profile/mood updater (§3.4, §14.8).

    SCAFFOLD: ``apply_event`` dispatches on an event ``kind`` string to a small
    table of nudges. Magnitudes live in ``rates`` so they are tunable / testable
    in one place. Returns the (mutated) ProfileState for chaining.

    Filled-in-later: swap the static table for a learned update (bandit /
    preference learning) — keep the same signature so callers don't change.
    """

    def __init__(self, rates: Dict[str, float] | None = None):
        # Default nudge magnitudes. Small so behaviour drifts, not jumps.
        self.rates = {
            "trait_nudge": 0.03,
            "mood_nudge": 0.20,
            "mood_decay": 1.0,    # global multiplier on per-state STATE_SPECS decay
        }
        if rates:
            self.rates.update(rates)

    # -- per-turn mood relaxation -------------------------------------------
    def decay_mood(self, state: ProfileState) -> ProfileState:
        """Relax every transient state toward its baseline each turn (§14.5).

        Data-driven: per-state baseline + decay come from
        :data:`agent_sdk.lived.core.schema.STATE_SPECS`, scaled by the global
        ``mood_decay`` multiplier."""
        g = self.rates["mood_decay"]
        m = state.mood
        for name, spec in STATE_SPECS.items():
            cur = getattr(m, name)
            d = min(1.0, spec.decay * g)
            setattr(m, name, _clamp(cur + (spec.baseline - cur) * d))
        return state

    # max |Δtrait| any single event may cause (§5.3 "repeated 最多 ~0.03")
    MAX_TRAIT_DELTA = 0.03

    # -- one plasticity-scaled stable-trait nudge ---------------------------
    def _nudge_trait(self, state: ProfileState, trait: str, realized: float,
                     *, turn: int = -1, cause: str = "") -> float:
        """Move one stable trait by an already-scaled ``realized`` delta,
        hard-capped at ±MAX_TRAIT_DELTA. Writes the flat ProfileVector (scorer
        source of truth) and mirrors into the rich TraitNode (history/confidence)
        if it has been built. Returns the post-clamp realized delta."""
        if trait not in TRAIT_SPECS or realized == 0.0:
            return 0.0
        realized = max(-self.MAX_TRAIT_DELTA, min(self.MAX_TRAIT_DELTA, realized))
        cur = float(getattr(state.profile, trait))
        new = _clamp(cur + realized)
        realized = new - cur
        setattr(state.profile, trait, new)
        if state.traits:  # mirror into the rich node for provenance
            node = state.traits.get(trait)
            if node is not None:
                node.current_value = new
                node.confidence = min(1.0, node.confidence + 0.05)
                node.update_history.append(
                    {"turn": turn, "delta": round(realized, 5), "cause": cause}
                )
        return realized

    # -- event-driven update -------------------------------------------------
    def apply_event(self, state: ProfileState, kind: str, **ctx: Any) -> ProfileState:
        """Apply one event's effect to the profile/mood.

        ``kind`` examples (design doc §3.4 / §14.8):
          helped_by, betrayed_by, winter_starvation, successful_leadership,
          successful_teaching, unjust_punishment.
        ``ctx`` carries event specifics (e.g. ``source_uid``) — currently
        unused by the stub rules but threaded through for the learned variant.
        """
        t = self.rates["trait_nudge"]
        mn = self.rates["mood_nudge"]
        p, m = state.profile, state.mood

        if kind == "helped_by":
            m.gratitude = _clamp(m.gratitude + mn)
            p.reciprocity = _clamp(p.reciprocity + t)
        elif kind == "betrayed_by":
            p.distrust_sensitivity = _clamp(p.distrust_sensitivity + t)
            p.risk_aversion = _clamp(p.risk_aversion + t)
            m.anger = _clamp(m.anger + mn)
        elif kind == "winter_starvation":
            p.long_termism = _clamp(p.long_termism + t)
            p.risk_aversion = _clamp(p.risk_aversion + t)
            m.fear = _clamp(m.fear + mn)
        elif kind == "successful_leadership":
            p.dominance = _clamp(p.dominance + t)
            m.confidence = _clamp(m.confidence + mn)
        elif kind == "successful_teaching":
            p.altruism = _clamp(p.altruism + t)
            m.valence = _clamp(m.valence + mn * 0.5)
        elif kind == "unjust_punishment":
            # grievance ~ conformity down, anger up (§3.4 "depends on fairness belief")
            p.conformity = _clamp(p.conformity - t)
            m.anger = _clamp(m.anger + mn)
        # Unknown kinds are a no-op — keep the loop resilient.
        return state

    # -- appraisal-driven update (§5.2 steps 3 + 7) -------------------------
    def apply_appraisal(self, state: ProfileState, appraisal: Any) -> Dict[str, Any]:
        """Consume a structured :class:`~agent_sdk.lived.cognition.appraisal.EventAppraisal`:
        update transient states now, and *slowly* update stable traits — but
        only if the event's intensity tier permits (minor → mood-only, §5.3).

        Returns a trace ``{"mood_deltas", "trait_deltas", "intensity"}`` for the
        explainability output (§8.10). This is the new main update path; the
        legacy ``apply_event(kind=...)`` shim above remains for callers that pass
        a bare kind string."""
        mood_deltas = self._apply_mood_from_appraisal(state, appraisal)

        trait_deltas: Dict[str, float] = {}
        scale = INTENSITY_SCALE.get(appraisal.intensity, 0.0)
        turn = int(getattr(appraisal, "turn", -1))
        kinds = list(getattr(appraisal, "trigger_kinds", []) or [])
        if scale > 0.0 and kinds:
            # One event moves a given trait AT MOST ONCE (§5.3 per-event budget):
            # corroborating signals don't stack. Determine each trait's net sign
            # from matching triggers, then apply a single plasticity*scale nudge.
            for tname, spec in TRAIT_SPECS.items():
                pos = any(k in spec.positive_triggers for k in kinds)
                neg = any(k in spec.negative_triggers for k in kinds)
                if pos == neg:        # neither matched, or contradictory -> skip
                    continue
                sign = 1.0 if pos else -1.0
                causes = [k for k in kinds
                          if k in (spec.positive_triggers if pos else spec.negative_triggers)]
                d = self._nudge_trait(
                    state, tname, sign * spec.plasticity * scale,
                    turn=turn, cause="+".join(causes),
                )
                trait_deltas[tname] = d
        return {
            "mood_deltas": mood_deltas,
            "trait_deltas": trait_deltas,
            "intensity": appraisal.intensity.value,
        }

    def _apply_mood_from_appraisal(self, state: ProfileState, appraisal: Any) -> Dict[str, float]:
        """Map appraisal signals onto transient states (§5.2 step 3). Mood moves
        freely (bounded perturbation, no intensity gate — that gate is only for
        the *stable* traits)."""
        m = state.mood
        mn = self.rates["mood_nudge"]
        before = m.to_dict()

        valence_shift = float(getattr(appraisal, "emotional_valence", 0.0)) * mn
        m.valence = _clamp(m.valence + valence_shift)
        if getattr(appraisal, "social_support_received", 0.0) > 0:
            m.gratitude = _clamp(m.gratitude + mn * appraisal.social_support_received)
        if getattr(appraisal, "social_harm_received", 0.0) > 0:
            m.anger = _clamp(m.anger + mn * appraisal.social_harm_received)
        if getattr(appraisal, "risk_realized", 0.0) > 0:
            m.fear = _clamp(m.fear + mn * appraisal.risk_realized)
        if getattr(appraisal, "fairness_violation_observed", False):
            m.fairness_salience = _clamp(m.fairness_salience + mn)
            m.anger = _clamp(m.anger + mn * 0.5)
        if getattr(appraisal, "survival_delta", 0.0) < 0:
            m.hunger_pressure = _clamp(m.hunger_pressure - appraisal.survival_delta)
        if getattr(appraisal, "prototype_result", "") == "success" or \
                getattr(appraisal, "knowledge_gain", 0.0) > 0:
            m.confidence = _clamp(m.confidence + mn * 0.5)
        if getattr(appraisal, "prototype_result", "") == "fail" or not getattr(appraisal, "success", True):
            m.frustration = _clamp(m.frustration + mn)

        after = m.to_dict()
        return {k: round(after[k] - before[k], 5) for k in after if after[k] != before[k]}
