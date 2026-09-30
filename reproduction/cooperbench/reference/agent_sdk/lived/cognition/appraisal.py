"""Event Appraisal + ordered update pipeline (Priority 4).

The rule "**appraise first, then update the graph**" (§5) is what stops every
turn from double-counting persona change. After an action executes, the system
produces a *structured* :class:`EventAppraisal` (LLM may help annotate it, but
it may NOT write persona directly). The appraisal is then consumed, in a fixed
order (§5.2), by the transient-state / social / memory / skill / stable-trait
updates.

This module owns:
  * :class:`EventAppraisal` — the frozen appraisal schema (§5.1).
  * :class:`EventAppraiser` — turns ``(action, result, signals)`` into an
    appraisal: fills structured fields, classifies an :class:`Intensity` tier,
    and emits the closed-set ``trigger_kinds`` (subset of
    :data:`agent_sdk.lived.core.schema.EVENT_KINDS`) that drive *stable* trait
    updates.
  * :func:`apply_event_pipeline` — the §5.2 ordered update orchestrator. It
    delegates persona updates to :class:`~agent_sdk.lived.persona.profile.ProfileUpdater`
    and graph updates to the typed graphs that are passed in; anything not
    supplied is simply skipped (so it runs in a unit test with just a profile).

Persona update *magnitudes* and the trait-trigger table live in
``agent_sdk.lived.core.schema`` (TRAIT_SPECS) so this file stays declarative.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from agent_sdk.lived.core.schema import EVENT_KINDS, Intensity


# --------------------------------------------------------------------------- #
# §5.1 Event appraisal schema (frozen)
# --------------------------------------------------------------------------- #
@dataclass
class EventAppraisal:
    """Structured appraisal of one executed action / micro-event (§5.1).

    The numeric ``*_delta`` / ``*_received`` fields are normalized [-1,1] (or
    [0,1] for the non-signed ones). ``trigger_kinds`` is the bridge to the
    stable-trait update table: each entry is a canonical
    :data:`~agent_sdk.lived.core.schema.EVENT_KINDS` string. ``intensity`` gates
    whether *stable* traits may move at all (minor → mood-only).
    """
    actor: str
    action_type: str
    turn: int = 0
    target: Optional[str] = None
    success: bool = True
    survival_delta: float = 0.0
    energy_delta: float = 0.0
    resource_delta: float = 0.0
    risk_realized: float = 0.0
    social_support_received: float = 0.0
    social_harm_received: float = 0.0
    fairness_violation_observed: bool = False
    promise_kept_or_broken: str = ""          # "kept" | "broken" | ""
    reputation_delta: float = 0.0
    knowledge_gain: float = 0.0
    prototype_result: str = ""                # "success" | "fail" | "partial" | ""
    rule_compliance_result: str = ""          # "complied" | "violated" | "rewarded" | "unjust_punishment" | ""
    emotional_valence: float = 0.0            # [-1,1] signed feeling of the outcome
    intensity: Intensity = Intensity.MINOR
    observed_by: List[str] = field(default_factory=list)
    visibility: str = "private"               # "private" | "witnessed" | "public"
    related_episode_id: Optional[str] = None
    primary_episode_id: Optional[str] = None
    trigger_kinds: List[str] = field(default_factory=list)
    # --- Stage B action-layer fields (§10) ------------------------------- #
    event_id: Optional[str] = None
    fatigue_delta: float = 0.0
    possible_bottleneck_signal: bool = False
    promise_created: bool = False
    communication_sent: bool = False
    exhaustion_event: bool = False
    related_plan_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["intensity"] = self.intensity.value
        return d


# --------------------------------------------------------------------------- #
# §5 Event appraiser — structured fields + intensity + trait-trigger kinds
# --------------------------------------------------------------------------- #
class EventAppraiser:
    """Builds :class:`EventAppraisal` from action outcome + env signals.

    Rule-based / deterministic by default (cost control, §4.2). An LLM annotator
    can be slotted behind the same ``appraise`` signature later; it may refine
    ``emotional_valence`` / ``fairness_violation_observed`` but must still go
    through this object so it never writes persona directly (§5).
    """

    # thresholds for intensity classification
    MAJOR_T = 0.5
    MODERATE_T = 0.15

    def appraise(
        self,
        *,
        actor: str,
        action_type: str,
        turn: int = 0,
        target: Optional[str] = None,
        success: bool = True,
        kind: str = "",
        repeat_count: int = 0,
        **signals: Any,
    ) -> EventAppraisal:
        """Create an appraisal. ``kind`` (optional) names a canonical trigger
        directly; otherwise it is inferred from fields. ``repeat_count`` ≥ 2
        promotes the intensity to ``REPEATED`` (the §6.1 "don't recompute every
        turn — but a sustained pattern counts" rule). Extra ``**signals`` map
        onto the appraisal fields."""
        ap = EventAppraisal(actor=actor, action_type=action_type, turn=turn,
                            target=target, success=success)
        for k, v in signals.items():
            if hasattr(ap, k):
                setattr(ap, k, v)

        ap.trigger_kinds = self._derive_trigger_kinds(ap, kind)
        ap.intensity = self._classify_intensity(ap, repeat_count)
        if not ap.emotional_valence:
            ap.emotional_valence = self._infer_valence(ap)
        return ap

    # -- trigger kinds (-> stable trait updates) ---------------------------- #
    def _derive_trigger_kinds(self, ap: EventAppraisal, explicit: str) -> List[str]:
        kinds: List[str] = []
        if explicit:
            kinds.append(explicit)
        if ap.social_support_received > 0:
            kinds.append("helped_by")
        if ap.promise_kept_or_broken == "broken" and ap.social_harm_received > 0:
            kinds += ["promise_broken_against", "betrayed_by"]
        elif ap.social_harm_received >= self.MAJOR_T:
            kinds.append("betrayed_by")
        if ap.fairness_violation_observed:
            kinds.append("fairness_violation_observed")
        if ap.survival_delta <= -self.MAJOR_T:
            kinds.append("winter_starvation")
        if ap.prototype_result == "success":
            kinds.append("prototype_success")
        elif ap.prototype_result == "fail":
            kinds.append("repeated_experiment_failure")
        if ap.rule_compliance_result == "rewarded":
            kinds.append("rule_compliance_rewarded")
        elif ap.rule_compliance_result == "unjust_punishment":
            kinds.append("unjust_punishment")
        if ap.reputation_delta >= self.MODERATE_T:
            kinds.append("public_praise")
        elif ap.reputation_delta <= -self.MODERATE_T:
            kinds.append("public_blame")
        # keep only canonical kinds, de-duplicated, order-preserving
        seen = set()
        out = []
        for k in kinds:
            if k in EVENT_KINDS and k not in seen:
                seen.add(k)
                out.append(k)
        return out

    # -- intensity tier ----------------------------------------------------- #
    def _classify_intensity(self, ap: EventAppraisal, repeat_count: int) -> Intensity:
        if repeat_count >= 2:
            return Intensity.REPEATED
        mag = max(
            abs(ap.survival_delta), abs(ap.social_harm_received),
            abs(ap.social_support_received), abs(ap.risk_realized),
            abs(ap.reputation_delta), abs(ap.knowledge_gain),
            1.0 if ap.fairness_violation_observed else 0.0,
            1.0 if ap.prototype_result == "success" else 0.0,
            1.0 if ap.promise_kept_or_broken == "broken" else 0.0,
        )
        if mag >= self.MAJOR_T:
            return Intensity.MAJOR
        if mag >= self.MODERATE_T:
            return Intensity.MODERATE
        return Intensity.MINOR

    def _infer_valence(self, ap: EventAppraisal) -> float:
        v = (ap.survival_delta + ap.energy_delta + ap.social_support_received
             + ap.reputation_delta + ap.knowledge_gain
             - ap.social_harm_received - ap.risk_realized)
        return max(-1.0, min(1.0, v))


# --------------------------------------------------------------------------- #
# §5.2 Ordered update orchestrator
# --------------------------------------------------------------------------- #
def apply_event_pipeline(
    *,
    appraisal: EventAppraisal,
    profile_state: Any,                       # ProfileState (avoid import cycle)
    profile_updater: Any,                     # ProfileUpdater
    event_graph: Any = None,
    social_graph: Any = None,
    knowledge_graph: Any = None,
    episode_manager: Any = None,
) -> Dict[str, Any]:
    """Run the §5.2 ordered update for one appraised event.

    Order (skips any stage whose collaborator is None):
      1. write event graph
      2. (appraisal already generated by the caller)
      3. update transient states (mood)
      4. update social graph
      5. write memory graph (event node already serves as memory ref)
      6. update skill / knowledge graph
      7. slow stable-trait update (gated by intensity)
      8. update episode progress

    Returns a trace dict for the explainability output (§8.10)."""
    trace: Dict[str, Any] = {"appraisal": appraisal.to_dict()}

    # 1. event graph (fact base) -------------------------------------------
    if event_graph is not None:
        ev = event_graph.new_event(
            timestamp=appraisal.turn, actor=appraisal.actor,
            action_type=appraisal.action_type, target=appraisal.target,
            success=appraisal.success,
            summary=f"{appraisal.action_type} (intensity={appraisal.intensity.value})",
        )
        trace["event_id"] = ev.event_id
        # 5. memory ref: remember high-intensity events on the profile
        if appraisal.intensity in (Intensity.MAJOR, Intensity.REPEATED):
            if ev.event_id not in profile_state.important_memories:
                profile_state.important_memories.append(ev.event_id)

    # 3 + 7. persona (mood now, stable trait if intensity allows) ----------
    trace["persona"] = profile_updater.apply_appraisal(profile_state, appraisal)

    # 4. social graph ------------------------------------------------------
    if social_graph is not None and appraisal.target:
        a, t = appraisal.actor, appraisal.target
        if appraisal.social_support_received > 0:
            social_graph.help_during_crisis(t, a, amount=appraisal.social_support_received)
        if appraisal.promise_kept_or_broken == "kept":
            social_graph.keep_promise(a, t)
        elif appraisal.promise_kept_or_broken == "broken":
            social_graph.break_promise(a, t)
        trace["social_updated"] = True

    # 6. knowledge graph (+ couple teaching into the social graph) ----------
    if knowledge_graph is not None and appraisal.knowledge_gain > 0:
        kid = f"skill::{appraisal.action_type}"
        if appraisal.prototype_result == "success":
            knowledge_graph.discover(kid, appraisal.actor, turn=appraisal.turn)
        elif appraisal.target:  # teaching: actor -> target
            knowledge_graph.teach(kid, appraisal.actor, appraisal.target, turn=appraisal.turn)
            # teaching builds a teaching edge + student->teacher trust (§5/§9)
            if social_graph is not None:
                social_graph.teach_skill(appraisal.actor, appraisal.target)
                trace["social_updated"] = True
        trace["knowledge_updated"] = True

    # 8. episode progress --------------------------------------------------
    if episode_manager is not None:
        matched = episode_manager.route_event(appraisal)
        trace["episode"] = matched

    return trace
