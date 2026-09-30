"""
auto_trigger/scoring.py — Reflex scoring framework (Level 4/5).

Implements the four core scalars that drive reflex life-cycle decisions:

    c_x   confidence         Laplace-smoothed quality estimate per kind.
    v_x   vitality           Time-decaying visibility subject to reinforcement
                             and displacement penalties. Clamped to [0, 1].
    r_x   adoption score     Per-turn sort key used in place of raw priority.
    u_x   utility trace      5-axis per-fire record feeding c_x and vitality.

Plus small helpers for match strength, displacement risk and cost-saving
estimation. All of these are pure functions so they can be unit-tested in
isolation and do not depend on the concrete agent / store implementation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .types import (
    AgentReflexEntry,
    ModeActivation,
    ReflexContext,
    ReflexKind,
    ReflexTemplate,
    TemplateStatus,
    UtilityTrace,
)


# ---------------------------------------------------------------------------
# Default weights (overridable via engine config)
# ---------------------------------------------------------------------------

DEFAULT_WEIGHTS: Dict[str, float] = {
    "w_match": 1.0,
    "w_vitality": 0.5,
    "w_confidence": 0.8,
    "w_group": 0.4,
    "w_cost": 0.3,
    "w_disp": 0.6,
    # priority fallback — normalised to [0,1] and added as a mild bias
    "w_priority": 0.2,
}

DEFAULT_VITALITY: Dict[str, float] = {
    "decay": 0.98,
    "bonus_up": 0.10,
    "bonus_down": 0.15,
    "disp_start": 0.15,      # displacement fraction below this is free
    "disp_scale": 2.0,       # how steeply to penalise past the start fraction
}


# ---------------------------------------------------------------------------
# 1. Confidence (kind-aware Laplace smoothing)
# ---------------------------------------------------------------------------

def compute_confidence(template: ReflexTemplate) -> float:
    """Laplace-smoothed confidence over kind-specific positive/negative fires.

    All three kinds use the same formula, but callers feed different positive
    definitions into ``template.positive_fires`` / ``negative_fires`` via
    :func:`apply_trace_to_template`.

        c = (positive + 1) / (positive + negative + 2)

    This keeps a new template at 0.5, punishes early failures moderately and
    converges to the true success rate as sample size grows. It matches the
    existing CollectiveMemory credibility model.
    """
    pos = max(0, int(template.positive_fires))
    neg = max(0, int(template.negative_fires))
    return (pos + 1.0) / (pos + neg + 2.0)


# ---------------------------------------------------------------------------
# 2. Vitality (time decay + reinforce - displacement)
# ---------------------------------------------------------------------------

def advance_vitality(
    template: ReflexTemplate,
    *,
    positive: bool = False,
    negative: bool = False,
    displacement_penalty: float = 0.0,
    cfg: Optional[Dict[str, float]] = None,
) -> float:
    """Advance the vitality scalar by one step.

    ``positive`` / ``negative`` describe the outcome of a single fire (or a
    single external upvote / downvote event). ``displacement_penalty`` is a
    non-negative scalar contributed by the aggregate reflex pressure on the
    firing agent (see :func:`displacement_penalty`).

    Returns the new vitality and writes it back onto the template.
    """
    c = dict(DEFAULT_VITALITY)
    if cfg:
        c.update(cfg)
    v = float(template.vitality) * float(c["decay"])
    if positive:
        v += float(c["bonus_up"])
    if negative:
        v -= float(c["bonus_down"])
    v -= float(displacement_penalty)
    v = max(0.0, min(1.0, v))
    template.vitality = v
    return v


def displacement_penalty(recent_fire_fraction: float, cfg: Optional[Dict[str, float]] = None) -> float:
    """Convert a 0..1 fire fraction into a vitality penalty.

    The fraction is typically ``reflex_fires_in_last_N_turns / N``. Below
    ``disp_start`` the penalty is zero; beyond that it scales linearly.
    """
    c = dict(DEFAULT_VITALITY)
    if cfg:
        c.update(cfg)
    over = max(0.0, float(recent_fire_fraction) - float(c["disp_start"]))
    return over * float(c["disp_scale"]) * float(c["bonus_down"])


# ---------------------------------------------------------------------------
# 3. Match strength (cheap heuristic; env can override later)
# ---------------------------------------------------------------------------

def match_strength(template: ReflexTemplate, ctx: ReflexContext) -> float:
    """Rough 0..1 alignment between template cues and current context.

    We avoid fully re-evaluating the JSON condition here (that is the job
    of :class:`ConditionEvaluator`) and instead inspect well-known cues:

    * A matching pressure signal (``resource_pressure`` / ``danger_level``)
      boosts the score.
    * Low inventory of an item mentioned in the condition boosts the score.
    * Mode reflexes inspect ``ctx.trends`` and ``ctx.group_metrics`` — the
      env is expected to populate those before evaluation.
    """
    score = 0.0
    cond = template.condition or {}
    # Walk ANDed subconditions; tolerate a flat dict too.
    subs: List[dict] = []
    if cond.get("type") == "and":
        subs = list(cond.get("conditions", []))
    elif cond:
        subs = [cond]

    for sub in subs:
        ctype = sub.get("type", "")
        if ctype == "threshold":
            field = sub.get("field", "")
            op = sub.get("op", "lt")
            target = float(sub.get("value", 0.0) or 0.0)
            if field == "energy_pct":
                actual = float(ctx.energy_pct or 0.0)
                score += _close_to(op, actual, target)
            elif field == "hp_pct":
                actual = float(ctx.hp_pct or 0.0)
                score += _close_to(op, actual, target)
        elif ctype == "inventory":
            item = sub.get("item", "")
            target = float(sub.get("value", 0.0) or 0.0)
            op = sub.get("op", "lt")
            actual = float(ctx.inventory.get(item, 0.0) or 0.0)
            score += _close_to(op, actual, target)
        elif ctype == "entity_nearby":
            key = sub.get("entity_type", "")
            state = sub.get("entity_state", "")
            lookup = f"{key}_{state}" if state else key
            ent = ctx.nearest_entities.get(lookup) or ctx.nearest_entities.get(key)
            if ent is not None:
                max_d = float(sub.get("max_distance", 10) or 10)
                score += max(0.0, 1.0 - (float(ent.distance) / max_d))
        elif ctype == "signal":
            field = sub.get("field", "")
            want = str(sub.get("value", "")).lower()
            have = str(getattr(ctx, field, "")).lower()
            if have and want:
                order = {"none": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}
                score += 0.5 + 0.125 * max(0, order.get(have, 0) - order.get(want, 0))
        elif ctype == "signal_received":
            sig_type = sub.get("signal_type", "")
            if any(s.signal_type == sig_type for s in ctx.received_signals):
                score += 1.0
        elif ctype == "bool_flag":
            key = sub.get("key", "")
            expected = bool(sub.get("expected", True))
            have = bool(ctx.observation_metadata.get(key, False))
            score += 1.0 if have == expected else 0.0
        elif ctype == "mode_active":
            mid = sub.get("mode_id", "")
            score += 1.0 if mid in ctx.active_modes else 0.0
        elif ctype == "group_pressure":
            key = sub.get("field", "")
            want = float(sub.get("value", 0.0) or 0.0)
            have = float(ctx.group_metrics.get(key, 0.0) or 0.0)
            score += 1.0 if have >= want else max(0.0, have / max(want, 1e-6))
        elif ctype == "trend":
            key = sub.get("field", "")
            want = float(sub.get("value", 0.0) or 0.0)
            have = float(ctx.trends.get(key, 0.0) or 0.0)
            score += 1.0 if (sub.get("op", "lt") == "lt" and have < want) or \
                              (sub.get("op", "lt") == "gt" and have > want) else 0.0

    if not subs:
        return 0.5
    return max(0.0, min(1.0, score / float(len(subs))))


def _close_to(op: str, actual: float, target: float) -> float:
    """Small helper: how 'satisfied' a numeric comparison is, 0..1."""
    if op in ("lt", "lte"):
        if actual <= target:
            return 1.0
        # gently decay as we move above the threshold
        return max(0.0, 1.0 - (actual - target) / max(abs(target), 1.0))
    if op in ("gt", "gte"):
        if actual >= target:
            return 1.0
        return max(0.0, 1.0 - (target - actual) / max(abs(target), 1.0))
    if op == "eq":
        return 1.0 if abs(actual - target) < 1e-6 else 0.0
    return 0.0


# ---------------------------------------------------------------------------
# 4. Displacement risk per agent
# ---------------------------------------------------------------------------

def reflex_fire_fraction(
    fire_history: List[Tuple[int, str, str]],
    current_turn: int,
    window: int = 20,
) -> float:
    """Return the fraction of recent turns that were consumed by reflex fires.

    ``fire_history`` matches AgentBase._reflex_fire_history: a list of
    (turn, name, outcome_snippet) tuples.
    """
    if window <= 0:
        return 0.0
    lo = current_turn - window + 1
    hit_turns = {t for (t, _name, _out) in fire_history if t >= lo}
    return min(1.0, len(hit_turns) / float(window))


# ---------------------------------------------------------------------------
# 5. Cost saving estimate (u_cost prior)
# ---------------------------------------------------------------------------

def estimated_cost_saving(template: ReflexTemplate) -> float:
    """Rough 0..1 estimate of how much external inference this reflex saves.

    Heuristic: FREE-action templates (``expected_free=True``) save nothing
    because the agent would have gotten a free turn anyway. Normal templates
    save a full Stage A+B cycle; high-priority low-cooldown reflexes thus
    score higher. Mode templates save substantial LLM if they avoid repeated
    deliberation — but only when actually active.
    """
    if template.kind == ReflexKind.MODE.value:
        return 0.6
    base = 0.0 if template.expected_free else 1.0
    # Short cooldown means more frequent savings per run
    cd = max(1, int(template.cooldown_turns or 1))
    return base * (1.0 / (1.0 + 0.05 * cd))


# ---------------------------------------------------------------------------
# 6. Adoption score — the replacement for priority-sort selection
# ---------------------------------------------------------------------------

@dataclass
class AdoptionScoreBreakdown:
    """Diagnostic breakdown of r_x. Helpful for logs/tests."""
    match: float = 0.0
    vitality: float = 0.0
    confidence: float = 0.0
    group: float = 0.0
    cost: float = 0.0
    displacement: float = 0.0
    priority: float = 0.0
    total: float = 0.0


def compute_adoption_score(
    template: ReflexTemplate,
    ctx: ReflexContext,
    *,
    recent_fire_fraction: float = 0.0,
    weights: Optional[Dict[str, float]] = None,
    match_override: Optional[float] = None,
    group_fit_override: Optional[float] = None,
) -> AdoptionScoreBreakdown:
    w = dict(DEFAULT_WEIGHTS)
    if weights:
        w.update(weights)
    match = float(match_override) if match_override is not None else match_strength(template, ctx)
    vitality = float(template.vitality)
    confidence = compute_confidence(template)
    group = float(group_fit_override) if group_fit_override is not None else _group_fit(template, ctx)
    cost = estimated_cost_saving(template)
    disp_pen = displacement_penalty(recent_fire_fraction)
    prio = max(0.0, min(1.0, float(template.priority) / 100.0))

    total = (
        w["w_match"] * match
        + w["w_vitality"] * vitality
        + w["w_confidence"] * confidence
        + w["w_group"] * group
        + w["w_cost"] * cost
        - w["w_disp"] * disp_pen
        + w["w_priority"] * prio
    )
    return AdoptionScoreBreakdown(
        match=match, vitality=vitality, confidence=confidence,
        group=group, cost=cost, displacement=disp_pen,
        priority=prio, total=total,
    )


def _group_fit(template: ReflexTemplate, ctx: ReflexContext) -> float:
    """How well the template suits the current group/mode situation (0..1).

    * PAIR templates score higher when there is a coordination_need signal.
    * MODE templates score higher when their activation conditions are
      already met (delegated to match_strength) OR when there is a strongly
      aligned pressure signal.
    * SELF templates default to 0.5 — neutral.
    """
    kind = template.kind
    if kind == ReflexKind.PAIR.value:
        order = {"none": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}
        return min(1.0, order.get(ctx.coordination_need, 0) / 4.0)
    if kind == ReflexKind.MODE.value:
        # Prefer higher pressure of any kind.
        order = {"none": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}
        rp = order.get(ctx.resource_pressure, 0)
        dl = order.get(ctx.danger_level, 0)
        return min(1.0, max(rp, dl) / 4.0)
    return 0.5


# ---------------------------------------------------------------------------
# 7. UtilityTrace construction helpers
# ---------------------------------------------------------------------------

def build_utility_trace(
    template: ReflexTemplate,
    *,
    turn: int,
    agent_id: str,
    outcome: str,
    free_action: bool = False,
    pair_closed: Optional[bool] = None,
    mode_delta: Optional[float] = None,
    energy_delta_pct: float = 0.0,
    hp_delta_pct: float = 0.0,
    metric_delta: float = 0.0,
    recent_fire_fraction: float = 0.0,
) -> UtilityTrace:
    """Derive a UtilityTrace from an execution outcome.

    The agent core calls this once per reflex fire. It encodes the policy
    choices in the upgrade doc Section 7:

    * u_cost: 1 if the fire shortcut a Stage A+B decision, else 0.
    * u_surv: +energy_delta_pct + hp_delta_pct, signed.
    * u_task: metric_delta as reported by the env (already bounded-ish).
    * u_disp: negative penalty derived from recent fire fraction.
    * u_coord: +1 for pair closure, +mode_delta for mode windows, else 0.
    """
    kind = template.kind

    # LLM cost saved — only if the fire prevented a decision cycle
    if free_action or kind == ReflexKind.MODE.value:
        u_cost = 0.3  # mode / free does not fully replace a decision
    elif outcome == "success":
        u_cost = 1.0
    else:
        u_cost = 0.0

    u_surv = float(energy_delta_pct) + float(hp_delta_pct)
    u_task = float(metric_delta)

    # Displacement penalty feeds a NEGATIVE u_disp channel directly.
    disp = displacement_penalty(recent_fire_fraction)
    u_disp = -disp

    u_coord = 0.0
    if pair_closed is True:
        u_coord += 1.0
    elif pair_closed is False:
        u_coord -= 0.5
    if mode_delta is not None:
        u_coord += float(mode_delta)

    return UtilityTrace(
        turn=turn,
        agent_id=agent_id,
        template_id=template.template_id,
        kind=kind,
        u_cost=u_cost,
        u_surv=u_surv,
        u_task=u_task,
        u_disp=u_disp,
        u_coord=u_coord,
        outcome=outcome,
    )


# ---------------------------------------------------------------------------
# 8. Apply a trace to a template (store-side update)
# ---------------------------------------------------------------------------

def apply_trace_to_template(
    template: ReflexTemplate,
    trace: UtilityTrace,
    *,
    pair_closed: Optional[bool] = None,
) -> None:
    """Update template aggregate counters + vitality from a UtilityTrace.

    Positive/negative classification is kind-specific:

    * SELF:  positive iff trace.outcome == "success".
    * PAIR:  positive iff pair_closed is True.
    * MODE:  positive iff trace.total() > 0.
    """
    template.total_fires += 1
    template.last_seen_turn = max(int(template.last_seen_turn), int(trace.turn))

    positive = False
    negative = False
    if template.kind == ReflexKind.PAIR.value:
        if pair_closed is True:
            positive = True
        elif pair_closed is False:
            negative = True
    elif template.kind == ReflexKind.MODE.value:
        positive = trace.total() > 0.0
        negative = not positive
    else:
        positive = trace.outcome == "success"
        negative = not positive

    if positive:
        template.positive_fires += 1
    if negative:
        template.negative_fires += 1

    template.confidence = compute_confidence(template)
    # Vitality absorbs displacement penalty baked into trace.u_disp (<=0).
    disp = max(0.0, -float(trace.u_disp))
    advance_vitality(
        template,
        positive=positive,
        negative=negative,
        displacement_penalty=disp,
    )


# ---------------------------------------------------------------------------
# 9. Mode-window utility accumulator
# ---------------------------------------------------------------------------

def accumulate_mode_window(
    activation: ModeActivation,
    *,
    llm_cost_saved: float = 0.0,
    energy_delta_pct: float = 0.0,
    hp_delta_pct: float = 0.0,
    metric_delta: float = 0.0,
    pair_closure_delta: float = 0.0,
    thrash_delta: float = 0.0,
) -> None:
    """Feed one turn of per-agent metrics into an open mode window."""
    activation.u_cost += float(llm_cost_saved)
    activation.u_surv += float(energy_delta_pct) + float(hp_delta_pct)
    activation.u_task += float(metric_delta)
    activation.u_coord += float(pair_closure_delta)
    activation.u_thrash += float(thrash_delta)


__all__ = [
    "DEFAULT_WEIGHTS",
    "DEFAULT_VITALITY",
    "AdoptionScoreBreakdown",
    "compute_confidence",
    "advance_vitality",
    "displacement_penalty",
    "match_strength",
    "reflex_fire_fraction",
    "estimated_cost_saving",
    "compute_adoption_score",
    "build_utility_trace",
    "apply_trace_to_template",
    "accumulate_mode_window",
]
