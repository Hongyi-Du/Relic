"""Persona-Conditioned Bounded Softmax Policy (PCBSP) — the v1 decision model.

This is how a SocioGenesis v1 agent selects an action from a *shared*, state-
derived candidate pool. The persona only re-weights and samples; it never adds
or removes candidates, and the LLM never decides the utility, formula, weights,
or persona update (§0).

Pipeline (§1):
  observe → retrieve subgraph → shared candidate pool → annotate features (§4/§5)
  → feasibility + survival guards (§13) → transform features (§6)
  → utility = SurvivalScore + TraitScore + RelationScore + MoodShift
              + EpisodeValue − CostScore                                (§12)
  → satisficing shortlist (§14) → seeded softmax sample (§15)
  → policy trace (§16)

Utility components:
  * SurvivalScore — profile-independent survival drive, amplified by the §6.4
    hunger/hp pressure (so a starving agent's eat/gather utility spikes).
  * TraitScore    — Σ_t Σ_f trait[t]·W[t][f]·transformed_feature[f] (§8/§12.1).
  * RelationScore — target-specific social-graph modifiers (§9).
  * MoodShift     — transient-state perturbation (§10).
  * EpisodeValue  — foreground-episode continuation bonuses (§11).
  * CostScore     — profile-independent baseline over the §12.2 cost set.

Everything env-specific (feature annotation, vitals, social graph, episodes)
arrives via :class:`DecisionContext` so ``agent_sdk`` stays env-agnostic.
"""
from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from agent_sdk.lived.core.contracts import (
    ActionCandidate,
    ActionFeatures,
    MoodState,
    ProfileVector,
    ScoredCandidate,
)
from agent_sdk.lived.core.ports import FeatureExtractorPort, NoOpFeatureExtractor
from agent_sdk.lived.core.schema import COSTSCORE_FEATURES, SURVIVAL_FEATURES
from agent_sdk.lived.core.transforms import SurvivalPressure, transform_features
from agent_sdk.lived.core.wmatrix import profile_to_weights

# §15.1 temperature constants.
TAU_BASE = 0.7
TAU_MIN = 0.2
TAU_MAX = 1.5
# §14 satisficing.
DEFAULT_TOP_K = 5
DEFAULT_DELTA = 1.0
# §13 survival-guard thresholds.
ENERGY_URGENT = 30.0
HP_URGENT = 40.0
ENERGY_CRITICAL = 10.0
HP_CRITICAL = 20.0


# --------------------------------------------------------------------------- #
# Decision context — the env seam (§2.1 inputs)
# --------------------------------------------------------------------------- #
@dataclass
class DecisionContext:
    """All optional, env-supplied inputs for one decision (§2.1)."""
    turn_id: int = 0
    decision_id: str = "0"
    run_seed: int = 0
    energy: Optional[float] = None
    hp: Optional[float] = None
    state: Any = None                       # passed to the feature annotator
    social: Any = None                      # SocialGraph-like (trust/debt/grievance)
    episodes: Any = None                    # EpisodeManager-like
    groups: Dict[str, str] = field(default_factory=dict)   # uid -> group_id
    mask_fn: Optional[Callable[[ActionCandidate], bool]] = None  # True == feasible
    # PlanValue (§5): the PlanMonitor's per-action value modifiers — active
    # plans raise the utility of the actions that advance their current step
    # (e.g. RECOVER -> rest, MATERIAL_PROBLEM_SOLVING -> gather/attempt_craft).
    # PlanMonitorResult.plan_value_modifiers was designed for exactly this
    # consumer; empty dict == no active-plan shaping (fully backwards compatible).
    plan_modifiers: Dict[str, float] = field(default_factory=dict)

    @property
    def pressure(self) -> SurvivalPressure:
        return SurvivalPressure.from_vitals(self.energy, self.hp)


# --------------------------------------------------------------------------- #
# §16 Policy trace
# --------------------------------------------------------------------------- #
@dataclass
class PolicyTrace:
    agent_id: str
    turn_id: int = 0
    candidate_pool: List[str] = field(default_factory=list)
    masked_actions: List[str] = field(default_factory=list)
    feature_annotations: Dict[str, Dict[str, float]] = field(default_factory=dict)
    scored: List[ScoredCandidate] = field(default_factory=list)
    shortlist: List[str] = field(default_factory=list)
    probabilities: Dict[str, float] = field(default_factory=dict)
    selected_action: Optional[str] = None
    selected_parameters: Dict[str, Any] = field(default_factory=dict)
    action_probability: float = 0.0
    sampling_temperature: float = 0.0
    random_draw: float = 0.0
    explanation_paths: List[str] = field(default_factory=list)

    @property
    def chosen(self) -> Optional[ScoredCandidate]:
        for s in self.scored:
            if s.candidate.action_type == self.selected_action:
                return s
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id, "turn_id": self.turn_id,
            "candidate_pool": list(self.candidate_pool),
            "masked_actions": list(self.masked_actions),
            "feature_annotations": self.feature_annotations,
            "utility_breakdown": {
                s.candidate.action_type: {"utility": round(s.utility, 4),
                                          **{k: round(v, 4) for k, v in s.contributions.items()}}
                for s in self.scored
            },
            "shortlist": list(self.shortlist),
            "probabilities": {k: round(v, 4) for k, v in self.probabilities.items()},
            "selected_action": self.selected_action,
            "selected_parameters": dict(self.selected_parameters),
            "action_probability": round(self.action_probability, 4),
            "sampling_temperature": round(self.sampling_temperature, 4),
            "random_draw": round(self.random_draw, 4),
            "explanation_paths": list(self.explanation_paths),
        }


# --------------------------------------------------------------------------- #
# §11 episode continuation bonuses (action_type substring -> bonus)
# --------------------------------------------------------------------------- #
_EPISODE_BONUS: Dict[str, Dict[str, float]] = {
    "material_prototype": {"continue": 0.5, "session": 0.5, "collect": 0.4,
                           "gather": 0.4, "test_prototype": 0.6, "test": 0.5},
    "teaching": {"continue_teaching": 0.5, "teach": 0.5, "complete_lesson": 0.7,
                 "demonstrate": 0.4},
    "build_project": {"build": 0.4, "gather": 0.4, "deposit": 0.4, "construct": 0.5,
                      "haul": 0.3},
    "civic_trial": {"follow": 0.5, "comply": 0.5, "deposit": 0.5, "record": 0.5,
                    "support": 0.4, "enforce": 0.4, "vote": 0.3},
    "dispute": {"accuse": 0.3, "witness": 0.3, "mediate": 0.4, "appeal": 0.3,
                "compensate": 0.3},
}


# --------------------------------------------------------------------------- #
# The policy
# --------------------------------------------------------------------------- #
class PCBSPPolicy:
    """Persona-Conditioned Bounded Softmax Policy (§12-§15)."""

    def __init__(self, feature_extractor: Optional[FeatureExtractorPort] = None,
                 rng: Optional[random.Random] = None):
        self.feature_extractor: FeatureExtractorPort = feature_extractor or NoOpFeatureExtractor()
        self.rng = rng or random.Random()

    # ===== component scores ============================================== #
    def survival_score(self, tf: Dict[str, float], pressure: SurvivalPressure) -> float:
        """§13.2/§6.4 profile-independent survival drive, amplified by emergency
        pressure so eat/gather/food spikes when energy/hp is low."""
        base = (tf.get("survival_gain", 0.0)
                + 0.8 * tf.get("energy_gain", 0.0)
                + 0.6 * tf.get("resource_gain", 0.0))
        return base * (1.0 + 2.0 * pressure.max)

    def trait_score(self, tf: Dict[str, float], profile: ProfileVector) -> float:
        """§12.1 Σ_t Σ_f trait·W·transformed_feature, via the compiled weights."""
        w = profile_to_weights(profile).weights
        return sum(w.get(f, 0.0) * v for f, v in tf.items())

    def cost_score(self, tf: Dict[str, float]) -> float:
        """§12.2 profile-independent baseline cost (the seven physical/risk
        costs). Trait modulation of these costs lives in TraitScore via negative
        W cells, so this is only the baseline (base + modulation, no double base)."""
        return sum(tf.get(f, 0.0) for f in COSTSCORE_FEATURES)

    def mood_shift(self, mood: MoodState, candidate: ActionCandidate) -> float:
        """§10 transient-state perturbation (bounded; never edits stable traits)."""
        a = candidate.action_type.lower()
        s = 0.0
        if mood.hunger_pressure > 0.5 and any(k in a for k in ("eat", "gather", "withdraw", "forage", "hunt")):
            s += 0.4 * mood.hunger_pressure
        if mood.fatigue > 0.5 and any(k in a for k in ("rest", "sleep", "idle")):
            s += 0.3 * mood.fatigue
        if mood.anger > 0.5 and any(k in a for k in ("accuse", "blame", "refuse", "confront", "challenge", "punish")):
            s += 0.3 * mood.anger
        if mood.gratitude > 0.5 and any(k in a for k in ("help", "repay", "teach", "share", "give")):
            s += 0.3 * mood.gratitude
        if mood.confidence > 0.5 and any(k in a for k in ("experiment", "propose", "lead", "challenge", "coordinate", "announce")):
            s += 0.25 * mood.confidence
        if mood.frustration > 0.5 and any(k in a for k in ("ask_help", "abandon", "retry", "wish")):
            s += 0.25 * mood.frustration
        if mood.fear > 0.5 and any(k in a for k in ("avoid", "return_home", "flee", "seek_help", "retreat")):
            s += 0.3 * mood.fear
        if mood.fairness_salience > 0.5 and any(k in a for k in ("propose_rule", "accuse", "tally", "mark", "fair")):
            s += 0.3 * mood.fairness_salience
        return s

    def relation_score(self, candidate: ActionCandidate, profile: ProfileVector,
                       ctx: DecisionContext, agent_id: str) -> float:
        """§9 target-specific social-graph modifiers."""
        social = ctx.social
        tgt = candidate.target_uid
        if social is None or not tgt:
            return 0.0
        a = candidate.action_type.lower()
        score = 0.0
        trust = _safe(lambda: social.trust(agent_id, tgt), 0.0)
        debt = _safe(lambda: social.debts_owed_by(agent_id).get(tgt, 0.0), 0.0)
        grievance = _safe(lambda: social.store.edge_attr(agent_id, tgt, "grievance", "strength", 0.0), 0.0)
        same_group = 1.0 if (ctx.groups.get(agent_id) and ctx.groups.get(agent_id) == ctx.groups.get(tgt)) else 0.0

        if any(k in a for k in ("help", "share", "teach", "give", "support", "transfer")):
            score += profile.reciprocity * debt * 1.5
            score += profile.altruism * trust * 0.5
            score += profile.group_loyalty * same_group * 1.0
        if any(k in a for k in ("repay",)):
            score += profile.reciprocity * debt * 2.0
        if any(k in a for k in ("trust_claim", "believe", "accept_claim")):
            score += trust * 1.0
            score -= profile.distrust_sensitivity * grievance * 1.5
        if any(k in a for k in ("accuse", "verify", "refuse", "blame")):
            score += profile.fairness * grievance * 1.0
            score += grievance * 1.0
        return score

    def episode_value(self, candidate: ActionCandidate, ctx: DecisionContext,
                      agent_id: str, mood: MoodState, profile: ProfileVector) -> float:
        """§11 foreground-episode continuation bonus."""
        em = ctx.episodes
        if em is None:
            return 0.0
        ep = _safe(lambda: em.foreground_of(agent_id), None)
        if ep is None:
            return 0.0
        etype = getattr(getattr(ep, "episode_type", None), "value", None)
        table = _EPISODE_BONUS.get(etype, {})
        a = candidate.action_type.lower()
        bonus = 0.0
        for key, val in table.items():
            if key in a:
                bonus = max(bonus, val)
        # abandoning a foreground episode is penalized unless frustration is high
        if any(k in a for k in ("abandon", "leave", "quit")):
            bonus = -0.5 + (0.7 if mood.frustration > 0.6 else 0.0)
        # civic enforcement bonus scales with fairness/conformity (§11.1)
        if etype == "civic_trial" and any(k in a for k in ("enforce", "support")):
            bonus += 0.4 * max(profile.fairness, profile.conformity)
        return bonus

    # ===== §13 survival guard ============================================ #
    def survival_guard(self, candidate: ActionCandidate, tf: Dict[str, float],
                       ctx: DecisionContext) -> float:
        """Soft urgency boosts + critical penalties (§13.2). Hard masks are
        handled separately in :meth:`rank`."""
        e, hp = ctx.energy, ctx.hp
        a = candidate.action_type.lower()
        boost = 0.0
        is_survival = any(tf.get(f, 0.0) > 0 for f in SURVIVAL_FEATURES)
        if e is not None and e < ENERGY_URGENT and any(
            k in a for k in ("eat", "gather", "return_home", "withdraw", "forage", "hunt")
        ):
            boost += 1.5
        if hp is not None and hp < HP_URGENT and any(
            k in a for k in ("rest", "seek_help", "avoid", "retreat", "heal")
        ):
            boost += 1.5
        # critical: penalize long non-survival actions while dying (§13.2)
        critical = (e is not None and e < ENERGY_CRITICAL) or (hp is not None and hp < HP_CRITICAL)
        if critical and not is_survival and tf.get("time_cost", 0.0) > 0.3:
            boost -= 3.0
        return boost

    # ===== scoring ======================================================= #
    def score(self, *, agent_id: str, candidate: ActionCandidate,
              raw_features: ActionFeatures, profile: ProfileVector, mood: MoodState,
              ctx: DecisionContext) -> ScoredCandidate:
        tf = transform_features(raw_features)
        survival = self.survival_score(tf, ctx.pressure)
        trait = self.trait_score(tf, profile)
        relation = self.relation_score(candidate, profile, ctx, agent_id)
        moods = self.mood_shift(mood, candidate)
        episode = self.episode_value(candidate, ctx, agent_id, mood, profile)
        cost = self.cost_score(tf)
        guard = self.survival_guard(candidate, tf, ctx)
        plan = float((ctx.plan_modifiers or {}).get(candidate.action_type, 0.0))
        utility = survival + trait + relation + moods + episode - cost + guard + plan
        contributions = {
            "survival": survival, "trait": trait, "relation": relation,
            "mood": moods, "episode": episode, "cost": -cost, "guard": guard,
            "plan": plan,
        }
        return ScoredCandidate(candidate=candidate, features=raw_features,
                               utility=utility, contributions=contributions)

    # ===== §1 steps 4-7: annotate + score + sort ========================= #
    def rank(self, *, agent_id: str, candidates: List[ActionCandidate],
             profile: ProfileVector, mood: Optional[MoodState] = None, state: Any = None,
             tctx: Any = None, ctx: Optional[DecisionContext] = None,
             ) -> List[ScoredCandidate]:
        """Annotate, transform, guard, score; return masked-out-removed list
        sorted by utility desc. ``tctx`` is accepted for back-compat (ignored)."""
        mood = mood or MoodState()
        ctx = ctx or DecisionContext(state=state)
        if state is not None and ctx.state is None:
            ctx.state = state
        scored: List[ScoredCandidate] = []
        for cand in candidates:
            if ctx.mask_fn is not None and not _safe(lambda: ctx.mask_fn(cand), True):
                continue  # §13.1 hard mask
            feats = self.feature_extractor.extract(agent_id=agent_id, candidate=cand, state=ctx.state)
            scored.append(self.score(agent_id=agent_id, candidate=cand,
                                     raw_features=feats, profile=profile, mood=mood, ctx=ctx))
        scored.sort(key=lambda s: s.utility, reverse=True)
        return scored

    # ===== §14 satisficing shortlist ===================================== #
    def shortlist(self, scored: List[ScoredCandidate], *,
                  top_k: int = DEFAULT_TOP_K, delta: float = DEFAULT_DELTA) -> List[ScoredCandidate]:
        if not scored:
            return []
        # top_k<=1 is the deterministic ablation arm (§15.1): pure argmax, no
        # delta band, so it never collapses into stochastic sampling.
        if top_k <= 1:
            return [scored[0]]
        max_u = scored[0].utility
        keep = {id(s) for s in scored[:top_k]}
        return [s for s in scored if id(s) in keep or s.utility >= max_u - delta]

    # ===== §15 temperature + seeded softmax ============================== #
    def temperature(self, profile: ProfileVector, mood: MoodState,
                    uncertainty: float = 0.0) -> float:
        tau = TAU_BASE * (1.0 + 0.5 * mood.stress + 0.3 * mood.fatigue
                          + 0.3 * uncertainty + 0.2 * mood.frustration
                          - 0.2 * mood.confidence - 0.2 * profile.conformity)
        return max(TAU_MIN, min(TAU_MAX, tau))

    def _rng_for(self, ctx: DecisionContext, agent_id: str) -> random.Random:
        key = f"{ctx.run_seed}|{agent_id}|{ctx.turn_id}|{ctx.decision_id}|action_policy"
        h = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return random.Random(int(h[:16], 16))

    def sample(self, scored: List[ScoredCandidate], *, profile: ProfileVector,
               mood: Optional[MoodState] = None, top_k: int = DEFAULT_TOP_K,
               delta: float = DEFAULT_DELTA, ctx: Optional[DecisionContext] = None,
               agent_id: str = "") -> Optional[ScoredCandidate]:
        """Satisficing shortlist (§14) + seeded softmax (§15). With top_k=1 this
        degrades to argmax (deterministic ablation arm)."""
        mood = mood or MoodState()
        pool = self.shortlist(scored, top_k=top_k, delta=delta)
        if not pool:
            return None
        if len(pool) == 1:
            return pool[0]
        tau = self.temperature(profile, mood)
        mx = max(s.utility for s in pool)
        exps = [math.exp((s.utility - mx) / tau) for s in pool]
        total = sum(exps) or 1.0
        probs = [e / total for e in exps]
        rng = self._rng_for(ctx, agent_id) if ctx is not None else self.rng
        r = rng.random()
        acc = 0.0
        for s, p in zip(pool, probs):
            acc += p
            if r <= acc:
                return s
        return pool[-1]

    # ===== §1 full pipeline + §16 trace ================================== #
    def select(self, *, agent_id: str, candidates: List[ActionCandidate],
               profile: ProfileVector, mood: Optional[MoodState] = None,
               ctx: Optional[DecisionContext] = None,
               top_k: int = DEFAULT_TOP_K, delta: float = DEFAULT_DELTA) -> PolicyTrace:
        ctx = ctx or DecisionContext()
        mood = mood or MoodState()
        trace = PolicyTrace(agent_id=agent_id, turn_id=ctx.turn_id,
                            candidate_pool=[c.action_type for c in candidates])
        # mask + annotate + score
        scored: List[ScoredCandidate] = []
        for cand in candidates:
            if ctx.mask_fn is not None and not _safe(lambda: ctx.mask_fn(cand), True):
                trace.masked_actions.append(cand.action_type)
                continue
            feats = self.feature_extractor.extract(agent_id=agent_id, candidate=cand, state=ctx.state)
            trace.feature_annotations[cand.action_type] = {
                k: v for k, v in feats.to_dict().items() if v
            }
            scored.append(self.score(agent_id=agent_id, candidate=cand,
                                     raw_features=feats, profile=profile, mood=mood, ctx=ctx))
        scored.sort(key=lambda s: s.utility, reverse=True)
        trace.scored = scored
        if not scored:
            return trace

        pool = self.shortlist(scored, top_k=top_k, delta=delta)
        trace.shortlist = [s.candidate.action_type for s in pool]
        tau = self.temperature(profile, mood)
        trace.sampling_temperature = tau

        # probabilities over the shortlist (§15)
        if len(pool) == 1:
            trace.probabilities = {pool[0].candidate.action_type: 1.0}
            chosen, draw = pool[0], 0.0
        else:
            mx = max(s.utility for s in pool)
            exps = [math.exp((s.utility - mx) / tau) for s in pool]
            total = sum(exps) or 1.0
            probs = [e / total for e in exps]
            trace.probabilities = {s.candidate.action_type: p for s, p in zip(pool, probs)}
            rng = self._rng_for(ctx, agent_id)
            draw = rng.random()
            chosen = pool[-1]
            acc = 0.0
            for s, p in zip(pool, probs):
                acc += p
                if draw <= acc:
                    chosen = s
                    break

        trace.selected_action = chosen.candidate.action_type
        trace.selected_parameters = dict(chosen.candidate.parameters)
        trace.action_probability = trace.probabilities.get(chosen.candidate.action_type, 1.0)
        trace.random_draw = draw
        trace.explanation_paths = _explain(chosen)
        return trace


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _safe(fn: Callable[[], Any], default: Any) -> Any:
    try:
        return fn()
    except Exception:
        return default


def _explain(chosen: ScoredCandidate, top_n: int = 4) -> List[str]:
    """Build §16 explanation paths from the component + top-feature contributions."""
    paths: List[str] = []
    comp = sorted(chosen.contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)
    for name, val in comp:
        if abs(val) < 1e-6:
            continue
        sign = "+" if val >= 0 else "-"
        paths.append(f"{name} {sign}{abs(val):.2f}")
        if len(paths) >= top_n:
            break
    return paths


# Back-compat alias: the controller + tests refer to ProfileToPolicy. PCBSP is
# the v1 model; the old centered-linear scorer is superseded by it.
ProfileToPolicy = PCBSPPolicy
