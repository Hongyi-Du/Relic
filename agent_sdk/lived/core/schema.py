"""Persona Graph schema — the FROZEN, machine-readable persona vocabulary.

This module is **Priority 1 + Priority 2** of the "Persona-Conditioned Policy
and Episode Dynamics" line: it pins down, as data (not prose), the typed
heterogeneous persona graph and the action-feature schema so the rest of the
system (scorer, updater, episode manager, explainability trace) reads one
authoritative source instead of scattering magic strings.

What lives here:
  * :class:`NodeType` / :class:`EdgeType` — the persona graph's typed node/edge
    ontology (§2.1 / §3.1).
  * :class:`TraitSpec` + :data:`TRAIT_SPECS` — the 12 frozen stable traits, each
    with base_value / stability / plasticity / linked_action_features /
    positive_triggers / negative_triggers (§2.2, §8.1, §8.6). This table is what
    makes the experience-driven update rules *data-driven* and auditable.
  * :class:`TraitNode` — the per-agent *runtime* trait object (base_value,
    current_value, plasticity, confidence, update_history). Wraps a single
    trait scalar with provenance so a profile change is explainable (§8.1).
  * :class:`StateSpec` + :data:`STATE_SPECS` — the transient-state nodes
    (baseline + decay) so per-turn mood relaxation is data-driven too (§2.3).
  * :class:`FeatureSpec` + :data:`FEATURE_SPECS` — Priority 2: definition,
    value range, and *who fills it* (deterministic env vs. optional LLM
    annotation) for every :class:`~agent_sdk.lived.core.contracts.ActionFeatures`
    field (§4.3).

Everything is environment-agnostic. The W matrix that maps traits→feature
weights lives next door in ``agent_sdk.lived.core.wmatrix``; this file only declares
*what exists*, not *how strongly trait couples to feature*.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Tuple

from agent_sdk.lived.core.contracts import ActionFeatures, MoodState, ProfileVector


# --------------------------------------------------------------------------- #
# §2.1 Persona graph node types  /  §3.1 edge types
# --------------------------------------------------------------------------- #
class NodeType(str, Enum):
    """Typed nodes of the heterogeneous persona graph (§2.1)."""
    STABLE_TRAIT = "stable_trait"            # long-term tendency (ProfileVector)
    TRANSIENT_STATE = "transient_state"      # short-term mood / physiology (MoodState)
    MOTIVATION = "motivation"                # current drive pressure
    SKILL = "skill"                          # skill / material / terrain knowledge
    KNOWLEDGE = "knowledge"
    MEMORY = "memory"                        # a concrete experience (event-node id)
    IDENTITY = "identity"                    # home / terrain / known path / group
    SOCIAL_RELATION = "social_relation"      # trust / debt / grievance / reputation
    ACTION_FEATURE = "action_feature"        # a policy behaviour-feature dimension
    EPISODE = "episode"                      # a multi-turn activity object
    AFFORDANCE_ITEM = "affordance_item"      # known item / attemptable skill / prototype
    INSTITUTION_BELIEF = "institution_belief"  # trust / resistance toward rules & records


class EdgeType(str, Enum):
    """Typed edges of the persona graph (§3.1). The graph's value is in the
    edges, not the fields: these carry the explainable provenance."""
    SUPPORTS = "supports"                    # positively supports a tendency / feature
    SUPPRESSES = "suppresses"                # suppresses a behaviour tendency
    CONFLICTS_WITH = "conflicts_with"        # two tendencies conflict
    GROUNDS = "grounds"                      # knowledge/action has real grounding
    ENABLES = "enables"                      # skill/item/mechanism makes action attemptable
    UPDATES = "updates"                      # event updates state/relation/skill/trait
    MODULATES = "modulates"                  # one state modulates another's strength
    CAUSED_BY = "caused_by"                  # causal source of an event
    OBSERVED_IN = "observed_in"              # whether a memory/action was observed
    ASSOCIATED_WITH = "associated_with"      # weak association


# --------------------------------------------------------------------------- #
# §2.2 / §8.6 Stable trait specifications (FROZEN registry)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TraitSpec:
    """Static spec for one of the 12 stable traits.

    * ``base_value``  — population prior / reproduction default (§2.2).
    * ``stability``   — [0,1]; high = resists change. Used as ``(1-stability)``
      multiplier so a "rock-solid" trait barely moves (§2.2 "不能频繁大幅变化").
    * ``plasticity``  — the per-event nudge magnitude at a **major** event
      (§8.6). Intensity tiers scale this (see ``INTENSITY_SCALE``). Default 0.02
      matches the design's "major event ≈ 0.02" budget (§5.3).
    * ``linked_action_features`` — which :class:`ActionFeatures` fields this
      trait couples to (mirror of the W matrix's non-zero columns; kept here for
      the explainability trace, §8.10).
    * ``positive_triggers`` / ``negative_triggers`` — event *kinds* (see
      :data:`EVENT_KINDS`) that push the trait up / down. This is what makes the
      stable-trait update rules data-driven (Priority 4).
    """
    name: str
    base_value: float = 0.5
    stability: float = 0.85
    plasticity: float = 0.02
    linked_action_features: Tuple[str, ...] = ()
    positive_triggers: Tuple[str, ...] = ()
    negative_triggers: Tuple[str, ...] = ()


# The 12 traits, frozen. Names MUST match ProfileVector fields exactly.
TRAIT_SPECS: Dict[str, TraitSpec] = {
    # linked_action_features == the trait's non-zero columns in wmatrix.W_MATRIX
    # (§8.1); validate_matrix() enforces this mirror so the two never drift.
    "risk_aversion": TraitSpec(
        name="risk_aversion", stability=0.85, plasticity=0.02,
        linked_action_features=("future_security_gain", "storage_value", "rule_compliance",
                                "risk_cost", "uncertainty_cost", "conflict_risk",
                                "reputation_risk", "experiment_value"),
        positive_triggers=("betrayed_by", "winter_starvation", "risk_realized_harm"),
        negative_triggers=("risk_paid_off",),
    ),
    "altruism": TraitSpec(
        name="altruism", stability=0.88, plasticity=0.015,
        linked_action_features=("altruistic_gain", "trust_gain", "reputation_gain",
                                "public_good_gain", "group_loyalty_gain", "private_gain",
                                "opportunistic_gain", "public_harm"),
        positive_triggers=("successful_teaching", "help_reciprocated"),
        negative_triggers=("altruism_exploited",),
    ),
    "fairness": TraitSpec(
        name="fairness", stability=0.9, plasticity=0.015,
        linked_action_features=("fairness_gain", "violation_detection", "rule_compliance",
                                "institution_gain", "public_good_gain", "fairness_cost",
                                "opportunistic_gain", "public_harm"),
        positive_triggers=("fairness_violation_observed", "unfair_treatment_received"),
        negative_triggers=(),
    ),
    "dominance": TraitSpec(
        name="dominance", stability=0.85, plasticity=0.02,
        linked_action_features=("dominance_gain", "institution_gain", "coordination_gain",
                                "reputation_risk"),
        positive_triggers=("successful_leadership",),
        negative_triggers=("leadership_rejected",),
    ),
    "curiosity": TraitSpec(
        name="curiosity", stability=0.88, plasticity=0.015,
        linked_action_features=("information_gain", "experiment_value", "novelty_value",
                                "skill_gain", "future_security_gain", "time_cost",
                                "uncertainty_cost"),
        positive_triggers=("prototype_success", "exploration_positive"),
        negative_triggers=("repeated_experiment_failure",),
    ),
    "group_loyalty": TraitSpec(
        name="group_loyalty", stability=0.88, plasticity=0.015,
        linked_action_features=("group_loyalty_gain", "public_good_gain", "trust_gain",
                                "public_harm", "opportunistic_gain"),
        positive_triggers=("group_defended_me", "group_success"),
        negative_triggers=("group_abandoned_me",),
    ),
    "reciprocity": TraitSpec(
        name="reciprocity", stability=0.85, plasticity=0.015,
        linked_action_features=("reciprocity_gain", "trust_gain", "reputation_gain",
                                "trust_cost", "public_harm"),
        positive_triggers=("helped_by", "debt_repaid_to_me"),
        negative_triggers=("help_unreturned",),
    ),
    "opportunism": TraitSpec(
        name="opportunism", stability=0.85, plasticity=0.02,
        linked_action_features=("private_gain", "opportunistic_gain", "autonomy_gain",
                                "dominance_gain", "fairness_gain", "rule_compliance",
                                "altruistic_gain", "reputation_risk"),
        positive_triggers=("opportunism_paid_off",),
        negative_triggers=("opportunism_punished",),
    ),
    "conformity": TraitSpec(
        name="conformity", stability=0.85, plasticity=0.02,
        linked_action_features=("rule_compliance", "institution_gain", "reputation_gain",
                                "public_good_gain", "conformity_cost", "violation_gain",
                                "opportunistic_gain"),
        positive_triggers=("rule_compliance_rewarded",),
        negative_triggers=("unjust_punishment",),
    ),
    "distrust_sensitivity": TraitSpec(
        name="distrust_sensitivity", stability=0.8, plasticity=0.03,  # the most plastic (§8.6)
        linked_action_features=("verification_value", "information_gain", "autonomy_gain",
                                "trust_gain"),
        positive_triggers=("betrayed_by", "promise_broken_against", "false_accusation_against"),
        negative_triggers=("trust_restored",),
    ),
    "reputation_concern": TraitSpec(
        name="reputation_concern", stability=0.87, plasticity=0.015,
        linked_action_features=("reputation_gain", "rule_compliance", "altruistic_gain",
                                "reputation_risk", "trust_cost", "violation_gain"),
        positive_triggers=("public_praise", "public_blame"),
        negative_triggers=(),
    ),
    "long_termism": TraitSpec(
        name="long_termism", stability=0.88, plasticity=0.02,
        linked_action_features=("future_security_gain", "storage_value", "skill_gain",
                                "institution_gain", "public_good_gain", "time_cost"),
        positive_triggers=("winter_starvation", "storage_success"),
        negative_triggers=(),
    ),
}

# Sanity: every ProfileVector trait must have a spec and vice-versa.
assert set(TRAIT_SPECS) == set(ProfileVector.trait_names()), (
    "TRAIT_SPECS drifted from ProfileVector fields: "
    f"{set(TRAIT_SPECS) ^ set(ProfileVector.trait_names())}"
)


# --------------------------------------------------------------------------- #
# §5.3 Event-intensity tiers — gate whether a stable trait may move at all
# --------------------------------------------------------------------------- #
class Intensity(str, Enum):
    MINOR = "minor"
    MODERATE = "moderate"
    MAJOR = "major"
    REPEATED = "repeated"


# Multiplier applied to a trait's ``plasticity`` per intensity tier (§5.3):
# minor → no stable-trait update at all; moderate ≈ 0.005; major ≈ 0.02;
# repeated pattern ≈ 0.03 (capped). With default plasticity 0.02 these line up.
INTENSITY_SCALE: Dict[Intensity, float] = {
    Intensity.MINOR: 0.0,
    Intensity.MODERATE: 0.25,
    Intensity.MAJOR: 1.0,
    Intensity.REPEATED: 1.5,
}

# The closed set of event *kinds* the appraiser may emit and the updater +
# trait triggers understand. Keeping it closed makes the update path auditable.
EVENT_KINDS: Tuple[str, ...] = (
    "helped_by", "betrayed_by", "winter_starvation", "successful_leadership",
    "successful_teaching", "unjust_punishment", "risk_realized_harm",
    "risk_paid_off", "help_reciprocated", "altruism_exploited",
    "fairness_violation_observed", "unfair_treatment_received",
    "leadership_rejected", "prototype_success", "exploration_positive",
    "repeated_experiment_failure", "group_defended_me", "group_success",
    "group_abandoned_me", "debt_repaid_to_me", "help_unreturned",
    "opportunism_paid_off", "opportunism_punished", "rule_compliance_rewarded",
    "promise_broken_against", "false_accusation_against", "trust_restored",
    "public_praise", "public_blame", "storage_success",
)


# --------------------------------------------------------------------------- #
# §8.1 Per-agent runtime trait node (with provenance / update history)
# --------------------------------------------------------------------------- #
@dataclass
class TraitNode:
    """A single stable trait as a live persona-graph node (§8.1).

    The flat :class:`ProfileVector` is the scorer's fast input; ``TraitNode`` is
    the explainable, slowly-updated source of one of its scalars. ``confidence``
    rises as evidence accumulates (more updates → more certain the value is
    real). ``update_history`` keeps the audit trail (turn, delta, cause) so any
    profile drift can be traced to events (the [§8.10] explainability claim).
    """
    name: str
    base_value: float = 0.5
    current_value: float = 0.5
    stability: float = 0.85
    plasticity: float = 0.02
    confidence: float = 0.5
    linked_action_features: Tuple[str, ...] = ()
    positive_triggers: Tuple[str, ...] = ()
    negative_triggers: Tuple[str, ...] = ()
    update_history: List[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_spec(cls, spec: TraitSpec, value: float | None = None) -> "TraitNode":
        v = spec.base_value if value is None else float(value)
        return cls(
            name=spec.name, base_value=spec.base_value, current_value=v,
            stability=spec.stability, plasticity=spec.plasticity,
            linked_action_features=spec.linked_action_features,
            positive_triggers=spec.positive_triggers,
            negative_triggers=spec.negative_triggers,
        )

    def nudge(self, delta: float, *, turn: int = -1, cause: str = "") -> float:
        """Apply a (already-scaled) realized ``delta``, clamp to [0,1], log
        provenance, and bump confidence. Returns the clamped realized delta.

        ``delta`` is the *final* magnitude (the caller scales ``plasticity`` by
        the intensity tier). ``stability`` is metadata describing how resistant
        the trait is — it informed the plasticity choice and is reserved for an
        optional reversion-to-base; it is NOT re-applied here, to avoid double
        damping (§5.3 magnitudes line up directly with plasticity)."""
        new_val = max(0.0, min(1.0, self.current_value + delta))
        realized = new_val - self.current_value
        self.current_value = new_val
        # Confidence asymptotically approaches 1 as evidence accrues.
        self.confidence = min(1.0, self.confidence + 0.05)
        self.update_history.append(
            {"turn": turn, "delta": round(realized, 5), "cause": cause}
        )
        return realized


def default_trait_nodes() -> Dict[str, TraitNode]:
    """Build a fresh per-agent trait-node map from the frozen specs."""
    return {name: TraitNode.from_spec(spec) for name, spec in TRAIT_SPECS.items()}


# --------------------------------------------------------------------------- #
# §2.3 Transient-state specs (baseline + decay) — data-driven mood relaxation
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class StateSpec:
    """Spec for one transient (mood/physiology) state.

    ``baseline`` is where the state relaxes to each turn; ``decay`` is the
    per-turn fraction of the gap closed. ``env_driven`` states are expected to
    be overwritten by the FeatureExtractor from world state — their decay is
    only a fallback (§2.3 note).
    """
    name: str
    baseline: float = 0.0
    decay: float = 0.1
    env_driven: bool = False


STATE_SPECS: Dict[str, StateSpec] = {
    "valence": StateSpec("valence", baseline=0.5, decay=0.1),
    "stress": StateSpec("stress", baseline=0.0, decay=0.1),
    "fatigue": StateSpec("fatigue", baseline=0.0, decay=0.1),
    "anger": StateSpec("anger", baseline=0.0, decay=0.12),
    "gratitude": StateSpec("gratitude", baseline=0.0, decay=0.1),
    "confidence": StateSpec("confidence", baseline=0.5, decay=0.08),
    "fear": StateSpec("fear", baseline=0.0, decay=0.12),
    "hunger_pressure": StateSpec("hunger_pressure", baseline=0.0, decay=0.05, env_driven=True),
    "frustration": StateSpec("frustration", baseline=0.0, decay=0.12),
    "fairness_salience": StateSpec("fairness_salience", baseline=0.0, decay=0.15),
    "resource_insecurity": StateSpec("resource_insecurity", baseline=0.0, decay=0.05, env_driven=True),
}

assert set(STATE_SPECS) == set(MoodState.state_names()), (
    "STATE_SPECS drifted from MoodState fields: "
    f"{set(STATE_SPECS) ^ set(MoodState.state_names())}"
)


# --------------------------------------------------------------------------- #
# §4.3 Action-feature specs (Priority 2: definition + range + who fills it)
# --------------------------------------------------------------------------- #
class FilledBy(str, Enum):
    """Who is allowed to populate a feature (cost-control rule, §4.2 / §5)."""
    ENV = "env"        # deterministic function of world state (cheap, default)
    LLM = "llm"        # semantic judgement (target fairness/altruism) — sparing
    MIXED = "mixed"    # env baseline, LLM may refine


class TransformKind(str, Enum):
    """Which §6 nonlinear transform a feature's intensity goes through before it
    enters the utility. ``transforms.py`` owns the math."""
    BOUNDED_LINEAR = "bounded_linear"      # §6.1 identity-ish, clamped to [0,1]
    DIMINISHING_GAIN = "diminishing_gain"  # §6.2 log marginal-decreasing gain
    LOSS_AVERSION = "loss_aversion"        # §6.3 amplified, λ-weighted loss
    SURVIVAL = "survival"                  # §6.4 feeds SurvivalScore + guard


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    definition: str
    transform: TransformKind = TransformKind.BOUNDED_LINEAR
    lo: float = 0.0
    hi: float = 1.0
    is_cost: bool = False            # stored positive; policy applies the sign
    filled_by: FilledBy = FilledBy.ENV


# Discrete intensities the annotator may emit (§5).
FEATURE_INTENSITIES: Tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)

_BL = TransformKind.BOUNDED_LINEAR
_DG = TransformKind.DIMINISHING_GAIN
_LA = TransformKind.LOSS_AVERSION
_SV = TransformKind.SURVIVAL

# Definitions + ranges + transform for every ActionFeatures field (PCBSP §4/§6).
FEATURE_SPECS: Dict[str, FeatureSpec] = {
    # §4.1 survival / resource (diminishing gains; the three pure-survival ones
    # feed SurvivalScore via the SURVIVAL transform).
    "survival_gain": FeatureSpec("survival_gain", "Immediate survival value (food/HP now).", _SV, filled_by=FilledBy.ENV),
    "energy_gain": FeatureSpec("energy_gain", "Net energy restored (eat/rest).", _SV, filled_by=FilledBy.ENV),
    "resource_gain": FeatureSpec("resource_gain", "Food/material/tool acquired.", _SV, filled_by=FilledBy.ENV),
    "future_security_gain": FeatureSpec("future_security_gain", "Improves future survival safety (winter buffer).", _DG, filled_by=FilledBy.ENV),
    "storage_value": FeatureSpec("storage_value", "Improves storage capacity/stability.", _DG, filled_by=FilledBy.ENV),
    # §4.2 cost / risk (loss aversion)
    "energy_cost": FeatureSpec("energy_cost", "Physical energy spent.", _LA, is_cost=True, filled_by=FilledBy.ENV),
    "time_cost": FeatureSpec("time_cost", "Action time / opportunity cost.", _LA, is_cost=True, filled_by=FilledBy.ENV),
    "material_cost": FeatureSpec("material_cost", "Materials consumed.", _LA, is_cost=True, filled_by=FilledBy.ENV),
    "risk_cost": FeatureSpec("risk_cost", "Injury/loss/failure risk.", _LA, is_cost=True, filled_by=FilledBy.ENV),
    "uncertainty_cost": FeatureSpec("uncertainty_cost", "Outcome uncertainty.", _LA, is_cost=True, filled_by=FilledBy.MIXED),
    "conflict_risk": FeatureSpec("conflict_risk", "Risk of provoking conflict.", _LA, is_cost=True, filled_by=FilledBy.MIXED),
    "reputation_risk": FeatureSpec("reputation_risk", "Reputation damage if observed.", _LA, is_cost=True, filled_by=FilledBy.MIXED),
    # §4.3 information / invention (bounded linear)
    "information_gain": FeatureSpec("information_gain", "New information acquired.", _BL, filled_by=FilledBy.ENV),
    "experiment_value": FeatureSpec("experiment_value", "Experiment / prototype / probe value.", _BL, filled_by=FilledBy.ENV),
    "novelty_value": FeatureSpec("novelty_value", "Value of trying a new method.", _BL, filled_by=FilledBy.ENV),
    "skill_gain": FeatureSpec("skill_gain", "Learn / improve a skill.", _BL, filled_by=FilledBy.ENV),
    # §4.4 social (bounded linear)
    "altruistic_gain": FeatureSpec("altruistic_gain", "Degree of helping others.", _BL, filled_by=FilledBy.MIXED),
    "reciprocity_gain": FeatureSpec("reciprocity_gain", "Repaying help / upholding reciprocity.", _BL, filled_by=FilledBy.MIXED),
    "trust_gain": FeatureSpec("trust_gain", "Strengthens a trust relation.", _BL, filled_by=FilledBy.MIXED),
    "reputation_gain": FeatureSpec("reputation_gain", "Raises reputation.", _BL, filled_by=FilledBy.MIXED),
    "group_loyalty_gain": FeatureSpec("group_loyalty_gain", "Helps home group / familiars.", _BL, filled_by=FilledBy.ENV),
    # §4.5 civic / institutional (bounded linear)
    "fairness_gain": FeatureSpec("fairness_gain", "Reduces unfairness / upholds fairness.", _BL, filled_by=FilledBy.MIXED),
    "violation_detection": FeatureSpec("violation_detection", "Detects / exposes a violation.", _BL, filled_by=FilledBy.MIXED),
    "rule_compliance": FeatureSpec("rule_compliance", "Complies with an adopted trial rule.", _BL, filled_by=FilledBy.ENV),
    "institution_gain": FeatureSpec("institution_gain", "Maintains / tests / strengthens a mechanism.", _BL, filled_by=FilledBy.ENV),
    "public_good_gain": FeatureSpec("public_good_gain", "Improves shared resource / infrastructure.", _BL, filled_by=FilledBy.ENV),
    # §4.6 private strategy (bounded linear)
    "private_gain": FeatureSpec("private_gain", "Private individual payoff.", _BL, filled_by=FilledBy.ENV),
    "opportunistic_gain": FeatureSpec("opportunistic_gain", "Exploiting unobserved / asymmetric opening.", _BL, filled_by=FilledBy.MIXED),
    "dominance_gain": FeatureSpec("dominance_gain", "Gains control / command / influence.", _BL, filled_by=FilledBy.MIXED),
    "autonomy_gain": FeatureSpec("autonomy_gain", "Reduces dependence on others.", _BL, filled_by=FilledBy.ENV),
    "coordination_gain": FeatureSpec("coordination_gain", "Coordinates the group / public influence (camp_announce).", _BL, filled_by=FilledBy.ENV),
    # §4.7 negative civic / social (loss aversion, stored positive)
    "fairness_cost": FeatureSpec("fairness_cost", "Creates unfairness.", _LA, is_cost=True, filled_by=FilledBy.MIXED),
    "conformity_cost": FeatureSpec("conformity_cost", "Violates an existing rule/norm/trial.", _LA, is_cost=True, filled_by=FilledBy.ENV),
    "trust_cost": FeatureSpec("trust_cost", "Damages trust.", _LA, is_cost=True, filled_by=FilledBy.MIXED),
    "public_harm": FeatureSpec("public_harm", "Harms public resource / others' safety.", _LA, is_cost=True, filled_by=FilledBy.MIXED),
    # §8 extras
    "verification_value": FeatureSpec("verification_value", "Verifying a claim/record before trusting.", _BL, filled_by=FilledBy.ENV),
    "violation_gain": FeatureSpec("violation_gain", "Perceived selfish payoff of breaking a rule.", _BL, filled_by=FilledBy.ENV),
}

assert set(FEATURE_SPECS) == set(ActionFeatures.feature_names()), (
    "FEATURE_SPECS drifted from ActionFeatures fields: "
    f"{set(FEATURE_SPECS) ^ set(ActionFeatures.feature_names())}"
)

COST_FEATURES: Tuple[str, ...] = tuple(
    n for n, s in FEATURE_SPECS.items() if s.is_cost
)
# The three pure-survival gains that feed SurvivalScore (§6.4 / §13.2).
SURVIVAL_FEATURES: Tuple[str, ...] = tuple(
    n for n, s in FEATURE_SPECS.items() if s.transform == TransformKind.SURVIVAL
)
# CostScore baseline set (§12.2): the seven physical/risk costs only. The four
# negative-civic features (fairness_cost/conformity_cost/trust_cost/public_harm)
# are handled purely via the W matrix to avoid double-counting.
COSTSCORE_FEATURES: Tuple[str, ...] = (
    "energy_cost", "time_cost", "material_cost", "risk_cost",
    "uncertainty_cost", "conflict_risk", "reputation_risk",
)
