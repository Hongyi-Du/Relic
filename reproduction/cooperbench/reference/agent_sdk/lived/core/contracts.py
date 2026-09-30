"""Lived-Agent decision system — frozen schema DTOs (design doc Phase 1).

These dataclasses are the **only** vocabulary the rest of ``agent_sdk.lived``
speaks in. They mirror the design doc "Graph-Grounded Lived Agents for
Emergent Primitive Societies":

  * ``ProfileVector``  — §3.2 long-term tendency variables
  * ``MoodState``      — §3.2 short-term state variables
  * ``ActionFeatures`` — §14.3 action feature vector
  * ``ProfileWeights`` — §14.4 profile weight vector
  * ``ActionCandidate``/``ScoredCandidate`` — §14.2 / §14.7
  * ``Need``           — §11.2 need schema (wish parser output)
  * ``AffordanceProposal`` — §12 synthesizer/verifier output
  * ``SpeechAct``      — §15 speech-as-action

Everything here is environment-agnostic: no world/grid/energy references.
Env-specific extraction (e.g. how ``survival_gain`` is computed from a tile)
lives behind the ports in ``agent_sdk.lived.core.ports``.

NOTE (scaffold): fields are intentionally *complete* (the schema is frozen
here so downstream modules can be filled in without churn) but the logic that
populates them lives in stubs elsewhere. Keep these dataclasses stable.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


# --------------------------------------------------------------------------- #
# §3.2 Profile — long-term tendency variables
# --------------------------------------------------------------------------- #
@dataclass
class ProfileVector:
    """Long-term persona tendencies. Range convention: [0.0, 1.0].

    These are the *lived state* center of an agent's behaviour. They change
    slowly via :class:`agent_sdk.lived.persona.profile.ProfileUpdater` (experience
    driven), never within a single turn.
    """
    risk_aversion: float = 0.5
    altruism: float = 0.5
    fairness: float = 0.5
    dominance: float = 0.5
    curiosity: float = 0.5
    group_loyalty: float = 0.5
    reciprocity: float = 0.5
    opportunism: float = 0.5
    conformity: float = 0.5
    distrust_sensitivity: float = 0.5
    reputation_concern: float = 0.5
    long_termism: float = 0.5

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ProfileVector":
        known = {f.name for f in fields(cls)}
        return cls(**{k: float(v) for k, v in (d or {}).items() if k in known})

    @classmethod
    def trait_names(cls) -> Tuple[str, ...]:
        return tuple(f.name for f in fields(cls))


# --------------------------------------------------------------------------- #
# §3.2 Mood — short-term state variables (bounded perturbation, never override)
# --------------------------------------------------------------------------- #
@dataclass
class MoodState:
    """Short-term / transient state. Range convention: [0.0, 1.0].

    These are the "Transient State" nodes of the persona graph (Persona-
    Conditioned Policy §2.3). They change fast (every turn) and modulate — but
    never override — the long-term :class:`ProfileVector` (§14.5).

    ``valence`` is the only signed-feeling axis (0=very negative, 1=very
    positive); the rest are intensity scalars in [0,1].

    Two of these (``hunger_pressure``, ``resource_insecurity``) are physiology
    /environment-driven: the env FeatureExtractor is expected to OVERWRITE them
    each turn from world state; their per-turn decay is only a fallback when the
    env does not refresh them.
    """
    valence: float = 0.5
    stress: float = 0.0
    fatigue: float = 0.0
    anger: float = 0.0
    gratitude: float = 0.0
    confidence: float = 0.5
    fear: float = 0.0
    # --- added for the full §2.3 transient-state set ---------------------- #
    hunger_pressure: float = 0.0      # ↑ eat / gather / withdraw food (env-driven)
    frustration: float = 0.0          # ↑ abandon / ask_help / retry / wish
    fairness_salience: float = 0.0    # transiently ↑ fairness-related action utility
    resource_insecurity: float = 0.0  # ↑ storage / private security / gathering (env-driven)

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MoodState":
        known = {f.name for f in fields(cls)}
        return cls(**{k: float(v) for k, v in (d or {}).items() if k in known})

    @classmethod
    def state_names(cls) -> Tuple[str, ...]:
        return tuple(f.name for f in fields(cls))


# --------------------------------------------------------------------------- #
# §2.4 Identity / grounding fields — spatial + group anchoring for each agent
# --------------------------------------------------------------------------- #
@dataclass
class IdentityGrounding:
    """Spatial / group grounding for one agent (Persona-Conditioned Policy
    §2.4). These are the ``identity``/``grounding`` nodes of the persona graph:
    they tie a profile to a *place*, a *group* and a *terrain skill identity*,
    which is the basis of the 16-agents / 8-groups initialization (DESIGN §6A).

    All fields are env-meaningful but env-agnostic in type (ids / coords as
    opaque values), so ``agent_sdk`` never needs to import the world. The env
    fills these at spawn via the wiring layer.
    """
    home_location: Optional[Any] = None            # (x, y) or tile id
    home_storage_id: Optional[str] = None          # private cache id
    campfire_location: Optional[Any] = None
    home_group_id: Optional[str] = None            # 1 of the 8 groups
    group_partner: Optional[str] = None            # paired agent uid
    nearby_terrain: List[str] = field(default_factory=list)         # biome ids around home
    known_paths: List[str] = field(default_factory=list)            # path ids agent can navigate
    familiar_resource_zones: List[str] = field(default_factory=list)
    terrain_skill_identity: str = ""               # e.g. "grain", "reed", "river", "forest"
    initial_private_inventory: Dict[str, int] = field(default_factory=dict)
    known_materials: List[str] = field(default_factory=list)
    known_resource_sites: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "IdentityGrounding":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (d or {}).items() if k in known})


# --------------------------------------------------------------------------- #
# §14.3 Action feature vector
# --------------------------------------------------------------------------- #
@dataclass
class ActionFeatures:
    """Structured feature vector for one candidate action in one state.

    This is the FROZEN action-feature schema of the **Persona-Conditioned
    Bounded Softmax Policy (PCBSP) §4**. The scorer maps it through fixed
    nonlinear transforms (`transforms.py`, §6) and a fixed trait-feature matrix
    (`wmatrix.py`, §8). The env / LLM annotator fills it but may ONLY use these
    fields and only the discrete intensities {0, .25, .5, .75, 1.0} (§5).

    See :data:`agent_sdk.lived.core.schema.FEATURE_SPECS` for each field's
    definition, range, transform type, and who fills it.

    Range convention: every field is a non-negative [0,1] *magnitude*; costs /
    risks / harms are stored POSITIVE and the policy applies the negative sign
    (via the loss-aversion transform in CostScore + negative W cells).
    """
    # --- §4.1 survival / resource ---------------------------------------- #
    survival_gain: float = 0.0
    energy_gain: float = 0.0
    resource_gain: float = 0.0
    future_security_gain: float = 0.0
    storage_value: float = 0.0
    # --- §4.2 cost / risk (stored positive) ------------------------------ #
    energy_cost: float = 0.0
    time_cost: float = 0.0
    material_cost: float = 0.0
    risk_cost: float = 0.0
    uncertainty_cost: float = 0.0
    conflict_risk: float = 0.0
    reputation_risk: float = 0.0
    # --- §4.3 information / invention ------------------------------------ #
    information_gain: float = 0.0
    experiment_value: float = 0.0
    novelty_value: float = 0.0
    skill_gain: float = 0.0
    # --- §4.4 social ----------------------------------------------------- #
    altruistic_gain: float = 0.0
    reciprocity_gain: float = 0.0
    trust_gain: float = 0.0
    reputation_gain: float = 0.0
    group_loyalty_gain: float = 0.0
    # --- §4.5 civic / institutional -------------------------------------- #
    fairness_gain: float = 0.0
    violation_detection: float = 0.0
    rule_compliance: float = 0.0
    institution_gain: float = 0.0
    public_good_gain: float = 0.0
    # --- §4.6 private strategy ------------------------------------------- #
    private_gain: float = 0.0
    opportunistic_gain: float = 0.0
    dominance_gain: float = 0.0
    autonomy_gain: float = 0.0
    coordination_gain: float = 0.0      # §8 dominance: coordination / public influence
    # --- §4.7 negative civic / social (stored positive) ------------------ #
    fairness_cost: float = 0.0
    conformity_cost: float = 0.0
    trust_cost: float = 0.0
    public_harm: float = 0.0
    # --- §8-referenced extras -------------------------------------------- #
    verification_value: float = 0.0     # §8 distrust_sensitivity
    violation_gain: float = 0.0         # §8 conformity / reputation_concern (selfish rule-break payoff)

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)

    @classmethod
    def feature_names(cls) -> Tuple[str, ...]:
        return tuple(f.name for f in fields(cls))


# --------------------------------------------------------------------------- #
# §14.4 Profile weight vector — maps ProfileVector -> per-feature weights
# --------------------------------------------------------------------------- #
@dataclass
class ProfileWeights:
    """Per-feature weights used by the linear utility scorer (§14.4).

    Produced from a :class:`ProfileVector` by
    ``agent_sdk.lived.persona.policy.profile_to_weights``. Stored as a plain mapping so
    the scorer stays a simple dot product and remains interpretable /
    counterfactual-editable (§22.3 mitigation).
    """
    weights: Dict[str, float] = field(default_factory=dict)

    def get(self, feature_name: str, default: float = 0.0) -> float:
        return self.weights.get(feature_name, default)


# --------------------------------------------------------------------------- #
# §14.2 / §14.7 Action candidate + scored candidate
# --------------------------------------------------------------------------- #
class CandidateSource(str, Enum):
    """Where a candidate action came from (§14.2). Useful for telemetry /
    ablation: e.g. "remove profile scorer" should collapse the persona-sourced
    candidate distribution."""
    ENVIRONMENT = "environment"     # nearby resource / tool / building
    NEED = "need"                   # hunger / cold / injury driven
    PERSONA = "persona"             # altruism->help, dominance->lead, ...
    SOCIAL = "social"               # trust / debt / betrayal driven
    INSTITUTION = "institution"     # enforce / appeal / vote / challenge
    WISH = "wish"                   # ReAct thought -> wish parser -> affordance


@dataclass
class ActionCandidate:
    """A proposed action before scoring. Wraps the (type, parameters) shape of
    :class:`agent_sdk.contracts.action.AgentAction` plus provenance + an
    optional natural-language rationale produced by the LLM (§14.2)."""
    action_type: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    source: CandidateSource = CandidateSource.ENVIRONMENT
    rationale: str = ""
    # Optional target agent uid (for social actions: give/accuse/teach/...).
    target_uid: Optional[str] = None

    def to_action_dict(self) -> Dict[str, Any]:
        return {"type": self.action_type, "parameters": dict(self.parameters)}


@dataclass
class ScoredCandidate:
    """An ActionCandidate after feature extraction + utility scoring (§14.4)."""
    candidate: ActionCandidate
    features: ActionFeatures
    utility: float
    # Optional breakdown for interpretability / debugging (feature -> contribution).
    contributions: Dict[str, float] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# §11.2 Need schema (wish parser output)
# --------------------------------------------------------------------------- #
class NeedType(str, Enum):
    STORAGE = "storage"
    FAIRNESS = "fairness"
    TEACHING = "teaching"
    TRADE = "trade"
    DEFENSE = "defense"
    LEADERSHIP = "leadership"
    RECORDING = "recording"
    PUNISHMENT = "punishment"
    COORDINATION = "coordination"


@dataclass
class Need:
    """Structured need extracted from a ReAct thought (§11.2)."""
    need_type: NeedType
    motivation: str = ""
    target_object: str = ""
    expected_effect: str = ""
    required_participants: List[str] = field(default_factory=list)
    required_resources: Dict[str, int] = field(default_factory=dict)
    urgency: float = 0.5  # [0,1]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["need_type"] = self.need_type.value
        return d


# --------------------------------------------------------------------------- #
# §12 Affordance proposal (synthesizer output -> verifier input)
# --------------------------------------------------------------------------- #
class VerifierVerdict(str, Enum):
    UNLOCKED = "unlocked"               # added to agent's action set
    PENDING_PROPOSAL = "pending"        # added to institution graph as proposal
    REJECTED = "rejected"


@dataclass
class AffordanceProposal:
    """A candidate affordance synthesized from a Need (§12.2).

    ``primitive_decomposition`` is the list of primitive action_types this
    affordance compiles down to — the verifier (§12.3) requires it to be
    non-empty and drawn from the registered primitive action set.
    """
    name: str                                   # e.g. "build_granary"
    need: Need
    primitive_decomposition: List[str] = field(default_factory=list)
    required_resources: Dict[str, int] = field(default_factory=dict)
    cost: Dict[str, float] = field(default_factory=dict)
    failure_modes: List[str] = field(default_factory=list)
    observable: bool = True
    # Filled by the verifier:
    verdict: Optional[VerifierVerdict] = None
    reject_reason: str = ""


# --------------------------------------------------------------------------- #
# §15 Speech act
# --------------------------------------------------------------------------- #
class SpeechIntent(str, Enum):
    ASK_FOR_HELP = "ask_for_help"
    OFFER_HELP = "offer_help"
    WARN = "warn"
    COMFORT = "comfort"
    ACCUSE = "accuse"
    APOLOGIZE = "apologize"
    NEGOTIATE = "negotiate"
    TEACH = "teach"
    PROMISE = "promise"
    THREATEN = "threaten"
    PROPOSE_RULE = "propose_rule"
    RALLY_SUPPORT = "rally_support"
    CHALLENGE_LEADER = "challenge_leader"
    VERIFY_CLAIM = "verify_claim"


@dataclass
class SpeechStyle:
    """§15.5 style variables (all [0,1])."""
    directness: float = 0.5
    warmth: float = 0.5
    assertiveness: float = 0.5
    politeness: float = 0.5
    verbosity: float = 0.5
    emotionality: float = 0.5
    honesty: float = 1.0
    strategic_ambiguity: float = 0.0
    collectivism_language: float = 0.5
    self_disclosure: float = 0.5


@dataclass
class SpeechAct:
    """A planned utterance (§15.2 pipeline output). ``utterance`` is filled by
    the LLM in the final stage; the graph updates it triggers are described in
    ``graph_effects`` (e.g. {"social": "promise", "target": "agent_3"})."""
    intent: SpeechIntent
    target_uid: Optional[str] = None
    public: bool = True
    content_refs: Dict[str, Any] = field(default_factory=dict)  # §15.4 memory/rule/record refs
    style: SpeechStyle = field(default_factory=SpeechStyle)
    utterance: str = ""
    graph_effects: Dict[str, Any] = field(default_factory=dict)
