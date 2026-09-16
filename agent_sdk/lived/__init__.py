"""agent_sdk.lived — Graph-Grounded Lived Agents decision system (scaffold).

Environment-agnostic skeleton for the design doc "Graph-Grounded Lived Agents
for Emergent Primitive Societies". Mirrors the (removed) ``agent_sdk.evolving``
slot: it is the new decision-system package the harness path plugs into.

Boundary rule (CLAUDE.md): this package NEVER imports ``environments``. All
env-specific logic (feature extraction, candidate sources, world-state graph
population) arrives via the Protocol ports in ``agent_sdk.lived.core.ports``.

Layout (each maps to design doc sections):
  contracts.py   — frozen DTOs / schema           (§3, §11, §12, §14, §15)
  schema.py      — Persona Graph schema: traits/states/features (Priority 1+2)
  wmatrix.py     — Trait→Feature weight matrix (Priority 3, the W matrix)
  appraisal.py   — EventAppraisal + ordered update pipeline (Priority 4, §5)
  episode.py     — EpisodeManager: multi-turn activity objects (Priority 5, §6)
  profile.py     — ProfileState + ProfileUpdater  (§3, §14.8)
  ports.py       — SDK<->Env Protocol seam         (§14.3 feature extraction)
  policy.py      — Profile-to-Policy scorer        (§14, the core mechanism)
  wish.py        — wish parser                      (§11)
  affordance.py  — synthesizer + verifier           (§12)
  speech.py      — profile-conditioned speech       (§15)
  graphs/        — profile/event/social/knowledge/institution graphs (§3-7)
  emergence/     — structural emergence detector     (§17)
  controller.py  — LivedDecisionController (ReAct loop orchestrator, §13)

The "Persona-Conditioned Policy and Episode Dynamics" line (schema.py +
wmatrix.py + appraisal.py + episode.py + the policy/profile updates) answers:
how do an agent's persona, experience, state and the multi-turn episodes it is
in shape its next action in an interpretable, reproducible, non-prompt-
engineering way?

Import is lazy/opt-in: nothing here is pulled in by the v2 path. Wire it in the
harness path only (see controller.py docstring for the Act-phase hook plan).

The full public surface is re-exported here so downstream code (and the future
harness wiring) can do ``from agent_sdk.lived import X`` for any major symbol.
"""
# --- contracts / schema (§3, §11, §12, §14, §15) -------------------------- #
from agent_sdk.lived.core.contracts import (
    ActionCandidate,
    ActionFeatures,
    AffordanceProposal,
    CandidateSource,
    IdentityGrounding,
    MoodState,
    Need,
    NeedType,
    ProfileVector,
    ProfileWeights,
    ScoredCandidate,
    SpeechAct,
    SpeechIntent,
    SpeechStyle,
    VerifierVerdict,
)

# --- persona graph schema (Priority 1+2: traits/states/features frozen) --- #
from agent_sdk.lived.core.schema import (
    COST_FEATURES,
    EVENT_KINDS,
    FEATURE_SPECS,
    INTENSITY_SCALE,
    STATE_SPECS,
    TRAIT_SPECS,
    EdgeType,
    FeatureSpec,
    FilledBy,
    Intensity,
    NodeType,
    StateSpec,
    TraitNode,
    TraitSpec,
    default_trait_nodes,
)

# --- trait->feature weight matrix (PCBSP §8) ------------------------------ #
from agent_sdk.lived.core.wmatrix import (
    ALLOWED_W_LEVELS,
    W_MATRIX,
    profile_to_weights,
    validate_matrix,
    validate_monotonicity,
    weight_sensitivity,
)
# --- feature transforms (PCBSP §6) ---------------------------------------- #
from agent_sdk.lived.core.transforms import (
    SurvivalPressure,
    hp_pressure,
    hunger_pressure,
    transform_features,
    transform_value,
)
# --- annotation + shared candidate pool (PCBSP §3/§4/§5) ------------------ #
from agent_sdk.lived.persona.annotator import (
    DEFAULT_ACTION_FEATURES,
    RuleBasedAnnotator,
    build_candidate_pool,
    sanitize_features,
    snap_intensity,
)

# --- event appraisal + ordered update pipeline (Priority 4) --------------- #
from agent_sdk.lived.cognition.appraisal import (
    EventAppraisal,
    EventAppraiser,
    apply_event_pipeline,
)

# --- episode manager (Priority 5) ----------------------------------------- #
from agent_sdk.lived.cognition.episode import (
    EPISODE_RULES,
    AgentAttention,
    Episode,
    EpisodeManager,
    EpisodeRule,
    EpisodeStatus,
    EpisodeType,
    RouteResult,
    TerminalAppraisal,
)

# --- profile state + updater (§3, §14.8) ---------------------------------- #
from agent_sdk.lived.persona.profile import ProfileState, ProfileUpdater

# --- ports + NoOp fallbacks (§14.3 seam) ---------------------------------- #
from agent_sdk.lived.core.ports import (
    AffordanceSynthPort,
    CandidateSourcePort,
    FeatureExtractorPort,
    GraphStorePort,
    NoOpCandidateSource,
    NoOpFeatureExtractor,
    WishParserPort,
)

# --- policy: PCBSP scorer (§12-§15) --------------------------------------- #
from agent_sdk.lived.persona.policy import (
    ProfileToPolicy,
    TriggerContext,
    mood_shift,
    sampling_temperature,
)
from agent_sdk.lived.persona.pcbsp import (
    DecisionContext,
    PCBSPPolicy,
    PolicyTrace,
)

# --- wish parser (§11) ---------------------------------------------------- #
from agent_sdk.lived.cognition.wish import RuleBasedWishParser

# --- affordance synth + verifier (§12) ------------------------------------ #
from agent_sdk.lived.perceive.affordance import (
    DEFAULT_TEMPLATES,
    AffordanceSynthesizer,
    AffordanceTemplate,
    AffordanceVerifier,
)

# --- speech policy (§15) -------------------------------------------------- #
from agent_sdk.lived.cognition.speech import (
    INTENT_GRAPH_EFFECT,
    SpeechPolicy,
    derive_style,
)

# --- graphs (§3-7) -------------------------------------------------------- #
from agent_sdk.lived.graphs import (
    Edge,
    EventGraph,
    EventRecord,
    GraphStore,
    InstitutionGraph,
    KnowledgeGraph,
    Node,
    ProfileGraph,
    RuleState,
    SocialGraph,
)

# --- emergence detector (§17) --------------------------------------------- #
from agent_sdk.lived.emergence import DetectionResult, EmergenceDetector

# --- controller / orchestrator (§13) -------------------------------------- #
from agent_sdk.lived.loop.controller import DecisionResult, LivedDecisionController

# --- frontend aggregator (read-only viz snapshot, §17/§18) ---------------- #
from agent_sdk.lived.perceive.view import EMPTY_LIVED_SNAPSHOT, lived_society_snapshot
# --- deep snapshot + run recorder (Inspector replay, §28) ----------------- #
from agent_sdk.lived.record.recorder import (
    RunRecorder,
    graph_detail,
    lived_full_snapshot,
    profile_detail,
)

# --- infrastructure: LLM engine / logs / perception / memory (infra task) -- #
from agent_sdk.lived.llm.llm_engine import (
    DEFAULT_MODEL_ROLES,
    EVIDENCE_KINDS,
    DecodingConfig,
    LLMEngine,
    MockProvider,
    ReplayCacheMiss,
    validate_schema,
)
# Real OpenAI transport (opt-in; `openai` imported lazily inside).
from agent_sdk.lived.llm.providers_openai import (
    DEFAULT_MODEL_MAP,
    OpenAILivedProvider,
    build_openai_lived_engine,
)
from agent_sdk.lived.record.logs import (
    ALL_STREAMS,
    IMPLEMENTED_STREAMS,
    PLACEHOLDER_STREAMS,
    Journal,
)
from agent_sdk.lived.perceive.perception import (
    AgentGT,
    GroundTruthScene,
    PerceptionAppraisal,
    PerceptionBuilder,
    PerceptionPacket,
    SelfStatePercept,
    derive_self_signals,
)
from agent_sdk.lived.cognition.memory import (
    MemoryContext,
    MemoryNode,
    MemoryRetrievalQuery,
    MemoryRetriever,
    MemoryStore,
    query_from_perception,
)

# --- Stage B: action layer + live control loop ----------------------------- #
from agent_sdk.lived.core.actions import (
    ACTION_REGISTRY,
    SPEECH_ACTIONS,
    ActionCategory,
    ActionSpec,
    WishCandidate,
    get_spec,
)
from agent_sdk.lived.loop.handlers import (
    ActionExecutionResult,
    SpeechEvent,
    execute_action,
)
from agent_sdk.lived.perceive.candidates import (
    CandidatePoolGenerator,
    EnvAwareFeatureExtractor,
    PrecomputedFeatureExtractor,
)
from agent_sdk.lived.loop.control_loop import LiveLoop, build_smoke_world, decide_for_agent

# --- async tick loop + day/night + LLM queue (async infra) ----------------- #
from agent_sdk.lived.world.clock import (
    WorldClock,
    gather_yield_multiplier,
    move_cost_multiplier,
    passive_energy_decay,
    sleep_recovery,
    visual_radius,
)
from agent_sdk.lived.loop.scheduler import AsyncTickLoop, seeded_rng
from agent_sdk.lived.llm.llm_queue import (
    DecisionSnapshot,
    LLMJob,
    LLMQueue,
    capture_snapshot,
)

__all__ = [
    # contracts
    "ProfileVector", "MoodState", "ActionFeatures", "ProfileWeights",
    "ActionCandidate", "ScoredCandidate", "CandidateSource",
    "Need", "NeedType", "AffordanceProposal", "VerifierVerdict",
    "SpeechAct", "SpeechIntent", "SpeechStyle", "IdentityGrounding",
    # persona graph schema (Priority 1+2)
    "NodeType", "EdgeType", "TraitSpec", "TRAIT_SPECS", "TraitNode",
    "default_trait_nodes", "StateSpec", "STATE_SPECS",
    "FeatureSpec", "FEATURE_SPECS", "FilledBy", "COST_FEATURES",
    "Intensity", "INTENSITY_SCALE", "EVENT_KINDS",
    # W matrix (PCBSP §8) + transforms (§6)
    "W_MATRIX", "ALLOWED_W_LEVELS", "profile_to_weights",
    "validate_matrix", "validate_monotonicity", "weight_sensitivity",
    "transform_features", "transform_value",
    "hunger_pressure", "hp_pressure", "SurvivalPressure",
    # annotation + shared pool (PCBSP §3/§4/§5)
    "RuleBasedAnnotator", "DEFAULT_ACTION_FEATURES", "build_candidate_pool",
    "sanitize_features", "snap_intensity",
    # appraisal + update pipeline (Priority 4)
    "EventAppraisal", "EventAppraiser", "apply_event_pipeline",
    # episode manager (Priority 5)
    "EpisodeManager", "Episode", "EpisodeType", "EpisodeStatus",
    "EpisodeRule", "EPISODE_RULES", "TerminalAppraisal", "RouteResult",
    "AgentAttention",
    # profile
    "ProfileState", "ProfileUpdater",
    # ports
    "FeatureExtractorPort", "CandidateSourcePort", "WishParserPort",
    "AffordanceSynthPort", "GraphStorePort",
    "NoOpFeatureExtractor", "NoOpCandidateSource",
    # policy (PCBSP §12-§15)
    "ProfileToPolicy", "PCBSPPolicy", "DecisionContext", "PolicyTrace",
    "TriggerContext", "mood_shift", "sampling_temperature",
    # wish / affordance
    "RuleBasedWishParser",
    "AffordanceSynthesizer", "AffordanceVerifier", "AffordanceTemplate",
    "DEFAULT_TEMPLATES",
    # speech
    "SpeechPolicy", "derive_style", "INTENT_GRAPH_EFFECT",
    # graphs
    "GraphStore", "Node", "Edge",
    "EventGraph", "EventRecord", "SocialGraph", "KnowledgeGraph",
    "InstitutionGraph", "RuleState", "ProfileGraph",
    # emergence
    "EmergenceDetector", "DetectionResult",
    # controller
    "LivedDecisionController", "DecisionResult",
    # frontend aggregator
    "lived_society_snapshot", "EMPTY_LIVED_SNAPSHOT",
    # deep snapshot + recorder (Inspector replay)
    "lived_full_snapshot", "RunRecorder", "profile_detail", "graph_detail",
    # infrastructure: LLM engine / logs / perception / memory (infra task)
    "LLMEngine", "MockProvider", "DecodingConfig", "ReplayCacheMiss",
    "validate_schema", "DEFAULT_MODEL_ROLES", "EVIDENCE_KINDS",
    "OpenAILivedProvider", "build_openai_lived_engine", "DEFAULT_MODEL_MAP",
    "Journal", "ALL_STREAMS", "IMPLEMENTED_STREAMS", "PLACEHOLDER_STREAMS",
    "PerceptionBuilder", "PerceptionPacket", "SelfStatePercept", "PerceptionAppraisal",
    "GroundTruthScene", "AgentGT", "derive_self_signals",
    "MemoryNode", "MemoryStore", "MemoryRetriever", "MemoryRetrievalQuery",
    "MemoryContext", "query_from_perception",
    # Stage B: action layer + live control loop
    "ACTION_REGISTRY", "ActionSpec", "ActionCategory", "WishCandidate", "get_spec",
    "SPEECH_ACTIONS", "execute_action", "ActionExecutionResult", "SpeechEvent",
    "CandidatePoolGenerator", "EnvAwareFeatureExtractor", "PrecomputedFeatureExtractor",
    "LiveLoop", "build_smoke_world", "decide_for_agent",
    # async tick loop + day/night + LLM queue
    "WorldClock", "visual_radius", "move_cost_multiplier", "gather_yield_multiplier",
    "passive_energy_decay", "sleep_recovery",
    "AsyncTickLoop", "seeded_rng",
    "LLMQueue", "LLMJob", "DecisionSnapshot", "capture_snapshot",
]
