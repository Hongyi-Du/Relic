"""Domain decoupling interfaces — the SocioGenesis Core ↔ Domain seam (DESIGN §32).

The lived Core (perception / memory / PCBSP / plan / wish / affordance / detector /
LLMEngine / logging) is environment-agnostic. Each environment (nature_env,
org_env, …) implements a **Domain Adapter** that maps its world into these
DTOs/ports; the Core never imports an environment.

Boundary rule (CLAUDE.md): this module imports NOTHING from ``environments``.
It is deliberately self-contained: domain-side DTOs are concrete dataclasses,
and the adapter Protocols use ``Any`` for *core* types (PerceptionPacket,
ActionCandidate, ActionFeatures, EventAppraisal) so the seam stays thin and free
of fragile cross-imports. Each env's adapter does the actual mapping to/from the
core types.

Layout maps to DESIGN §32.2/§32.3:
  DTOs:   VitalState · DomainEntity · DomainAgentState · DomainAction ·
          DomainState · DomainScenarioConfig
  Ports:  DomainPerceptionAdapter · DomainActionMapper · DomainExecutionAdapter ·
          DomainFeatureExtractor · DomainEventAppraisalAdapter ·
          DomainReplayFormatter  (+ composite DomainAdapter)

Status: 📐 skeleton — interfaces frozen here; env implementations are stubs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


# --------------------------------------------------------------------------- #
# DTOs (domain-side; concrete dataclasses)
# --------------------------------------------------------------------------- #
@dataclass
class VitalState:
    """Generalized self-state (DESIGN §32.4 route 1).

    The Core does not hard-code energy/satiety. A domain supplies a named bag of
    vital variables with its own semantics + death/guard condition:
      * nature: {energy, fatigue, satiety, hp}
      * org:    {attention, fatigue, stress, deadline_pressure, workload, reputation_pressure}
    ``guard_violated`` lets the PCBSP "vital guard" stay domain-pluggable (a
    survival guard for nature; for org there is no "starve to death", so it may
    always return False).
    """
    variables: Dict[str, float] = field(default_factory=dict)

    def get(self, key: str, default: float = 0.0) -> float:
        return float(self.variables.get(key, default))

    def guard_violated(self) -> bool:
        """Override per domain. Default: never (no vital collapse)."""
        return False


@dataclass
class DomainEntity:
    """A non-agent object in the world (§32.3). ``affordances`` lists the action
    types this entity currently supports."""
    entity_id: str
    entity_type: str
    owner: Optional[str] = None
    visibility: str = "public"            # public | team | private | channel
    location_or_channel: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    state: Dict[str, Any] = field(default_factory=dict)
    affordances: List[str] = field(default_factory=list)


@dataclass
class DomainAgentState:
    """Per-agent domain state (§32.3). ``vitals`` replaces nature-specific
    energy/satiety with a generalized bag (see VitalState)."""
    agent_id: str
    agent_name: str = ""
    role: str = ""
    location_or_workspace: Optional[Any] = None
    vitals: VitalState = field(default_factory=VitalState)
    private_state: Dict[str, Any] = field(default_factory=dict)
    public_state: Dict[str, Any] = field(default_factory=dict)
    inventory_or_workspace_access: Dict[str, Any] = field(default_factory=dict)
    active_tasks_or_plans: List[str] = field(default_factory=list)
    skills: Dict[str, float] = field(default_factory=dict)
    relationships: Dict[str, Any] = field(default_factory=dict)
    permissions: List[str] = field(default_factory=list)


@dataclass
class DomainAction:
    """A domain action descriptor (§32.3). The action *registry* (per env) holds
    these; ``DomainActionMapper`` turns the agent's available ones into core
    ActionCandidates for the PCBSP policy."""
    action_type: str
    action_category: str = ""             # work | artifact | protocol | comm | bridge | survival | ...
    actor_id: Optional[str] = None
    target_entity_id: Optional[str] = None
    target_agent_id: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    preconditions: List[str] = field(default_factory=list)
    duration: int = 1
    cost: Dict[str, float] = field(default_factory=dict)
    visibility: str = "public"
    result_schema: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DomainState:
    """A snapshot of one environment at the current tick (§32.3). The Core reads
    this through the adapter; it never touches env internals directly."""
    run_id: str = "run"
    world_tick: int = 0
    agents: Dict[str, DomainAgentState] = field(default_factory=dict)
    entities: Dict[str, DomainEntity] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    messages: List[Dict[str, Any]] = field(default_factory=list)
    shared_resources: Dict[str, Any] = field(default_factory=dict)
    public_records: List[Dict[str, Any]] = field(default_factory=list)
    domain_clock: Dict[str, Any] = field(default_factory=dict)
    domain_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DomainScenarioConfig:
    """Reproducible scenario knobs (§32.2 / acceptance ⑫). ``corpus_version`` +
    ``seed`` must pin a run; for org this also freezes the external snapshot."""
    name: str = "default"
    seed: int = 0
    corpus_version: str = ""
    params: Dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Adapter Ports (Protocols). Core types are ``Any`` to keep the seam thin.
# --------------------------------------------------------------------------- #
@runtime_checkable
class DomainPerceptionAdapter(Protocol):
    """(agent, DomainState) -> core PerceptionPacket (+ SelfState)."""
    def build_perception(self, *, agent_id: str, state: DomainState) -> Any: ...


@runtime_checkable
class DomainActionMapper(Protocol):
    """Available DomainActions for an agent -> core ActionCandidates."""
    def available_actions(self, *, agent_id: str, state: DomainState) -> List[DomainAction]: ...
    def to_core_candidates(self, actions: List[DomainAction]) -> List[Any]: ...


@runtime_checkable
class DomainExecutionAdapter(Protocol):
    """Execute a chosen action against the world; return a result/effect dict."""
    def execute(self, *, agent_id: str, action: DomainAction, state: DomainState) -> Dict[str, Any]: ...


@runtime_checkable
class DomainFeatureExtractor(Protocol):
    """(agent, candidate, state) -> core ActionFeatures for the PCBSP scorer.

    Per §32.4 the feature schema should split into core_features (generic:
    risk/conformity/dominance/…) + domain_features (env-injected). For the
    skeleton this returns whatever the core scorer consumes.
    """
    def extract(self, *, agent_id: str, candidate: Any, state: DomainState) -> Any: ...


@runtime_checkable
class DomainEventAppraisalAdapter(Protocol):
    """Map a domain action-result/event into a core EventAppraisal."""
    def appraise(self, *, agent_id: str, event: Dict[str, Any], state: DomainState) -> Any: ...


@runtime_checkable
class DomainReplayFormatter(Protocol):
    """Render a DomainState into a JSON-serializable frame for the Inspector."""
    def format_frame(self, *, state: DomainState) -> Dict[str, Any]: ...


@runtime_checkable
class DomainAdapter(Protocol):
    """Composite seam an environment exposes to the lived Core (§32.2).

    Bundles the sub-adapters + scenario + a tick interface. The Core's control
    loop drives: get_state -> perception -> action_mapper -> feature_extractor ->
    PCBSP -> execution -> appraisal, all through this adapter.
    """
    perception: DomainPerceptionAdapter
    action_mapper: DomainActionMapper
    execution: DomainExecutionAdapter
    feature_extractor: DomainFeatureExtractor
    appraisal: DomainEventAppraisalAdapter
    replay: DomainReplayFormatter

    def scenario(self) -> DomainScenarioConfig: ...
    def get_state(self) -> DomainState: ...
    def agent_ids(self) -> List[str]: ...
