"""Environment-neutral contracts used by the Relic research runtime.

Environment implementations map their state and actions through these data
objects and adapter protocols. This module intentionally imports nothing from
``environments`` so the dependency always points from an environment to the
shared contracts, never the other way around.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


# --------------------------------------------------------------------------- #
# DTOs (domain-side; concrete dataclasses)
# --------------------------------------------------------------------------- #
@dataclass
class VitalState:
    """Named runtime state variables supplied by an environment adapter."""
    variables: dict[str, float] = field(default_factory=dict)

    def get(self, key: str, default: float = 0.0) -> float:
        return float(self.variables.get(key, default))

    def guard_violated(self) -> bool:
        """Return whether an environment-defined safety guard was violated."""
        return False


@dataclass
class DomainEntity:
    """A non-agent object and the actions it currently affords."""
    entity_id: str
    entity_type: str
    owner: str | None = None
    visibility: str = "public"  # public | team | private | channel
    location_or_channel: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    state: dict[str, Any] = field(default_factory=dict)
    affordances: list[str] = field(default_factory=list)


@dataclass
class DomainAgentState:
    """Environment state visible through one persistent member."""
    agent_id: str
    agent_name: str = ""
    role: str = ""
    location_or_workspace: Any | None = None
    vitals: VitalState = field(default_factory=VitalState)
    private_state: dict[str, Any] = field(default_factory=dict)
    public_state: dict[str, Any] = field(default_factory=dict)
    inventory_or_workspace_access: dict[str, Any] = field(default_factory=dict)
    active_tasks_or_plans: list[str] = field(default_factory=list)
    skills: dict[str, float] = field(default_factory=dict)
    relationships: dict[str, Any] = field(default_factory=dict)
    permissions: list[str] = field(default_factory=list)


@dataclass
class DomainAction:
    """An action descriptor exposed by an environment registry."""
    action_type: str
    action_category: str = ""
    actor_id: str | None = None
    target_entity_id: str | None = None
    target_agent_id: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    preconditions: list[str] = field(default_factory=list)
    duration: int = 1
    cost: dict[str, float] = field(default_factory=dict)
    visibility: str = "public"
    result_schema: dict[str, Any] = field(default_factory=dict)


@dataclass
class DomainState:
    """A snapshot of one environment at the current tick."""
    run_id: str = "run"
    world_tick: int = 0
    agents: dict[str, DomainAgentState] = field(default_factory=dict)
    entities: dict[str, DomainEntity] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list)
    shared_resources: dict[str, Any] = field(default_factory=dict)
    public_records: list[dict[str, Any]] = field(default_factory=list)
    domain_clock: dict[str, Any] = field(default_factory=dict)
    domain_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DomainScenarioConfig:
    """Reproducible scenario parameters pinned by corpus version and seed."""
    name: str = "default"
    seed: int = 0
    corpus_version: str = ""
    params: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Adapter ports. Decision-layer types stay ``Any`` to keep the seam thin.
# --------------------------------------------------------------------------- #
@runtime_checkable
class DomainPerceptionAdapter(Protocol):
    """(agent, DomainState) -> core PerceptionPacket (+ SelfState)."""
    def build_perception(self, *, agent_id: str, state: DomainState) -> Any: ...


@runtime_checkable
class DomainActionMapper(Protocol):
    """Available DomainActions for an agent -> core ActionCandidates."""
    def available_actions(self, *, agent_id: str, state: DomainState) -> list[DomainAction]: ...
    def to_core_candidates(self, actions: list[DomainAction]) -> list[Any]: ...


@runtime_checkable
class DomainExecutionAdapter(Protocol):
    """Execute a chosen action against the world; return a result/effect dict."""
    def execute(self, *, agent_id: str, action: DomainAction, state: DomainState) -> dict[str, Any]: ...


@runtime_checkable
class DomainFeatureExtractor(Protocol):
    """Map an agent, candidate, and state to decision-policy features."""
    def extract(self, *, agent_id: str, candidate: Any, state: DomainState) -> Any: ...


@runtime_checkable
class DomainEventAppraisalAdapter(Protocol):
    """Map a domain action-result/event into a core EventAppraisal."""
    def appraise(self, *, agent_id: str, event: dict[str, Any], state: DomainState) -> Any: ...


@runtime_checkable
class DomainReplayFormatter(Protocol):
    """Render a DomainState into a JSON-serializable frame for the Inspector."""
    def format_frame(self, *, state: DomainState) -> dict[str, Any]: ...


@runtime_checkable
class DomainAdapter(Protocol):
    """Composite adapter exposed by a Relic environment."""
    perception: DomainPerceptionAdapter
    action_mapper: DomainActionMapper
    execution: DomainExecutionAdapter
    feature_extractor: DomainFeatureExtractor
    appraisal: DomainEventAppraisalAdapter
    replay: DomainReplayFormatter

    def scenario(self) -> DomainScenarioConfig: ...
    def get_state(self) -> DomainState: ...
    def agent_ids(self) -> list[str]: ...
