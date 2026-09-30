"""EnvAdapter Protocol: the contract between Agent SDK and any environment."""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Protocol, TypedDict, runtime_checkable

from .agent_status import AgentStatus
from .observation import EnvObservation
from .tools import ToolDescriptor
from .action import AgentAction
from .result import EnvResult
from .eval import EvalSnapshot
from .shard import MetricBound


@runtime_checkable
class EnvAdapter(Protocol):
    """
    The boundary interface between the Agent SDK and any environment.

    Any environment that wants to host agents must provide a concrete
    implementation of this protocol.  The Agent SDK never imports
    environment internals — it only interacts through this interface.

    Implementations are free to hold references to environment-specific
    objects (e.g., ``world``) internally; the protocol surface only
    exposes contract DTOs.
    """

    def get_agent_status(self, agent_uid: str) -> AgentStatus:
        """Return the lifecycle status of a specific agent."""
        ...

    def observe(self, agent_uid: str) -> EnvObservation:
        """
        Build a full observation for the agent.

        Includes rendered perception text, domain signals, last-action
        feedback, visible agents, and possible interaction options.
        """
        ...

    def list_tools(self, agent_uid: str, intent: str = "") -> List[ToolDescriptor]:
        """
        Return the tools available to the agent this turn.

        The ``intent`` string (from Stage A) allows the environment
        to contextually filter which tools are relevant.
        """
        ...

    def execute(self, agent_uid: str, action: AgentAction) -> EnvResult:
        """
        Execute an agent action in the environment.

        Returns an EnvResult with outcome, feedback, and metric delta.
        """
        ...

    def get_eval_snapshot(self) -> EvalSnapshot:
        """Return a point-in-time evaluation snapshot of the simulation."""
        ...

    def should_trigger_round_eval(self) -> bool:
        """Check whether the environment's round-end conditions are met."""
        ...

    # -- Lifecycle helpers (environment-specific, called by runner) --

    def tick_lifecycle(self, agent_uid: str) -> None:
        """
        Advance per-turn lifecycle for an agent (energy decay, temperature,
        exploration, death checks, etc.).

        Called by the runner *before* the agent's decision step.
        """
        ...

    # -- Optional spatial methods (only for grid/spatial environments) --
    # Non-spatial environments (e.g., research, trading) need NOT implement these.
    # The SDK guards calls with hasattr() / try-except, so missing is safe.

    def request_pathfind(
        self, agent_uid: str, target_x: int, target_y: int
    ) -> List[tuple]:
        """
        (Optional) Request a path from the agent's current position to (target_x, target_y).

        Only relevant for spatial/grid-based environments.
        Returns a list of (x, y) waypoints, or empty list if no path found.
        """
        ...

    def step_movement(self, agent_uid: str) -> Dict[str, Any]:
        """
        (Optional) Continue an agent's in-progress movement along its current path.

        Only relevant for spatial/grid-based environments.
        Returns a dict with movement results (e.g., path taken, blocked, arrived).
        """
        ...

    # -- Optional communication context (for Stage C.comm) --

    def get_pending_comm_context(self, agent_uid: str) -> dict:
        """
        (Optional) Return pending communication context for Stage C prompt.

        Returns a dict with:
          - "text": formatted string for the LLM prompt (pending invitations, etc.)
          - "available_tools": list of comm tool names available this turn
        Default: no invitations, basic tools.
        """
        ...


class EnvSnapshotV1(TypedDict):
    """Cold-start prompt payload delivered to the meta-evolver."""
    turn: int
    season: Optional[str]
    alive_count: int
    recent_deaths_window: int
    top_pressures: list[str]
    resource_scarcity: Mapping[str, float]
    shard_window_keywords: list[str]


@runtime_checkable
class HarnessEnvAdapter(EnvAdapter, Protocol):
    """Extended EnvAdapter for harness-driven environments.

    Adds the three F3-metric methods used by the meta-evolver. Inherits
    from EnvAdapter so isinstance checks satisfy both protocols.
    """

    def metric(self, agent_id: str, t_start: int, t_end: int) -> Mapping[str, float]:
        ...

    def metric_bounds(self) -> Mapping[str, MetricBound]:
        ...

    def snapshot(self) -> EnvSnapshotV1:
        ...
