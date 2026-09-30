"""
auto_trigger/port.py — AutoTriggerPort Protocol.

The boundary interface any environment can implement to support the
Conditioned Reflex system. SDK calls this; the env (engine) provides the
implementation.

Level 4/5 additions (2026-04-18):
  * Mode activation / query / overlay hooks — delegate to ModeManager.
  * Pair emit / consume / sweep hooks — delegate to PairClosureTracker.
  * ``ingest_utility_trace`` — unified channel for Score updates.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, Set, Tuple, runtime_checkable

from agent_sdk.contracts.observation import EnvObservation
from .types import (
    ModeActivation,
    ReflexContext,
    ReflexSignal,
    ReflexTemplate,
    UtilityTrace,
)


@runtime_checkable
class AutoTriggerPort(Protocol):
    """Protocol for the auto-trigger (conditioned reflex) mechanism."""

    # -- Template management --

    def publish_template(self, template: ReflexTemplate) -> ReflexTemplate:
        """Publish a new template to the shared store. Returns the stored template."""
        ...

    def get_template(self, template_id: str) -> Optional[ReflexTemplate]:
        """Retrieve a template by ID."""
        ...

    def list_templates(
        self, tags: Optional[List[str]] = None, top_k: int = 20
    ) -> List[ReflexTemplate]:
        """List templates, optionally filtered by tags."""
        ...

    def update_template(
        self, template_id: str, updates: Dict[str, Any], requester_id: str
    ) -> Optional[ReflexTemplate]:
        """Update a template (only creator can edit). Returns updated template or None."""
        ...

    def delete_template(self, template_id: str, requester_id: str) -> bool:
        """Delete/deprecate a template (only creator). Returns success."""
        ...

    # -- Voting --

    def apply_vote(
        self, template_id: str, vote: str, voter_id: str, turn: int
    ) -> float:
        """Apply a vote (like/dislike). Returns confidence delta."""
        ...

    def get_top_templates(self, n: int = 5) -> List[ReflexTemplate]:
        """Get top-N templates by confidence."""
        ...

    # -- Unlock --

    def unlock_templates(self, template_ids: List[str]) -> None:
        """Transition templates from locked → active."""
        ...

    # -- Context building (env-specific, delegated to adapter) --

    def build_reflex_context(
        self, agent_id: str, obs: EnvObservation
    ) -> ReflexContext:
        """Build a ReflexContext snapshot from an EnvObservation."""
        ...

    # -- Signal bus --

    def emit_signal(self, signal: ReflexSignal) -> None:
        """Emit a reflex signal to nearby agents."""
        ...

    def get_signals(
        self, agent_id: str, max_age: int, turn: int
    ) -> List[ReflexSignal]:
        """Get pending signals for an agent."""
        ...

    def consume_signal(self, agent_id: str, signal_id: str) -> None:
        """Remove a consumed signal from the bus."""
        ...

    def get_direction_away_from(
        self, agent_id: str, position: Tuple[int, int]
    ) -> Tuple[int, int]:
        """Calculate a safe position away from the given danger position."""
        ...

    # -- L5 mode management --

    def activate_mode(
        self, agent_id: str, mode_id: str, turn: int,
        ttl_override: Optional[int] = None,
    ) -> Optional[ModeActivation]:
        """Activate a mode window for an agent."""
        ...

    def get_active_modes(self, agent_id: str, turn: int) -> Set[str]:
        """Return the set of active mode_ids for an agent."""
        ...

    def get_active_mode_overlays(self, agent_id: str, turn: int) -> List[str]:
        """Return ordered overlay names for all active modes on an agent."""
        ...

    def record_mode_turn_metrics(
        self, agent_id: str, turn: int, *,
        llm_cost_saved: float = 0.0,
        energy_delta_pct: float = 0.0,
        hp_delta_pct: float = 0.0,
        metric_delta: float = 0.0,
        pair_closure_delta: float = 0.0,
        thrash_delta: float = 0.0,
    ) -> None:
        """Feed one turn of metrics into any open mode windows on ``agent_id``."""
        ...

    # -- L4 pair tracking --

    def register_pair_emit(
        self, pair_id: str, emit_id: str, emitter_agent_id: str, turn: int
    ) -> None:
        """Record that a pair emitter fired and is awaiting closure."""
        ...

    def register_pair_consume(
        self, pair_id: str, emit_id: str, responder_agent_id: str, turn: int
    ) -> Optional[Any]:
        """Record that a responder consumed a pair signal; returns a
        :class:`ClosureEvent` when the consumption matches a pending emit."""
        ...

    def sweep_pair_timeouts(self, turn: int) -> None:
        """Turn off pending emits that exceeded their closure window."""
        ...

    # -- Utility trace ingestion --

    def ingest_utility_trace(
        self, trace: UtilityTrace, *, pair_closed: Optional[bool] = None
    ) -> None:
        """Feed a UtilityTrace into store aggregates (confidence + vitality)."""
        ...
