"""
AgentBase — environment-agnostic base class for LLM agents.

This is the SDK-level agent. It has identity, context window,
and orchestration hooks. It does NOT have position, energy, HP, or
any other environment-specific attributes.

Environments extend this class to add their own attributes
(e.g., NatureLLMAgent adds x, y, energy, hp, inventory).
"""
from __future__ import annotations
from typing import Dict, Any, Optional, List

from agent_sdk.base.context import AgentContext


class AgentBase:
    """SDK-level agent base class. Environment-agnostic."""

    def __init__(
        self,
        agent_id: str,
        name: str,
        config: Dict[str, Any],
        role: str = "agent",
        system_context: Optional[str] = None,
    ):
        self.id = agent_id
        self.name = name
        self.config = config
        self.role = role

        # Context window (structured)
        self.ctx = AgentContext()
        self.system_context = system_context or ""

        # Legacy compatibility
        self.message_history: List[Dict[str, str]] = []

        # Status
        self.is_dead = False
        self.current_action = "idle"

        # Mechanism ports — injected at wire time by the engine
        # communication_port satisfies agent_sdk.mechanisms.communication.CommunicationPort
        self.communication_port: Optional[Any] = None

        # Conditioned Reflex: per-agent adopted templates
        # Dict[template_id, AgentReflexEntry]
        self._reflexes: Dict[str, Any] = {}
        self._auto_trigger_port: Optional[Any] = None

        # Reflex unlock gate: tracks LLM-chosen action counts.
        # Stage C reflex tools only appear after any action type reaches
        # the threshold (default 5).
        self._action_type_counts: Dict[str, int] = {}
        self._reflex_unlocked: bool = False

        # Reflex firing history for Stage C review.
        # List of recent (turn, template_name, outcome_snippet) tuples.
        self._reflex_fire_history: List[tuple] = []

        # Milestone-hint buffer for Stage C reflex review. Engine's milestone
        # callback appends dicts of {turn, milestone, hint}; keep only last ~10.
        self._recent_milestone_hints: List[Dict[str, Any]] = []

        # Action-component unlock state (Level 1-2 component gate).
        # ``_unlocked_components[name] = {"turn": int, "source": "milestone"|"attempts"}``.
        # Stage C reflex review uses this to tell the LLM which specialised
        # action components are currently available inside reflex templates.
        self._unlocked_components: Dict[str, Dict[str, Any]] = {}
        # Per-component successful-execution counter (for the 5-attempt gate).
        self._component_attempts: Dict[str, int] = {}
        # Buffer of pending unlock announcements to surface in Stage C.
        self._pending_component_unlocks: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Reflex management
    # ------------------------------------------------------------------

    def adopt_reflex(self, template, turn: int = 0) -> None:
        """Adopt a ReflexTemplate. Idempotent by template_id."""
        from agent_sdk.mechanisms.auto_trigger.types import AgentReflexEntry
        tid = template.template_id
        if tid in self._reflexes:
            return
        self._reflexes[tid] = AgentReflexEntry(
            template=template, adopted_turn=turn,
        )

    def remove_reflex(self, template_id: str) -> bool:
        """Remove an adopted reflex. Returns True if removed."""
        return self._reflexes.pop(template_id, None) is not None

    def enable_reflex(self, template_id: str, enabled: bool = True) -> None:
        """Enable or disable an adopted reflex."""
        entry = self._reflexes.get(template_id)
        if entry is not None:
            entry.enabled = enabled

    def list_reflexes(self) -> List[Any]:
        """Return all adopted AgentReflexEntry objects."""
        return list(self._reflexes.values())

    def get_reflex_entry(self, template_id: str) -> Optional[Any]:
        """Get a specific reflex entry."""
        return self._reflexes.get(template_id)

    # ------------------------------------------------------------------
    # Reflex unlock tracking
    # ------------------------------------------------------------------

    def record_action_type(self, action_type: str, threshold: int = 5) -> None:
        """Increment LLM-chosen action counter; unlock reflex when threshold met."""
        if not action_type or self._reflex_unlocked:
            return
        self._action_type_counts[action_type] = self._action_type_counts.get(action_type, 0) + 1
        if self._action_type_counts[action_type] >= threshold:
            self._reflex_unlocked = True
            print(f"[Reflex-Unlock] {self.id}: '{action_type}' x{threshold}, unlocked")

    def record_reflex_fire(self, turn: int, template_name: str, outcome: str) -> None:
        """Append a reflex firing record for Stage C review (keep last 30)."""
        self._reflex_fire_history.append((turn, template_name, outcome[:120]))
        if len(self._reflex_fire_history) > 30:
            self._reflex_fire_history = self._reflex_fire_history[-30:]

    # ------------------------------------------------------------------
    # Action-component unlock API (shared with env-specific subclasses)
    # ------------------------------------------------------------------

    def unlock_component(self, name: str, *, turn: int = 0,
                         source: str = "milestone", extra: Optional[Dict[str, Any]] = None) -> bool:
        """Mark an action component as unlocked for this agent.

        ``source`` is one of ``"milestone"`` / ``"attempts"`` / ``"default"``.
        Returns True on first unlock (so callers can suppress duplicates).
        """
        if not name or name in self._unlocked_components:
            return False
        record = {"turn": int(turn), "source": str(source)}
        if extra:
            record.update(extra)
        self._unlocked_components[name] = record
        # Push a pending announcement for the next Stage C reflex review
        announce = {"turn": int(turn), "component": name, "source": str(source)}
        if extra and "hint" in extra:
            announce["hint"] = extra["hint"]
        self._pending_component_unlocks.append(announce)
        if len(self._pending_component_unlocks) > 12:
            self._pending_component_unlocks = self._pending_component_unlocks[-12:]
        return True

    def record_component_attempt(
        self, component_name: str, *, success: bool = True,
        turn: int = 0, threshold: int = 5,
    ) -> bool:
        """Bump the per-component success counter; unlock at threshold.

        Returns True when this call flipped the component to unlocked.
        """
        if not component_name:
            return False
        if not success:
            return False
        if component_name in self._unlocked_components:
            return False
        self._component_attempts[component_name] = (
            self._component_attempts.get(component_name, 0) + 1
        )
        if self._component_attempts[component_name] >= int(threshold):
            return self.unlock_component(
                component_name, turn=turn, source="attempts",
            )
        return False

    def has_component(self, name: str) -> bool:
        """True if the component is unlocked (or is one of the defaults)."""
        return name in self._unlocked_components

    def drain_pending_component_unlocks(self) -> List[Dict[str, Any]]:
        """Return and clear the buffer of recent unlock announcements."""
        pending = list(self._pending_component_unlocks)
        self._pending_component_unlocks = []
        return pending

    def step(self, **kwargs):
        """Override in subclass."""
        raise NotImplementedError

    def get_env_state(self) -> Dict[str, Any]:
        """Return env-specific state as a dict. Override in env subclass.

        Engines can inject env_state via ``agent._env_state = {...}``
        for prompt template substitution without requiring a subclass.
        """
        return getattr(self, "_env_state", {}) or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "role": self.role,
            "is_dead": self.is_dead,
            "action": self.current_action,
        }
