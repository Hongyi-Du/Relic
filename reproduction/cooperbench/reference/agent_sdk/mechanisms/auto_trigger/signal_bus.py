"""
auto_trigger/signal_bus.py — ReflexSignalBus.

Lightweight in-memory signal bus for reflex-to-reflex communication.
Isolated from LLM chat — only reflex templates can emit/receive signals.
"""
from __future__ import annotations

from typing import Any, Dict, List

from .types import ReflexSignal


class ReflexSignalBus:
    """
    Signal bus for inter-reflex communication.

    Mounted on world (env-side). Written by ActionProgramExecutor's
    emit_signal node; read by ConditionEvaluator's signal_received condition.
    """

    def __init__(self):
        self.pending: Dict[str, List[ReflexSignal]] = {}

    def emit(self, signal: ReflexSignal, agents_positions: Dict[str, tuple]) -> None:
        """Broadcast signal to all agents within radius (Chebyshev distance)."""
        sx, sy = signal.sender_position
        for agent_id, (ax, ay) in agents_positions.items():
            if agent_id == signal.sender_id:
                continue
            dist = max(abs(sx - ax), abs(sy - ay))
            if dist <= signal.radius:
                self.pending.setdefault(agent_id, []).append(signal)

    def consume(self, agent_id: str, signal_id: str) -> None:
        """Remove a consumed signal so it doesn't trigger again."""
        if agent_id in self.pending:
            self.pending[agent_id] = [
                s for s in self.pending[agent_id] if s.signal_id != signal_id
            ]

    def get_signals(self, agent_id: str, max_age: int, turn: int) -> List[ReflexSignal]:
        """Get pending signals for an agent, filtered by age."""
        return [
            s for s in self.pending.get(agent_id, [])
            if turn - s.turn_sent <= max_age
        ]

    def cleanup_old(self, turn: int, max_age: int = 10) -> None:
        """Prune signals older than max_age turns."""
        for agent_id in list(self.pending.keys()):
            self.pending[agent_id] = [
                s for s in self.pending[agent_id]
                if turn - s.turn_sent <= max_age
            ]
            if not self.pending[agent_id]:
                del self.pending[agent_id]
