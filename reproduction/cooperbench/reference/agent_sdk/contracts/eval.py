"""Evaluation DTOs for civilization-level assessment."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass(frozen=True)
class EvalSnapshot:
    """
    Point-in-time snapshot of civilization metrics.

    Used by the CivilJudge (or equivalent evaluator) to assess
    progress across a generation / round.
    """
    turn: int = 0
    alive_count: int = 0
    total_gdp: float = 0.0
    knowledge_count: int = 0
    avg_energy: float = 0.0
    avg_age: float = 0.0
    per_agent: Dict[str, Any] = field(default_factory=dict)
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RoundEvalTrigger:
    """
    Signal that a round-level evaluation should be triggered.

    The environment emits this when its round-end conditions are met
    (e.g., all agents dead, max turns reached).
    """
    reason: str = ""
    round_number: int = 0
    snapshot: EvalSnapshot = field(default_factory=EvalSnapshot)
