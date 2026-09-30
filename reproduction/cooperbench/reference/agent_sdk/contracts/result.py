"""Environment result DTO: the environment's response after executing an action."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EnvResult:
    """
    The result of executing an AgentAction in the environment.

    * ``outcome`` — "success" | "failure" | "partial"
    * ``feedback`` — human-readable description of what happened
    * ``metric_delta`` — score change caused by this action
    * ``terminal`` — whether the agent's episode ended (e.g. death)
    * ``free_action`` — if True, agent gets a bonus decision this turn
    """
    outcome: str = "unknown"
    feedback: str = ""
    metric_delta: float = 0.0
    terminal: bool = False
    free_action: bool = False
