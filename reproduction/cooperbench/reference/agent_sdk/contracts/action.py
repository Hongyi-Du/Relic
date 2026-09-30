"""Agent action DTO: the agent's output to the environment each turn."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass(frozen=True)
class AgentAction:
    """
    A single action the agent wants to perform.

    Compatible with the existing ``decision`` dict format::

        {"type": "move", "parameters": {"x": 10, "y": 20}}

    becomes::

        AgentAction(type="move", parameters={"x": 10, "y": 20})
    """
    type: str
    parameters: Dict[str, Any] = field(default_factory=dict)

    # -- Convenience helpers --

    def to_dict(self) -> Dict[str, Any]:
        return {"type": self.type, "parameters": dict(self.parameters)}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AgentAction":
        return cls(
            type=str(d.get("type", "idle")),
            parameters=dict(d.get("parameters", {})),
        )
