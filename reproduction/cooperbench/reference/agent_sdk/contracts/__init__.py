"""
agent_sdk.contracts — shared DTOs between Agent SDK and any Environment.

This package is the *only* shared dependency — neither the Agent SDK
internals nor any environment implementation should import each other's
internals.  Only contract DTOs cross the boundary.
"""
from .agent_status import AgentStatus
from .observation import (
    AgentView,
    DomainSignalsV1,
    EnvObservation,
    ProgressState,
    SignalLevel,
)
from .tools import ToolDescriptor, ToolEntry
from .action import AgentAction
from .result import EnvResult
from .eval import EvalSnapshot, RoundEvalTrigger
from .env_adapter import EnvAdapter

__all__ = [
    "AgentStatus",
    "AgentView",
    "DomainSignalsV1",
    "EnvObservation",
    "ProgressState",
    "SignalLevel",
    "ToolDescriptor",
    "ToolEntry",
    "AgentAction",
    "EnvResult",
    "EvalSnapshot",
    "RoundEvalTrigger",
    "EnvAdapter",
]
