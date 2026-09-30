"""Tool descriptor DTOs: describe skills/tools available to the agent."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass(frozen=True)
class ToolDescriptor:
    """
    A single tool (skill) that the agent can invoke.

    Mirrors the OpenAI function-calling schema shape so it can be passed
    directly to the LLM ``tools`` parameter.
    """
    name: str
    description: str = ""
    schema: Dict[str, Any] = field(default_factory=dict)
    available: bool = True


@dataclass(frozen=True)
class ToolEntry:
    """
    Lightweight tool entry for Stage A tool-list rendering (ToolsIndex)
    AND for harness_sdk Act-phase BasicTool input_schema population.

    Environments construct these to register their available tools.
    Use ToolDescriptor for full JSON-schema tools passed to the LLM.

    The optional ``schema`` field holds the JSON Schema for the tool's
    arguments (i.e. the OpenAI function-calling ``parameters`` object).
    When provided, harness_sdk.runtime_adapter converters will wire it
    into ``BasicTool.input_schema`` so the LLM sees the full hard spec
    (required / properties / enum) in the ``tools=`` channel. When
    empty, consumers fall back to a permissive default — but that
    fallback is the S4 bug we want to close (see spec 2026-04-19 §5.5).
    """
    name: str
    description: str
    source: str = "env"  # "env" | "sdk"
    schema: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolUnlock:
    """
    Notification that a passive tool has become available this turn.

    Delivered via EnvObservation.unlocked_tools.
    """
    name: str
    reason: str
    description: str = ""  # L1 ultra-brief description for system prompt catalog
