"""Agent lifecycle status enum shared between Agent SDK and Environment."""
from __future__ import annotations

from enum import Enum


class AgentStatus(str, Enum):
    """High-level lifecycle state of an agent."""
    IN_PROGRESS = "in_progress"   # actively executing
    IDLE = "idle"                 # paused, can be resumed
    TERMINATED = "terminated"     # permanently done
