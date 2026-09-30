"""agent_sdk.quick — simple facade for standalone agent usage."""
from .tool_decorator import tool
from .agent import Agent
from .team import Team

__all__ = ["Agent", "tool", "Team"]
