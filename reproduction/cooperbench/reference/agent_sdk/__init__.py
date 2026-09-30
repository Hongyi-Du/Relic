"""
EvolvingAgents SDK — multi-agent orchestration library.

Quick API:
    from agent_sdk import Agent, tool, Team

Advanced:
    from agent_sdk.llm import LLMCore
    from agent_sdk.core import LLMAgentV2
    from agent_sdk.contracts import EnvAdapter, AgentAction, EnvResult
"""
from agent_sdk.quick import Agent, tool, Team

__all__ = ["Agent", "tool", "Team"]
