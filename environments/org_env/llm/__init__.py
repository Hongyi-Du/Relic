"""OrgEnv LLM cognitive layer.

Principle: **LLM proposes / reasons / composes / evaluates / verbalizes; the System
validates / executes / records / commits.** The LLM never mutates the world — it only
emits candidate intents / reflections / wishes / proposals / drafts / summaries that
go through validators + managers + ``world.apply_action`` before anything changes.

``client.py`` is the provider-agnostic wrapper (Mock for tests, OpenAI/HTTP for
runtime); each cognitive module (action decision, surface realizer, reflection,
wish, proposal, tool, institution, summary, evaluation) builds on it with a JSON
schema + a deterministic template fallback.
"""
from environments.org_env.llm.client import (
    LLMError,
    MockOrgLLMClient,
    OrgLLMClient,
    build_org_llm_client,
)

__all__ = ["OrgLLMClient", "MockOrgLLMClient", "LLMError", "build_org_llm_client"]
