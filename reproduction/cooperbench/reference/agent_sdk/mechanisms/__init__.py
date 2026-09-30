"""
Mechanism Port Abstractions + Concrete Implementations
========================================================
Sub-packages:
  collective_memory/ — Gene+Event asset store, retrieval, feedback
  strategy_mutation/ — StrategyMutationPort + PersonaTraits + MutationResult
  communication/     — NegotiationView, CommunicationPort
  auto_trigger/      — Conditioned Reflex system (rule-based automatic actions)
"""
from agent_sdk.mechanisms.collective_memory import (
    CollectiveMemoryPort,
    AssetStore,
    Gene, Event,
    RetrievalEngine,
    FeedbackLedger,
)
from agent_sdk.mechanisms.strategy_mutation import StrategyMutationPort, PersonaTraits, MutationResult
from agent_sdk.mechanisms.auto_trigger import (
    AutoTriggerPort,
    AutoTriggerStore,
    ReflexTemplate,
    ReflexContext,
    AgentReflexEntry,
    ConditionEvaluator,
    ActionProgramExecutor,
)

__all__ = [
    # Collective Memory
    "CollectiveMemoryPort",
    "AssetStore",
    "Gene", "Event",
    "RetrievalEngine",
    "FeedbackLedger",
    # Strategy Mutation
    "StrategyMutationPort",
    "PersonaTraits",
    "MutationResult",
    # Auto Trigger (Conditioned Reflex)
    "AutoTriggerPort",
    "AutoTriggerStore",
    "ReflexTemplate",
    "ReflexContext",
    "AgentReflexEntry",
    "ConditionEvaluator",
    "ActionProgramExecutor",
]
