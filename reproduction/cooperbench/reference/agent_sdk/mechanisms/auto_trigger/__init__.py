"""
agent_sdk.mechanisms.auto_trigger — Conditioned Reflex System (Level 1-5).

Provides rule-based reflex objects that bypass LLM inference for
well-characterised action patterns. Upgraded to Level 4/5 on 2026-04-18:

  * Level 1-2: self reflexes (individual survival / routine labour).
  * Level 3:    signal-coupled cooperation via ReflexSignalBus.
  * Level 4:    ReflexPairTemplate + PairClosureTracker — cooperative
                emit/respond pairs evaluated by closure.
  * Level 5:    ModeTemplate + ModeManager — TTL-bounded mode activations
                with prompt-overlay injection.

Key components:
  * ``types``            — dataclasses (incl. ReflexKind, UtilityTrace,
                           ReflexPairTemplate, ModeTemplate, ModeActivation)
  * ``port``             — AutoTriggerPort Protocol (boundary interface)
  * ``store``            — AutoTriggerStore (publish/vote/deprecation with
                           three-pool Top-K + exploration quota + kind-aware
                           Laplace confidence + UtilityTrace ingestion)
  * ``evaluator``        — ConditionEvaluator (JSON DSL → bool+signal)
  * ``executor``         — ActionProgramExecutor (node tree → action,
                           signals, mode activations)
  * ``signal_bus``       — ReflexSignalBus (in-memory broadcast)
  * ``pair``             — PairRegistry + PairClosureTracker
  * ``mode``             — ModeManager (per-agent TTL / exclusion / cooldown)
  * ``scoring``          — confidence / vitality / adoption score
  * ``builtin_templates``— empty registry (envs provide their own)
"""
from .types import (
    AgentReflexEntry,
    AgentSnapshot,
    EntitySnapshot,
    ModeActivation,
    ModeTemplate,
    PairRole,
    ReflexContext,
    ReflexKind,
    ReflexPairTemplate,
    ReflexSignal,
    ReflexSignalType,
    ReflexTemplate,
    TemplateStatus,
    TriggerResult,
    UtilityTrace,
)
from .port import AutoTriggerPort
from .store import AutoTriggerStore
from .evaluator import ConditionEvaluator
from .executor import ActionProgramExecutor
from .signal_bus import ReflexSignalBus
from .pair import ClosureEvent, PairClosureTracker, PairRegistry
from .mode import ModeManager
from . import scoring
from .builtin_templates import ALL_BUILTIN_TEMPLATES, MILESTONE_TO_TEMPLATES

__all__ = [
    # types
    "AgentReflexEntry",
    "AgentSnapshot",
    "EntitySnapshot",
    "ModeActivation",
    "ModeTemplate",
    "PairRole",
    "ReflexContext",
    "ReflexKind",
    "ReflexPairTemplate",
    "ReflexSignal",
    "ReflexSignalType",
    "ReflexTemplate",
    "TemplateStatus",
    "TriggerResult",
    "UtilityTrace",
    # ports / infra
    "AutoTriggerPort",
    "AutoTriggerStore",
    "ConditionEvaluator",
    "ActionProgramExecutor",
    "ReflexSignalBus",
    # L4 / L5
    "PairRegistry",
    "PairClosureTracker",
    "ClosureEvent",
    "ModeManager",
    "scoring",
    # builtin registry (kept empty at SDK level)
    "ALL_BUILTIN_TEMPLATES",
    "MILESTONE_TO_TEMPLATES",
]
