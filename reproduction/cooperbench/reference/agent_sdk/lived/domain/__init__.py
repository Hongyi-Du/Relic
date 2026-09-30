"""agent_sdk.lived.domain — Core ↔ Domain decoupling seam (DESIGN §32).

Environment-agnostic interfaces every domain adapter (nature_env, org_env, …)
conforms to. Imports nothing from ``environments``.
"""
from agent_sdk.lived.domain.interfaces import (
    DomainAction,
    DomainActionMapper,
    DomainAdapter,
    DomainAgentState,
    DomainEntity,
    DomainEventAppraisalAdapter,
    DomainExecutionAdapter,
    DomainFeatureExtractor,
    DomainPerceptionAdapter,
    DomainReplayFormatter,
    DomainScenarioConfig,
    DomainState,
    VitalState,
)

__all__ = [
    "VitalState", "DomainEntity", "DomainAgentState", "DomainAction",
    "DomainState", "DomainScenarioConfig",
    "DomainPerceptionAdapter", "DomainActionMapper", "DomainExecutionAdapter",
    "DomainFeatureExtractor", "DomainEventAppraisalAdapter",
    "DomainReplayFormatter", "DomainAdapter",
]
