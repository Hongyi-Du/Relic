"""OrgEnvAdapter — the OrgEnv DomainAdapter (DESIGN core §32.2 / env_org §33).

Composite seam the lived Core drives. Bundles the org sub-adapters + scenario +
world; ``get_state / scenario / agent_ids`` work in the skeleton so a smoke test
can construct + drive the contract. Perception / execution / feature / appraisal
are stubs (Stage O2–O4). Conforms to ``agent_sdk.lived.domain.DomainAdapter``.
"""
from __future__ import annotations

from typing import List, Optional

from agent_sdk.lived.domain.interfaces import DomainScenarioConfig, DomainState
from environments.org_env.backend.simulation import OrgWorld
from environments.org_env.config.scenarios import default_scenario
from environments.org_env.runtime_adapter.execution import (
    OrgActionMapper,
    OrgExecutionAdapter,
)
from environments.org_env.runtime_adapter.feature_extractor import (
    OrgEventAppraisal,
    OrgFeatureExtractor,
)
from environments.org_env.runtime_adapter.perception import OrgPerceptionAdapter
from environments.org_env.runtime_adapter.replay import OrgReplayFormatter


class OrgEnvAdapter:
    """OrgEnv implementation of the DomainAdapter Protocol."""

    def __init__(self, scenario: Optional[DomainScenarioConfig] = None):
        self._scenario = scenario or default_scenario()
        self.world = OrgWorld(self._scenario).build()
        self.perception = OrgPerceptionAdapter()
        self.action_mapper = OrgActionMapper()
        self.execution = OrgExecutionAdapter()
        self.feature_extractor = OrgFeatureExtractor()
        self.appraisal = OrgEventAppraisal()
        self.replay = OrgReplayFormatter()

    def scenario(self) -> DomainScenarioConfig:
        return self._scenario

    def get_state(self) -> DomainState:
        return self.world.get_state()

    def agent_ids(self) -> List[str]:
        return list(self.world.agents.keys())


__all__ = ["OrgEnvAdapter"]
