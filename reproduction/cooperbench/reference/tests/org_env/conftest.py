"""Shared helpers for O1 org_env tests."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent_sdk.lived.domain.interfaces import DomainScenarioConfig
from environments.org_env.backend.simulation.world import OrgWorld


def make_world(seed: int = 42, n: int = 8, policy_mode: str = "mock") -> OrgWorld:
    sc = DomainScenarioConfig(name="org_default", seed=seed, corpus_version="v0",
                              params={"num_internal_agents": n, "policy_mode": policy_mode})
    return OrgWorld(sc).build()


def make_oss_world(seed: int = 42, n: int = 8, dataset_id: str = "mini_cli_digest",
                   control: str = "none", mode: str = "dev", anonymize: bool = True) -> OrgWorld:
    """An OrgWorld seeded from the OSS time-machine substrate (brief §6). ``control`` selects an
    anti-scripting control condition (brief §12); ``mode='formal'`` rejects fixture substrates."""
    sc = DomainScenarioConfig(
        name="oss_time_machine", seed=seed, corpus_version="oss-v0",
        params={"num_internal_agents": n, "experiment_mode": mode,
                "company_config": {"product_substrate": {
                    "type": "oss_time_machine", "dataset_id": dataset_id, "anonymize": anonymize,
                    "control": control, "mode": mode}}})
    return OrgWorld(sc).build()


REAL_DATASET = "gitingest_v015_to_v030"


def make_real_oss_world(seed: int = 42, n: int = 8, mode: str = "dev",
                        control: str = "none") -> OrgWorld:
    """An OrgWorld seeded from the REAL gitingest OSS time-machine snapshot (anonymize off)."""
    return make_oss_world(seed=seed, n=n, dataset_id=REAL_DATASET, control=control, mode=mode,
                          anonymize=False)
