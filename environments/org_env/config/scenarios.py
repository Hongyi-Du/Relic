"""OrgEnv scenarios — reproducible configs (DESIGN env_org §34).

Flagship golden-chain scenario: API Price Shock (§34.1) — external API price ×3
-> internal budget pressure -> workflow wish -> budget ledger artifact ->
expensive-run approval protocol emergence. Schedule + corpus pin in Stage O3/O5.
"""
from __future__ import annotations

from agent_sdk.lived.domain.interfaces import DomainScenarioConfig


def default_scenario(seed: int = 0, corpus_version: str = "v0") -> DomainScenarioConfig:
    return DomainScenarioConfig(name="org_default", seed=seed, corpus_version=corpus_version,
                                params={"num_internal_agents": 8})


def api_price_shock(seed: int = 0, corpus_version: str = "v0",
                    shock_tick: int = 60, multiplier: float = 3.0) -> DomainScenarioConfig:
    """API price ×N shock at ``shock_tick`` (§34.1). SKELETON: declares knobs;
    signal injection + budget cost_multiplier bump wired in Stage O5."""
    return DomainScenarioConfig(
        name="api_price_shock", seed=seed, corpus_version=corpus_version,
        params={
            "num_internal_agents": 8,
            "shock_tick": shock_tick,
            "api_cost_multiplier": multiplier,
            "signal_type": "api_price_x3",
        },
    )


def _dataset_requires_manual_release_checks(dataset_id: str) -> bool:
    from environments.org_env.product.substrates.loader import (
        load_oss_substrate_spec,
    )
    from environments.org_env.product.substrates.final_evaluation import (
        manifest_requires_manual_release_checks,
    )

    spec = load_oss_substrate_spec(dataset_id)
    return manifest_requires_manual_release_checks(spec.manifest)


def oss_time_machine(seed: int = 0, corpus_version: str = "oss-v0",
                     dataset_id: str = "gitingest_v015_to_v030",
                     control: str = "none", mode: str = "dev",
                     anonymize: bool = False,
                     evaluator_config: dict = None) -> DomainScenarioConfig:
    """OSS time-machine substrate (brief §6.2): seed the company product from a REAL OSS project's
    early runnable release (frozen locally). Future code / release notes / hidden behavior tests are
    withheld; historical issues drive the work.

    ``dataset_id`` picks the frozen dataset under ``environments/org_env/data/oss_time_machine/
    {real,projects,fixtures}/``; the default is the complete ``gitingest_v015_to_v030`` snapshot.
    ``mode='formal'`` is the experiment mode and REFUSES a fixture substrate (use a real one).
    ``control`` selects an anti-scripting control condition (brief §12).

    ``evaluator_config`` is the first-class switch for objective OSS metrics.
    Formal and host-controlled pilot runs keep hidden tests evaluator-only until
    one final post-development evaluation; public tests and smoke remain
    available during development."""
    ec = dict(evaluator_config or {})
    if mode in {"formal", "pilot"}:
        ec["run_oss_hidden_tests"] = False              # evaluator-only once, after development
        ec["run_oss_final_evaluation"] = True
        ec["run_oss_manual_checks"] = (
            _dataset_requires_manual_release_checks(dataset_id)
            if mode == "formal"
            else False
        )
        ec["hidden_feedback_forbidden"] = True
        ec["run_oss_public_tests"] = True
        ec["prewarm_smoke"] = True                      # runtime readiness real at t0, no env var
    return DomainScenarioConfig(
        name="oss_time_machine", seed=seed, corpus_version=corpus_version,
        params={
            "num_internal_agents": 8,
            "experiment_mode": mode,
            "company_config": {
                "product_substrate": {
                    "type": "oss_time_machine",
                    "dataset_id": dataset_id,
                    "anonymize": anonymize,
                    "control": control,
                    "mode": mode,
                    "evaluator_config": ec,
                }
            },
        },
    )


def oss_time_machine_formal(seed: int = 0, dataset_id: str = "gitingest_v015_to_v030",
                            control: str = "none") -> DomainScenarioConfig:
    """Formal OSS time-machine experiment: real substrate REQUIRED (fixtures are rejected)."""
    return oss_time_machine(seed=seed, dataset_id=dataset_id, control=control, mode="formal")


SCENARIOS = {"org_default": default_scenario, "api_price_shock": api_price_shock,
             "oss_time_machine": oss_time_machine, "oss_time_machine_formal": oss_time_machine_formal}

__all__ = ["default_scenario", "api_price_shock", "oss_time_machine",
           "oss_time_machine_formal", "SCENARIOS"]
