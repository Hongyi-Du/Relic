"""Reproducible Relic scenarios for the frozen W01-W10 workloads."""
from __future__ import annotations

from relic.core.domain import DomainScenarioConfig


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
                     dataset_id: str = "mini_blobstore_v1",
                     control: str = "none", mode: str = "dev",
                     anonymize: bool = False,
                     evaluator_config: dict = None) -> DomainScenarioConfig:
    """OSS time-machine substrate (brief §6.2): seed the company product from a REAL OSS project's
    early runnable release (frozen locally). Future code / release notes / hidden behavior tests are
    withheld; historical issues drive the work.

    ``dataset_id`` picks a frozen pack under ``benchmarks/relic-main-v1/packs``.
    ``mode='formal'`` is the experiment mode and refuses fixture substrates.
    ``control`` selects an anti-scripting control condition (brief §12).

    ``evaluator_config`` is the first-class switch for objective OSS metrics.
    Formal runs keep hidden tests evaluator-only until one final post-development
    evaluation; public tests and smoke remain available during development."""
    ec = dict(evaluator_config or {})
    if mode == "formal":                               # objective metrics are mandatory in formal
        ec["run_oss_hidden_tests"] = False              # evaluator-only once, after development
        ec["run_oss_final_evaluation"] = True
        ec["run_oss_manual_checks"] = _dataset_requires_manual_release_checks(
            dataset_id
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


def oss_time_machine_formal(seed: int = 0, dataset_id: str = "mini_blobstore_v1",
                            control: str = "none") -> DomainScenarioConfig:
    """Formal OSS time-machine experiment: real substrate REQUIRED (fixtures are rejected)."""
    return oss_time_machine(seed=seed, dataset_id=dataset_id, control=control, mode="formal")


SCENARIOS = {
    "oss_time_machine": oss_time_machine,
    "oss_time_machine_formal": oss_time_machine_formal,
}

__all__ = ["oss_time_machine", "oss_time_machine_formal", "SCENARIOS"]
