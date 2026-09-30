"""Claim-specific robustness audit scaffolding."""

from __future__ import annotations

from dataclasses import dataclass, field

from .hashing import stable_hash
from .runtime import SocietyConfig, SocietyRuntime
from .schemas import Artifact


@dataclass(frozen=True)
class RobustnessResult:
    claim_id: str
    perturbation_family: str
    seeds: tuple[int, ...]
    metric_name: str
    metric_values: tuple[float, ...]
    stability_threshold: float
    stable_count: int
    failure_modes: tuple[str, ...] = ()

    @property
    def stability_rate(self) -> float:
        return self.stable_count / max(1, len(self.seeds))


def audit_artifact_diffusion_across_seeds(
    *,
    seeds: tuple[int, ...],
    population_size: int = 100,
    ticks: int = 6,
    threshold: float = 0.1,
) -> RobustnessResult:
    values: list[float] = []
    for seed in seeds:
        runtime = SocietyRuntime(SocietyConfig(population_size=population_size, seed=seed))
        runtime.inject_artifact(
            Artifact(
                id="lantern_scout",
                provider_id="lanternforge",
                artifact_kind="tool",
                public_claims=("evidence-grounded research assistance",),
            )
        )
        runtime.run(ticks)
        report = runtime.detectors.detect_artifact_diffusion(
            runtime.state.event_log,
            "lantern_scout",
            exposure_count=sum(
                1 for agent in runtime.state.agents.values() if "lantern_scout" in agent.artifact_access
            ),
        )
        values.append(report.continuous_score)
    stable_count = sum(1 for value in values if value >= threshold)
    failure_modes: list[str] = []
    if stable_count < len(seeds):
        failure_modes.append("low_or_no_independent_trials")
    return RobustnessResult(
        claim_id="artifact_diffusion_without_product_specific_actions",
        perturbation_family="seed",
        seeds=seeds,
        metric_name="ArtifactDiffusionScore",
        metric_values=tuple(values),
        stability_threshold=threshold,
        stable_count=stable_count,
        failure_modes=tuple(failure_modes),
    )


def robustness_manifest_hash(result: RobustnessResult) -> str:
    return stable_hash(result)
