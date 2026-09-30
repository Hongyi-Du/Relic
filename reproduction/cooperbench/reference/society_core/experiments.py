"""Experiment orchestration and proof-carrying reports for Society-Core."""

from __future__ import annotations

import json
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from time import monotonic
from typing import Callable, TypeVar

from .artifacts import community_checklist_artifact, lantern_scout_artifact, placebo_artifact
from .controls import ArtifactCovariateControlReport, build_artifact_covariate_control_report, extract_agent_covariates
from .detectors import DetectorReport
from .hashing import canonicalize, stable_hash
from .institutions import InstitutionLongRunReport, build_institution_long_run_report
from .llm_intent import DEFAULT_LLM_INTENT_STANDARD_MODEL
from .registries import ClaimAudit
from .robustness import RobustnessResult, audit_artifact_diffusion_across_seeds
from .runtime import SocietyConfig, SocietyRuntime
from .scenarios import (
    inject_health_or_fatigue_shock,
    inject_public_conflict,
    inject_resource_shock,
    inject_rumor,
)
from .schemas import RunManifest
from .schemas import ExternalEvent


SCENARIO_NAMES = ("resource_shock", "rumor", "public_conflict", "health_shock")
T = TypeVar("T")


@dataclass(frozen=True)
class SingleRunResult:
    seed: int
    population_size: int
    ticks: int
    scenario_names: tuple[str, ...]
    with_lantern_scout: bool
    with_non_company_artifact: bool
    with_placebo_artifact: bool
    initial_agent_covariates: dict[str, dict[str, float]]
    manifest: RunManifest
    event_count: int
    event_log_hash: str
    state_hash: str
    detector_reports: dict[str, DetectorReport]

    @property
    def detector_scores(self) -> dict[str, float]:
        return {
            name: report.continuous_score
            for name, report in sorted(self.detector_reports.items())
        }


@dataclass(frozen=True)
class SingleRunPayload:
    result: SingleRunResult
    artifact_control_reports: dict[str, ArtifactCovariateControlReport]
    event_log: tuple[ExternalEvent, ...] = ()


@dataclass(frozen=True)
class SeedMetricInterval:
    metric_name: str
    primary_unit: str
    uncertainty_method: str
    confidence_level: float
    seeds: tuple[int, ...]
    values: tuple[float, ...]
    missing_seed_count: int
    mean: float
    sample_sd: float
    ci_low: float
    ci_high: float
    caveats: tuple[str, ...] = ()


@dataclass(frozen=True)
class SeedLevelUncertaintyReport:
    report_id: str
    primary_unit: str
    forbidden_method: str
    run_count: int
    scenario_names: tuple[str, ...]
    metric_intervals: dict[str, SeedMetricInterval]

    def hash(self) -> str:
        return stable_hash(self)


@dataclass(frozen=True)
class InstitutionLongRunSeedReport:
    seed: int
    event_count: int
    event_log_hash: str
    detector_stage: str | None
    detector_score: float
    long_run_report: InstitutionLongRunReport


@dataclass(frozen=True)
class InstitutionLongRunExperimentReport:
    report_id: str
    primary_unit: str
    run_count: int
    population_size: int
    ticks: int
    scenario_names: tuple[str, ...]
    window_size: int
    min_stable_windows: int
    seed_reports: tuple[InstitutionLongRunSeedReport, ...]
    metric_intervals: dict[str, SeedMetricInterval]
    claim_ready_seed_count: int
    claim_ready_rate: float
    claim_status: str

    def hash(self) -> str:
        return stable_hash(self)


def apply_named_scenarios(runtime: SocietyRuntime, scenario_names: tuple[str, ...]) -> None:
    for scenario in scenario_names:
        if scenario == "resource_shock":
            inject_resource_shock(runtime.state)
        elif scenario == "rumor":
            inject_rumor(runtime.state)
        elif scenario == "public_conflict":
            inject_public_conflict(runtime.state)
        elif scenario == "health_shock":
            inject_health_or_fatigue_shock(runtime.state)
        else:
            raise ValueError(f"Unknown Society-Core scenario: {scenario}")


def run_single_seed(
    *,
    seed: int,
    population_size: int,
    ticks: int,
    scenario_names: tuple[str, ...] = (),
    with_lantern_scout: bool = False,
    with_non_company_artifact: bool = False,
    with_placebo_artifact: bool = False,
    detectors_during_run: bool = False,
    state_hash_mode: str = "merkle",
    network_enabled: bool = True,
    artifact_initial_access_fraction: float = 0.25,
    max_social_out_degree: int = 8,
    network_holdout_fraction: float = 0.25,
    peer_signal_control_fraction: float = 0.5,
    peer_signal_utility_enabled: bool = True,
    body_affect_coupling_enabled: bool = True,
    llm_intent_provider: str = "heuristic",
    llm_intent_model: str = DEFAULT_LLM_INTENT_STANDARD_MODEL,
    llm_intent_timeout_seconds: float = 20.0,
    scenario_id: str = "society_core_v16_experiment",
) -> tuple[SocietyRuntime, SingleRunResult]:
    runtime = SocietyRuntime(
        SocietyConfig(
            population_size=population_size,
            seed=seed,
            detectors_enabled_during_run=detectors_during_run,
            llm_enabled=llm_intent_provider == "openai",
            llm_intent_provider=llm_intent_provider,
            llm_intent_model=llm_intent_model,
            llm_intent_timeout_seconds=llm_intent_timeout_seconds,
            scenario_id=scenario_id,
            state_hash_mode=state_hash_mode,
            network_enabled=network_enabled,
            artifact_initial_access_fraction=artifact_initial_access_fraction,
            max_social_out_degree=max_social_out_degree,
            network_holdout_fraction=network_holdout_fraction,
            peer_signal_control_fraction=peer_signal_control_fraction,
            peer_signal_utility_enabled=peer_signal_utility_enabled,
            body_affect_coupling_enabled=body_affect_coupling_enabled,
        )
    )
    initial_agent_covariates = extract_agent_covariates(runtime.state)
    if with_non_company_artifact:
        runtime.inject_artifact(community_checklist_artifact())
    if with_lantern_scout:
        runtime.inject_artifact(lantern_scout_artifact())
    if with_placebo_artifact:
        runtime.inject_artifact(placebo_artifact())
    apply_named_scenarios(runtime, scenario_names)
    runtime.run(ticks)
    reports = runtime.detectors.run_all(runtime.state)
    result = SingleRunResult(
        seed=seed,
        population_size=population_size,
        ticks=ticks,
        scenario_names=scenario_names,
        with_lantern_scout=with_lantern_scout,
        with_non_company_artifact=with_non_company_artifact,
        with_placebo_artifact=with_placebo_artifact,
        initial_agent_covariates=initial_agent_covariates,
        manifest=runtime.manifest(),
        event_count=len(runtime.state.event_log),
        event_log_hash=runtime.event_log_hash(),
        state_hash=runtime.state_hash(),
        detector_reports=reports,
    )
    return runtime, result


def _run_single_seed_from_kwargs(kwargs: dict) -> tuple[SocietyRuntime, SingleRunResult]:
    return run_single_seed(**kwargs)


def run_single_seed_payload(
    *,
    include_event_log: bool = False,
    **kwargs,
) -> SingleRunPayload:
    runtime, result = run_single_seed(**kwargs)
    artifact_control_reports = {
        artifact_id: build_artifact_covariate_control_report(
            state=runtime.state,
            initial_covariates=result.initial_agent_covariates,
            artifact_id=artifact_id,
        )
        for artifact_id in sorted(runtime.state.artifacts)
    }
    return SingleRunPayload(
        result=result,
        artifact_control_reports=artifact_control_reports,
        event_log=tuple(runtime.state.event_log) if include_event_log else (),
    )


def _run_single_seed_payload_from_kwargs(kwargs: dict) -> SingleRunPayload:
    include_event_log = bool(kwargs.pop("include_event_log", False))
    return run_single_seed_payload(include_event_log=include_event_log, **kwargs)


def _terminate_process_pool(executor: ProcessPoolExecutor) -> None:
    terminate_workers = getattr(executor, "terminate_workers", None)
    if callable(terminate_workers):
        terminate_workers()
        return
    # Python <3.14 has no public hard-stop API for ProcessPoolExecutor.
    for process in tuple((getattr(executor, "_processes", {}) or {}).values()):
        process.terminate()
    executor.shutdown(wait=False, cancel_futures=True)


def _collect_parallel_seed_outputs(
    *,
    run_kwargs: list[dict],
    workers: int,
    worker_fn: Callable[[dict], T],
    seed_timeout_seconds: float | None,
) -> list[T]:
    executor = ProcessPoolExecutor(max_workers=workers)
    executor_terminated = False
    try:
        future_to_seed = {
            executor.submit(worker_fn, kwargs): kwargs["seed"]
            for kwargs in run_kwargs
        }
        started_at = {future: monotonic() for future in future_to_seed}
        pending = set(future_to_seed)
        results_by_seed: dict[int, T] = {}
        while pending:
            done, pending = wait(pending, timeout=1.0, return_when=FIRST_COMPLETED)
            if not done:
                if seed_timeout_seconds is not None:
                    now = monotonic()
                    timed_out = tuple(
                        sorted(
                            future_to_seed[future]
                            for future in pending
                            if now - started_at[future] > seed_timeout_seconds
                        )
                    )
                    if timed_out:
                        executor_terminated = True
                        _terminate_process_pool(executor)
                        raise TimeoutError(
                            "seed run exceeded "
                            f"{seed_timeout_seconds:.1f}s timeout for seeds={timed_out}"
                        )
                continue
            for future in done:
                seed = future_to_seed[future]
                try:
                    results_by_seed[seed] = future.result()
                except Exception as exc:
                    executor_terminated = True
                    _terminate_process_pool(executor)
                    raise RuntimeError(f"seed run failed for seed={seed}") from exc
        return [results_by_seed[kwargs["seed"]] for kwargs in run_kwargs]
    finally:
        if not executor_terminated:
            executor.shutdown(wait=True)


def _student_t_975(df: int) -> float:
    table = {
        1: 12.706,
        2: 4.303,
        3: 3.182,
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
        10: 2.228,
        11: 2.201,
        12: 2.179,
        13: 2.16,
        14: 2.145,
        15: 2.131,
        16: 2.12,
        17: 2.11,
        18: 2.101,
        19: 2.093,
        20: 2.086,
        21: 2.08,
        22: 2.074,
        23: 2.069,
        24: 2.064,
        25: 2.06,
        26: 2.056,
        27: 2.052,
        28: 2.048,
        29: 2.045,
        30: 2.042,
    }
    if df <= 0:
        return 0.0
    if df in table:
        return table[df]
    if df <= 60:
        return 2.0
    return 1.96


def seed_level_interval(metric_name: str, seeds: tuple[int, ...], values: tuple[float, ...]) -> SeedMetricInterval:
    if len(seeds) != len(values):
        raise ValueError("seed and value counts must match")
    n = len(values)
    mean = sum(values) / n if n else 0.0
    if n <= 1:
        sd = 0.0
        low = mean
        high = mean
    else:
        sd = sqrt(sum((value - mean) ** 2 for value in values) / (n - 1))
        margin = _student_t_975(n - 1) * sd / sqrt(n)
        low = max(0.0, mean - margin)
        high = min(1.0, mean + margin)
    caveats: list[str] = []
    if n < 3:
        caveats.append("fewer_than_three_independent_seeds")
    return SeedMetricInterval(
        metric_name=metric_name,
        primary_unit="independent_seed",
        uncertainty_method="seed_level_t_interval",
        confidence_level=0.95,
        seeds=seeds,
        values=values,
        missing_seed_count=0,
        mean=mean,
        sample_sd=sd,
        ci_low=low,
        ci_high=high,
        caveats=tuple(caveats),
    )


def seed_level_raw_interval(metric_name: str, seeds: tuple[int, ...], values: tuple[float, ...]) -> SeedMetricInterval:
    if len(seeds) != len(values):
        raise ValueError("seed and value counts must match")
    n = len(values)
    mean = sum(values) / n if n else 0.0
    if n <= 1:
        sd = 0.0
        low = mean
        high = mean
    else:
        sd = sqrt(sum((value - mean) ** 2 for value in values) / (n - 1))
        margin = _student_t_975(n - 1) * sd / sqrt(n)
        low = mean - margin
        high = mean + margin
    caveats: list[str] = []
    if n < 3:
        caveats.append("fewer_than_three_independent_seeds")
    return SeedMetricInterval(
        metric_name=metric_name,
        primary_unit="independent_seed",
        uncertainty_method="seed_level_t_interval",
        confidence_level=0.95,
        seeds=seeds,
        values=values,
        missing_seed_count=0,
        mean=mean,
        sample_sd=sd,
        ci_low=low,
        ci_high=high,
        caveats=tuple(caveats),
    )


def build_seed_level_uncertainty_report(results: tuple[SingleRunResult, ...]) -> SeedLevelUncertaintyReport:
    metric_names = sorted({name for result in results for name in result.detector_scores})
    intervals: dict[str, SeedMetricInterval] = {}
    for metric_name in metric_names:
        seeds: list[int] = []
        values: list[float] = []
        detector_caveats: list[str] = []
        missing = 0
        for result in results:
            score = result.detector_scores.get(metric_name)
            if score is None:
                missing += 1
                continue
            seeds.append(result.seed)
            values.append(score)
            detector_caveats.extend(result.detector_reports[metric_name].caveats)
        interval = seed_level_interval(metric_name, tuple(seeds), tuple(values))
        caveats = list(interval.caveats)
        caveats.extend(sorted(set(detector_caveats)))
        if missing:
            caveats.append("metric_missing_for_some_seeds")
        intervals[metric_name] = SeedMetricInterval(
            metric_name=interval.metric_name,
            primary_unit=interval.primary_unit,
            uncertainty_method=interval.uncertainty_method,
            confidence_level=interval.confidence_level,
            seeds=interval.seeds,
            values=interval.values,
            missing_seed_count=missing,
            mean=interval.mean,
            sample_sd=interval.sample_sd,
            ci_low=interval.ci_low,
            ci_high=interval.ci_high,
            caveats=tuple(caveats),
        )
    scenario_names = tuple(sorted({name for result in results for name in result.scenario_names}))
    return SeedLevelUncertaintyReport(
        report_id="society_core_seed_level_uncertainty_v15",
        primary_unit="independent_seed",
        forbidden_method="naive_event_level_bootstrap_for_main_claims",
        run_count=len(results),
        scenario_names=scenario_names,
        metric_intervals=intervals,
    )


def build_claim_audit_report(
    claim_audit: ClaimAudit,
    uncertainty_report: SeedLevelUncertaintyReport,
    robustness_result: RobustnessResult | None = None,
) -> dict:
    records = {
        claim_id: canonicalize(record)
        for claim_id, record in sorted(claim_audit.records.items())
    }
    overclaim_records = [
        claim_id
        for claim_id, record in claim_audit.records.items()
        if record.claim_level > record.evidence_level
    ]
    return {
        "report_id": "society_core_claim_audit_v15",
        "claim_level_ceiling_enforced": not overclaim_records,
        "overclaim_records": overclaim_records,
        "records": records,
        "uncertainty_report_hash": uncertainty_report.hash(),
        "robustness_report_hash": stable_hash(robustness_result) if robustness_result else None,
        "allowed_main_claim_ceiling": "L1 until construct-validity and robustness packs are expanded",
    }


def run_institution_long_run_experiment(
    *,
    seeds: tuple[int, ...],
    population_size: int,
    ticks: int,
    scenario_names: tuple[str, ...] = ("rumor", "public_conflict"),
    window_size: int | None = None,
    min_stable_windows: int = 3,
    detectors_during_run: bool = False,
    workers: int = 1,
    seed_timeout_seconds: float | None = None,
) -> InstitutionLongRunExperimentReport:
    effective_window_size = window_size or max(10, ticks // max(1, min_stable_windows))
    payloads = run_seed_payload_batch(
        seeds=seeds,
        population_size=population_size,
        ticks=ticks,
        scenario_names=scenario_names,
        detectors_during_run=detectors_during_run,
        include_event_log=True,
        workers=workers,
        seed_timeout_seconds=seed_timeout_seconds,
    )
    seed_reports: list[InstitutionLongRunSeedReport] = []
    for payload in payloads:
        result = payload.result
        detector = result.detector_reports["institution_emergence"]
        long_run_report = build_institution_long_run_report(
            event_log=payload.event_log,
            window_size=effective_window_size,
            min_stable_windows=min_stable_windows,
        )
        seed_reports.append(
            InstitutionLongRunSeedReport(
                seed=result.seed,
                event_count=result.event_count,
                event_log_hash=result.event_log_hash,
                detector_stage=detector.stage_label,
                detector_score=detector.continuous_score,
                long_run_report=long_run_report,
            )
        )
    seed_tuple = tuple(report.seed for report in seed_reports)
    metric_intervals = {
        "institution_long_run_claim_ready": seed_level_interval(
            "institution_long_run_claim_ready",
            seed_tuple,
            tuple(float(report.long_run_report.claim_ready) for report in seed_reports),
        ),
        "institution_chain_completeness": seed_level_interval(
            "institution_chain_completeness",
            seed_tuple,
            tuple(report.long_run_report.chain_completeness for report in seed_reports),
        ),
        "role_persistence_score": seed_level_interval(
            "role_persistence_score",
            seed_tuple,
            tuple(report.long_run_report.role_persistence_score for report in seed_reports),
        ),
        "procedure_persistence_score": seed_level_interval(
            "procedure_persistence_score",
            seed_tuple,
            tuple(report.long_run_report.procedure_persistence_score for report in seed_reports),
        ),
        "stable_window_count": seed_level_raw_interval(
            "stable_window_count",
            seed_tuple,
            tuple(float(report.long_run_report.stable_window_count) for report in seed_reports),
        ),
        "institution_detector_score": seed_level_interval(
            "institution_detector_score",
            seed_tuple,
            tuple(report.detector_score for report in seed_reports),
        ),
    }
    claim_ready_seed_count = sum(1 for report in seed_reports if report.long_run_report.claim_ready)
    claim_ready_rate = claim_ready_seed_count / len(seed_reports) if seed_reports else 0.0
    report_payload = {
        "seeds": seed_tuple,
        "population_size": population_size,
        "ticks": ticks,
        "scenario_names": scenario_names,
        "window_size": effective_window_size,
        "min_stable_windows": min_stable_windows,
        "claim_ready_rate": claim_ready_rate,
        "seed_report_hashes": tuple(report.long_run_report.report_id for report in seed_reports),
    }
    return InstitutionLongRunExperimentReport(
        report_id="institution_long_run_experiment_" + stable_hash(report_payload)[:24],
        primary_unit="independent_seed",
        run_count=len(seed_reports),
        population_size=population_size,
        ticks=ticks,
        scenario_names=scenario_names,
        window_size=effective_window_size,
        min_stable_windows=min_stable_windows,
        seed_reports=tuple(seed_reports),
        metric_intervals=metric_intervals,
        claim_ready_seed_count=claim_ready_seed_count,
        claim_ready_rate=claim_ready_rate,
        claim_status="supported" if claim_ready_rate >= 0.5 else "insufficient_evidence",
    )


def run_seed_batch(
    *,
    seeds: tuple[int, ...],
    population_size: int,
    ticks: int,
    scenario_names: tuple[str, ...] = (),
    with_lantern_scout: bool = False,
    with_non_company_artifact: bool = False,
    with_placebo_artifact: bool = False,
    detectors_during_run: bool = False,
    state_hash_mode: str = "merkle",
    network_enabled: bool = True,
    artifact_initial_access_fraction: float = 0.25,
    max_social_out_degree: int = 8,
    network_holdout_fraction: float = 0.25,
    peer_signal_control_fraction: float = 0.5,
    peer_signal_utility_enabled: bool = True,
    body_affect_coupling_enabled: bool = True,
    llm_intent_provider: str = "heuristic",
    llm_intent_model: str = DEFAULT_LLM_INTENT_STANDARD_MODEL,
    llm_intent_timeout_seconds: float = 20.0,
    workers: int = 1,
    seed_timeout_seconds: float | None = None,
) -> tuple[list[SocietyRuntime], tuple[SingleRunResult, ...]]:
    run_kwargs = [
        {
            "seed": seed,
            "population_size": population_size,
            "ticks": ticks,
            "scenario_names": scenario_names,
            "with_lantern_scout": with_lantern_scout,
            "with_non_company_artifact": with_non_company_artifact,
            "with_placebo_artifact": with_placebo_artifact,
            "detectors_during_run": detectors_during_run,
            "state_hash_mode": state_hash_mode,
            "network_enabled": network_enabled,
            "artifact_initial_access_fraction": artifact_initial_access_fraction,
            "max_social_out_degree": max_social_out_degree,
            "network_holdout_fraction": network_holdout_fraction,
            "peer_signal_control_fraction": peer_signal_control_fraction,
            "peer_signal_utility_enabled": peer_signal_utility_enabled,
            "body_affect_coupling_enabled": body_affect_coupling_enabled,
            "llm_intent_provider": llm_intent_provider,
            "llm_intent_model": llm_intent_model,
            "llm_intent_timeout_seconds": llm_intent_timeout_seconds,
        }
        for seed in seeds
    ]
    if workers <= 1 or len(run_kwargs) <= 1:
        pairs = [_run_single_seed_from_kwargs(kwargs) for kwargs in run_kwargs]
    else:
        pairs = _collect_parallel_seed_outputs(
            run_kwargs=run_kwargs,
            workers=workers,
            worker_fn=_run_single_seed_from_kwargs,
            seed_timeout_seconds=seed_timeout_seconds,
        )
    runtimes = [runtime for runtime, _ in pairs]
    results = [result for _, result in pairs]
    return runtimes, tuple(results)


def run_seed_payload_batch(
    *,
    seeds: tuple[int, ...],
    population_size: int,
    ticks: int,
    scenario_names: tuple[str, ...] = (),
    with_lantern_scout: bool = False,
    with_non_company_artifact: bool = False,
    with_placebo_artifact: bool = False,
    detectors_during_run: bool = False,
    state_hash_mode: str = "merkle",
    network_enabled: bool = True,
    artifact_initial_access_fraction: float = 0.25,
    max_social_out_degree: int = 8,
    network_holdout_fraction: float = 0.25,
    peer_signal_control_fraction: float = 0.5,
    peer_signal_utility_enabled: bool = True,
    body_affect_coupling_enabled: bool = True,
    llm_intent_provider: str = "heuristic",
    llm_intent_model: str = DEFAULT_LLM_INTENT_STANDARD_MODEL,
    llm_intent_timeout_seconds: float = 20.0,
    include_event_log: bool = False,
    workers: int = 1,
    seed_timeout_seconds: float | None = None,
) -> tuple[SingleRunPayload, ...]:
    run_kwargs = [
        {
            "seed": seed,
            "population_size": population_size,
            "ticks": ticks,
            "scenario_names": scenario_names,
            "with_lantern_scout": with_lantern_scout,
            "with_non_company_artifact": with_non_company_artifact,
            "with_placebo_artifact": with_placebo_artifact,
            "detectors_during_run": detectors_during_run,
            "state_hash_mode": state_hash_mode,
            "network_enabled": network_enabled,
            "artifact_initial_access_fraction": artifact_initial_access_fraction,
            "max_social_out_degree": max_social_out_degree,
            "network_holdout_fraction": network_holdout_fraction,
            "peer_signal_control_fraction": peer_signal_control_fraction,
            "peer_signal_utility_enabled": peer_signal_utility_enabled,
            "body_affect_coupling_enabled": body_affect_coupling_enabled,
            "llm_intent_provider": llm_intent_provider,
            "llm_intent_model": llm_intent_model,
            "llm_intent_timeout_seconds": llm_intent_timeout_seconds,
            "include_event_log": include_event_log,
        }
        for seed in seeds
    ]
    if workers <= 1 or len(run_kwargs) <= 1:
        payloads = [_run_single_seed_payload_from_kwargs(dict(kwargs)) for kwargs in run_kwargs]
    else:
        payloads = _collect_parallel_seed_outputs(
            run_kwargs=run_kwargs,
            workers=workers,
            worker_fn=_run_single_seed_payload_from_kwargs,
            seed_timeout_seconds=seed_timeout_seconds,
        )
    return tuple(payloads)


def build_robustness_report(
    *,
    seeds: tuple[int, ...],
    population_size: int,
    ticks: int,
    threshold: float = 0.1,
) -> RobustnessResult:
    return audit_artifact_diffusion_across_seeds(
        seeds=seeds,
        population_size=population_size,
        ticks=ticks,
        threshold=threshold,
    )


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(canonicalize(value), indent=2, sort_keys=True), encoding="utf-8")
