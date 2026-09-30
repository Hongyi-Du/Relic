"""Scenario matrix runner for Society-Core evidence packs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .construct_validity import run_construct_validity_suite
from .controls import ArmControlReport, build_arm_control_report_from_seed_reports
from .experiments import (
    SeedLevelUncertaintyReport,
    SingleRunPayload,
    SingleRunResult,
    build_seed_level_uncertainty_report,
    run_seed_payload_batch,
    write_json,
)
from .hashing import stable_hash


@dataclass(frozen=True)
class ScenarioMatrixArm:
    arm_id: str
    scenario_names: tuple[str, ...] = ()
    with_lantern_scout: bool = False
    with_non_company_artifact: bool = False
    with_placebo_artifact: bool = False


@dataclass(frozen=True)
class ScenarioMatrixArmReport:
    arm_id: str
    scenario_names: tuple[str, ...]
    with_lantern_scout: bool
    with_non_company_artifact: bool
    with_placebo_artifact: bool
    run_count: int
    seeds: tuple[int, ...]
    output_path: str
    uncertainty_report_hash: str
    control_report_hash: str
    primary_metric_intervals: dict[str, dict | None]
    control_summaries: dict[str, dict]
    single_run_hashes: dict[str, dict[str, str]]


@dataclass(frozen=True)
class ScenarioMatrixReport:
    report_id: str
    matrix_version: str
    event_log_mode: str
    state_hash_mode: str
    network_enabled: bool
    artifact_initial_access_fraction: float
    max_social_out_degree: int
    network_holdout_fraction: float
    peer_signal_control_fraction: float
    peer_signal_utility_enabled: bool
    body_affect_coupling_enabled: bool
    seed_timeout_seconds: float | None
    workers: int
    population_size: int
    ticks: int
    seeds: tuple[int, ...]
    arm_reports: tuple[ScenarioMatrixArmReport, ...]
    construct_validity_report_hash: str

    def hash(self) -> str:
        return stable_hash(self)


PRIMARY_MATRIX_METRICS = (
    "artifact_diffusion:lantern_scout",
    "artifact_diffusion:community_checklist",
    "artifact_diffusion:placebo_tool",
    "artifact_payment:lantern_scout",
    "artifact_payment:community_checklist",
    "artifact_payment:placebo_tool",
    "attention_cluster",
    "norm_emergence",
    "institution_emergence",
    "trust_shift",
    "organizational_capability_l1_plus",
    "organizational_capability_l2_plus",
    "organizational_capability_l3_plus",
    "organizational_capability_transfer_l4",
)


def default_scenario_matrix() -> tuple[ScenarioMatrixArm, ...]:
    return (
        ScenarioMatrixArm("baseline"),
        ScenarioMatrixArm("resource_shock", scenario_names=("resource_shock",)),
        ScenarioMatrixArm("rumor", scenario_names=("rumor",)),
        ScenarioMatrixArm("public_conflict", scenario_names=("public_conflict",)),
        ScenarioMatrixArm("health_shock", scenario_names=("health_shock",)),
        ScenarioMatrixArm("non_company_artifact_only", with_non_company_artifact=True),
        ScenarioMatrixArm("placebo_artifact_only", with_placebo_artifact=True),
        ScenarioMatrixArm("lantern_scout_only", with_lantern_scout=True),
        ScenarioMatrixArm(
            "lantern_plus_non_company",
            with_lantern_scout=True,
            with_non_company_artifact=True,
        ),
        ScenarioMatrixArm(
            "lantern_plus_placebo",
            with_lantern_scout=True,
            with_placebo_artifact=True,
        ),
        ScenarioMatrixArm(
            "all_mixed",
            scenario_names=("resource_shock", "rumor", "public_conflict", "health_shock"),
            with_lantern_scout=True,
            with_non_company_artifact=True,
        ),
        ScenarioMatrixArm(
            "all_mixed_with_placebo",
            scenario_names=("resource_shock", "rumor", "public_conflict", "health_shock"),
            with_lantern_scout=True,
            with_non_company_artifact=True,
            with_placebo_artifact=True,
        ),
    )


def _run_summary(result: SingleRunResult) -> dict:
    return {
        "agents": result.population_size,
        "ticks": result.ticks,
        "seed": result.seed,
        "scenario_names": result.scenario_names,
        "with_lantern_scout": result.with_lantern_scout,
        "with_non_company_artifact": result.with_non_company_artifact,
        "with_placebo_artifact": result.with_placebo_artifact,
        "event_log_hash": result.event_log_hash,
        "state_hash": result.state_hash,
        "state_hash_mode": result.manifest.state_hash_mode,
        "events": result.event_count,
        "detector_reports": result.detector_reports,
    }


def _interval_summary(report: SeedLevelUncertaintyReport, metric_name: str) -> dict | None:
    interval = report.metric_intervals.get(metric_name)
    if interval is None:
        return None
    return {
        "mean": interval.mean,
        "ci_low": interval.ci_low,
        "ci_high": interval.ci_high,
        "sample_sd": interval.sample_sd,
        "missing_seed_count": interval.missing_seed_count,
        "caveats": interval.caveats,
    }


def _control_summary(report: ArmControlReport) -> dict[str, dict]:
    return {
        artifact_id: {
            "trial_rate": artifact_report.trial_rate,
            "trial_actor_count": artifact_report.trial_actor_count,
            "payment_rate": artifact_report.payment_rate,
            "payment_actor_count": artifact_report.payment_actor_count,
            "matched_control_count": artifact_report.matched_control_count,
            "max_abs_standardized_mean_difference": artifact_report.max_abs_standardized_mean_difference,
            "release_exposure_count": artifact_report.release_exposure_count,
            "network_exposure_count": artifact_report.network_exposure_count,
            "network_trial_actor_count": artifact_report.network_trial_actor_count,
            "network_payment_actor_count": artifact_report.network_payment_actor_count,
            "mean_network_distance": artifact_report.mean_network_distance,
            "exposure_source_counts": artifact_report.exposure_source_counts,
            "randomized_control_design": artifact_report.randomized_control_design,
            "network_eligible_count": artifact_report.network_eligible_count,
            "network_holdout_count": artifact_report.network_holdout_count,
            "network_eligible_trial_rate": artifact_report.network_eligible_trial_rate,
            "network_holdout_trial_rate": artifact_report.network_holdout_trial_rate,
            "network_intent_to_treat_lift": artifact_report.network_intent_to_treat_lift,
            "randomized_group_max_abs_smd": artifact_report.randomized_group_max_abs_smd,
            "network_diffusion_claim_allowed": artifact_report.network_diffusion_claim_allowed,
            "peer_signal_design": artifact_report.peer_signal_design,
            "peer_signal_eligible_count": artifact_report.peer_signal_eligible_count,
            "access_only_control_count": artifact_report.access_only_control_count,
            "peer_signal_trial_rate": artifact_report.peer_signal_trial_rate,
            "access_only_trial_rate": artifact_report.access_only_trial_rate,
            "peer_signal_intent_to_treat_lift": artifact_report.peer_signal_intent_to_treat_lift,
            "peer_signal_payment_rate": artifact_report.peer_signal_payment_rate,
            "access_only_payment_rate": artifact_report.access_only_payment_rate,
            "peer_signal_payment_intent_to_treat_lift": artifact_report.peer_signal_payment_intent_to_treat_lift,
            "peer_signal_group_max_abs_smd": artifact_report.peer_signal_group_max_abs_smd,
            "peer_signal_trial_actor_count": artifact_report.peer_signal_trial_actor_count,
            "peer_signal_payment_actor_count": artifact_report.peer_signal_payment_actor_count,
            "peer_signal_randomization_p_value": artifact_report.peer_signal_randomization_p_value,
            "peer_signal_confidence_low": artifact_report.peer_signal_confidence_low,
            "peer_signal_confidence_high": artifact_report.peer_signal_confidence_high,
            "peer_signal_power_proxy": artifact_report.peer_signal_power_proxy,
            "peer_signal_payment_randomization_p_value": artifact_report.peer_signal_payment_randomization_p_value,
            "peer_signal_payment_confidence_low": artifact_report.peer_signal_payment_confidence_low,
            "peer_signal_payment_confidence_high": artifact_report.peer_signal_payment_confidence_high,
            "peer_signal_payment_power_proxy": artifact_report.peer_signal_payment_power_proxy,
            "early_window_ticks": artifact_report.early_window_ticks,
            "peer_signal_early_trial_rate": artifact_report.peer_signal_early_trial_rate,
            "access_only_early_trial_rate": artifact_report.access_only_early_trial_rate,
            "peer_signal_early_trial_intent_to_treat_lift": artifact_report.peer_signal_early_trial_intent_to_treat_lift,
            "peer_signal_early_payment_rate": artifact_report.peer_signal_early_payment_rate,
            "access_only_early_payment_rate": artifact_report.access_only_early_payment_rate,
            "peer_signal_early_payment_intent_to_treat_lift": artifact_report.peer_signal_early_payment_intent_to_treat_lift,
            "peer_signal_early_trial_actor_count": artifact_report.peer_signal_early_trial_actor_count,
            "peer_signal_early_payment_actor_count": artifact_report.peer_signal_early_payment_actor_count,
            "peer_signal_early_trial_randomization_p_value": artifact_report.peer_signal_early_trial_randomization_p_value,
            "peer_signal_early_trial_confidence_low": artifact_report.peer_signal_early_trial_confidence_low,
            "peer_signal_early_trial_confidence_high": artifact_report.peer_signal_early_trial_confidence_high,
            "peer_signal_early_trial_power_proxy": artifact_report.peer_signal_early_trial_power_proxy,
            "peer_signal_early_payment_randomization_p_value": artifact_report.peer_signal_early_payment_randomization_p_value,
            "peer_signal_early_payment_confidence_low": artifact_report.peer_signal_early_payment_confidence_low,
            "peer_signal_early_payment_confidence_high": artifact_report.peer_signal_early_payment_confidence_high,
            "peer_signal_early_payment_power_proxy": artifact_report.peer_signal_early_payment_power_proxy,
            "peer_signal_mean_first_trial_tick": artifact_report.peer_signal_mean_first_trial_tick,
            "access_only_mean_first_trial_tick": artifact_report.access_only_mean_first_trial_tick,
            "peer_signal_mean_first_payment_tick": artifact_report.peer_signal_mean_first_payment_tick,
            "access_only_mean_first_payment_tick": artifact_report.access_only_mean_first_payment_tick,
            "peer_early_trial_claim_allowed": artifact_report.peer_early_trial_claim_allowed,
            "peer_early_payment_claim_allowed": artifact_report.peer_early_payment_claim_allowed,
            "peer_trial_claim_allowed": artifact_report.peer_trial_claim_allowed,
            "peer_payment_claim_allowed": artifact_report.peer_payment_claim_allowed,
            "peer_influence_claim_allowed": artifact_report.peer_influence_claim_allowed,
            "influence_claim_allowed": artifact_report.influence_claim_allowed,
            "caveats": artifact_report.caveats,
        }
        for artifact_id, artifact_report in sorted(report.artifact_reports.items())
    }


def _write_arm_outputs(
    *,
    arm_dir: Path,
    payloads: tuple[SingleRunPayload, ...],
    results: tuple[SingleRunResult, ...],
    uncertainty_report: SeedLevelUncertaintyReport,
    control_report: ArmControlReport,
    write_event_logs: bool,
) -> None:
    arm_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = arm_dir / "runs"
    for payload, result in zip(payloads, results, strict=True):
        seed_dir = runs_dir / f"seed_{result.seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        write_json(seed_dir / "manifest.json", result.manifest)
        write_json(seed_dir / "summary.json", _run_summary(result))
        write_json(seed_dir / "artifact_control_reports.json", payload.artifact_control_reports)
        if write_event_logs:
            write_json(seed_dir / "event_log.json", payload.event_log)
    write_json(arm_dir / "seed_runs.json", tuple(_run_summary(result) for result in results))
    write_json(arm_dir / "uncertainty_report.json", uncertainty_report)
    write_json(arm_dir / "control_report.json", control_report)


def run_scenario_matrix(
    *,
    seeds: tuple[int, ...],
    population_size: int,
    ticks: int,
    output_dir: Path,
    arms: tuple[ScenarioMatrixArm, ...] | None = None,
    event_log_mode: str = "hash_only",
    state_hash_mode: str = "merkle",
    network_enabled: bool = True,
    artifact_initial_access_fraction: float = 0.25,
    max_social_out_degree: int = 8,
    network_holdout_fraction: float = 0.25,
    peer_signal_control_fraction: float = 0.5,
    peer_signal_utility_enabled: bool = True,
    body_affect_coupling_enabled: bool = True,
    workers: int = 1,
    seed_timeout_seconds: float | None = None,
) -> ScenarioMatrixReport:
    if not seeds:
        raise ValueError("scenario matrix requires at least one seed")
    if event_log_mode not in {"full", "hash_only"}:
        raise ValueError(f"unsupported event_log_mode: {event_log_mode}")
    arms = arms or default_scenario_matrix()
    output_dir.mkdir(parents=True, exist_ok=True)
    construct_validity_report = run_construct_validity_suite()
    write_json(output_dir / "construct_validity_report.json", construct_validity_report)

    arm_reports: list[ScenarioMatrixArmReport] = []
    for arm in arms:
        arm_dir = output_dir / "arms" / arm.arm_id
        payloads = run_seed_payload_batch(
            seeds=seeds,
            population_size=population_size,
            ticks=ticks,
            scenario_names=arm.scenario_names,
            with_lantern_scout=arm.with_lantern_scout,
            with_non_company_artifact=arm.with_non_company_artifact,
            with_placebo_artifact=arm.with_placebo_artifact,
            state_hash_mode=state_hash_mode,
            network_enabled=network_enabled,
            artifact_initial_access_fraction=artifact_initial_access_fraction,
            max_social_out_degree=max_social_out_degree,
            network_holdout_fraction=network_holdout_fraction,
            peer_signal_control_fraction=peer_signal_control_fraction,
            peer_signal_utility_enabled=peer_signal_utility_enabled,
            body_affect_coupling_enabled=body_affect_coupling_enabled,
            include_event_log=event_log_mode == "full",
            workers=workers,
            seed_timeout_seconds=seed_timeout_seconds,
        )
        results = tuple(payload.result for payload in payloads)
        uncertainty_report = build_seed_level_uncertainty_report(results)
        control_report = build_arm_control_report_from_seed_reports(
            arm_id=arm.arm_id,
            seed_artifact_reports=tuple(payload.artifact_control_reports for payload in payloads),
        )
        _write_arm_outputs(
            arm_dir=arm_dir,
            payloads=payloads,
            results=results,
            uncertainty_report=uncertainty_report,
            control_report=control_report,
            write_event_logs=event_log_mode == "full",
        )
        arm_reports.append(
            ScenarioMatrixArmReport(
                arm_id=arm.arm_id,
                scenario_names=arm.scenario_names,
                with_lantern_scout=arm.with_lantern_scout,
                with_non_company_artifact=arm.with_non_company_artifact,
                with_placebo_artifact=arm.with_placebo_artifact,
                run_count=len(results),
                seeds=tuple(result.seed for result in results),
                output_path=str(arm_dir),
                uncertainty_report_hash=uncertainty_report.hash(),
                control_report_hash=control_report.hash(),
                primary_metric_intervals={
                    metric_name: _interval_summary(uncertainty_report, metric_name)
                    for metric_name in PRIMARY_MATRIX_METRICS
                },
                control_summaries=_control_summary(control_report),
                single_run_hashes={
                    str(result.seed): {
                        "manifest_hash": stable_hash(result.manifest),
                        "event_log_hash": result.event_log_hash,
                        "state_hash": result.state_hash,
                    }
                    for result in results
                },
            )
        )
    report = ScenarioMatrixReport(
        report_id="society_core_scenario_matrix_v15",
        matrix_version="v15",
        event_log_mode=event_log_mode,
        state_hash_mode=state_hash_mode,
        network_enabled=network_enabled,
        artifact_initial_access_fraction=artifact_initial_access_fraction,
        max_social_out_degree=max_social_out_degree,
        network_holdout_fraction=network_holdout_fraction,
        peer_signal_control_fraction=peer_signal_control_fraction,
        peer_signal_utility_enabled=peer_signal_utility_enabled,
        body_affect_coupling_enabled=body_affect_coupling_enabled,
        seed_timeout_seconds=seed_timeout_seconds,
        workers=workers,
        population_size=population_size,
        ticks=ticks,
        seeds=seeds,
        arm_reports=tuple(arm_reports),
        construct_validity_report_hash=construct_validity_report.hash(),
    )
    write_json(output_dir / "matrix_report.json", report)
    write_json(
        output_dir / "matrix_index.json",
        {
            "report_hash": report.hash(),
            "matrix_version": report.matrix_version,
            "arm_count": len(arm_reports),
            "arms": tuple(arm.arm_id for arm in arms),
            "event_log_mode": event_log_mode,
            "state_hash_mode": state_hash_mode,
            "network_enabled": network_enabled,
            "artifact_initial_access_fraction": artifact_initial_access_fraction,
            "max_social_out_degree": max_social_out_degree,
            "network_holdout_fraction": network_holdout_fraction,
            "peer_signal_control_fraction": peer_signal_control_fraction,
            "peer_signal_utility_enabled": peer_signal_utility_enabled,
            "body_affect_coupling_enabled": body_affect_coupling_enabled,
            "seed_timeout_seconds": seed_timeout_seconds,
            "workers": workers,
            "seeds": seeds,
        },
    )
    return report
