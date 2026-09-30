"""End-to-end historical product experiments."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from time import monotonic
from typing import Any

from .company_development import (
    CompanyDevelopmentRun,
    DEFAULT_COMPANY_CODING_MODEL,
    apply_and_verify_project_edits,
    build_company_coding_agent,
    build_workspace_manifest,
)
from .code_landing.event_store import SQLiteEventStore
from .code_landing.environment import build_workspace_execution_profile
from .code_landing.evidence import evidence_level_at_least
from .execution import CommandExecutor
from .company_optimizer import DEFAULT_COMPANY_OPTIMIZER_MODEL, build_company_optimizer
from .company_safety import (
    CompanySafetyReview,
    DEFAULT_COMPANY_SAFETY_MODEL,
    build_company_safety_review_agent,
)
from .claim_evidence import (
    assess_evidence_debts,
    build_evaluation_freeze_boundary,
    build_society_claim_certificate,
    select_weakest_public_witness,
)
from .hashing import canonicalize, stable_hash
from .historical_products import (
    HistoricalProductCase,
    gitingest_2025_case,
    gitingest_2025_v015_to_v030_case,
    snowpack_2020_case,
    vite_2020_case,
)
from .llm_intent import (
    DEFAULT_LLM_INTENT_HIGH_MODEL,
    DEFAULT_LLM_INTENT_LOW_MODEL,
    DEFAULT_LLM_INTENT_MAX_OUTPUT_TOKENS,
    DEFAULT_LLM_INTENT_REQUEST_ATTEMPTS,
    DEFAULT_LLM_INTENT_STANDARD_MODEL,
)
from .org_adapter import (
    CompanyOptimizationProposal,
    build_company_public_feedback_report,
    extract_company_visible_public_traces,
)
from .organizational_capabilities import CapabilityLedger, build_capability_ledger
from .profiles import (
    DEFAULT_INNOVATION_EARLY_BUILDER_FRACTION,
    DEFAULT_INNOVATION_ORIGINATOR_FRACTION,
    DEFAULT_INNOVATION_PRAGMATIC_ADAPTER_FRACTION,
    DEFAULT_TECHNICAL_EXPERT_FRACTION,
    DEFAULT_TECHNICAL_PRACTITIONER_FRACTION,
)
from .real_product_experience import build_real_product_experience_harness
from .runtime import SocietyConfig, SocietyRuntime
from .time_machine_evaluation import (
    TimeMachineEvaluationPlan,
    TimeMachineEvaluationResult,
    attach_time_machine_evidence,
    build_time_machine_evaluation_plan,
    evaluate_time_machine_candidate,
)
from .workspace_agent import (
    CodingAgentLoopResult,
    DEFAULT_WORKSPACE_CODING_EXPLORATION_ITERATIONS,
    DEFAULT_WORKSPACE_CODING_MAX_ITERATIONS,
    DEFAULT_WORKSPACE_CODING_MAX_SELECTED_FILES,
    DEFAULT_WORKSPACE_CODING_MODEL,
    build_development_intent_specs,
    build_workspace_coding_agent,
    run_coding_agent_development_loop,
)

TIME_MACHINE_CONTRACT_VISIBILITIES = frozenset(
    {"agent_visible", "evaluator_only"}
)


@dataclass(frozen=True)
class HistoricalExperimentConfig:
    case_id: str = "vite_2020_initial_to_v2"
    population_size: int = 200
    seed: int = 2020
    simulated_days: int = 7
    ticks_per_day: int = 1
    artifact_initial_access_fraction: float = 0.0
    technical_expert_fraction: float = DEFAULT_TECHNICAL_EXPERT_FRACTION
    technical_practitioner_fraction: float = (
        DEFAULT_TECHNICAL_PRACTITIONER_FRACTION
    )
    innovation_originator_fraction: float = DEFAULT_INNOVATION_ORIGINATOR_FRACTION
    innovation_early_builder_fraction: float = (
        DEFAULT_INNOVATION_EARLY_BUILDER_FRACTION
    )
    innovation_pragmatic_adapter_fraction: float = (
        DEFAULT_INNOVATION_PRAGMATIC_ADAPTER_FRACTION
    )
    llm_intent_provider: str = "heuristic"
    llm_intent_model: str = DEFAULT_LLM_INTENT_STANDARD_MODEL
    llm_intent_low_model: str | None = DEFAULT_LLM_INTENT_LOW_MODEL
    llm_intent_standard_model: str | None = DEFAULT_LLM_INTENT_STANDARD_MODEL
    llm_intent_high_model: str | None = DEFAULT_LLM_INTENT_HIGH_MODEL
    llm_intent_timeout_seconds: float = 20.0
    llm_intent_request_attempts: int = DEFAULT_LLM_INTENT_REQUEST_ATTEMPTS
    llm_intent_max_output_tokens: int = DEFAULT_LLM_INTENT_MAX_OUTPUT_TOKENS
    llm_intent_response_cache_dir: str | None = None
    llm_intent_audit_path: str | None = None
    llm_intent_progress_path: str | None = None
    llm_intent_retry_backoff_seconds: float = 0.0
    llm_intent_replay_reference_audit_path: str | None = None
    llm_intent_max_concurrency: int = 1
    require_live_llm_product_experience: bool = False
    society_network_enabled: bool = True
    company_optimizer_provider: str = "heuristic"
    company_optimizer_model: str = DEFAULT_COMPANY_OPTIMIZER_MODEL
    company_optimizer_timeout_seconds: float = 300.0
    company_safety_provider: str = "heuristic"
    company_safety_model: str = DEFAULT_COMPANY_SAFETY_MODEL
    company_safety_timeout_seconds: float = 300.0
    company_coding_agent_provider: str = "heuristic"
    company_coding_agent_model: str = DEFAULT_COMPANY_CODING_MODEL
    company_coding_agent_timeout_seconds: float = 300.0
    require_live_company_agents: bool = False
    company_agent_request_attempts: int = 2
    company_agent_retry_backoff_seconds: float = 0.0
    company_workspace_root: str | None = None
    real_product_experience_enabled: bool = False
    real_product_experience_workspace_root: str | None = None
    real_product_experience_timeout_seconds: float = 20.0
    real_product_experience_max_tasks: int = 8
    real_product_experience_observation_cache_path: str | None = None
    real_product_experience_evidence_context_hash: str = ""
    real_product_experience_require_natural_journey_tasks: bool = False
    evidence_only_upgrade_suggestions: bool = False
    apply_company_development: bool = False
    allow_company_replacements: bool = False
    guarded_company_replacements: bool = True
    verify_company_development: bool = True
    company_verification_timeout_seconds: float = 120.0
    workspace_coding_agent_enabled: bool = False
    workspace_coding_agent_provider: str = "heuristic"
    workspace_coding_agent_model: str = DEFAULT_WORKSPACE_CODING_MODEL
    workspace_coding_agent_timeout_seconds: float = 300.0
    workspace_coding_agent_verification_timeout_seconds: float = 120.0
    workspace_coding_agent_max_iterations: int = DEFAULT_WORKSPACE_CODING_MAX_ITERATIONS
    workspace_coding_agent_exploration_iterations: int = (
        DEFAULT_WORKSPACE_CODING_EXPLORATION_ITERATIONS
    )
    workspace_coding_agent_max_selected_files: int = (
        DEFAULT_WORKSPACE_CODING_MAX_SELECTED_FILES
    )
    workspace_coding_agent_include_agent_verification_commands: bool = True
    workspace_coding_agent_verification_commands: tuple[str, ...] = ()
    workspace_coding_agent_enforce_spec_maturity_gate: bool = False
    workspace_coding_agent_candidate_count: int = 1
    workspace_coding_agent_event_attempt: int = 1
    capability_transfer_probe_enabled: bool = False
    capability_transfer_probe_verification_commands: tuple[str, ...] = ()
    capability_transfer_probe_timeout_seconds: float = 120.0
    capability_transfer_probe_max_iterations: int = (
        DEFAULT_WORKSPACE_CODING_MAX_ITERATIONS
    )
    capability_transfer_probe_max_selected_files: int = (
        DEFAULT_WORKSPACE_CODING_MAX_SELECTED_FILES
    )
    capability_transfer_probe_include_agent_verification_commands: bool = True
    time_machine_evaluation_dataset_id: str | None = None
    time_machine_evaluation_timeout_seconds: int = 120
    time_machine_public_contract_visibility: str = "agent_visible"
    max_company_visible_traces: int = 1000


@dataclass(frozen=True)
class HistoricalExperimentReport:
    report_id: str
    case_id: str
    product_name: str
    source_urls: tuple[str, ...]
    population_size: int
    seed: int
    simulated_days: int
    ticks_per_day: int
    panel_experienced_agents: int
    agents_with_experience: int
    paid_agents: int
    public_trace_count: int
    real_elapsed_days: int
    proposed_direction: str
    proposed_themes: tuple[str, ...]
    historical_target_overlap: float | None
    matched_target_themes: tuple[str, ...]
    missed_target_themes: tuple[str, ...]
    extra_proposed_themes: tuple[str, ...]
    verified_patch_historical_target_overlap: float | None
    verified_patch_matched_target_themes: tuple[str, ...]
    verified_patch_missed_target_themes: tuple[str, ...]
    verified_patch_extra_themes: tuple[str, ...]
    behavior_verified_historical_target_overlap: float | None
    behavior_verified_matched_target_themes: tuple[str, ...]
    behavior_verified_missed_target_themes: tuple[str, ...]
    profile_tier_counts: dict[str, dict[str, int]]
    experience_summary: dict[str, Any]
    real_product_experience_summary: dict[str, Any]
    llm_product_experience_summary: dict[str, Any]
    upgrade_suggestion_summary: dict[str, Any]
    payment_summary: dict[str, int]
    public_feedback_summary: dict[str, Any]
    demand_coverage: dict[str, int]
    community_coverage: dict[str, int]
    development_status: dict[str, Any]
    workspace_development_status: dict[str, Any]
    closed_loop_evidence_status: dict[str, Any]
    llm_intent_tier_summary: dict[str, Any]
    provider_provenance: dict[str, Any]
    paper_evidence_certificate: dict[str, Any]
    capability_transfer_status: dict[str, Any]
    organizational_capability_summary: dict[str, Any]
    organizational_capability_ledger: CapabilityLedger
    proposal: CompanyOptimizationProposal
    safety_review: CompanySafetyReview
    development_run: CompanyDevelopmentRun
    workspace_development_run: CodingAgentLoopResult | None
    capability_transfer_run: CodingAgentLoopResult | None
    time_machine_evaluation: dict[str, Any]
    state_hash: str
    started_at: str
    completed_at: str
    elapsed_sec: float


def historical_case_by_id(case_id: str) -> HistoricalProductCase:
    if case_id == "vite_2020_initial_to_v2":
        return vite_2020_case()
    if case_id == "snowpack_2020_v2_to_v3":
        return snowpack_2020_case()
    if case_id == "gitingest_2025_v015_to_v020":
        return gitingest_2025_case()
    if case_id == "gitingest_2025_v015_to_v030":
        return gitingest_2025_v015_to_v030_case()
    raise ValueError(f"Unsupported historical product case: {case_id}")


def _validate_live_llm_product_experience_config(
    config: HistoricalExperimentConfig,
) -> None:
    if not config.require_live_llm_product_experience:
        return
    if config.llm_intent_provider.lower() != "openai":
        raise ValueError(
            "require_live_llm_product_experience requires llm_intent_provider=openai"
        )
    if not config.real_product_experience_enabled:
        raise ValueError(
            "require_live_llm_product_experience requires real_product_experience_enabled"
        )
    tier_models = (
        config.llm_intent_low_model,
        config.llm_intent_standard_model,
        config.llm_intent_high_model,
    )
    normalized = tuple(model.strip() for model in tier_models if model and model.strip())
    if len(normalized) != 3 or len(set(normalized)) != 3:
        raise ValueError(
            "require_live_llm_product_experience requires three distinct explicit tier models"
        )


def run_historical_product_experiment(
    config: HistoricalExperimentConfig | None = None,
    *,
    output_dir: Path | None = None,
    executor: CommandExecutor | None = None,
    event_store: SQLiteEventStore | None = None,
) -> HistoricalExperimentReport:
    started_monotonic = monotonic()
    started_at = _utc_now_iso()
    config = config or HistoricalExperimentConfig()
    _validate_live_llm_product_experience_config(config)
    _validate_time_machine_contract_visibility(config)
    _validate_live_company_agent_config(config)
    if config.apply_company_development and config.workspace_coding_agent_enabled:
        raise ValueError(
            "apply_company_development cannot be combined with the transactional "
            "workspace coding agent"
        )
    if config.apply_company_development and not config.verify_company_development:
        raise ValueError(
            "unverified_company_workspace_mutation_disabled; enable "
            "verify_company_development"
        )
    execution_required = any(
        (
            config.real_product_experience_enabled,
            config.workspace_coding_agent_enabled,
            config.capability_transfer_probe_enabled,
            config.apply_company_development,
            bool(config.time_machine_evaluation_dataset_id),
        )
    )
    if execution_required and executor is None:
        raise ValueError("historical_experiment_executor_required")
    command_executor = executor
    case = historical_case_by_id(config.case_id)
    time_machine_plan = _prepare_time_machine_plan(
        config,
        executor=command_executor,
    )
    runtime = SocietyRuntime(
        SocietyConfig(
            population_size=config.population_size,
            seed=config.seed,
            artifact_initial_access_fraction=config.artifact_initial_access_fraction,
            llm_intent_provider=config.llm_intent_provider,
            llm_intent_model=config.llm_intent_model,
            llm_intent_low_model=config.llm_intent_low_model,
            llm_intent_standard_model=config.llm_intent_standard_model,
            llm_intent_high_model=config.llm_intent_high_model,
            llm_intent_timeout_seconds=config.llm_intent_timeout_seconds,
            llm_intent_request_attempts=config.llm_intent_request_attempts,
            llm_intent_max_output_tokens=config.llm_intent_max_output_tokens,
            llm_intent_strict_live=(
                config.require_live_llm_product_experience
            ),
            llm_intent_response_cache_dir=(
                config.llm_intent_response_cache_dir
            ),
            llm_intent_audit_path=config.llm_intent_audit_path,
            llm_intent_progress_path=config.llm_intent_progress_path,
            llm_intent_retry_backoff_seconds=(
                config.llm_intent_retry_backoff_seconds
            ),
            llm_intent_replay_reference_audit_path=(
                config.llm_intent_replay_reference_audit_path
            ),
            llm_intent_max_concurrency=(
                config.llm_intent_max_concurrency
            ),
            evidence_only_upgrade_suggestions=(
                config.evidence_only_upgrade_suggestions
            ),
            network_enabled=config.society_network_enabled,
            technical_expert_fraction=config.technical_expert_fraction,
            technical_practitioner_fraction=(
                config.technical_practitioner_fraction
            ),
            innovation_originator_fraction=config.innovation_originator_fraction,
            innovation_early_builder_fraction=(
                config.innovation_early_builder_fraction
            ),
            innovation_pragmatic_adapter_fraction=(
                config.innovation_pragmatic_adapter_fraction
            ),
        )
    )
    runtime.inject_artifact(case.initial_artifact)
    real_experience_harness = None
    if config.real_product_experience_enabled:
        real_workspace_root = (
            config.real_product_experience_workspace_root
            or config.company_workspace_root
        )
        if not real_workspace_root:
            raise ValueError(
                "real_product_experience_enabled requires real_product_experience_workspace_root"
            )
        real_experience_harness = build_real_product_experience_harness(
            artifact=case.initial_artifact,
            workspace_root=Path(real_workspace_root),
            timeout_seconds=config.real_product_experience_timeout_seconds,
            max_tasks=config.real_product_experience_max_tasks,
            executor=command_executor,
            observation_cache_path=(
                Path(config.real_product_experience_observation_cache_path)
                if config.real_product_experience_observation_cache_path
                else None
            ),
            evidence_context_hash=(
                config.real_product_experience_evidence_context_hash
            ),
            require_natural_journey_tasks=(
                config.real_product_experience_require_natural_journey_tasks
            ),
        )
        runtime.product_experience_packet_provider = (
            real_experience_harness.packet_for_agent
        )
    panel_count = runtime.run_product_experience_panel(
        case.initial_artifact.id,
        packet_provider=real_experience_harness.packet_for_agent
        if real_experience_harness
        else None,
    )
    real_product_experience_summary = (
        real_experience_harness.summary() if real_experience_harness else {}
    )
    panel_llm_product_experience_summary = _llm_product_experience_summary(
        agents=tuple(runtime.state.agents.values()),
        artifact_id=case.initial_artifact.id,
        event_log=tuple(runtime.state.event_log),
        objective_execution_observation_count=int(
            real_product_experience_summary.get("observation_count", 0) or 0
        ),
        objective_execution_observation_refs=tuple(
            real_product_experience_summary.get("observation_ids", ()) or ()
        ),
        objective_execution_observation_manifest_hash=str(
            real_product_experience_summary.get("observation_hash", "") or ""
        ),
    )
    if (
        config.require_live_llm_product_experience
        and not panel_llm_product_experience_summary["claim_ready"]
    ):
        missing = ",".join(
            panel_llm_product_experience_summary["missing_for_claim"]
        )
        raise RuntimeError(f"live_llm_product_experience_incomplete:{missing}")
    runtime.run(config.simulated_days * config.ticks_per_day)
    llm_product_experience_summary = _llm_product_experience_summary(
        agents=tuple(runtime.state.agents.values()),
        artifact_id=case.initial_artifact.id,
        event_log=tuple(runtime.state.event_log),
        objective_execution_observation_count=int(
            real_product_experience_summary.get("observation_count", 0) or 0
        ),
        objective_execution_observation_refs=tuple(
            real_product_experience_summary.get("observation_ids", ()) or ()
        ),
        objective_execution_observation_manifest_hash=str(
            real_product_experience_summary.get("observation_hash", "") or ""
        ),
    )
    llm_product_experience_summary["panel_summary"] = (
        panel_llm_product_experience_summary
    )
    if (
        config.require_live_llm_product_experience
        and not llm_product_experience_summary["claim_ready"]
    ):
        missing = ",".join(llm_product_experience_summary["missing_for_claim"])
        raise RuntimeError(
            f"live_llm_product_experience_incomplete_after_simulation:{missing}"
        )
    traces = extract_company_visible_public_traces(
        runtime.state,
        max_items=config.max_company_visible_traces,
    )
    published_objective_observation_refs = tuple(
        dict.fromkeys(
            ref
            for trace in traces
            for ref in trace.support_refs
            if ref.startswith("real_xp_obs_")
        )
    )
    feedback_report = build_company_public_feedback_report(traces)
    company_optimizer = build_company_optimizer(
        provider=config.company_optimizer_provider,
        model=config.company_optimizer_model,
        timeout_seconds=config.company_optimizer_timeout_seconds,
        strict_live=config.require_live_company_agents,
        request_attempts=config.company_agent_request_attempts,
        retry_backoff_seconds=config.company_agent_retry_backoff_seconds,
    )
    proposal = company_optimizer.propose(
        report=feedback_report,
        artifact_id=case.initial_artifact.id,
        public_traces=traces,
    )
    safety_agent = build_company_safety_review_agent(
        provider=config.company_safety_provider,
        model=config.company_safety_model,
        timeout_seconds=config.company_safety_timeout_seconds,
        strict_live=config.require_live_company_agents,
        request_attempts=config.company_agent_request_attempts,
        retry_backoff_seconds=config.company_agent_retry_backoff_seconds,
    )
    proposal, safety_review = safety_agent.review(
        proposal=proposal,
        report=feedback_report,
        public_traces=traces,
        artifact=case.initial_artifact,
    )
    workspace_manifest = (
        build_workspace_manifest(Path(config.company_workspace_root))
        if config.company_workspace_root
        else None
    )
    coding_agent = build_company_coding_agent(
        provider=config.company_coding_agent_provider,
        model=config.company_coding_agent_model,
        timeout_seconds=config.company_coding_agent_timeout_seconds,
        strict_live=config.require_live_company_agents,
        request_attempts=config.company_agent_request_attempts,
        retry_backoff_seconds=config.company_agent_retry_backoff_seconds,
    )
    development_run = coding_agent.develop(
        proposal=proposal,
        report=feedback_report,
        public_traces=traces,
        workspace_manifest=workspace_manifest,
    )
    if config.apply_company_development and config.company_workspace_root:
        development_run = apply_and_verify_project_edits(
            development_run,
            Path(config.company_workspace_root),
            allow_replace_existing=config.allow_company_replacements,
            guarded_replacement=config.guarded_company_replacements,
            timeout_seconds=config.company_verification_timeout_seconds,
            executor=command_executor,
        )
    workspace_development_run = None
    if config.workspace_coding_agent_enabled:
        if not config.company_workspace_root:
            raise ValueError(
                "workspace_coding_agent_enabled requires company_workspace_root"
            )
        workspace_agent = build_workspace_coding_agent(
            provider=config.workspace_coding_agent_provider,
            model=config.workspace_coding_agent_model,
            timeout_seconds=config.workspace_coding_agent_timeout_seconds,
            executor=command_executor,
            event_store=event_store,
        )
        workspace_development_run = run_coding_agent_development_loop(
            source_report=_workspace_source_report(
                config=config,
                case=case,
                feedback_report=feedback_report,
                proposal=proposal,
                safety_review=safety_review,
                development_run=development_run,
                real_product_experience_summary=real_product_experience_summary,
                time_machine_plan=time_machine_plan,
            ),
            workspace_root=Path(config.company_workspace_root),
            agent=workspace_agent,
            verification_commands=_workspace_verification_commands(
                config=config,
                time_machine_plan=time_machine_plan,
                real_experience_harness=real_experience_harness,
                published_objective_observation_refs=(
                    published_objective_observation_refs
                ),
            ),
            max_iterations=config.workspace_coding_agent_max_iterations,
            timeout_seconds=config.workspace_coding_agent_verification_timeout_seconds,
            max_selected_files=config.workspace_coding_agent_max_selected_files,
            include_agent_verification_commands=config.workspace_coding_agent_include_agent_verification_commands,
            exploration_iterations=config.workspace_coding_agent_exploration_iterations,
            enforce_spec_maturity_gate=config.workspace_coding_agent_enforce_spec_maturity_gate,
            executor=command_executor,
            candidate_count=config.workspace_coding_agent_candidate_count,
            event_store=event_store,
            event_attempt=config.workspace_coding_agent_event_attempt,
        )
    provisional_capability_ledger = build_capability_ledger(
        case_id=case.case_id,
        artifact_id=case.initial_artifact.id,
        proposal=proposal,
        safety_review=safety_review,
        development_run=development_run,
        workspace_development_run=workspace_development_run,
        public_trace_count=len(traces),
        real_product_experience_summary=real_product_experience_summary,
    )
    capability_transfer_run = None
    transfer_themes = tuple(
        capability.theme
        for capability in provisional_capability_ledger.capabilities
        if capability.kind == "organizational"
        and capability.criteria.get("adoption")
        and any(
            event.capability_id == capability.capability_id
            and event.stage == "materialization"
            for event in provisional_capability_ledger.events
        )
    )
    if config.capability_transfer_probe_enabled and transfer_themes:
        if not config.company_workspace_root:
            raise ValueError(
                "capability_transfer_probe_enabled requires company_workspace_root"
            )
        transfer_agent = build_workspace_coding_agent(
            provider=config.workspace_coding_agent_provider,
            model=config.workspace_coding_agent_model,
            timeout_seconds=config.workspace_coding_agent_timeout_seconds,
            executor=command_executor,
            event_store=event_store,
        )
        capability_transfer_run = run_coding_agent_development_loop(
            source_report=_capability_transfer_source_report(
                config=config,
                case=case,
                proposal=proposal,
                prior_ledger=provisional_capability_ledger,
                transfer_themes=transfer_themes,
            ),
            workspace_root=Path(config.company_workspace_root),
            agent=transfer_agent,
            verification_commands=config.capability_transfer_probe_verification_commands,
            max_iterations=config.capability_transfer_probe_max_iterations,
            timeout_seconds=config.capability_transfer_probe_timeout_seconds,
            max_selected_files=config.capability_transfer_probe_max_selected_files,
            include_agent_verification_commands=config.capability_transfer_probe_include_agent_verification_commands,
            exploration_iterations=0,
            enforce_spec_maturity_gate=config.workspace_coding_agent_enforce_spec_maturity_gate,
            executor=command_executor,
            candidate_count=config.workspace_coding_agent_candidate_count,
            event_store=event_store,
            event_attempt=config.workspace_coding_agent_event_attempt,
        )
    time_machine_result = None
    if time_machine_plan is not None:
        time_machine_result = evaluate_time_machine_candidate(
            time_machine_plan,
            Path(str(config.company_workspace_root)),
            timeout_seconds=config.time_machine_evaluation_timeout_seconds,
            executor=command_executor,
        )
        if capability_transfer_run is not None:
            capability_transfer_run = attach_time_machine_evidence(
                capability_transfer_run,
                time_machine_result,
            )
        elif workspace_development_run is not None:
            workspace_development_run = attach_time_machine_evidence(
                workspace_development_run,
                time_machine_result,
            )
    completed_at = _utc_now_iso()
    elapsed_sec = round(monotonic() - started_monotonic, 6)
    report = _build_report(
        config=config,
        case=case,
        runtime=runtime,
        panel_count=panel_count,
        public_trace_count=len(traces),
        feedback_report=feedback_report,
        proposal=proposal,
        safety_review=safety_review,
        development_run=development_run,
        workspace_development_run=workspace_development_run,
        capability_transfer_run=capability_transfer_run,
        time_machine_plan=time_machine_plan,
        time_machine_result=time_machine_result,
        real_product_experience_summary=real_product_experience_summary,
        llm_product_experience_summary=llm_product_experience_summary,
        started_at=started_at,
        completed_at=completed_at,
        elapsed_sec=elapsed_sec,
    )
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(output_dir / "historical_experiment_report.json", report)
    return report


def _prepare_time_machine_plan(
    config: HistoricalExperimentConfig,
    *,
    executor: CommandExecutor | None,
) -> TimeMachineEvaluationPlan | None:
    dataset_id = config.time_machine_evaluation_dataset_id
    if not dataset_id:
        return None
    if not config.workspace_coding_agent_enabled:
        raise ValueError("time_machine_evaluation_requires_workspace_coding_agent")
    if not config.company_workspace_root:
        raise ValueError("time_machine_evaluation_requires_company_workspace_root")
    if executor is None:
        raise ValueError("time_machine_evaluation_executor_required")
    if config.time_machine_evaluation_timeout_seconds <= 0:
        raise ValueError("time_machine_evaluation_timeout_must_be_positive")
    plan = build_time_machine_evaluation_plan(
        dataset_id=dataset_id,
        timeout_seconds=config.time_machine_evaluation_timeout_seconds,
        executor=executor,
    )
    if not plan.formal_ready:
        raise ValueError(
            "time_machine_evaluation_plan_not_ready:" + ",".join(plan.blocking_reasons)
        )
    workspace_digest = build_workspace_execution_profile(
        Path(config.company_workspace_root)
    ).repo_hash
    if workspace_digest != plan.starter_repo_digest:
        raise ValueError(
            "time_machine_starter_digest_mismatch:"
            f"expected={plan.starter_repo_digest}:actual={workspace_digest}"
        )
    return plan


def _validate_time_machine_contract_visibility(
    config: HistoricalExperimentConfig,
) -> None:
    if (
        config.time_machine_public_contract_visibility
        not in TIME_MACHINE_CONTRACT_VISIBILITIES
    ):
        allowed = ",".join(sorted(TIME_MACHINE_CONTRACT_VISIBILITIES))
        raise ValueError(
            "unsupported_time_machine_public_contract_visibility:"
            f"{config.time_machine_public_contract_visibility}:allowed={allowed}"
        )


def _validate_live_company_agent_config(
    config: HistoricalExperimentConfig,
) -> None:
    if config.company_agent_request_attempts < 1:
        raise ValueError("company_agent_request_attempts_must_be_positive")
    if config.company_agent_retry_backoff_seconds < 0:
        raise ValueError("company_agent_retry_backoff_must_be_nonnegative")
    if not config.require_live_company_agents:
        return
    providers = {
        "optimizer": config.company_optimizer_provider,
        "safety": config.company_safety_provider,
        "coding": config.company_coding_agent_provider,
    }
    invalid = tuple(
        name
        for name, provider in providers.items()
        if provider.strip().lower() != "openai"
    )
    if invalid:
        raise ValueError(
            "require_live_company_agents_requires_openai_providers:"
            + ",".join(invalid)
        )


def _agent_visible_time_machine_contracts(
    *,
    config: HistoricalExperimentConfig,
    time_machine_plan: TimeMachineEvaluationPlan | None,
) -> tuple[dict[str, Any], ...]:
    if (
        time_machine_plan is None
        or config.time_machine_public_contract_visibility == "evaluator_only"
    ):
        return ()
    return time_machine_plan.public_contracts


def _workspace_verification_commands(
    *,
    config: HistoricalExperimentConfig,
    time_machine_plan: TimeMachineEvaluationPlan | None,
    real_experience_harness,
    published_objective_observation_refs: tuple[str, ...],
) -> tuple[str, ...]:
    agent_visible_contracts = _agent_visible_time_machine_contracts(
        config=config,
        time_machine_plan=time_machine_plan,
    )
    public_experience_reproducers = (
        real_experience_harness.failed_verification_commands(
            allowed_observation_ids=published_objective_observation_refs
        )
        if real_experience_harness
        else ()
    )
    return tuple(
        dict.fromkeys(
            (
                *config.workspace_coding_agent_verification_commands,
                *(
                    command
                    for contract in agent_visible_contracts
                    for command in contract["acceptance_commands"]
                ),
                *public_experience_reproducers,
            )
        )
    )


def _time_machine_evaluation_summary(
    config: HistoricalExperimentConfig,
    plan: TimeMachineEvaluationPlan,
    result: TimeMachineEvaluationResult,
) -> dict[str, Any]:
    return {
        "dataset_id": plan.dataset_id,
        "product_name": plan.product_name,
        "starter_ref": plan.starter_ref,
        "reference_ref": plan.reference_ref,
        "starter_repo_digest": plan.starter_repo_digest,
        "reference_repo_digest": plan.reference_repo_digest,
        "candidate_repo_digest": result.candidate_repo_digest,
        "hidden_suite_hash": plan.hidden_suite_hash,
        "evaluator_environment_hash": plan.evaluator_environment_hash,
        "plan_hash": plan.plan_hash,
        "result_hash": result.result_hash,
        "formal_ready": plan.formal_ready,
        "formal_claim_ready": result.formal_claim_ready,
        "status": result.status,
        "oracle_count": len(plan.oracles),
        "causal_fix_count": result.causal_fix_count,
        "unresolved_count": result.unresolved_count,
        "regression_count": result.regression_count,
        "infrastructure_error_count": result.infrastructure_error_count,
        "baseline_pass_rate": result.baseline_pass_rate,
        "candidate_pass_rate": result.candidate_pass_rate,
        "reference_pass_rate": result.reference_pass_rate,
        "pass_rate_uplift": result.pass_rate_uplift,
        "outcomes": tuple(
            {
                "test_id": item.test_id,
                "issue_ids": item.issue_ids,
                "baseline_status": item.baseline_status,
                "reference_status": item.reference_status,
                "candidate_status": item.candidate_status,
                "causal_fix": item.causal_fix,
                "unresolved": item.unresolved,
                "regression": item.regression,
            }
            for item in result.outcomes
        ),
        "evidence_record_count": len(result.evidence_records),
        "evidence_hashes": tuple(
            item.evidence_hash for item in result.evidence_records
        ),
        "public_contract_visibility": (
            config.time_machine_public_contract_visibility
        ),
        "evaluation_lane": (
            "blind_direction_discovery"
            if config.time_machine_public_contract_visibility == "evaluator_only"
            else "specified_contract_implementation"
        ),
        "privacy_boundary": (
            "starter_and_declared_agent_inputs_visible;"
            "reference_hidden_suite_and_evaluator_only_contracts_withheld"
        ),
    }


def _build_report(
    *,
    config: HistoricalExperimentConfig,
    case: HistoricalProductCase,
    runtime: SocietyRuntime,
    panel_count: int,
    public_trace_count: int,
    feedback_report,
    proposal: CompanyOptimizationProposal,
    safety_review: CompanySafetyReview,
    development_run: CompanyDevelopmentRun,
    workspace_development_run: CodingAgentLoopResult | None,
    capability_transfer_run: CodingAgentLoopResult | None,
    time_machine_plan: TimeMachineEvaluationPlan | None,
    time_machine_result: TimeMachineEvaluationResult | None,
    real_product_experience_summary: dict[str, Any],
    llm_product_experience_summary: dict[str, Any],
    started_at: str,
    completed_at: str,
    elapsed_sec: float,
) -> HistoricalExperimentReport:
    artifact_id = case.initial_artifact.id
    agents = tuple(runtime.state.agents.values())
    experiences = [
        agent.latest_artifact_experience[artifact_id]
        for agent in agents
        if artifact_id in agent.latest_artifact_experience
    ]
    paid_agents = sum(
        1 for agent in agents if agent.artifact_payment_state.get(artifact_id) == "paid"
    )
    feedback = feedback_report.artifact_feedback.get(artifact_id)
    suggestions = tuple(runtime.state.upgrade_suggestions.values())
    report_payload = {
        "case_id": case.case_id,
        "seed": config.seed,
        "population_size": config.population_size,
        "simulated_days": config.simulated_days,
        "state_hash": runtime.state_hash(),
        "proposal": proposal,
        "safety_review": safety_review,
        "development_run": development_run,
        "workspace_development_run": workspace_development_run,
        "capability_transfer_run": capability_transfer_run,
        "time_machine_evaluation": (
            _time_machine_evaluation_summary(
                config,
                time_machine_plan,
                time_machine_result,
            )
            if time_machine_plan and time_machine_result
            else {}
        ),
        "real_product_experience_summary": real_product_experience_summary,
        "llm_product_experience_summary": llm_product_experience_summary,
    }
    theme_alignment = _theme_alignment(
        proposed_themes=proposal.priority_themes,
        target_themes=case.target_improvement_themes,
    )
    verified_patch_themes = _workspace_verified_patch_themes(workspace_development_run)
    patch_theme_alignment = _theme_alignment(
        proposed_themes=verified_patch_themes,
        target_themes=case.target_improvement_themes,
    )
    behavior_verified_themes = _workspace_behavior_verified_themes(
        workspace_development_run
    )
    behavior_theme_alignment = _theme_alignment(
        proposed_themes=behavior_verified_themes,
        target_themes=case.target_improvement_themes,
    )
    capability_ledger = build_capability_ledger(
        case_id=case.case_id,
        artifact_id=case.initial_artifact.id,
        proposal=proposal,
        safety_review=safety_review,
        development_run=development_run,
        workspace_development_run=workspace_development_run,
        public_trace_count=public_trace_count,
        real_product_experience_summary=real_product_experience_summary,
        transfer_evidence_by_theme=_transfer_evidence_by_theme(capability_transfer_run),
    )
    development_status = _development_status(development_run)
    workspace_development_status = _workspace_development_status(
        workspace_development_run
    )
    closed_loop_evidence_status = _closed_loop_evidence_status(
        config=config,
        panel_count=panel_count,
        agents_with_experience=len(experiences),
        public_trace_count=public_trace_count,
        feedback_themes=proposal.priority_themes,
        behavior_verified_themes=behavior_verified_themes,
        real_product_experience_summary=real_product_experience_summary,
        llm_product_experience_summary=llm_product_experience_summary,
        workspace_development_run=workspace_development_run,
    )
    provider_provenance = _provider_provenance(
        config=config,
        proposal=proposal,
        safety_review=safety_review,
        development_run=development_run,
        workspace_development_run=workspace_development_run,
        capability_transfer_run=capability_transfer_run,
        agents=agents,
        llm_product_experience_summary=llm_product_experience_summary,
        intent_provider=runtime.intent_provider,
    )
    capability_transfer_status = _capability_transfer_status(
        config=config,
        prior_ledger=capability_ledger,
        transfer_run=capability_transfer_run,
    )
    paper_evidence_certificate = _paper_evidence_certificate(
        config=config,
        case=case,
        runtime=runtime,
        feedback=feedback,
        proposal=proposal,
        theme_alignment=theme_alignment,
        patch_theme_alignment=patch_theme_alignment,
        behavior_theme_alignment=behavior_theme_alignment,
        real_product_experience_summary=real_product_experience_summary,
        workspace_development_status=workspace_development_status,
        behavior_verified_overlap=(
            _overlap_from_alignment(
                behavior_theme_alignment,
                target_themes=case.target_improvement_themes,
            )
            if behavior_verified_themes
            else None
        ),
        provider_provenance=provider_provenance,
        frozen_at_utc=started_at,
    )
    return HistoricalExperimentReport(
        report_id=f"historical_experiment_{stable_hash(report_payload)[:24]}",
        case_id=case.case_id,
        product_name=case.product_name,
        source_urls=case.source_urls,
        population_size=config.population_size,
        seed=config.seed,
        simulated_days=config.simulated_days,
        ticks_per_day=config.ticks_per_day,
        panel_experienced_agents=panel_count,
        agents_with_experience=len(experiences),
        paid_agents=paid_agents,
        public_trace_count=public_trace_count,
        real_elapsed_days=case.real_elapsed_days,
        proposed_direction=proposal.requested_direction,
        proposed_themes=proposal.priority_themes,
        historical_target_overlap=_overlap_from_alignment(
            theme_alignment,
            target_themes=case.target_improvement_themes,
        ),
        matched_target_themes=theme_alignment["matched"],
        missed_target_themes=theme_alignment["missed"],
        extra_proposed_themes=theme_alignment["extra"],
        verified_patch_historical_target_overlap=(
            _overlap_from_alignment(
                patch_theme_alignment,
                target_themes=case.target_improvement_themes,
            )
            if verified_patch_themes
            else None
        ),
        verified_patch_matched_target_themes=patch_theme_alignment["matched"],
        verified_patch_missed_target_themes=patch_theme_alignment["missed"],
        verified_patch_extra_themes=patch_theme_alignment["extra"],
        behavior_verified_historical_target_overlap=(
            _overlap_from_alignment(
                behavior_theme_alignment,
                target_themes=case.target_improvement_themes,
            )
            if behavior_verified_themes
            else None
        ),
        behavior_verified_matched_target_themes=behavior_theme_alignment["matched"],
        behavior_verified_missed_target_themes=behavior_theme_alignment["missed"],
        profile_tier_counts=_profile_tier_counts(agents),
        experience_summary=_experience_summary(experiences),
        real_product_experience_summary=real_product_experience_summary,
        llm_product_experience_summary=llm_product_experience_summary,
        upgrade_suggestion_summary=_upgrade_suggestion_summary(
            suggestions, artifact_id, agents
        ),
        payment_summary=_payment_summary(agents, artifact_id),
        public_feedback_summary=canonicalize(feedback) if feedback else {},
        demand_coverage=_demand_coverage(agents),
        community_coverage=_community_coverage(agents),
        development_status=development_status,
        workspace_development_status=workspace_development_status,
        closed_loop_evidence_status=closed_loop_evidence_status,
        llm_intent_tier_summary=_llm_intent_tier_summary(agents),
        provider_provenance=provider_provenance,
        paper_evidence_certificate=paper_evidence_certificate,
        capability_transfer_status=capability_transfer_status,
        organizational_capability_summary=capability_ledger.summary,
        organizational_capability_ledger=capability_ledger,
        proposal=proposal,
        safety_review=safety_review,
        development_run=development_run,
        workspace_development_run=workspace_development_run,
        capability_transfer_run=capability_transfer_run,
        time_machine_evaluation=(
            _time_machine_evaluation_summary(
                config,
                time_machine_plan,
                time_machine_result,
            )
            if time_machine_plan and time_machine_result
            else {}
        ),
        state_hash=runtime.state_hash(),
        started_at=started_at,
        completed_at=completed_at,
        elapsed_sec=elapsed_sec,
    )


def _utc_now_iso() -> str:
    return (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )


def _theme_alignment(
    *,
    proposed_themes: tuple[str, ...],
    target_themes: tuple[str, ...],
) -> dict[str, tuple[str, ...]]:
    proposed_set = set(proposed_themes)
    target_set = set(target_themes)
    return {
        "matched": tuple(theme for theme in target_themes if theme in proposed_set),
        "missed": tuple(theme for theme in target_themes if theme not in proposed_set),
        "extra": tuple(theme for theme in proposed_themes if theme not in target_set),
    }


def _overlap_from_alignment(
    alignment: dict[str, tuple[str, ...]],
    *,
    target_themes: tuple[str, ...],
) -> float | None:
    if not target_themes:
        return None
    return len(alignment["matched"]) / len(target_themes)


def _paper_evidence_certificate(
    *,
    config: HistoricalExperimentConfig,
    case: HistoricalProductCase,
    runtime: SocietyRuntime,
    feedback,
    proposal: CompanyOptimizationProposal,
    theme_alignment: dict[str, tuple[str, ...]],
    patch_theme_alignment: dict[str, tuple[str, ...]],
    behavior_theme_alignment: dict[str, tuple[str, ...]],
    real_product_experience_summary: dict[str, Any],
    workspace_development_status: dict[str, Any],
    behavior_verified_overlap: float | None,
    provider_provenance: dict[str, Any],
    frozen_at_utc: str,
) -> dict[str, Any]:
    public_inputs = {
        "case_id": case.case_id,
        "product_name": case.product_name,
        "artifact_id": case.initial_artifact.id,
        "seed": config.seed,
        "population_size": config.population_size,
        "simulated_days": config.simulated_days,
        "public_trace_hash": proposal.public_trace_hash,
        "proposal_id": proposal.proposal_id,
        "proposal_themes": proposal.priority_themes,
        "proposal_theme_decisions": proposal.theme_decisions,
        "safety_obligation_themes": proposal.safety_obligation_themes,
        "public_feedback_support_refs": feedback.support_refs if feedback else (),
        "state_hash": runtime.state_hash(),
    }
    evaluator_private_inputs = {
        "target_improvement_themes": case.target_improvement_themes,
        "real_elapsed_days": case.real_elapsed_days,
        "matched_target_themes": theme_alignment["matched"],
        "missed_target_themes": theme_alignment["missed"],
        "verified_patch_matched_themes": patch_theme_alignment["matched"],
        "behavior_verified_matched_themes": behavior_theme_alignment["matched"],
    }
    freeze_boundary = build_evaluation_freeze_boundary(
        case_id=case.case_id,
        public_inputs=public_inputs,
        evaluator_private_inputs=evaluator_private_inputs,
        frozen_at_utc=frozen_at_utc,
    )
    provider_with_cutoff = {
        **provider_provenance,
        "model_cutoff_policy": _model_cutoff_policy(config),
    }
    hidden_eval_coverage = _overlap_from_alignment(
        behavior_theme_alignment,
        target_themes=case.target_improvement_themes,
    )
    evidence_debts = assess_evidence_debts(
        real_product_experience_summary=real_product_experience_summary,
        workspace_development_status=workspace_development_status,
        behavior_verified_overlap=behavior_verified_overlap,
        hidden_eval_coverage=hidden_eval_coverage,
        provider_provenance=provider_with_cutoff,
    )
    theme_support = (
        feedback.theme_support if feedback and feedback.theme_support else {}
    )
    support_refs_by_theme = _claim_support_refs_by_theme(
        feedback=feedback,
        proposal=proposal,
    )
    weakest_witness = select_weakest_public_witness(
        theme_support=theme_support,
        support_refs_by_theme=support_refs_by_theme,
        eligible_themes=theme_alignment["matched"] or proposal.priority_themes,
    )
    claim_level = (
        4
        if behavior_verified_overlap is not None and behavior_verified_overlap > 0.0
        else 3
        if theme_alignment["matched"]
        else 1
    )
    certificate = build_society_claim_certificate(
        claim_id="historical_product_direction_overlap",
        claim_level=claim_level,
        supported_claim=(
            "Public external-society product feedback identifies historical next-version improvement themes "
            "under the declared frozen evaluation boundary."
        ),
        freeze_boundary=freeze_boundary,
        evidence_debts=evidence_debts,
        weakest_witness=weakest_witness,
    )
    return canonicalize(certificate)


def _claim_support_refs_by_theme(
    *,
    feedback,
    proposal: CompanyOptimizationProposal,
) -> dict[str, tuple[str, ...]]:
    theme_support = (
        feedback.theme_support if feedback and feedback.theme_support else {}
    )
    feedback_refs = (
        feedback.theme_support_refs
        if feedback and feedback.theme_support_refs
        else {}
    )
    decision_refs = {
        decision.theme: decision.support_refs
        for decision in proposal.theme_decisions
    }
    return {
        theme: tuple(
            decision_refs.get(theme)
            or feedback_refs.get(theme)
            or (tuple(feedback.support_refs[:10]) if feedback else ())
        )
        for theme in theme_support
    }


def _workspace_verified_patch_themes(
    run: CodingAgentLoopResult | None,
) -> tuple[str, ...]:
    if run is None or not evidence_level_at_least(
        run.evidence_level,
        "targeted_verified",
    ):
        return ()
    effective_patch_ids = {
        result.patch_id
        for result in run.patch_results
        if result.status == "applied"
        and result.before_hash != result.after_hash
        and _counts_as_product_patch_path(result.path)
    }
    if not effective_patch_ids:
        return ()
    themes: list[str] = []
    for proposal in run.proposals:
        if len(proposal.themes) != 1:
            continue
        if any(patch.patch_id in effective_patch_ids for patch in proposal.patches):
            themes.extend(proposal.themes)
    return tuple(dict.fromkeys(themes))


def _workspace_behavior_verified_themes(
    run: CodingAgentLoopResult | None,
) -> tuple[str, ...]:
    if run is None or not evidence_level_at_least(
        run.evidence_level,
        "targeted_verified",
    ):
        return ()
    patch_coverage = run.patch_coverage or {}
    behavior_paths = set(patch_coverage.get("behavior_covered_paths", ()))
    if not behavior_paths:
        return ()
    themes: list[str] = []
    for proposal in run.proposals:
        if len(proposal.themes) != 1:
            continue
        if any(patch.path in behavior_paths for patch in proposal.patches):
            themes.extend(proposal.themes)
    return tuple(dict.fromkeys(themes))


def _counts_as_product_patch_path(path: str) -> bool:
    return Path(path).as_posix() != "company-development/workspace-agent-report.json"


def _profile_tier_counts(agents) -> dict[str, dict[str, int]]:
    counts = {
        "activity_tier": {},
        "intelligence_tier": {},
        "llm_model_tier": {},
        "primary_domain": {},
        "technical_role": {},
        "innovation_role": {},
    }
    for agent in agents:
        counts["activity_tier"][agent.profile.activity_tier] = (
            counts["activity_tier"].get(agent.profile.activity_tier, 0) + 1
        )
        counts["intelligence_tier"][agent.profile.intelligence_tier] = (
            counts["intelligence_tier"].get(agent.profile.intelligence_tier, 0) + 1
        )
        counts["llm_model_tier"][agent.profile.llm_model_tier] = (
            counts["llm_model_tier"].get(agent.profile.llm_model_tier, 0) + 1
        )
        counts["primary_domain"][agent.profile.primary_domain] = (
            counts["primary_domain"].get(agent.profile.primary_domain, 0) + 1
        )
        counts["technical_role"][agent.profile.technical_role] = (
            counts["technical_role"].get(agent.profile.technical_role, 0) + 1
        )
        counts["innovation_role"][agent.profile.innovation_role] = (
            counts["innovation_role"].get(agent.profile.innovation_role, 0) + 1
        )
    return counts


def _llm_product_experience_summary(
    *,
    agents,
    artifact_id: str,
    event_log=(),
    objective_execution_observation_count: int,
    objective_execution_observation_refs: tuple[str, ...] = (),
    objective_execution_observation_manifest_hash: str = "",
) -> dict[str, Any]:
    dimensions = {
        "by_activity_tier": "activity_tier",
        "by_intelligence_tier": "intelligence_tier",
        "by_model_tier": "llm_model_tier",
        "by_technical_role": "technical_role",
        "by_innovation_role": "innovation_role",
    }
    buckets: dict[str, dict[str, dict[str, int]]] = {
        name: {} for name in dimensions
    }
    source_counts: dict[str, int] = {}
    input_evidence_hashes: set[str] = set()
    agent_with_session_count = 0
    subjective_session_count = 0
    live_session_count = 0
    heuristic_session_count = 0
    fallback_session_count = 0
    grounded_session_count = 0
    structured_feedback_count = 0
    provenance_complete_count = 0
    objective_ref_linked_session_count = 0
    objective_linked_session_count = 0
    declared_observation_refs = tuple(objective_execution_observation_refs)
    declared_observation_set = set(declared_observation_refs)
    declared_observation_manifest_hash = (
        objective_execution_observation_manifest_hash.strip()
    )
    event_experience_index = _event_experience_evidence_index(
        event_log=event_log,
        artifact_id=artifact_id,
    )
    event_bound_session_count = 0

    for agent in agents:
        records = tuple(
            record
            for record in agent.llm_intent_session_history
            if record.artifact_id == artifact_id
        )
        if not records and artifact_id in agent.latest_llm_intents:
            records = (agent.latest_llm_intents[artifact_id],)
        agent_with_session_count += int(bool(records))
        experience_by_id = {
            packet.experience_id: canonicalize(packet)
            for packet in agent.artifact_experience_history.get(artifact_id, ())
        }
        latest_experience = agent.latest_artifact_experience.get(artifact_id)
        if latest_experience:
            experience_by_id[latest_experience.experience_id] = canonicalize(
                latest_experience
            )
        experience_by_id.update(
            event_experience_index.get(agent.id, {})
        )

        for bucket_name, profile_field in dimensions.items():
            tier = str(getattr(agent.profile, profile_field))
            row = buckets[bucket_name].setdefault(
                tier,
                {
                    "agent_count": 0,
                    "subjective_session_count": 0,
                    "live_session_count": 0,
                    "grounded_session_count": 0,
                    "structured_feedback_count": 0,
                    "fallback_session_count": 0,
                },
            )
            row["agent_count"] += 1
        for record in records:
            is_fallback = "fallback" in record.source
            is_live = bool(
                record.source.startswith("openai_live_v17:")
                and not is_fallback
                and record.blocked_reason is None
            )
            is_heuristic = record.source == "heuristic_frozen_llm_surrogate_v16"
            is_grounded = bool(
                record.experience_ref
                and record.experience_ref in experience_by_id
            )
            is_event_bound = bool(
                record.experience_ref
                and record.experience_ref
                in event_experience_index.get(agent.id, {})
            )
            experience = (
                experience_by_id.get(record.experience_ref)
                if record.experience_ref
                else None
            )
            is_objective_ref_linked = bool(
                experience
                and declared_observation_set
                and experience.get("objective_observation_refs")
                and len(experience.get("objective_observation_refs", ()))
                == len(set(experience.get("objective_observation_refs", ())))
                and set(experience.get("objective_observation_refs", ())).issubset(
                    declared_observation_set
                )
                and experience.get("objective_observation_set_hash")
                == stable_hash(
                    tuple(experience.get("objective_observation_refs", ()))
                )
            )
            is_objective_linked = bool(
                is_objective_ref_linked
                and declared_observation_manifest_hash
                and experience.get("objective_observation_manifest_hash")
                == declared_observation_manifest_hash
            )
            has_structured_feedback = bool(
                record.experience_summary.strip()
                and record.product_feedback.strip()
                and record.suggested_improvements
                and all(item.strip() for item in record.suggested_improvements)
            )
            has_provenance = bool(
                record.input_evidence_hash and record.output_evidence_hash
            )
            subjective_session_count += 1
            source_counts[record.source] = source_counts.get(record.source, 0) + 1
            live_session_count += int(is_live)
            heuristic_session_count += int(is_heuristic)
            fallback_session_count += int(is_fallback)
            grounded_session_count += int(is_grounded)
            event_bound_session_count += int(is_event_bound)
            structured_feedback_count += int(has_structured_feedback)
            provenance_complete_count += int(has_provenance)
            objective_ref_linked_session_count += int(is_objective_ref_linked)
            objective_linked_session_count += int(is_objective_linked)
            if record.input_evidence_hash:
                input_evidence_hashes.add(record.input_evidence_hash)
            for bucket_name, profile_field in dimensions.items():
                tier = str(getattr(agent.profile, profile_field))
                row = buckets[bucket_name][tier]
                row["subjective_session_count"] += 1
                row["live_session_count"] += int(is_live)
                row["grounded_session_count"] += int(is_grounded)
                row["structured_feedback_count"] += int(has_structured_feedback)
                row["fallback_session_count"] += int(is_fallback)

    agent_count = len(agents)
    missing = []
    if objective_execution_observation_count <= 0:
        missing.append("objective_workspace_execution_observations")
    if (
        objective_execution_observation_count != len(declared_observation_refs)
        or len(declared_observation_set) != len(declared_observation_refs)
    ):
        missing.append("consistent_declared_objective_observation_manifest")
    if not declared_observation_manifest_hash:
        missing.append("declared_full_objective_observation_manifest_hash")
    if agent_with_session_count != agent_count or agent_count <= 0:
        missing.append("at_least_one_subjective_session_per_agent")
    if live_session_count != subjective_session_count or not subjective_session_count:
        missing.append("all_subjective_sessions_are_live_llm")
    if grounded_session_count != subjective_session_count or not subjective_session_count:
        missing.append("all_sessions_grounded_in_agent_experience")
    if event_bound_session_count != subjective_session_count or not subjective_session_count:
        missing.append("all_sessions_bound_to_append_only_experience_events")
    if (
        structured_feedback_count != subjective_session_count
        or not subjective_session_count
    ):
        missing.append("all_sessions_have_structured_feedback")
    if provenance_complete_count != subjective_session_count or not subjective_session_count:
        missing.append("all_sessions_have_input_output_provenance")
    if (
        objective_ref_linked_session_count != subjective_session_count
        or not subjective_session_count
    ):
        missing.append("all_sessions_bound_to_declared_objective_observations")
    if (
        objective_linked_session_count != subjective_session_count
        or not subjective_session_count
    ):
        missing.append("all_sessions_bound_to_full_objective_observation_manifest")
    if len(input_evidence_hashes) != subjective_session_count:
        missing.append("unique_input_evidence_per_session")
    if fallback_session_count:
        missing.append("zero_live_provider_fallbacks")
    return {
        "agent_count": agent_count,
        "agent_with_session_count": agent_with_session_count,
        "objective_execution_observation_count": objective_execution_observation_count,
        "objective_execution_observation_refs": declared_observation_refs,
        "objective_execution_observation_ref_set_hash": stable_hash(
            declared_observation_refs
        ),
        "objective_execution_observation_manifest_hash": (
            declared_observation_manifest_hash
        ),
        "subjective_session_count": subjective_session_count,
        "live_session_count": live_session_count,
        "heuristic_session_count": heuristic_session_count,
        "fallback_session_count": fallback_session_count,
        "grounded_session_count": grounded_session_count,
        "event_bound_session_count": event_bound_session_count,
        "experience_event_count": sum(
            len(experiences)
            for experiences in event_experience_index.values()
        ),
        "experience_event_manifest_hash": stable_hash(
            event_experience_index
        ),
        "structured_feedback_count": structured_feedback_count,
        "provenance_complete_count": provenance_complete_count,
        "objective_ref_linked_session_count": objective_ref_linked_session_count,
        "objective_linked_session_count": objective_linked_session_count,
        "unique_input_evidence_hash_count": len(input_evidence_hashes),
        "source_counts": dict(sorted(source_counts.items())),
        "by_activity_tier": dict(sorted(buckets["by_activity_tier"].items())),
        "by_intelligence_tier": dict(
            sorted(buckets["by_intelligence_tier"].items())
        ),
        "by_model_tier": dict(sorted(buckets["by_model_tier"].items())),
        "by_technical_role": dict(
            sorted(buckets["by_technical_role"].items())
        ),
        "by_innovation_role": dict(
            sorted(buckets["by_innovation_role"].items())
        ),
        "objective_evidence_scope": "shared_workspace_execution_observations",
        "subjective_session_scope": "independent_profile_conditioned_llm_review",
        "claim_ready": not missing,
        "missing_for_claim": tuple(missing),
    }


def _event_experience_evidence_index(
    *,
    event_log,
    artifact_id: str,
) -> dict[str, dict[str, dict[str, Any]]]:
    index: dict[str, dict[str, dict[str, Any]]] = {}
    for event in event_log:
        actor_id = str(getattr(event, "actor_id", "") or "")
        typed_payload = getattr(event, "typed_payload", {})
        if not actor_id or not isinstance(typed_payload, Mapping):
            continue
        action_payload = typed_payload.get("action_payload", {})
        if not isinstance(action_payload, Mapping):
            continue
        packet = action_payload.get("product_experience")
        if not isinstance(packet, Mapping):
            continue
        canonical_packet = canonicalize(dict(packet))
        if canonical_packet.get("artifact_id") != artifact_id:
            continue
        experience_id = str(canonical_packet.get("experience_id", "") or "")
        if not experience_id:
            continue
        agent_index = index.setdefault(actor_id, {})
        previous = agent_index.get(experience_id)
        if previous is not None and stable_hash(previous) != stable_hash(
            canonical_packet
        ):
            raise RuntimeError(
                "conflicting_product_experience_event:"
                f"agent={actor_id}:experience={experience_id}"
            )
        agent_index[experience_id] = canonical_packet
    return index


def _llm_intent_tier_summary(agents) -> dict[str, Any]:
    summary: dict[str, dict[str, Any]] = {}
    for agent in agents:
        tier = agent.profile.llm_model_tier
        row = summary.setdefault(
            tier,
            {
                "agent_count": 0,
                "intent_count": 0,
                "sources": {},
                "fallback_count": 0,
                "mean_intent_to_pay": None,
                "_intent_to_pay_values": [],
            },
        )
        row["agent_count"] += 1
        records = (
            tuple(agent.llm_intent_session_history)
            or tuple(agent.latest_llm_intents.values())
        )
        for record in records:
            row["intent_count"] += 1
            row["sources"][record.source] = row["sources"].get(record.source, 0) + 1
            if "fallback" in record.source or record.blocked_reason:
                row["fallback_count"] += int(
                    bool(record.blocked_reason and "fallback" in record.source)
                )
            row["_intent_to_pay_values"].append(record.intent_to_pay)
    for row in summary.values():
        values = row.pop("_intent_to_pay_values")
        row["mean_intent_to_pay"] = mean(values) if values else None
        row["sources"] = dict(sorted(row["sources"].items()))
    return dict(sorted(summary.items()))


def _provider_provenance(
    *,
    config: HistoricalExperimentConfig,
    proposal: CompanyOptimizationProposal,
    safety_review: CompanySafetyReview,
    development_run: CompanyDevelopmentRun,
    workspace_development_run: CodingAgentLoopResult | None,
    capability_transfer_run: CodingAgentLoopResult | None,
    agents,
    llm_product_experience_summary: dict[str, Any],
    intent_provider,
) -> dict[str, Any]:
    intent_records = [
        record
        for agent in agents
        for record in (
            tuple(agent.llm_intent_session_history)
            or tuple(agent.latest_llm_intents.values())
        )
    ]
    fallback_or_blocked_records = [
        record
        for record in intent_records
        if "fallback" in record.source or record.blocked_reason
    ]
    panel_summary = llm_product_experience_summary.get("panel_summary", {})
    execution_summary_fn = getattr(intent_provider, "execution_summary", None)
    intent_execution_summary = (
        execution_summary_fn() if callable(execution_summary_fn) else {}
    )
    return {
        "llm_intent": {
            "configured_provider": config.llm_intent_provider,
            "configured_default_model": config.llm_intent_model,
            "configured_tier_models": {
                "low": config.llm_intent_low_model,
                "standard": config.llm_intent_standard_model,
                "high": config.llm_intent_high_model,
            },
            "record_count": len(intent_records),
            "source_counts": _source_counts(record.source for record in intent_records),
            "full_run_subjective_session_count": llm_product_experience_summary.get(
                "subjective_session_count", 0
            ),
            "full_run_source_counts": llm_product_experience_summary.get(
                "source_counts", {}
            ),
            "full_run_fallback_session_count": llm_product_experience_summary.get(
                "fallback_session_count", 0
            ),
            "panel_subjective_session_count": panel_summary.get(
                "subjective_session_count", 0
            ),
            "panel_source_counts": panel_summary.get("source_counts", {}),
            "panel_fallback_session_count": panel_summary.get(
                "fallback_session_count", 0
            ),
            "fallback_or_blocked_count": len(fallback_or_blocked_records),
            "execution": intent_execution_summary,
        },
        "company_optimizer": {
            "configured_provider": config.company_optimizer_provider,
            "configured_model": config.company_optimizer_model,
            "actual_source": proposal.optimizer_source,
            "actual_model": proposal.optimizer_model,
        },
        "company_safety": {
            "configured_provider": config.company_safety_provider,
            "configured_model": config.company_safety_model,
            "actual_source": safety_review.source,
            "actual_model": safety_review.model,
            "obligation_count": len(safety_review.obligations),
        },
        "company_development": {
            "configured_provider": config.company_coding_agent_provider,
            "configured_model": config.company_coding_agent_model,
            "actual_source": development_run.source,
            "actual_model": development_run.model,
            "blocked_reason": development_run.blocked_reason,
        },
        "workspace_development": _workspace_run_provider_provenance(
            config_provider=config.workspace_coding_agent_provider,
            config_model=config.workspace_coding_agent_model,
            run=workspace_development_run,
        ),
        "capability_transfer": _workspace_run_provider_provenance(
            config_provider=config.workspace_coding_agent_provider,
            config_model=config.workspace_coding_agent_model,
            run=capability_transfer_run,
        ),
    }


def _model_cutoff_policy(config: HistoricalExperimentConfig) -> str:
    providers = {
        config.llm_intent_provider,
        config.company_optimizer_provider,
        config.company_safety_provider,
        config.company_coding_agent_provider,
        config.workspace_coding_agent_provider,
    }
    if providers <= {"heuristic"}:
        return "deterministic_no_model_training_contamination"
    if "openai" in providers:
        return "live_model_retrospective_validation_not_contamination_free_prediction"
    return "unknown"


def _workspace_run_provider_provenance(
    *,
    config_provider: str,
    config_model: str,
    run: CodingAgentLoopResult | None,
) -> dict[str, Any]:
    if run is None:
        return {
            "configured_provider": config_provider,
            "configured_model": config_model,
            "run_id": None,
            "source_counts": {},
            "fallback_or_blocked_count": 0,
        }
    return {
        "configured_provider": config_provider,
        "configured_model": config_model,
        "run_id": run.run_id,
        "developer_mode": run.developer_mode,
        "final_mode": run.final_mode,
        "reasoning_effort": run.reasoning_effort,
        "requested_reasoning_effort": run.requested_reasoning_effort,
        "run_config": run.run_config,
        "patch_coverage": run.patch_coverage,
        "code_landing_metrics": run.code_landing_metrics,
        "source_counts": _source_counts(proposal.source for proposal in run.proposals),
        "fallback_or_blocked_count": sum(
            1 for proposal in run.proposals if proposal.blocked_reason
        ),
        "failed_command_count": len(run.failed_commands),
    }


def _source_counts(values) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value or "unknown")
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _experience_summary(experiences) -> dict[str, Any]:
    if not experiences:
        return {
            "count": 0,
            "mean_experienced_task_success": None,
            "mean_raw_execution_success": None,
            "mean_success_gain": None,
            "mean_friction": None,
            "mean_time_saved": None,
            "mean_diagnostic_clarity": None,
            "mean_workaround_success": None,
            "mean_first_value_time": None,
            "mean_repeat_use_value": None,
            "blocked_stages": {},
        }
    return {
        "count": len(experiences),
        "mean_experienced_task_success": mean(
            packet.experienced_task_success for packet in experiences
        ),
        "mean_raw_execution_success": _optional_mean(
            packet.raw_execution_success for packet in experiences
        ),
        "mean_success_gain": mean(packet.success_gain for packet in experiences),
        "mean_friction": mean(packet.friction for packet in experiences),
        "mean_time_saved": mean(packet.time_saved for packet in experiences),
        "mean_diagnostic_clarity": mean(
            packet.diagnostic_clarity for packet in experiences
        ),
        "mean_workaround_success": mean(
            packet.workaround_success for packet in experiences
        ),
        "mean_first_value_time": mean(
            packet.first_value_time for packet in experiences
        ),
        "mean_repeat_use_value": mean(
            packet.repeat_use_value for packet in experiences
        ),
        "blocked_stages": {
            stage: sum(1 for packet in experiences if packet.blocked_stage == stage)
            for stage in sorted(
                {packet.blocked_stage for packet in experiences if packet.blocked_stage}
            )
        },
        "failure_events": {
            event: sum(1 for packet in experiences if packet.failure_event == event)
            for event in sorted(
                {packet.failure_event for packet in experiences if packet.failure_event}
            )
        },
    }


def _optional_mean(values) -> float | None:
    observed = tuple(value for value in values if value is not None)
    return mean(observed) if observed else None


def _payment_summary(agents, artifact_id: str) -> dict[str, int]:
    summary: dict[str, int] = {}
    for agent in agents:
        state = agent.artifact_payment_state.get(artifact_id, "unattempted")
        summary[state] = summary.get(state, 0) + 1
    return summary


def _upgrade_suggestion_summary(
    suggestions, artifact_id: str, agents
) -> dict[str, Any]:
    relevant = [
        suggestion
        for suggestion in suggestions
        if suggestion.artifact_id == artifact_id
    ]
    if not relevant:
        return {
            "count": 0,
            "public_count": 0,
            "mean_priority": None,
            "mean_confidence": None,
            "themes": {},
            "by_activity_tier": {},
            "by_intelligence_tier": {},
            "by_model_tier": {},
            "by_technical_role": {},
            "by_innovation_role": {},
            "contribution_modes": {},
            "mean_technical_depth": None,
            "mean_novelty_score": None,
            "llm_feature_idea_count": 0,
            "live_llm_feedback_count": 0,
            "public_live_llm_feedback_count": 0,
        }
    themes = {
        theme: sum(1 for suggestion in relevant if suggestion.theme == theme)
        for theme in sorted({suggestion.theme for suggestion in relevant})
    }
    agents_by_id = {agent.id: agent for agent in agents}
    return {
        "count": len(relevant),
        "public_count": sum(
            1 for suggestion in relevant if suggestion.visibility_scope == "public"
        ),
        "mean_priority": mean(suggestion.priority for suggestion in relevant),
        "mean_confidence": mean(suggestion.confidence for suggestion in relevant),
        "themes": themes,
        "by_activity_tier": _suggestion_counts_by_profile_field(
            relevant, agents_by_id, "activity_tier"
        ),
        "by_intelligence_tier": _suggestion_counts_by_profile_field(
            relevant, agents_by_id, "intelligence_tier"
        ),
        "by_model_tier": _suggestion_counts_by_profile_field(
            relevant, agents_by_id, "llm_model_tier"
        ),
        "by_technical_role": _suggestion_counts_by_profile_field(
            relevant, agents_by_id, "technical_role"
        ),
        "by_innovation_role": _suggestion_counts_by_profile_field(
            relevant, agents_by_id, "innovation_role"
        ),
        "contribution_modes": {
            mode: sum(
                1 for suggestion in relevant if suggestion.contribution_mode == mode
            )
            for mode in sorted(
                {suggestion.contribution_mode for suggestion in relevant}
            )
        },
        "mean_technical_depth": mean(
            suggestion.technical_depth for suggestion in relevant
        ),
        "mean_novelty_score": mean(
            suggestion.novelty_score for suggestion in relevant
        ),
        "llm_feature_idea_count": sum(
            1 for suggestion in relevant if suggestion.llm_feature_idea
        ),
        "live_llm_feedback_count": sum(
            1 for suggestion in relevant if suggestion.llm_intent_ref
        ),
        "public_live_llm_feedback_count": sum(
            1
            for suggestion in relevant
            if suggestion.llm_intent_ref and suggestion.visibility_scope == "public"
        ),
    }


def _suggestion_counts_by_profile_field(
    suggestions, agents_by_id: dict[str, Any], profile_field: str
) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for suggestion in suggestions:
        agent = agents_by_id.get(suggestion.author_id)
        if agent is None:
            continue
        tier = str(getattr(agent.profile, profile_field))
        row = counts.setdefault(tier, {"count": 0, "public_count": 0})
        row["count"] += 1
        row["public_count"] += int(suggestion.visibility_scope == "public")
    return dict(sorted(counts.items()))


def _demand_coverage(agents) -> dict[str, int]:
    counts: dict[str, int] = {}
    for agent in agents:
        for task_id, weight in agent.profile.task_demand_weights.items():
            if weight <= 0:
                continue
            counts[task_id] = counts.get(task_id, 0) + 1
    return dict(sorted(counts.items()))


def _community_coverage(agents) -> dict[str, int]:
    counts: dict[str, int] = {}
    for agent in agents:
        for community_id in agent.profile.community_memberships:
            counts[community_id] = counts.get(community_id, 0) + 1
    return dict(sorted(counts.items()))


def _development_status(run: CompanyDevelopmentRun) -> dict[str, Any]:
    verification_exit_codes = [
        result.exit_code
        for result in run.verification_results
        if result.exit_code is not None
    ]
    return {
        "mode": run.mode,
        "source": run.source,
        "model": run.model,
        "change_unit_count": len(run.change_units),
        "edit_count": len(run.edits),
        "applied_paths": run.applied_paths,
        "blocked_reason": run.blocked_reason,
        "patch_set_hash": run.patch_set_hash,
        "private_state_boundary": run.private_state_boundary,
        "verification_result_count": len(run.verification_results),
        "verification_passed": (
            all(result.status == "passed" for result in run.verification_results)
            if run.verification_results
            else None
        ),
        "verification_exit_codes": verification_exit_codes,
    }


def _workspace_source_report(
    *,
    config: HistoricalExperimentConfig,
    case: HistoricalProductCase,
    feedback_report,
    proposal: CompanyOptimizationProposal,
    safety_review: CompanySafetyReview,
    development_run: CompanyDevelopmentRun,
    real_product_experience_summary: dict[str, Any],
    time_machine_plan: TimeMachineEvaluationPlan | None = None,
) -> dict[str, Any]:
    evaluation_episode_id = _agent_visible_episode_id(case)
    agent_visible_contracts = _agent_visible_time_machine_contracts(
        config=config,
        time_machine_plan=time_machine_plan,
    )
    payload = {
        "evaluation_episode_id": evaluation_episode_id,
        "artifact_id": case.initial_artifact.id,
        "product_name": case.product_name,
        "seed": config.seed,
        "population_size": config.population_size,
        "simulated_days": config.simulated_days,
        "proposal_id": proposal.proposal_id,
        "public_feedback_report_id": feedback_report.report_id,
        "safety_review_id": safety_review.review_id,
        "development_run_id": development_run.run_id,
    }
    report = {
        "report_id": f"historical_workspace_source_{stable_hash(payload)[:24]}",
        "evaluation_episode_id": evaluation_episode_id,
        "product_name": case.product_name,
        "artifact_id": case.initial_artifact.id,
        "source_urls": case.agent_visible_source_urls,
        "seed": config.seed,
        "population_size": config.population_size,
        "simulated_days": config.simulated_days,
        "proposed_direction": proposal.requested_direction,
        "proposed_themes": proposal.priority_themes,
        "real_product_experience_summary": real_product_experience_summary,
        "public_feedback_summary": canonicalize(feedback_report),
        "proposal": _public_proposal_payload(proposal),
        "safety_review": canonicalize(safety_review),
        "development_run": canonicalize(development_run),
        "time_machine_public_contracts": (
            canonicalize(agent_visible_contracts)
        ),
        "time_machine_public_contract_visibility": (
            config.time_machine_public_contract_visibility
        ),
        "maintainer_landing_gate": {
            "enabled": True,
            "require_intent_trace": True,
            "require_behavior_verification": True,
            "require_test_or_repro": True,
        },
        "privacy_boundary": (
            "base_release_sources_public_society_feedback_and_workspace_files_only"
        ),
    }
    public_contract_specs = tuple(
        {
            "intent_id": f"release_contract_{contract['issue_id']}",
            "theme": contract["theme"],
            "task_type": "feature",
            "user_pain": contract["user_pain"],
            "observed_behavior": (
                "The frozen base release does not satisfy this public contract."
            ),
            "expected_behavior": contract["expected_behavior"],
            "reproduction_steps": contract["reproduction_steps"],
            "reproduction_expected_failure": (
                "The frozen base release should fail the public reproducer."
            ),
            "acceptance_tests": contract["acceptance_commands"],
            "contract_dimensions": contract["contract_dimensions"],
            "candidate_path_hints": contract["candidate_path_hints"],
            "relevant_symbols_hint": contract["relevant_symbols_hint"],
            "support_refs": (
                contract["source_url"],
                f"public_contract:{contract['issue_id']}",
            ),
            "evidence_refs": (f"public_contract:{contract['issue_id']}",),
            "risk_level": "medium",
            "non_goals": (
                "Do not copy or reconstruct the withheld reference release.",
                "Do not rely on evaluator-private tests or future release notes.",
            ),
        }
        for contract in agent_visible_contracts
    )
    report["development_intent_specs"] = (
        *public_contract_specs,
        *build_development_intent_specs(
            {
                **report,
                "development_intent_specs": (),
            }
        ),
    )
    return report


def _agent_visible_episode_id(case: HistoricalProductCase) -> str:
    identity = {
        "product_name": case.product_name,
        "artifact_id": case.initial_artifact.id,
        "base_version": case.initial_artifact.version,
        "base_sources": case.agent_visible_source_urls,
    }
    return f"historical_episode_{stable_hash(identity)[:24]}"


def _capability_transfer_source_report(
    *,
    config: HistoricalExperimentConfig,
    case: HistoricalProductCase,
    proposal: CompanyOptimizationProposal,
    prior_ledger: CapabilityLedger,
    transfer_themes: tuple[str, ...],
) -> dict[str, Any]:
    payload = {
        "case_id": case.case_id,
        "artifact_id": case.initial_artifact.id,
        "seed": config.seed,
        "prior_ledger_id": prior_ledger.ledger_id,
        "transfer_themes": transfer_themes,
    }
    report = {
        "report_id": f"capability_transfer_probe_{stable_hash(payload)[:24]}",
        "evaluation_stage": "capability_transfer_probe",
        "case_id": case.case_id,
        "product_name": case.product_name,
        "artifact_id": case.initial_artifact.id,
        "seed": config.seed,
        "proposed_direction": f"reuse_prior_capabilities_for_held_out_episode:{proposal.requested_direction}",
        "proposed_themes": transfer_themes,
        "capability_transfer_probe": {
            "prior_ledger_id": prior_ledger.ledger_id,
            "prior_ledger_hash": prior_ledger.hash(),
            "eligible_themes": transfer_themes,
            "required_evidence": (
                "same_capability_theme_reused",
                "held_out_or_later_episode_workspace_patch",
                "evaluator_owned_verification_passed",
                "effective_patch_or_verified_state_impact",
            ),
        },
        "prior_capability_summary": prior_ledger.summary,
        "proposal": _public_proposal_payload(proposal),
        "maintainer_landing_gate": {
            "enabled": True,
            "require_intent_trace": True,
            "require_behavior_verification": True,
            "require_test_or_repro": True,
        },
        "privacy_boundary": "public_capability_ledger_and_workspace_files_only",
    }
    report["development_intent_specs"] = build_development_intent_specs(report)
    return report


def _public_proposal_payload(proposal: CompanyOptimizationProposal) -> dict[str, Any]:
    return {
        "proposal_id": proposal.proposal_id,
        "artifact_id": proposal.artifact_id,
        "requested_direction": proposal.requested_direction,
        "priority_themes": proposal.priority_themes,
        "theme_decisions": proposal.theme_decisions,
        "safety_obligation_themes": proposal.safety_obligation_themes,
        "support_refs": proposal.support_refs,
        "public_trace_hash": proposal.public_trace_hash,
        "optimizer_source": proposal.optimizer_source,
        "optimizer_model": proposal.optimizer_model,
        "optimizer_reasoning": proposal.optimizer_reasoning,
    }


def _workspace_development_status(run: CodingAgentLoopResult | None) -> dict[str, Any]:
    if run is None:
        return {
            "enabled": False,
            "final_mode": None,
            "verified": None,
            "applied_patch_count": 0,
        }
    applied_results = tuple(
        result for result in run.patch_results if result.status == "applied"
    )
    no_op_results = tuple(
        result for result in run.patch_results if result.status == "no_op"
    )
    targeted_evidence = evidence_level_at_least(
        run.evidence_level,
        "targeted_verified",
    )
    release_evidence = evidence_level_at_least(
        run.evidence_level,
        "release_candidate",
    )
    return {
        "enabled": True,
        "run_id": run.run_id,
        "source_report_id": run.source_report_id,
        "developer_mode": run.developer_mode,
        "development_stage": run.development_stage,
        "final_mode": run.final_mode,
        "verified": release_evidence,
        "execution_completed": run.final_mode == "verified",
        "targeted_verified": targeted_evidence,
        "release_candidate": release_evidence,
        "evidence_level": run.evidence_level,
        "evidence_hash": run.evidence_hash,
        "evidence_summary": run.evidence_summary,
        "verification_passed": release_evidence,
        "iterations": run.iterations,
        "exploration_attempts": run.exploration_attempts,
        "promoted_candidates": run.promoted_candidates,
        "verified_effective_patches": run.verified_effective_patches,
        "rejected_by_gate": run.rejected_by_gate,
        "repair_iterations": run.repair_iterations,
        "applied_patch_count": len(applied_results),
        "no_op_patch_count": len(no_op_results),
        "selected_files": run.selected_files,
        "failed_commands": run.failed_commands,
        "reasoning_effort": run.reasoning_effort,
        "requested_reasoning_effort": run.requested_reasoning_effort,
        "run_config": run.run_config,
        "patch_coverage": run.patch_coverage,
        "code_landing_metrics": run.code_landing_metrics,
        "provenance_hash": run.provenance_hash,
        "privacy_boundary": run.privacy_boundary,
    }


def _transfer_evidence_by_theme(
    run: CodingAgentLoopResult | None,
) -> dict[str, tuple[str, ...]]:
    if run is None:
        return {}
    if not evidence_level_at_least(run.evidence_level, "release_candidate"):
        return {}
    if run.verified_effective_patches <= 0:
        return {}
    refs = tuple(
        dict.fromkeys(
            (
                f"transfer_run:{run.run_id}",
                f"transfer_provenance:{run.provenance_hash}",
                f"transfer_final_mode:{run.final_mode}",
                *(f"transfer_file:{path}" for path in run.selected_files[:8]),
                *(
                    f"transfer_check:{result.command}"
                    for result in run.verification_results
                    if result.status == "passed"
                ),
            )
        )
    )
    themes = tuple(
        dict.fromkeys(theme for proposal in run.proposals for theme in proposal.themes)
    )
    return {theme: refs for theme in themes}


def _capability_transfer_status(
    *,
    config: HistoricalExperimentConfig,
    prior_ledger: CapabilityLedger,
    transfer_run: CodingAgentLoopResult | None,
) -> dict[str, Any]:
    evidence_by_theme = _transfer_evidence_by_theme(transfer_run)
    return {
        "enabled": config.capability_transfer_probe_enabled,
        "claim_ready": bool(
            evidence_by_theme and prior_ledger.summary.get("transfer_claim_ready")
        ),
        "transfer_theme_count": len(evidence_by_theme),
        "transfer_themes": tuple(sorted(evidence_by_theme)),
        "run_id": transfer_run.run_id if transfer_run else None,
        "final_mode": transfer_run.final_mode if transfer_run else None,
        "verified_effective_patches": transfer_run.verified_effective_patches
        if transfer_run
        else 0,
        "failed_commands": transfer_run.failed_commands if transfer_run else (),
        "missing_for_l4_claim": ()
        if evidence_by_theme and prior_ledger.summary.get("transfer_claim_ready")
        else _capability_transfer_missing_reasons(
            config=config,
            ledger=prior_ledger,
            transfer_run=transfer_run,
            evidence_by_theme=evidence_by_theme,
        ),
        "privacy_boundary": "public_capability_ledger_and_workspace_files_only",
    }


def _capability_transfer_missing_reasons(
    *,
    config: HistoricalExperimentConfig,
    ledger: CapabilityLedger,
    transfer_run: CodingAgentLoopResult | None,
    evidence_by_theme: dict[str, tuple[str, ...]],
) -> tuple[str, ...]:
    missing: list[str] = []
    if not config.capability_transfer_probe_enabled:
        missing.append("transfer_probe_disabled")
    if not any(capability.level >= 2 for capability in ledger.capabilities):
        missing.append("no_prior_l2_persistent_capability")
    if transfer_run is None:
        missing.append("no_transfer_run")
    elif not evidence_level_at_least(
        transfer_run.evidence_level,
        "release_candidate",
    ):
        missing.append("transfer_workspace_not_verified")
    if not evidence_by_theme:
        missing.append("no_theme_specific_transfer_evidence")
    if not ledger.summary.get("transfer_claim_ready"):
        missing.append("ledger_has_no_l4_capability")
    return tuple(dict.fromkeys(missing))


def _closed_loop_evidence_status(
    *,
    config: HistoricalExperimentConfig,
    panel_count: int,
    agents_with_experience: int,
    public_trace_count: int,
    feedback_themes: tuple[str, ...],
    behavior_verified_themes: tuple[str, ...],
    real_product_experience_summary: dict[str, Any],
    llm_product_experience_summary: dict[str, Any],
    workspace_development_run: CodingAgentLoopResult | None,
) -> dict[str, Any]:
    real_experience_observations = int(
        real_product_experience_summary.get("observation_count", 0) or 0
    )
    real_product_experience_ready = (
        config.real_product_experience_enabled
        and real_experience_observations > 0
        and agents_with_experience == panel_count
        and panel_count > 0
    )
    workspace_development_verified = (
        config.workspace_coding_agent_enabled
        and workspace_development_run is not None
        and evidence_level_at_least(
            workspace_development_run.evidence_level,
            "release_candidate",
        )
        and bool(workspace_development_run.evidence_hash)
        and workspace_development_run.verified_effective_patches > 0
    )
    feedback_theme_set = set(feedback_themes)
    linked_verified_themes = tuple(
        theme
        for theme in behavior_verified_themes
        if theme in feedback_theme_set
    )
    theme_linked_workspace_development = (
        workspace_development_verified and bool(linked_verified_themes)
    )
    public_feedback_ready = public_trace_count > 0 and bool(feedback_themes)
    live_llm_product_experience_ready = bool(
        llm_product_experience_summary.get("claim_ready")
    )
    claim_ready = (
        real_product_experience_ready
        and public_feedback_ready
        and theme_linked_workspace_development
        and (
            not config.require_live_llm_product_experience
            or live_llm_product_experience_ready
        )
    )
    missing = []
    if not real_product_experience_ready:
        missing.append("real_workspace_product_experience")
    if not public_feedback_ready:
        missing.append("company_visible_public_feedback")
    if not workspace_development_verified:
        missing.append("verified_workspace_development")
    if not theme_linked_workspace_development:
        missing.append("theme_linked_verified_development")
    if (
        config.require_live_llm_product_experience
        and not live_llm_product_experience_ready
    ):
        missing.append("live_profile_conditioned_llm_product_experience")
    return {
        "claim_ready": claim_ready,
        "real_product_experience_ready": real_product_experience_ready,
        "real_product_experience_observations": real_experience_observations,
        "agents_with_experience": agents_with_experience,
        "panel_experienced_agents": panel_count,
        "live_llm_product_experience_required": (
            config.require_live_llm_product_experience
        ),
        "live_llm_product_experience_ready": live_llm_product_experience_ready,
        "public_feedback_ready": public_feedback_ready,
        "public_trace_count": public_trace_count,
        "workspace_development_verified": workspace_development_verified,
        "theme_linked_workspace_development": theme_linked_workspace_development,
        "feedback_themes": feedback_themes,
        "behavior_verified_themes": behavior_verified_themes,
        "linked_verified_themes": linked_verified_themes,
        "workspace_final_mode": workspace_development_run.final_mode
        if workspace_development_run
        else None,
        "workspace_evidence_level": workspace_development_run.evidence_level
        if workspace_development_run
        else None,
        "missing_for_full_closed_loop_claim": tuple(missing),
        "privacy_boundary": "public_feedback_and_workspace_files_only",
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(canonicalize(asdict(value)), indent=2, sort_keys=True),
        encoding="utf-8",
    )
