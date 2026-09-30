"""Typed artifacts for the Code-Max workspace landing runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from society_core.workspace_update import (
    WorkspaceFilePatch,
    WorkspacePatchResult,
    WorkspaceVerificationResult,
)


@dataclass(frozen=True)
class WorkspaceExecutionProfile:
    repo_hash: str
    os_image: str
    language_stack: tuple[str, ...]
    package_manager: str | None
    install_commands: tuple[str, ...]
    setup_status: str
    blocked_reason: str | None
    discovered_test_commands: tuple[str, ...]
    discovered_lint_commands: tuple[str, ...]
    discovered_build_commands: tuple[str, ...]
    smoke_commands: tuple[str, ...]
    env_vars_required: tuple[str, ...]
    services_required: tuple[str, ...]
    flaky_commands: tuple[str, ...]
    command_timeouts: dict[str, int]
    setup_log_hash: str
    missing_declared_dependencies: tuple[str, ...] = ()


@dataclass(frozen=True)
class SymbolLocation:
    symbol: str
    path: str
    line: int
    kind: str


@dataclass(frozen=True)
class RepoIntelligenceFabric:
    repo_hash: str
    file_tree: tuple[str, ...]
    source_roots: tuple[str, ...]
    test_roots: tuple[str, ...]
    config_files: tuple[str, ...]
    docs_files: tuple[str, ...]
    generated_files: tuple[str, ...]
    package_manager: str | None
    entrypoints: tuple[str, ...]
    public_api_surfaces: tuple[str, ...]
    symbol_index: dict[str, tuple[SymbolLocation, ...]]
    import_graph: dict[str, tuple[str, ...]]
    reverse_dependency_graph: dict[str, tuple[str, ...]]
    source_to_tests: dict[str, tuple[str, ...]]
    test_to_sources: dict[str, tuple[str, ...]]
    coverage_map: dict[str, tuple[str, ...]]
    error_patterns: tuple[str, ...]
    style_conventions: tuple[str, ...]
    architecture_notes: tuple[str, ...]
    semantic_search_index_id: str
    code_knowledge_graph_id: str
    candidate_files_by_theme: dict[str, tuple[str, ...]] = field(default_factory=dict)


@dataclass(frozen=True)
class AcceptanceOracleSpec:
    oracle_id: str
    task_id: str
    kind: str
    command: str
    files_created: tuple[str, ...]
    expected_on_base: str
    expected_on_patch: str
    base_observed: str | None
    base_status: str
    confidence: float
    false_positive_risk: str
    evidence_hash: str
    dimension_ids: tuple[str, ...] = ()
    witness_role: str = "repair"
    required: bool = True


@dataclass(frozen=True)
class LocalizationHypothesis:
    hypothesis_id: str
    task_id: str
    candidate_files: tuple[str, ...]
    candidate_symbols: tuple[str, ...]
    candidate_lines: tuple[tuple[str, int, int], ...]
    rationale: str
    evidence_refs: tuple[str, ...]
    scores: dict[str, float]
    confidence: float
    risk_notes: tuple[str, ...]
    missing_context: tuple[str, ...]


@dataclass(frozen=True)
class PatchStrategyPlan:
    task_id: str
    strategies: tuple[str, ...]
    required_candidate_count: int
    allowed_files: tuple[str, ...]
    protected_files: tuple[str, ...]
    required_tests: tuple[str, ...]
    reviewer_roles_required: tuple[str, ...]
    max_diff_lines: int
    escalation_policy: str


@dataclass(frozen=True)
class CandidatePatch:
    patch_id: str
    task_id: str
    strategy: str
    produced_by_agent: str
    unified_diff: str
    workspace_patches: tuple[WorkspaceFilePatch, ...]
    changed_files: tuple[str, ...]
    changed_symbols: tuple[str, ...]
    added_tests: tuple[str, ...]
    rationale: str
    expected_behavior_change: str
    risk_notes: tuple[str, ...]
    compatibility_notes: tuple[str, ...]
    support_refs: tuple[str, ...]
    parent_localization_hypothesis_id: str
    parent_oracle_ids: tuple[str, ...]
    verification_commands: tuple[str, ...] = ()


@dataclass(frozen=True)
class PatchValidationResult:
    patch_id: str
    applies_cleanly: bool
    patch_results: tuple[WorkspacePatchResult, ...]
    oracle_results: dict[str, WorkspaceVerificationResult]
    repro_passed: bool
    targeted_tests_passed: bool
    regression_tests_passed: bool | None
    lint_passed: bool | None
    typecheck_passed: bool | None
    static_analysis_passed: bool | None
    security_scan_passed: bool | None
    performance_probe_passed: bool | None
    diff_size: int
    files_touched_count: int
    public_api_changed: bool
    config_changed: bool
    test_only_patch: bool
    broad_rewrite_detected: bool
    failure_summary: str | None
    score: float
    evidence_hash: str
    candidate_verification_results: dict[str, WorkspaceVerificationResult] = field(
        default_factory=dict
    )
    candidate_verification_passed: bool = False
    candidate_behavior_verification_passed: bool = False
    added_test_verification_passed: bool = False
    added_test_verification_status: str = "not_applicable"
    public_contract_continuity_passed: bool = True
    public_contract_continuity_issues: tuple[str, ...] = ()
    required_dimension_ids: tuple[str, ...] = ()
    dimension_status_by_id: dict[str, str] = field(default_factory=dict)
    dimension_coverage_complete: bool = True
    qualified_added_contract_count: int = 0
    cross_candidate_contract_pass_count: int = 0
    cross_candidate_contract_total: int = 0
    cross_candidate_contract_failures: tuple[str, ...] = ()


@dataclass(frozen=True)
class PatchSelectionDecision:
    task_id: str | None
    selected_patch_id: str | None
    rejected_patch_ids: tuple[str, ...]
    selection_reason: str
    remaining_risks: tuple[str, ...]
    required_followup: tuple[str, ...]


@dataclass(frozen=True)
class ReviewerFinding:
    finding_id: str
    patch_id: str
    reviewer_role: str
    severity: str
    summary: str
    evidence_refs: tuple[str, ...]
    suggested_probe: str | None
    suggested_patch_constraint: str | None
    code: str = "unspecified_review_finding"
    remediation_owner: str = "candidate"
    retryable: bool = True
    details: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class ReviewDecision:
    patch_id: str | None
    status: str
    findings: tuple[ReviewerFinding, ...]
    reviewer_roles: tuple[str, ...]
    claim_ceiling: str


@dataclass(frozen=True)
class DevelopmentEpic:
    epic_id: str
    source_theme: str
    public_evidence_refs: tuple[str, ...]
    ordered_task_ids: tuple[str, ...]
    dependency_edges: tuple[tuple[str, str], ...]
    integration_oracles: tuple[str, ...]
    release_strategy: str
    stop_condition: str


@dataclass(frozen=True)
class IntegrationDecision:
    epic_id: str
    status: str
    candidate_coherent: bool
    reasons: tuple[str, ...]
    required_repairs: tuple[str, ...]


@dataclass(frozen=True)
class GuardedPatchDecision:
    patch_id: str
    accepted: bool
    applied_diff_hash: str | None
    blocked_reasons: tuple[str, ...]
    changed_files: tuple[str, ...]
    effective_product_change: bool


@dataclass(frozen=True)
class LandingDecision:
    task_id: str | None
    patch_id: str | None
    status: str
    claim_level: str
    reasons: tuple[str, ...]
    evidence_refs: tuple[str, ...]


@dataclass(frozen=True)
class CodeLandingTrajectory:
    trajectory_id: str
    task_id: str | None
    repo_hash: str
    stages: tuple[str, ...]
    tool_calls: tuple[dict[str, Any], ...]
    observations: tuple[dict[str, Any], ...]
    patches: tuple[str, ...]
    validation_results: tuple[str, ...]
    reviewer_findings: tuple[str, ...]
    final_outcome: str
    failure_attribution: tuple[str, ...]
    reusable_lessons: tuple[str, ...]


@dataclass(frozen=True)
class CodeMaxRunResult:
    runtime_id: str
    environment: WorkspaceExecutionProfile
    repo: RepoIntelligenceFabric
    oracles: tuple[AcceptanceOracleSpec, ...]
    localizations: tuple[LocalizationHypothesis, ...]
    strategies: tuple[PatchStrategyPlan, ...]
    validations: tuple[PatchValidationResult, ...]
    selection: PatchSelectionDecision
    review_decision: ReviewDecision
    integration_decision: IntegrationDecision
    guarded_decision: GuardedPatchDecision
    landing_decision: LandingDecision
    trajectory: CodeLandingTrajectory
