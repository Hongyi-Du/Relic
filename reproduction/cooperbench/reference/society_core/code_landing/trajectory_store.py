"""Replayable trajectory records for Code-Max runs."""

from __future__ import annotations

from typing import Any

from society_core.hashing import canonicalize, stable_hash

from .schemas import (
    CodeLandingTrajectory,
    GuardedPatchDecision,
    IntegrationDecision,
    LandingDecision,
    LocalizationHypothesis,
    PatchSelectionDecision,
    PatchStrategyPlan,
    PatchValidationResult,
    ReviewDecision,
)


def build_code_landing_trajectory(
    *,
    repo_hash: str,
    localizations: tuple[LocalizationHypothesis, ...],
    strategies: tuple[PatchStrategyPlan, ...],
    validations: tuple[PatchValidationResult, ...],
    selection: PatchSelectionDecision,
    review_decision: ReviewDecision,
    integration_decision: IntegrationDecision,
    guarded_decision: GuardedPatchDecision,
    landing_decision: LandingDecision,
) -> CodeLandingTrajectory:
    """Build a compact, deterministic trajectory artifact."""

    stages = (
        "environment",
        "repo_intelligence",
        "oracle",
        "localization",
        "strategy",
        "validation",
        "selection",
        "review",
        "guarded_kernel",
        "candidate_validation",
        "candidate_integration",
    )
    observations = (
        {"stage": "localization", "count": len(localizations)},
        {"stage": "strategy", "count": len(strategies)},
        {
            "stage": "validation",
            "passed": sum(1 for result in validations if result.repro_passed),
            "total": len(validations),
        },
        {"stage": "review", "status": review_decision.status},
        {"stage": "candidate_validation", "status": landing_decision.status},
    )
    failure_attribution = _failure_attribution(
        validations=validations,
        selection=selection,
        review_decision=review_decision,
        integration_decision=integration_decision,
        guarded_decision=guarded_decision,
        landing_decision=landing_decision,
    )
    lessons = _reusable_lessons(
        localizations=localizations,
        validations=validations,
        landing_decision=landing_decision,
    )
    payload: dict[str, Any] = {
        "repo_hash": repo_hash,
        "task_id": selection.task_id,
        "stages": stages,
        "observations": observations,
        "selection": selection,
        "review": review_decision,
        "integration": integration_decision,
        "landing": landing_decision,
    }
    return CodeLandingTrajectory(
        trajectory_id=f"trajectory_{stable_hash(payload)[:24]}",
        task_id=selection.task_id,
        repo_hash=repo_hash,
        stages=stages,
        tool_calls=(),
        observations=canonicalize(observations),
        patches=(selection.selected_patch_id,) if selection.selected_patch_id else (),
        validation_results=tuple(result.evidence_hash for result in validations),
        reviewer_findings=tuple(
            finding.finding_id for finding in review_decision.findings
        ),
        final_outcome=landing_decision.status,
        failure_attribution=failure_attribution,
        reusable_lessons=lessons,
    )


def _failure_attribution(
    *,
    validations: tuple[PatchValidationResult, ...],
    selection: PatchSelectionDecision,
    review_decision: ReviewDecision,
    integration_decision: IntegrationDecision,
    guarded_decision: GuardedPatchDecision,
    landing_decision: LandingDecision,
) -> tuple[str, ...]:
    failures: list[str] = []
    failures.extend(
        result.failure_summary
        for result in validations
        if result.failure_summary is not None
    )
    if selection.selected_patch_id is None:
        failures.append("no_selected_patch")
    if review_decision.status == "blocked":
        failures.append("semantic_review_blocked")
    if not guarded_decision.accepted:
        failures.extend(guarded_decision.blocked_reasons)
    if not integration_decision.candidate_coherent:
        failures.extend(integration_decision.reasons)
    if landing_decision.status not in {
        "candidate_validated",
        "candidate_validated_with_claim_downgrade",
    }:
        failures.extend(landing_decision.reasons)
    return tuple(dict.fromkeys(failures))


def _reusable_lessons(
    *,
    localizations: tuple[LocalizationHypothesis, ...],
    validations: tuple[PatchValidationResult, ...],
    landing_decision: LandingDecision,
) -> tuple[str, ...]:
    lessons: list[str] = []
    if any(loc.confidence < 0.5 for loc in localizations):
        lessons.append("increase_localization_budget_before_patch_generation")
    if any(result.test_only_patch for result in validations):
        lessons.append("discard_test_only_candidates_before_reranking")
    if any(result.broad_rewrite_detected for result in validations):
        lessons.append("split_broad_rewrites_into_behavior_backed_micro_patches")
    if landing_decision.status == "candidate_validated":
        lessons.append("reuse_confirmed_repro_and_localization_pattern")
    return tuple(dict.fromkeys(lessons))
