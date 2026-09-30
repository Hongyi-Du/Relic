"""Patch candidate reranking for Code-Max runs."""

from __future__ import annotations

from .schemas import CandidatePatch, PatchSelectionDecision, PatchValidationResult


def select_best_patch(
    *,
    candidates: tuple[CandidatePatch, ...],
    validations: tuple[PatchValidationResult, ...],
) -> PatchSelectionDecision:
    validation_by_id = {result.patch_id: result for result in validations}
    rejected: list[str] = []
    eligible: list[tuple[PatchValidationResult, CandidatePatch]] = []
    for candidate in candidates:
        validation = validation_by_id.get(candidate.patch_id)
        if validation is None:
            rejected.append(candidate.patch_id)
            continue
        if not validation.applies_cleanly:
            rejected.append(candidate.patch_id)
            continue
        if validation.test_only_patch:
            rejected.append(candidate.patch_id)
            continue
        if not validation.repro_passed:
            rejected.append(candidate.patch_id)
            continue
        if (
            validation.candidate_verification_results
            and not validation.candidate_verification_passed
        ):
            rejected.append(candidate.patch_id)
            continue
        if not getattr(validation, "public_contract_continuity_passed", True):
            rejected.append(candidate.patch_id)
            continue
        if not getattr(validation, "dimension_coverage_complete", True):
            rejected.append(candidate.patch_id)
            continue
        eligible.append((validation, candidate))
    if not eligible:
        return PatchSelectionDecision(
            task_id=candidates[0].task_id if candidates else None,
            selected_patch_id=None,
            rejected_patch_ids=tuple(dict.fromkeys(rejected)),
            selection_reason="no_candidate_passed_required_validation",
            remaining_risks=("no_landed_candidate",),
            required_followup=("generate_more_candidates_or_strengthen_oracles",),
        )
    has_peer_contracts = any(
        getattr(validation, "cross_candidate_contract_total", 0) > 0
        or getattr(validation, "qualified_added_contract_count", 0) > 0
        for validation, _ in eligible
    )
    eligible.sort(
        key=lambda item: (
            -_peer_contract_tier(item[0], has_peer_contracts=has_peer_contracts),
            -_functional_contract_rate(item[0]),
            -int(item[0].candidate_behavior_verification_passed),
            -_ownership_strategy_tier(item[1]),
            -int(item[0].added_test_verification_passed),
            -_functional_contract_count(item[0]),
            -getattr(item[0], "qualified_added_contract_count", 0),
            -item[0].score,
            item[0].diff_size,
            item[0].files_touched_count,
            item[1].patch_id,
        )
    )
    selected_validation, selected = eligible[0]
    rejected.extend(candidate.patch_id for _, candidate in eligible[1:])
    remaining_risks = list(selected.risk_notes)
    required_followup: list[str] = []
    peer_total = getattr(selected_validation, "cross_candidate_contract_total", 0)
    peer_passed = getattr(selected_validation, "cross_candidate_contract_pass_count", 0)
    if peer_total > peer_passed:
        remaining_risks.append(f"peer_contract_residual:{peer_passed}/{peer_total}")
        remaining_risks.extend(
            getattr(selected_validation, "cross_candidate_contract_failures", ())
        )
        required_followup.append("resolve_failed_cross_candidate_contracts")
    elif has_peer_contracts and peer_total == 0:
        remaining_risks.append("not_independently_challenged_by_peer_contract")
    return PatchSelectionDecision(
        task_id=selected.task_id,
        selected_patch_id=selected.patch_id,
        rejected_patch_ids=tuple(dict.fromkeys(rejected)),
        selection_reason=(
            "selected_function_first:"
            "contract_tier:"
            f"{_peer_contract_tier(selected_validation, has_peer_contracts=has_peer_contracts)}:"
            f"validation_score:{selected_validation.score:.4f}:"
            "peer_contracts:"
            f"{getattr(selected_validation, 'cross_candidate_contract_pass_count', 0)}/"
            f"{getattr(selected_validation, 'cross_candidate_contract_total', 0)}:"
            f"ownership_tier:{_ownership_strategy_tier(selected)}:"
            f"effective_diff_size:{selected_validation.diff_size}:"
            f"files_touched:{selected_validation.files_touched_count}"
        ),
        remaining_risks=tuple(dict.fromkeys(remaining_risks)),
        required_followup=tuple(dict.fromkeys(required_followup)),
    )


def _ownership_strategy_tier(candidate: CandidatePatch) -> int:
    strategy = candidate.strategy
    if "declared_component_ownership" in strategy:
        return 2
    if any(
        marker in strategy
        for marker in (
            "adversarial_boundary_matrix",
            "architecture_preserving_systemic",
            "integration_synthesis",
        )
    ):
        return 1
    return 0


def _cross_candidate_contract_rate(validation: PatchValidationResult) -> float:
    total = getattr(validation, "cross_candidate_contract_total", 0)
    if total <= 0:
        return 0.0
    return getattr(validation, "cross_candidate_contract_pass_count", 0) / total


def _functional_contract_rate(validation: PatchValidationResult) -> float:
    peer_rate = _cross_candidate_contract_rate(validation)
    if getattr(validation, "cross_candidate_contract_total", 0) > 0:
        return peer_rate
    return 1.0 if getattr(validation, "qualified_added_contract_count", 0) else 0.0


def _functional_contract_count(validation: PatchValidationResult) -> int:
    return getattr(validation, "cross_candidate_contract_pass_count", 0) + getattr(
        validation, "qualified_added_contract_count", 0
    )


def _peer_contract_tier(
    validation: PatchValidationResult,
    *,
    has_peer_contracts: bool,
) -> int:
    """Rank independently challenged behavior before scalar patch quality."""

    total = getattr(validation, "cross_candidate_contract_total", 0)
    passed = getattr(validation, "cross_candidate_contract_pass_count", 0)
    if total > 0:
        return 3 if passed == total else 0
    if getattr(validation, "qualified_added_contract_count", 0) > 0:
        return 3
    return 1 if has_peer_contracts else 0
