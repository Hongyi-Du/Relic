"""Final guarded patch and landing gate for Code-Max runs."""

from __future__ import annotations

from pathlib import Path

from society_core.hashing import stable_hash
from society_core.workspace_update import apply_workspace_patches

from .schemas import (
    CandidatePatch,
    GuardedPatchDecision,
    LandingDecision,
    PatchSelectionDecision,
    PatchValidationResult,
    ReviewDecision,
)
from .validation_matrix import complex_change_evidence_ready


def guard_selected_patch(
    *,
    workspace_root: Path,
    selection: PatchSelectionDecision,
    candidates: tuple[CandidatePatch, ...],
    validations: tuple[PatchValidationResult, ...],
) -> GuardedPatchDecision:
    """Run the final non-mutating patch gate on the selected candidate."""

    candidate = _candidate(selection, candidates)
    if candidate is None:
        return GuardedPatchDecision(
            patch_id="none",
            accepted=False,
            applied_diff_hash=None,
            blocked_reasons=("no_selected_patch",),
            changed_files=(),
            effective_product_change=False,
        )
    validation = _validation(candidate.patch_id, validations)
    dry_run = apply_workspace_patches(
        workspace_root,
        candidate.workspace_patches,
        dry_run=True,
    )
    blocked_reasons: list[str] = [
        result.blocked_reason or "patch_not_accepted"
        for result in dry_run
        if result.status == "blocked"
    ]
    if validation is None:
        blocked_reasons.append("missing_validation")
    elif not validation.applies_cleanly:
        blocked_reasons.append("validation_patch_did_not_apply")
    if validation and validation.test_only_patch:
        blocked_reasons.append("test_only_patch")
    if validation and not complex_change_evidence_ready(validation):
        blocked_reasons.append("broad_rewrite_without_behavior_evidence")
    effective_product_change = bool(candidate.changed_files) and not all(
        _is_test_file(path) for path in candidate.changed_files
    )
    if not effective_product_change:
        blocked_reasons.append("no_effective_product_change")
    accepted = not blocked_reasons and bool(dry_run)
    diff_hash = stable_hash(
        {
            "candidate": candidate.patch_id,
            "diff": candidate.unified_diff,
            "patches": candidate.workspace_patches,
            "dry_run": dry_run,
        }
    )
    return GuardedPatchDecision(
        patch_id=candidate.patch_id,
        accepted=accepted,
        applied_diff_hash=diff_hash if accepted else None,
        blocked_reasons=tuple(dict.fromkeys(blocked_reasons)),
        changed_files=candidate.changed_files,
        effective_product_change=effective_product_change,
    )


def decide_landing(
    *,
    selection: PatchSelectionDecision,
    review_decision: ReviewDecision,
    guarded_decision: GuardedPatchDecision,
    validations: tuple[PatchValidationResult, ...],
) -> LandingDecision:
    """Convert gate evidence into a bounded evidence claim."""

    if selection.selected_patch_id is None:
        return LandingDecision(
            task_id=selection.task_id,
            patch_id=None,
            status="blocked",
            claim_level="proposal_only",
            reasons=("no_selected_patch",),
            evidence_refs=selection.rejected_patch_ids,
        )
    validation = _validation(selection.selected_patch_id, validations)
    reasons: list[str] = []
    if not guarded_decision.accepted:
        reasons.extend(guarded_decision.blocked_reasons or ("guarded_kernel_rejected",))
    if review_decision.status == "blocked":
        reasons.append("semantic_review_blocked")
        reasons.extend(
            f"review:{finding.code}"
            for finding in review_decision.findings
            if finding.severity == "blocker"
        )
    if validation is None:
        reasons.append("missing_validation")
    elif not validation.repro_passed:
        reasons.append("behavior_oracle_not_passed")

    if reasons:
        return LandingDecision(
            task_id=selection.task_id,
            patch_id=selection.selected_patch_id,
            status="needs_revision",
            claim_level="patch_candidate",
            reasons=tuple(dict.fromkeys(reasons)),
            evidence_refs=_evidence_refs(validation, review_decision, guarded_decision),
        )

    claim_level = "verified_patch_candidate"
    if validation and validation.repro_passed:
        claim_level = "behavior_verified_patch_candidate"
    if review_decision.claim_ceiling == "architecture_api_aligned_patch":
        claim_level = "architecture_api_aligned_patch_candidate"
    review_ceiling = _candidate_claim(review_decision.claim_ceiling)
    status = "candidate_validated"
    if review_ceiling not in {claim_level, "behavior_verified_patch_candidate"}:
        status = "candidate_validated_with_claim_downgrade"
        claim_level = min(
            (claim_level, review_ceiling),
            key=_claim_rank,
        )
    return LandingDecision(
        task_id=selection.task_id,
        patch_id=selection.selected_patch_id,
        status=status,
        claim_level=claim_level,
        reasons=("candidate_passed_guarded_kernel_and_review",),
        evidence_refs=_evidence_refs(validation, review_decision, guarded_decision),
    )


def _evidence_refs(
    validation: PatchValidationResult | None,
    review_decision: ReviewDecision,
    guarded_decision: GuardedPatchDecision,
) -> tuple[str, ...]:
    refs: list[str] = []
    if validation is not None:
        refs.append(validation.evidence_hash)
    refs.extend(finding.finding_id for finding in review_decision.findings)
    if guarded_decision.applied_diff_hash:
        refs.append(guarded_decision.applied_diff_hash)
    return tuple(dict.fromkeys(refs))


def _candidate(
    selection: PatchSelectionDecision,
    candidates: tuple[CandidatePatch, ...],
) -> CandidatePatch | None:
    if selection.selected_patch_id is None:
        return None
    return next(
        (candidate for candidate in candidates if candidate.patch_id == selection.selected_patch_id),
        None,
    )


def _validation(
    patch_id: str,
    validations: tuple[PatchValidationResult, ...],
) -> PatchValidationResult | None:
    return next((result for result in validations if result.patch_id == patch_id), None)


def _is_test_file(path: str) -> bool:
    lowered = path.lower()
    return lowered.startswith(("test/", "tests/", "__tests__/")) or lowered.endswith(
        (".test.js", ".spec.js", "_test.py")
    )


def _claim_rank(claim: str) -> int:
    ranks = {
        "proposal_only": 0,
        "patch_candidate": 1,
        "patch_applied": 2,
        "verified_patch_candidate": 3,
        "behavior_verified_patch_candidate": 4,
        "architecture_api_aligned_patch_candidate": 5,
        "release_candidate": 6,
    }
    return ranks.get(claim, 0)


def _candidate_claim(claim: str) -> str:
    aliases = {
        "verified_workspace_patch": "verified_patch_candidate",
        "behavior_verified_patch": "behavior_verified_patch_candidate",
        "architecture_api_aligned_patch": "architecture_api_aligned_patch_candidate",
    }
    return aliases.get(claim, claim)
