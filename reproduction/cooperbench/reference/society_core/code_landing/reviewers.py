"""Semantic review board for Code-Max candidate patches."""

from __future__ import annotations

from typing import Any, Mapping

from society_core.hashing import stable_hash

from .schemas import (
    CandidatePatch,
    PatchSelectionDecision,
    PatchValidationResult,
    ReviewDecision,
    ReviewerFinding,
)
from .surfaces import has_security_surface
from .validation_matrix import complex_change_evidence_ready


def review_selected_patch(
    *,
    selection: PatchSelectionDecision,
    candidates: tuple[CandidatePatch, ...],
    validations: tuple[PatchValidationResult, ...],
    task_specs: tuple[Any, ...] = (),
    reviewer_roles: tuple[str, ...] = (),
) -> ReviewDecision:
    """Attack the selected patch before it may become evidence."""

    selected = _selected_candidate(selection, candidates)
    if selected is None:
        return ReviewDecision(
            patch_id=None,
            status="blocked",
            findings=(
                _finding(
                    patch_id="none",
                    role="evidence_boundary",
                    severity="blocker",
                    code="no_patch_survived_selection",
                    summary="No patch survived validation and selection.",
                    evidence_refs=selection.rejected_patch_ids,
                ),
            ),
            reviewer_roles=("evidence_boundary",),
            claim_ceiling="proposal_only",
        )

    validation = _validation_for(selected.patch_id, validations)
    task = _task_for(selected.task_id, task_specs)
    roles = reviewer_roles or _roles_for_candidate(selected, task)
    findings: list[ReviewerFinding] = []
    for role in roles:
        findings.extend(_review_role(role, selected, validation, task))

    status = "passed"
    if any(finding.severity == "blocker" for finding in findings):
        status = "blocked"
    elif any(finding.severity == "warning" for finding in findings):
        status = "passed_with_warnings"
    return ReviewDecision(
        patch_id=selected.patch_id,
        status=status,
        findings=tuple(findings),
        reviewer_roles=roles,
        claim_ceiling=_claim_ceiling(validation, findings),
    )


def _review_role(
    role: str,
    candidate: CandidatePatch,
    validation: PatchValidationResult | None,
    task: Any | None,
) -> tuple[ReviewerFinding, ...]:
    if role == "intent":
        return _intent_findings(candidate, validation, task)
    if role == "api":
        return _api_findings(candidate, validation, task)
    if role == "security":
        return _security_findings(candidate, validation)
    if role == "regression":
        return _regression_findings(candidate, validation)
    if role == "architecture":
        return _architecture_findings(candidate, validation)
    if role == "developer_experience":
        return _dx_findings(candidate)
    if role == "test_quality":
        return _test_quality_findings(candidate, validation)
    if role == "evidence_boundary":
        return _evidence_boundary_findings(candidate, validation)
    return ()


def _intent_findings(
    candidate: CandidatePatch,
    validation: PatchValidationResult | None,
    task: Any | None,
) -> tuple[ReviewerFinding, ...]:
    if validation is None or not validation.repro_passed:
        return (
            _finding(
                patch_id=candidate.patch_id,
                role="intent",
                severity="blocker",
                code="behavior_oracle_failed",
                summary="Selected patch does not pass a behavior oracle.",
                evidence_refs=candidate.parent_oracle_ids,
                suggested_probe="Run or strengthen the issue-specific repro.",
            ),
        )
    expected = _field(task, "expected_behavior", "")
    if expected and not candidate.expected_behavior_change:
        return (
            _finding(
                patch_id=candidate.patch_id,
                role="intent",
                severity="warning",
                code="weak_expected_behavior_rationale",
                summary="Patch has behavior evidence but weak expected-behavior rationale.",
                evidence_refs=candidate.support_refs,
            ),
        )
    return ()


def _api_findings(
    candidate: CandidatePatch,
    validation: PatchValidationResult | None,
    task: Any | None,
) -> tuple[ReviewerFinding, ...]:
    if validation is None or not validation.public_api_changed:
        return ()
    task_type = str(_field(task, "task_type", ""))
    behavior_surface = tuple(
        str(item).lower() for item in _field(task, "behavior_surface", ())
    )
    allows_api = task_type in {
        "api_change",
        "api_correctness",
        "compatibility",
        "feature",
    } or _has_api_surface(behavior_surface)
    compatibility_evidence = getattr(
        validation,
        "public_contract_continuity_passed",
        None,
    )
    if (allows_api or candidate.compatibility_notes) and compatibility_evidence is False:
        return (
            _finding(
                patch_id=candidate.patch_id,
                role="api",
                severity="blocker",
                code="public_api_continuity_failed",
                summary="Patch changes a public API surface and compatibility checks failed.",
                evidence_refs=candidate.changed_files,
                suggested_probe="Repair and rerun public API continuity checks.",
            ),
        )
    if allows_api or any("api" in note.lower() for note in candidate.compatibility_notes):
        return (
            _finding(
                patch_id=candidate.patch_id,
                role="api",
                severity="warning",
                code="public_api_evidence_required",
                summary="Patch changes a public API surface; compatibility evidence is required.",
                evidence_refs=candidate.changed_files,
                suggested_probe="Run public API compatibility examples.",
            ),
        )
    return (
        _finding(
            patch_id=candidate.patch_id,
            role="api",
            severity="blocker",
            code="undeclared_public_api_change",
            summary="Patch changes a public API surface without an API-change task.",
            evidence_refs=candidate.changed_files,
            suggested_patch_constraint="Preserve public exports or update the task spec.",
        ),
    )


def _security_findings(
    candidate: CandidatePatch,
    validation: PatchValidationResult | None,
) -> tuple[ReviewerFinding, ...]:
    risky_text = "\n".join(
        (candidate.unified_diff, candidate.rationale, candidate.expected_behavior_change)
    ).lower()
    if any(token in risky_text for token in ("secret", "token", "api_key", "password")):
        return (
            _finding(
                patch_id=candidate.patch_id,
                role="security",
                severity="blocker",
                code="credential_like_material",
                summary="Patch text appears to introduce credential-like material.",
                evidence_refs=candidate.changed_files,
                suggested_patch_constraint="Remove secrets and use environment variables.",
            ),
        )
    if validation and validation.security_scan_passed is False:
        return (
            _finding(
                patch_id=candidate.patch_id,
                role="security",
                severity="blocker",
                code="security_scan_failed",
                summary="Security scan failed for the selected patch.",
                evidence_refs=candidate.changed_files,
            ),
        )
    return ()


def _regression_findings(
    candidate: CandidatePatch,
    validation: PatchValidationResult | None,
) -> tuple[ReviewerFinding, ...]:
    if validation is None:
        return ()
    if validation.regression_tests_passed is False:
        return (
            _finding(
                patch_id=candidate.patch_id,
                role="regression",
                severity="blocker",
                code="regression_tests_failed",
                summary="Regression tests failed for the selected patch.",
                evidence_refs=candidate.changed_files,
            ),
        )
    if validation.broad_rewrite_detected and not complex_change_evidence_ready(
        validation
    ):
        return (
            _finding(
                patch_id=candidate.patch_id,
                role="regression",
                severity="blocker",
                code="broad_rewrite_evidence_missing",
                summary="Broad rewrite detected without enough regression evidence.",
                evidence_refs=candidate.changed_files,
                suggested_probe="Run a wider regression subset before promotion.",
            ),
        )
    return ()


def _architecture_findings(
    candidate: CandidatePatch,
    validation: PatchValidationResult | None,
) -> tuple[ReviewerFinding, ...]:
    if (
        validation is None
        or not validation.broad_rewrite_detected
        or complex_change_evidence_ready(validation)
    ):
        return ()
    return (
        _finding(
            patch_id=candidate.patch_id,
            role="architecture",
            severity="blocker",
            code="architecture_review_missing",
            summary="Architecture-scale change lacks an explicit architecture review.",
            evidence_refs=candidate.changed_files,
            suggested_patch_constraint="Split into smaller behavior-backed patches.",
        ),
    )


def _dx_findings(candidate: CandidatePatch) -> tuple[ReviewerFinding, ...]:
    if "diagnostic" not in candidate.rationale.lower() and not any(
        "readme" in path.lower() or "docs" in path.lower()
        for path in candidate.changed_files
    ):
        return ()
    if candidate.expected_behavior_change:
        return ()
    return (
        _finding(
            patch_id=candidate.patch_id,
            role="developer_experience",
            severity="warning",
            code="user_visible_change_unspecified",
            summary="Developer-experience patch should state the user-visible change.",
            evidence_refs=candidate.changed_files,
        ),
    )


def _test_quality_findings(
    candidate: CandidatePatch,
    validation: PatchValidationResult | None,
) -> tuple[ReviewerFinding, ...]:
    if (
        validation
        and candidate.added_tests
        and validation.added_test_verification_status != "passed"
    ):
        status = validation.added_test_verification_status
        return (
            _finding(
                patch_id=candidate.patch_id,
                role="test_quality",
                severity="blocker",
                code="added_tests_not_verified",
                summary=(
                    "Candidate changes tests, but those exact tests were not verified "
                    f"successfully (status={status})."
                ),
                evidence_refs=candidate.added_tests,
                suggested_probe=(
                    "Run an isolated command that names every changed test. If the "
                    "project test environment is unavailable, remove unverified test "
                    "edits and rely on evaluator-owned behavior probes."
                ),
                suggested_patch_constraint=(
                    "Do not include test-file changes that cannot be executed in the "
                    "frozen environment."
                ),
                details=(("verification_status", status),),
            ),
        )
    if validation and validation.test_only_patch:
        return (
            _finding(
                patch_id=candidate.patch_id,
                role="test_quality",
                severity="blocker",
                code="test_only_patch",
                summary="Selected patch only changes tests.",
                evidence_refs=candidate.changed_files,
            ),
        )
    return ()


def _evidence_boundary_findings(
    candidate: CandidatePatch,
    validation: PatchValidationResult | None,
) -> tuple[ReviewerFinding, ...]:
    if validation is None:
        return (
            _finding(
                patch_id=candidate.patch_id,
                role="evidence_boundary",
                severity="blocker",
                code="missing_validation",
                summary="Selected patch has no validation record.",
                evidence_refs=(candidate.patch_id,),
            ),
        )
    if not validation.applies_cleanly:
        return (
            _finding(
                patch_id=candidate.patch_id,
                role="evidence_boundary",
                severity="blocker",
                code="patch_did_not_apply",
                summary="Selected patch did not apply cleanly.",
                evidence_refs=(validation.patch_id,),
            ),
        )
    if not validation.repro_passed:
        return (
            _finding(
                patch_id=candidate.patch_id,
                role="evidence_boundary",
                severity="blocker",
                code="behavior_claim_unsupported",
                summary="Behavior claim is not supported by an oracle pass.",
                evidence_refs=(validation.patch_id,),
            ),
        )
    if not validation.dimension_coverage_complete:
        return (
            _finding(
                patch_id=candidate.patch_id,
                role="evidence_boundary",
                severity="blocker",
                code="required_contract_dimension_failed",
                summary="One or more required public contract dimensions did not pass.",
                evidence_refs=tuple(
                    f"{dimension_id}:{status}"
                    for dimension_id, status in validation.dimension_status_by_id.items()
                    if status != "passed"
                ),
                suggested_probe=(
                    "Repair every failed public dimension witness and rerun the full "
                    "controller-owned dimension plan."
                ),
            ),
        )
    return ()


def _claim_ceiling(
    validation: PatchValidationResult | None,
    findings: list[ReviewerFinding],
) -> str:
    if any(finding.severity == "blocker" for finding in findings):
        return "patch_candidate"
    if validation is None:
        return "proposal_only"
    if validation.repro_passed and not validation.public_api_changed:
        return "behavior_verified_patch"
    if validation.repro_passed and validation.public_api_changed:
        return "architecture_api_aligned_patch"
    if validation.applies_cleanly:
        return "verified_workspace_patch"
    return "patch_candidate"


def _roles_for_candidate(candidate: CandidatePatch, task: Any | None) -> tuple[str, ...]:
    roles = ["intent", "regression", "test_quality", "evidence_boundary"]
    surface = tuple(str(item) for item in _field(task, "behavior_surface", ()))
    theme = str(_field(task, "source_theme", ""))
    if any("api" in path.lower() for path in candidate.changed_files) or _has_api_surface(
        tuple(item.lower() for item in surface)
    ):
        roles.append("api")
    if theme in {"path_security", "dependency_resolution"} or has_security_surface(
        surface, (theme,)
    ):
        roles.append("security")
    if theme in {"architecture_generalization", "framework_generalization"}:
        roles.append("architecture")
    if theme == "diagnostics_and_recovery":
        roles.append("developer_experience")
    return tuple(dict.fromkeys(roles))


def _has_api_surface(surface: tuple[str, ...]) -> bool:
    return any(
        item == "api" or item.startswith("api_") or item.endswith("_api")
        for item in surface
    )


def _selected_candidate(
    selection: PatchSelectionDecision,
    candidates: tuple[CandidatePatch, ...],
) -> CandidatePatch | None:
    if selection.selected_patch_id is None:
        return None
    return next(
        (candidate for candidate in candidates if candidate.patch_id == selection.selected_patch_id),
        None,
    )


def _validation_for(
    patch_id: str,
    validations: tuple[PatchValidationResult, ...],
) -> PatchValidationResult | None:
    return next((result for result in validations if result.patch_id == patch_id), None)


def _task_for(task_id: str, task_specs: tuple[Any, ...]) -> Any | None:
    return next((task for task in task_specs if _field(task, "task_id", None) == task_id), None)


def _finding(
    *,
    patch_id: str,
    role: str,
    severity: str,
    code: str,
    summary: str,
    evidence_refs: tuple[str, ...],
    suggested_probe: str | None = None,
    suggested_patch_constraint: str | None = None,
    remediation_owner: str = "candidate",
    retryable: bool = True,
    details: tuple[tuple[str, str], ...] = (),
) -> ReviewerFinding:
    payload = {
        "patch_id": patch_id,
        "role": role,
        "severity": severity,
        "code": code,
        "summary": summary,
        "evidence_refs": evidence_refs,
        "remediation_owner": remediation_owner,
        "retryable": retryable,
        "details": details,
    }
    return ReviewerFinding(
        finding_id=f"finding_{stable_hash(payload)[:16]}",
        patch_id=patch_id,
        reviewer_role=role,
        severity=severity,
        summary=summary,
        evidence_refs=evidence_refs,
        suggested_probe=suggested_probe,
        suggested_patch_constraint=suggested_patch_constraint,
        code=code,
        remediation_owner=remediation_owner,
        retryable=retryable,
        details=details,
    )


def _field(obj: Any | None, name: str, default: Any) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)
