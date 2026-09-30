"""Claim-evidence certificates for Society-Core paper runs."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from .hashing import stable_hash


@dataclass(frozen=True)
class EvaluationFreezeBoundary:
    case_id: str
    public_input_hash: str
    evaluator_private_hash: str
    freeze_hash: str
    frozen_at_utc: str
    public_manifest: dict[str, Any]

    def hash(self) -> str:
        return stable_hash(self)


@dataclass(frozen=True)
class EvidenceDebt:
    debt_id: str
    severity: str
    reason: str
    claim_effect: str


@dataclass(frozen=True)
class ClaimWitness:
    witness_id: str
    theme: str
    support_stage: str
    support_score: float
    support_refs: tuple[str, ...]


@dataclass(frozen=True)
class SocietyClaimCertificate:
    certificate_id: str
    claim_id: str
    claim_level: int
    claim_status: str
    supported_claim: str
    freeze_boundary: EvaluationFreezeBoundary
    evidence_debts: tuple[EvidenceDebt, ...]
    weakest_witness: ClaimWitness | None
    support_hash: str
    challenges: tuple[dict[str, Any], ...] = ()

    def hash(self) -> str:
        return stable_hash(self)


def build_evaluation_freeze_boundary(
    *,
    case_id: str,
    public_inputs: dict[str, Any],
    evaluator_private_inputs: dict[str, Any],
    frozen_at_utc: str,
) -> EvaluationFreezeBoundary:
    public_input_hash = stable_hash(public_inputs)
    evaluator_private_hash = stable_hash(evaluator_private_inputs)
    public_manifest = {
        "case_id": case_id,
        "frozen_at_utc": frozen_at_utc,
        "public_input_hash": public_input_hash,
        "evaluator_private_hash": evaluator_private_hash,
        "privacy_boundary": "public_hashes_only_private_payload_excluded",
    }
    freeze_hash = stable_hash(
        {
            "case_id": case_id,
            "public_input_hash": public_input_hash,
            "evaluator_private_hash": evaluator_private_hash,
            "frozen_at_utc": frozen_at_utc,
        }
    )
    return EvaluationFreezeBoundary(
        case_id=case_id,
        public_input_hash=public_input_hash,
        evaluator_private_hash=evaluator_private_hash,
        freeze_hash=freeze_hash,
        frozen_at_utc=frozen_at_utc,
        public_manifest=public_manifest,
    )


def assess_evidence_debts(
    *,
    real_product_experience_summary: dict[str, Any],
    workspace_development_status: dict[str, Any],
    behavior_verified_overlap: float | None,
    hidden_eval_coverage: float | None,
    provider_provenance: dict[str, Any],
) -> tuple[EvidenceDebt, ...]:
    debts: list[EvidenceDebt] = []
    real_count = _safe_int(real_product_experience_summary.get("observation_count"))
    if real_count <= 0:
        debts.append(
            EvidenceDebt(
                debt_id="missing_real_product_experience",
                severity="major",
                reason="The claim lacks workspace-backed product-experience observations.",
                claim_effect="limit_to_simulated_experience_or_public_feedback_claim",
            )
        )
    if not bool(workspace_development_status.get("verified")):
        debts.append(
            EvidenceDebt(
                debt_id="workspace_patch_not_verified",
                severity="major",
                reason="Company-side code changes were not verified by runnable workspace checks.",
                claim_effect="do_not_claim_behavioral_product_improvement",
            )
        )
    if behavior_verified_overlap is None or behavior_verified_overlap <= 0.0:
        debts.append(
            EvidenceDebt(
                debt_id="missing_behavior_verified_overlap",
                severity="major",
                reason=(
                    "Historical target overlap has no positive support from "
                    "behavior-covered workspace patches."
                ),
                claim_effect="downgrade_from_behavior_match_to_directional_theme_match",
            )
        )
    if hidden_eval_coverage is not None and hidden_eval_coverage < 0.5:
        debts.append(
            EvidenceDebt(
                debt_id="low_hidden_eval_coverage",
                severity="minor",
                reason="Evaluator-private tests cover less than half of declared target behavior.",
                claim_effect="report_coverage_debt_with_overlap",
            )
        )
    cutoff_policy = str(provider_provenance.get("model_cutoff_policy") or "unknown")
    if cutoff_policy == "unknown":
        debts.append(
            EvidenceDebt(
                debt_id="unknown_model_cutoff_policy",
                severity="minor",
                reason="Model cutoff or contamination policy is not declared for this run.",
                claim_effect="keep_time_machine_result_as_retrospective_validation",
            )
        )
    return tuple(debts)


def select_weakest_public_witness(
    *,
    theme_support: dict[str, dict[str, float]] | None,
    support_refs_by_theme: dict[str, tuple[str, ...]] | None = None,
    eligible_themes: tuple[str, ...] | None = None,
) -> ClaimWitness | None:
    if not theme_support:
        return None
    support_refs_by_theme = support_refs_by_theme or {}
    eligible = set(eligible_themes) if eligible_themes is not None else None
    candidates: list[ClaimWitness] = []
    for theme, row in sorted(theme_support.items()):
        if eligible is not None and theme not in eligible:
            continue
        score = _support_score(row)
        refs = tuple(support_refs_by_theme.get(theme, ()))
        payload = {
            "theme": theme,
            "support_score": score,
            "support_refs": refs,
        }
        candidates.append(
            ClaimWitness(
                witness_id=f"public_witness_{stable_hash(payload)[:24]}",
                theme=theme,
                support_stage="public_feedback",
                support_score=score,
                support_refs=refs,
            )
        )
    if not candidates:
        return None
    return min(candidates, key=lambda witness: (witness.support_score, witness.theme))


def build_society_claim_certificate(
    *,
    claim_id: str,
    claim_level: int,
    supported_claim: str,
    freeze_boundary: EvaluationFreezeBoundary,
    evidence_debts: tuple[EvidenceDebt, ...],
    weakest_witness: ClaimWitness | None,
) -> SocietyClaimCertificate:
    adjusted_level = min(_clamp_claim_level(claim_level), _claim_level_ceiling(evidence_debts))
    status = "claim_ready" if adjusted_level >= claim_level and not evidence_debts else "claim_debt_limited"
    support_payload = {
        "claim_id": claim_id,
        "claim_level": adjusted_level,
        "supported_claim": supported_claim,
        "freeze_hash": freeze_boundary.freeze_hash,
        "evidence_debts": evidence_debts,
        "weakest_witness": weakest_witness,
    }
    support_hash = stable_hash(support_payload)
    return SocietyClaimCertificate(
        certificate_id=f"claim_certificate_{stable_hash(support_payload)[:24]}",
        claim_id=claim_id,
        claim_level=adjusted_level,
        claim_status=status,
        supported_claim=supported_claim,
        freeze_boundary=freeze_boundary,
        evidence_debts=evidence_debts,
        weakest_witness=weakest_witness,
        support_hash=support_hash,
    )


def apply_claim_challenge(
    certificate: SocietyClaimCertificate,
    *,
    challenge_id: str,
    requested_claim_level: int,
    severity: str,
    reason: str,
) -> SocietyClaimCertificate:
    requested = _clamp_claim_level(requested_claim_level)
    downgrade = {"minor": 1, "major": 2, "blocking": 5}.get(severity, 1)
    challenged_level = max(0, certificate.claim_level - downgrade)
    accepted = min(certificate.claim_level, requested, challenged_level)
    challenge = {
        "challenge_id": challenge_id,
        "requested_claim_level": requested,
        "accepted_claim_level": accepted,
        "severity": severity,
        "reason": reason,
    }
    support_payload = {
        "previous_support_hash": certificate.support_hash,
        "challenge": challenge,
    }
    return replace(
        certificate,
        certificate_id=f"claim_certificate_{stable_hash((certificate, challenge))[:24]}",
        claim_level=accepted,
        claim_status="challenged_downgraded" if accepted < certificate.claim_level else "challenged_preserved",
        support_hash=stable_hash(support_payload),
        challenges=(*certificate.challenges, challenge),
    )


def _support_score(row: dict[str, float]) -> float:
    count = max(0.0, float(row.get("count", 0.0)))
    priority = _safe_float01(row.get("mean_priority"))
    confidence = _safe_float01(row.get("mean_confidence"))
    return round(min(1.0, count / 10.0) * 0.4 + priority * 0.25 + confidence * 0.35, 12)


def _claim_level_ceiling(evidence_debts: tuple[EvidenceDebt, ...]) -> int:
    if any(debt.severity == "blocking" for debt in evidence_debts):
        return 1
    if any(debt.severity == "major" for debt in evidence_debts):
        return 3
    return 5


def _clamp_claim_level(value: int) -> int:
    return max(0, min(5, int(value)))


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _safe_float01(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, numeric))
