"""Independent evidence ledger and explicit code-landing claim levels."""

from __future__ import annotations

from dataclasses import dataclass

from society_core.hashing import canonicalize, stable_hash


_OWNERS = frozenset({"evaluator", "project", "generated", "agent"})
_STATUSES = frozenset({"not_run", "passed", "failed", "blocked", "timeout", "infra_error"})
_EVIDENCE_RANK = {
    "candidate_failed": 0,
    "security_blocked": 0,
    "regression": 0,
    "legacy_verified_unclassified": 0,
    "proposal_ready": 0,
    "candidate_applied": 1,
    "targeted_verified": 2,
    "project_suite_verified": 3,
    "integration_verified": 4,
    "release_candidate": 5,
}


@dataclass(frozen=True)
class VerificationEvidence:
    evidence_id: str
    task_id: str
    command: str
    owner: str
    kind: str
    base_status: str
    candidate_status: str
    required: bool
    evidence_hash: str
    baseline_repo_digest: str = ""
    candidate_repo_digest: str = ""
    execution_policy_hash: str = ""
    exit_code: int | None = None
    stdout_hash: str | None = None
    stderr_hash: str | None = None

    @classmethod
    def create(
        cls,
        *,
        task_id: str,
        command: str,
        owner: str,
        kind: str,
        base_status: str,
        candidate_status: str,
        required: bool,
        baseline_repo_digest: str = "",
        candidate_repo_digest: str = "",
        execution_policy_hash: str = "",
        exit_code: int | None = None,
        stdout_hash: str | None = None,
        stderr_hash: str | None = None,
    ) -> VerificationEvidence:
        if owner not in _OWNERS:
            raise ValueError(f"unsupported_evidence_owner:{owner}")
        if base_status not in _STATUSES:
            raise ValueError(f"unsupported_base_status:{base_status}")
        if candidate_status not in _STATUSES:
            raise ValueError(f"unsupported_candidate_status:{candidate_status}")
        if not command.strip():
            raise ValueError("evidence_command_required")
        payload = {
            "task_id": task_id,
            "command": command,
            "owner": owner,
            "kind": kind,
            "base_status": base_status,
            "candidate_status": candidate_status,
            "required": required,
            "baseline_repo_digest": baseline_repo_digest,
            "candidate_repo_digest": candidate_repo_digest,
            "execution_policy_hash": execution_policy_hash,
            "exit_code": exit_code,
            "stdout_hash": stdout_hash,
            "stderr_hash": stderr_hash,
        }
        evidence_hash = stable_hash(payload)
        return cls(
            evidence_id=f"verification_evidence_{evidence_hash[:24]}",
            task_id=task_id,
            command=command,
            owner=owner,
            kind=kind,
            base_status=base_status,
            candidate_status=candidate_status,
            required=required,
            evidence_hash=evidence_hash,
            baseline_repo_digest=baseline_repo_digest,
            candidate_repo_digest=candidate_repo_digest,
            execution_policy_hash=execution_policy_hash,
            exit_code=exit_code,
            stdout_hash=stdout_hash,
            stderr_hash=stderr_hash,
        )


@dataclass(frozen=True)
class EvidenceLedger:
    records: tuple[VerificationEvidence, ...] = ()

    def append(self, evidence: VerificationEvidence) -> EvidenceLedger:
        if any(item.evidence_hash == evidence.evidence_hash for item in self.records):
            return self
        return EvidenceLedger(records=(*self.records, evidence))

    def extend(self, evidence: tuple[VerificationEvidence, ...]) -> EvidenceLedger:
        ledger = self
        for item in evidence:
            ledger = ledger.append(item)
        return ledger

    @property
    def ledger_hash(self) -> str:
        return stable_hash(tuple(canonicalize(item) for item in self.records))

    @property
    def replay_commands(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                item.command
                for item in self.current_records
                if item.required and item.candidate_status == "passed"
            )
        )

    @property
    def current_records(self) -> tuple[VerificationEvidence, ...]:
        latest: dict[tuple[str, str, str, str], VerificationEvidence] = {}
        order: list[tuple[str, str, str, str]] = []
        for item in self.records:
            key = (item.task_id, item.command, item.owner, item.kind)
            if key not in latest:
                order.append(key)
            latest[key] = item
        return tuple(latest[key] for key in order)


@dataclass(frozen=True)
class LandingEvidenceAssessment:
    level: str
    causal_oracle_count: int
    project_gate_count: int
    replay_commands: tuple[str, ...]
    failed_required_commands: tuple[str, ...]
    blocking_reasons: tuple[str, ...]
    evidence_hash: str
    context_hash: str = ""


def evidence_level_at_least(level: str, minimum: str) -> bool:
    """Compare claim levels without treating unknown or legacy labels as proof."""

    if minimum not in _EVIDENCE_RANK:
        raise ValueError(f"unknown_minimum_evidence_level:{minimum}")
    return _EVIDENCE_RANK.get(level, 0) >= _EVIDENCE_RANK[minimum]


def assess_landing_evidence(
    *,
    ledger: EvidenceLedger,
    effective_patch: bool,
    integration_replayed: bool,
    promotion_status: str | None,
    promotion_records: tuple[dict[str, object], ...] = (),
    event_chain_head: str | None = None,
) -> LandingEvidenceAssessment:
    required = tuple(item for item in ledger.current_records if item.required)
    failed_required = tuple(
        dict.fromkeys(
            item.command for item in required if item.candidate_status != "passed"
        )
    )
    causal = tuple(
        item
        for item in required
        if item.owner == "evaluator"
        and item.kind == "behavior"
        and item.base_status == "failed"
        and item.candidate_status == "passed"
    )
    project_gates = tuple(
        item
        for item in required
        if item.owner == "project" and item.kind == "test_suite"
    )
    security_failures = tuple(
        item
        for item in required
        if item.kind == "security" and item.candidate_status != "passed"
    )

    blocking_reasons: list[str] = []
    if not effective_patch:
        level = "candidate_failed"
        blocking_reasons.append("missing_effective_patch")
    elif security_failures:
        level = "security_blocked"
        blocking_reasons.append("required_security_gate_failed")
    elif failed_required:
        level = "regression" if integration_replayed or promotion_status else "candidate_failed"
        blocking_reasons.append("required_verification_failed")
    elif not causal:
        level = "candidate_applied"
        blocking_reasons.append("missing_evaluator_owned_causal_oracle")
    else:
        level = "targeted_verified"
        if not project_gates:
            blocking_reasons.append("project_suite_not_executed")
        elif all(item.candidate_status == "passed" for item in project_gates):
            level = "project_suite_verified"
            if integration_replayed:
                level = "integration_verified"
            else:
                blocking_reasons.append("integration_oracles_not_replayed")
        if (
            promotion_status == "release_candidate"
            and integration_replayed
            and project_gates
            and level == "integration_verified"
        ):
            level = "release_candidate"
        elif promotion_status not in {None, "release_candidate"}:
            blocking_reasons.append(f"promotion_status:{promotion_status}")

    payload = {
        "ledger_hash": ledger.ledger_hash,
        "effective_patch": effective_patch,
        "integration_replayed": integration_replayed,
        "promotion_status": promotion_status,
        "level": level,
        "failed_required": failed_required,
        "blocking_reasons": blocking_reasons,
        "promotion_records": promotion_records,
        "event_chain_head": event_chain_head,
    }
    context_hash = stable_hash(
        {
            "promotion_records": promotion_records,
            "event_chain_head": event_chain_head,
        }
    )
    return LandingEvidenceAssessment(
        level=level,
        causal_oracle_count=len(causal),
        project_gate_count=len(project_gates),
        replay_commands=ledger.replay_commands,
        failed_required_commands=failed_required,
        blocking_reasons=tuple(blocking_reasons),
        evidence_hash=stable_hash(payload),
        context_hash=context_hash,
    )
