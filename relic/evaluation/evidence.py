"""Content-addressed evidence records emitted by the Relic evaluator."""

from __future__ import annotations

from dataclasses import dataclass

from relic.research.hashing import stable_hash

_OWNERS = frozenset({"evaluator", "project", "generated", "agent"})
_STATUSES = frozenset({"not_run", "passed", "failed", "blocked", "timeout", "infra_error"})


@dataclass(frozen=True)
class VerificationEvidence:
    """One independently hashable verification observation."""

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


__all__ = ["VerificationEvidence"]
