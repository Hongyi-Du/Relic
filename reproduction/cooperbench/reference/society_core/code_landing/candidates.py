"""Adapters from workspace-agent proposals to typed landing candidates."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from society_core.hashing import stable_hash

from .schemas import CandidatePatch


def candidate_from_proposal(
    proposal: Any,
    *,
    task_id: str,
    strategy: str,
    localization_hypothesis_id: str,
    oracle_ids: tuple[str, ...],
) -> CandidatePatch:
    patches = tuple(getattr(proposal, "patches", ()))
    changed_files = tuple(dict.fromkeys(str(patch.path) for patch in patches))
    added_tests = tuple(path for path in changed_files if _is_test_file(path))
    source = str(getattr(proposal, "source", "workspace_agent"))
    model = getattr(proposal, "model", None)
    producer = f"{source}:{model}" if model else source
    payload = {
        "proposal_id": getattr(proposal, "proposal_id", None),
        "task_id": task_id,
        "strategy": strategy,
        "patches": patches,
        "oracle_ids": oracle_ids,
        "verification_commands": tuple(
            str(command) for command in getattr(proposal, "verification_commands", ())
        ),
    }
    return CandidatePatch(
        patch_id=f"candidate_{stable_hash(payload)[:24]}",
        task_id=task_id,
        strategy=strategy,
        produced_by_agent=producer,
        unified_diff="",
        workspace_patches=patches,
        changed_files=changed_files,
        changed_symbols=(),
        added_tests=added_tests,
        rationale=str(getattr(proposal, "rationale", ""))[:4000],
        expected_behavior_change=str(getattr(proposal, "rationale", ""))[:2000],
        risk_notes=(),
        compatibility_notes=(),
        support_refs=tuple(getattr(proposal, "support_refs", ())),
        parent_localization_hypothesis_id=localization_hypothesis_id,
        parent_oracle_ids=oracle_ids,
        verification_commands=tuple(
            str(command) for command in getattr(proposal, "verification_commands", ())
        ),
    )


def _is_test_file(path: str) -> bool:
    normalized = Path(path).as_posix().lower()
    name = Path(normalized).name
    return (
        normalized.startswith(("test/", "tests/", "__tests__/"))
        or "/test/" in normalized
        or "/tests/" in normalized
        or name.startswith("test_")
        or name.endswith(("_test.py", ".test.js", ".spec.js", ".test.ts", ".spec.ts"))
    )
