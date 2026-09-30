"""Acceptance-oracle construction for Code-Max runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from society_core.execution import CommandExecutor
from society_core.hashing import stable_hash
from society_core.workspace_transaction import CandidateWorkspace
from society_core.workspace_update import run_workspace_verification

from .schemas import AcceptanceOracleSpec, RepoIntelligenceFabric


def build_acceptance_oracles(
    *,
    source_report: Mapping[str, Any],
    task_specs: tuple[Any, ...],
    workspace_root: Path,
    repo: RepoIntelligenceFabric,
    confirm_base: bool = False,
    timeout_seconds: float = 20.0,
    executor: CommandExecutor | None = None,
) -> tuple[AcceptanceOracleSpec, ...]:
    del source_report
    oracles: list[AcceptanceOracleSpec] = []
    for spec in task_specs:
        task_id = _field(spec, "task_id", "task")
        raw_oracles = tuple(_field(spec, "acceptance_oracles", ()))
        command_records = tuple(
            (oracle, str(_field(oracle, "command", "")))
            for oracle in raw_oracles
            if _field(oracle, "command", None)
        )
        if not command_records:
            command_records = tuple(
                (None, command) for command in _fallback_commands(spec, repo)
            )
        deduplicated: dict[str, Any | None] = {}
        for raw_oracle, command in command_records:
            existing = deduplicated.get(command)
            existing_dimensions = (
                tuple(_field(existing, "dimension_ids", ()))
                if existing is not None
                else ()
            )
            candidate_dimensions = (
                tuple(_field(raw_oracle, "dimension_ids", ()))
                if raw_oracle is not None
                else ()
            )
            if command not in deduplicated or (
                candidate_dimensions and not existing_dimensions
            ):
                deduplicated[command] = raw_oracle
        for index, (command, raw_oracle) in enumerate(deduplicated.items()):
            base_status = "not_run"
            base_observed = None
            if confirm_base:
                with CandidateWorkspace.create(
                    workspace_root
                ) as verification_workspace:
                    result = run_workspace_verification(
                        verification_workspace.root,
                        (command,),
                        timeout_seconds=timeout_seconds,
                        executor=executor,
                    )[0]
                base_observed = "\n".join(
                    part for part in (result.stdout_tail, result.stderr_tail) if part
                )[-2000:]
                if result.status == "failed":
                    base_status = "confirmed"
                elif result.status == "passed":
                    base_status = "not_reproduced"
                elif result.status == "timeout":
                    base_status = "flaky"
                else:
                    base_status = "blocked"
            payload = {
                "task_id": task_id,
                "index": index,
                "command": command,
                "base_status": base_status,
                "base_observed": base_observed,
                "dimension_ids": tuple(_field(raw_oracle, "dimension_ids", ()))
                if raw_oracle is not None
                else (),
                "witness_role": str(_field(raw_oracle, "witness_role", "repair"))
                if raw_oracle is not None
                else "repair",
                "required": bool(_field(raw_oracle, "required", True))
                if raw_oracle is not None
                else True,
            }
            expected_on_base = (
                str(_field(raw_oracle, "expected_on_base", "not_applicable"))
                if raw_oracle is not None
                else ("fail" if base_status == "confirmed" else "not_applicable")
            )
            expected_on_patch = (
                str(_field(raw_oracle, "expected_on_patch", "pass"))
                if raw_oracle is not None
                else "pass"
            )
            source_evidence_hash = (
                str(_field(raw_oracle, "evidence_hash", ""))
                if raw_oracle is not None
                else ""
            )
            oracles.append(
                AcceptanceOracleSpec(
                    oracle_id=f"oracle_{stable_hash(payload)[:16]}",
                    task_id=str(task_id),
                    kind=_oracle_kind(command),
                    command=command,
                    files_created=(),
                    expected_on_base=expected_on_base,
                    expected_on_patch=expected_on_patch,
                    base_observed=base_observed,
                    base_status=base_status,
                    confidence=_oracle_confidence(command, base_status),
                    false_positive_risk="low"
                    if base_status == "confirmed"
                    else "medium",
                    evidence_hash=(
                        source_evidence_hash
                        if len(source_evidence_hash) == 64
                        and all(
                            character in "0123456789abcdef"
                            for character in source_evidence_hash
                        )
                        else stable_hash(payload)
                    ),
                    dimension_ids=payload["dimension_ids"],
                    witness_role=payload["witness_role"],
                    required=payload["required"],
                )
            )
    return tuple(oracles)


def _fallback_commands(spec: Any, repo: RepoIntelligenceFabric) -> tuple[str, ...]:
    behavior_surface = tuple(_field(spec, "behavior_surface", ()))
    if repo.source_to_tests:
        first_tests = next(iter(repo.source_to_tests.values()))
        if first_tests:
            test_file = first_tests[0]
            if test_file.endswith(".py"):
                return (f"pytest {test_file}",)
            return (f"node --test {test_file}",)
    if "docs" in behavior_surface and repo.docs_files:
        return (
            "python -c \"import pathlib; assert pathlib.Path('README.md').read_text().strip()\"",
        )
    return repo.entrypoints[:1] and tuple(
        _syntax_command(path) for path in repo.entrypoints[:1] if _syntax_command(path)
    )


def _syntax_command(path: str) -> str | None:
    if path.endswith(".py"):
        return f"python -m py_compile {path}"
    if path.endswith((".js", ".mjs", ".cjs")):
        return f"node --check {path}"
    if path.endswith(".json"):
        return f"python -m json.tool {path}"
    return None


def _oracle_kind(command: str) -> str:
    lower = command.lower()
    if "pytest" in lower:
        return "pytest"
    if "node --test" in lower:
        return "unit_test"
    if "node --check" in lower or "py_compile" in lower:
        return "syntax"
    if " -c " in f" {lower} ":
        return "script"
    if "npm" in lower or "yarn" in lower or "pnpm" in lower:
        return "build" if "build" in lower else "script"
    return "cli"


def _oracle_confidence(command: str, base_status: str) -> float:
    base = 0.45
    if _oracle_kind(command) in {"pytest", "unit_test", "script"}:
        base += 0.25
    if base_status == "confirmed":
        base += 0.25
    elif base_status == "not_reproduced":
        base -= 0.15
    elif base_status == "blocked":
        base -= 0.25
    return max(0.0, min(1.0, round(base, 3)))


def _field(obj: Any, name: str, default: Any) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)
