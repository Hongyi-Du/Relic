"""Evaluator-private counterfactual qualification for frozen OSS time machines."""

from __future__ import annotations

import hashlib
import os
import platform
import re
import shutil
import stat
import sys
import tempfile
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from environments.org_env.product.substrates.base import (
    HiddenTestSpec,
    OSSSubstrateSpec,
)
from environments.org_env.product.substrates.eval_assets import (
    run_oss_hidden_tests_for_spec,
)
from environments.org_env.product.substrates.loader import (
    load_hidden_test_specs,
    load_oss_substrate_spec,
    read_repo_files,
)

from .code_landing.environment import build_workspace_execution_profile
from .code_landing.evidence import (
    EvidenceLedger,
    VerificationEvidence,
    assess_landing_evidence,
)
from .hashing import canonicalize, stable_hash
from .execution import (
    ApptainerCommandExecutor,
    BubblewrapCommandExecutor,
    CommandExecutor,
    CommandOutcome,
    DockerCommandExecutor,
)
from .programbench_evaluation import (
    is_programbench_spec,
    programbench_formal_blocking_reasons,
    programbench_hidden_suite_hash,
    programbench_runtime_identity,
    run_programbench_evaluation,
)
from .workspace_transaction import CandidateWorkspace


HiddenRunner = Callable[[OSSSubstrateSpec, str, int], Mapping[str, Any]]
TIME_MACHINE_EVALUATOR_RUNTIME_VERSION = "9"
_VALID_STATUSES = frozenset(
    {"not_run", "passed", "failed", "blocked", "timeout", "infra_error"}
)
_EVALUATOR_ROOT = "/evaluator"
_EVALUATOR_HIDDEN_ROOT = f"{_EVALUATOR_ROOT}/hidden"
_EVALUATOR_PYTEST_CONFIG = f"{_EVALUATOR_HIDDEN_ROOT}/pytest.ini"
_ISOLATED_PYTEST_RUNNER = (
    "import os\n"
    "import sys\n"
    "import pytest\n"
    "root = os.getcwd()\n"
    "sys.path[:0] = [root, os.path.join(root, 'src')]\n"
    "raise SystemExit(pytest.main(sys.argv[1:]))\n"
)
_ISOLATED_WORKSPACE_PYTEST_RUNNER = (
    "import os\n"
    "import sys\n"
    "root = os.getcwd()\n"
    "sys.path[:0] = [root, os.path.join(root, 'src')]\n"
    "import pytest\n"
    "raise SystemExit(pytest.main(sys.argv[1:]))\n"
)
_ISOLATED_PYTHON_RUNNER = (
    "import os\n"
    "import runpy\n"
    "import sys\n"
    "root = os.getcwd()\n"
    "sys.path[:0] = [root, os.path.join(root, 'src')]\n"
    "mode = sys.argv[1]\n"
    "if mode == '-c':\n"
    "    source = sys.argv[2]\n"
    "    sys.argv = ['-c', *sys.argv[3:]]\n"
    "    namespace = {'__name__': '__main__', '__file__': '<hidden-evaluator>'}\n"
    "    exec(compile(source, '<hidden-evaluator>', 'exec'), namespace)\n"
    "elif mode == '-m':\n"
    "    module = sys.argv[2]\n"
    "    sys.argv = [module, *sys.argv[3:]]\n"
    "    runpy.run_module(module, run_name='__main__', alter_sys=True)\n"
    "else:\n"
    "    sys.argv = sys.argv[1:]\n"
    "    runpy.run_path(mode, run_name='__main__')\n"
)


def _evaluator_bundle_parent() -> Path:
    """Return a private host path visible to local container runtimes."""

    configured = os.environ.get("SOCIETY_CORE_EVALUATOR_ROOT")
    root = (
        Path(configured).expanduser()
        if configured
        else Path.home() / ".society_core" / "evaluator-bundles"
    )
    is_junction = getattr(root, "is_junction", None)
    if (
        not root.is_absolute()
        or root.is_symlink()
        or bool(is_junction is not None and is_junction())
    ):
        raise RuntimeError("unsafe_evaluator_bundle_root")
    try:
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        metadata = root.lstat()
        if not stat.S_ISDIR(metadata.st_mode):
            raise RuntimeError("unsafe_evaluator_bundle_root")
        if hasattr(os, "getuid") and metadata.st_uid != os.getuid():
            raise RuntimeError("unsafe_evaluator_bundle_root")
        if os.name == "posix":
            root.chmod(0o700)
            if stat.S_IMODE(root.lstat().st_mode) != 0o700:
                raise RuntimeError("unsafe_evaluator_bundle_root")
    except OSError as exc:
        raise RuntimeError("unsafe_evaluator_bundle_root") from exc
    return root


@dataclass(frozen=True)
class TimeMachineOracleQualification:
    test_id: str
    issue_ids: tuple[str, ...]
    baseline_status: str
    reference_status: str
    introduced_in: str = ""


@dataclass(frozen=True)
class TimeMachineEvaluationPlan:
    dataset_id: str
    product_name: str
    starter_ref: str
    reference_ref: str
    starter_repo_digest: str
    reference_repo_digest: str
    hidden_suite_hash: str
    evaluator_environment_hash: str
    oracles: tuple[TimeMachineOracleQualification, ...]
    public_contracts: tuple[dict[str, Any], ...]
    operational_ready: bool
    formal_environment_ready: bool
    formal_ready: bool
    blocking_reasons: tuple[str, ...]
    plan_hash: str
    _spec: OSSSubstrateSpec = field(repr=False, compare=False)
    _hidden_specs: tuple[HiddenTestSpec, ...] = field(repr=False, compare=False)
    required_release_coverage: tuple[str, ...] = ()
    covered_release_versions: tuple[str, ...] = ()


@dataclass(frozen=True)
class TimeMachineCandidateOutcome:
    test_id: str
    issue_ids: tuple[str, ...]
    baseline_status: str
    reference_status: str
    candidate_status: str
    causal_fix: bool
    unresolved: bool
    regression: bool


@dataclass(frozen=True)
class TimeMachineEvaluationResult:
    dataset_id: str
    plan_hash: str
    candidate_repo_digest: str
    status: str
    outcomes: tuple[TimeMachineCandidateOutcome, ...]
    causal_fix_count: int
    unresolved_count: int
    regression_count: int
    infrastructure_error_count: int
    baseline_pass_rate: float
    candidate_pass_rate: float
    reference_pass_rate: float
    pass_rate_uplift: float
    formal_claim_ready: bool
    evidence_records: tuple[VerificationEvidence, ...]
    result_hash: str
    # Whether any oracle could have reported a regression at all. False means
    # regression_count is a tautology, not a measurement — see _summarise.
    regression_detectable: bool = False


def attach_time_machine_evidence(run: Any, result: TimeMachineEvaluationResult) -> Any:
    """Return a coding-loop result re-assessed with post-run evaluator-owned evidence."""

    ledger = EvidenceLedger(records=tuple(run.evidence_records)).extend(
        result.evidence_records
    )
    promotion_records = tuple(
        dict(item)
        for item in (run.run_config.get("promotion_records") or ())
        if isinstance(item, Mapping)
    )
    latest_promotion_status = (
        str(promotion_records[-1].get("status")) if promotion_records else None
    )
    integration_replayed = (
        run.final_mode == "verified" and latest_promotion_status == "release_candidate"
    )
    assessment = assess_landing_evidence(
        ledger=ledger,
        effective_patch=bool(run.verified_effective_patches),
        integration_replayed=integration_replayed,
        promotion_status=(
            latest_promotion_status if run.final_mode == "verified" else run.final_mode
        ),
        promotion_records=promotion_records,
    )
    public_evaluation_summary = {
        "dataset_id": result.dataset_id,
        "plan_hash": result.plan_hash,
        "result_hash": result.result_hash,
        "status": result.status,
        "causal_fix_count": result.causal_fix_count,
        "unresolved_count": result.unresolved_count,
        "regression_count": result.regression_count,
        "regression_detectable": result.regression_detectable,
        "infrastructure_error_count": result.infrastructure_error_count,
        "baseline_pass_rate": result.baseline_pass_rate,
        "candidate_pass_rate": result.candidate_pass_rate,
        "reference_pass_rate": result.reference_pass_rate,
        "pass_rate_uplift": result.pass_rate_uplift,
        "formal_claim_ready": result.formal_claim_ready,
        "test_statuses": tuple(
            {
                "test_id": item.test_id,
                "candidate_status": item.candidate_status,
                "causal_fix": item.causal_fix,
            }
            for item in result.outcomes
        ),
    }
    provenance_hash = stable_hash(
        {
            "prior_provenance_hash": run.provenance_hash,
            "time_machine_result_hash": result.result_hash,
            "evidence_hash": assessment.evidence_hash,
            "evidence_records": tuple(item.evidence_hash for item in ledger.records),
        }
    )
    return replace(
        run,
        run_id=f"{run.run_id}_evaluated_{provenance_hash[:12]}",
        provenance_hash=provenance_hash,
        run_config={
            **run.run_config,
            "time_machine_evaluation": public_evaluation_summary,
        },
        evidence_level=assessment.level,
        evidence_hash=assessment.evidence_hash,
        evidence_summary=canonicalize(assessment),
        evidence_records=ledger.records,
    )


def oracle_blocking_reasons(
    test_id: str,
    *,
    baseline_status: str,
    reference_status: str,
) -> list[str]:
    """Why this oracle cannot qualify, if it cannot. One rule, one place.

    What the baseline has to establish is that the starter does not satisfy the
    contract. A failed assertion says that; so does an error raised on the way
    to the assertion, and on a greenfield pack the second is the ordinary case
    -- the suite drives the product through fixtures, and a fixture that
    touches a hollowed module raises during setup, which pytest reports as an
    error rather than a failure.

    Reading that as a disqualification cost three packs. It is only safe to
    read it as "unsatisfied" while the reference passes the same contract, in
    the same container, under the same command: that is what rules out a
    harness fault, and it is required here rather than assumed.

    The rule was written here and separately in the substrate seeder and the
    pack builder, and only this copy was corrected -- so preflight and the
    hash tool called traffic_watch formal-ready while the seeder refused to
    build a world for it, and every greenfield pack was unrunnable on the
    formal path for a reason the checks said was fixed.
    """
    reasons: list[str] = []
    unsatisfied = baseline_status == "failed" or (
        baseline_status == "infra_error" and reference_status == "passed"
    )
    if not unsatisfied:
        reasons.append(f"baseline_not_failed:{test_id}:{baseline_status}")
    if reference_status != "passed":
        reasons.append(f"reference_not_passed:{test_id}:{reference_status}")
    return reasons


def build_time_machine_evaluation_plan(
    *,
    dataset_id: str | None = None,
    spec: OSSSubstrateSpec | None = None,
    hidden_specs: Sequence[HiddenTestSpec] | None = None,
    timeout_seconds: int = 60,
    runner: HiddenRunner = run_oss_hidden_tests_for_spec,
    executor: CommandExecutor | None = None,
) -> TimeMachineEvaluationPlan:
    """Qualify a hidden suite before an agent run using frozen base/reference outcomes."""

    resolved_spec = spec or load_oss_substrate_spec(str(dataset_id or ""))
    if dataset_id is not None and resolved_spec.project_id != dataset_id:
        requested_path = Path(dataset_id).expanduser()
        resolved_path = Path(resolved_spec.dataset_dir).expanduser()
        if not (
            requested_path.is_absolute()
            and requested_path.resolve() == resolved_path.resolve()
        ):
            raise ValueError("time_machine_dataset_id_mismatch")
    specs = tuple(hidden_specs or load_hidden_test_specs(resolved_spec))
    public_contracts = _public_contracts_from_manifest(resolved_spec.manifest)
    test_ids = tuple(item.test_id for item in specs)
    blocking_reasons: list[str] = []
    if not specs:
        blocking_reasons.append("hidden_suite_empty")
    if len(set(test_ids)) != len(test_ids):
        blocking_reasons.append("duplicate_hidden_test_ids")
    required_release_coverage = tuple(
        str(version)
        for version in (
            (resolved_spec.manifest.get("evaluation") or {}).get(
                "required_hidden_release_coverage", ()
            )
            or ()
        )
    )
    covered_release_versions = tuple(
        sorted({str(item.introduced_in) for item in specs if str(item.introduced_in)})
    )
    if required_release_coverage:
        for item in specs:
            if not item.introduced_in:
                blocking_reasons.append(
                    f"hidden_test_missing_introduced_in:{item.test_id}"
                )
    for version in required_release_coverage:
        if version not in covered_release_versions:
            blocking_reasons.append(f"hidden_release_coverage_missing:{version}")

    baseline = _run_frozen_repo(
        resolved_spec,
        Path(resolved_spec.starter_repo_dir),
        timeout_seconds=timeout_seconds,
        runner=runner,
        hidden_specs=specs,
        executor=executor,
        evaluation_role="baseline",
    )
    reference = _run_frozen_repo(
        resolved_spec,
        Path(resolved_spec.reference_repo_dir),
        timeout_seconds=timeout_seconds,
        runner=runner,
        hidden_specs=specs,
        executor=executor,
        evaluation_role="reference",
    )
    baseline_statuses = _status_by_test(baseline, test_ids)
    reference_statuses = _status_by_test(reference, test_ids)
    oracles = tuple(
        TimeMachineOracleQualification(
            test_id=item.test_id,
            issue_ids=tuple(item.issue_ids),
            introduced_in=item.introduced_in,
            baseline_status=baseline_statuses[item.test_id],
            reference_status=reference_statuses[item.test_id],
        )
        for item in specs
    )
    for oracle in oracles:
        blocking_reasons.extend(
            oracle_blocking_reasons(
                oracle.test_id,
                baseline_status=oracle.baseline_status,
                reference_status=oracle.reference_status,
            )
        )

    starter_digest = _repo_digest(Path(resolved_spec.starter_repo_dir))
    reference_digest = _repo_digest(Path(resolved_spec.reference_repo_dir))
    hidden_suite_hash = (
        programbench_hidden_suite_hash(resolved_spec)
        if is_programbench_spec(resolved_spec)
        else stable_hash(read_repo_files(resolved_spec.hidden_tests_dir))
    )
    evaluator_environment_hash = _evaluator_environment_hash(
        runner=runner,
        executor=executor,
        hidden_suite_hash=hidden_suite_hash,
        spec=resolved_spec,
    )
    operational_ready = not blocking_reasons
    formal_environment_reasons = _formal_executor_blocking_reasons(
        executor,
        spec=resolved_spec,
    )
    formal_environment_ready = not formal_environment_reasons
    all_blocking_reasons = tuple(
        dict.fromkeys((*blocking_reasons, *formal_environment_reasons))
    )
    payload = {
        "dataset_id": resolved_spec.project_id,
        "product_name": resolved_spec.product_name,
        "starter_ref": str(resolved_spec.manifest.get("starter_ref") or ""),
        "reference_ref": str(resolved_spec.manifest.get("reference_ref") or ""),
        "starter_repo_digest": starter_digest,
        "reference_repo_digest": reference_digest,
        "hidden_suite_hash": hidden_suite_hash,
        "evaluator_environment_hash": evaluator_environment_hash,
        "required_release_coverage": required_release_coverage,
        "covered_release_versions": covered_release_versions,
        "oracles": canonicalize(oracles),
        "public_contracts": canonicalize(public_contracts),
        "blocking_reasons": all_blocking_reasons,
        "operational_ready": operational_ready,
        "formal_environment_ready": formal_environment_ready,
    }
    return TimeMachineEvaluationPlan(
        dataset_id=resolved_spec.project_id,
        product_name=resolved_spec.product_name,
        starter_ref=str(resolved_spec.manifest.get("starter_ref") or ""),
        reference_ref=str(resolved_spec.manifest.get("reference_ref") or ""),
        starter_repo_digest=starter_digest,
        reference_repo_digest=reference_digest,
        hidden_suite_hash=hidden_suite_hash,
        evaluator_environment_hash=evaluator_environment_hash,
        required_release_coverage=required_release_coverage,
        covered_release_versions=covered_release_versions,
        oracles=oracles,
        public_contracts=public_contracts,
        operational_ready=operational_ready,
        formal_environment_ready=formal_environment_ready,
        formal_ready=operational_ready and formal_environment_ready,
        blocking_reasons=all_blocking_reasons,
        plan_hash=stable_hash(payload),
        _spec=resolved_spec,
        _hidden_specs=specs,
    )


def _public_contracts_from_manifest(
    manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    """Return only agent-visible, behavior-oriented release contracts."""

    rows = manifest.get("public_contracts")
    if not isinstance(rows, (list, tuple)):
        return ()
    contracts: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"time_machine_public_contract_invalid:{index}")
        issue_id = str(row.get("issue_id") or "").strip()
        theme = str(row.get("theme") or "").strip()
        title = str(row.get("title") or "").strip()
        user_pain = str(row.get("user_pain") or row.get("body") or "").strip()
        expected_behavior = str(row.get("expected_behavior") or "").strip()
        acceptance_commands = _manifest_string_tuple(row.get("acceptance_commands"))
        contract_dimensions = _manifest_string_tuple(row.get("contract_dimensions"))
        if not (
            issue_id
            and theme
            and title
            and user_pain
            and expected_behavior
            and acceptance_commands
            and contract_dimensions
        ):
            raise ValueError(f"time_machine_public_contract_incomplete:{index}")
        contracts.append(
            {
                "issue_id": issue_id[:120],
                "theme": theme[:80],
                "title": title[:500],
                "user_pain": user_pain[:2000],
                "expected_behavior": expected_behavior[:2000],
                "reproduction_steps": _manifest_string_tuple(
                    row.get("reproduction_steps")
                ),
                "acceptance_commands": acceptance_commands,
                "contract_dimensions": contract_dimensions,
                "candidate_path_hints": _manifest_string_tuple(
                    row.get("candidate_path_hints")
                ),
                "relevant_symbols_hint": _manifest_string_tuple(
                    row.get("relevant_symbols_hint")
                ),
                "source_url": str(row.get("source_url") or "")[:1000],
            }
        )
    issue_ids = tuple(item["issue_id"] for item in contracts)
    if len(set(issue_ids)) != len(issue_ids):
        raise ValueError("time_machine_public_contract_duplicate_issue_id")
    return tuple(contracts)


def _manifest_string_tuple(value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(
        dict.fromkeys(str(item).strip()[:2000] for item in value if str(item).strip())
    )


def evaluate_time_machine_candidate(
    plan: TimeMachineEvaluationPlan,
    candidate_root: Path,
    *,
    timeout_seconds: int = 60,
    runner: HiddenRunner = run_oss_hidden_tests_for_spec,
    executor: CommandExecutor | None = None,
) -> TimeMachineEvaluationResult:
    """Run a qualified hidden suite once against a post-development candidate."""

    candidate_digest = _repo_digest(candidate_root)
    evaluator_environment_hash = _evaluator_environment_hash(
        runner=runner,
        executor=executor,
        hidden_suite_hash=plan.hidden_suite_hash,
        spec=plan._spec,
    )
    if evaluator_environment_hash != plan.evaluator_environment_hash:
        raise ValueError("time_machine_evaluator_environment_mismatch")
    if not plan.operational_ready:
        outcomes = tuple(
            TimeMachineCandidateOutcome(
                test_id=oracle.test_id,
                issue_ids=oracle.issue_ids,
                baseline_status=oracle.baseline_status,
                reference_status=oracle.reference_status,
                candidate_status="not_run",
                causal_fix=False,
                unresolved=False,
                regression=False,
            )
            for oracle in plan.oracles
        )
        return _build_result(
            plan=plan,
            candidate_digest=candidate_digest,
            status="blocked_invalid_plan",
            outcomes=outcomes,
            evidence_records=(),
        )

    raw = _run_frozen_repo(
        plan._spec,
        candidate_root,
        timeout_seconds=timeout_seconds,
        runner=runner,
        hidden_specs=plan._hidden_specs,
        executor=executor,
        evaluation_role="candidate",
    )
    statuses = _status_by_test(raw, tuple(item.test_id for item in plan.oracles))
    outcomes = tuple(
        TimeMachineCandidateOutcome(
            test_id=oracle.test_id,
            issue_ids=oracle.issue_ids,
            baseline_status=oracle.baseline_status,
            reference_status=oracle.reference_status,
            candidate_status=statuses[oracle.test_id],
            causal_fix=(
                oracle.baseline_status == "failed"
                and statuses[oracle.test_id] == "passed"
            ),
            unresolved=(
                oracle.baseline_status == "failed"
                and statuses[oracle.test_id] == "failed"
            ),
            regression=(
                oracle.baseline_status == "passed"
                and statuses[oracle.test_id] != "passed"
            ),
        )
        for oracle in plan.oracles
    )
    test_strategy = plan._spec.manifest.get("test_strategy") or {}
    evidence_kind = (
        str(test_strategy.get("kind"))
        if isinstance(test_strategy, Mapping) and test_strategy.get("kind")
        else "behavior"
    )
    evidence = tuple(
        VerificationEvidence.create(
            task_id=f"time_machine:{plan.dataset_id}:{outcome.test_id}",
            command=f"private-evaluator://{plan.dataset_id}/{outcome.test_id}",
            owner="evaluator",
            kind=evidence_kind,
            base_status=outcome.baseline_status,
            candidate_status=outcome.candidate_status,
            required=True,
            baseline_repo_digest=plan.starter_repo_digest,
            candidate_repo_digest=candidate_digest,
            execution_policy_hash=plan.evaluator_environment_hash,
        )
        for outcome in outcomes
    )
    candidate_statuses = tuple(outcome.candidate_status for outcome in outcomes)
    if any(
        status in {"infra_error", "timeout", "blocked"} for status in candidate_statuses
    ):
        status = "infra_error"
    elif all(status == "passed" for status in candidate_statuses):
        status = "passed"
    else:
        status = "incomplete"
    return _build_result(
        plan=plan,
        candidate_digest=candidate_digest,
        status=status,
        outcomes=outcomes,
        evidence_records=evidence,
    )


def _run_frozen_repo(
    spec: OSSSubstrateSpec,
    source_root: Path,
    *,
    timeout_seconds: int,
    runner: HiddenRunner,
    hidden_specs: Sequence[HiddenTestSpec],
    executor: CommandExecutor | None,
    evaluation_role: str,
) -> Mapping[str, Any]:
    if not source_root.is_dir():
        return {"by_test": ()}
    with CandidateWorkspace.create(
        source_root,
        recover_interrupted_promotion=False,
    ) as isolated_workspace:
        isolated_root = isolated_workspace.root
        before_digest = _repo_digest(isolated_root)
        if is_programbench_spec(spec):
            result = run_programbench_evaluation(
                spec,
                str(isolated_root),
                timeout=timeout_seconds,
                role=evaluation_role,
            )
        elif executor is not None:
            result = _run_hidden_specs_with_executor(
                spec=spec,
                hidden_specs=hidden_specs,
                repo_root=isolated_root,
                timeout_seconds=timeout_seconds,
                executor=executor,
            )
        else:
            result = runner(spec, str(isolated_root), timeout_seconds)
        if _repo_digest(isolated_root) != before_digest:
            raise ValueError(
                "time_machine_candidate_mutated_during_evaluation"
            )
        return result


def _run_hidden_specs_with_executor(
    *,
    spec: OSSSubstrateSpec,
    hidden_specs: Sequence[HiddenTestSpec],
    repo_root: Path,
    timeout_seconds: int,
    executor: CommandExecutor,
) -> Mapping[str, Any]:
    hidden_source = Path(spec.hidden_tests_dir)
    if not hidden_source.is_dir():
        return {"by_test": ()}
    run_evaluator = getattr(executor, "run_with_read_only_evaluator", None)
    for path in hidden_source.rglob("*"):
        if path.is_symlink():
            raise ValueError("evaluator_hidden_suite_symlinks_forbidden")
    with tempfile.TemporaryDirectory(
        prefix="sociogenesis_evaluator_bundle_",
        dir=_evaluator_bundle_parent(),
    ) as temporary_root:
        evaluator_root = Path(temporary_root)
        hidden_target = evaluator_root / "hidden"
        shutil.copytree(
            hidden_source,
            hidden_target,
            dirs_exist_ok=True,
            symlinks=False,
        )
        (hidden_target / "pytest.ini").write_text(
            "[pytest]\n",
            encoding="utf-8",
        )
        by_test: list[dict[str, Any]] = []
        workspace_owns_pytest = _workspace_owns_pytest(
            Path(spec.starter_repo_dir)
        )
        python_executable = (
            "python"
            if isinstance(
                executor,
                (
                    DockerCommandExecutor,
                    ApptainerCommandExecutor,
                    BubblewrapCommandExecutor,
                ),
            )
            else sys.executable
        )
        evaluator_hidden_root = (
            _EVALUATOR_HIDDEN_ROOT
            if isinstance(
                executor,
                (
                    DockerCommandExecutor,
                    ApptainerCommandExecutor,
                    BubblewrapCommandExecutor,
                ),
            )
            else str(hidden_target)
        )
        for hidden_spec in hidden_specs:
            with CandidateWorkspace.create(
                repo_root,
                recover_interrupted_promotion=False,
            ) as oracle_workspace:
                oracle_root = oracle_workspace.root
                before_digest = _repo_digest(oracle_root)
                argv = _executor_hidden_command(
                    hidden_spec.command,
                    workspace_owns_pytest=workspace_owns_pytest,
                    python_executable=python_executable,
                    evaluator_hidden_root=evaluator_hidden_root,
                )
                if not argv:
                    status = "infra_error"
                else:
                    if callable(run_evaluator):
                        outcome = run_evaluator(
                            root=oracle_root,
                            evaluator_root=evaluator_root,
                            argv=argv,
                            timeout_seconds=float(timeout_seconds),
                        )
                    else:
                        outcome = executor.run(
                            root=oracle_root,
                            argv=argv,
                            timeout_seconds=float(timeout_seconds),
                        )
                    status = _hidden_outcome_status(
                        outcome,
                        hidden_spec.command,
                    )
                if _repo_digest(oracle_root) != before_digest:
                    status = "infra_error"
            by_test.append(
                {
                    "test_id": hidden_spec.test_id,
                    "issue_ids": tuple(hidden_spec.issue_ids),
                    "status": status,
                    "passed": status == "passed",
                }
            )
        return {"by_test": tuple(by_test)}


def _executor_hidden_command(
    command: Sequence[str],
    *,
    workspace_owns_pytest: bool = False,
    python_executable: str = "python",
    evaluator_hidden_root: str = _EVALUATOR_HIDDEN_ROOT,
) -> tuple[str, ...]:
    argv = tuple(str(token) for token in command if str(token))
    if not argv:
        return ()
    is_pytest = _is_pytest_command(argv)
    if is_pytest:
        pytest_args = (
            argv[3:] if len(argv) >= 3 and argv[1:3] == ("-m", "pytest") else argv[1:]
        )
        pytest_args = tuple(
            _rewrite_hidden_path(
                token, evaluator_hidden_root=evaluator_hidden_root
            )
            for token in pytest_args
        )
        if "-c" not in pytest_args and "--config-file" not in pytest_args:
            pytest_args = (
                *pytest_args,
                "-c",
                f"{evaluator_hidden_root}/pytest.ini",
            )
        # The pack may name a confcutdir of its own, relative to its own tree.
        # That path does not exist here: the suite is mounted away from the
        # candidate, and pytest refuses a confcutdir that is not a directory
        # with a usage error, which reads back as an infrastructure failure on
        # every contract. mini_blobstore asks for `tests/hidden` and had all
        # five oracles recorded not_run across thirteen runs for that reason,
        # baseline and reference alike, so the plan was blocked and nothing the
        # organization did could be scored. Where the suite actually sits is
        # not the pack's to decide.
        pytest_args = tuple(t for t in pytest_args
                            if not t.startswith("--confcutdir="))
        pytest_args = (*pytest_args, f"--confcutdir={evaluator_hidden_root}")
        if "no:cacheprovider" not in pytest_args:
            pytest_args = (*pytest_args, "-p", "no:cacheprovider")
        if not any(token.startswith("--tb") for token in pytest_args):
            pytest_args = (*pytest_args, "--tb=no")
        pytest_runner = (
            _ISOLATED_WORKSPACE_PYTEST_RUNNER
            if workspace_owns_pytest
            else _ISOLATED_PYTEST_RUNNER
        )
        argv = (python_executable, "-I", "-c", pytest_runner, *pytest_args)
    elif Path(argv[0]).name in {"python", "python3"}:
        if len(argv) < 2 or (argv[1].startswith("-") and argv[1] not in {"-c", "-m"}):
            return ()
        python_args = tuple(
            _rewrite_hidden_path(
                token, evaluator_hidden_root=evaluator_hidden_root
            )
            for token in argv[1:]
        )
        argv = (
            python_executable,
            "-I",
            "-c",
            _ISOLATED_PYTHON_RUNNER,
            *python_args,
        )
    else:
        argv = tuple(
            _rewrite_hidden_path(
                token, evaluator_hidden_root=evaluator_hidden_root
            )
            for token in argv
        )
    return ("env", "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1", "PY_COLORS=0", *argv)


def _rewrite_hidden_path(
    token: str,
    *,
    evaluator_hidden_root: str = _EVALUATOR_HIDDEN_ROOT,
) -> str:
    if token == "tests/hidden":
        return evaluator_hidden_root
    prefix = "tests/hidden/"
    if token.startswith(prefix):
        return f"{evaluator_hidden_root}/{token[len(prefix) :]}"
    return token


def _workspace_owns_pytest(starter_root: Path) -> bool:
    source_root = starter_root / "src"
    return (source_root / "pytest" / "__init__.py").is_file() and (
        source_root / "_pytest" / "__init__.py"
    ).is_file()


def _hidden_outcome_status(
    outcome: CommandOutcome,
    command: Sequence[str],
) -> str:
    if outcome.status != "failed" or not _is_pytest_command(command):
        return outcome.status
    combined = f"{outcome.stdout_tail}\n{outcome.stderr_tail}".casefold()
    infrastructure_markers = (
        "error during collection",
        "error collecting",
        "internalerror>",
        "no tests ran",
        "file or directory not found",
        "importerror while loading conftest",
        "error importing plugin",
        "unrecognized arguments:",
    )
    if outcome.exit_code in {2, 3, 4, 5} or any(
        marker in combined for marker in infrastructure_markers
    ):
        return "infra_error"
    return "failed"


def _is_pytest_command(command: Sequence[str]) -> bool:
    argv = tuple(str(token) for token in command)
    return (len(argv) >= 3 and argv[1:3] == ("-m", "pytest")) or bool(
        argv and Path(argv[0]).name in {"pytest", "py.test"}
    )


def _evaluator_environment_hash(
    *,
    runner: HiddenRunner,
    executor: CommandExecutor | None,
    hidden_suite_hash: str,
    spec: OSSSubstrateSpec | None = None,
) -> str:
    programbench_spec = spec is not None and is_programbench_spec(spec)
    runtime_code_hash = _evaluator_runtime_code_hash(
        include_programbench=programbench_spec
    )
    if programbench_spec:
        runtime = {
            "kind": "programbench_official",
            "evaluator_runtime_version": TIME_MACHINE_EVALUATOR_RUNTIME_VERSION,
            "evaluator_runtime_code_hash": runtime_code_hash,
            "programbench": canonicalize(programbench_runtime_identity(spec)),
        }
    elif executor is None:
        runtime = {
            "kind": "host_runner",
            "evaluator_runtime_version": TIME_MACHINE_EVALUATOR_RUNTIME_VERSION,
            "evaluator_runtime_code_hash": runtime_code_hash,
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": sys.platform,
            "runner": (
                f"{runner.__module__}."
                f"{getattr(runner, '__qualname__', runner.__name__)}"
            ),
        }
    else:
        policy = asdict(executor.policy)
        policy.pop("docker_host", None)
        runtime = {
            "kind": "command_executor",
            "evaluator_runtime_version": TIME_MACHINE_EVALUATOR_RUNTIME_VERSION,
            "evaluator_runtime_code_hash": runtime_code_hash,
            "executor": (
                f"{executor.__class__.__module__}.{executor.__class__.__qualname__}"
            ),
            "policy": canonicalize(policy),
        }
        if isinstance(executor, BubblewrapCommandExecutor):
            runtime["virtualenv_fingerprint"] = _virtualenv_fingerprint(
                executor.virtualenv_path
            )
    return stable_hash(
        {
            "runtime": runtime,
            "hidden_suite_hash": hidden_suite_hash,
        }
    )


def _virtualenv_fingerprint(virtualenv_path: Path) -> str:
    """Content-address a controller-owned evaluator venv.

    Bytecode and tool caches are execution residue and excluded. Installed
    source, metadata, native extensions, entry points, the Python binding, and
    prefetched evaluator data remain part of the identity.
    """

    try:
        root = virtualenv_path.expanduser().resolve(strict=True)
    except OSError as exc:
        raise RuntimeError("evaluator_virtualenv_unavailable") from exc
    if not root.is_dir():
        raise RuntimeError("evaluator_virtualenv_not_directory")

    ignored_directories = {"__pycache__", ".pytest_cache", ".ruff_cache"}
    ignored_suffixes = {".pyc", ".pyo"}
    records: list[dict[str, str]] = []
    for current, directories, filenames in os.walk(root, followlinks=False):
        directories[:] = sorted(
            name for name in directories if name not in ignored_directories
        )
        current_path = Path(current)
        for filename in sorted(filenames):
            path = current_path / filename
            if path.suffix in ignored_suffixes:
                continue
            relative = path.relative_to(root).as_posix()
            if path.is_symlink():
                records.append(
                    {
                        "path": relative,
                        "kind": "symlink",
                        "target": os.readlink(path),
                    }
                )
                continue
            try:
                mode = path.stat(follow_symlinks=False).st_mode
            except OSError as exc:
                raise RuntimeError(
                    f"evaluator_virtualenv_entry_unavailable:{relative}"
                ) from exc
            if not stat.S_ISREG(mode):
                continue
            digest = hashlib.sha256()
            try:
                with path.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(chunk)
            except OSError as exc:
                raise RuntimeError(
                    f"evaluator_virtualenv_entry_unreadable:{relative}"
                ) from exc
            records.append(
                {
                    "path": relative,
                    "kind": "file",
                    "sha256": digest.hexdigest(),
                }
            )
    return stable_hash(records)


def _evaluator_runtime_code_hash(*, include_programbench: bool = False) -> str:
    """Bind formal evaluator receipts to the executing source implementation."""

    module_names = (
        "society_core.execution",
        "society_core.time_machine_evaluation",
        "society_core.workspace_transaction",
        "environments.org_env.product.substrates.eval_assets",
        "environments.org_env.product.substrates.loader",
    )
    if include_programbench:
        module_names = (*module_names, "society_core.programbench_evaluation")
    records: list[dict[str, str]] = []
    for module_name in module_names:
        module = sys.modules.get(module_name)
        source_name = str(getattr(module, "__file__", "") or "")
        source_path = Path(source_name)
        if source_path.suffix == ".pyc":
            candidate = source_path.with_suffix(".py")
            if candidate.is_file():
                source_path = candidate
        try:
            payload = source_path.resolve(strict=True).read_bytes()
        except OSError as exc:
            raise RuntimeError(
                f"evaluator_runtime_source_unavailable:{module_name}"
            ) from exc
        records.append(
            {
                "module": module_name,
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    return stable_hash(records)


def _formal_executor_blocking_reasons(
    executor: CommandExecutor | None,
    *,
    spec: OSSSubstrateSpec | None = None,
) -> tuple[str, ...]:
    if spec is not None and is_programbench_spec(spec):
        # The official ProgramBench evaluator owns the pinned Docker boundary;
        # accepting a second, generic executor here would create two subtly
        # different qualification paths.
        return tuple(programbench_formal_blocking_reasons(spec))
    if not isinstance(
        executor,
        (DockerCommandExecutor, ApptainerCommandExecutor),
    ):
        return ("formal_container_executor_required",)
    policy = executor.policy
    image = str(policy.container_image or "")
    digest_pinned = bool(
        re.fullmatch(r"sha256:[0-9a-fA-F]{64}", image)
        or re.search(r"@sha256:[0-9a-fA-F]{64}$", image)
    )
    reasons: list[str] = []
    if policy.backend not in {"docker", "apptainer"}:
        reasons.append("formal_container_backend_required")
    if policy.trust_level != "untrusted":
        reasons.append("formal_untrusted_execution_required")
    if policy.network_enabled:
        reasons.append("formal_network_isolation_required")
    if not digest_pinned:
        reasons.append("formal_digest_pinned_image_required")
    if not policy.container_platform:
        reasons.append("formal_container_platform_required")
    if not callable(
        getattr(executor, "run_with_read_only_evaluator", None)
    ):
        reasons.append("formal_read_only_evaluator_mount_required")
    return tuple(reasons)


def _status_by_test(
    raw: Mapping[str, Any],
    expected_test_ids: tuple[str, ...],
) -> dict[str, str]:
    statuses: dict[str, str] = {}
    for row in raw.get("by_test", ()) or ():
        if not isinstance(row, Mapping):
            continue
        test_id = str(row.get("test_id") or "")
        if not test_id or test_id in statuses:
            continue
        status = str(
            row.get("status") or ("passed" if bool(row.get("passed")) else "failed")
        )
        statuses[test_id] = status if status in _VALID_STATUSES else "infra_error"
    return {
        test_id: statuses.get(test_id, "infra_error") for test_id in expected_test_ids
    }


def _repo_digest(root: Path) -> str:
    if not root.is_dir():
        return stable_hash({"missing_repo": root.name})
    return build_workspace_execution_profile(root).repo_hash


def _build_result(
    *,
    plan: TimeMachineEvaluationPlan,
    candidate_digest: str,
    status: str,
    outcomes: tuple[TimeMachineCandidateOutcome, ...],
    evidence_records: tuple[VerificationEvidence, ...],
) -> TimeMachineEvaluationResult:
    total = len(outcomes)
    baseline_passed = sum(item.baseline_status == "passed" for item in outcomes)
    candidate_passed = sum(item.candidate_status == "passed" for item in outcomes)
    reference_passed = sum(item.reference_status == "passed" for item in outcomes)
    baseline_rate = baseline_passed / total if total else 0.0
    candidate_rate = candidate_passed / total if total else 0.0
    reference_rate = reference_passed / total if total else 0.0
    causal_fix_count = sum(item.causal_fix for item in outcomes)
    unresolved_count = sum(item.unresolved for item in outcomes)
    regression_count = sum(item.regression for item in outcomes)
    # A regression is "baseline passed and the candidate did not", and plan
    # construction requires every oracle to FAIL on the starter tree, so the two
    # conditions are mutually exclusive: in any run that produces numbers at all
    # regression_count can only be zero. Reported as zero it reads as a measured
    # finding — one pilot summary said "seventeen merged PRs and not one
    # regression" — when the suite by construction contains nothing that could
    # detect one. Two runs each replaced a 43k-character module with a 9.5k
    # rewrite, dropping a public symbol and every docstring, and the number
    # stayed zero. Say whether the question was askable.
    regression_detectable = any(
        item.baseline_status == "passed" for item in outcomes
    )
    infrastructure_error_count = sum(
        item.candidate_status in {"infra_error", "timeout", "blocked"}
        for item in outcomes
    )
    formal_claim_ready = (
        plan.formal_ready
        and total > 0
        and candidate_passed == total
        and infrastructure_error_count == 0
    )
    payload = {
        "dataset_id": plan.dataset_id,
        "plan_hash": plan.plan_hash,
        "candidate_repo_digest": candidate_digest,
        "status": status,
        "outcomes": canonicalize(outcomes),
        "evidence_hashes": tuple(item.evidence_hash for item in evidence_records),
        "formal_claim_ready": formal_claim_ready,
    }
    return TimeMachineEvaluationResult(
        dataset_id=plan.dataset_id,
        plan_hash=plan.plan_hash,
        candidate_repo_digest=candidate_digest,
        status=status,
        outcomes=outcomes,
        causal_fix_count=causal_fix_count,
        unresolved_count=unresolved_count,
        regression_count=regression_count,
        regression_detectable=regression_detectable,
        infrastructure_error_count=infrastructure_error_count,
        baseline_pass_rate=baseline_rate,
        candidate_pass_rate=candidate_rate,
        reference_pass_rate=reference_rate,
        pass_rate_uplift=candidate_rate - baseline_rate,
        formal_claim_ready=formal_claim_ready,
        evidence_records=evidence_records,
        result_hash=stable_hash(payload),
    )
