"""Repository preparation and filesystem-boundary audit for the paper protocol."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from environments.org_env.experiments.provenance import directory_content_hash
from environments.org_env.product.substrates.loader import (
    find_dataset_dir,
    load_oss_substrate_spec,
)

from .code_landing.environment import build_workspace_execution_profile
from .hashing import stable_hash
from .release_benchmark import (
    MaterializedReleaseBenchmark,
    load_release_benchmark,
    materialize_release_benchmark,
)


REPOSITORY_PREPARATION_SCHEMA_VERSION = (
    "organization_experiment_repositories_v2"
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PINNED_IMAGE_RE = re.compile(
    r"(?:^sha256:[0-9a-f]{64}$|@sha256:[0-9a-f]{64}$)"
)
_FORMAL_SOURCE_KINDS = frozenset(
    {
        "frozen_org_env_pack",
        "verified_npm_release_materialization",
    }
)


@dataclass(frozen=True)
class DatasetBoundaryAudit:
    dataset_id: str
    valid: bool
    agent_visible_roots: tuple[str, ...]
    evaluator_only_roots: tuple[str, ...]
    blocking_reasons: tuple[str, ...]
    audit_hash: str


@dataclass(frozen=True)
class ExperimentRepositoryBinding:
    repository_id: str
    dataset_dir: str
    starter_repo_digest: str
    reference_repo_digest: str
    hidden_suite_hash: str
    manifest_hash: str
    materialization_hash: str
    horizon_days: float | None
    boundary_audit_hash: str
    source_kind: str
    binding_hash: str
    evaluator_backend: str = ""
    evaluator_container_image: str = ""
    evaluator_container_platform: str = ""
    evaluator_environment_hash: str = ""
    qualification_plan_hash: str = ""
    evidence_kind: str = ""

    def __post_init__(self) -> None:
        for field in (
            "starter_repo_digest",
            "reference_repo_digest",
            "hidden_suite_hash",
            "manifest_hash",
            "materialization_hash",
            "boundary_audit_hash",
            "binding_hash",
        ):
            if not _SHA256_RE.fullmatch(str(getattr(self, field))):
                raise ValueError(f"invalid_repository_binding_hash:{field}")
        if self.starter_repo_digest == self.reference_repo_digest:
            raise ValueError("starter_and_reference_repo_digests_are_identical")
        if self.horizon_days is not None and self.horizon_days <= 0:
            raise ValueError("repository_horizon_must_be_positive")
        if self.source_kind in _FORMAL_SOURCE_KINDS:
            if self.evaluator_backend not in {"docker", "apptainer"}:
                raise ValueError("formal_repository_evaluator_backend_required")
            if not _PINNED_IMAGE_RE.search(self.evaluator_container_image):
                raise ValueError(
                    "formal_repository_evaluator_image_must_be_digest_pinned"
                )
            if not re.fullmatch(
                r"[a-z0-9][a-z0-9._-]*/[a-z0-9][a-z0-9._-]*"
                r"(?:/[a-z0-9][a-z0-9._-]*)?",
                self.evaluator_container_platform,
            ):
                raise ValueError(
                    "formal_repository_evaluator_platform_required"
                )
            for field in (
                "evaluator_environment_hash",
                "qualification_plan_hash",
            ):
                if not _SHA256_RE.fullmatch(str(getattr(self, field))):
                    raise ValueError(
                        f"invalid_repository_binding_hash:{field}"
                    )
            if not self.evidence_kind:
                raise ValueError("formal_repository_evidence_kind_required")
        expected = stable_hash(repository_binding_payload(self))
        if self.binding_hash != expected:
            raise ValueError("repository_binding_hash_mismatch")


def repository_binding_payload(
    binding: ExperimentRepositoryBinding | Mapping[str, Any],
) -> dict[str, Any]:
    def value(field: str) -> Any:
        if isinstance(binding, Mapping):
            return binding[field]
        return getattr(binding, field)

    return {
        "schema_version": REPOSITORY_PREPARATION_SCHEMA_VERSION,
        "repository_id": value("repository_id"),
        "starter_repo_digest": value("starter_repo_digest"),
        "reference_repo_digest": value("reference_repo_digest"),
        "hidden_suite_hash": value("hidden_suite_hash"),
        "manifest_hash": value("manifest_hash"),
        "materialization_hash": value("materialization_hash"),
        "horizon_days": value("horizon_days"),
        "boundary_audit_hash": value("boundary_audit_hash"),
        "source_kind": value("source_kind"),
        "evaluator_backend": value("evaluator_backend"),
        "evaluator_container_image": value("evaluator_container_image"),
        "evaluator_container_platform": value(
            "evaluator_container_platform"
        ),
        "evaluator_environment_hash": value("evaluator_environment_hash"),
        "qualification_plan_hash": value("qualification_plan_hash"),
        "evidence_kind": value("evidence_kind"),
    }


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _resolved(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def audit_dataset_boundary(dataset_id: str) -> DatasetBoundaryAudit:
    """Verify that evaluator-only assets cannot enter the starter workspace."""

    spec = load_oss_substrate_spec(dataset_id)
    dataset_root = _resolved(spec.dataset_dir)
    agent_roots = tuple(
        _resolved(path)
        for path in (
            spec.starter_repo_dir,
            spec.public_issues_dir,
            spec.external_signals_dir,
        )
    )
    evaluator_roots = tuple(
        _resolved(path)
        for path in (
            spec.reference_repo_dir,
            spec.hidden_tests_dir,
            spec.heldout_issues_dir,
            spec.contamination_dir,
        )
    )
    reasons: list[str] = []
    if not dataset_root.is_dir():
        reasons.append("dataset_root_missing")
    for root in (*agent_roots, *evaluator_roots):
        if not root.exists():
            reasons.append(f"required_root_missing:{root.name}")
        if not _inside(root, dataset_root):
            reasons.append(f"root_outside_dataset:{root.name}")
    starter = agent_roots[0]
    for private_root in evaluator_roots:
        if _inside(private_root, starter) or _inside(starter, private_root):
            reasons.append(
                f"starter_evaluator_root_overlap:{private_root.name}"
            )
    for visible_root in agent_roots:
        if not visible_root.exists():
            continue
        for path in visible_root.rglob("*"):
            if not path.is_symlink():
                continue
            try:
                target = path.resolve(strict=True)
            except OSError:
                reasons.append(
                    "agent_visible_contains_unresolvable_symlink:"
                    f"{visible_root.name}:"
                    f"{path.relative_to(visible_root).as_posix()}"
                )
                continue
            if any(
                _inside(target, private_root)
                for private_root in evaluator_roots
            ):
                reasons.append(
                    "agent_visible_symlink_reaches_evaluator:"
                    f"{visible_root.name}:"
                    f"{path.relative_to(visible_root).as_posix()}"
                )
    payload = {
        "schema_version": REPOSITORY_PREPARATION_SCHEMA_VERSION,
        "dataset_id": spec.project_id,
        "agent_visible_roots": [
            root.relative_to(dataset_root).as_posix()
            for root in agent_roots
            if _inside(root, dataset_root)
        ],
        "evaluator_only_roots": [
            root.relative_to(dataset_root).as_posix()
            for root in evaluator_roots
            if _inside(root, dataset_root)
        ],
        "blocking_reasons": reasons,
    }
    return DatasetBoundaryAudit(
        dataset_id=spec.project_id,
        valid=not reasons,
        agent_visible_roots=tuple(str(root) for root in agent_roots),
        evaluator_only_roots=tuple(str(root) for root in evaluator_roots),
        blocking_reasons=tuple(dict.fromkeys(reasons)),
        audit_hash=stable_hash(payload),
    )


def _parse_date(value: Any) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _manifest_horizon_days(manifest: Mapping[str, Any]) -> float | None:
    start = _parse_date(
        manifest.get("starter_release_date")
        or manifest.get("base_released_at")
    )
    end = _parse_date(
        manifest.get("reference_release_date")
        or manifest.get("reference_released_at")
    )
    if start is None or end is None:
        return None
    return round((end - start).total_seconds() / 86_400.0, 9)


def build_repository_binding(
    dataset_id: str,
    *,
    source_kind: str,
    materialization_hash: str | None = None,
    evaluator_backend: str = "",
    evaluator_container_image: str = "",
    evaluator_container_platform: str = "",
    evaluator_environment_hash: str = "",
    qualification_plan_hash: str = "",
    evidence_kind: str = "",
) -> ExperimentRepositoryBinding:
    spec = load_oss_substrate_spec(dataset_id)
    audit = audit_dataset_boundary(dataset_id)
    if not audit.valid:
        raise ValueError(
            "dataset_boundary_audit_failed:"
            + ",".join(audit.blocking_reasons)
        )
    starter_digest = build_workspace_execution_profile(
        _resolved(spec.starter_repo_dir)
    ).repo_hash
    reference_digest = build_workspace_execution_profile(
        _resolved(spec.reference_repo_dir)
    ).repo_hash
    if starter_digest == reference_digest:
        raise ValueError("starter_and_reference_repo_digests_are_identical")
    from .programbench_evaluation import (
        is_programbench_spec,
        programbench_hidden_suite_hash,
    )

    hidden_hash = (
        programbench_hidden_suite_hash(spec)
        if is_programbench_spec(spec)
        else directory_content_hash(spec.hidden_tests_dir)
    )
    manifest_hash = stable_hash(spec.manifest)
    fields = {
        "repository_id": spec.project_id,
        "dataset_dir": str(_resolved(spec.dataset_dir)),
        "starter_repo_digest": starter_digest,
        "reference_repo_digest": reference_digest,
        "hidden_suite_hash": hidden_hash,
        "manifest_hash": manifest_hash,
        "materialization_hash": materialization_hash or manifest_hash,
        "horizon_days": _manifest_horizon_days(spec.manifest),
        "boundary_audit_hash": audit.audit_hash,
        "source_kind": source_kind,
        "evaluator_backend": evaluator_backend,
        "evaluator_container_image": evaluator_container_image,
        "evaluator_container_platform": evaluator_container_platform,
        "evaluator_environment_hash": evaluator_environment_hash,
        "qualification_plan_hash": qualification_plan_hash,
        "evidence_kind": evidence_kind,
    }
    return ExperimentRepositoryBinding(
        **fields,
        binding_hash=stable_hash(repository_binding_payload(fields)),
    )


def validate_repository_binding(
    binding: ExperimentRepositoryBinding,
    *,
    verify_filesystem: bool = False,
) -> None:
    """Validate semantic integrity and optionally rehash the bound dataset."""

    ExperimentRepositoryBinding(**asdict(binding))
    if not verify_filesystem:
        return
    observed = build_repository_binding(
        binding.dataset_dir,
        source_kind=binding.source_kind,
        materialization_hash=binding.materialization_hash,
        evaluator_backend=binding.evaluator_backend,
        evaluator_container_image=binding.evaluator_container_image,
        evaluator_container_platform=binding.evaluator_container_platform,
        evaluator_environment_hash=binding.evaluator_environment_hash,
        qualification_plan_hash=binding.qualification_plan_hash,
        evidence_kind=binding.evidence_kind,
    )
    if observed != binding:
        raise ValueError(f"repository_binding_filesystem_drift:{binding.repository_id}")


def prepare_organizational_experiment_repositories(
    *,
    main_repository_ids: Sequence[str],
    third_repository_definition: Path,
    materialization_root: Path,
    evaluator_bindings: Mapping[str, Mapping[str, str]],
    qualification_timeout_seconds: int = 300,
    refresh: bool = False,
) -> tuple[ExperimentRepositoryBinding, ...]:
    """Qualify repositories in their frozen evaluator runtimes."""

    release_spec = load_release_benchmark(third_repository_definition)
    materialized: MaterializedReleaseBenchmark = materialize_release_benchmark(
        release_spec,
        cache_root=materialization_root,
        refresh=refresh,
    )
    datasets = [
        (
            repository_id,
            find_dataset_dir(repository_id),
            "frozen_org_env_pack",
            None,
        )
        for repository_id in main_repository_ids
    ]
    datasets.append(
        (
            materialized.benchmark_id,
            str(materialized.dataset_dir),
            "verified_npm_release_materialization",
            materialized.materialization_hash,
        )
    )
    bindings: list[ExperimentRepositoryBinding] = []
    for repository_id, dataset_id, source_kind, materialization_hash in datasets:
        runtime = dict(evaluator_bindings.get(repository_id) or {})
        backend = str(runtime.get("backend") or "docker")
        image = str(runtime.get("container_image") or "")
        platform = str(runtime.get("container_platform") or "")
        from .execution import ExecutionPolicy, build_command_executor
        from .time_machine_evaluation import build_time_machine_evaluation_plan

        executor = build_command_executor(
            ExecutionPolicy(
                trust_level="untrusted",
                backend=backend,
                container_image=image,
                container_platform=platform,
                network_enabled=False,
            )
        )
        qualification = build_time_machine_evaluation_plan(
            dataset_id=str(dataset_id),
            timeout_seconds=qualification_timeout_seconds,
            executor=executor,
        )
        if not qualification.formal_ready:
            raise ValueError(
                "repository_formal_qualification_failed:"
                f"{repository_id}:"
                + ",".join(qualification.blocking_reasons)
            )
        spec = load_oss_substrate_spec(str(dataset_id))
        strategy = spec.manifest.get("test_strategy") or {}
        evidence_kind = (
            str(strategy.get("kind"))
            if isinstance(strategy, Mapping) and strategy.get("kind")
            else "behavior"
        )
        bindings.append(
            build_repository_binding(
                str(dataset_id),
                source_kind=source_kind,
                materialization_hash=materialization_hash,
                evaluator_backend=backend,
                evaluator_container_image=image,
                evaluator_container_platform=platform,
                evaluator_environment_hash=(
                    qualification.evaluator_environment_hash
                ),
                qualification_plan_hash=qualification.plan_hash,
                evidence_kind=evidence_kind,
            )
        )
    return tuple(bindings)


def repository_preparation_manifest(
    bindings: Sequence[ExperimentRepositoryBinding],
) -> dict[str, Any]:
    payload = {
        "schema_version": REPOSITORY_PREPARATION_SCHEMA_VERSION,
        "bindings": [asdict(binding) for binding in bindings],
    }
    return {
        **payload,
        "valid": bool(bindings)
        and len({binding.repository_id for binding in bindings})
        == len(bindings),
        "preparation_hash": stable_hash(
            {
                "schema_version": REPOSITORY_PREPARATION_SCHEMA_VERSION,
                "binding_hashes": [
                    binding.binding_hash for binding in bindings
                ],
            }
        ),
    }


__all__ = [
    "DatasetBoundaryAudit",
    "ExperimentRepositoryBinding",
    "REPOSITORY_PREPARATION_SCHEMA_VERSION",
    "audit_dataset_boundary",
    "build_repository_binding",
    "prepare_organizational_experiment_repositories",
    "repository_binding_payload",
    "repository_preparation_manifest",
    "validate_repository_binding",
]
