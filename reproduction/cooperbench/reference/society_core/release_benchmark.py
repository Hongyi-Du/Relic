"""Frozen cross-release OSS benchmark materialization and comparison."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import shutil
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import yaml

from .code_landing.environment import build_workspace_execution_profile
from .hashing import stable_hash


RELEASE_BENCHMARK_SCHEMA_VERSION = "release_time_machine_v1"
RELEASE_MATERIALIZATION_SCHEMA_VERSION = "release_time_machine_materialization_v1"
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_IDENTIFIER_RE = re.compile(r"^[a-z0-9][a-z0-9_]{2,79}$")
_REPOSITORY_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_DOWNLOAD_ATTEMPTS = 4
_RETRYABLE_HTTP_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})


@dataclass(frozen=True)
class ReleaseArtifactSpec:
    version: str
    released_at: str
    commit: str
    npm_tarball_url: str
    npm_sha1: str
    npm_integrity: str


@dataclass(frozen=True)
class ReleaseContractSpec:
    issue_id: str
    theme: str
    title: str
    user_pain: str
    expected_behavior: str
    reproduction_steps: tuple[str, ...]
    acceptance_commands: tuple[str, ...]
    contract_dimensions: tuple[str, ...]
    candidate_path_hints: tuple[str, ...]
    relevant_symbols_hint: tuple[str, ...]
    hidden_test_id: str
    hidden_test_source: Path
    hidden_test_sha256: str


@dataclass(frozen=True)
class ReleaseBenchmarkSpec:
    benchmark_id: str
    repository: str
    source_path: Path
    minimum_horizon_days: float
    minimum_major_version_gap: int
    base: ReleaseArtifactSpec
    reference: ReleaseArtifactSpec
    contracts: tuple[ReleaseContractSpec, ...]

    @property
    def horizon_days(self) -> float:
        elapsed = _parse_timestamp(self.reference.released_at) - _parse_timestamp(
            self.base.released_at
        )
        return elapsed.total_seconds() / 86_400.0

    @property
    def base_major_version(self) -> int:
        return _major_version(self.base.version)

    @property
    def reference_major_version(self) -> int:
        return _major_version(self.reference.version)

    @property
    def major_version_gap(self) -> int:
        return self.reference_major_version - self.base_major_version


@dataclass(frozen=True)
class MaterializedReleaseBenchmark:
    benchmark_id: str
    dataset_dir: Path
    starter_repo_dir: Path
    reference_repo_dir: Path
    starter_repo_digest: str
    reference_repo_digest: str
    horizon_days: float
    materialization_hash: str
    reused: bool


@dataclass(frozen=True)
class ReleaseDeltaComparison:
    historical_changed_paths: tuple[str, ...]
    candidate_changed_paths: tuple[str, ...]
    matched_historical_paths: tuple[str, ...]
    candidate_extra_paths: tuple[str, ...]
    historical_missing_paths: tuple[str, ...]
    path_precision: float
    path_recall: float
    path_f1: float


def load_release_benchmark(path: Path) -> ReleaseBenchmarkSpec:
    """Load a release benchmark and validate its immutable source bindings."""

    source_path = path.expanduser().resolve()
    raw = yaml.safe_load(source_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, Mapping):
        raise ValueError("release_benchmark_must_be_mapping")
    if raw.get("schema_version") != RELEASE_BENCHMARK_SCHEMA_VERSION:
        raise ValueError("unsupported_release_benchmark_schema")
    benchmark_id = _identifier(raw, "benchmark_id")
    repository = str(raw.get("repository") or "")
    if not _REPOSITORY_RE.fullmatch(repository):
        raise ValueError("invalid_release_benchmark_repository")
    minimum_horizon_days = float(raw.get("minimum_horizon_days") or 0.0)
    if minimum_horizon_days < 90.0:
        raise ValueError("release_benchmark_minimum_horizon_below_90_days")
    minimum_major_version_gap = int(raw.get("minimum_major_version_gap") or 1)
    if minimum_major_version_gap < 1:
        raise ValueError("release_benchmark_minimum_major_gap_invalid")
    base = _artifact(raw.get("base"), owner="base")
    reference = _artifact(raw.get("reference"), owner="reference")
    if base.commit == reference.commit or base.version == reference.version:
        raise ValueError("release_benchmark_base_reference_identical")
    contracts_raw = raw.get("public_contracts")
    if not isinstance(contracts_raw, list) or not contracts_raw:
        raise ValueError("release_benchmark_public_contracts_required")
    contracts = tuple(
        _contract(item, source_path.parent, index)
        for index, item in enumerate(contracts_raw)
    )
    issue_ids = tuple(item.issue_id for item in contracts)
    test_ids = tuple(item.hidden_test_id for item in contracts)
    if len(set(issue_ids)) != len(issue_ids):
        raise ValueError("release_benchmark_duplicate_issue_id")
    if len(set(test_ids)) != len(test_ids):
        raise ValueError("release_benchmark_duplicate_hidden_test_id")
    spec = ReleaseBenchmarkSpec(
        benchmark_id=benchmark_id,
        repository=repository,
        source_path=source_path,
        minimum_horizon_days=minimum_horizon_days,
        minimum_major_version_gap=minimum_major_version_gap,
        base=base,
        reference=reference,
        contracts=contracts,
    )
    if spec.horizon_days < minimum_horizon_days:
        raise ValueError("release_benchmark_horizon_too_short")
    if spec.major_version_gap < minimum_major_version_gap:
        raise ValueError("release_benchmark_major_version_gap_too_small")
    return spec


def build_release_public_source_report(
    spec: ReleaseBenchmarkSpec,
) -> dict[str, Any]:
    """Build an agent-visible report without reference code or hidden tests."""

    intents: list[dict[str, Any]] = []
    support_refs: list[str] = []
    for contract in spec.contracts:
        public_fingerprint = stable_hash(
            {
                "issue_id": contract.issue_id,
                "theme": contract.theme,
                "title": contract.title,
                "user_pain": contract.user_pain,
                "expected_behavior": contract.expected_behavior,
                "reproduction_steps": contract.reproduction_steps,
                "acceptance_commands": contract.acceptance_commands,
                "contract_dimensions": contract.contract_dimensions,
            }
        )[:16]
        support_ref = f"release_feedback:opaque_{public_fingerprint}"
        support_refs.append(support_ref)
        intents.append(
            {
                "intent_id": f"release_intent_{public_fingerprint}",
                "theme": contract.theme,
                "task_type": "cross_major_product_upgrade",
                "user_pain": contract.user_pain,
                "observed_behavior": contract.user_pain,
                "expected_behavior": contract.expected_behavior,
                "reproduction_steps": contract.reproduction_steps,
                "reproduction_expected_failure": contract.user_pain,
                "acceptance_criteria": (contract.expected_behavior,),
                "contract_dimensions": contract.contract_dimensions,
                "acceptance_tests": contract.acceptance_commands,
                "candidate_path_hints": contract.candidate_path_hints,
                "relevant_symbols_hint": contract.relevant_symbols_hint,
                "support_refs": (support_ref,),
                "evidence_refs": (support_ref,),
                "risk_level": "medium",
                "behavior_surface": ("public_node_api", contract.theme),
                "non_goals": (
                    "Do not retrieve or copy a later upstream implementation.",
                    "Do not change unrelated behavior merely to satisfy examples.",
                ),
            }
        )
    report_fingerprint = stable_hash(
        {
            "benchmark_id": spec.benchmark_id,
            "base_version": spec.base.version,
            "intents": intents,
        }
    )[:20]
    themes = tuple(dict.fromkeys(contract.theme for contract in spec.contracts))
    return {
        "report_id": f"release_public_report_{report_fingerprint}",
        "case_id": f"opaque_{report_fingerprint}",
        "product_name": spec.repository.rsplit("/", 1)[-1],
        "starting_version": spec.base.version,
        "proposed_themes": themes,
        "public_feedback_summary": {
            "title": "Cross-major user-facing capability upgrade",
            "body": "Users reported multiple independently reproducible API gaps.",
            "source_refs": tuple(support_refs),
        },
        "development_intent_specs": tuple(intents),
        "proposal": {
            "proposal_id": f"release_proposal_{report_fingerprint}",
            "artifact_id": f"release_artifact_{report_fingerprint}",
            "requested_direction": "Resolve every supported public behavior gap.",
            "priority_themes": themes,
            "support_refs": tuple(support_refs),
        },
        "maintainer_landing_gate": {
            "enabled": True,
            "require_behavior_verification": True,
            "require_intent_trace": True,
            "require_test_or_repro": True,
        },
        "execution_environment": {
            "network_available_to_workspace_checks": False,
            "dependency_installation_allowed": False,
        },
        "claim_boundary": (
            "Agent-visible evidence contains the base package and public behavior "
            "contracts only; reference code and hidden tests remain evaluator-private."
        ),
    }


def materialize_release_benchmark(
    spec: ReleaseBenchmarkSpec,
    *,
    cache_root: Path,
    refresh: bool = False,
) -> MaterializedReleaseBenchmark:
    """Download, verify, and atomically materialize both release artifacts."""

    dataset_dir = cache_root.expanduser().resolve() / spec.benchmark_id
    prior = _load_json(dataset_dir / "materialization.json")
    source_hash = _source_definition_hash(spec)
    if (
        not refresh
        and prior.get("source_definition_hash") == source_hash
        and (dataset_dir / "starter_repo").is_dir()
        and (dataset_dir / "reference_repo").is_dir()
        and prior.get("materialization_hash")
    ):
        starter_digest = build_workspace_execution_profile(
            dataset_dir / "starter_repo"
        ).repo_hash
        reference_digest = build_workspace_execution_profile(
            dataset_dir / "reference_repo"
        ).repo_hash
        if starter_digest == prior.get(
            "starter_repo_digest"
        ) and reference_digest == prior.get("reference_repo_digest"):
            return MaterializedReleaseBenchmark(
                benchmark_id=spec.benchmark_id,
                dataset_dir=dataset_dir,
                starter_repo_dir=dataset_dir / "starter_repo",
                reference_repo_dir=dataset_dir / "reference_repo",
                starter_repo_digest=starter_digest,
                reference_repo_digest=reference_digest,
                horizon_days=spec.horizon_days,
                materialization_hash=str(prior["materialization_hash"]),
                reused=True,
            )

    dataset_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(
            prefix=f".{spec.benchmark_id}_",
            dir=dataset_dir.parent,
        )
    )
    try:
        base_payload = _download(spec.base.npm_tarball_url)
        reference_payload = _download(spec.reference.npm_tarball_url)
        _verify_npm_artifact(base_payload, spec.base, owner="base")
        _verify_npm_artifact(reference_payload, spec.reference, owner="reference")
        _extract_npm_package(base_payload, stage / "starter_repo")
        _extract_npm_package(reference_payload, stage / "reference_repo")
        _write_dataset_metadata(stage, spec)
        starter_digest = build_workspace_execution_profile(
            stage / "starter_repo"
        ).repo_hash
        reference_digest = build_workspace_execution_profile(
            stage / "reference_repo"
        ).repo_hash
        if starter_digest == reference_digest:
            raise ValueError("release_benchmark_materialized_trees_identical")
        payload = {
            "schema_version": RELEASE_MATERIALIZATION_SCHEMA_VERSION,
            "benchmark_id": spec.benchmark_id,
            "repository": spec.repository,
            "base_version": spec.base.version,
            "reference_version": spec.reference.version,
            "base_commit": spec.base.commit,
            "reference_commit": spec.reference.commit,
            "base_released_at": spec.base.released_at,
            "reference_released_at": spec.reference.released_at,
            "horizon_days": round(spec.horizon_days, 9),
            "minimum_horizon_days": spec.minimum_horizon_days,
            "base_major_version": spec.base_major_version,
            "reference_major_version": spec.reference_major_version,
            "major_version_gap": spec.major_version_gap,
            "minimum_major_version_gap": spec.minimum_major_version_gap,
            "source_definition_hash": source_hash,
            "base_archive_sha256": hashlib.sha256(base_payload).hexdigest(),
            "reference_archive_sha256": hashlib.sha256(reference_payload).hexdigest(),
            "starter_repo_digest": starter_digest,
            "reference_repo_digest": reference_digest,
            "hidden_suite_hash": stable_hash(
                {
                    item.hidden_test_id: item.hidden_test_sha256
                    for item in spec.contracts
                }
            ),
        }
        payload["materialization_hash"] = stable_hash(payload)
        _write_json(stage / "materialization.json", payload)
        backup = dataset_dir.parent / f".{spec.benchmark_id}.previous"
        if backup.exists():
            shutil.rmtree(backup)
        if dataset_dir.exists():
            os.replace(dataset_dir, backup)
        os.replace(stage, dataset_dir)
        if backup.exists():
            shutil.rmtree(backup)
        return MaterializedReleaseBenchmark(
            benchmark_id=spec.benchmark_id,
            dataset_dir=dataset_dir,
            starter_repo_dir=dataset_dir / "starter_repo",
            reference_repo_dir=dataset_dir / "reference_repo",
            starter_repo_digest=starter_digest,
            reference_repo_digest=reference_digest,
            horizon_days=spec.horizon_days,
            materialization_hash=str(payload["materialization_hash"]),
            reused=False,
        )
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def compare_release_delta(
    *,
    base_root: Path,
    candidate_root: Path,
    reference_root: Path,
) -> ReleaseDeltaComparison:
    """Compare candidate and historical release deltas by changed product paths."""

    base = _file_hashes(base_root)
    candidate = _file_hashes(candidate_root)
    reference = _file_hashes(reference_root)
    historical = _changed_paths(base, reference)
    proposed = _changed_paths(base, candidate)
    matched = historical & proposed
    precision = len(matched) / len(proposed) if proposed else 0.0
    recall = len(matched) / len(historical) if historical else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return ReleaseDeltaComparison(
        historical_changed_paths=tuple(sorted(historical)),
        candidate_changed_paths=tuple(sorted(proposed)),
        matched_historical_paths=tuple(sorted(matched)),
        candidate_extra_paths=tuple(sorted(proposed - historical)),
        historical_missing_paths=tuple(sorted(historical - proposed)),
        path_precision=round(precision, 6),
        path_recall=round(recall, 6),
        path_f1=round(f1, 6),
    )


def _artifact(raw: Any, *, owner: str) -> ReleaseArtifactSpec:
    if not isinstance(raw, Mapping):
        raise ValueError(f"release_benchmark_{owner}_required")
    commit = str(raw.get("commit") or "").lower()
    if not _COMMIT_RE.fullmatch(commit):
        raise ValueError(f"release_benchmark_{owner}_commit_invalid")
    released_at = str(raw.get("released_at") or "")
    _parse_timestamp(released_at)
    tarball_url = str(raw.get("npm_tarball_url") or "")
    if not tarball_url.startswith("https://registry.npmjs.org/"):
        raise ValueError(f"release_benchmark_{owner}_tarball_url_invalid")
    sha1 = str(raw.get("npm_sha1") or "").lower()
    if not re.fullmatch(r"[0-9a-f]{40}", sha1):
        raise ValueError(f"release_benchmark_{owner}_sha1_invalid")
    integrity = str(raw.get("npm_integrity") or "")
    if not integrity.startswith("sha512-"):
        raise ValueError(f"release_benchmark_{owner}_integrity_invalid")
    return ReleaseArtifactSpec(
        version=str(raw.get("version") or "")[:80],
        released_at=released_at,
        commit=commit,
        npm_tarball_url=tarball_url,
        npm_sha1=sha1,
        npm_integrity=integrity,
    )


def _contract(raw: Any, root: Path, index: int) -> ReleaseContractSpec:
    if not isinstance(raw, Mapping):
        raise ValueError(f"release_benchmark_contract_invalid:{index}")
    issue_id = _identifier(raw, "issue_id")
    theme = _identifier(raw, "theme")
    hidden = raw.get("hidden_test")
    if not isinstance(hidden, Mapping):
        raise ValueError(f"release_benchmark_hidden_test_required:{issue_id}")
    source = (root / str(hidden.get("source") or "")).resolve()
    try:
        source.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(
            f"release_benchmark_hidden_test_outside_root:{issue_id}"
        ) from exc
    if not source.is_file() or source.is_symlink():
        raise ValueError(f"release_benchmark_hidden_test_invalid:{issue_id}")
    expected_hash = str(hidden.get("sha256") or "").lower()
    observed_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    if expected_hash != observed_hash:
        raise ValueError(f"release_benchmark_hidden_test_hash_mismatch:{issue_id}")
    commands = _strings(raw.get("acceptance_commands"))
    dimensions = _strings(raw.get("contract_dimensions"))
    if not commands or not dimensions:
        raise ValueError(f"release_benchmark_contract_incomplete:{issue_id}")
    return ReleaseContractSpec(
        issue_id=issue_id,
        theme=theme,
        title=_required_text(raw, "title", issue_id),
        user_pain=_required_text(raw, "user_pain", issue_id),
        expected_behavior=_required_text(raw, "expected_behavior", issue_id),
        reproduction_steps=_strings(raw.get("reproduction_steps")),
        acceptance_commands=commands,
        contract_dimensions=dimensions,
        candidate_path_hints=_strings(raw.get("candidate_path_hints")),
        relevant_symbols_hint=_strings(raw.get("relevant_symbols_hint")),
        hidden_test_id=_identifier(hidden, "test_id"),
        hidden_test_source=source,
        hidden_test_sha256=expected_hash,
    )


def _write_dataset_metadata(root: Path, spec: ReleaseBenchmarkSpec) -> None:
    public_dir = root / "issues" / "public"
    hidden_dir = root / "tests" / "hidden"
    heldout_dir = root / "issues" / "heldout"
    external_dir = root / "external_signals"
    contamination_dir = root / "contamination"
    for directory in (
        public_dir,
        hidden_dir,
        heldout_dir,
        external_dir,
        contamination_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    public_contracts: list[dict[str, Any]] = []
    hidden_specs: list[dict[str, Any]] = []
    for contract in spec.contracts:
        public_payload = {
            "issue_id": contract.issue_id,
            "theme": contract.theme,
            "title": contract.title,
            "body": contract.user_pain,
            "user_pain": contract.user_pain,
            "expected_behavior": contract.expected_behavior,
            "reproduction_steps": list(contract.reproduction_steps),
            "acceptance_commands": list(contract.acceptance_commands),
            "contract_dimensions": list(contract.contract_dimensions),
            "candidate_path_hints": list(contract.candidate_path_hints),
            "relevant_symbols_hint": list(contract.relevant_symbols_hint),
            "issue_type": "feature",
            "severity": "major",
            "component": "public_node_api",
            "source": "frozen_release_public_contract",
            "hidden": False,
            "source_url": (
                f"https://github.com/{spec.repository}/compare/"
                f"{spec.base.commit}...{spec.reference.commit}"
            ),
        }
        _write_json(public_dir / f"{contract.issue_id}.json", public_payload)
        public_contracts.append(public_payload)
        shutil.copy2(
            contract.hidden_test_source,
            hidden_dir / contract.hidden_test_source.name,
        )
        hidden_specs.append(
            {
                "test_id": contract.hidden_test_id,
                "name": f"Evaluator oracle for {contract.issue_id}",
                "command": [
                    "node",
                    f"tests/hidden/{contract.hidden_test_source.name}",
                ],
                "issue_ids": [contract.issue_id],
                "expected_behavior": contract.expected_behavior,
                "rel_path": contract.hidden_test_source.name,
            }
        )
    _write_json(hidden_dir / "specs.json", hidden_specs)
    _write_json(external_dir / "seed_posts.json", [])
    _write_json(
        contamination_dir / "probes.json",
        {
            "recognition_probe": "required",
            "future_recall_probe": "required",
            "reference_tree_agent_visible": False,
            "hidden_test_source_agent_visible": False,
            "known_training_overlap_risk": True,
        },
    )
    manifest = {
        "project_id": spec.benchmark_id,
        "source_project_name": spec.repository.rsplit("/", 1)[-1],
        "anonymized_product_name": spec.repository.rsplit("/", 1)[-1],
        "repo_url": f"https://github.com/{spec.repository}",
        "source_kind": "cross_release",
        "starter_ref": spec.base.version,
        "starter_commit": spec.base.commit,
        "reference_ref": spec.reference.version,
        "reference_commit": spec.reference.commit,
        "base_released_at": spec.base.released_at,
        "reference_released_at": spec.reference.released_at,
        "horizon_days": round(spec.horizon_days, 9),
        "minimum_horizon_days": spec.minimum_horizon_days,
        "base_major_version": spec.base_major_version,
        "reference_major_version": spec.reference_major_version,
        "major_version_gap": spec.major_version_gap,
        "minimum_major_version_gap": spec.minimum_major_version_gap,
        "artifact_provenance": {
            "base_tarball": spec.base.npm_tarball_url,
            "base_integrity": spec.base.npm_integrity,
            "reference_tarball": spec.reference.npm_tarball_url,
            "reference_integrity": spec.reference.npm_integrity,
        },
        "agent_visible": {
            "starter_repo": "starter_repo",
            "public_issues": "issues/public",
            "external_signals": "external_signals",
        },
        "private_evaluator_only": {
            "reference_repo": "reference_repo",
            "hidden_tests": "tests/hidden",
            "heldout_issues": "issues/heldout",
            "contamination_probes": "contamination",
        },
        "public_contracts": public_contracts,
        "hidden_tests": {"command": ["node", "tests/hidden"]},
        "evaluation": {
            "primary_metric": "hidden_behavior_causal_fix_rate",
            "secondary_metrics": [
                "historical_direction_overlap",
                "release_delta_path_overlap",
            ],
            "claim_scope": "retrospective_long_horizon_coding_reconstruction",
        },
        "contamination_controls": {
            "reference_tree_agent_visible": False,
            "hidden_test_source_agent_visible": False,
            "historical_training_contamination_excluded": False,
            "prediction_claim_allowed": False,
        },
    }
    (root / "manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False),
        encoding="utf-8",
    )


def _download(url: str) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "SocioGenesis-release-benchmark"},
    )
    last_error: Exception | None = None
    for attempt in range(_DOWNLOAD_ATTEMPTS):
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code not in _RETRYABLE_HTTP_STATUS:
                break
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
        if attempt + 1 < _DOWNLOAD_ATTEMPTS:
            time.sleep(2**attempt)
    raise RuntimeError(f"release_benchmark_download_failed:{type(last_error).__name__}")


def _verify_npm_artifact(
    payload: bytes,
    artifact: ReleaseArtifactSpec,
    *,
    owner: str,
) -> None:
    if hashlib.sha1(payload).hexdigest() != artifact.npm_sha1:
        raise ValueError(f"release_benchmark_{owner}_sha1_mismatch")
    expected = base64.b64decode(artifact.npm_integrity.removeprefix("sha512-"))
    if hashlib.sha512(payload).digest() != expected:
        raise ValueError(f"release_benchmark_{owner}_integrity_mismatch")


def _extract_npm_package(payload: bytes, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    with tarfile.open(fileobj=_bytes_reader(payload), mode="r:gz") as archive:
        for member in archive.getmembers():
            path = PurePosixPath(member.name)
            if not path.parts or path.parts[0] != "package":
                continue
            relative = PurePosixPath(*path.parts[1:])
            if not relative.parts:
                continue
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("release_benchmark_archive_path_unsafe")
            target = destination.joinpath(*relative.parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise ValueError("release_benchmark_archive_link_or_special_file")
            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise ValueError("release_benchmark_archive_member_unreadable")
            with target.open("wb") as output:
                shutil.copyfileobj(source, output)
            target.chmod(member.mode & 0o777)


def _bytes_reader(payload: bytes):
    import io

    return io.BytesIO(payload)


def _source_definition_hash(spec: ReleaseBenchmarkSpec) -> str:
    return stable_hash(
        {
            "schema_version": RELEASE_BENCHMARK_SCHEMA_VERSION,
            "repository": spec.repository,
            "minimum_horizon_days": spec.minimum_horizon_days,
            "minimum_major_version_gap": spec.minimum_major_version_gap,
            "base": spec.base,
            "reference": spec.reference,
            "contracts": tuple(
                {
                    "issue_id": item.issue_id,
                    "theme": item.theme,
                    "title": item.title,
                    "user_pain": item.user_pain,
                    "expected_behavior": item.expected_behavior,
                    "reproduction_steps": item.reproduction_steps,
                    "acceptance_commands": item.acceptance_commands,
                    "contract_dimensions": item.contract_dimensions,
                    "candidate_path_hints": item.candidate_path_hints,
                    "relevant_symbols_hint": item.relevant_symbols_hint,
                    "hidden_test_id": item.hidden_test_id,
                    "hidden_test_sha256": item.hidden_test_sha256,
                }
                for item in spec.contracts
            ),
        }
    )


def _file_hashes(root: Path) -> dict[str, str]:
    ignored = {".git", "node_modules", ".time_machine_evaluator", "__pycache__"}
    result: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(root)
        if any(part in ignored for part in relative.parts):
            continue
        result[relative.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def _changed_paths(base: Mapping[str, str], other: Mapping[str, str]) -> set[str]:
    return {
        path for path in set(base) | set(other) if base.get(path) != other.get(path)
    }


def _parse_timestamp(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("release_benchmark_timestamp_invalid") from exc
    if parsed.tzinfo is None:
        raise ValueError("release_benchmark_timestamp_timezone_required")
    return parsed


def _major_version(value: str) -> int:
    match = re.fullmatch(r"[vV]?(\d+)(?:\.\d+){0,2}(?:[-+].*)?", value.strip())
    if match is None:
        raise ValueError("release_benchmark_version_not_semver")
    return int(match.group(1))


def _identifier(raw: Mapping[str, Any], field: str) -> str:
    value = str(raw.get(field) or "")
    if not _IDENTIFIER_RE.fullmatch(value):
        raise ValueError(f"release_benchmark_identifier_invalid:{field}")
    return value


def _required_text(raw: Mapping[str, Any], field: str, owner: str) -> str:
    value = str(raw.get(field) or "").strip()
    if not value:
        raise ValueError(f"release_benchmark_text_required:{owner}:{field}")
    return value[:2000]


def _strings(value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(
        dict.fromkeys(str(item).strip()[:2000] for item in value if str(item).strip())
    )


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
