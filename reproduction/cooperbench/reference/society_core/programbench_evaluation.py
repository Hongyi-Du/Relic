"""Fail-closed adapter for the official ProgramBench 1.2.4 evaluator.

The adapter deliberately does not implement ProgramBench's scoring rules.  It
validates the evaluator-private pack, creates the official submission/blob
layout, hardens ProgramBench's process-global Docker arguments, and delegates
the compile/test lifecycle to :class:`programbench.eval.eval.Evaluator`.
"""

from __future__ import annotations

import base64
import csv
import gzip
import hashlib
import importlib
import importlib.metadata
import io
import json
import os
import re
import shutil
import stat
import subprocess
import tarfile
import tempfile
import threading
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal


PROGRAMBENCH_RUNNER = "programbench_official_v1"

_CONFIG_SCHEMA = "programbench_pack_eval_v1"
_PROGRAMBENCH_VERSION = "1.2.4"
_PROGRAMBENCH_WHEEL_SHA256 = (
    "bc24d2879a9d0249ae7e4f7770a7df6c069767f491444d846fe0a75070c572b8"
)
_PROGRAMBENCH_PACKAGE_FILE_COUNT = 435
_PROGRAMBENCH_PACKAGE_TOTAL_BYTES = 38_377_225
_PROGRAMBENCH_PACKAGE_INVENTORY_SHA256 = (
    "b3f3645267bc93d1a902f7b37cab3081352a78b84fa436b911e51bd39f2cc10b"
)
_PYTEST_TIMEOUT_VERSION = "2.4.0"
_CLEANROOM_DATASET_VERSION = "v6"
_CONTRACT_ENTRYPOINT = "compile.sh"
_OUTPUT_PATH = "executable"
_IMAGE_TAG = "task_cleanroom_v6"
_PLATFORM = "linux/amd64"
_DOCKER_EXECUTABLE = "docker"
_EXPECTED_DOCKER_CPUS = 4
_EXPECTED_MEMORY_LIMIT_MB = 4096
_EXPECTED_PIDS_LIMIT = 256
_EXPECTED_BRANCH_WORKERS = 1
_EXPECTED_BRANCH_RETRIES = 1
_MAX_CONFIG_BYTES = 4 * 1024 * 1024
_MAX_REPOSITORY_BYTES = 1024 * 1024 * 1024
_MAX_REPOSITORY_ENTRIES = 200_000
_MAX_ATTESTATION_ARCHIVE_BYTES = _MAX_REPOSITORY_BYTES + 256 * 1024 * 1024
_MAX_EXECUTABLE_BYTES = 512 * 1024 * 1024
_MAX_TRUSTED_DEPENDENCY_FILE_BYTES = 4 * 1024 * 1024
_MAX_DOCKER_ERROR_BYTES = 64 * 1024
_ATTESTATION_COPY_TIMEOUT_SECONDS = 300
_CANDIDATE_LAUNCHER = b'#!/bin/bash\nexec -a "$0" /candidate/executable "$@"\n'
_CANDIDATE_LAUNCHER_SHA256 = hashlib.sha256(_CANDIDATE_LAUNCHER).hexdigest()
# Restores the workspace git repository the pinned cleanroom image ships and the
# hardened sandbox drops.  Fixed author and committer dates make the seeded
# commit a function of the tree alone, so repeated runs over the same workspace
# agree; the identity is written locally so no host or global git config is read.
# ``add -A`` is what lets ``git grep`` see the branch fixtures, since git grep
# searches tracked content only.
_WORKSPACE_GIT_SEED_COMMAND = (
    "GIT_AUTHOR_DATE='2000-01-01T00:00:00Z' "
    "GIT_COMMITTER_DATE='2000-01-01T00:00:00Z' "
    "git -c init.defaultBranch=master init -q . && "
    "git config user.email evaluator@programbench.local && "
    "git config user.name ProgramBench && "
    "git add -A && "
    "GIT_AUTHOR_DATE='2000-01-01T00:00:00Z' "
    "GIT_COMMITTER_DATE='2000-01-01T00:00:00Z' "
    "git -c commit.gpgsign=false commit -q --allow-empty "
    "-m 'programbench workspace baseline'"
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_GIT_SHA_RE = re.compile(r"[0-9a-f]{40}")
_BRANCH_RE = re.compile(r"[0-9a-f]{12}")
_INSTANCE_RE = re.compile(r"[A-Za-z0-9_.-]+__[A-Za-z0-9_.-]+\.([0-9a-f]{7})")
_DOCKER_REPOSITORY_RE = re.compile(
    r"[a-z0-9]+(?:[._-][a-z0-9]+)*(?:/[a-z0-9]+(?:[._-][a-z0-9]+)*)+"
)

_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "runner",
        "programbench_version",
        "cleanroom",
        "task",
        "official_eval_clean_hashes",
        "reference_executable",
        "candidate_rejected_hashes",
        "branches",
        "active_test_ids",
        "provenance",
    }
)
_CLEANROOM_KEYS = frozenset(
    {"dataset_version", "contract_entrypoint", "output_path", "resources", "image"}
)
_RESOURCE_KEYS = frozenset(
    {
        "docker_cpus",
        "memory_limit_mb",
        "pids_limit",
        "branch_workers",
        "branch_retries",
    }
)
_IMAGE_KEYS = frozenset(
    {"repository", "tag", "digest", "reference", "platform", "pull_policy"}
)
_TASK_KEYS = frozenset(
    {"instance_id", "repository", "commit", "language", "difficulty"}
)
_REFERENCE_KEYS = frozenset({"sha256", "size", "private_path"})
_BRANCH_KEYS = frozenset(
    {
        "name",
        "blob_path",
        "sha256",
        "size",
        "ignored",
        "ignore_reason",
        "tests",
        "ignored_tests",
        "active_test_ids",
    }
)
_PROVENANCE_KEYS = frozenset(
    {
        "programbench_commit",
        "hf_revision",
        "task_sha256",
        "tests_sha256",
        "attribution_sha256",
        "license_sha256",
    }
)
_CANDIDATE_BUILD_FAILURES = frozenset(
    {"compile_failed", "copy_executable_failed", "hash_executable_failed"}
)
_VALID_ROLES = frozenset({"baseline", "reference", "candidate"})
_PROCESS_GLOBALS_LOCK = threading.RLock()


class ProgramBenchConfigurationError(ValueError):
    """The evaluator-private ProgramBench pack is invalid."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


class ProgramBenchDependencyError(RuntimeError):
    """The pinned official ProgramBench dependency cannot be used."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


class _RepositoryRejected(ValueError):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


class _TrustedAttestationError(RuntimeError):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class _Branch:
    name: str
    blob_path: str
    sha256: str
    size: int
    ignored: bool
    ignore_reason: str
    tests: tuple[str, ...]
    ignored_tests: tuple[str, ...]
    declared_active_test_ids: tuple[str, ...]

    @property
    def active_tests(self) -> tuple[str, ...]:
        ignored = frozenset(self.ignored_tests)
        return tuple(test for test in self.tests if test not in ignored)


@dataclass(frozen=True)
class _Resources:
    docker_cpus: int
    memory_limit_mb: int
    pids_limit: int
    branch_workers: int
    branch_retries: int

    @property
    def docker_run_args(self) -> tuple[str, ...]:
        return (
            "--pull",
            "never",
            "--network",
            "none",
            "--platform",
            _PLATFORM,
            "--cap-drop",
            "ALL",
            "--cap-add",
            "DAC_OVERRIDE",
            "--security-opt",
            "no-new-privileges",
            "--memory",
            f"{self.memory_limit_mb}m",
            "--pids-limit",
            str(self.pids_limit),
        )

    def as_mapping(self) -> Mapping[str, int]:
        return {
            "docker_cpus": self.docker_cpus,
            "memory_limit_mb": self.memory_limit_mb,
            "pids_limit": self.pids_limit,
            "branch_workers": self.branch_workers,
            "branch_retries": self.branch_retries,
        }


@dataclass(frozen=True)
class _Config:
    hidden_root: Path
    config_path: Path
    config_sha256: str
    instance_id: str
    repository: str
    commit: str
    language: str
    difficulty: str
    image_repository: str
    image_tag: str
    image_digest: str
    image_platform: str
    resources: _Resources
    official_eval_clean_hashes: tuple[str, ...]
    reference_sha256: str
    reference_size: int
    candidate_rejected_hashes: tuple[str, ...]
    branches: tuple[_Branch, ...]
    active_test_ids: tuple[str, ...]
    provenance: Mapping[str, str]

    @property
    def image_ref(self) -> str:
        return f"{self.image_repository}:{self.image_tag}@{self.image_digest}"

    @property
    def forbidden_hashes(self) -> tuple[str, ...]:
        return self.candidate_rejected_hashes

    @property
    def active_branches(self) -> tuple[_Branch, ...]:
        return tuple(branch for branch in self.branches if not branch.ignored)

    @property
    def ignored_test_ids(self) -> frozenset[str]:
        return frozenset(
            f"{branch.name}/{test}"
            for branch in self.branches
            for test in branch.ignored_tests
        )


@dataclass(frozen=True)
class _OfficialApi:
    version: str
    evaluator_cls: type[Any]
    eval_step_error_cls: type[Exception]
    eval_module: Any
    constants_module: Any
    package_root: Path


@dataclass(frozen=True)
class _TrustedDependencyFile:
    root: Path
    path: Path
    relative: str
    sha256: str
    size: int


@dataclass(frozen=True)
class _PostCompileAttestation:
    executable_sha256: str
    executable_size: int
    workspace_hashes: frozenset[str]
    workspace_identity: tuple[tuple[str, str, int], ...] = ()


@dataclass(frozen=True)
class _TreeEntry:
    relative: str
    path: Path
    kind: Literal["directory", "file"]
    mode: int
    size: int = 0
    sha256: str = ""


def is_programbench_spec(spec: Any) -> bool:
    """Return whether *spec* selects the official ProgramBench runner.

    Merely having a malformed ``programbench.json`` still selects this path so
    an invalid evaluator pack cannot silently fall back to the legacy host
    runner.
    """

    manifest = getattr(spec, "manifest", {}) or {}
    strategy = manifest.get("test_strategy") if isinstance(manifest, Mapping) else None
    if isinstance(strategy, Mapping) and strategy.get("runner") == PROGRAMBENCH_RUNNER:
        return True
    hidden = str(getattr(spec, "hidden_tests_dir", "") or "")
    return bool(hidden and (Path(hidden) / "programbench.json").exists())


def programbench_hidden_suite_hash(spec: Any) -> str:
    """Hash every evaluator-private file, including opaque branch archives."""

    config = _load_config(spec)
    entries = _walk_tree(
        config.hidden_root,
        error_type=ProgramBenchConfigurationError,
        error_prefix="programbench_hidden_suite",
        byte_limit=None,
        entry_limit=None,
    )
    return _stable_hash(
        {
            "schema": "programbench_hidden_suite_v1",
            "files": [
                {
                    "path": entry.relative,
                    "kind": entry.kind,
                    "size": entry.size,
                    "sha256": entry.sha256,
                }
                for entry in entries
            ],
        }
    )


def programbench_runtime_identity(spec: Any) -> Mapping[str, Any]:
    """Return a host-independent identity for the exact official runtime."""

    config = _load_config(spec)
    api = _load_official_api()
    official_files = _official_file_hashes(api)
    trusted_dependency_files = _trusted_test_dependency_files()
    return {
        "schema_version": "programbench_official_runtime_v1",
        "runner": PROGRAMBENCH_RUNNER,
        "official_package": {
            "name": "programbench",
            "version": api.version,
            "wheel_sha256": _PROGRAMBENCH_WHEEL_SHA256,
            "package_inventory_sha256": _PROGRAMBENCH_PACKAGE_INVENTORY_SHA256,
            "files": official_files,
        },
        "trusted_test_dependencies": {
            "pytest-timeout": {
                "version": _PYTEST_TIMEOUT_VERSION,
                "files": {
                    entry.relative: {
                        "sha256": entry.sha256,
                        "size": entry.size,
                    }
                    for entry in trusted_dependency_files
                },
            }
        },
        "config_sha256": config.config_sha256,
        "cleanroom": {
            "dataset_version": _CLEANROOM_DATASET_VERSION,
            "contract_entrypoint": _CONTRACT_ENTRYPOINT,
            "output_path": _OUTPUT_PATH,
            "resources": config.resources.as_mapping(),
            "image": {
                "repository": config.image_repository,
                "tag": config.image_tag,
                "digest": config.image_digest,
                "platform": config.image_platform,
                "reference": config.image_ref,
                "pull_policy": "never",
            },
        },
        "task": {
            "instance_id": config.instance_id,
            "repository": config.repository,
            "commit": config.commit,
            "language": config.language,
            "difficulty": config.difficulty,
        },
        "official_eval_clean_hashes": config.official_eval_clean_hashes,
        "reference_executable": {
            "sha256": config.reference_sha256,
            "size": config.reference_size,
            "private_path": ".programbench/reference-executable",
        },
        "candidate_rejected_hashes": config.candidate_rejected_hashes,
        "branches": tuple(
            {
                "name": branch.name,
                "sha256": branch.sha256,
                "size": branch.size,
                "ignored": branch.ignored,
                "ignore_reason": branch.ignore_reason,
                "tests": branch.tests,
                "ignored_tests": branch.ignored_tests,
                "active_test_ids": branch.declared_active_test_ids,
            }
            for branch in config.branches
        ),
        "active_test_ids": config.active_test_ids,
        "provenance": dict(config.provenance),
        "security_boundary": {
            "docker_run_args": config.resources.docker_run_args,
            "docker_cpus": config.resources.docker_cpus,
            "repository_max_bytes": _MAX_REPOSITORY_BYTES,
            "repository_max_entries": _MAX_REPOSITORY_ENTRIES,
            "candidate_build_rootfs_committed": False,
            "candidate_tree_path": "/candidate",
            "hidden_workspace_path": "/workspace",
            "candidate_launcher_sha256": _CANDIDATE_LAUNCHER_SHA256,
        },
    }


def programbench_formal_blocking_reasons(spec: Any) -> tuple[str, ...]:
    """Report distinct config, dependency, Docker, and local-image blockers."""

    reasons: list[str] = []
    if not is_programbench_spec(spec):
        return ("programbench_runner_not_selected",)
    try:
        config = _load_config(spec)
    except ProgramBenchConfigurationError as error:
        return (f"programbench_config_invalid:{error.code}",)
    try:
        api = _load_official_api()
        _official_file_hashes(api)
        _trusted_test_dependency_files()
    except ProgramBenchDependencyError as error:
        reasons.append(error.code)
    reasons.extend(_docker_preflight_reasons(config.image_ref))
    return tuple(dict.fromkeys(reasons))


def run_programbench_evaluation(
    spec: Any,
    repo_dir: str | Path,
    timeout: int = 60,
    role: Literal["baseline", "reference", "candidate"] = "candidate",
) -> Mapping[str, Any]:
    """Evaluate a repository through ProgramBench's official ``Evaluator``.

    ``role`` is intentionally an explicit closed set.  Reference privileges
    cannot be enabled through a truthy flag supplied by a candidate caller.
    """

    if not isinstance(role, str) or role not in _VALID_ROLES:
        raise ValueError("programbench_invalid_evaluation_role")
    if isinstance(timeout, bool) or not isinstance(timeout, int) or timeout <= 0:
        raise ValueError("programbench_timeout_must_be_positive_integer")
    try:
        config = _load_config(spec)
    except ProgramBenchConfigurationError as error:
        return _unconfigured_result(role, f"programbench_config_invalid:{error.code}")

    try:
        api = _load_official_api()
    except ProgramBenchDependencyError as error:
        return _uniform_result(config, role, "infra_error", error.code)

    repository = Path(repo_dir)
    try:
        source_entries = _walk_tree(
            repository,
            error_type=_RepositoryRejected,
            error_prefix="programbench_repository",
            byte_limit=_MAX_REPOSITORY_BYTES,
            entry_limit=_MAX_REPOSITORY_ENTRIES,
        )
    except (OSError, _RepositoryRejected) as error:
        code = getattr(error, "code", "programbench_repository_unavailable")
        status = "infra_error" if role == "reference" else "blocked"
        return _uniform_result(config, role, status, str(code))

    source_hashes = frozenset(
        entry.sha256 for entry in source_entries if entry.kind == "file"
    )
    forbidden = frozenset(config.forbidden_hashes)
    if role != "reference" and source_hashes & forbidden:
        return _uniform_result(
            config, role, "blocked", "programbench_forbidden_artifact_in_submission"
        )
    if role == "reference" and config.reference_sha256 not in source_hashes:
        return _uniform_result(
            config, role, "infra_error", "programbench_reference_artifact_missing"
        )

    try:
        with tempfile.TemporaryDirectory(prefix="programbench_adapter_") as raw_tmp:
            temporary_root = Path(raw_tmp)
            submission_archive = temporary_root / "submission.tar.gz"
            _write_submission_archive(
                repository,
                source_entries,
                submission_archive,
            )
            blob_root = temporary_root / "blobs"
            _stage_branch_blobs(config, blob_root)
            result = _run_official_evaluator(
                api=api,
                config=config,
                submission_archive=submission_archive,
                blob_root=blob_root,
                timeout=timeout,
                role=role,
            )
    except _RepositoryRejected as error:
        status = "infra_error" if role == "reference" else "blocked"
        return _uniform_result(config, role, status, error.code)
    except ProgramBenchConfigurationError as error:
        return _uniform_result(
            config,
            role,
            "infra_error",
            f"programbench_assets_changed:{error.code}",
        )
    except Exception as error:  # Official Docker boundary: fail closed.
        return _uniform_result(
            config,
            role,
            "infra_error",
            "programbench_official_evaluator_exception",
            diagnostic_extra={"exception_type": type(error).__name__},
        )

    return _translate_official_result(config, role, result)


def _load_config(spec: Any) -> _Config:
    hidden_root = _validated_hidden_root(spec)
    config_path = hidden_root / "programbench.json"
    raw = _read_regular_bytes(
        hidden_root,
        config_path,
        max_bytes=_MAX_CONFIG_BYTES,
        error_type=ProgramBenchConfigurationError,
        error_prefix="programbench_config",
    )
    try:
        payload = json.loads(raw, object_pairs_hook=_reject_duplicate_json_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProgramBenchConfigurationError("invalid_json") from error
    root = _mapping(payload, "root")
    _exact_keys(root, _TOP_LEVEL_KEYS, "root")
    _equal_string(root, "schema_version", _CONFIG_SCHEMA)
    _equal_string(root, "runner", PROGRAMBENCH_RUNNER)
    _equal_string(root, "programbench_version", _PROGRAMBENCH_VERSION)

    cleanroom = _mapping(root.get("cleanroom"), "cleanroom")
    _exact_keys(cleanroom, _CLEANROOM_KEYS, "cleanroom")
    _equal_string(cleanroom, "dataset_version", _CLEANROOM_DATASET_VERSION)
    _equal_string(cleanroom, "contract_entrypoint", _CONTRACT_ENTRYPOINT)
    _equal_string(cleanroom, "output_path", _OUTPUT_PATH)

    raw_resources = _mapping(cleanroom.get("resources"), "cleanroom_resources")
    _exact_keys(raw_resources, _RESOURCE_KEYS, "cleanroom_resources")
    resources = _Resources(
        docker_cpus=_positive_int(
            raw_resources.get("docker_cpus"), "cleanroom_docker_cpus"
        ),
        memory_limit_mb=_positive_int(
            raw_resources.get("memory_limit_mb"), "cleanroom_memory_limit_mb"
        ),
        pids_limit=_positive_int(
            raw_resources.get("pids_limit"), "cleanroom_pids_limit"
        ),
        branch_workers=_positive_int(
            raw_resources.get("branch_workers"), "cleanroom_branch_workers"
        ),
        branch_retries=_nonnegative_int(
            raw_resources.get("branch_retries"), "cleanroom_branch_retries"
        ),
    )
    expected_resources = _Resources(
        docker_cpus=_EXPECTED_DOCKER_CPUS,
        memory_limit_mb=_EXPECTED_MEMORY_LIMIT_MB,
        pids_limit=_EXPECTED_PIDS_LIMIT,
        branch_workers=_EXPECTED_BRANCH_WORKERS,
        branch_retries=_EXPECTED_BRANCH_RETRIES,
    )
    if resources != expected_resources:
        raise ProgramBenchConfigurationError("unsupported_cleanroom_resources")

    image = _mapping(cleanroom.get("image"), "cleanroom_image")
    _exact_keys(image, _IMAGE_KEYS, "image")
    image_repository = _string(image, "repository")
    image_tag = _string(image, "tag")
    image_digest = _sha256(image.get("digest"), "image_digest", prefixed=True)
    image_reference = _string(image, "reference")
    image_platform = _string(image, "platform")
    image_pull_policy = _string(image, "pull_policy")
    if not _DOCKER_REPOSITORY_RE.fullmatch(image_repository):
        raise ProgramBenchConfigurationError("invalid_image_repository")
    if image_tag != _IMAGE_TAG:
        raise ProgramBenchConfigurationError("unsupported_cleanroom_image_tag")
    if image_platform != _PLATFORM:
        raise ProgramBenchConfigurationError("unsupported_image_platform")
    expected_image_reference = f"{image_repository}:{image_tag}@{image_digest}"
    if image_reference != expected_image_reference:
        raise ProgramBenchConfigurationError("image_reference_mismatch")
    if image_pull_policy != "never":
        raise ProgramBenchConfigurationError("unsupported_image_pull_policy")

    task = _mapping(root.get("task"), "task")
    _exact_keys(task, _TASK_KEYS, "task")
    instance_id = _string(task, "instance_id")
    repository = _string(task, "repository")
    commit = _string(task, "commit")
    language = _string(task, "language")
    difficulty = _string(task, "difficulty")
    instance_match = _INSTANCE_RE.fullmatch(instance_id)
    if instance_match is None:
        raise ProgramBenchConfigurationError("invalid_task_instance_id")
    if _GIT_SHA_RE.fullmatch(commit) is None:
        raise ProgramBenchConfigurationError("invalid_task_commit")
    if instance_match.group(1) != commit[:7]:
        raise ProgramBenchConfigurationError("task_instance_commit_mismatch")
    expected_task_repository = instance_id.rsplit(".", 1)[0].replace("__", "/")
    if repository != expected_task_repository:
        raise ProgramBenchConfigurationError("task_repository_mismatch")
    expected_image_repository = f"programbench/{instance_id.replace('__', '_1776_')}"
    if image_repository != expected_image_repository:
        raise ProgramBenchConfigurationError("task_image_repository_mismatch")

    raw_clean_hashes = _sequence(root.get("official_eval_clean_hashes"), "clean_hashes")
    official_eval_clean_hashes = tuple(
        _sha256(value, f"official_eval_clean_hashes_{index}")
        for index, value in enumerate(raw_clean_hashes)
    )
    if not official_eval_clean_hashes:
        raise ProgramBenchConfigurationError("official_eval_clean_hashes_empty")
    if len(set(official_eval_clean_hashes)) != len(official_eval_clean_hashes):
        raise ProgramBenchConfigurationError("duplicate_official_eval_clean_hash")

    reference = _mapping(root.get("reference_executable"), "reference_executable")
    _exact_keys(reference, _REFERENCE_KEYS, "reference_executable")
    reference_sha256 = _sha256(reference.get("sha256"), "reference_executable_sha256")
    reference_size = _positive_int(reference.get("size"), "reference_executable_size")
    if _string(reference, "private_path") != ".programbench/reference-executable":
        raise ProgramBenchConfigurationError(
            "reference_executable_private_path_mismatch"
        )

    raw_rejected_hashes = _sequence(
        root.get("candidate_rejected_hashes"), "candidate_rejected_hashes"
    )
    candidate_rejected_hashes = tuple(
        _sha256(value, f"candidate_rejected_hashes_{index}")
        for index, value in enumerate(raw_rejected_hashes)
    )
    if len(set(candidate_rejected_hashes)) != len(candidate_rejected_hashes):
        raise ProgramBenchConfigurationError("duplicate_candidate_rejected_hash")
    expected_rejected_hashes = tuple(
        sorted({*official_eval_clean_hashes, reference_sha256})
    )
    if candidate_rejected_hashes != expected_rejected_hashes:
        raise ProgramBenchConfigurationError("candidate_rejected_hashes_mismatch")

    raw_branches = _sequence(root.get("branches"), "branches")
    if not raw_branches:
        raise ProgramBenchConfigurationError("branches_empty")
    branches: list[_Branch] = []
    for index, value in enumerate(raw_branches):
        branch = _mapping(value, f"branch_{index}")
        _exact_keys(branch, _BRANCH_KEYS, f"branch_{index}")
        name = _string(branch, "name")
        if _BRANCH_RE.fullmatch(name) is None:
            raise ProgramBenchConfigurationError("invalid_branch_name")
        blob_path = _relative_posix_path(branch.get("blob_path"), "branch_blob_path")
        blob_sha256 = _sha256(branch.get("sha256"), "branch_blob_sha256")
        blob_size = _positive_int(branch.get("size"), "branch_blob_size")
        ignored = branch.get("ignored")
        if not isinstance(ignored, bool):
            raise ProgramBenchConfigurationError("branch_ignored_not_boolean")
        ignore_reason = branch.get("ignore_reason")
        if not isinstance(ignore_reason, str) or "\x00" in ignore_reason:
            raise ProgramBenchConfigurationError("branch_ignore_reason_invalid")
        tests = _string_tuple(branch.get("tests"), "branch_tests")
        if not tests:
            raise ProgramBenchConfigurationError("branch_tests_empty")
        ignored_tests = _ignored_test_tuple(branch.get("ignored_tests"))
        if len(set(tests)) != len(tests):
            raise ProgramBenchConfigurationError("duplicate_branch_test")
        if len(set(ignored_tests)) != len(ignored_tests):
            raise ProgramBenchConfigurationError("duplicate_ignored_test")
        if not set(ignored_tests) <= set(tests):
            raise ProgramBenchConfigurationError("ignored_test_not_in_catalog")
        declared_active_test_ids = _string_tuple(
            branch.get("active_test_ids"), "branch_active_test_ids"
        )
        expected_branch_active_test_ids = (
            ()
            if ignored
            else tuple(
                f"{name}/{test}"
                for test in tests
                if test not in frozenset(ignored_tests)
            )
        )
        if declared_active_test_ids != expected_branch_active_test_ids:
            raise ProgramBenchConfigurationError("branch_active_test_catalog_mismatch")
        _verify_declared_file(
            hidden_root,
            blob_path,
            expected_sha256=blob_sha256,
            expected_size=blob_size,
            error_prefix="programbench_branch_blob",
        )
        branches.append(
            _Branch(
                name=name,
                blob_path=blob_path,
                sha256=blob_sha256,
                size=blob_size,
                ignored=ignored,
                ignore_reason=ignore_reason,
                tests=tests,
                ignored_tests=ignored_tests,
                declared_active_test_ids=declared_active_test_ids,
            )
        )
    if len({branch.name for branch in branches}) != len(branches):
        raise ProgramBenchConfigurationError("duplicate_branch_name")
    if len({branch.blob_path for branch in branches}) != len(branches):
        raise ProgramBenchConfigurationError("duplicate_branch_blob_path")

    active_test_ids = _string_tuple(root.get("active_test_ids"), "active_test_ids")
    expected_active_ids = tuple(
        sorted(
            test_id
            for branch in branches
            for test_id in branch.declared_active_test_ids
        )
    )
    if not active_test_ids:
        raise ProgramBenchConfigurationError("active_test_catalog_empty")
    if active_test_ids != expected_active_ids:
        raise ProgramBenchConfigurationError("active_test_catalog_mismatch")

    raw_provenance = _mapping(root.get("provenance"), "provenance")
    _exact_keys(raw_provenance, _PROVENANCE_KEYS, "provenance")
    provenance = {key: _string(raw_provenance, key) for key in _PROVENANCE_KEYS}
    for key in ("programbench_commit", "hf_revision"):
        if _GIT_SHA_RE.fullmatch(provenance[key]) is None:
            raise ProgramBenchConfigurationError(f"invalid_provenance_{key}")
    for key in (
        "task_sha256",
        "tests_sha256",
        "attribution_sha256",
        "license_sha256",
    ):
        if _SHA256_RE.fullmatch(provenance[key]) is None:
            raise ProgramBenchConfigurationError(f"invalid_provenance_{key}")

    return _Config(
        hidden_root=hidden_root,
        config_path=config_path,
        config_sha256=hashlib.sha256(raw).hexdigest(),
        instance_id=instance_id,
        repository=repository,
        commit=commit,
        language=language,
        difficulty=difficulty,
        image_repository=image_repository,
        image_tag=image_tag,
        image_digest=image_digest,
        image_platform=image_platform,
        resources=resources,
        official_eval_clean_hashes=official_eval_clean_hashes,
        reference_sha256=reference_sha256,
        reference_size=reference_size,
        candidate_rejected_hashes=candidate_rejected_hashes,
        branches=tuple(branches),
        active_test_ids=active_test_ids,
        provenance=provenance,
    )


def _validated_hidden_root(spec: Any) -> Path:
    raw_hidden = str(getattr(spec, "hidden_tests_dir", "") or "")
    if not raw_hidden:
        raise ProgramBenchConfigurationError("hidden_tests_dir_missing")
    lexical = Path(raw_hidden).absolute()
    try:
        metadata = lexical.lstat()
        resolved = lexical.resolve(strict=True)
    except OSError as error:
        raise ProgramBenchConfigurationError("hidden_tests_dir_unavailable") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ProgramBenchConfigurationError("hidden_tests_dir_not_real_directory")
    raw_dataset = str(getattr(spec, "dataset_dir", "") or "")
    if raw_dataset:
        try:
            dataset = Path(raw_dataset).resolve(strict=True)
            resolved.relative_to(dataset)
        except (OSError, ValueError) as error:
            raise ProgramBenchConfigurationError(
                "hidden_tests_dir_outside_dataset"
            ) from error
    return resolved


def _reject_duplicate_json_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ProgramBenchConfigurationError("duplicate_json_key")
        result[key] = value
    return result


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProgramBenchConfigurationError(f"{name}_not_object")
    return value


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise ProgramBenchConfigurationError(f"{name}_not_array")
    return value


def _exact_keys(value: Mapping[str, Any], expected: frozenset[str], name: str) -> None:
    observed = frozenset(str(key) for key in value)
    if observed != expected:
        raise ProgramBenchConfigurationError(f"{name}_keys_mismatch")


def _string(value: Mapping[str, Any], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item or "\x00" in item or item != item.strip():
        raise ProgramBenchConfigurationError(f"{key}_not_canonical_string")
    return item


def _equal_string(value: Mapping[str, Any], key: str, expected: str) -> None:
    if _string(value, key) != expected:
        raise ProgramBenchConfigurationError(f"unsupported_{key}")


def _sha256(value: Any, name: str, *, prefixed: bool = False) -> str:
    if not isinstance(value, str):
        raise ProgramBenchConfigurationError(f"{name}_invalid")
    candidate = value
    if prefixed:
        if not candidate.startswith("sha256:"):
            raise ProgramBenchConfigurationError(f"{name}_invalid")
        candidate = candidate.removeprefix("sha256:")
    if _SHA256_RE.fullmatch(candidate) is None:
        raise ProgramBenchConfigurationError(f"{name}_invalid")
    return f"sha256:{candidate}" if prefixed else candidate


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ProgramBenchConfigurationError(f"{name}_invalid")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ProgramBenchConfigurationError(f"{name}_invalid")
    return value


def _string_tuple(value: Any, name: str) -> tuple[str, ...]:
    rows = _sequence(value, name)
    result: list[str] = []
    for row in rows:
        if (
            not isinstance(row, str)
            or not row
            or row != row.strip()
            or "\x00" in row
            or "\n" in row
            or "\r" in row
        ):
            raise ProgramBenchConfigurationError(f"{name}_invalid_item")
        result.append(row)
    if len(set(result)) != len(result):
        raise ProgramBenchConfigurationError(f"{name}_duplicate_item")
    return tuple(result)


def _ignored_test_tuple(value: Any) -> tuple[str, ...]:
    rows = _sequence(value, "ignored_tests")
    normalized: list[str] = []
    for row in rows:
        if isinstance(row, Mapping):
            if frozenset(row) != {"name"}:
                raise ProgramBenchConfigurationError("ignored_test_keys_mismatch")
            row = row.get("name")
        if not isinstance(row, str) or not row or row != row.strip():
            raise ProgramBenchConfigurationError("ignored_test_invalid")
        normalized.append(row)
    return tuple(normalized)


def _relative_posix_path(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value or "\\" in value:
        raise ProgramBenchConfigurationError(f"{name}_invalid")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or ":" in value
        or path.as_posix() != value
    ):
        raise ProgramBenchConfigurationError(f"{name}_invalid")
    return value


def _verify_declared_file(
    root: Path,
    relative: str,
    *,
    expected_sha256: str,
    expected_size: int,
    error_prefix: str,
) -> None:
    path = root.joinpath(*PurePosixPath(relative).parts)
    payload_hash, size = _hash_regular_file(
        root,
        path,
        error_type=ProgramBenchConfigurationError,
        error_prefix=error_prefix,
    )
    if size != expected_size:
        raise ProgramBenchConfigurationError(f"{error_prefix}_size_mismatch")
    if payload_hash != expected_sha256:
        raise ProgramBenchConfigurationError(f"{error_prefix}_hash_mismatch")


def _read_regular_bytes(
    root: Path,
    path: Path,
    *,
    max_bytes: int,
    error_type: type[ProgramBenchConfigurationError],
    error_prefix: str,
) -> bytes:
    with _open_regular_file(
        root, path, error_type=error_type, error_prefix=error_prefix
    ) as handle:
        metadata = os.fstat(handle.fileno())
        if metadata.st_size > max_bytes:
            raise error_type(f"{error_prefix}_too_large")
        payload = handle.read(max_bytes + 1)
        if len(payload) > max_bytes:
            raise error_type(f"{error_prefix}_too_large")
        return payload


@contextmanager
def _open_regular_file(
    root: Path,
    path: Path,
    *,
    error_type: type[Exception],
    error_prefix: str,
) -> Iterator[Any]:
    try:
        resolved_root = root.resolve(strict=True)
        relative = path.absolute().relative_to(root.absolute())
    except (OSError, ValueError) as error:
        raise error_type(f"{error_prefix}_outside_root") from error
    cursor = root.absolute()
    try:
        for component in relative.parts[:-1]:
            cursor /= component
            metadata = cursor.lstat()
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                raise error_type(f"{error_prefix}_unsafe_parent")
        target = root.absolute() / relative
        before = target.lstat()
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
            raise error_type(f"{error_prefix}_not_regular")
        target.resolve(strict=True).relative_to(resolved_root)
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(target, flags)
        after_open = os.fstat(descriptor)
        if not stat.S_ISREG(after_open.st_mode) or (
            before.st_dev,
            before.st_ino,
        ) != (after_open.st_dev, after_open.st_ino):
            os.close(descriptor)
            raise error_type(f"{error_prefix}_changed_during_open")
    except error_type:
        raise
    except OSError as error:
        raise error_type(f"{error_prefix}_unavailable") from error
    handle = os.fdopen(descriptor, "rb", closefd=True)
    try:
        yield handle
        final = os.fstat(handle.fileno())
        if (final.st_size, final.st_mtime_ns) != (
            after_open.st_size,
            after_open.st_mtime_ns,
        ):
            raise error_type(f"{error_prefix}_changed_during_read")
    finally:
        handle.close()


def _hash_regular_file(
    root: Path,
    path: Path,
    *,
    error_type: type[Exception],
    error_prefix: str,
) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with _open_regular_file(
        root, path, error_type=error_type, error_prefix=error_prefix
    ) as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _walk_tree(
    root: Path,
    *,
    error_type: type[Exception],
    error_prefix: str,
    byte_limit: int | None,
    entry_limit: int | None,
) -> tuple[_TreeEntry, ...]:
    lexical_root = root.absolute()
    try:
        metadata = lexical_root.lstat()
        lexical_root.resolve(strict=True)
    except OSError as error:
        raise error_type(f"{error_prefix}_unavailable") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise error_type(f"{error_prefix}_not_real_directory")
    entries: list[_TreeEntry] = []
    total_bytes = 0

    def visit(directory: Path, relative_parent: PurePosixPath) -> None:
        nonlocal total_bytes
        try:
            children = sorted(os.scandir(directory), key=lambda item: item.name)
        except OSError as error:
            raise error_type(f"{error_prefix}_unreadable_directory") from error
        for child in children:
            child_path = Path(child.path)
            relative = (relative_parent / child.name).as_posix()
            try:
                child_metadata = child_path.lstat()
            except OSError as error:
                raise error_type(f"{error_prefix}_unreadable_entry") from error
            if stat.S_ISLNK(child_metadata.st_mode):
                raise error_type(f"{error_prefix}_symlink_forbidden")
            if stat.S_ISDIR(child_metadata.st_mode):
                entries.append(
                    _TreeEntry(
                        relative=relative,
                        path=child_path,
                        kind="directory",
                        mode=child_metadata.st_mode,
                    )
                )
                visit(child_path, relative_parent / child.name)
            elif stat.S_ISREG(child_metadata.st_mode):
                digest, size = _hash_regular_file(
                    lexical_root,
                    child_path,
                    error_type=error_type,
                    error_prefix=error_prefix,
                )
                total_bytes += size
                entries.append(
                    _TreeEntry(
                        relative=relative,
                        path=child_path,
                        kind="file",
                        mode=child_metadata.st_mode,
                        size=size,
                        sha256=digest,
                    )
                )
            else:
                raise error_type(f"{error_prefix}_special_file_forbidden")
            if entry_limit is not None and len(entries) > entry_limit:
                raise error_type(f"{error_prefix}_entry_limit_exceeded")
            if byte_limit is not None and total_bytes > byte_limit:
                raise error_type(f"{error_prefix}_byte_limit_exceeded")

    visit(lexical_root, PurePosixPath())
    return tuple(entries)


class _HashingReader:
    def __init__(self, handle: Any):
        self._handle = handle
        self.digest = hashlib.sha256()
        self.size = 0

    def read(self, size: int = -1) -> bytes:
        payload = self._handle.read(size)
        self.digest.update(payload)
        self.size += len(payload)
        return payload


def _write_submission_archive(
    root: Path,
    entries: tuple[_TreeEntry, ...],
    destination: Path,
) -> None:
    with destination.open("wb") as raw_output:
        with gzip.GzipFile(
            filename="", mode="wb", fileobj=raw_output, mtime=0
        ) as zipped:
            with tarfile.open(
                fileobj=zipped, mode="w", format=tarfile.PAX_FORMAT
            ) as archive:
                for entry in entries:
                    info = tarfile.TarInfo(entry.relative)
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    info.mtime = 0
                    info.mode = stat.S_IMODE(entry.mode)
                    if entry.kind == "directory":
                        info.type = tarfile.DIRTYPE
                        archive.addfile(info)
                        continue
                    info.size = entry.size
                    with _open_regular_file(
                        root,
                        entry.path,
                        error_type=_RepositoryRejected,
                        error_prefix="programbench_repository",
                    ) as handle:
                        reader = _HashingReader(handle)
                        archive.addfile(info, reader)
                        if (
                            reader.size != entry.size
                            or reader.digest.hexdigest() != entry.sha256
                        ):
                            raise _RepositoryRejected(
                                "programbench_repository_changed_during_pack"
                            )
    final_entries = _walk_tree(
        root,
        error_type=_RepositoryRejected,
        error_prefix="programbench_repository",
        byte_limit=_MAX_REPOSITORY_BYTES,
        entry_limit=_MAX_REPOSITORY_ENTRIES,
    )
    if _tree_identity(final_entries) != _tree_identity(entries):
        raise _RepositoryRejected("programbench_repository_changed_during_pack")


def _tree_identity(entries: tuple[_TreeEntry, ...]) -> tuple[tuple[Any, ...], ...]:
    return tuple(
        (entry.relative, entry.kind, entry.mode, entry.size, entry.sha256)
        for entry in entries
    )


def _stage_branch_blobs(config: _Config, blob_root: Path) -> None:
    target = blob_root / "tests"
    target.mkdir(mode=0o700, parents=True)
    for branch in config.active_branches:
        source = config.hidden_root.joinpath(*PurePosixPath(branch.blob_path).parts)
        destination = target / f"{branch.name}.tar.gz"
        with _open_regular_file(
            config.hidden_root,
            source,
            error_type=ProgramBenchConfigurationError,
            error_prefix="programbench_branch_blob",
        ) as input_handle:
            with destination.open("xb") as output_handle:
                shutil.copyfileobj(input_handle, output_handle, length=1024 * 1024)
        copied_hash, copied_size = _hash_regular_file(
            target,
            destination,
            error_type=ProgramBenchConfigurationError,
            error_prefix="programbench_staged_branch_blob",
        )
        if copied_hash != branch.sha256 or copied_size != branch.size:
            raise ProgramBenchConfigurationError("branch_blob_changed_during_stage")


def _load_official_api() -> _OfficialApi:
    try:
        version = importlib.metadata.version("programbench")
    except importlib.metadata.PackageNotFoundError as error:
        raise ProgramBenchDependencyError("programbench_dependency_missing") from error
    if version != _PROGRAMBENCH_VERSION:
        raise ProgramBenchDependencyError(
            f"programbench_version_mismatch:expected_{_PROGRAMBENCH_VERSION}:got_{version}"
        )
    try:
        package = importlib.import_module("programbench")
        eval_module = importlib.import_module("programbench.eval.eval")
        constants_module = importlib.import_module("programbench.constants")
        exceptions_module = importlib.import_module("programbench.exceptions")
        package_root = Path(package.__file__).resolve(strict=True).parent
        evaluator_cls = eval_module.Evaluator
        eval_step_error_cls = exceptions_module.EvalStepError
    except (AttributeError, ImportError, OSError) as error:
        raise ProgramBenchDependencyError(
            "programbench_official_api_unavailable"
        ) from error
    return _OfficialApi(
        version=version,
        evaluator_cls=evaluator_cls,
        eval_step_error_cls=eval_step_error_cls,
        eval_module=eval_module,
        constants_module=constants_module,
        package_root=package_root,
    )


def _official_file_hashes(api: _OfficialApi) -> Mapping[str, str]:
    """Verify every file from the locked 1.2.4 wheel, not just its version.

    ``importlib.metadata.version`` alone accepts a locally modified install.
    RECORD alone is also mutable, so the adapter pins the wheel digest and a
    compact fingerprint of all 435 ``programbench/`` RECORD rows.  Each row is
    then checked against the installed bytes.  Installer-specific console
    launchers and dist-info bookkeeping are deliberately excluded from the
    cross-platform package fingerprint.
    """

    try:
        distribution = importlib.metadata.distribution("programbench")
        record = distribution.read_text("RECORD")
        distribution_root = Path(distribution.locate_file("")).resolve(strict=True)
        installed_package_root = Path(distribution.locate_file("programbench")).resolve(
            strict=True
        )
    except (importlib.metadata.PackageNotFoundError, OSError) as error:
        raise ProgramBenchDependencyError(
            "programbench_official_distribution_unavailable"
        ) from error
    if distribution.version != _PROGRAMBENCH_VERSION or record is None:
        raise ProgramBenchDependencyError(
            "programbench_official_distribution_record_unavailable"
        )
    if installed_package_root != api.package_root:
        raise ProgramBenchDependencyError(
            "programbench_import_does_not_match_locked_distribution"
        )

    locked_rows: list[dict[str, Any]] = []
    record_entries: dict[str, tuple[str, int]] = {}
    try:
        rows = csv.reader(record.splitlines())
        for raw_path, raw_hash, raw_size in rows:
            if not raw_path.startswith("programbench/"):
                continue
            relative = raw_path.removeprefix("programbench/")
            posix = PurePosixPath(relative)
            if (
                not relative
                or posix.is_absolute()
                or posix.as_posix() != relative
                or any(part in {"", ".", ".."} for part in posix.parts)
                or relative in record_entries
                or not raw_hash.startswith("sha256=")
            ):
                raise ValueError
            size = int(raw_size)
            encoded_digest = raw_hash.removeprefix("sha256=")
            digest_bytes = base64.urlsafe_b64decode(
                encoded_digest + "=" * (-len(encoded_digest) % 4)
            )
            if size < 0 or len(digest_bytes) != hashlib.sha256().digest_size:
                raise ValueError
            digest = digest_bytes.hex()
            record_entries[relative] = (digest, size)
            locked_rows.append({"path": relative, "sha256": digest, "size": size})
    except (TypeError, ValueError) as error:
        raise ProgramBenchDependencyError(
            "programbench_official_distribution_record_invalid"
        ) from error

    locked_rows.sort(key=lambda row: row["path"])
    inventory_digest = _stable_hash(locked_rows)
    total_bytes = sum(int(row["size"]) for row in locked_rows)
    if (
        len(locked_rows) != _PROGRAMBENCH_PACKAGE_FILE_COUNT
        or total_bytes != _PROGRAMBENCH_PACKAGE_TOTAL_BYTES
        or inventory_digest != _PROGRAMBENCH_PACKAGE_INVENTORY_SHA256
    ):
        raise ProgramBenchDependencyError(
            "programbench_official_distribution_inventory_mismatch"
        )

    result: dict[str, str] = {}
    for relative, (record_digest, record_size) in sorted(record_entries.items()):
        path = api.package_root.joinpath(*PurePosixPath(relative).parts)
        try:
            digest, size = _hash_regular_file(
                distribution_root,
                path,
                error_type=ProgramBenchDependencyError,
                error_prefix="programbench_official_source",
            )
        except ProgramBenchDependencyError as error:
            raise ProgramBenchDependencyError(
                f"programbench_official_source_unavailable:{relative}"
            ) from error
        if digest != record_digest or size != record_size:
            raise ProgramBenchDependencyError(
                f"programbench_official_source_hash_mismatch:{relative}"
            )
        result[relative] = digest

    expected_paths = frozenset(record_entries)
    for directory, directory_names, file_names in os.walk(
        api.package_root, followlinks=False
    ):
        directory_path = Path(directory)
        for name in tuple(directory_names):
            child = directory_path / name
            try:
                metadata = child.lstat()
            except OSError as error:
                raise ProgramBenchDependencyError(
                    "programbench_official_package_tree_unavailable"
                ) from error
            if stat.S_ISLNK(metadata.st_mode):
                raise ProgramBenchDependencyError(
                    "programbench_official_package_tree_symlink"
                )
        for name in file_names:
            path = directory_path / name
            relative = path.relative_to(api.package_root).as_posix()
            if "__pycache__" in PurePosixPath(relative).parts:
                continue
            if relative not in expected_paths:
                raise ProgramBenchDependencyError(
                    f"programbench_official_package_unlisted_file:{relative}"
                )
    return result


def _trusted_test_dependency_files() -> tuple[_TrustedDependencyFile, ...]:
    """Inventory the pinned offline pytest plugin injected before submission.

    ProgramBench branch archives install ``pytest-timeout`` at runtime and use
    its ``--timeout`` flag.  Resolving that dependency from the network would
    expose hidden-test containers to the internet and make the evaluator drift.
    The adapter instead freezes the locally locked distribution bytes into the
    runtime identity and copies those exact bytes into the pristine cleanroom
    before the candidate archive is unpacked.
    """

    try:
        distribution = importlib.metadata.distribution("pytest-timeout")
    except importlib.metadata.PackageNotFoundError as error:
        raise ProgramBenchDependencyError(
            "programbench_pytest_timeout_dependency_missing"
        ) from error
    if distribution.version != _PYTEST_TIMEOUT_VERSION:
        raise ProgramBenchDependencyError(
            "programbench_pytest_timeout_version_mismatch:"
            f"expected_{_PYTEST_TIMEOUT_VERSION}:got_{distribution.version}"
        )
    raw_files = distribution.files
    if not raw_files:
        raise ProgramBenchDependencyError(
            "programbench_pytest_timeout_file_inventory_missing"
        )
    try:
        root = Path(distribution.locate_file("")).resolve(strict=True)
    except OSError as error:
        raise ProgramBenchDependencyError(
            "programbench_pytest_timeout_root_unavailable"
        ) from error
    entries: list[_TrustedDependencyFile] = []
    for package_path in sorted(raw_files, key=lambda value: str(value)):
        relative = str(package_path).replace("\\", "/")
        posix = PurePosixPath(relative)
        if (
            not relative
            or posix.is_absolute()
            or posix.as_posix() != relative
            or any(part in {"", ".", ".."} for part in posix.parts)
            or ":" in relative
        ):
            raise ProgramBenchDependencyError(
                "programbench_pytest_timeout_unsafe_file_path"
            )
        try:
            path = Path(distribution.locate_file(package_path)).resolve(strict=True)
            path.relative_to(root)
        except (OSError, ValueError) as error:
            raise ProgramBenchDependencyError(
                "programbench_pytest_timeout_file_outside_distribution"
            ) from error
        digest, size = _hash_regular_file(
            root,
            path,
            error_type=ProgramBenchDependencyError,
            error_prefix="programbench_pytest_timeout_file",
        )
        entries.append(
            _TrustedDependencyFile(
                root=root,
                path=path,
                relative=relative,
                sha256=digest,
                size=size,
            )
        )
    required = {
        "pytest_timeout.py",
        f"pytest_timeout-{_PYTEST_TIMEOUT_VERSION}.dist-info/METADATA",
        f"pytest_timeout-{_PYTEST_TIMEOUT_VERSION}.dist-info/entry_points.txt",
    }
    if not required <= {entry.relative for entry in entries}:
        raise ProgramBenchDependencyError(
            "programbench_pytest_timeout_required_files_missing"
        )
    return tuple(entries)


def _write_trusted_dependency_archive(
    entries: tuple[_TrustedDependencyFile, ...], destination: Path
) -> str:
    digest = hashlib.sha256()
    with destination.open("xb") as raw_output:
        with tarfile.open(
            fileobj=raw_output, mode="w", format=tarfile.PAX_FORMAT
        ) as archive:
            for entry in entries:
                info = tarfile.TarInfo(entry.relative)
                info.uid = 0
                info.gid = 0
                info.uname = ""
                info.gname = ""
                info.mtime = 0
                info.mode = 0o644
                info.size = entry.size
                with _open_regular_file(
                    entry.root,
                    entry.path,
                    error_type=ProgramBenchDependencyError,
                    error_prefix="programbench_pytest_timeout_file",
                ) as handle:
                    reader = _HashingReader(handle)
                    archive.addfile(info, reader)
                    if (
                        reader.size != entry.size
                        or reader.digest.hexdigest() != entry.sha256
                    ):
                        raise ProgramBenchDependencyError(
                            "programbench_pytest_timeout_file_changed_during_pack"
                        )
    with destination.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_candidate_launcher_archive(destination: Path) -> None:
    with destination.open("xb") as raw_output:
        with tarfile.open(
            fileobj=raw_output, mode="w", format=tarfile.PAX_FORMAT
        ) as archive:
            info = tarfile.TarInfo("executable")
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.mtime = 0
            info.mode = 0o755
            info.size = len(_CANDIDATE_LAUNCHER)
            archive.addfile(info, fileobj=io.BytesIO(_CANDIDATE_LAUNCHER))


def _execute_container_utf8(
    env: Any, command: str, *, timeout: int | None = None
) -> Mapping[str, Any]:
    """Run the official docker-exec shape with deterministic host decoding.

    ProgramBench 1.2.4 uses ``text=True`` without an encoding.  On Windows that
    selects the active ANSI code page (GBK on the Gate 0 host), so arbitrary
    candidate output can crash the subprocess reader thread and surface as a
    ``TypeError``.  UTF-8 with replacement keeps stdout diagnostic-only while
    preserving the official return code and in-container timeout cleanup.
    """

    effective_timeout = timeout or env.default_timeout
    argv = [
        env.executable,
        "exec",
        "-w",
        env.cwd,
        env.container_id,
        "bash",
        "-lc",
        command,
    ]
    try:
        completed = subprocess.run(
            argv,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=effective_timeout,
        )
        return {
            "output": (completed.stdout or "") + (completed.stderr or ""),
            "returncode": completed.returncode,
            "exception_info": "",
        }
    except subprocess.TimeoutExpired:
        try:
            subprocess.run(
                [
                    env.executable,
                    "exec",
                    env.container_id,
                    "bash",
                    "-c",
                    "kill -KILL -1 2>/dev/null; sleep 0.2; "
                    "kill -KILL -1 2>/dev/null; true",
                ],
                capture_output=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
            )
        except Exception:
            pass
        return {
            "output": "",
            "returncode": -1,
            "exception_info": f"Command timed out after {effective_timeout}s",
        }


def _docker_control(env: Any, action: str) -> None:
    try:
        completed = subprocess.run(
            [env.executable, action, env.container_id],
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise _TrustedAttestationError(
            f"programbench_trusted_container_{action}_failed"
        ) from error
    if completed.returncode != 0:
        raise _TrustedAttestationError(
            f"programbench_trusted_container_{action}_failed"
        )


@contextmanager
def _frozen_container(env: Any, *, keep_paused: bool) -> Iterator[None]:
    """Freeze every candidate process while the host attests container bytes."""

    _docker_control(env, "pause")
    body_failed = False
    try:
        yield
    except BaseException:
        body_failed = True
        raise
    finally:
        if not keep_paused:
            try:
                _docker_control(env, "unpause")
            except _TrustedAttestationError:
                if not body_failed:
                    raise


def _capture_container_archive(
    env: Any,
    container_path: str,
    destination: Path,
    *,
    archive_byte_limit: int,
    timeout: int,
) -> None:
    """Capture ``docker cp CONTAINER:PATH -`` without unbounded RAM or disk.

    Docker's daemon, rather than any tool inside the candidate filesystem,
    creates the tar stream.  Reader threads cap stdout before writing it and
    continuously drain bounded stderr so a hostile tree cannot deadlock the
    host process or fill the evaluation disk indefinitely.
    """

    command = [
        env.executable,
        "cp",
        f"{env.container_id}:{container_path}",
        "-",
    ]
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as error:
        raise _TrustedAttestationError(
            "programbench_trusted_container_copy_failed"
        ) from error
    if process.stdout is None or process.stderr is None:
        process.kill()
        raise _TrustedAttestationError("programbench_trusted_container_copy_failed")

    overflow = threading.Event()
    reader_errors: list[BaseException] = []
    stderr_chunks: list[bytes] = []

    def read_stdout() -> None:
        total = 0
        try:
            with destination.open("xb") as output:
                while chunk := process.stdout.read(1024 * 1024):
                    total += len(chunk)
                    if total > archive_byte_limit:
                        overflow.set()
                        process.kill()
                        return
                    output.write(chunk)
        except BaseException as error:
            reader_errors.append(error)
            try:
                process.kill()
            except OSError:
                pass

    def read_stderr() -> None:
        retained = 0
        try:
            while chunk := process.stderr.read(64 * 1024):
                if retained < _MAX_DOCKER_ERROR_BYTES:
                    kept = chunk[: _MAX_DOCKER_ERROR_BYTES - retained]
                    stderr_chunks.append(kept)
                    retained += len(kept)
        except BaseException as error:
            reader_errors.append(error)
            try:
                process.kill()
            except OSError:
                pass

    stdout_thread = threading.Thread(target=read_stdout, daemon=True)
    stderr_thread = threading.Thread(target=read_stderr, daemon=True)
    stdout_thread.start()
    stderr_thread.start()
    timed_out = False
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        process.kill()
        process.wait(timeout=30)
    stdout_thread.join(timeout=30)
    stderr_thread.join(timeout=30)
    if stdout_thread.is_alive() or stderr_thread.is_alive():
        try:
            process.kill()
        except OSError:
            pass
        raise _TrustedAttestationError("programbench_trusted_container_copy_failed")
    if timed_out:
        raise _TrustedAttestationError("programbench_trusted_container_copy_timeout")
    if overflow.is_set():
        raise _TrustedAttestationError(
            "programbench_trusted_container_archive_byte_limit_exceeded"
        )
    if reader_errors or process.returncode != 0:
        raise _TrustedAttestationError("programbench_trusted_container_copy_failed")


def _hash_container_archive(
    archive_path: Path,
    *,
    byte_limit: int,
    entry_limit: int,
) -> Mapping[str, tuple[str, int]]:
    """Hash regular members of a Docker-created archive without extracting it."""

    hashes: dict[str, tuple[str, int]] = {}
    total_bytes = 0
    entry_count = 0
    try:
        with tarfile.open(archive_path, mode="r:*") as archive:
            for member in archive:
                entry_count += 1
                if entry_count > entry_limit:
                    raise _TrustedAttestationError(
                        "programbench_trusted_container_entry_limit_exceeded"
                    )
                if member.name in {".", "./"} and member.isdir():
                    continue
                relative = member.name.removeprefix("./")
                posix = PurePosixPath(relative)
                if (
                    not relative
                    or posix.is_absolute()
                    or posix.as_posix() != relative
                    or any(part in {"", ".", ".."} for part in posix.parts)
                ):
                    raise _TrustedAttestationError(
                        "programbench_trusted_container_unsafe_archive_path"
                    )
                if member.isdir():
                    continue
                if not member.isfile() or relative in hashes:
                    raise _TrustedAttestationError(
                        "programbench_trusted_container_unsafe_archive_entry"
                    )
                total_bytes += member.size
                if member.size < 0 or total_bytes > byte_limit:
                    raise _TrustedAttestationError(
                        "programbench_trusted_container_byte_limit_exceeded"
                    )
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise _TrustedAttestationError(
                        "programbench_trusted_container_archive_read_failed"
                    )
                digest = hashlib.sha256()
                observed_size = 0
                while chunk := extracted.read(1024 * 1024):
                    digest.update(chunk)
                    observed_size += len(chunk)
                    if observed_size > member.size:
                        raise _TrustedAttestationError(
                            "programbench_trusted_container_archive_size_mismatch"
                        )
                if observed_size != member.size:
                    raise _TrustedAttestationError(
                        "programbench_trusted_container_archive_size_mismatch"
                    )
                hashes[relative] = (digest.hexdigest(), observed_size)
    except _TrustedAttestationError:
        raise
    except (OSError, tarfile.TarError) as error:
        raise _TrustedAttestationError(
            "programbench_trusted_container_archive_read_failed"
        ) from error
    return hashes


def _trusted_container_archive_hashes(
    env: Any,
    container_path: str,
    *,
    byte_limit: int,
    entry_limit: int,
    archive_byte_limit: int,
    archive_destination: Path | None = None,
) -> Mapping[str, tuple[str, int]]:
    if archive_destination is not None:
        _capture_container_archive(
            env,
            container_path,
            archive_destination,
            archive_byte_limit=archive_byte_limit,
            timeout=_ATTESTATION_COPY_TIMEOUT_SECONDS,
        )
        return _hash_container_archive(
            archive_destination,
            byte_limit=byte_limit,
            entry_limit=entry_limit,
        )
    with tempfile.TemporaryDirectory(prefix="programbench_attestation_") as raw_tmp:
        archive_path = Path(raw_tmp) / "container.tar"
        _capture_container_archive(
            env,
            container_path,
            archive_path,
            archive_byte_limit=archive_byte_limit,
            timeout=_ATTESTATION_COPY_TIMEOUT_SECONDS,
        )
        return _hash_container_archive(
            archive_path,
            byte_limit=byte_limit,
            entry_limit=entry_limit,
        )


def _trusted_container_regular_file_hash(
    env: Any,
    container_path: str,
    *,
    byte_limit: int,
    archive_destination: Path | None = None,
) -> tuple[str, int]:
    hashes = _trusted_container_archive_hashes(
        env,
        container_path,
        byte_limit=byte_limit,
        entry_limit=1,
        archive_byte_limit=byte_limit + 1024 * 1024,
        archive_destination=archive_destination,
    )
    if len(hashes) != 1:
        raise _TrustedAttestationError(
            "programbench_trusted_container_expected_regular_file"
        )
    return next(iter(hashes.values()))


def _trusted_post_compile_attestation(
    env: Any,
    *,
    purelib: str,
    trusted_dependency_files: tuple[_TrustedDependencyFile, ...],
    scan_workspace: bool,
    workspace_archive: Path | None = None,
    executable_archive: Path | None = None,
) -> _PostCompileAttestation:
    if trusted_dependency_files and (
        not purelib.startswith("/")
        or "\x00" in purelib
        or "\n" in purelib
        or "\r" in purelib
    ):
        raise _TrustedAttestationError(
            "programbench_trusted_dependency_path_unavailable"
        )

    with _frozen_container(env, keep_paused=True):
        executable_sha256, executable_size = _trusted_container_regular_file_hash(
            env,
            "/opt/programbench-stashed-executable-do-not-modify",
            byte_limit=_MAX_EXECUTABLE_BYTES,
            archive_destination=executable_archive,
        )
        workspace_hashes: frozenset[str] = frozenset()
        workspace_identity: tuple[tuple[str, str, int], ...] = ()
        if scan_workspace:
            workspace = _trusted_container_archive_hashes(
                env,
                "/workspace/.",
                byte_limit=_MAX_REPOSITORY_BYTES,
                entry_limit=_MAX_REPOSITORY_ENTRIES,
                archive_byte_limit=_MAX_ATTESTATION_ARCHIVE_BYTES,
                archive_destination=workspace_archive,
            )
            workspace_hashes = frozenset(digest for digest, _size in workspace.values())
            workspace_identity = tuple(
                (path, digest, size)
                for path, (digest, size) in sorted(workspace.items())
            )

        for entry in trusted_dependency_files:
            observed_digest, observed_size = _trusted_container_regular_file_hash(
                env,
                f"{purelib.rstrip('/')}/{entry.relative}",
                byte_limit=min(entry.size, _MAX_TRUSTED_DEPENDENCY_FILE_BYTES),
            )
            if observed_digest != entry.sha256 or observed_size != entry.size:
                raise _TrustedAttestationError("trusted_test_dependency_tampered")

    return _PostCompileAttestation(
        executable_sha256=executable_sha256,
        executable_size=executable_size,
        workspace_hashes=workspace_hashes,
        workspace_identity=workspace_identity,
    )


def _trusted_restore_candidate_executable(
    env: Any, launcher_archive: Path
) -> tuple[str, int]:
    for argv in (
        [
            "/bin/rm",
            "-f",
            "--",
            "/candidate/executable",
            "/workspace/executable",
        ],
        [
            "/bin/mv",
            "--",
            "/opt/programbench-stashed-executable-do-not-modify",
            "/candidate/executable",
        ],
        ["/bin/chmod", "0755", "--", "/candidate/executable"],
    ):
        try:
            completed = subprocess.run(
                [
                    env.executable,
                    "exec",
                    "-w",
                    "/",
                    env.container_id,
                    *argv,
                ],
                capture_output=True,
                encoding="utf-8",
                errors="replace",
                timeout=300,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise _TrustedAttestationError(
                "programbench_trusted_executable_restore_failed"
            ) from error
        if completed.returncode != 0:
            raise _TrustedAttestationError(
                "programbench_trusted_executable_restore_failed"
            )

    _trusted_copy_archive_into_container(env, launcher_archive, "/workspace/")
    with _frozen_container(env, keep_paused=False):
        candidate_hash = _trusted_container_regular_file_hash(
            env,
            "/candidate/executable",
            byte_limit=_MAX_EXECUTABLE_BYTES,
        )
        launcher_hash = _trusted_container_regular_file_hash(
            env,
            "/workspace/executable",
            byte_limit=len(_CANDIDATE_LAUNCHER),
        )
        if launcher_hash != (
            _CANDIDATE_LAUNCHER_SHA256,
            len(_CANDIDATE_LAUNCHER),
        ):
            raise _TrustedAttestationError("programbench_candidate_launcher_mismatch")
        return candidate_hash


def _trusted_copy_archive_into_container(
    env: Any, archive_path: Path, destination: str
) -> None:
    """Ask the host Docker daemon to extract a validated archive.

    This avoids the official ``copy_in_tar`` implementation's in-container
    ``tar`` process.  The hardened container intentionally lacks CHOWN and
    FOWNER capabilities, so replaying a Docker-authored workspace archive via
    GNU tar cannot restore its mixed root/agent metadata.  Docker's archive
    endpoint imports the same validated bytes without granting those broad
    capabilities or trusting candidate rootfs tools.
    """

    try:
        with archive_path.open("rb") as source:
            completed = subprocess.run(
                [
                    env.executable,
                    "cp",
                    "-",
                    f"{env.container_id}:{destination}",
                ],
                stdin=source,
                capture_output=True,
                timeout=_ATTESTATION_COPY_TIMEOUT_SECONDS,
            )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise _TrustedAttestationError(
            "programbench_pristine_archive_import_failed"
        ) from error
    if completed.returncode != 0:
        raise _TrustedAttestationError("programbench_pristine_archive_import_failed")


def _trusted_reset_pristine_workspace(env: Any) -> None:
    for argv in (
        ["/bin/rm", "-rf", "--", "/workspace", "/candidate"],
        ["/bin/mkdir", "-p", "--", "/workspace", "/candidate"],
    ):
        try:
            completed = subprocess.run(
                [
                    env.executable,
                    "exec",
                    "-w",
                    "/",
                    env.container_id,
                    *argv,
                ],
                capture_output=True,
                encoding="utf-8",
                errors="replace",
                timeout=120,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise _TrustedAttestationError(
                "programbench_pristine_workspace_reset_failed"
            ) from error
        if completed.returncode != 0:
            raise _TrustedAttestationError(
                "programbench_pristine_workspace_reset_failed"
            )


def _rehydrate_pristine_compile_container(
    evaluator: Any,
    candidate_env: Any,
    *,
    workspace_archive: Path,
    executable_archive: Path,
    trusted_dependency_archive: Path,
    purelib: str,
    trusted_dependency_files: tuple[_TrustedDependencyFile, ...],
    attestation: _PostCompileAttestation,
) -> Any:
    """Create the committed state from the pinned image plus attested bytes.

    The candidate build container runs as root in ProgramBench's pinned image.
    Committing it would therefore preserve arbitrary changes to ``/bin``, the
    Python runtime, pytest, and shell startup files.  Instead this function
    starts a second pristine container, imports only the host-validated
    workspace, executable, and locked pytest plugin, then freezes and verifies
    that reconstructed state before the official evaluator commits it.
    """

    source_image = getattr(candidate_env, "_programbench_source_image", "")
    if not isinstance(source_image, str) or not source_image:
        raise _TrustedAttestationError("programbench_pristine_source_image_unavailable")
    try:
        pristine_env = evaluator._new_env(source_image)
    except Exception as error:
        raise _TrustedAttestationError(
            "programbench_pristine_container_start_failed"
        ) from error
    try:
        try:
            _trusted_copy_archive_into_container(
                pristine_env, trusted_dependency_archive, purelib
            )
        except Exception as error:
            raise _TrustedAttestationError(
                "programbench_pristine_dependency_import_failed"
            ) from error
        _trusted_reset_pristine_workspace(pristine_env)
        try:
            _trusted_copy_archive_into_container(
                pristine_env, workspace_archive, "/candidate/"
            )
        except Exception as error:
            raise _TrustedAttestationError(
                "programbench_pristine_workspace_import_failed"
            ) from error
        try:
            _trusted_copy_archive_into_container(
                pristine_env, executable_archive, "/opt/"
            )
        except Exception as error:
            raise _TrustedAttestationError(
                "programbench_pristine_executable_import_failed"
            ) from error

        with _frozen_container(pristine_env, keep_paused=True):
            hydrated_workspace = _trusted_container_archive_hashes(
                pristine_env,
                "/candidate/.",
                byte_limit=_MAX_REPOSITORY_BYTES,
                entry_limit=_MAX_REPOSITORY_ENTRIES,
                archive_byte_limit=_MAX_ATTESTATION_ARCHIVE_BYTES,
            )
            hydrated_identity = tuple(
                (path, digest, size)
                for path, (digest, size) in sorted(hydrated_workspace.items())
            )
            if hydrated_identity != attestation.workspace_identity:
                raise _TrustedAttestationError(
                    "programbench_pristine_workspace_mismatch"
                )
            hydrated_hidden_workspace = _trusted_container_archive_hashes(
                pristine_env,
                "/workspace/.",
                byte_limit=0,
                entry_limit=1,
                archive_byte_limit=1024 * 1024,
            )
            if hydrated_hidden_workspace:
                raise _TrustedAttestationError(
                    "programbench_pristine_hidden_workspace_not_empty"
                )
            executable_sha256, executable_size = _trusted_container_regular_file_hash(
                pristine_env,
                "/opt/programbench-stashed-executable-do-not-modify",
                byte_limit=_MAX_EXECUTABLE_BYTES,
            )
            if (
                executable_sha256 != attestation.executable_sha256
                or executable_size != attestation.executable_size
            ):
                raise _TrustedAttestationError(
                    "programbench_pristine_executable_mismatch"
                )
            for entry in trusted_dependency_files:
                observed_digest, observed_size = _trusted_container_regular_file_hash(
                    pristine_env,
                    f"{purelib.rstrip('/')}/{entry.relative}",
                    byte_limit=min(entry.size, _MAX_TRUSTED_DEPENDENCY_FILE_BYTES),
                )
                if observed_digest != entry.sha256 or observed_size != entry.size:
                    raise _TrustedAttestationError(
                        "programbench_pristine_trusted_dependency_mismatch"
                    )
        return pristine_env
    except BaseException:
        pristine_env.cleanup()
        raise


def _swap_in_pristine_compile_container(
    evaluator: Any, candidate_env: Any, pristine_env: Any
) -> None:
    candidate_container_id = candidate_env.container_id
    try:
        removed = subprocess.run(
            [candidate_env.executable, "rm", "-f", candidate_container_id],
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        pristine_env.cleanup()
        raise _TrustedAttestationError(
            "programbench_candidate_container_remove_failed"
        ) from error
    if removed.returncode != 0:
        pristine_env.cleanup()
        raise _TrustedAttestationError("programbench_candidate_container_remove_failed")
    candidate_env.container_id = pristine_env.container_id
    evaluator._programbench_pristine_compile_env = pristine_env


@contextmanager
def _hardened_official_globals(
    api: _OfficialApi, resources: _Resources
) -> Iterator[None]:
    with _PROCESS_GLOBALS_LOCK:
        lists: dict[int, tuple[list[str], list[str]]] = {}
        for module in (api.constants_module, api.eval_module):
            value = getattr(module, "DOCKER_RUN_ARGS", None)
            if not isinstance(value, list):
                raise ProgramBenchDependencyError(
                    "programbench_docker_run_args_api_unavailable"
                )
            lists.setdefault(id(value), (value, list(value)))
        old_eval_executable = getattr(api.eval_module, "DOCKER_EXECUTABLE", None)
        try:
            for value, _old in lists.values():
                value[:] = resources.docker_run_args
            api.eval_module.DOCKER_EXECUTABLE = _DOCKER_EXECUTABLE
            yield
        finally:
            for value, old in lists.values():
                value[:] = old
            api.eval_module.DOCKER_EXECUTABLE = old_eval_executable


def _hardened_evaluator_class(
    api: _OfficialApi,
    *,
    forbidden_hashes: tuple[str, ...],
    timeout: int,
    role: str,
    trusted_dependency_archive: Path | None = None,
    trusted_dependency_archive_sha256: str = "",
    trusted_dependency_files: tuple[_TrustedDependencyFile, ...] = (),
    candidate_launcher_archive: Path | None = None,
) -> type[Any]:
    base = api.evaluator_cls
    eval_step_error = api.eval_step_error_cls
    evaluation_timeout = timeout

    class HardenedOfficialEvaluator(base):  # type: ignore[misc, valid-type]
        _trusted_dependency_purelib = ""

        def _new_env(self, image: str, *, serial_pytest: bool = False) -> Any:
            # Officially ``serial_pytest`` is a recovery lever: _evaluate_branch
            # flips it on only after an xdist worker has already crashed. That
            # leaves the first attempt parallel, and parallel attempts do not
            # agree with each other. On delta the same reference produced 12, 17
            # and 18 active-oracle failures across runs of one unchanged pack,
            # and four of the oracles that fail under `-n auto` pass serially in
            # a correct environment. A qualification verdict that moves between
            # runs is not a verdict, so xdist is pinned to a single worker for
            # every container instead of only on the retry path. The incoming
            # flag is subsumed: serial is now the floor, not the fallback.
            env = super()._new_env(image, serial_pytest=True)
            env._programbench_source_image = image

            def execute(
                command: str, *, timeout: int | None = None
            ) -> Mapping[str, Any]:
                return _execute_container_utf8(env, command, timeout=timeout)

            env.execute = execute
            return env

        def _seed_workspace_git_repository(
            self, env: Any, log_buf: list[dict]
        ) -> None:
            """Give the workspace back the git repository the suite assumes.

            The pinned cleanroom image ships ``/workspace`` as a real git work
            tree, and the official ``seed_git`` compile step recreates one when a
            submission arrives without it.  Neither survives to the test
            container here: ``_rehydrate_pristine_compile_container`` imports the
            compiled tree into ``/candidate`` and then *requires* ``/workspace``
            to be empty, so the suite runs against a workspace holding only the
            candidate launcher and the branch fixtures.

            Suites that shell out to git from the workspace root therefore see
            ``fatal: not a git repository``.  delta's ``eval/tests/conftest.py``
            sets ``cwd`` there, so seven ``test_external_subcommands`` cases
            failed on their own reference, and five ``test_grep`` cases failed
            because ``git grep`` searches tracked content and the fixtures were
            untracked.  Seeding after the fixtures land fixes both.

            This restores parity with the image's own shipped state rather than
            inventing an environment, and it runs inside the evaluation
            container after the candidate's output is frozen, so it cannot reach
            the agent.  Both steps are fatal on failure: a missing or broken git
            has to read as an infrastructure error, never as test failures.
            """

            self._run_step(
                "command -v git >/dev/null",
                env=env,
                log_buf=log_buf,
                step_name="verify_workspace_git_available",
                timeout=30,
            )
            self._run_step(
                _WORKSPACE_GIT_SEED_COMMAND,
                env=env,
                log_buf=log_buf,
                step_name="seed_workspace_git",
                timeout=300,
            )

        def _run_step(self, command: str, **kwargs: Any) -> Any:
            requested = kwargs.get("timeout", 20)
            kwargs["timeout"] = min(int(requested), evaluation_timeout)
            # ``run_tests`` is the official step that launches the suite, and it
            # is reached only once the branch fixtures and the restored
            # executable are both in place -- the one point where seeding the
            # repository sees the same tree the tests will.
            if kwargs.get("step_name") == "run_tests":
                self._seed_workspace_git_repository(
                    kwargs["env"], kwargs["log_buf"]
                )
            return super()._run_step(command, **kwargs)

        def _copy_file_from_container(
            self,
            *,
            env: Any,
            log_buf: list[dict],
            container_path: str,
            step_name: str,
            timeout: int = 60,
        ) -> str:
            """Preserve the official Docker-copy contract without a Windows lock.

            ProgramBench 1.2.4 creates a host temporary file with ``mkstemp``
            but discards the still-open descriptor.  Windows consequently
            rejects ``docker cp`` and then rejects cleanup of the locked file.
            Closing that descriptor is the only behavioral difference here;
            the command, log shape, errors, decoding, and timeout semantics stay
            aligned with the pinned official method.
            """

            descriptor, temporary_name = tempfile.mkstemp(
                suffix=Path(container_path).suffix or ".out"
            )
            os.close(descriptor)
            host_tmp = Path(temporary_name)
            command_parts = [
                env.executable,
                "cp",
                f"{env.container_id}:{container_path}",
                str(host_tmp),
            ]
            command = " ".join(command_parts)
            started = time.monotonic()
            effective_timeout = min(int(timeout), evaluation_timeout)
            try:
                try:
                    copied = subprocess.run(
                        command_parts,
                        capture_output=True,
                        text=True,
                        timeout=effective_timeout,
                    )
                    return_code = copied.returncode
                    detail = (copied.stdout + copied.stderr).strip()
                except subprocess.TimeoutExpired:
                    return_code = -1
                    detail = f"docker cp timed out after {effective_timeout}s"
                wall_time = time.monotonic() - started
                if return_code != 0:
                    log_buf.append(
                        {
                            "step": step_name,
                            "command": command,
                            "wall_time": wall_time,
                            "output": detail,
                            "returncode": return_code,
                            "exception_info": "",
                        }
                    )
                    raise eval_step_error(f"{step_name}_failed", detail)
                contents = host_tmp.read_text(encoding="utf-8", errors="replace")
                log_buf.append(
                    {
                        "step": step_name,
                        "command": command,
                        "wall_time": wall_time,
                        "output": contents,
                        "returncode": 0,
                        "exception_info": "",
                    }
                )
                return contents
            finally:
                host_tmp.unlink(missing_ok=True)

        def _install_trusted_test_dependency(
            self, env: Any, log_buf: list[dict]
        ) -> None:
            if trusted_dependency_archive is None:
                return
            located = self._run_step(
                "python3 -c \"import sysconfig; print(sysconfig.get_path('purelib'))\"",
                env=env,
                log_buf=log_buf,
                step_name="locate_python_purelib",
                timeout=30,
            )
            purelib = str(located.get("output") or "").strip()
            if (
                not purelib.startswith("/")
                or "\x00" in purelib
                or "\n" in purelib
                or "\r" in purelib
            ):
                raise eval_step_error(
                    "locate_python_purelib_failed", "non-canonical purelib path"
                )
            self._trusted_dependency_purelib = purelib
            started = time.monotonic()
            try:
                env.copy_in_tar(trusted_dependency_archive, purelib)
            except Exception as error:
                raise eval_step_error(
                    "trusted_test_dependency_copy_failed", type(error).__name__
                ) from error
            log_buf.append(
                {
                    "step": "trusted_test_dependency_copied",
                    "command": "offline evaluator dependency injection",
                    "wall_time": time.monotonic() - started,
                    "output": trusted_dependency_archive_sha256,
                    "returncode": 0,
                    "exception_info": "",
                }
            )
            self._run_step(
                'python3 -c "import importlib.metadata as m, pytest_timeout; '
                f"assert m.version('pytest-timeout') == '{_PYTEST_TIMEOUT_VERSION}'\"",
                env=env,
                log_buf=log_buf,
                step_name="verify_trusted_test_dependency",
                timeout=30,
            )

        def _compile_executable(self, env: Any, log_buf: list[dict]) -> None:
            self._install_trusted_test_dependency(env, log_buf)
            super()._compile_executable(env, log_buf)
            started = time.monotonic()
            try:
                if trusted_dependency_archive is None:
                    attestation = _trusted_post_compile_attestation(
                        env,
                        purelib=self._trusted_dependency_purelib,
                        trusted_dependency_files=trusted_dependency_files,
                        scan_workspace=role != "reference",
                    )
                else:
                    with tempfile.TemporaryDirectory(
                        prefix="programbench_rehydrate_"
                    ) as raw_tmp:
                        snapshot_root = Path(raw_tmp)
                        workspace_archive = snapshot_root / "workspace.tar"
                        executable_archive = snapshot_root / "executable.tar"
                        attestation = _trusted_post_compile_attestation(
                            env,
                            purelib=self._trusted_dependency_purelib,
                            trusted_dependency_files=trusted_dependency_files,
                            scan_workspace=True,
                            workspace_archive=workspace_archive,
                            executable_archive=executable_archive,
                        )
                        observed_hashes = attestation.workspace_hashes | {
                            attestation.executable_sha256
                        }
                        if role != "reference" and observed_hashes & frozenset(
                            forbidden_hashes
                        ):
                            raise eval_step_error("forbidden_gold_artifact", "")
                        pristine_env = _rehydrate_pristine_compile_container(
                            self,
                            env,
                            workspace_archive=workspace_archive,
                            executable_archive=executable_archive,
                            trusted_dependency_archive=trusted_dependency_archive,
                            purelib=self._trusted_dependency_purelib,
                            trusted_dependency_files=trusted_dependency_files,
                            attestation=attestation,
                        )
                        _swap_in_pristine_compile_container(self, env, pristine_env)
            except _TrustedAttestationError as error:
                raise eval_step_error(error.code, "") from error
            self.result.executable_hash = attestation.executable_sha256
            observed_hashes = attestation.workspace_hashes | {
                attestation.executable_sha256
            }
            if role != "reference" and observed_hashes & frozenset(forbidden_hashes):
                raise eval_step_error("forbidden_gold_artifact", "")
            log_buf.append(
                {
                    "step": "trusted_post_compile_attestation",
                    "command": "host attestation and pristine rehydration",
                    "wall_time": time.monotonic() - started,
                    "output": attestation.executable_sha256,
                    "returncode": 0,
                    "exception_info": "",
                }
            )

        def _restore_executable(self, env: Any, log_buf: list[dict]) -> None:
            if candidate_launcher_archive is None:
                raise eval_step_error("candidate_launcher_unavailable", "")
            started = time.monotonic()
            try:
                observed_hash, _observed_size = _trusted_restore_candidate_executable(
                    env, candidate_launcher_archive
                )
            except _TrustedAttestationError as error:
                raise eval_step_error(error.code, "") from error
            if observed_hash != self.result.executable_hash:
                raise eval_step_error(
                    "executable_hash_mismatch",
                    "trusted host hash differs from compiled executable hash",
                )
            log_buf.append(
                {
                    "step": "trusted_restore_executable",
                    "command": "host-verified /candidate executable and launcher",
                    "wall_time": time.monotonic() - started,
                    "output": observed_hash,
                    "returncode": 0,
                    "exception_info": "",
                }
            )

    HardenedOfficialEvaluator.__name__ = "HardenedOfficialEvaluator"
    return HardenedOfficialEvaluator


def _run_official_evaluator(
    *,
    api: _OfficialApi,
    config: _Config,
    submission_archive: Path,
    blob_root: Path,
    timeout: int,
    role: str,
) -> Any:
    dependency_archive = submission_archive.parent / "pytest-timeout.tar"
    launcher_archive = submission_archive.parent / "candidate-launcher.tar"
    _write_candidate_launcher_archive(launcher_archive)
    trusted_dependency_files = _trusted_test_dependency_files()
    dependency_archive_sha256 = _write_trusted_dependency_archive(
        trusted_dependency_files, dependency_archive
    )
    evaluator_cls = _hardened_evaluator_class(
        api,
        forbidden_hashes=config.forbidden_hashes,
        timeout=timeout,
        role=role,
        trusted_dependency_archive=dependency_archive,
        trusted_dependency_archive_sha256=dependency_archive_sha256,
        trusted_dependency_files=trusted_dependency_files,
        candidate_launcher_archive=launcher_archive,
    )
    tests_by_branch = {
        branch.name: list(branch.tests) for branch in config.active_branches
    }
    ignored_tests = set(config.ignored_test_ids)
    ignored_branches = {branch.name for branch in config.branches if branch.ignored}
    remove_hashes = (
        [] if role == "reference" else list(config.official_eval_clean_hashes)
    )
    with _hardened_official_globals(api, config.resources):
        evaluator = evaluator_cls(
            image_name=config.image_repository,
            solution_branch=role,
            submission_archive=submission_archive,
            blob_dir=blob_root,
            tests_branches=[branch.name for branch in config.active_branches],
            remove_hashes=remove_hashes,
            image_tag=f"{config.image_tag}@{config.image_digest}",
            tests_by_branch=tests_by_branch,
            ignored_tests=ignored_tests,
            ignored_branches=ignored_branches,
            instance_id=config.instance_id,
            docker_cpus=config.resources.docker_cpus,
            branch_workers=config.resources.branch_workers,
            branch_retries=config.resources.branch_retries,
        )
        return evaluator.run()


def _translate_official_result(
    config: _Config, role: str, result: Any
) -> Mapping[str, Any]:
    official_error = _attribute(result, "error_code")
    executable_hash = _attribute(result, "executable_hash")
    raw_result = _sanitized_official_result(result, config.forbidden_hashes)
    trust_boundary_codes = {
        "forbidden_gold_artifact": "programbench_forbidden_artifact_after_compile",
        "trusted_test_dependency_tampered": (
            "programbench_trusted_dependency_tampered_after_compile"
        ),
    }
    if official_error in trust_boundary_codes:
        return _uniform_result(
            config,
            role,
            "blocked",
            trust_boundary_codes[str(official_error)],
            official_result=raw_result,
        )
    if official_error:
        if role != "reference" and official_error in _CANDIDATE_BUILD_FAILURES:
            return _uniform_result(
                config,
                role,
                "failed",
                "programbench_candidate_compile_failed",
                diagnostic_extra={"official_error_code": official_error},
                official_result=raw_result,
            )
        return _uniform_result(
            config,
            role,
            "infra_error",
            "programbench_official_evaluator_error",
            diagnostic_extra={"official_error_code": official_error},
            official_result=raw_result,
        )
    if role == "reference" and executable_hash != config.reference_sha256:
        return _uniform_result(
            config,
            role,
            "infra_error",
            "programbench_reference_executable_hash_mismatch",
            official_result=raw_result,
        )
    if role != "reference" and executable_hash in frozenset(config.forbidden_hashes):
        return _uniform_result(
            config,
            role,
            "blocked",
            "programbench_forbidden_executable_after_compile",
            official_result=raw_result,
        )

    statuses: dict[str, str] = {}
    unexpected_branches: set[str] = set()
    ignored_ids = config.ignored_test_ids
    expected_ids = frozenset(config.active_test_ids)
    raw_tests = _attribute(result, "test_results") or []
    for test in raw_tests:
        branch = str(_attribute(test, "branch") or "")
        name = str(_attribute(test, "name") or "")
        test_id = f"{branch}/{name}" if branch else name
        if test_id in ignored_ids:
            continue
        if test_id not in expected_ids or test_id in statuses:
            unexpected_branches.add(branch)
            continue
        official_status = str(_attribute(test, "status") or "")
        if official_status == "passed":
            statuses[test_id] = "passed"
        elif official_status in {"failure", "error", "skipped"}:
            statuses[test_id] = "failed"
        else:
            statuses[test_id] = "infra_error"

    raw_branch_errors = _attribute(result, "test_branch_errors") or {}
    branch_errors = {
        str(branch)
        for branch, errors in (
            raw_branch_errors.items() if isinstance(raw_branch_errors, Mapping) else ()
        )
        if errors
    }
    branch_errors.update(unexpected_branches)
    for branch in config.active_branches:
        for test in branch.active_tests:
            test_id = f"{branch.name}/{test}"
            if branch.name in branch_errors:
                statuses[test_id] = "infra_error"
            elif test_id not in statuses:
                statuses[test_id] = "infra_error"

    overall = _overall_status(statuses.values())
    diagnostic_code = (
        "programbench_official_evaluation_complete"
        if overall in {"passed", "failed"}
        else "programbench_official_results_incomplete"
    )
    return _result_mapping(
        config,
        role,
        statuses,
        overall,
        diagnostic_code,
        diagnostic_extra={"branch_error_count": len(branch_errors)},
        official_result=raw_result,
    )


def _attribute(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _sanitized_official_result(result: Any, secrets: tuple[str, ...]) -> Any:
    if hasattr(result, "model_dump"):
        raw = result.model_dump(mode="python")
    elif isinstance(result, Mapping):
        raw = dict(result)
    else:
        raw = {
            name: getattr(result, name)
            for name in (
                "test_results",
                "error_code",
                "error_details",
                "log",
                "solution_branch",
                "test_branches",
                "test_branch_errors",
                "executable_hash",
                "warnings",
            )
            if hasattr(result, name)
        }
    return _redact_secrets(raw, secrets)


def _redact_secrets(value: Any, secrets: tuple[str, ...]) -> Any:
    if isinstance(value, str):
        redacted = value
        for secret in secrets:
            redacted = re.sub(
                re.escape(secret), "<redacted-programbench-hash>", redacted, flags=re.I
            )
        return redacted
    if isinstance(value, Mapping):
        return {str(key): _redact_secrets(item, secrets) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return tuple(_redact_secrets(item, secrets) for item in value)
    if hasattr(value, "model_dump"):
        return _redact_secrets(value.model_dump(mode="python"), secrets)
    return value


def _unconfigured_result(role: str, code: str) -> Mapping[str, Any]:
    return {
        "runner": PROGRAMBENCH_RUNNER,
        "role": role,
        "ok": False,
        "status": "infra_error",
        "total": 0,
        "passed": 0,
        "failed": 0,
        "pass_rate": 0.0,
        "by_test": (),
        "diagnostic": {"code": code, "role": role},
        "official_result": None,
    }


def _uniform_result(
    config: _Config,
    role: str,
    status: str,
    code: str,
    *,
    diagnostic_extra: Mapping[str, Any] | None = None,
    official_result: Any = None,
) -> Mapping[str, Any]:
    statuses = {test_id: status for test_id in config.active_test_ids}
    return _result_mapping(
        config,
        role,
        statuses,
        status,
        code,
        diagnostic_extra=diagnostic_extra,
        official_result=official_result,
    )


def _result_mapping(
    config: _Config,
    role: str,
    statuses: Mapping[str, str],
    overall: str,
    code: str,
    *,
    diagnostic_extra: Mapping[str, Any] | None = None,
    official_result: Any = None,
) -> Mapping[str, Any]:
    by_test = tuple(
        {
            "test_id": test_id,
            "issue_ids": (),
            "status": statuses.get(test_id, "infra_error"),
            "passed": statuses.get(test_id) == "passed",
        }
        for test_id in config.active_test_ids
    )
    passed = sum(row["passed"] for row in by_test)
    diagnostic = {"code": code, "role": role, **dict(diagnostic_extra or {})}
    return {
        "runner": PROGRAMBENCH_RUNNER,
        "role": role,
        "ok": overall in {"passed", "failed"},
        "status": overall,
        "total": len(by_test),
        "passed": passed,
        "failed": len(by_test) - passed,
        "pass_rate": passed / len(by_test) if by_test else 0.0,
        "by_test": by_test,
        "diagnostic": diagnostic,
        "official_result": official_result,
    }


def _overall_status(statuses: Sequence[str] | Any) -> str:
    values = tuple(statuses)
    if not values or "infra_error" in values:
        return "infra_error"
    if "blocked" in values:
        return "blocked"
    if "timeout" in values:
        return "timeout"
    if all(value == "passed" for value in values):
        return "passed"
    return "failed"


def _docker_preflight_reasons(image_ref: str) -> tuple[str, ...]:
    if shutil.which(_DOCKER_EXECUTABLE) is None:
        return ("programbench_docker_executable_not_found",)
    try:
        daemon = subprocess.run(
            [_DOCKER_EXECUTABLE, "version", "--format", "{{.Server.Version}}"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ("programbench_docker_daemon_unavailable",)
    if daemon.returncode != 0 or not daemon.stdout.strip():
        return ("programbench_docker_daemon_unavailable",)
    try:
        image = subprocess.run(
            [
                _DOCKER_EXECUTABLE,
                "image",
                "inspect",
                "--format",
                "{{.Os}}/{{.Architecture}}",
                image_ref,
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ("programbench_pinned_image_inspection_failed",)
    if image.returncode != 0:
        return ("programbench_pinned_image_unavailable",)
    if image.stdout.strip() != _PLATFORM:
        return ("programbench_pinned_image_platform_mismatch",)
    return ()


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "PROGRAMBENCH_RUNNER",
    "ProgramBenchConfigurationError",
    "ProgramBenchDependencyError",
    "is_programbench_spec",
    "programbench_formal_blocking_reasons",
    "programbench_hidden_suite_hash",
    "programbench_runtime_identity",
    "run_programbench_evaluation",
]
