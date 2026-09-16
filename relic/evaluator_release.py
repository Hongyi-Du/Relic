"""Public, source-derived local evaluator build and qualification helpers.

The implementation follows the evaluator build/hash/preflight path in
``SocioGenesis/hci-human-seat@dda36fb563375060ae8d8850300db01eb4695d29``.
It intentionally draws a release boundary around that path: local image IDs and
locally computed hashes are diagnostic evidence, never a substitute for the
author-published image digest and qualification bindings required by the paper
runner.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from relic.benchmark import load_benchmark_manifest
from relic.paths import project_root
from relic.research.hashing import stable_hash


SOURCE_REVISION = "hci-human-seat@dda36fb563375060ae8d8850300db01eb4695d29"
BUILD_SCHEMA_VERSION = "relic-local-evaluator-build-v1"
HASH_SCHEMA_VERSION = "relic-local-evaluator-hashes-v1"
PREFLIGHT_SCHEMA_VERSION = "relic-evaluator-preflight-v1"
DEFAULT_LOCAL_IMAGE_TAG = "relic-oss-evaluator:local"
FORMAL_PLATFORM = "linux/amd64"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_LOCAL_IMAGE_ID = re.compile(r"^sha256:[0-9a-f]{64}$")
_REGISTRY_IMAGE_DIGEST = re.compile(r"^.+@sha256:[0-9a-f]{64}$")
_LOCAL_TAG = re.compile(r"^[a-z0-9][a-z0-9._/-]*(?::[a-z0-9][a-z0-9._-]*)?$")


class EvaluatorReleaseError(ValueError):
    """A stable public error code for evaluator release tooling."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def released_pack_ids() -> frozenset[str]:
    """Return exactly the evaluator pack IDs in the frozen public benchmark."""

    manifest = load_benchmark_manifest()
    workloads = manifest.get("workloads")
    if not isinstance(workloads, list):
        raise EvaluatorReleaseError("released_benchmark_manifest_invalid")
    packs = {
        str(item.get("pack") or "").strip()
        for item in workloads
        if isinstance(item, Mapping)
    }
    if not packs or "" in packs:
        raise EvaluatorReleaseError("released_benchmark_manifest_invalid")
    return frozenset(packs)


def require_released_pack(dataset_id: str) -> str:
    """Reject non-release and ProgramBench evaluator inputs before execution."""

    pack = str(dataset_id or "").strip()
    lowered = pack.casefold()
    if "programbench" in lowered:
        raise EvaluatorReleaseError("programbench_evaluator_not_available_in_relic_release")
    if not pack or "/" in pack or "\\" in pack or Path(pack).is_absolute():
        raise EvaluatorReleaseError("evaluator_released_pack_id_required")
    if pack not in released_pack_ids():
        raise EvaluatorReleaseError("evaluator_pack_not_in_relic_main_v1")
    return pack


def _require_platform(platform: str) -> str:
    value = str(platform or "").strip().lower()
    if value != FORMAL_PLATFORM:
        raise EvaluatorReleaseError("evaluator_platform_must_be_linux_amd64")
    return value


def _require_local_tag(tag: str) -> str:
    value = str(tag or "").strip()
    if not _LOCAL_TAG.fullmatch(value) or "@" in value:
        raise EvaluatorReleaseError("evaluator_local_image_tag_invalid")
    return value


def _run_checked(
    argv: Sequence[str],
    *,
    failure_code: str,
    timeout_seconds: float | None = None,
) -> subprocess.CompletedProcess[str]:
    try:
        completed = subprocess.run(
            tuple(argv),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        del exc
        raise EvaluatorReleaseError(failure_code) from None
    if completed.returncode != 0:
        # Build output can contain an arbitrary registry URL or proxy detail.
        # Keep the public CLI and persisted reports free of such values.
        raise EvaluatorReleaseError(failure_code)
    return completed


def _evaluator_dockerfile() -> Path:
    path = project_root() / "evaluator" / "Dockerfile"
    if not path.is_file():
        raise EvaluatorReleaseError("evaluator_build_context_missing")
    return path


def _inspect_local_image(
    *, docker_binary: str, tag: str, platform: str
) -> str:
    completed = _run_checked(
        (
            docker_binary,
            "image",
            "inspect",
            "--format",
            "{{.Id}} {{.Os}}/{{.Architecture}}",
            tag,
        ),
        failure_code="evaluator_local_image_inspection_failed",
        timeout_seconds=15,
    )
    fields = completed.stdout.strip().split()
    if len(fields) != 2 or not _LOCAL_IMAGE_ID.fullmatch(fields[0].lower()):
        raise EvaluatorReleaseError("evaluator_local_image_identity_invalid")
    if fields[1].lower() != platform:
        raise EvaluatorReleaseError("evaluator_local_image_platform_mismatch")
    return fields[0].lower()


def _smoke_local_image(*, docker_binary: str, tag: str, platform: str) -> None:
    """Run import-only source-closure smoke under the formal network boundary."""

    probe = (
        "import annotated_types, attr, bs4, cv2, docx, fastapi, flask, "
        "httpx, hypothesis, immutables, jose, kombu, numpy, openpyxl, "
        "outcome, passlib, pdfplumber, pydantic, pydantic_core, pypdf, "
        "pytest, pytest_asyncio, pytest_mock, reportlab, sniffio, sqlalchemy, "
        "trio, typing_extensions, uvicorn, yarl; print('relic evaluator imports ok')"
    )
    _run_checked(
        (
            docker_binary,
            "run",
            "--rm",
            "--platform",
            platform,
            "--network",
            "none",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--user",
            "65534:65534",
            "--tmpfs",
            "/tmp:rw,nosuid,nodev,noexec,size=64m",
            "-e",
            "HOME=/tmp",
            "-e",
            "PYTHONDONTWRITEBYTECODE=1",
            tag,
            "python",
            "-c",
            probe,
        ),
        failure_code="evaluator_local_image_smoke_failed",
        timeout_seconds=90,
    )


def build_local_evaluator(
    *,
    tag: str = DEFAULT_LOCAL_IMAGE_TAG,
    platform: str = FORMAL_PLATFORM,
    smoke: bool = False,
    docker_binary: str = "docker",
) -> dict[str, Any]:
    """Build the public local image without producing a paper evaluator binding."""

    tag = _require_local_tag(tag)
    platform = _require_platform(platform)
    if shutil.which(docker_binary) is None:
        raise EvaluatorReleaseError("docker_executable_not_found")
    dockerfile = _evaluator_dockerfile()
    _run_checked(
        (
            docker_binary,
            "build",
            "--platform",
            platform,
            "--file",
            str(dockerfile),
            "--tag",
            tag,
            str(dockerfile.parent),
        ),
        failure_code="evaluator_local_image_build_failed",
    )
    image_id = _inspect_local_image(
        docker_binary=docker_binary,
        tag=tag,
        platform=platform,
    )
    if smoke:
        _smoke_local_image(
            docker_binary=docker_binary,
            tag=tag,
            platform=platform,
        )
    return {
        "schema_version": BUILD_SCHEMA_VERSION,
        "status": "passed",
        "source_revision": SOURCE_REVISION,
        "dockerfile_sha256": hashlib.sha256(dockerfile.read_bytes()).hexdigest(),
        "local_image_tag": tag,
        "local_image_id": image_id,
        "platform": platform,
        "smoke": "passed" if smoke else "not_run",
        "paper_binding_status": "author_published_binding_required",
        "paper_binding_note": (
            "A local image ID and local qualification hashes are not a paper "
            "evaluator binding and are rejected by the source main runner."
        ),
    }


def _build_executor(*, backend: str, image: str, platform: str):
    from relic.evaluation.execution import ExecutionPolicy, build_command_executor

    return build_command_executor(
        ExecutionPolicy(
            trust_level="untrusted",
            backend=backend,  # type: ignore[arg-type]
            container_image=image,
            container_platform=platform,
            network_enabled=False,
        )
    )


def _image_identity_kind(image: str) -> str:
    if _LOCAL_IMAGE_ID.fullmatch(image.lower()):
        return "local_image_id"
    if _REGISTRY_IMAGE_DIGEST.fullmatch(image.lower()):
        return "registry_digest"
    return "invalid_or_unpinned"


def _hash_report(*, plan: Any, backend: str, image: str, platform: str) -> dict[str, Any]:
    return {
        "schema_version": HASH_SCHEMA_VERSION,
        "status": "passed" if plan.operational_ready else "failed",
        "source_revision": SOURCE_REVISION,
        "dataset_id": plan.dataset_id,
        "backend": backend,
        "container_platform": platform,
        "container_image_identity_kind": _image_identity_kind(image),
        "operational_ready": plan.operational_ready,
        "formal_runtime_ready": plan.formal_ready,
        "blocking_reasons": list(plan.blocking_reasons),
        "local_evaluator_environment_hash": plan.evaluator_environment_hash,
        "local_qualification_plan_hash": plan.plan_hash,
        "paper_binding_status": "author_published_binding_required",
        "paper_binding_note": (
            "These locally calculated values are diagnostic evidence only; do not "
            "construct a paper binding from them."
        ),
    }


def calculate_local_evaluator_hashes(
    *,
    dataset_id: str,
    backend: str,
    container_image: str,
    container_platform: str = FORMAL_PLATFORM,
    timeout_seconds: int = 900,
) -> dict[str, Any]:
    """Compute HCI-source-compatible plan hashes for one released pack.

    The container qualification is real and can fail, but the result remains a
    local diagnostic until the authors publish a corresponding binding.
    """

    pack = require_released_pack(dataset_id)
    backend = str(backend or "").strip().lower()
    if backend not in {"docker", "apptainer"}:
        raise EvaluatorReleaseError("evaluator_backend_not_supported")
    image = str(container_image or "").strip()
    if not image:
        raise EvaluatorReleaseError("evaluator_container_image_required")
    platform = _require_platform(container_platform)
    if timeout_seconds <= 0:
        raise EvaluatorReleaseError("evaluator_timeout_must_be_positive")
    try:
        executor = _build_executor(
            backend=backend,
            image=image,
            platform=platform,
        )
        from relic.evaluation.time_machine import build_time_machine_evaluation_plan

        plan = build_time_machine_evaluation_plan(
            dataset_id=pack,
            timeout_seconds=timeout_seconds,
            executor=executor,
        )
    except EvaluatorReleaseError:
        raise
    except Exception as exc:
        del exc
        raise EvaluatorReleaseError("evaluator_hash_calculation_failed") from None
    return _hash_report(
        plan=plan,
        backend=backend,
        image=image,
        platform=platform,
    )


def _require_sha256(value: str, *, code: str) -> str:
    normalized = str(value or "").strip().lower()
    if not _SHA256.fullmatch(normalized):
        raise EvaluatorReleaseError(code)
    return normalized


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    target = path.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(rendered)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)
    if os.name != "nt":
        directory = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    return target


def _preflight_payload(
    *,
    repository_id: str,
    dataset_id: str,
    backend: str,
    container_image: str,
    container_platform: str,
    expected_environment_hash: str,
    expected_qualification_hash: str,
    plan: Any | None,
    reasons: Sequence[str],
) -> dict[str, Any]:
    """Build a stable-hashable preflight receipt without persisting image URLs."""

    fields: dict[str, Any] = {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "source_revision": SOURCE_REVISION,
        "repository_id": repository_id,
        "dataset_id": dataset_id,
        "backend": backend,
        "container_platform": container_platform,
        "container_image_identity_sha256": hashlib.sha256(
            container_image.encode("utf-8")
        ).hexdigest(),
        "container_image_identity_kind": _image_identity_kind(container_image),
        "expected_environment_hash": expected_environment_hash,
        "expected_qualification_hash": expected_qualification_hash,
        "observed_environment_hash": (
            plan.evaluator_environment_hash if plan is not None else None
        ),
        "observed_qualification_hash": plan.plan_hash if plan is not None else None,
        "oracle_count": len(plan.oracles) if plan is not None else 0,
        "status": "passed" if not reasons else "failed",
        "blocking_reasons": list(dict.fromkeys(reasons)),
        "paper_binding_status": "author_published_binding_required",
    }
    # Wall-clock time is useful to a local operator but must not make the
    # attestation itself nondeterministic.
    fields["attestation_hash"] = stable_hash(fields)
    fields["checked_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    return fields


def preflight_local_evaluator(
    *,
    repository_id: str,
    dataset_id: str,
    backend: str,
    container_image: str,
    container_platform: str,
    expected_environment_hash: str,
    expected_qualification_hash: str,
    timeout_seconds: int,
    output: Path,
) -> tuple[dict[str, Any], Path]:
    """Requalify one released pack and write a local, non-paper receipt."""

    reasons: list[str] = []
    plan: Any | None = None
    normalized_repository = str(repository_id or "").strip()
    normalized_dataset = str(dataset_id or "").strip()
    normalized_backend = str(backend or "").strip().lower()
    normalized_image = str(container_image or "").strip()
    normalized_platform = str(container_platform or "").strip().lower()
    normalized_environment_hash = str(expected_environment_hash or "").strip().lower()
    normalized_qualification_hash = str(expected_qualification_hash or "").strip().lower()
    try:
        pack = require_released_pack(normalized_dataset)
        if normalized_repository != pack:
            reasons.append("evaluator_preflight_repository_id_mismatch")
        if normalized_backend not in {"docker", "apptainer"}:
            raise EvaluatorReleaseError("evaluator_backend_not_supported")
        _require_platform(normalized_platform)
        _require_sha256(
            normalized_environment_hash,
            code="evaluator_expected_environment_hash_invalid",
        )
        _require_sha256(
            normalized_qualification_hash,
            code="evaluator_expected_qualification_hash_invalid",
        )
        if not normalized_image:
            raise EvaluatorReleaseError("evaluator_container_image_required")
        if timeout_seconds <= 0:
            raise EvaluatorReleaseError("evaluator_timeout_must_be_positive")
        executor = _build_executor(
            backend=normalized_backend,
            image=normalized_image,
            platform=normalized_platform,
        )
        from relic.evaluation.time_machine import build_time_machine_evaluation_plan

        plan = build_time_machine_evaluation_plan(
            dataset_id=pack,
            timeout_seconds=timeout_seconds,
            executor=executor,
        )
        if plan.formal_ready is not True:
            reasons.extend(plan.blocking_reasons)
        if plan.evaluator_environment_hash != normalized_environment_hash:
            reasons.append("evaluator_preflight_environment_hash_mismatch")
        if plan.plan_hash != normalized_qualification_hash:
            reasons.append("evaluator_preflight_qualification_hash_mismatch")
    except EvaluatorReleaseError as exc:
        reasons.append(exc.code)
    except Exception as exc:
        del exc
        reasons.append("evaluator_preflight_failed")
    payload = _preflight_payload(
        repository_id=normalized_repository,
        dataset_id=normalized_dataset,
        backend=normalized_backend,
        container_image=normalized_image,
        container_platform=normalized_platform,
        expected_environment_hash=normalized_environment_hash,
        expected_qualification_hash=normalized_qualification_hash,
        plan=plan,
        reasons=reasons,
    )
    return payload, _atomic_write_json(output, payload)


__all__ = [
    "BUILD_SCHEMA_VERSION",
    "DEFAULT_LOCAL_IMAGE_TAG",
    "EvaluatorReleaseError",
    "FORMAL_PLATFORM",
    "HASH_SCHEMA_VERSION",
    "PREFLIGHT_SCHEMA_VERSION",
    "SOURCE_REVISION",
    "build_local_evaluator",
    "calculate_local_evaluator_hashes",
    "preflight_local_evaluator",
    "released_pack_ids",
    "require_released_pack",
]
