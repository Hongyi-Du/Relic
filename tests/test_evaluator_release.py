from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from relic import evaluator_release
from relic.source_runner import SourceMainRunnerError, load_evaluator_bindings


ROOT = Path(__file__).resolve().parents[1]
_HEX = "a" * 64


@pytest.mark.release
def test_source_derived_evaluator_context_has_no_proxy_or_private_build_input() -> None:
    dockerfile = (ROOT / "evaluator" / "Dockerfile").read_text(encoding="utf-8")
    readme = (ROOT / "evaluator" / "README.md").read_text(encoding="utf-8")

    assert evaluator_release.SOURCE_REVISION in dockerfile
    assert "python@sha256:" in dockerfile
    assert "pytest==8.3.4" in dockerfile
    assert "pydantic-core==2.47.0" in dockerfile
    assert "PIPPROXY" not in dockerfile
    assert "--proxy" not in dockerfile
    assert "COPY " not in dockerfile
    assert "/root/" not in dockerfile
    assert "ProgramBench" not in dockerfile
    assert "author-published" in readme


# Retain a tiny helper instead of coupling tests to an implementation traceback.
def _assert_error_code(callable_, expected: str) -> None:
    with pytest.raises(evaluator_release.EvaluatorReleaseError) as exc_info:
        callable_()
    assert exc_info.value.code == expected


@pytest.mark.unit
def test_evaluator_pack_guard_uses_stable_error_codes() -> None:
    packs = evaluator_release.released_pack_ids()
    assert len(packs) == 10
    assert "mini_blobstore_v1" in packs
    assert evaluator_release.require_released_pack("mini_blobstore_v1") == "mini_blobstore_v1"
    _assert_error_code(
        lambda: evaluator_release.require_released_pack("ProgramBench"),
        "programbench_evaluator_not_available_in_relic_release",
    )
    _assert_error_code(
        lambda: evaluator_release.require_released_pack("/tmp/not-a-release-pack"),
        "evaluator_released_pack_id_required",
    )
    _assert_error_code(
        lambda: evaluator_release.require_released_pack("not-a-pack"),
        "evaluator_pack_not_in_relic_main_v1",
    )


@pytest.mark.unit
def test_local_build_uses_minimal_context_and_never_mints_a_paper_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, ...]] = []

    def fake_run(argv, **_kwargs) -> subprocess.CompletedProcess[str]:
        command = tuple(argv)
        calls.append(command)
        if command[1:3] == ("image", "inspect"):
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=f"sha256:{_HEX} linux/amd64\n",
                stderr="",
            )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(evaluator_release.shutil, "which", lambda _binary: "/usr/bin/docker")
    monkeypatch.setattr(evaluator_release.subprocess, "run", fake_run)

    result = evaluator_release.build_local_evaluator()

    build = calls[0]
    dockerfile = ROOT / "evaluator" / "Dockerfile"
    assert build[:5] == ("docker", "build", "--platform", "linux/amd64", "--file")
    assert build[5] == str(dockerfile)
    assert build[-1] == str(dockerfile.parent)
    assert "--build-arg" not in build
    assert "--network" not in build
    assert result["local_image_id"] == f"sha256:{_HEX}"
    assert result["paper_binding_status"] == "author_published_binding_required"


@pytest.mark.unit
def test_hash_report_and_preflight_attestation_are_deterministic() -> None:
    plan = SimpleNamespace(
        dataset_id="mini_blobstore_v1",
        operational_ready=True,
        formal_ready=True,
        blocking_reasons=(),
        evaluator_environment_hash="b" * 64,
        plan_hash="c" * 64,
        oracles=(object(), object()),
    )
    first_hash_report = evaluator_release._hash_report(  # type: ignore[attr-defined]
        plan=plan,
        backend="docker",
        image=f"sha256:{_HEX}",
        platform="linux/amd64",
    )
    second_hash_report = evaluator_release._hash_report(  # type: ignore[attr-defined]
        plan=plan,
        backend="docker",
        image=f"sha256:{_HEX}",
        platform="linux/amd64",
    )
    assert first_hash_report == second_hash_report

    image = f"private.example/hidden@sha256:{_HEX}"
    first = evaluator_release._preflight_payload(  # type: ignore[attr-defined]
        repository_id="mini_blobstore_v1",
        dataset_id="mini_blobstore_v1",
        backend="docker",
        container_image=image,
        container_platform="linux/amd64",
        expected_environment_hash="b" * 64,
        expected_qualification_hash="c" * 64,
        plan=plan,
        reasons=(),
    )
    second = evaluator_release._preflight_payload(  # type: ignore[attr-defined]
        repository_id="mini_blobstore_v1",
        dataset_id="mini_blobstore_v1",
        backend="docker",
        container_image=image,
        container_platform="linux/amd64",
        expected_environment_hash="b" * 64,
        expected_qualification_hash="c" * 64,
        plan=plan,
        reasons=(),
    )
    assert first["attestation_hash"] == second["attestation_hash"]
    assert image not in json.dumps(first, sort_keys=True)
    assert first["container_image_identity_kind"] == "registry_digest"


@pytest.mark.unit
def test_programbench_preflight_fails_closed_without_invoking_docker(tmp_path: Path) -> None:
    payload, receipt = evaluator_release.preflight_local_evaluator(
        repository_id="ProgramBench",
        dataset_id="ProgramBench",
        backend="docker",
        container_image=f"private.example/hidden@sha256:{_HEX}",
        container_platform="linux/amd64",
        expected_environment_hash="b" * 64,
        expected_qualification_hash="c" * 64,
        timeout_seconds=1,
        output=tmp_path / "preflight.json",
    )

    assert payload["status"] == "failed"
    assert payload["blocking_reasons"] == [
        "programbench_evaluator_not_available_in_relic_release"
    ]
    assert receipt.is_file()
    assert "private.example" not in receipt.read_text(encoding="utf-8")


@pytest.mark.unit
def test_local_image_id_cannot_be_used_as_a_source_main_binding(tmp_path: Path) -> None:
    binding = {
        "mini_blobstore_v1": {
            "backend": "docker",
            "container_image": f"sha256:{_HEX}",
            "container_platform": "linux/amd64",
            "environment_hash": "b" * 64,
            "qualification_plan_hash": "c" * 64,
        }
    }
    path = tmp_path / "local-binding.json"
    path.write_text(json.dumps(binding), encoding="utf-8")

    with pytest.raises(SourceMainRunnerError, match="evaluator_binding_image_not_digest_pinned"):
        load_evaluator_bindings(path)


@pytest.mark.docker
def test_source_derived_evaluator_image_build_and_smoke_opt_in() -> None:
    if os.environ.get("RELIC_RUN_DOCKER_TESTS") != "1":
        pytest.skip("set RELIC_RUN_DOCKER_TESTS=1 to build the evaluator image")
    result = evaluator_release.build_local_evaluator(
        tag="relic-oss-evaluator:test",
        smoke=True,
    )
    assert result["status"] == "passed"
    assert result["smoke"] == "passed"
