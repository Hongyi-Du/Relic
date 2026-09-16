from __future__ import annotations

import json
import subprocess

import pytest

from relic import release_checks
from relic.cli import main


def _by_code(report: dict, code: str) -> list[dict]:
    return [check for check in report["checks"] if check["code"] == code]


@pytest.mark.unit
def test_mock_smoke_never_constructs_a_provider_or_formal_evaluator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ORG_LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("RELIC_EVALUATOR_BACKEND", raising=False)
    monkeypatch.delenv("RELIC_EVALUATOR_CONTAINER_IMAGE", raising=False)
    monkeypatch.delenv("RELIC_EVALUATOR_CONTAINER_PLATFORM", raising=False)

    report = release_checks.smoke(mode="mock")

    assert report["status"] == "passed"
    assert report["exit_code"] == release_checks.EXIT_OK
    assert report["provider_calls_made"] == 0
    assert report["formal_experiment_started"] is False
    assert _by_code(report, "mock_smoke_runtime_and_output_validated")
    assert _by_code(report, "mock_smoke_is_not_a_formal_experiment")


@pytest.mark.unit
def test_formal_gate_reports_missing_current_binding_with_stable_exit_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RELIC_EVALUATOR_BACKEND", raising=False)
    monkeypatch.delenv("RELIC_EVALUATOR_CONTAINER_IMAGE", raising=False)
    monkeypatch.delenv("RELIC_EVALUATOR_CONTAINER_PLATFORM", raising=False)
    monkeypatch.delenv("ORG_LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    report = release_checks.check_environment(scope="formal", model="gpt-5.6-terra")

    assert report["status"] == "failed"
    assert report["exit_code"] == release_checks.EXIT_FORMAL_GATE_FAILED
    missing = _by_code(report, "formal_evaluator_runtime_binding_missing")
    assert {check["details"]["missing"] for check in missing} == {
        "backend",
        "container_image",
        "container_platform",
    }
    assert release_checks.report_exit_code(report) == release_checks.EXIT_FORMAL_GATE_FAILED


@pytest.mark.unit
def test_reports_never_echo_credential_values(monkeypatch: pytest.MonkeyPatch) -> None:
    secret = "sk-release-check-secret-value"
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    monkeypatch.setenv("RELIC_EVALUATOR_CONTAINER_IMAGE", secret)

    report = release_checks.check_environment(scope="formal", model="gpt-5.6-terra")
    rendered = json.dumps(report, sort_keys=True)

    assert secret not in rendered
    credential = _by_code(report, "model_credential_available")
    assert credential[0]["details"]["credential_configured"] is True
    assert credential[0]["details"]["accepted_environment_variables"] == [
        "ORG_LLM_API_KEY",
        "OPENAI_API_KEY",
    ]
    optional = _by_code(report, "optional_model_credential_available")
    assert optional[0]["details"]["credential_configured"] is True


@pytest.mark.unit
def test_memory_recommendation_and_parallelism_warning_are_structured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(release_checks, "visible_memory_gib", lambda: 64.0)

    report = release_checks.check_environment(scope="core", requested_parallelism=5)

    warning = _by_code(report, "requested_parallelism_above_recommendation")
    assert len(warning) == 1
    assert warning[0]["status"] == "warn"
    assert warning[0]["details"] == {
        "requested_parallelism": 5,
        "visible_memory_gib": 64.0,
        "recommended_max_parallel": 4,
        "budget_gib_per_active_cell": 16,
    }
    assert report["exit_code"] == release_checks.EXIT_OK


@pytest.mark.unit
def test_docker_image_qualification_requires_linux_amd64(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def inspected(*_args, **_kwargs) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=("docker", "image", "inspect"),
            returncode=0,
            stdout="linux/arm64\n",
            stderr="",
        )

    monkeypatch.setattr(release_checks.subprocess, "run", inspected)

    qualified, code = release_checks._docker_image_is_linux_amd64(  # type: ignore[attr-defined]
        "docker", "example.invalid/evaluator@sha256:" + "a" * 64
    )

    assert qualified is False
    assert code == "docker_image_platform_mismatch"


@pytest.mark.unit
def test_apptainer_cache_probe_is_read_only_and_digest_specific(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    digest = "a" * 64

    def cached(argv, **_kwargs) -> subprocess.CompletedProcess[str]:
        assert argv == ("apptainer", "cache", "list", "-v")
        return subprocess.CompletedProcess(
            args=argv,
            returncode=0,
            stdout=f"blob sha256.{digest}\n",
            stderr="",
        )

    monkeypatch.setattr(release_checks.subprocess, "run", cached)
    qualified, code = release_checks._apptainer_image_is_cached(  # type: ignore[attr-defined]
        "apptainer", f"docker://example.invalid/evaluator@sha256:{digest}"
    )

    assert qualified is True
    assert code == "apptainer_image_cache_entry_available"


@pytest.mark.unit
def test_formal_smoke_uses_network_isolated_executor_roundtrip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from relic.evaluation import execution
    from relic.evaluation.execution import CommandOutcome

    image = "example.invalid/evaluator@sha256:" + "a" * 64
    monkeypatch.setenv("RELIC_EVALUATOR_BACKEND", "docker")
    monkeypatch.setenv("RELIC_EVALUATOR_CONTAINER_IMAGE", image)
    monkeypatch.setenv("RELIC_EVALUATOR_CONTAINER_PLATFORM", "linux/amd64")
    environment_requests = []

    def environment_gate(**kwargs):
        environment_requests.append(kwargs)
        return {
            "status": "passed",
            "checks": [],
            "recommendations": {},
        }

    monkeypatch.setattr(release_checks, "check_environment", environment_gate)

    seen = {}

    class FakeExecutor:
        def __init__(self, policy) -> None:
            self.policy = policy

        def run(self, *, root, argv, timeout_seconds):
            del timeout_seconds
            seen["root"] = root
            seen["policy"] = self.policy
            (root / argv[5]).write_text(argv[6], encoding="utf-8")
            return CommandOutcome(
                status="passed",
                exit_code=0,
                elapsed_sec=0.0,
                backend="docker",
            )

    monkeypatch.setattr(
        execution,
        "build_command_executor",
        lambda policy: FakeExecutor(policy),
    )

    report = release_checks.smoke(mode="formal")

    assert report["status"] == "passed"
    assert _by_code(report, "formal_smoke_executor_workspace_roundtrip_passed")
    assert seen["policy"].network_enabled is False
    assert seen["policy"].backend == "docker"
    assert seen["policy"].container_image == image
    assert environment_requests == [
        {"scope": "formal", "model": "gpt-5.6-terra"}
    ]


@pytest.mark.unit
def test_invalid_inputs_have_a_stable_usage_exit_code() -> None:
    report = release_checks.check_environment(scope="not-a-scope")  # type: ignore[arg-type]
    smoke_report = release_checks.smoke(mode="not-a-mode")  # type: ignore[arg-type]

    assert report["exit_code"] == release_checks.EXIT_INVALID_REQUEST
    assert report["checks"][0]["code"] == "release_check_scope_invalid"
    assert smoke_report["exit_code"] == release_checks.EXIT_INVALID_REQUEST
    assert smoke_report["checks"][0]["code"] == "release_smoke_mode_invalid"


@pytest.mark.unit
def test_cli_exposes_check_env_and_mock_smoke(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    core_report = {
        "schema_version": release_checks.REPORT_SCHEMA_VERSION,
        "status": "passed",
        "exit_code": 0,
    }
    monkeypatch.setattr(release_checks, "check_environment", lambda **_kwargs: core_report)
    monkeypatch.setattr(release_checks, "smoke", lambda **_kwargs: core_report)

    assert main(["check-env", "--scope", "core", "--max-parallel", "2"]) == 0
    assert '"status": "passed"' in capsys.readouterr().out
    assert main(["smoke", "--mode", "mock"]) == 0
    assert '"exit_code": 0' in capsys.readouterr().out



@pytest.mark.unit
def test_cli_invalid_release_check_inputs_use_stable_usage_exit(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["check-env", "--scope", "invalid"]) == release_checks.EXIT_INVALID_REQUEST
    assert '"code": "release_check_scope_invalid"' in capsys.readouterr().out
    assert main(["smoke", "--mode", "invalid"]) == release_checks.EXIT_INVALID_REQUEST
    assert '"code": "release_smoke_mode_invalid"' in capsys.readouterr().out
