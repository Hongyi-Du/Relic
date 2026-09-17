"""Acceptance tests for the optional host evaluator reproduction path."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from environments.org_env.product.substrates.final_evaluation import (
    _frozen_evaluator_executor,
)
from relic.cell_spec import compile_cell_spec
from relic.cell_worker import _preflight_evaluator
from relic.source_runner import (
    SourceMainRunnerError,
    build_source_main_manifest,
    run_source_main,
)
from relic.transfer_runner import (
    TransferRunnerError,
    _plan_digest as transfer_plan_digest,
    run_transfer,
)
from tools.run_org_baselines import build_case_environment


def _fake_source_batch(argv: list[str], calls: list[list[str]]) -> int:
    """Materialize the child manifest without starting a provider process."""

    calls.append(list(argv))
    root = Path(argv[argv.index("--output-root") + 1])
    root.mkdir(parents=True, exist_ok=True)
    mode = argv[argv.index("--evaluator-mode") + 1]
    backend = (
        argv[argv.index("--evaluator-backend") + 1]
        if "--evaluator-backend" in argv
        else mode
    )
    manifest = {
        "evaluator": {
            "mode": mode,
            "backend": backend,
            "binding_supplied": mode == "container",
            "strict_reproducibility": "--strict-reproducibility" in argv,
            "observed": [],
        },
        "cases": [],
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return 0


def test_main_defaults_to_local_evaluator_without_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(
        "relic.source_runner._invoke_source_batch",
        lambda argv, _runtime: _fake_source_batch(list(argv), calls),
    )

    result = run_source_main(
        model="gpt-5.6-terra",
        output_root=tmp_path / "main",
        workloads=("w01",),
        seeds=(1401,),
    )

    assert result.status == "partial"
    assert len(calls) == 1
    assert calls[0][calls[0].index("--evaluator-mode") + 1] == "local"
    assert "--evaluator-backend" not in calls[0]
    manifest = json.loads(
        (tmp_path / "main" / "source_main_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["execution"]["evaluator"]["mode"] == "local"
    assert manifest["execution"]["evaluator"]["bindings"] == {}


@pytest.mark.parametrize("model_source", ["environment", "cli"])
def test_unresolved_dry_plan_resolves_model_before_first_execution(tmp_path, monkeypatch, model_source):
    for key in ("RELIC_RUNTIME_MODEL", "ORG_LLM_RUNTIME_MODEL", "OPENAI_MODEL", "ORG_LLM_MODEL",
                "RELIC_CLAUDE_OPUS_4_6_MODEL"):
        monkeypatch.delenv(key, raising=False)
    calls = []
    monkeypatch.setattr("relic.source_runner._invoke_source_batch",
                        lambda argv, _runtime: _fake_source_batch(list(argv), calls))
    result = run_source_main(model="claude-opus-4.6", output_root=tmp_path,
                             dry_run=True, workloads=("w01",), seeds=(1401,))
    before = json.loads(result.manifest_path.read_text())
    assert before["plan"]["model"]["runtime_model"] == ""
    overrides = {}
    if model_source == "environment":
        monkeypatch.setenv("RELIC_CLAUDE_OPUS_4_6_MODEL", "my-deployment")
    else:
        overrides["runtime_model"] = "my-deployment"
    run_source_main(manifest_path=result.manifest_path, resume=True,
                    workloads=("w01",), seeds=(1401,), **overrides)
    after = json.loads(result.manifest_path.read_text())
    assert after["plan"]["model"]["runtime_model"] == "my-deployment"
    assert after["plan"]["model"]["runtime"]["ORG_LLM_MODEL"] == "my-deployment"
    assert after["plan"]["plan_sha256"] != before["plan"]["plan_sha256"]
    with pytest.raises(SourceMainRunnerError, match="resume_runtime_model_mismatch"):
        run_source_main(manifest_path=result.manifest_path, resume=True,
                        runtime_model="a-different-model", workloads=("w01",), seeds=(1401,))


@pytest.mark.parametrize("model_source", ["environment", "cli"])
def test_transfer_unresolved_dry_plan_resolves_once_before_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model_source: str
) -> None:
    for key in (
        "RELIC_RUNTIME_MODEL",
        "ORG_LLM_RUNTIME_MODEL",
        "OPENAI_MODEL",
        "ORG_LLM_MODEL",
    ):
        monkeypatch.delenv(key, raising=False)
    calls: list[list[str]] = []
    monkeypatch.setattr(
        "relic.transfer_runner._invoke_source_batch",
        lambda argv, _runtime: _fake_source_batch(list(argv), calls),
    )

    result = run_transfer(
        output_root=tmp_path / "transfer",
        arm="text",
        dry_run=True,
        workloads=("w01",),
        seeds=(1401,),
    )
    path = result.manifest_path
    payload = json.loads(path.read_text(encoding="utf-8"))
    # Simulate a provider-free plan from a model configuration whose deployment
    # name is unresolved.  The fixed transfer paper ID remains unchanged.
    payload["plan"]["model"]["runtime_model"] = ""
    payload["plan"]["model"]["runtime"]["ORG_LLM_MODEL"] = ""
    payload["plan"]["plan_sha256"] = transfer_plan_digest(payload["plan"])
    path.write_text(json.dumps(payload), encoding="utf-8")

    overrides: dict[str, str] = {}
    if model_source == "environment":
        monkeypatch.setenv("RELIC_RUNTIME_MODEL", "transfer-deployment")
    else:
        overrides["runtime_model"] = "transfer-deployment"
    run_transfer(
        manifest_path=path,
        resume=True,
        arm="text",
        workloads=("w01",),
        seeds=(1401,),
        **overrides,
    )
    after = json.loads(path.read_text(encoding="utf-8"))
    assert after["plan"]["model"]["runtime_model"] == "transfer-deployment"
    assert after["plan"]["model"]["runtime"]["ORG_LLM_MODEL"] == "transfer-deployment"
    assert len(calls) == 2

    with pytest.raises(
        TransferRunnerError, match="transfer_resume_runtime_model_mismatch"
    ):
        run_transfer(
            manifest_path=path,
            resume=True,
            arm="text",
            runtime_model="a-different-model",
            workloads=("w01",),
            seeds=(1401,),
        )


def test_transfer_defaults_to_local_evaluator_without_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(
        "relic.transfer_runner._invoke_source_batch",
        lambda argv, _runtime: _fake_source_batch(list(argv), calls),
    )

    result = run_transfer(
        output_root=tmp_path / "transfer",
        workloads=("w01",),
        seeds=(1401,),
    )

    assert result.status == "partial"
    assert len(calls) == 2
    assert all(
        call[call.index("--evaluator-mode") + 1] == "local"
        and "--evaluator-backend" not in call
        for call in calls
    )
    manifest = json.loads(
        (tmp_path / "transfer" / "transfer_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["execution"]["evaluator"]["mode"] == "local"


def test_non_strict_binding_is_optional_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(
        "relic.source_runner._invoke_source_batch",
        lambda argv, _runtime: _fake_source_batch(list(argv), calls),
    )
    bindings = tmp_path / "bindings.json"
    bindings.write_text(
        json.dumps(
            {
                "mini_blobstore_v1": {
                    "backend": "docker",
                    "container_image": "relic-oss-evaluator:operator",
                    "container_platform": "",
                    "environment_hash": "",
                    "qualification_plan_hash": "",
                }
            }
        ),
        encoding="utf-8",
    )

    result = run_source_main(
        model="gpt-5.6-terra",
        output_root=tmp_path / "main",
        workloads=("w01",),
        seeds=(1401,),
        evaluator_bindings_path=bindings,
    )

    assert result.status == "partial"
    assert calls[0][calls[0].index("--evaluator-mode") + 1] == "container"
    assert calls[0][calls[0].index("--evaluator-backend") + 1] == "docker"
    assert "--strict-reproducibility" not in calls[0]
    manifest = json.loads(
        (tmp_path / "main" / "source_main_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["execution"]["evaluator"]["bindings"]["mini_blobstore_v1"] == {
        "backend": "docker",
        "container_image": "relic-oss-evaluator:operator",
        "container_platform": "",
        "environment_hash": "",
        "qualification_plan_hash": "",
    }


def test_strict_mode_requires_pinned_binding_before_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "relic.source_runner._invoke_source_batch",
        lambda *_args, **_kwargs: pytest.fail("strict preflight must stop first"),
    )

    with pytest.raises(
        SourceMainRunnerError, match="^formal_evaluator_bindings_required$"
    ):
        run_source_main(
            model="gpt-5.6-terra",
            output_root=tmp_path / "strict",
            workloads=("w01",),
            seeds=(1401,),
            strict_reproducibility=True,
        )


@pytest.mark.parametrize(
    ("field", "value", "error"),
    (
        ("container_image", "relic-oss-evaluator:operator", "not_digest_pinned"),
        ("container_platform", "linux/arm64", "platform_invalid"),
        ("environment_hash", "not-a-hash", "environment_hash_invalid"),
        ("qualification_plan_hash", "not-a-hash", "qualification_hash_invalid"),
    ),
)
def test_strict_binding_validates_digest_platform_and_hashes(
    tmp_path: Path, field: str, value: str, error: str
) -> None:
    binding = {
        "backend": "docker",
        "container_image": "registry.example/evaluator@sha256:" + "a" * 64,
        "container_platform": "linux/amd64",
        "environment_hash": "b" * 64,
        "qualification_plan_hash": "c" * 64,
    }
    binding[field] = value
    path = tmp_path / "bindings.json"
    path.write_text(json.dumps({"mini_blobstore_v1": binding}), encoding="utf-8")
    with pytest.raises(SourceMainRunnerError, match=error):
        run_source_main(
            model="gpt-5.6-terra",
            output_root=tmp_path / "strict",
            workloads=("w01",),
            seeds=(1401,),
            evaluator_bindings_path=path,
            strict_reproducibility=True,
            dry_run=True,
        )


def test_local_cell_preflight_runs_public_host_evaluator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = compile_cell_spec(
        model="gpt-5.6-terra",
        workload="W01",
        arm="B3",
        seed=1401,
        output_root=tmp_path,
    )
    monkeypatch.setenv("RELIC_EVALUATOR_MODE", "local")
    monkeypatch.delenv("RELIC_EVALUATOR_STRICT_REPRODUCIBILITY", raising=False)
    monkeypatch.delenv("RELIC_EVALUATOR_BACKEND", raising=False)
    monkeypatch.delenv("RELIC_EVALUATOR_CONTAINER_IMAGE", raising=False)
    monkeypatch.delenv("RELIC_EVALUATOR_CONTAINER_PLATFORM", raising=False)

    binding = _preflight_evaluator(spec)

    assert binding["execution_policy"]["backend"] == "local"
    assert binding["execution_policy"]["strict_reproducibility"] is False
    assert len(binding["qualification_plan_sha256"]) == 64
    assert len(binding["evaluator_environment_sha256"]) == 64


def test_frozen_evaluator_strict_mode_requires_a_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "RELIC_EVALUATOR_MODE",
        "RELIC_EVALUATOR_BACKEND",
        "RELIC_EVALUATOR_CONTAINER_IMAGE",
        "RELIC_EVALUATOR_CONTAINER_PLATFORM",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("RELIC_EVALUATOR_STRICT_REPRODUCIBILITY", "1")

    with pytest.raises(
        RuntimeError, match="^strict_reproducibility_requires_container_evaluator$"
    ):
        _frozen_evaluator_executor()


def test_frozen_evaluator_explicit_container_requires_backend_and_image(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RELIC_EVALUATOR_MODE", "container")
    monkeypatch.setenv("RELIC_EVALUATOR_BACKEND", "docker")
    monkeypatch.delenv("RELIC_EVALUATOR_CONTAINER_IMAGE", raising=False)
    monkeypatch.delenv("RELIC_EVALUATOR_CONTAINER_PLATFORM", raising=False)
    monkeypatch.delenv("RELIC_EVALUATOR_STRICT_REPRODUCIBILITY", raising=False)

    with pytest.raises(
        RuntimeError,
        match="^evaluator_runtime_binding_missing:container_image$",
    ):
        _frozen_evaluator_executor()


def test_frozen_evaluator_partial_ambient_binding_defaults_to_local(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RELIC_EVALUATOR_MODE", raising=False)
    monkeypatch.setenv("RELIC_EVALUATOR_BACKEND", "docker")
    monkeypatch.delenv("RELIC_EVALUATOR_CONTAINER_IMAGE", raising=False)
    monkeypatch.setenv("RELIC_EVALUATOR_CONTAINER_PLATFORM", "linux/amd64")
    monkeypatch.delenv("RELIC_EVALUATOR_STRICT_REPRODUCIBILITY", raising=False)

    assert _frozen_evaluator_executor() is None


def test_runtime_model_cli_and_environment_overrides_preserve_canonical_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RELIC_RUNTIME_MODEL", "env-deployment")
    env_plan = build_source_main_manifest(
        model="gpt-5.6-terra", output_root=tmp_path / "env"
    )
    cli_plan = build_source_main_manifest(
        model="gpt-5.6-terra",
        output_root=tmp_path / "cli",
        runtime_model="cli-deployment",
    )

    assert env_plan["plan"]["model"]["canonical_model"] == "gpt-5.6-terra"
    assert env_plan["plan"]["model"]["runtime_model"] == "env-deployment"
    assert cli_plan["plan"]["model"]["canonical_model"] == "gpt-5.6-terra"
    assert cli_plan["plan"]["model"]["runtime_model"] == "cli-deployment"


def test_direct_baseline_bridges_openai_key_and_base_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = build_case_environment(
        {
            "PATH": "/usr/bin",
            "OPENAI_API_KEY": "provider-secret",
            "OPENAI_BASE_URL": "https://gateway.example/v1",
            "RELIC_OPENAI_DEFAULT_HEADERS_JSON": '{"x-route":"test"}',
        },
        condition_id="b0_single_agent_founder",
        dataset="mini_blobstore_v1",
        seed=1401,
        sprint_ticks=168,
        llm=True,
        llm_actions=True,
        run_tag="test",
        llm_provider="openai",
        llm_model="deployment-name",
        evaluator_mode="local",
    )

    assert environment["ORG_LLM_API_KEY"] == "provider-secret"
    assert environment["ORG_LLM_BASE_URL"] == "https://gateway.example/v1"
    assert environment["ORG_LLM_DEFAULT_HEADERS_JSON"] == '{"x-route":"test"}'
    assert environment["ORG_EVALUATOR_MODE"] == "local"
    assert environment["ORG_EVALUATOR_BACKEND"] == "local"


def test_claude_dry_plan_can_be_created_before_gateway_alias(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("RELIC_CLAUDE_OPUS_4_6_MODEL", raising=False)
    monkeypatch.delenv("RELIC_RUNTIME_MODEL", raising=False)
    monkeypatch.delenv("ORG_LLM_RUNTIME_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.delenv("ORG_LLM_MODEL", raising=False)

    payload = build_source_main_manifest(
        model="claude-opus-4.6", output_root=tmp_path
    )

    assert payload["plan"]["model"]["canonical_model"] == "claude-opus-4.6"
    assert payload["plan"]["model"]["runtime_model"] == ""


def test_missing_runtime_model_error_explains_how_to_set_deployment_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in (
        "RELIC_CLAUDE_OPUS_4_6_MODEL",
        "RELIC_RUNTIME_MODEL",
        "ORG_LLM_RUNTIME_MODEL",
        "OPENAI_MODEL",
        "ORG_LLM_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(
        SourceMainRunnerError,
        match=(
            r"runtime_model_binding_missing:RELIC_CLAUDE_OPUS_4_6_MODEL; "
            r".*provider deployment name.*--runtime-model"
        ),
    ):
        run_source_main(
            model="claude-opus-4.6",
            output_root=tmp_path,
            workloads=("w01",),
            seeds=(1401,),
        )
