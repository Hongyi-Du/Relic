from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from environments.org_env.llm.client import MockOrgLLMClient
from relic.cell_spec import (
    SOURCE_BRANCH,
    SOURCE_COMMIT,
    compile_cell_spec,
    load_frozen_cell_spec,
    stable_sha256,
)
from relic.cell_worker import (
    CellWorkerError,
    _build_openai_client,
    _build_world,
    _checkpoint_world,
    _execution_binding,
    _identity_environment,
    _latest_checkpoint,
    _scoped_environment,
    _validate_execution_binding,
    _validate_world_binding,
    _write_public_trace,
    _write_status,
    inspect_cell,
    run_cell,
)
from relic.cli import main
from relic.replay import load_trace
from relic.replay.source_export import public_trace_path


def _spec(tmp_path: Path, *, arm: str = "B3", model: str = "gpt-5.6-terra"):
    return compile_cell_spec(
        model=model,
        workload="W01",
        arm=arm,
        seed=1401,
        output_root=tmp_path,
    )


def _binding(spec):
    policy = {
        "trust_level": "untrusted",
        "backend": "docker",
        "container_image": "example.invalid/relic@sha256:" + "b" * 64,
        "container_platform": "linux/amd64",
        "network_enabled": False,
    }
    return _execution_binding(
        spec,
        model_binding={
            "provider": "openai",
            "runtime_model": "gpt-5.6-terra",
            "reasoning_effort": "low",
            "routing_context_fingerprint": "a" * 64,
        },
        evaluator_binding={
            "execution_policy": policy,
            "execution_policy_sha256": stable_sha256(policy),
            "qualification_plan_sha256": "d" * 64,
            "evaluator_environment_sha256": "e" * 64,
            "dataset_id": spec.dataset_id,
        },
    )


def _execute_marker(path: str):
    Path(path).write_text("unpickled", encoding="utf-8")
    return None


class _MaliciousOnLoad:
    def __init__(self, marker: Path):
        self.marker = marker

    def __reduce__(self):
        return _execute_marker, (str(self.marker),)


def test_all_120_canonical_cell_specs_are_unique_and_frozen(tmp_path: Path) -> None:
    fingerprints: set[str] = set()
    cell_ids: set[str] = set()
    for workload in (f"W{index:02d}" for index in range(1, 11)):
        for seed in (1401, 2711, 4013):
            for arm in ("B0", "B1", "B2", "B3"):
                spec = compile_cell_spec(
                    model="gpt-5.6-terra",
                    workload=workload,
                    arm=arm,
                    seed=seed,
                    output_root=tmp_path,
                    verify_pack=False,
                )
                fingerprints.add(spec.fingerprint)
                cell_ids.add(spec.cell_id)
                assert spec.source["branch"] == SOURCE_BRANCH
                assert spec.source["commit"] == SOURCE_COMMIT
                assert spec.mechanism_ablations == ("work_rhythm",)
                assert spec.ticks == 336
                assert spec.checkpoint_every == 24
    assert len(fingerprints) == 120
    assert len(cell_ids) == 120


@pytest.mark.parametrize(
    ("arm", "condition", "roster", "action", "profile", "capability", "institution"),
    (
        ("B0", "b0_single_agent_founder", 1, "llm_direct", False, False, False),
        ("B1", "b1_persistent_role_org", 8, "llm_direct", False, False, False),
        ("B2", "b2_policy_conditioned_org", 8, "profile_policy", True, True, False),
        ("B3", "b3_full_sociogenesis", 8, "profile_policy", True, True, True),
    ),
)
def test_arm_yaml_and_runtime_condition_are_bound_fail_closed(
    tmp_path: Path,
    arm: str,
    condition: str,
    roster: int,
    action: str,
    profile: bool,
    capability: bool,
    institution: bool,
) -> None:
    spec = _spec(tmp_path, arm=arm)
    assert spec.condition_id == condition
    assert spec.arm_config["roster_size"] == roster
    assert spec.arm_config["action_selection"] == action
    assert spec.arm_config["profile_conditioning"] is profile
    assert spec.arm_config["capability_learning"] is capability
    assert spec.arm_config["institutionalization"] is institution

    world = _build_world(spec, MockOrgLLMClient())
    _validate_world_binding(world, spec, require_client=True)
    assert world.action_selection_mode == action
    assert world.time.rhythm_enabled is False
    assert world.approval_mode == "semi_auto"


def test_cell_spec_identity_ignores_output_location_and_round_trips(tmp_path: Path) -> None:
    first = _spec(tmp_path / "one")
    second = _spec(tmp_path / "two")
    assert first.fingerprint == second.fingerprint
    assert first.cell_id == second.cell_id

    cell_dir = tmp_path / "frozen"
    cell_dir.mkdir()
    (cell_dir / "cell-spec.json").write_text(
        json.dumps(first.document()), encoding="utf-8"
    )
    loaded = load_frozen_cell_spec(cell_dir)
    assert loaded.fingerprint == first.fingerprint


def test_noncanonical_seed_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="seed_not_in_main_study"):
        compile_cell_spec(
            model="gpt-5.6-terra",
            workload="W01",
            arm="B3",
            seed=7,
            output_root=tmp_path,
        )


def test_claude_uses_the_openai_compatible_gateway_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = _spec(tmp_path, model="claude-opus-4.6")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("RELIC_CLAUDE_OPUS_4_6_MODEL", "claude-opus-4-6")

    client, binding = _build_openai_client(spec)

    assert client.provider == "openai"
    assert client.model == "claude-opus-4-6"
    assert client.wire_api == "chat_completions"
    assert client.json_transport == "prompt_only"
    assert client.effective_reasoning_effort == "low"
    assert binding["provider"] == "openai"
    assert binding["wire_api"] == "chat_completions"
    assert binding["json_transport"] == "prompt_only"


def test_evaluator_preflight_happens_before_client_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = _spec(tmp_path)
    client_called = False

    def fail_preflight(_spec):
        raise CellWorkerError("formal_evaluator_runtime_binding_missing:container_image")

    def forbidden_client(_spec):
        nonlocal client_called
        client_called = True
        raise AssertionError("client must not be built before evaluator preflight")

    monkeypatch.setattr("relic.cell_worker._preflight_evaluator", fail_preflight)
    monkeypatch.setattr("relic.cell_worker._build_openai_client", forbidden_client)
    with pytest.raises(CellWorkerError, match="formal_evaluator_runtime_binding_missing"):
        run_cell(spec)
    assert client_called is False
    status = json.loads((spec.cell_dir / "public" / "status.json").read_text())
    assert status["stage"] == "preflight"
    assert status["tick"] == 0
    assert status["failure_code"].startswith("formal_evaluator_runtime_binding_missing")
    frozen_status = (spec.cell_dir / "public" / "status.json").read_bytes()
    with pytest.raises(CellWorkerError, match="cell_directory_not_empty"):
        run_cell(spec)
    assert (spec.cell_dir / "public" / "status.json").read_bytes() == frozen_status


def test_public_projection_uses_only_explicit_source_fields(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    secret = "sk-private-canary-never-publish"

    world = _build_world(spec, MockOrgLLMClient())
    world.messages = [{"body": secret}]
    world.memory = {"paul": [{"secret": secret}]}
    task = next(iter(world.tasks.values()))
    task.description = secret
    _write_public_trace(spec.cell_dir, spec, status="running", world=world)

    path = public_trace_path(spec.cell_dir)
    payload = load_trace(path)
    encoded = path.read_text(encoding="utf-8")
    assert secret not in encoded
    assert payload["schema_version"] == "relic-trace-v1"
    assert payload["privacy"] == {
        "private_reflections_included": False,
        "private_memories_included": False,
        "provider_messages_included": False,
    }
    assert "description" not in payload["frames"][0]["organization"]["tasks"][0]


def test_status_verifies_sidecar_without_unpickling_checkpoint(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    binding = _binding(spec)
    marker = tmp_path / "unpickle-marker"
    world = SimpleNamespace(
        world_tick=24,
        scenario=SimpleNamespace(seed=spec.seed),
        llm_client=None,
        llm_decides_actions=False,
        approval_mode=spec.approval_mode,
        payload=_MaliciousOnLoad(marker),
    )
    spec.cell_dir.mkdir(parents=True)
    (spec.cell_dir / "private" / "checkpoints").mkdir(parents=True)
    (spec.cell_dir / "public").mkdir(parents=True)
    (spec.cell_dir / "cell-spec.json").write_text(
        json.dumps(spec.document()), encoding="utf-8"
    )
    (spec.cell_dir / "private" / "execution-binding.json").write_text(
        json.dumps(binding), encoding="utf-8"
    )
    with _scoped_environment(_identity_environment(spec, binding)):
        _checkpoint_world(world, spec.cell_dir, spec)
    _write_status(
        spec.cell_dir,
        spec,
        status="running",
        stage="rollout",
        tick=24,
        started_at="2026-09-16T00:00:00+00:00",
    )

    status = inspect_cell(spec.cell_dir)

    assert marker.exists() is False
    assert status["checkpoints"] == [
        {
            "tick": 24,
            "created": status["checkpoints"][0]["created"],
            "verified": True,
        }
    ]


def test_resume_rejects_checkpoint_model_identity_mismatch(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    binding = _binding(spec)
    world = SimpleNamespace(
        world_tick=24,
        scenario=SimpleNamespace(seed=spec.seed),
        llm_client=None,
        llm_decides_actions=False,
        approval_mode=spec.approval_mode,
    )
    (spec.cell_dir / "private" / "checkpoints").mkdir(parents=True)
    with _scoped_environment(_identity_environment(spec, binding)):
        _checkpoint_world(world, spec.cell_dir, spec)

    changed = dict(binding)
    changed["model_binding_sha256"] = "f" * 64
    with pytest.raises(CellWorkerError, match="checkpoint_identity_mismatch:model_binding"):
        _latest_checkpoint(spec.cell_dir, spec, changed)


def test_execution_binding_tampering_is_rejected(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    binding = _binding(spec)
    _validate_execution_binding(spec, binding)

    changed = json.loads(json.dumps(binding))
    changed["model"]["runtime_model"] = "different-model"
    with pytest.raises(CellWorkerError, match="execution_binding_hash_mismatch"):
        _validate_execution_binding(spec, changed)


def test_runner_refuses_symlink_cell_directory(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "cell-link"
    link.symlink_to(target, target_is_directory=True)

    with pytest.raises(CellWorkerError, match="cell_directory_symlink_forbidden"):
        run_cell(spec, cell_dir=link)


def test_completed_resume_is_side_effect_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = _spec(tmp_path)
    binding = _binding(spec)
    (spec.cell_dir / "private" / "checkpoints").mkdir(parents=True)
    (spec.cell_dir / "public").mkdir(parents=True)
    (spec.cell_dir / "cell-spec.json").write_text(
        json.dumps(spec.document()), encoding="utf-8"
    )
    (spec.cell_dir / "private" / "execution-binding.json").write_text(
        json.dumps(binding), encoding="utf-8"
    )
    world = SimpleNamespace(
        world_tick=spec.ticks,
        scenario=SimpleNamespace(seed=spec.seed),
        llm_client=None,
        llm_decides_actions=False,
        approval_mode=spec.approval_mode,
    )
    with _scoped_environment(_identity_environment(spec, binding)):
        _checkpoint_world(world, spec.cell_dir, spec)
    _write_status(
        spec.cell_dir,
        spec,
        status="completed",
        stage="complete",
        tick=spec.ticks,
        started_at="2026-09-16T00:00:00+00:00",
    )
    (spec.cell_dir / "run-record.json").write_text(
        json.dumps(
            {
                "schema_version": "orgenv_experiment_run_v2",
                "status": "completed",
                "replication_id": spec.cell_id,
                "seed": spec.seed,
            }
        ),
        encoding="utf-8",
    )
    calls = {"preflight": 0, "client": 0, "final": 0}

    def forbidden(name):
        def invoke(*_args, **_kwargs):
            calls[name] += 1
            raise AssertionError(f"completed resume called {name}")

        return invoke

    monkeypatch.setattr("relic.cell_worker._preflight_evaluator", forbidden("preflight"))
    monkeypatch.setattr("relic.cell_worker._build_openai_client", forbidden("client"))
    monkeypatch.setattr("relic.cell_worker.run_final_evaluation", forbidden("final"))

    result = run_cell(spec, resume=True)

    assert result.status == "completed"
    assert calls == {"preflight": 0, "client": 0, "final": 0}


def test_status_rejects_foreign_checkpoint_without_unpickling(tmp_path: Path) -> None:
    b2 = _spec(tmp_path, arm="B2")
    b3 = _spec(tmp_path, arm="B3")
    b2_binding = _binding(b2)
    b3_binding = _binding(b3)
    marker = tmp_path / "foreign-unpickle-marker"
    (b2.cell_dir / "private" / "checkpoints").mkdir(parents=True)
    (b2.cell_dir / "public").mkdir(parents=True)
    (b2.cell_dir / "cell-spec.json").write_text(
        json.dumps(b2.document()), encoding="utf-8"
    )
    (b2.cell_dir / "private" / "execution-binding.json").write_text(
        json.dumps(b2_binding), encoding="utf-8"
    )
    foreign_world = SimpleNamespace(
        world_tick=24,
        scenario=SimpleNamespace(seed=b3.seed),
        llm_client=None,
        llm_decides_actions=False,
        approval_mode=b3.approval_mode,
        payload=_MaliciousOnLoad(marker),
    )
    with _scoped_environment(_identity_environment(b3, b3_binding)):
        _checkpoint_world(foreign_world, b2.cell_dir, b3)
    _write_status(
        b2.cell_dir,
        b2,
        status="running",
        stage="rollout",
        tick=24,
        started_at="2026-09-16T00:00:00+00:00",
    )

    with pytest.raises(CellWorkerError, match="checkpoint_identity_mismatch"):
        inspect_cell(b2.cell_dir)
    assert marker.exists() is False


def test_trace_rejects_legacy_count_only_sidecar(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    (spec.cell_dir / "public").mkdir(parents=True)
    malicious = {
        "schema_version": "relic-public-trace-v1",
        "projection_profile": "relic-public-allowlist-v1",
        "cell_id": spec.cell_id,
        "study": spec.study,
        "model_label": spec.model_config["paper_label"],
        "workload": spec.workload.upper(),
        "arm": spec.arm.upper(),
        "seed": spec.seed,
        "target_tick": spec.ticks,
        "terminal_status": None,
        "frames": [{"tick": 24, "counts": {}, "private_canary": "secret"}],
    }
    (spec.cell_dir / "public" / "trace.json").write_text(
        json.dumps(malicious), encoding="utf-8"
    )
    world = _build_world(spec, MockOrgLLMClient())
    with pytest.raises(CellWorkerError, match="public_trace_legacy_sidecar_rejected"):
        _write_public_trace(spec.cell_dir, spec, status="running", world=world)


def test_trace_drops_unallowlisted_task_text(tmp_path: Path) -> None:
    secret = "private-status-canary"
    spec = _spec(tmp_path)
    world = _build_world(spec, MockOrgLLMClient())
    task = next(iter(world.tasks.values()))
    task.description = secret
    task.progress_evidence = [secret]
    _write_public_trace(spec.cell_dir, spec, status="running", world=world)

    assert secret not in public_trace_path(spec.cell_dir).read_text(encoding="utf-8")


def test_cli_exposes_canonical_runner_commands(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as stopped:
        main(["run-cell", "--help"])
    assert stopped.value.code == 0
    assert "--workload" in capsys.readouterr().out

    with pytest.raises(SystemExit) as stopped:
        main(["status", "--help"])
    assert stopped.value.code == 0
    assert "--cell-dir" in capsys.readouterr().out

    with pytest.raises(SystemExit) as stopped:
        main(["evaluate", "--help"])
    assert stopped.value.code == 0
    assert "--cell-dir" in capsys.readouterr().out


def test_cli_preflight_failure_is_json_and_persisted(tmp_path: Path) -> None:
    output = tmp_path / "cli-cell"
    environment = dict(os.environ)
    for name in (
        "RELIC_EVALUATOR_BACKEND",
        "RELIC_EVALUATOR_CONTAINER_IMAGE",
        "RELIC_EVALUATOR_CONTAINER_PLATFORM",
        "OPENAI_API_KEY",
        "ORG_LLM_API_KEY",
    ):
        environment.pop(name, None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        (
            sys.executable,
            "-m",
            "relic.cli",
            "run-cell",
            "--model",
            "gpt-5.6-terra",
            "--workload",
            "W01",
            "--arm",
            "B3",
            "--seed",
            "1401",
            "--output-dir",
            str(output),
        ),
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert completed.returncode == 2
    assert completed.stdout == ""
    error = json.loads(completed.stderr)
    assert error["status"] == "failed"
    # The default local evaluator preflight succeeds; the next required
    # binding is the provider credential for the model client.
    assert error["error"].startswith("model_credential_missing:openai")
    status = json.loads((output / "public" / "status.json").read_text())
    assert status["status"] == "infra_error"
    assert status["stage"] == "preflight"
