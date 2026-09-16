from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from environments.org_env.backend.simulation.world import OrgWorld
from environments.org_env.config.scenarios import oss_time_machine_formal
from environments.org_env.product.substrates.final_evaluation import (
    FINAL_EVIDENCE_SCHEMA_VERSION,
    run_final_evaluation,
)
from environments.org_env.product.substrates.loader import load_oss_substrate_spec
from relic.evaluation.execution import (
    CommandOutcome,
    DockerCommandExecutor,
    ExecutionPolicy,
    LocalCommandExecutor,
    _classify_docker_infrastructure_failure,
)
from relic.evaluation.time_machine import (
    build_time_machine_evaluation_plan,
    evaluate_time_machine_candidate,
)
from relic.evaluation.workspace import CandidateWorkspace, repository_digest

_PINNED_IMAGE = "example.invalid/relic-evaluator@sha256:" + "a" * 64


def test_untrusted_execution_requires_a_pinned_image() -> None:
    with pytest.raises(ValueError, match="container_image_must_be_digest_pinned"):
        ExecutionPolicy(
            trust_level="untrusted",
            backend="docker",
            container_image="example.invalid/relic-evaluator:latest",
        )


def test_local_executor_refuses_an_untrusted_workspace(tmp_path: Path) -> None:
    executor = LocalCommandExecutor(
        policy=ExecutionPolicy(trust_level="untrusted", backend="local")
    )

    outcome = executor.run(
        root=tmp_path,
        argv=("python", "-c", "print('should not run')"),
        timeout_seconds=1,
    )

    assert outcome.status == "blocked"
    assert outcome.blocked_reason == "untrusted_local_execution_forbidden"


def test_docker_command_has_the_formal_isolation_boundary(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    evaluator = tmp_path / "evaluator"
    workspace.mkdir()
    evaluator.mkdir()
    executor = DockerCommandExecutor(
        policy=ExecutionPolicy(
            trust_level="untrusted",
            backend="docker",
            container_image=_PINNED_IMAGE,
            container_platform="linux/amd64",
            network_enabled=False,
        ),
        docker_binary="docker-test",
    )

    command = executor.build_command(
        root=workspace,
        argv=("python", "-V"),
        read_only_mounts=((evaluator, "/evaluator"),),
        workspace_read_only=True,
    )

    assert command[command.index("--network") + 1] == "none"
    assert "--read-only" in command
    assert command[command.index("--cap-drop") + 1] == "ALL"
    assert f"{workspace.resolve()}:/workspace:ro" in command
    assert f"{evaluator.resolve()}:/evaluator:ro" in command
    assert command[command.index("--platform") + 1] == "linux/amd64"


def test_docker_executor_can_clear_a_task_image_entrypoint(tmp_path: Path) -> None:
    executor = DockerCommandExecutor(
        policy=ExecutionPolicy(
            trust_level="untrusted",
            backend="docker",
            container_image=_PINNED_IMAGE,
            clear_container_entrypoint=True,
        ),
        docker_binary="docker-test",
    )

    command = executor.build_command(
        root=tmp_path.resolve(),
        argv=("python", "-c", "print('ok')"),
    )

    index = command.index("--entrypoint")
    assert command[index + 1] == ""
    assert command[index + 2] == _PINNED_IMAGE
    assert command[-3:] == ("python", "-c", "print('ok')")


def test_docker_executor_stages_candidate_in_image_workspace(tmp_path: Path) -> None:
    executor = DockerCommandExecutor(
        policy=ExecutionPolicy(
            trust_level="untrusted",
            backend="docker",
            container_image=_PINNED_IMAGE,
            clear_container_entrypoint=True,
        ),
        docker_binary="docker-test",
    )

    command = executor.build_command(
        root=tmp_path.resolve(),
        argv=("python", "-m", "pytest", "-q"),
        image_workspace="/workspace/repo",
    )

    assert "type=volume,target=/workspace/repo" in command
    assert f"{tmp_path.resolve()}:/candidate:ro" in command
    assert f"{tmp_path.resolve()}:/workspace:rw" not in command
    assert command[command.index("--workdir") + 1] == "/workspace/repo"
    stage = command[command.index("/bin/bash") + 2]
    assert "git ls-files -z" in stage
    assert "cmp -s" in stage


def test_started_container_output_cannot_spoof_a_daemon_failure() -> None:
    failure = CommandOutcome(
        status="failed",
        exit_code=1,
        elapsed_sec=0.1,
        stderr_tail="Cannot connect to the Docker daemon",
        backend="docker",
    )

    before_start = _classify_docker_infrastructure_failure(failure)
    after_start = _classify_docker_infrastructure_failure(
        failure,
        container_started=True,
    )

    assert before_start.status == "infra_error"
    assert before_start.blocked_reason == "docker_daemon_unavailable"
    assert after_start == failure


def test_candidate_workspace_is_isolated_and_rejects_symlinks(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    source_digest = repository_digest(source)

    with CandidateWorkspace.create(source) as candidate:
        (candidate.root / "module.py").write_text("VALUE = 2\n", encoding="utf-8")
        assert repository_digest(candidate.root) != source_digest

    assert repository_digest(source) == source_digest

    if hasattr(os, "symlink"):
        os.symlink(source / "module.py", source / "link.py")
        with pytest.raises(ValueError, match="candidate_workspace_symlink_forbidden"):
            repository_digest(source)


@pytest.mark.integration
def test_mini_blobstore_evaluator_plan_and_candidate() -> None:
    spec = load_oss_substrate_spec("mini_blobstore_v1")
    plan = build_time_machine_evaluation_plan(spec=spec, timeout_seconds=30)

    result = evaluate_time_machine_candidate(
        plan,
        Path(spec.reference_repo_dir),
        timeout_seconds=30,
    )

    assert plan.operational_ready is True
    assert plan.formal_environment_ready is False
    assert plan.blocking_reasons == ("formal_container_executor_required",)
    assert result.status == "passed"
    assert result.candidate_pass_rate == 1.0
    assert result.formal_claim_ready is False
    assert len(result.evidence_records) == 5


@pytest.mark.integration
def test_final_evaluator_dependency_injection_uses_relic_schema(
    tmp_path: Path,
) -> None:
    spec = load_oss_substrate_spec("mini_blobstore_v1")
    plan = build_time_machine_evaluation_plan(spec=spec, timeout_seconds=30)
    world = OrgWorld(oss_time_machine_formal(seed=13, dataset_id="mini_blobstore_v1")).build()

    def export_reference(
        _world: OrgWorld,
        destination: str,
        *,
        prefer_mainline: bool,
    ) -> None:
        del _world, prefer_mainline
        shutil.copytree(spec.reference_repo_dir, destination)

    def evaluate(plan_arg, candidate_root: Path, *, timeout_seconds: int):
        return evaluate_time_machine_candidate(
            plan_arg,
            candidate_root,
            timeout_seconds=timeout_seconds,
        )

    artifact = run_final_evaluation(
        world,
        output_dir=tmp_path,
        timeout_seconds=30,
        plan_builder=lambda **_kwargs: plan,
        candidate_evaluator=evaluate,
        candidate_exporter=export_reference,
    )

    assert artifact is not None
    assert FINAL_EVIDENCE_SCHEMA_VERSION == "relic-oss-final-evaluation-v1"
    assert artifact.payload["schema_version"] == FINAL_EVIDENCE_SCHEMA_VERSION
    assert artifact.result["status"] == "passed"
