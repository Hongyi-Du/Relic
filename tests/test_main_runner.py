from __future__ import annotations

import fcntl
import json
import os
import signal
from pathlib import Path
from typing import Any

import pytest

import relic.main_runner as main_runner
from relic.cell_spec import SOURCE_BRANCH, SOURCE_COMMIT, compile_cell_spec, stable_sha256
from relic.cli import main
from relic.main_runner import MainRunnerError, build_run_manifest, run_main


@pytest.fixture
def unverified_cell_specs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep scheduler-unit tests focused on planning rather than pack hashing."""
    original = main_runner.compile_cell_spec

    def compile_without_pack_verification(**kwargs: Any):
        kwargs["verify_pack"] = False
        return original(**kwargs)

    monkeypatch.setattr(main_runner, "compile_cell_spec", compile_without_pack_verification)


def _write_payload(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _load_payload(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _freeze_run_manifest(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    output_root = tmp_path / "run-output"
    payload = build_run_manifest(
        model="gpt-5.6-terra",
        output_root=output_root,
        max_parallel=2,
    )
    manifest_path = output_root / "run_manifest.json"
    _write_payload(manifest_path, payload)
    return manifest_path, payload


def _cell_id_from_argv(argv: list[str]) -> str:
    def argument(name: str) -> str:
        return argv[argv.index(name) + 1]

    return "__".join(
        (
            argument("--model"),
            argument("--workload").upper(),
            argument("--arm").upper(),
            f"seed{argument('--seed')}",
        )
    )


class _CompletingPopen:
    """A child which creates only the public terminal status the scheduler may read."""

    next_pid = 41000

    def __init__(self, argv: list[str], **kwargs: Any) -> None:
        self.argv = list(argv)
        self.kwargs = kwargs
        self.pid = type(self).next_pid
        type(self).next_pid += 1
        self.returncode: int | None = None

        output_dir = Path(argv[argv.index("--output-dir") + 1])
        public = output_dir / "public"
        public.mkdir(parents=True, exist_ok=True)
        (public / "status.json").write_text(
            json.dumps(
                {
                    "cell_id": _cell_id_from_argv(argv),
                    "status": "completed",
                    "stage": "complete",
                }
            ),
            encoding="utf-8",
        )

    def poll(self) -> int:
        self.returncode = 0
        return self.returncode

    def wait(self) -> int:
        self.returncode = 0
        return self.returncode


@pytest.mark.unit
def test_run_manifest_freezes_all_120_canonical_cellspecs(
    tmp_path: Path, unverified_cell_specs: None
) -> None:
    payload = build_run_manifest(
        model="gpt-5.6-terra",
        output_root=tmp_path,
        max_parallel=2,
    )
    plan = payload["plan"]

    assert payload["schema_version"] == "relic-run-manifest-v2"
    assert plan["source"]["branch"] == SOURCE_BRANCH
    assert plan["source"]["commit"] == SOURCE_COMMIT
    assert payload["runtime"]["run_origin"]["kind"] == "user_new_run"
    assert len(payload["runtime"]["run_origin"]["instance_id"]) == 32
    assert payload["runtime"]["run_origin"]["created_at"]
    assert len(plan["cells"]) == len(payload["runtime"]["cells"]) == 120
    assert plan["plan_sha256"] == stable_sha256(
        {key: value for key, value in plan.items() if key != "plan_sha256"}
    )

    fingerprints: set[str] = set()
    output_paths: set[Path] = set()
    for cell in plan["cells"]:
        spec = compile_cell_spec(
            model=cell["model"],
            workload=cell["workload"],
            arm=cell["arm"],
            seed=cell["seed"],
            output_root=tmp_path,
            verify_pack=False,
        )
        fingerprints.add(cell["cell_spec_sha256"])
        output_paths.add(Path(cell["output_path"]))
        assert cell["cell_id"] == spec.cell_id
        assert cell["cell_spec_sha256"] == spec.fingerprint
        assert cell["cell_spec"] == spec.document()
        assert Path(cell["output_path"]) == spec.cell_dir

    assert len(fingerprints) == len(output_paths) == 120


@pytest.mark.unit
def test_run_manifest_requires_user_new_run_origin(
    tmp_path: Path, unverified_cell_specs: None
) -> None:
    manifest_path, payload = _freeze_run_manifest(tmp_path)
    del payload["runtime"]["run_origin"]
    _write_payload(manifest_path, payload)

    with pytest.raises(MainRunnerError, match="run_manifest_origin_missing"):
        run_main(
            manifest_path=manifest_path,
            output_root=tmp_path / "run-output",
            resume=True,
            dry_run=True,
        )


@pytest.mark.unit
def test_run_main_dry_run_writes_a_plan_without_spawning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, unverified_cell_specs: None
) -> None:
    manifest_path = tmp_path / "run_manifest.json"

    def spawn_forbidden(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("dry run must not start a cell subprocess")

    monkeypatch.setattr(main_runner.subprocess, "Popen", spawn_forbidden)

    result = run_main(
        model="gpt-5.6-terra",
        output_root=tmp_path,
        manifest_path=manifest_path,
        max_parallel=2,
        dry_run=True,
    )

    assert result.dry_run is True
    assert result.status == "planned"
    assert result.selected_cells == 120
    assert _load_payload(manifest_path)["runtime"]["status"] == "planned"


@pytest.mark.unit
def test_concurrent_manifest_lock_fails_closed(
    tmp_path: Path, unverified_cell_specs: None
) -> None:
    manifest_path = tmp_path / "run_manifest.json"
    lock_path = manifest_path.with_name(f".{manifest_path.name}.lock")
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(MainRunnerError, match="run_manifest_locked"):
            run_main(
                model="gpt-5.6-terra",
                output_root=tmp_path,
                manifest_path=manifest_path,
                dry_run=True,
            )
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


@pytest.mark.unit
def test_scheduler_limits_child_count_and_leaves_parent_environment_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, unverified_cell_specs: None
) -> None:
    manifest_path, _ = _freeze_run_manifest(tmp_path)
    invocations: list[_CompletingPopen] = []
    active = 0
    maximum_active = 0
    parent_before = dict(os.environ)

    class TrackingPopen(_CompletingPopen):
        def __init__(self, argv: list[str], **kwargs: Any) -> None:
            nonlocal active, maximum_active
            super().__init__(argv, **kwargs)
            invocations.append(self)
            active += 1
            maximum_active = max(maximum_active, active)

        def poll(self) -> int:
            nonlocal active
            if self.returncode is None:
                active -= 1
            return super().poll()

    monkeypatch.setattr(main_runner.subprocess, "Popen", TrackingPopen)

    result = run_main(
        manifest_path=manifest_path,
        output_root=tmp_path / "run-output",
        max_parallel=2,
        resume=True,
        poll_interval=0,
    )

    assert result.status == "complete"
    assert result.completed_cells == 120
    assert len(invocations) == 120
    assert maximum_active <= 2
    assert dict(os.environ) == parent_before
    assert all(call.kwargs["env"] == parent_before for call in invocations)
    assert all(call.kwargs["env"] is not os.environ for call in invocations)
    assert len({id(call.kwargs["env"]) for call in invocations}) == len(invocations)
    assert all(call.kwargs["cwd"] == main_runner.project_root() for call in invocations)
    assert all(call.kwargs["start_new_session"] is True for call in invocations)


@pytest.mark.unit
def test_resume_skips_completed_cells_and_retry_selects_only_retryable_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, unverified_cell_specs: None
) -> None:
    manifest_path, payload = _freeze_run_manifest(tmp_path)
    states = payload["runtime"]["cells"]
    cells = payload["plan"]["cells"]
    completed_id = cells[0]["cell_id"]
    retryable_ids = [cell["cell_id"] for cell in cells[1:4]]
    non_retryable_id = cells[4]["cell_id"]
    interrupted_id = cells[5]["cell_id"]
    states[completed_id]["status"] = "completed"
    for cell_id, failure_class in zip(
        retryable_ids,
        ("model_failure", "evaluator_failure", "infrastructure_failure"),
        strict=True,
    ):
        states[cell_id].update(status="failed", failure_class=failure_class)
    states[non_retryable_id].update(status="failed", failure_class="interrupted")
    states[interrupted_id].update(status="interrupted", failure_class="interrupted")
    _write_payload(manifest_path, payload)

    def spawn_forbidden(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("dry selection must not spawn")

    monkeypatch.setattr(main_runner.subprocess, "Popen", spawn_forbidden)
    resumed = run_main(
        manifest_path=manifest_path,
        output_root=tmp_path / "run-output",
        resume=True,
        dry_run=True,
    )
    retried = run_main(
        manifest_path=manifest_path,
        output_root=tmp_path / "run-output",
        resume=True,
        retry_failed=True,
        dry_run=True,
    )
    narrowed = run_main(
        manifest_path=manifest_path,
        output_root=tmp_path / "run-output",
        resume=True,
        retry_failed=True,
        cell_id=retryable_ids[0],
        dry_run=True,
    )

    assert resumed.selected_cells == 115
    assert retried.selected_cells == 3
    assert narrowed.selected_cells == 1

    for cell in cells:
        states[cell["cell_id"]]["status"] = "completed"
    _write_payload(manifest_path, payload)
    no_completed_rerun = run_main(
        manifest_path=manifest_path,
        output_root=tmp_path / "run-output",
        resume=True,
        poll_interval=0,
    )
    assert no_completed_rerun.selected_cells == 0
    assert no_completed_rerun.completed_cells == 120


@pytest.mark.unit
def test_retry_uses_worker_resume_flag_when_cell_artifacts_exist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, unverified_cell_specs: None
) -> None:
    manifest_path, payload = _freeze_run_manifest(tmp_path)
    retry_cell = payload["plan"]["cells"][0]
    retry_id = retry_cell["cell_id"]
    payload["runtime"]["cells"][retry_id].update(
        status="failed", failure_class="evaluator_failure", failure_code="evaluator_timeout"
    )
    cell_dir = Path(retry_cell["output_path"])
    cell_dir.mkdir(parents=True)
    (cell_dir / "cell-spec.json").write_text("{}", encoding="utf-8")
    _write_payload(manifest_path, payload)
    invocations: list[_CompletingPopen] = []

    class CapturingPopen(_CompletingPopen):
        def __init__(self, argv: list[str], **kwargs: Any) -> None:
            super().__init__(argv, **kwargs)
            invocations.append(self)

    monkeypatch.setattr(main_runner.subprocess, "Popen", CapturingPopen)
    result = run_main(
        manifest_path=manifest_path,
        output_root=tmp_path / "run-output",
        resume=True,
        retry_failed=True,
        cell_id=retry_id,
        poll_interval=0,
    )

    assert result.selected_cells == 1
    assert len(invocations) == 1
    assert "--resume" in invocations[0].argv


@pytest.mark.unit
def test_tampered_frozen_manifest_is_rejected_before_any_spawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, unverified_cell_specs: None
) -> None:
    manifest_path, payload = _freeze_run_manifest(tmp_path)
    plan = payload["plan"]
    plan["cells"][0]["cell_spec_sha256"] = "0" * 64
    plan["plan_sha256"] = stable_sha256(
        {key: value for key, value in plan.items() if key != "plan_sha256"}
    )
    _write_payload(manifest_path, payload)

    def spawn_forbidden(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("tampered plans must fail before spawning a child")

    monkeypatch.setattr(main_runner.subprocess, "Popen", spawn_forbidden)
    with pytest.raises(MainRunnerError, match="run_manifest_cell_spec_hash_mismatch"):
        run_main(
            manifest_path=manifest_path,
            output_root=tmp_path / "run-output",
            resume=True,
        )


@pytest.mark.unit
def test_post_spawn_manifest_failure_still_terminates_registered_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, unverified_cell_specs: None
) -> None:
    manifest_path, _ = _freeze_run_manifest(tmp_path)
    spawned: list[Any] = []
    terminated: list[set[str]] = []

    class LivePopen:
        def __init__(self, argv: list[str], **kwargs: Any) -> None:
            self.argv = argv
            self.kwargs = kwargs
            self.pid = 42000
            self.returncode = None
            spawned.append(self)

        def poll(self) -> None:
            return None

    original_bump = main_runner._bump_and_write
    writes = 0

    def fail_first_post_spawn_write(payload: dict[str, Any], path: Path) -> None:
        nonlocal writes
        writes += 1
        if writes == 3:
            raise OSError("simulated post-spawn persistence failure")
        original_bump(payload, path)

    def terminate(active: dict[str, Any], _grace_seconds: float) -> None:
        terminated.append(set(active))
        for child in active.values():
            child.process.returncode = -2

    monkeypatch.setattr(main_runner.subprocess, "Popen", LivePopen)
    monkeypatch.setattr(main_runner, "_bump_and_write", fail_first_post_spawn_write)
    monkeypatch.setattr(main_runner, "_terminate_children", terminate)

    with pytest.raises(MainRunnerError, match="scheduler_runtime_failure"):
        run_main(
            manifest_path=manifest_path,
            output_root=tmp_path / "run-output",
            resume=True,
            max_parallel=1,
            poll_interval=0,
        )

    assert len(spawned) == 1
    assert len(terminated) == 1
    assert len(terminated[0]) == 1


@pytest.mark.unit
def test_sigint_during_popen_registration_cannot_orphan_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, unverified_cell_specs: None
) -> None:
    manifest_path, _ = _freeze_run_manifest(tmp_path)
    terminated: list[set[str]] = []

    class InterruptingPopen:
        def __init__(self, _argv: list[str], **_kwargs: Any) -> None:
            self.pid = 43000
            self.returncode = None
            handler = signal.getsignal(signal.SIGINT)
            assert callable(handler)
            handler(signal.SIGINT, None)

        def poll(self) -> int | None:
            return self.returncode

    def terminate(active: dict[str, Any], _grace_seconds: float) -> None:
        terminated.append(set(active))
        for child in active.values():
            child.process.returncode = -2

    monkeypatch.setattr(main_runner.subprocess, "Popen", InterruptingPopen)
    monkeypatch.setattr(main_runner, "_terminate_children", terminate)

    with pytest.raises(KeyboardInterrupt):
        run_main(
            manifest_path=manifest_path,
            output_root=tmp_path / "run-output",
            resume=True,
            max_parallel=1,
            poll_interval=0,
        )

    assert len(terminated) == 1
    assert len(terminated[0]) == 1


@pytest.mark.unit
def test_process_group_cleanup_escalates_signals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class StubbornProcess:
        pid = 44000
        returncode = None

        def poll(self) -> None:
            return None

        def wait(self) -> int:
            self.returncode = -signal.SIGKILL
            return self.returncode

    stdout = (tmp_path / "stdout.log").open("w", encoding="utf-8")
    stderr = (tmp_path / "stderr.log").open("w", encoding="utf-8")
    child = main_runner._ActiveChild(
        "cell", StubbornProcess(), stdout, stderr, "start"
    )
    delivered: list[signal.Signals] = []

    def record_signal(_pid: int, sent: signal.Signals) -> None:
        delivered.append(sent)

    monkeypatch.setattr(main_runner.os, "killpg", record_signal)
    try:
        main_runner._terminate_children({"cell": child}, grace_seconds=0)
    finally:
        stdout.close()
        stderr.close()

    assert delivered == [signal.SIGINT, signal.SIGTERM, signal.SIGKILL]
    assert child.process.returncode == -signal.SIGKILL


@pytest.mark.unit
def test_claude_actual_run_fails_fast_but_dry_run_remains_safe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, unverified_cell_specs: None
) -> None:
    manifest_path = tmp_path / "claude_manifest.json"

    def spawn_forbidden(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("unsupported Claude runs must fail before child creation")

    monkeypatch.setattr(main_runner.subprocess, "Popen", spawn_forbidden)
    dry_result = run_main(
        model="claude-opus-4.6",
        output_root=tmp_path,
        manifest_path=manifest_path,
        dry_run=True,
    )
    assert dry_result.status == "planned"

    with pytest.raises(MainRunnerError, match="unsupported_model_provider:anthropic"):
        run_main(
            manifest_path=manifest_path,
            output_root=tmp_path,
            resume=True,
        )


@pytest.mark.unit
def test_cli_exposes_run_main_scheduler_options(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as stopped:
        main(["run-main", "--help"])

    assert stopped.value.code == 0
    help_text = capsys.readouterr().out
    assert "--dry-run" in help_text
    assert "--resume" in help_text
    assert "--retry-failed" in help_text
    assert "--cell-id" in help_text


@pytest.mark.unit
def test_cli_run_main_dry_run_uses_scheduler_without_spawning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    unverified_cell_specs: None,
) -> None:
    manifest_path = tmp_path / "cli_run_manifest.json"

    def spawn_forbidden(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("CLI dry run must not start a cell subprocess")

    monkeypatch.setattr(main_runner.subprocess, "Popen", spawn_forbidden)

    assert (
        main(
            [
                "run-main",
                "--model",
                "gpt-5.6-terra",
                "--output-root",
                str(tmp_path),
                "--manifest",
                str(manifest_path),
                "--dry-run",
            ]
        )
        == 0
    )
    rendered = capsys.readouterr().out
    assert '"dry_run": true' in rendered
    assert "Dry run only" in rendered
    assert _load_payload(manifest_path)["runtime"]["status"] == "planned"
