from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import relic.evaluation.user_run_batch as batch
from relic.cli import main


def _manifest(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    root = tmp_path / "output"
    manifest = root / "run_manifest.json"
    payload = {
        "schema_version": "relic-run-manifest-v2",
        "plan": {
            "study": "relic-main-v1",
            "model": "gpt-5.6-terra",
            "output_root": str(root),
            "plan_sha256": "plan-hash",
            "source": {
                "repository": "https://github.com/Hongyi-Du/SocioGenesis",
                "branch": "hci-human-seat",
                "commit": "dda36fb563375060ae8d8850300db01eb4695d29",
                "git_tree": "d7276c13312b13d4a030d82c3accc253349d708d",
            },
            "cells": [
                {
                    "cell_id": "cell-a",
                    "model": "gpt-5.6-terra",
                    "workload": "W01",
                    "arm": "B3",
                    "seed": 1401,
                    "cell_spec_sha256": "b" * 64,
                }
            ],
        },
        "runtime": {
            "revision": 0,
            "run_origin": {
                "kind": "user_new_run",
                "instance_id": "a" * 32,
                "created_at": "2026-09-16T00:00:00+00:00",
            },
            "cells": {
                "cell-a": {
                    "status": "completed",
                    "failure_class": None,
                }
            },
        },
    }
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    return manifest, payload


def _verified_cell(status: str = "completed", stage: str = "complete") -> dict[str, Any]:
    return {
        "spec": SimpleNamespace(),
        "binding": {},
        "cell_dir": Path("/safe/cell-a"),
        "status": {"status": status, "stage": stage},
        "scheduler_status": "completed",
        "scheduler_failure_class": None,
        "checkpoint_validation": {
            "final_checkpoint_sidecar_verified": True,
            "final_tick": 120,
            "unpickle_performed_by_batch": False,
        },
    }


@pytest.mark.unit
def test_output_root_uses_only_exact_scheduler_manifest_path(tmp_path: Path) -> None:
    root = tmp_path / "output"
    nested = root / "nested"
    nested.mkdir(parents=True)
    (nested / "run_manifest.json").write_text("{}", encoding="utf-8")

    assert batch.resolve_run_manifest(output_root=root) == (root / "run_manifest.json").resolve()
    with pytest.raises(batch.UserRunBatchError, match="exactly_one_manifest"):
        batch.resolve_run_manifest()


@pytest.mark.unit
def test_invalid_manifest_still_leaves_an_atomic_failure_receipt(tmp_path: Path) -> None:
    root = tmp_path / "output"
    root.mkdir()
    manifest = root / "run_manifest.json"
    manifest.write_text("not-json", encoding="utf-8")

    result = batch.evaluate_user_run_batch(output_root=root)

    receipt = json.loads((root / "evaluation_manifest.json").read_text(encoding="utf-8"))
    assert result.status == "failed"
    assert receipt["paper_snapshot_inputs"] == []
    assert receipt["run_origin"]["historical_author_run_inputs"] == []
    assert receipt["receipt_sha256"] == batch._receipt_hash(receipt)
    assert receipt["failures"][-1]["reason"] == "run_manifest_invalid_json"


@pytest.mark.unit
def test_dry_run_is_serialization_only_and_never_spawns_or_mutates_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, payload = _manifest(tmp_path)
    monkeypatch.setattr(batch, "_validate_manifest", lambda _payload: None)
    monkeypatch.setattr(batch, "_validate_cell_artifacts", lambda **_kwargs: _verified_cell())

    def spawn_forbidden(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("dry-run must not spawn evaluate")

    monkeypatch.setattr(batch.subprocess, "run", spawn_forbidden)
    before = json.loads(manifest.read_text(encoding="utf-8"))["runtime"]

    result = batch.evaluate_user_run_batch(manifest_path=manifest, dry_run=True)

    receipt = json.loads(result.receipt_path.read_text(encoding="utf-8"))
    assert result.status == "dry_run"
    assert result.selected_cells == 1
    assert receipt["evaluations"] == [
        {"cell_id": "cell-a", "status": "dry_run", "return_code": None}
    ]
    assert receipt["scheduler_runtime_unchanged"] is True
    assert json.loads(manifest.read_text(encoding="utf-8"))["runtime"] == before
    assert receipt["trusted_checkpoint_boundary"]["local_trusted_checkpoints_only"] is True
    assert receipt["manifest"]["study"] == "relic-main-v1"
    assert receipt["manifest"]["model"] == "gpt-5.6-terra"
    assert receipt["manifest"]["output_root"] == str((tmp_path / "output").resolve())
    assert receipt["manifest"]["run_origin"]["historical_author_run_inputs"] == []
    assert receipt["manifest"]["run_origin"]["historical_raw_run_inputs"] == []
    assert receipt["manifest"]["plan_cells"] == [
        {
            "cell_id": "cell-a",
            "model": "gpt-5.6-terra",
            "workload": "W01",
            "arm": "B3",
            "seed": 1401,
            "cell_spec_sha256": "b" * 64,
        }
    ]


@pytest.mark.unit
def test_incomplete_run_preselects_before_artifact_access_and_records_public_prestate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A partial scheduler run has no pending-cell directory to inspect."""

    manifest, payload = _manifest(tmp_path)
    pending = {
        "cell_id": "cell-pending",
        "model": "gpt-5.6-terra",
        "workload": "W02",
        "arm": "B3",
        "seed": 1401,
        "cell_spec_sha256": "c" * 64,
    }
    payload["plan"]["cells"].append(pending)
    payload["runtime"]["cells"]["cell-pending"] = {
        "status": "pending",
        "failure_class": None,
    }
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(batch, "_validate_manifest", lambda _payload: None)
    inspected: list[str] = []

    def fake_artifacts(**kwargs: Any) -> dict[str, Any]:
        inspected.append(kwargs["raw_cell"]["cell_id"])
        return _verified_cell()

    monkeypatch.setattr(batch, "_validate_cell_artifacts", fake_artifacts)
    monkeypatch.setattr(
        batch.subprocess, "run", lambda *_args, **_kwargs: SimpleNamespace(returncode=0)
    )
    monkeypatch.setattr(
        batch,
        "_analysis_from_verified_artifacts",
        lambda **_kwargs: {"evaluator": {}},
    )

    result = batch.evaluate_user_run_batch(manifest_path=manifest)

    receipt = json.loads(result.receipt_path.read_text(encoding="utf-8"))
    assert result.status == "completed"
    assert inspected == ["cell-a", "cell-a"]
    pending_row = next(cell for cell in receipt["cells"] if cell["cell_id"] == "cell-pending")
    assert pending_row == {
        **pending,
        "scheduler_status": "pending",
        "scheduler_failure_class": None,
        "outcome": "not_selected",
        "reason_code": "scheduler_cell_not_terminal",
    }
    selected = receipt["selection"]["eligible"]
    assert selected[0]["scheduler_failure_class"] is None
    assert selected[0]["public_pre_status"] == "completed"
    assert selected[0]["public_pre_stage"] == "complete"


@pytest.mark.unit
def test_failed_evaluation_policy_requires_scheduler_and_public_evaluation_state() -> None:
    terminal = {"status": "failed", "failure_class": "evaluator_failure"}
    assert (
        batch._selection_reason(
            policy="failed-evaluation",
            runtime_cell=terminal,
            status={"status": "infra_error", "stage": "evaluation"},
        )
        is None
    )
    assert (
        batch._selection_reason(
            policy="failed-evaluation",
            runtime_cell=terminal,
            status={"status": "completed", "stage": "complete"},
        )
        == "scheduler_public_status_not_eligible"
    )
    assert (
        batch._selection_reason(
            policy="all-eligible",
            runtime_cell={"status": "failed", "failure_class": "model_failure"},
            status={"status": "failed", "stage": "evaluation"},
        )
        == "scheduler_failure_not_evaluator"
    )


@pytest.mark.unit
def test_subprocess_failure_is_recorded_and_later_cells_are_not_needed_for_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, _ = _manifest(tmp_path)
    monkeypatch.setattr(batch, "_validate_manifest", lambda _payload: None)
    monkeypatch.setattr(batch, "_validate_cell_artifacts", lambda **_kwargs: _verified_cell())
    monkeypatch.setattr(
        batch.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=23),
    )

    result = batch.evaluate_user_run_batch(manifest_path=manifest)

    receipt = json.loads(result.receipt_path.read_text(encoding="utf-8"))
    assert result.status == "completed_with_failures"
    assert receipt["failures"] == [{"cell_id": "cell-a", "reason": "evaluator_subprocess_failed"}]
    assert receipt["evaluations"][0]["return_code"] == 23
    assert "stdout" not in receipt["evaluations"][0]
    assert "stderr" not in receipt["evaluations"][0]


@pytest.mark.unit
def test_evaluate_cli_dispatches_manifest_and_preserves_failure_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    captured: dict[str, Any] = {}

    class Result:
        failed_cells = 1

        def to_dict(self) -> dict[str, Any]:
            return {"status": "completed_with_failures"}

    def fake_batch(**kwargs: Any) -> Result:
        captured.update(kwargs)
        return Result()

    monkeypatch.setattr(batch, "evaluate_user_run_batch", fake_batch)
    code = main(
        [
            "evaluate",
            "--manifest",
            str(tmp_path / "run_manifest.json"),
            "--receipt-directory",
            str(tmp_path / "receipts"),
            "--selection",
            "all-eligible",
            "--dry-run",
        ]
    )

    assert code == 2
    assert captured["manifest_path"] == tmp_path / "run_manifest.json"
    assert captured["output_root"] is None
    assert captured["receipt_directory"] == tmp_path / "receipts"
    assert captured["selection"] == "all-eligible"
    assert captured["dry_run"] is True
    assert "completed_with_failures" in capsys.readouterr().out
