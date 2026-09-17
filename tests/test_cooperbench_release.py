"""Release boundary tests for the source-retained CooperBench paper subset."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from relic.cli import main
from relic.cooper_release import (
    EXPECTED_PAIR_COUNT,
    EXPECTED_REPOSITORY_COUNT,
    EXPECTED_TASK_COUNT,
    PAPER_SUBSET,
    SOURCE_BATCHES,
    CooperReleaseError,
    missing_input_report,
    public_preflight_command,
    read_upstream_summary,
    source_selection,
    upstream_eval_command,
    upstream_run_command,
    verify_upstream_subset,
)


def _subset_document() -> dict[str, object]:
    selection = source_selection()
    grouped: dict[tuple[str, int], list[list[int]]] = {}
    for pair in selection.pairs:
        grouped.setdefault((pair.repo, pair.task_id), []).append(list(pair.features))
    return {
        "tasks": [
            {"repo": repo, "task_id": task_id, "pairs": pairs}
            for (repo, task_id), pairs in grouped.items()
        ]
    }


def _write_subset(tmp_path: Path, document: dict[str, object]) -> Path:
    dataset = tmp_path / "dataset"
    subset = dataset / "subsets" / f"{PAPER_SUBSET}.json"
    subset.parent.mkdir(parents=True)
    subset.write_text(json.dumps(document), encoding="utf-8")
    return dataset


def test_source_selection_is_the_verbatim_16_plus_32_union() -> None:
    selection = source_selection()

    assert len(selection.pairs) == EXPECTED_PAIR_COUNT == 48
    assert len({(pair.repo, pair.task_id) for pair in selection.pairs}) == EXPECTED_TASK_COUNT == 30
    assert len({pair.repo for pair in selection.pairs}) == EXPECTED_REPOSITORY_COUNT == 12
    assert len(selection.keys) == 48
    assert "openai_tiktoken_task:0:1,5" in selection.keys
    assert [batch.path.name for batch in SOURCE_BATCHES] == [
        "b3_v108_new16_b001.json",
        "b3_v128_expand32_b001.json",
    ]
    assert all(hashlib.sha256(batch.path.read_bytes()).hexdigest() == batch.sha256 for batch in SOURCE_BATCHES)


def test_external_combined_subset_must_equal_source_union(tmp_path: Path) -> None:
    selection = source_selection()
    dataset = _write_subset(tmp_path, _subset_document())

    path = verify_upstream_subset(selection, dataset)

    assert path == dataset / "subsets" / f"{PAPER_SUBSET}.json"


def test_external_subset_cannot_be_a_smaller_or_different_batch(tmp_path: Path) -> None:
    selection = source_selection()
    document = _subset_document()
    tasks = document["tasks"]
    assert isinstance(tasks, list)
    tasks.pop()
    dataset = _write_subset(tmp_path, document)

    with pytest.raises(
        CooperReleaseError, match="^cooperbench_paper_subset_pair_count_mismatch$"
    ):
        verify_upstream_subset(selection, dataset)


def test_upstream_commands_are_source_subset_only_and_never_force(tmp_path: Path) -> None:
    run = upstream_run_command(
        cooperbench_binary="cooperbench",
        dataset_dir=tmp_path / "dataset",
        log_dir=tmp_path / "logs",
        run_name="relic-paper48-r001",
        model_name="gateway-opus-alias",
        concurrency=2,
        eval_concurrency=1,
        redis_url="redis://localhost:6379",
        agent_config=tmp_path / "b3_two_agent_case.yaml",
    )
    evaluate = upstream_eval_command(
        cooperbench_binary="cooperbench",
        dataset_dir=tmp_path / "dataset",
        log_dir=tmp_path / "logs",
        run_name="relic-paper48-r001",
        concurrency=2,
    )

    assert run[run.index("-s") + 1] == evaluate[evaluate.index("-s") + 1] == PAPER_SUBSET
    assert run[run.index("-a") + 1] == "orgenv_b3_two_agent"
    assert run[run.index("-m") + 1] == "gateway-opus-alias"
    assert "--force" not in run + evaluate


def test_public_preflight_accepts_only_source_selection_and_preserves_source_limit(
    tmp_path: Path,
) -> None:
    selection = source_selection()
    command = public_preflight_command(
        selection=selection,
        pair_key="dottxt_ai_outlines_task:1371:1,2",
        image="example/task:tag",
        dataset_dir=tmp_path / "dataset",
        output=tmp_path / "output",
        config=tmp_path / "smoke.yaml",
    )
    assert command[command.index("--repo") + 1] == "dottxt_ai_outlines_task"
    assert command[command.index("--features") + 1] == "1,2"

    with pytest.raises(
        CooperReleaseError, match="^cooperbench_public_preflight_task_id_zero_unsupported$"
    ):
        public_preflight_command(
            selection=selection,
            pair_key="openai_tiktoken_task:0:1,5",
            image="example/task:tag",
            dataset_dir=tmp_path / "dataset",
            output=tmp_path / "output",
            config=tmp_path / "smoke.yaml",
        )


def test_summary_is_emitted_as_upstream_bytes_without_reaggregation(tmp_path: Path) -> None:
    raw = b'{\n  "completed": 48, "successful": 29\n}\n'
    summary = tmp_path / "logs" / "run1" / "summary.json"
    summary.parent.mkdir(parents=True)
    summary.write_bytes(raw)

    assert read_upstream_summary(tmp_path / "logs", "run1") == raw


def test_missing_report_recognizes_retained_source_selection() -> None:
    report = missing_input_report()

    assert report["source_selection"]["pairs"] == 48
    assert "paper_48_manifest_missing" not in report["runtime_missing"]
    assert report["runtime_missing"] == [
        "cooperbench_checkout_required",
        "cooperbench_cli_required",
        "cooperbench_dataset_dir_required",
    ]
    assert "paper_48_historical_raw_artifacts_missing" in report["release_gaps"]


def test_cli_check_allows_new_runs_without_historical_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import relic.cooper_release as cooper_release

    dataset = _write_subset(tmp_path, _subset_document())
    monkeypatch.setattr(cooper_release, "verify_cooperbench_checkout", lambda _: tmp_path)
    monkeypatch.setattr(cooper_release, "resolve_cooperbench_binary", lambda _: "cooperbench")
    assert main([
        "check-cooper", "--cooperbench-root", str(tmp_path),
        "--cooperbench-bin", "cooperbench", "--dataset-dir", str(dataset),
    ]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["runtime_ready"] is True
    assert report["runtime_missing"] == []
    assert report["source_selection"]["pairs"] == 48
    assert "paper_48_historical_raw_artifacts_missing" in report["release_gaps"]
    assert "paper_48_historical_task_image_digest_ledger_missing" in report["release_gaps"]


def test_cli_dry_run_uses_source_subset_and_absolute_default_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import relic.cooper_release as cooper_release

    monkeypatch.setattr(cooper_release, "verify_cooperbench_checkout", lambda _: tmp_path)
    monkeypatch.setattr(cooper_release, "verify_upstream_subset", lambda *_: tmp_path / "subset.json")
    monkeypatch.setattr(cooper_release, "resolve_cooperbench_binary", lambda _: "cooperbench")

    assert (
        main(
            [
                "run-cooper",
                "--cooperbench-root",
                str(tmp_path),
                "--dataset-dir",
                str(tmp_path / "dataset"),
                "--log-dir",
                str(tmp_path / "logs"),
                "--run-name",
                "relic-paper48-r001",
                "--model",
                "gateway-opus-alias",
                "--dry-run",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    command = payload["command"]
    assert command[command.index("-s") + 1] == PAPER_SUBSET
    assert command[command.index("--agent-config") + 1].endswith(
        "/configs/cooperbench/b3_two_agent_case.yaml"
    )
    assert "--force" not in command
