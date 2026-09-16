from pathlib import Path

import pytest

from relic.paper_results import (
    load_source,
    public_payload,
    render_markdown,
    serialize_json,
)
from relic.paths import project_root


@pytest.mark.release
def test_committed_paper_results_are_reproducible() -> None:
    payload = public_payload(load_source())
    artifact_root = project_root() / "artifacts" / "paper_results"

    assert (artifact_root / "paper_results.json").read_text(encoding="utf-8") == serialize_json(
        payload
    )
    assert (artifact_root / "paper_results.md").read_text(encoding="utf-8") == render_markdown(
        payload
    )


@pytest.mark.unit
def test_programbench_is_reporting_only() -> None:
    payload = public_payload(load_source())
    programbench = payload["external_extensions"]["programbench"]

    assert programbench["reproduction_artifacts_included"] is False
    assert programbench["tasks_at_or_above_95_percent"] == {
        "official_mini_swe_agent": 2,
        "with_executable_protocols": 2,
    }


@pytest.mark.unit
def test_main_study_counts_match_paper() -> None:
    payload = public_payload(load_source(Path(project_root() / "reproduction/main_results/paper_results_source.yaml")))

    assert payload["main_study"]["runs"] == 240
    assert [model["runs"] for model in payload["main_study"]["models"]] == [120, 120]

