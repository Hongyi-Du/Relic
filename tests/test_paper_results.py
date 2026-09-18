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
    assert programbench["relative_gain_percent"] == 10.5
    assert programbench["sdl_enabled"] is False


@pytest.mark.unit
def test_main_study_counts_match_paper() -> None:
    payload = public_payload(load_source(Path(project_root() / "reproduction/main_results/paper_results_source.yaml")))

    assert payload["main_study"]["runs"] == 240
    assert [model["runs"] for model in payload["main_study"]["models"]] == [120, 120]
    assert payload["main_study"]["checkpoint_every"] == 24
    assert payload["main_study"]["sprint_ticks"] == 168
    assert payload["main_study"]["resource_policy"]["matched_hard_budget"] is False
    assert payload["main_study"]["workload_inventory"][0]["scoring_units"] == {
        "seeded": 5,
        "exposed": 35,
        "held_out": 0,
        "contracts": 5,
        "cases": 35,
    }
    assert payload["protocol_census"]["weak_or_strong_formed_lineages"] == 280
    assert payload["binding_ablation"]["executable_minus_text"]["bootstrap_replicates"] == 200000


@pytest.mark.unit
def test_corrected_complete_transfer_summary_matches_reviewed_table() -> None:
    transfer = public_payload(load_source())["internal_transfer"]
    endpoints = {row["id"]: row["arms"] for row in transfer["secondary_endpoint_metrics"]}
    assert transfer["target_runs"] == 60
    assert transfer["fresh_reference_runs_reused"] == 30
    assert endpoints["exposed_case_pass_rate"] == {"Fresh": 23.434, "Text": 39.7, "Exec": 44.4}
    assert endpoints["held_out_case_pass_rate"] == {"Fresh": 11.111, "Text": 11.1, "Exec": 25.9}
    assert endpoints["complete_contract_rate"] == {"Fresh": 18.574, "Text": 31.5, "Exec": 32.3}
    assert endpoints["workspace_behavioral_case_rate"] == {"Fresh": None, "Text": 44.4, "Exec": 51.0}
    assert endpoints["evaluator_confirmed_issue_rate"] == {"Fresh": 21.72, "Text": 41.4, "Exec": 45.1}

    process = {row["id"]: row["arms"] for row in transfer["process_metrics"]}
    assert len(process) == 11
    assert process["patches_generated_accepted"]["Exec"] == {"generated": 118.0, "accepted": 116.7}
    assert process["recorded_uses"]["Exec"] == 588.0
    assert process["recorded_binding_enforcements"]["Exec"] == 197.0

    markdown = render_markdown(public_payload(load_source()))
    assert "| Complete-contract rate | 18.574% | 31.5% | 32.3% |" in markdown
    assert "| Evaluator-confirmed issue rate | 21.720% | 41.4% | 45.1% |" in markdown
    assert "| Workspace behavioral-case rate | NA | 44.4% | 51.0% |" in markdown
