from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from relic.cell_spec import (
    SOURCE_BRANCH,
    SOURCE_COMMIT,
    SOURCE_GIT_TREE,
    SOURCE_REPOSITORY,
    compile_cell_spec,
    stable_sha256,
)
from relic.cli import main
import relic.evaluation.user_run_aggregate as aggregate_module
from relic.evaluation.user_run_aggregate import (
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    UserRunAggregateError,
    build_user_run_aggregate,
)


def _lineage(workload: str, model: str) -> dict[str, str]:
    return {
        "source_provenance_sha256": "a" * 64,
        "runtime_tree_sha256": "b" * 64,
        "model_binding_sha256": stable_sha256({"model": model}),
        "execution_policy_sha256": "d" * 64,
        "qualification_plan_sha256": stable_sha256({"qualification": workload}),
        "evaluator_environment_sha256": stable_sha256({"environment": workload}),
        "starter_repo_digest": stable_sha256({"starter": workload}),
        "reference_repo_digest": stable_sha256({"reference": workload}),
        "hidden_suite_hash": stable_sha256({"hidden": workload}),
    }


def _cell(
    workload: str,
    arm: str,
    seed: int,
    *,
    pass_rate: float,
    tokens: int,
    fixes: int,
    model: str = "gpt-5.6-terra",
    evaluator_status: str = "passed",
    infrastructure_error_count: int = 0,
) -> dict:
    cell_id = f"{model}__{workload}__{arm}__seed{seed}"
    evaluator = {
        "artifact_hash": stable_sha256({"artifact": cell_id}),
        "result_hash": stable_sha256({"result": cell_id}),
        "candidate_repo_digest": stable_sha256({"candidate": cell_id}),
        "status": evaluator_status,
    }
    return {
        "cell_id": cell_id,
        "model": model,
        "workload": workload,
        "arm": arm,
        "seed": seed,
        "scheduler_status": "completed",
        "outcome": "evaluated",
        "cell_spec_sha256": stable_sha256({"cell": cell_id}),
        "reason_code": None,
        "evaluator": evaluator,
        "analysis": {
            "model": model,
            "workload": workload,
            "arm": arm,
            "seed": seed,
            "dataset_id": workload.lower(),
            "run_record_status": "completed",
            "llm_usage": {"total_tokens": tokens},
            "final_evaluation": {
                "status": evaluator_status,
                "candidate_pass_rate": pass_rate,
                "causal_fix_count": fixes,
                "causal_fix_rate": pass_rate,
                "unresolved_count": 0,
                "regression_count": 0,
                "infrastructure_error_count": infrastructure_error_count,
                "formal_claim_ready": True,
            },
            "lineage": _lineage(workload, model),
            "evaluator": evaluator,
            "missing_reasons": {},
        },
    }


def _plan_cell(model: str, workload: str, arm: str, seed: int) -> dict:
    cell_id = f"{model}__{workload}__{arm}__seed{seed}"
    return {
        "cell_id": cell_id,
        "model": model,
        "workload": workload,
        "arm": arm,
        "seed": seed,
        "cell_spec_sha256": stable_sha256({"cell": cell_id}),
    }


def _receipt(
    path: Path,
    cells: list[dict],
    *,
    model: str = "gpt-5.6-terra",
    status: str = "completed",
    dry_run: bool = False,
    scheduler_runtime_unchanged: bool = True,
) -> Path:
    output_root = (path.parent / f"run-output-{model}").resolve()
    specs = [
        compile_cell_spec(
            model=model,
            workload=workload,
            arm=arm,
            seed=seed,
            output_root=output_root,
            verify_pack=False,
        )
        for workload in (f"W{index:02d}" for index in range(1, 11))
        for arm in ("B0", "B1", "B2", "B3")
        for seed in (1401, 2711, 4013)
    ]
    planned = [
        {
            "cell_id": spec.cell_id,
            "model": spec.model,
            "workload": spec.workload.upper(),
            "arm": spec.arm.upper(),
            "seed": spec.seed,
            "cell_spec_sha256": spec.fingerprint,
        }
        for spec in specs
    ]
    specs_by_id = {spec.cell_id: spec for spec in specs}
    selected = {str(cell["cell_id"]): cell for cell in cells}
    assert set(selected).issubset({str(cell["cell_id"]) for cell in planned})
    receipt_cells: list[dict] = []
    excluded: list[dict] = []
    eligible: list[dict] = []
    for plan_cell in planned:
        cell_id = str(plan_cell["cell_id"])
        if cell_id in selected:
            spec = specs_by_id[cell_id]
            cell = deepcopy(selected[cell_id])
            cell["cell_spec_sha256"] = spec.fingerprint
            cell["scheduler_status"] = "completed"
            cell["scheduler_failure_class"] = None
            cell["public_pre_status"] = "completed"
            cell["public_pre_stage"] = "complete"
            cell["analysis"]["dataset_id"] = spec.dataset_id
            cell["analysis"]["lineage"].update(
                {
                    "source_provenance_sha256": spec.source_provenance_fingerprint,
                    "runtime_tree_sha256": spec.runtime_source["tree_sha256"],
                    "model_binding_sha256": spec.model_binding_fingerprint,
                }
            )
            receipt_cells.append(cell)
            eligible.append(
                {
                    "cell_id": cell_id,
                    "cell_dir": str(spec.cell_dir),
                    "scheduler_status": "completed",
                    "scheduler_failure_class": None,
                    "public_pre_status": "completed",
                    "public_pre_stage": "complete",
                    "checkpoint_validation": {
                        "final_checkpoint_sidecar_verified": True,
                        "final_tick": spec.ticks,
                        "unpickle_performed_by_batch": False,
                    },
                }
            )
        else:
            reason = "scheduler_cell_not_terminal"
            receipt_cells.append(
                {
                    **plan_cell,
                    "scheduler_status": "pending",
                    "scheduler_failure_class": None,
                    "outcome": "not_selected",
                    "reason_code": reason,
                }
            )
            excluded.append({"cell_id": cell_id, "reason": reason})
    snapshot_sha256 = "8" * 64
    origin = {
        "kind": "user_new_run",
        "instance_id": "4" * 32,
        "created_at": "2026-09-16T00:00:00+00:00",
        "historical_author_run_inputs": [],
        "historical_raw_run_inputs": [],
    }
    source = {
        "repository": SOURCE_REPOSITORY,
        "branch": SOURCE_BRANCH,
        "commit": SOURCE_COMMIT,
        "git_tree": SOURCE_GIT_TREE,
    }
    plan = {
        "study": "relic-main-v1",
        "model": model,
        "output_root": str(output_root),
        "source": source,
        "cells": [
            {
                "cell_id": spec.cell_id,
                "model": spec.model,
                "workload": spec.workload.upper(),
                "arm": spec.arm.upper(),
                "seed": spec.seed,
                "output_path": str(spec.cell_dir),
                "cell_spec_sha256": spec.fingerprint,
                "cell_spec": spec.document(),
            }
            for spec in specs
        ],
    }
    plan_sha256 = stable_sha256(plan)
    payload = {
        "schema_version": "relic-user-run-evaluation-batch-v1",
        "created_at": "2026-09-16T00:00:00+00:00",
        "completed_at": "2026-09-16T00:01:00+00:00",
        "status": status,
        "dry_run": dry_run,
        "scheduler_runtime_unchanged": scheduler_runtime_unchanged,
        "scheduler_runtime_before_sha256": "7" * 64,
        "scheduler_runtime_after_sha256": "7" * 64,
        "scheduler_runtime_mutation": "batch_never_writes_scheduler_runtime",
        "paper_snapshot_inputs": [],
        "failures": [],
        "run_origin": {**origin, "manifest_sha256": snapshot_sha256},
        "trusted_checkpoint_boundary": {
            "local_trusted_checkpoints_only": True,
            "warning": "test trusted local checkpoints only",
        },
        "source_snapshot": source,
        "manifest_snapshot": {
            "path": str((path.parent / "run_manifest.json").resolve()),
            "sha256": snapshot_sha256,
            "schema_version": "relic-run-manifest-v2",
            "plan_sha256": plan_sha256,
        },
        "manifest": {
            "plan_sha256": plan_sha256,
            "schema_version": "relic-run-manifest-v2",
            "study": "relic-main-v1",
            "model": model,
            "output_root": str(output_root),
            "manifest_snapshot_sha256": snapshot_sha256,
            "runtime_revision": 0,
            "run_origin": origin,
            "upstream_source": source,
            "plan_cells": planned,
        },
        "selection": {
            "policy": "completed",
            "eligible": eligible,
            "excluded": excluded,
        },
        "evaluations": [
            {
                "cell_id": cell["cell_id"],
                "status": "completed",
                "return_code": 0,
                "analysis": cell["analysis"],
            }
            for cell in receipt_cells
            if cell["outcome"] == "evaluated"
        ],
        "cells": receipt_cells,
    }
    payload["receipt_sha256"] = stable_sha256(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _rewrite_receipt(path: Path, mutate) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    payload["receipt_sha256"] = stable_sha256(
        {key: value for key, value in payload.items() if key != "receipt_sha256"}
    )
    path.write_text(json.dumps(payload), encoding="utf-8")


def _complete_model_cells(model: str) -> list[dict]:
    return [
        _cell(
            workload,
            arm,
            seed,
            pass_rate=1.0,
            tokens=100,
            fixes=1,
            model=model,
        )
        for workload in (f"W{index:02d}" for index in range(1, 11))
        for arm in ("B0", "B1", "B2", "B3")
        for seed in (1401, 2711, 4013)
    ]


def test_partial_aggregate_uses_seed_then_block_macro_and_never_fills_paper_metrics(
    tmp_path: Path,
) -> None:
    cells: list[dict] = []
    # W01 contributes 0%; W02 contributes 100%. The block-macro result is 50%
    # regardless of different provider-token magnitudes.
    for seed in (1401, 2711, 4013):
        cells.extend(
            (
                _cell("W01", "B2", seed, pass_rate=0.0, tokens=100, fixes=0),
                _cell("W01", "B3", seed, pass_rate=0.1, tokens=120, fixes=1),
                _cell("W02", "B2", seed, pass_rate=1.0, tokens=1000, fixes=2),
                _cell("W02", "B3", seed, pass_rate=0.9, tokens=900, fixes=3),
            )
        )
    receipt = _receipt(tmp_path / "evaluation_manifest.json", cells)
    result = build_user_run_aggregate(
        [receipt], output_directory=tmp_path / "aggregate", allow_partial=True
    )
    payload = json.loads(result.json_path.read_text(encoding="utf-8"))

    candidate = payload["results"]["available_metrics"]["generic_candidate_pass_rate_percent"]
    assert candidate["by_arm"]["B2"]["value"] == pytest.approx(50.0)
    assert candidate["by_arm"]["B3"]["value"] == pytest.approx(50.0)
    assert candidate["b3_minus_b2"]["b3_minus_b2"] == pytest.approx(0.0)
    assert candidate["b3_minus_b2"]["eligible_blocks"] == 2
    assert candidate["b3_minus_b2"]["paired_cells"] == 6
    tokens = payload["results"]["available_metrics"]["average_tokens_per_run"]
    assert tokens["paper_metric"] is True
    assert tokens["unit"] == "million_tokens"

    paper = payload["results"]["paper_metrics_unavailable"]
    assert paper["complete_contracts"]["value"] is None
    assert paper["held_out_cases"]["workload_exception"] == {
        "W01": "inapplicable_no_held_out_scoring_units"
    }
    assert payload["paper_snapshot"] == {"mixed": False, "used_as_input": False}
    assert payload["aggregation"]["bootstrap"]["draws"] == BOOTSTRAP_DRAWS
    assert payload["aggregation"]["bootstrap"]["rng_seed"] == BOOTSTRAP_SEED
    unhashed = dict(payload)
    declared = unhashed.pop("aggregate_sha256")
    assert declared == stable_sha256(unhashed)
    assert result.markdown_path.read_text(encoding="utf-8").startswith(
        "# User-run aggregate — not paper results"
    )


def test_incomplete_design_requires_explicit_allow_partial(tmp_path: Path) -> None:
    receipt = _receipt(
        tmp_path / "evaluation_manifest.json",
        [_cell("W01", "B3", 1401, pass_rate=0.0, tokens=0, fixes=0)],
    )
    with pytest.raises(UserRunAggregateError, match="incomplete_user_run_design_use_allow_partial"):
        build_user_run_aggregate([receipt], output_directory=tmp_path / "aggregate")


def test_missing_pair_and_zero_fix_are_na_not_zero(tmp_path: Path) -> None:
    cells = [
        _cell("W01", "B2", 1401, pass_rate=0.0, tokens=50, fixes=0),
        _cell("W01", "B3", 1401, pass_rate=0.0, tokens=75, fixes=0),
        _cell("W02", "B3", 1401, pass_rate=0.5, tokens=100, fixes=1),
    ]
    receipt = _receipt(tmp_path / "evaluation_manifest.json", cells)
    result = build_user_run_aggregate(
        [receipt], output_directory=tmp_path / "aggregate", allow_partial=True
    )
    payload = json.loads(result.json_path.read_text(encoding="utf-8"))
    generic_cost = payload["results"]["available_metrics"]["generic_tokens_per_causal_fix_outcome"]
    b2_block = generic_cost["by_arm"]["B2"]["blocks"][0]
    assert b2_block["value"] is None
    assert b2_block["availability"] == "zero_denominator"
    contrast = payload["results"]["available_metrics"]["generic_candidate_pass_rate_percent"][
        "b3_minus_b2"
    ]
    assert contrast["paired_cells"] == 1
    assert contrast["excluded_pairs"]["missing_counterpart"] > 0


def test_rejects_paper_snapshot_input_and_duplicate_cells(tmp_path: Path) -> None:
    cell = _cell("W01", "B3", 1401, pass_rate=0.0, tokens=1, fixes=0)
    receipt = _receipt(tmp_path / "evaluation_manifest.json", [cell])
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload["paper_snapshot_inputs"] = ["artifacts/paper_results/paper_results.json"]
    payload["receipt_sha256"] = stable_sha256(
        {key: value for key, value in payload.items() if key != "receipt_sha256"}
    )
    receipt.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(UserRunAggregateError, match="paper_snapshot_input_forbidden"):
        build_user_run_aggregate(
            [receipt], output_directory=tmp_path / "aggregate", allow_partial=True
        )

    first = _receipt(tmp_path / "first.json", [cell])
    second = _receipt(tmp_path / "second.json", [cell])
    with pytest.raises(UserRunAggregateError, match="duplicate_evaluation_cell"):
        build_user_run_aggregate(
            [first, second], output_directory=tmp_path / "aggregate", allow_partial=True
        )


def test_receipt_allowlists_reject_extra_raw_fields_even_when_self_hashed(tmp_path: Path) -> None:
    receipt = _receipt(
        tmp_path / "evaluation_manifest.json",
        [_cell("W01", "B3", 1401, pass_rate=1.0, tokens=1, fixes=1)],
    )
    _rewrite_receipt(receipt, lambda payload: payload.update(raw_outcomes=[{"secret": "no"}]))
    with pytest.raises(UserRunAggregateError, match="evaluation_manifest_top_level_fields_invalid"):
        build_user_run_aggregate(
            [receipt], output_directory=tmp_path / "aggregate", allow_partial=True
        )

    receipt = _receipt(
        tmp_path / "second.json",
        [_cell("W01", "B3", 1401, pass_rate=1.0, tokens=1, fixes=1)],
    )
    _rewrite_receipt(
        receipt,
        lambda payload: next(cell for cell in payload["cells"] if cell["outcome"] == "evaluated")[
            "analysis"
        ].update(historical_rows=[]),
    )
    with pytest.raises(UserRunAggregateError, match="evaluation_manifest_analysis_fields_invalid"):
        build_user_run_aggregate(
            [receipt], output_directory=tmp_path / "second-aggregate", allow_partial=True
        )


def test_receipt_selection_policy_must_match_a_state_batch_can_generate(tmp_path: Path) -> None:
    receipt = _receipt(
        tmp_path / "evaluation_manifest.json",
        [_cell("W01", "B3", 1401, pass_rate=1.0, tokens=1, fixes=1)],
    )
    _rewrite_receipt(
        receipt, lambda payload: payload["selection"].update(policy="failed-evaluation")
    )
    with pytest.raises(
        UserRunAggregateError, match="evaluation_manifest_evaluated_selection_invalid"
    ):
        build_user_run_aggregate(
            [receipt], output_directory=tmp_path / "aggregate", allow_partial=True
        )

    _rewrite_receipt(receipt, lambda payload: payload["selection"].update(policy="all-eligible"))
    result = build_user_run_aggregate(
        [receipt], output_directory=tmp_path / "all-eligible", allow_partial=True
    )
    assert result.evaluated_cells == 1


def test_receipt_failed_evaluation_selection_requires_pre_evaluation_failure_state(
    tmp_path: Path,
) -> None:
    receipt = _receipt(
        tmp_path / "evaluation_manifest.json",
        [_cell("W01", "B3", 1401, pass_rate=1.0, tokens=1, fixes=1)],
    )

    def mark_failed_evaluation(payload: dict) -> None:
        payload["selection"]["policy"] = "failed-evaluation"
        evaluated = next(cell for cell in payload["cells"] if cell["outcome"] == "evaluated")
        evaluated.update(
            scheduler_status="failed",
            scheduler_failure_class="evaluator_failure",
            public_pre_status="infra_error",
            public_pre_stage="evaluation",
        )
        payload["selection"]["eligible"][0].update(
            scheduler_status="failed",
            scheduler_failure_class="evaluator_failure",
            public_pre_status="infra_error",
            public_pre_stage="evaluation",
        )

    _rewrite_receipt(receipt, mark_failed_evaluation)
    result = build_user_run_aggregate(
        [receipt], output_directory=tmp_path / "aggregate", allow_partial=True
    )
    assert result.evaluated_cells == 1


@pytest.mark.parametrize(
    ("mutate", "error"),
    [
        (
            lambda payload: payload["source_snapshot"].update(
                repository="https://example.invalid/not-the-frozen-source"
            ),
            "evaluation_manifest_source_mismatch",
        ),
        (
            lambda payload: payload["manifest"]["upstream_source"].update(git_tree="0" * 40),
            "evaluation_manifest_source_mismatch",
        ),
        (
            lambda payload: payload["source_snapshot"].update(branch="untrusted-branch"),
            "evaluation_manifest_source_mismatch",
        ),
        (
            lambda payload: payload["manifest"]["upstream_source"].update(commit="0" * 40),
            "evaluation_manifest_source_mismatch",
        ),
        (
            lambda payload: payload["run_origin"].update(
                historical_author_run_inputs=["author-output"]
            ),
            "evaluation_manifest_origin_historical_input_forbidden",
        ),
        (
            lambda payload: payload["manifest"]["run_origin"].update(
                historical_raw_run_inputs=["old-run"]
            ),
            "evaluation_manifest_manifest_origin_historical_input_forbidden",
        ),
        (lambda payload: payload.update(dry_run=True), "evaluation_manifest_dry_run_forbidden"),
        (
            lambda payload: payload.update(status="completed_with_failures"),
            "evaluation_manifest_batch_not_completed",
        ),
        (
            lambda payload: payload.update(scheduler_runtime_unchanged=False),
            "evaluation_manifest_scheduler_runtime_changed",
        ),
        (
            lambda payload: payload.update(scheduler_runtime_after_sha256="6" * 64),
            "evaluation_manifest_scheduler_runtime_invalid",
        ),
        (
            lambda payload: payload["manifest_snapshot"].update(plan_sha256="5" * 64),
            "evaluation_manifest_plan_invalid",
        ),
        (
            lambda payload: payload["cells"][0].update(outcome="failed"),
            "evaluation_manifest_outcome_invalid",
        ),
        (
            lambda payload: next(
                cell for cell in payload["cells"] if cell["outcome"] == "evaluated"
            )["analysis"].update(workload="W99"),
            "evaluation_manifest_analysis_identity_invalid",
        ),
        (
            lambda payload: payload["manifest"]["plan_cells"].pop(),
            "evaluation_manifest_plan_cells_invalid",
        ),
        (
            lambda payload: next(
                cell for cell in payload["cells"] if cell["outcome"] == "evaluated"
            ).update(cell_spec_sha256="0" * 64),
            "evaluation_manifest_cell_identity_invalid",
        ),
    ],
)
def test_aggregate_rejects_self_hashed_receipts_without_required_batch_provenance(
    tmp_path: Path, mutate, error: str
) -> None:
    receipt = _receipt(
        tmp_path / "evaluation_manifest.json",
        [_cell("W01", "B3", 1401, pass_rate=1.0, tokens=1, fixes=1)],
    )
    _rewrite_receipt(receipt, mutate)
    with pytest.raises(UserRunAggregateError, match=error):
        build_user_run_aggregate(
            [receipt], output_directory=tmp_path / "aggregate", allow_partial=True
        )


def test_evaluator_infrastructure_or_unavailable_status_is_excluded_not_low_score(
    tmp_path: Path,
) -> None:
    cells = [
        _cell(
            "W01",
            "B3",
            1401,
            pass_rate=0.0,
            tokens=100,
            fixes=0,
            evaluator_status="infra_error",
            infrastructure_error_count=1,
        ),
        _cell(
            "W02",
            "B3",
            1401,
            pass_rate=0.0,
            tokens=100,
            fixes=0,
            evaluator_status="not_run",
        ),
        _cell(
            "W03",
            "B3",
            1401,
            pass_rate=1.0,
            tokens=100,
            fixes=1,
            evaluator_status="passed",
        ),
        _cell(
            "W04",
            "B3",
            1401,
            pass_rate=0.5,
            tokens=100,
            fixes=1,
            evaluator_status="incomplete",
        ),
    ]
    receipt = _receipt(tmp_path / "evaluation_manifest.json", cells)
    result = build_user_run_aggregate(
        [receipt], output_directory=tmp_path / "aggregate", allow_partial=True
    )
    payload = json.loads(result.json_path.read_text(encoding="utf-8"))
    by_arm = payload["results"]["available_metrics"]["generic_candidate_pass_rate_percent"][
        "by_arm"
    ]["B3"]
    assert by_arm["value"] == pytest.approx(75.0)
    assert by_arm["eligible_blocks"] == 2
    assert payload["coverage"]["aggregate_exclusions"] == {"evaluator_status_unavailable": 2}


def test_full_design_requires_exactly_two_complete_single_model_receipts(
    tmp_path: Path,
) -> None:
    first = _receipt(tmp_path / "terra.json", _complete_model_cells("gpt-5.6-terra"))
    second = _receipt(
        tmp_path / "opus.json", _complete_model_cells("claude-opus-4.6"), model="claude-opus-4.6"
    )
    result = build_user_run_aggregate([first, second], output_directory=tmp_path / "aggregate")
    assert result.analysis_status == "full_design_limited_metrics_scoring_ledger_missing"

    with pytest.raises(UserRunAggregateError, match="duplicate_evaluation_cell"):
        build_user_run_aggregate(
            [first, second, first], output_directory=tmp_path / "rejected-aggregate"
        )

    receipts = []
    for path in (first, second):
        payload = json.loads(path.read_text(encoding="utf-8"))
        receipts.append((payload, "f" * 64, path))
    with pytest.raises(UserRunAggregateError, match="full_design_requires_two_complete_receipts"):
        aggregate_module._validate_full_design_receipts([*receipts, receipts[0]])


def test_full_design_rejects_mixed_provenance_even_with_recomputed_receipt_hash(
    tmp_path: Path,
) -> None:
    first = _receipt(tmp_path / "terra.json", _complete_model_cells("gpt-5.6-terra"))
    second = _receipt(
        tmp_path / "opus.json", _complete_model_cells("claude-opus-4.6"), model="claude-opus-4.6"
    )
    _rewrite_receipt(
        second,
        lambda payload: payload["cells"][0]["analysis"]["lineage"].update(
            runtime_tree_sha256="9" * 64
        ),
    )
    # Keep the duplicated public analysis in the evaluation record identical.
    _rewrite_receipt(
        second,
        lambda payload: payload["evaluations"][0]["analysis"]["lineage"].update(
            runtime_tree_sha256="9" * 64
        ),
    )
    with pytest.raises(
        UserRunAggregateError,
        match="evaluation_manifest_analysis_derived_lineage_mismatch",
    ):
        build_user_run_aggregate([first, second], output_directory=tmp_path / "aggregate")


def test_full_design_rejects_cross_model_workload_evaluator_mismatch(
    tmp_path: Path,
) -> None:
    first = _receipt(tmp_path / "terra.json", _complete_model_cells("gpt-5.6-terra"))
    second = _receipt(
        tmp_path / "opus.json",
        _complete_model_cells("claude-opus-4.6"),
        model="claude-opus-4.6",
    )

    def change_workload_environment(payload: dict) -> None:
        for collection in (payload["cells"], payload["evaluations"]):
            for item in collection:
                analysis = item.get("analysis")
                if isinstance(analysis, dict) and analysis.get("workload") == "W01":
                    analysis["lineage"]["evaluator_environment_sha256"] = "9" * 64

    _rewrite_receipt(second, change_workload_environment)
    with pytest.raises(
        UserRunAggregateError,
        match="full_design_evaluator_provenance_mismatch",
    ):
        build_user_run_aggregate([first, second], output_directory=tmp_path / "aggregate")


def test_aggregate_cli_forwards_multiple_receipts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict = {}

    class Result:
        def to_dict(self) -> dict:
            return {"analysis_status": "partial_descriptive_not_paper_comparable"}

    def fake_build(manifests, **kwargs):
        captured["manifests"] = manifests
        captured.update(kwargs)
        return Result()

    monkeypatch.setattr(aggregate_module, "build_user_run_aggregate", fake_build)
    first = tmp_path / "one.json"
    second = tmp_path / "two.json"
    code = main(
        [
            "aggregate-user-runs",
            "--evaluation-manifest",
            str(first),
            "--evaluation-manifest",
            str(second),
            "--output-directory",
            str(tmp_path / "aggregate"),
            "--allow-partial",
        ]
    )
    assert code == 0
    assert captured == {
        "manifests": [first, second],
        "output_directory": tmp_path / "aggregate",
        "allow_partial": True,
    }
