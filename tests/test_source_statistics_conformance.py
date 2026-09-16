"""Conformance checks for the hci paired-statistics boundary.

Relic's public receipt is intentionally narrower than an hci run record.  The
adapter may redact inputs, but it must not reimplement the point estimate or
bootstrap after it has produced source-compatible observations.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import relic.evaluation.user_run_aggregate as aggregate
from environments.org_env.experiments.statistics import (
    build_paired_units,
    normalize_condition,
    paired_block_bootstrap_interval,
    paired_block_weighted_mean,
    validate_observations,
)


class _ResourceBudget:
    def to_dict(self) -> dict[str, int]:
        return {"max_ticks": 336}


def _spec(model: str, workload: str, arm: str, seed: int) -> SimpleNamespace:
    return SimpleNamespace(
        model=model,
        workload=workload.lower(),
        arm=arm.lower(),
        seed=seed,
        dataset_id="source-conformance-pack",
        model_config={"provider": "openai"},
        resource_budget=_ResourceBudget(),
        mechanism_ablations=("work_rhythm",),
        benchmark_entry={"project_id": "source-conformance-pack"},
    )


def _cell(*, arm: str, seed: int, pass_rate: float) -> dict:
    model = "gpt-5.6-terra"
    workload = "W01"
    cell_id = f"{model}__{workload}__{arm}__seed{seed}"
    lineage = {
        "source_provenance_sha256": "a" * 64,
        "runtime_tree_sha256": "b" * 64,
        "model_binding_sha256": "c" * 64,
        "execution_policy_sha256": "d" * 64,
        "qualification_plan_sha256": "e" * 64,
        "evaluator_environment_sha256": "f" * 64,
        "starter_repo_digest": "1" * 64,
        "reference_repo_digest": "2" * 64,
        "hidden_suite_hash": "3" * 64,
    }
    evaluator = {
        "artifact_hash": "4" * 64,
        "result_hash": "5" * 64,
        "candidate_repo_digest": "6" * 64,
        "status": "passed",
    }
    return {
        "cell_id": cell_id,
        "model": model,
        "workload": workload,
        "arm": arm,
        "seed": seed,
        "analysis": {
            "model": model,
            "workload": workload,
            "arm": arm,
            "seed": seed,
            "llm_usage": {"total_tokens": 100},
            "lineage": lineage,
            "evaluator": evaluator,
            "final_evaluation": {
                "status": "passed",
                "formal_claim_ready": True,
                "candidate_pass_rate": pass_rate,
                "causal_fix_rate": pass_rate,
                "causal_fix_count": 1,
                "unresolved_count": 0,
                "regression_count": 0,
                "infrastructure_error_count": 0,
            },
        },
    }


def test_statistics_aliases_match_hci_source_ladder() -> None:
    assert normalize_condition("b3_full_sociogenesis") == "B3"


def test_aggregate_b3_minus_b2_uses_source_fixed_block_functions() -> None:
    cells = [
        _cell(arm=arm, seed=seed, pass_rate=value)
        for seed, (b2, b3) in zip(
            (1401, 2711, 4013),
            ((0.1, 0.4), (0.2, 0.45), (0.3, 0.5)),
            strict=True,
        )
        for arm, value in (("B2", b2), ("B3", b3))
    ]
    specs = {
        str(cell["cell_id"]): _spec(
            str(cell["model"]),
            str(cell["workload"]),
            str(cell["arm"]),
            int(cell["seed"]),
        )
        for cell in cells
    }
    metric = "generic_candidate_pass_rate_percent"
    observations = tuple(
        aggregate._source_statistics_observation(
            cell,
            spec=specs[str(cell["cell_id"])],
            metric=metric,
        )
        for cell in cells
    )
    validate_observations(
        observations,
        require_complete_conditions=False,
        require_matched_resources=True,
        require_matched_ablations=True,
    )
    units = build_paired_units(
        observations,
        metric=metric,
        treatment="B3",
        control="B2",
    )
    expected_interval = paired_block_bootstrap_interval(
        units,
        samples=aggregate.BOOTSTRAP_DRAWS,
        seed=aggregate.BOOTSTRAP_SEED,
    )

    result = aggregate._paired_contrast(cells, metric, specs_by_cell_id=specs)

    assert result["b3_minus_b2"] == pytest.approx(paired_block_weighted_mean(units))
    assert result["ci95"] == pytest.approx(expected_interval)
    assert result["eligible_blocks"] == 1
    assert result["paired_cells"] == 3
