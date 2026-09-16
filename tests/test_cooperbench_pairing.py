from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import threading
import time
from pathlib import Path

import pytest

from environments.org_env.cooperbench.contract import (
    CONTRACT_SCHEMA_VERSION,
    DELIVERY_MODE,
    TREATMENT_ID,
    CooperContractError,
    FeatureCall,
    PairOutcome,
    PairRequest,
)
from environments.org_env.cooperbench.pairing import PairCoordinator


def _call(
    tmp_path: Path,
    agent_id: str,
    *,
    model: str = "gpt-5",
    run_id: str = "pair1234",
) -> FeatureCall:
    return FeatureCall.from_adapter_call(
        task=f"Implement public feature for {agent_id}",
        image="cooperbench/example:task1",
        agent_id=agent_id,
        agents=["agent1", "agent2"],
        model_name=model,
        config={
            "run_id": run_id,
            "backend": "docker",
            "orgenv_ticks": 48,
        },
        log_dir=str((tmp_path / run_id).resolve()),
    )


def _outcome(request: PairRequest) -> PairOutcome:
    return PairOutcome(
        schema_version=CONTRACT_SCHEMA_VERSION,
        treatment_id=TREATMENT_ID,
        delivery_mode=DELIVERY_MODE,
        run_id=request.run_id,
        agents=request.agents,
        status="Submitted",
        joint_patch="diff --git a/x.py b/x.py\n",
        usage_totals={"calls": 3, "total_tokens": 100},
    ).validated()


def test_pair_request_is_exactly_two_agents_and_b3_two_agent_only(tmp_path: Path):
    first = _call(tmp_path, "agent1")
    second = _call(tmp_path, "agent2")

    request = PairRequest.from_calls({"agent1": first, "agent2": second})

    assert request.treatment_id == (
        "b3_two_agent_cooperbench_v159_internal_method_rename_source"
    )
    assert request.delivery_mode == "identical_joint_mainline"
    assert request.agents == ("agent1", "agent2")
    assert set(request.tasks) == {"agent1", "agent2"}
    assert "condition" not in request.to_dict()


def test_adapter_contract_rejects_non_pair_and_non_docker(tmp_path: Path):
    with pytest.raises(CooperContractError, match="exactly_two_unique_agents_required"):
        FeatureCall.from_adapter_call(
            task="feature",
            image="image",
            agent_id="agent1",
            agents=["agent1", "agent2", "agent3"],
            model_name="gpt-5",
            config={"run_id": "run", "backend": "docker"},
            log_dir=str(tmp_path.resolve()),
        )

    with pytest.raises(CooperContractError, match="docker_backend_required"):
        FeatureCall.from_adapter_call(
            task="feature",
            image="image",
            agent_id="agent1",
            agents=["agent1", "agent2"],
            model_name="gpt-5",
            config={"run_id": "run", "backend": "modal"},
            log_dir=str(tmp_path.resolve()),
        )


def test_pair_coordinator_runs_one_executor_and_returns_one_joint_outcome(tmp_path: Path):
    executions: list[PairRequest] = []

    def executor(request: PairRequest, _timeout: int) -> PairOutcome:
        executions.append(request)
        return _outcome(request)

    coordinator = PairCoordinator(executor)
    results: dict[str, PairOutcome] = {}
    errors: list[BaseException] = []

    def submit(agent: str) -> None:
        try:
            results[agent] = coordinator.submit(
                _call(tmp_path, agent), rendezvous_timeout_seconds=2
            )
        except BaseException as error:  # pragma: no cover - assertion reports it
            errors.append(error)

    threads = [threading.Thread(target=submit, args=(agent,)) for agent in ("agent1", "agent2")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert not errors
    assert len(executions) == 1
    assert results["agent1"] is results["agent2"]
    assert results["agent1"].joint_patch == "diff --git a/x.py b/x.py\n"
    assert coordinator.pending_run_ids() == ()


def test_eight_parallel_pairs_never_cross_rendezvous_state(tmp_path: Path):
    pair_count = 8
    executor_barrier = threading.Barrier(pair_count)
    execution_requests: list[PairRequest] = []
    execution_lock = threading.Lock()

    def executor(request: PairRequest, _timeout: int) -> PairOutcome:
        with execution_lock:
            execution_requests.append(request)
        executor_barrier.wait(timeout=5)
        return _outcome(request)

    coordinator = PairCoordinator(executor)

    def submit(run_id: str, agent_id: str) -> tuple[str, str, PairOutcome]:
        outcome = coordinator.submit(
            _call(tmp_path, agent_id, run_id=run_id),
            rendezvous_timeout_seconds=5,
        )
        return run_id, agent_id, outcome

    expected_run_ids = {f"pair{i:04d}" for i in range(pair_count)}
    with ThreadPoolExecutor(max_workers=pair_count * 2) as pool:
        futures = [
            pool.submit(submit, run_id, agent_id)
            for run_id in sorted(expected_run_ids)
            for agent_id in ("agent1", "agent2")
        ]
        results = [future.result(timeout=10) for future in futures]

    assert {request.run_id for request in execution_requests} == expected_run_ids
    assert len(execution_requests) == pair_count
    assert len(results) == pair_count * 2
    assert all(outcome.run_id == run_id for run_id, _agent, outcome in results)
    for run_id in expected_run_ids:
        pair_outcomes = [
            outcome for result_run, _agent, outcome in results
            if result_run == run_id
        ]
        assert len(pair_outcomes) == 2
        assert pair_outcomes[0] is pair_outcomes[1]
    assert coordinator.pending_run_ids() == ()


def test_paired_call_uses_worker_budget_not_short_rendezvous_budget(tmp_path: Path):
    def executor(request: PairRequest, _timeout: int) -> PairOutcome:
        time.sleep(0.1)
        return _outcome(request)

    coordinator = PairCoordinator(executor)
    results: list[PairOutcome] = []
    errors: list[BaseException] = []

    def submit(agent: str) -> None:
        try:
            results.append(
                coordinator.submit(
                    _call(tmp_path, agent), rendezvous_timeout_seconds=0.05
                )
            )
        except BaseException as error:  # pragma: no cover - assertion reports it
            errors.append(error)

    first = threading.Thread(target=submit, args=("agent1",))
    second = threading.Thread(target=submit, args=("agent2",))
    first.start()
    time.sleep(0.01)
    second.start()
    first.join(timeout=2)
    second.join(timeout=2)

    assert not errors
    assert len(results) == 2


def test_pair_coordinator_fails_a_lone_feature_call(tmp_path: Path):
    coordinator = PairCoordinator(lambda request, timeout: _outcome(request))

    with pytest.raises(TimeoutError, match="cooper_pair_rendezvous_timeout"):
        coordinator.submit(
            _call(tmp_path, "agent1"), rendezvous_timeout_seconds=0.05
        )

    assert coordinator.pending_run_ids() == ()


def test_pair_request_rejects_mismatched_model(tmp_path: Path):
    with pytest.raises(CooperContractError, match="pair_execution_config_mismatch"):
        PairRequest.from_calls(
            {
                "agent1": _call(tmp_path, "agent1", model="gpt-5"),
                "agent2": _call(tmp_path, "agent2", model="gpt-5-mini"),
            }
        )
