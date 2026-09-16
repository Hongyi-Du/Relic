from __future__ import annotations

import importlib
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from environments.org_env.cooperbench.contract import (
    CONTRACT_SCHEMA_VERSION,
    DELIVERY_MODE,
    TREATMENT_ID,
    PairOutcome,
)


@dataclass
class _AgentResult:
    status: str
    patch: str
    cost: float
    steps: int
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    messages: list[dict[str, Any]] = field(default_factory=list)
    sent_messages: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None


def _load_adapter(monkeypatch, registered: list[str]):
    agents_module = types.ModuleType("cooperbench.agents")
    agents_module.AgentResult = _AgentResult

    def register(name: str):
        registered.append(name)
        return lambda cls: cls

    registry_module = types.ModuleType("cooperbench.agents.registry")
    registry_module.register = register
    package = types.ModuleType("cooperbench")
    package.__path__ = []
    monkeypatch.setitem(sys.modules, "cooperbench", package)
    monkeypatch.setitem(sys.modules, "cooperbench.agents", agents_module)
    monkeypatch.setitem(
        sys.modules, "cooperbench.agents.registry", registry_module
    )
    sys.modules.pop("environments.org_env.cooperbench.adapter", None)
    return importlib.import_module("environments.org_env.cooperbench.adapter")


def test_external_adapter_registers_and_partitions_shared_usage(
    tmp_path: Path, monkeypatch
):
    registered: list[str] = []
    adapter = _load_adapter(monkeypatch, registered)

    outcome = PairOutcome(
        schema_version=CONTRACT_SCHEMA_VERSION,
        treatment_id=TREATMENT_ID,
        delivery_mode=DELIVERY_MODE,
        run_id="run1",
        agents=("agent1", "agent2"),
        status="Submitted",
        joint_patch="diff --git a/x.py b/x.py\n",
        usage_totals={
            "calls": 5,
            "prompt_tokens": 101,
            "completion_tokens": 51,
            "cached_prompt_tokens": 11,
        },
        communications_by_agent={
            "agent1": [{"to": "agent2", "content": "review this"}],
            "agent2": [],
        },
    ).validated()

    class FakeCoordinator:
        def submit(self, call, *, rendezvous_timeout_seconds):
            assert call.agent_id == "agent1"
            assert rendezvous_timeout_seconds == 120
            return outcome

    monkeypatch.setattr(adapter, "_COORDINATOR", FakeCoordinator())
    result = adapter.OrgEnvB3TwoAgentRunner().run(
        task="feature one",
        image="image",
        agent_id="agent1",
        model_name="gpt-5",
        agents=["agent1", "agent2"],
        config={"run_id": "run1", "backend": "docker", "orgenv_ticks": 48},
        log_dir=str(tmp_path.resolve()),
    )

    assert registered == ["orgenv_b3_two_agent"]
    assert result.status == "Submitted"
    assert result.patch == outcome.joint_patch
    assert result.steps == 3
    assert result.input_tokens == 51
    assert result.output_tokens == 26
    assert result.cache_read_tokens == 6
    assert result.cost == 0.0
    assert result.sent_messages == [{"to": "agent2", "content": "review this"}]


def test_external_adapter_returns_error_for_a_non_pair(tmp_path: Path, monkeypatch):
    adapter = _load_adapter(monkeypatch, [])

    result = adapter.OrgEnvB3TwoAgentRunner().run(
        task="feature",
        image="image",
        agent_id="agent1",
        model_name="gpt-5",
        agents=["agent1", "agent2", "agent3"],
        config={"run_id": "run", "backend": "docker"},
        log_dir=str(tmp_path.resolve()),
    )

    assert result.status == "Error"
    assert result.patch == ""
    assert "exactly_two_unique_agents_required" in result.error
