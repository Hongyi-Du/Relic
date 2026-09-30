"""CooperBench external-agent registration for the two-member B3 treatment.

Set ``COOPERBENCH_EXTERNAL_AGENTS=environments.org_env.cooperbench.adapter``
in the CooperBench runner process, then select agent framework
``orgenv_b3_two_agent``.
"""
from __future__ import annotations

import json
from typing import Any, Mapping

from environments.org_env.cooperbench.contract import (
    ADAPTER_NAME,
    FeatureCall,
    PairOutcome,
)
from environments.org_env.cooperbench.execution import execute_pair_via_worker
from environments.org_env.cooperbench.pairing import PairCoordinator

try:  # CooperBench is an optional external dependency of this repository.
    # CooperBench imports external adapters while ``cooperbench.agents`` is
    # still initializing.  ``AgentResult`` already exists at that point, but
    # the convenience re-export of ``register`` does not.  Import the
    # decorator from its defining module so official auto-registration does
    # not silently fall through to the optional-dependency stub.
    from cooperbench.agents import AgentResult
    from cooperbench.agents.registry import register
except ImportError:  # pragma: no cover - exercised by import smoke tests here
    AgentResult = None  # type: ignore[assignment,misc]

    def register(_name: str):
        return lambda cls: cls


_COORDINATOR = PairCoordinator(execute_pair_via_worker)


def _bounded_timeout(config: Mapping[str, Any] | None) -> float:
    raw = dict(config or {}).get("orgenv_rendezvous_timeout_seconds", 120)
    try:
        parsed = float(raw)
    except (TypeError, ValueError) as error:
        raise ValueError("orgenv_rendezvous_timeout_seconds_invalid") from error
    if not 1 <= parsed <= 600:
        raise ValueError("orgenv_rendezvous_timeout_seconds_out_of_range")
    return parsed


def _share(total: Any, index: int, count: int = 2) -> int:
    try:
        value = max(0, int(total or 0))
    except (TypeError, ValueError):
        value = 0
    quotient, remainder = divmod(value, count)
    return quotient + (1 if index < remainder else 0)


def _agent_result(
    outcome: PairOutcome,
    *,
    agent_id: str,
) -> Any:
    if AgentResult is None:
        raise RuntimeError(
            "CooperBench is not installed; import this adapter from a "
            "CooperBench runner environment"
        )
    index = outcome.agents.index(agent_id)
    usage = dict(outcome.usage_totals)
    summary = {
        "treatment_id": outcome.treatment_id,
        "delivery_mode": outcome.delivery_mode,
        "run_id": outcome.run_id,
        "organization_status": outcome.status,
        "shared_usage_allocation": "integer_partition_across_two_adapter_results",
        "cost_usd": None,
    }
    organization = dict((outcome.receipt or {}).get("organization") or {})
    runtime = dict((outcome.receipt or {}).get("runtime") or {})
    call_budget = dict(runtime.get("model_call_budget") or {})
    external_to_internal = dict(organization.get("external_to_internal") or {})
    internal_id = external_to_internal.get(agent_id)
    exact_calls = dict(call_budget.get("logical_calls_by_member") or {}).get(
        internal_id
    )
    steps = (
        max(0, int(exact_calls))
        if exact_calls is not None
        else _share(usage.get("calls"), index)
    )
    summary["step_accounting"] = (
        "exact_per_member_logical_model_queries"
        if exact_calls is not None
        else "legacy_integer_partition_of_team_calls"
    )
    return AgentResult(
        status=outcome.status,
        patch=outcome.joint_patch if outcome.status == "Submitted" else "",
        # OrgEnv currently records token/call counts but not provider pricing.
        # Zero is the only valid AgentResult scalar; the receipt states that it
        # is unavailable rather than claiming the run was free.
        cost=0.0,
        steps=steps,
        input_tokens=_share(usage.get("prompt_tokens"), index),
        output_tokens=_share(usage.get("completion_tokens"), index),
        cache_read_tokens=_share(usage.get("cached_prompt_tokens"), index),
        cache_write_tokens=0,
        messages=[
            {
                "role": "system",
                "content": json.dumps(summary, sort_keys=True),
            }
        ],
        sent_messages=list(
            (outcome.communications_by_agent or {}).get(agent_id, [])
        ),
        error=outcome.error,
    )


@register(ADAPTER_NAME)
class OrgEnvB3TwoAgentRunner:
    """Rendezvous two Cooper features and run one B3-2 organization."""

    def run(
        self,
        task: str,
        image: str,
        *,
        agent_id: str = "agent",
        model_name: str = "gpt-4o",
        agents: list[str] | None = None,
        comm_url: str | None = None,
        git_server_url: str | None = None,
        git_enabled: bool = False,
        messaging_enabled: bool = True,
        config: dict[str, Any] | None = None,
        agent_config: str | None = None,
        log_dir: str | None = None,
    ) -> Any:
        del comm_url, git_server_url, git_enabled, messaging_enabled, agent_config
        try:
            call = FeatureCall.from_adapter_call(
                task=task,
                image=image,
                agent_id=agent_id,
                agents=agents,
                model_name=model_name,
                config=config,
                log_dir=log_dir,
            )
            outcome = _COORDINATOR.submit(
                call,
                rendezvous_timeout_seconds=_bounded_timeout(config),
            )
            return _agent_result(outcome, agent_id=agent_id)
        except Exception as error:
            if AgentResult is None:
                raise
            return AgentResult(
                status="Error",
                patch="",
                cost=0.0,
                steps=0,
                messages=[],
                sent_messages=[],
                error=f"{type(error).__name__}:{error}"[:1000],
            )


__all__ = ["OrgEnvB3TwoAgentRunner"]
