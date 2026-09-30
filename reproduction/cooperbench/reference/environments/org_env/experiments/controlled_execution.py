"""Execution adapter adding evaluator-owned resource and ablation gates."""
from __future__ import annotations

from typing import Any

from environments.org_env.experiments.ablations import (
    EXTERNAL_BRIDGE,
    INSTITUTIONALIZATION,
    PROTOCOL_ENFORCEMENT,
    mechanism_disabled,
)
from environments.org_env.experiments.resources import (
    PRIMARY_ACTIONS,
    reserve_world_resources,
)
from environments.org_env.runtime_adapter.execution import (
    ExecutionResult,
    OrgExecutionAdapter,
)


class ExperimentControlledExecutionAdapter(OrgExecutionAdapter):
    """Default-compatible adapter; gates activate only in explicit experiment arms."""

    _INSTITUTIONAL_ACTIONS = frozenset(
        {
            "propose_protocol",
            "amend_protocol",
            "support_protocol",
            "follow_protocol",
        }
    )

    def execute(self, agent_id: str, action: Any, org_world: Any) -> ExecutionResult:
        action_type = str(action.action_type)
        if not reserve_world_resources(
            org_world,
            {PRIMARY_ACTIONS: 1},
            agent_id=agent_id,
            detail=action_type,
        ):
            tick = int(getattr(org_world, "world_tick", 0))
            return ExecutionResult(
                action_id=f"act_{agent_id}_{tick}_{action_type}",
                agent_id=agent_id,
                action_type=action_type,
                success=False,
                failure_reason="experiment_resource_exhausted:primary_actions",
                events=[
                    {
                        "type": "experiment_resource_event",
                        "subtype": "primary_action_denied",
                        "agent_id": agent_id,
                        "action_type": action_type,
                        "tick": tick,
                    }
                ],
            )
        if (
            action_type in self._INSTITUTIONAL_ACTIONS
            and (
                not getattr(org_world, "institutionalization_enabled", True)
                or mechanism_disabled(org_world, INSTITUTIONALIZATION)
            )
        ):
            tick = int(getattr(org_world, "world_tick", 0))
            return ExecutionResult(
                action_id=f"act_{agent_id}_{tick}_{action_type}",
                agent_id=agent_id,
                action_type=action_type,
                success=False,
                failure_reason="mechanism_ablation:institutionalization",
                events=[
                    {
                        "type": "mechanism_ablation_event",
                        "subtype": "institutional_action_blocked",
                        "agent_id": agent_id,
                        "action_type": action_type,
                        "tick": tick,
                    }
                ],
            )
        return super().execute(agent_id, action, org_world)

    def _enforce_protocols(self, w, aid, at, params, res, tick) -> None:
        if mechanism_disabled(w, PROTOCOL_ENFORCEMENT):
            return
        super()._enforce_protocols(w, aid, at, params, res, tick)

    def _h_collect_post_launch_feedback(self, w, aid, params, res, tick) -> None:
        if mechanism_disabled(w, EXTERNAL_BRIDGE):
            res.success = False
            res.failure_reason = "mechanism_ablation:external_bridge"
            res.events.append(
                {
                    "type": "mechanism_ablation_event",
                    "subtype": "external_feedback_blocked",
                    "agent_id": aid,
                    "tick": tick,
                }
            )
            return
        super()._h_collect_post_launch_feedback(w, aid, params, res, tick)


__all__ = ["ExperimentControlledExecutionAdapter"]
