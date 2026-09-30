"""Auditable Cooper-only mapping from simulated hours to decision windows.

CooperBench budgets agents by actions, while OrgEnv's generic calendar budgets
them again by nights, weekends and persona availability.  A 336-tick pair can
therefore give one member fewer than a third of the other's decision windows.
This module removes that synthetic-calendar coupling for the two-person
treatment. CooperBench measures software delivery under a model-call budget,
not simulated human physiology, so its compressed schedule also disables the
lived-body work-rhythm layer while retaining real activity duration.
"""
from __future__ import annotations

from typing import Any, Mapping


SCHEMA = "cooperbench_compressed_decision_schedule_v2"
_STORE = "_cooperbench_work_schedule"
_CALL_BUDGET_STORE = "_cooperbench_model_call_budget"
MODEL_CALL_LIMIT_PER_MEMBER = 500
_RECEIPT = {
    "schema_version": SCHEMA,
    "mode": "compressed_continuous_decision_windows",
    "calendar_hard_gates": False,
    "calendar_utility_priors": False,
    "calendar_overtime_weekend_costs": False,
    "work_rhythm_enabled": False,
    "model_call_limit_per_member": MODEL_CALL_LIMIT_PER_MEMBER,
    "model_call_semantics": "logical_model_queries_including_failed_queries_excluding_worker_transport_preflight",
    "preserved_constraints": [
        "blocking_and_background_activity_duration",
        "agent_action_choice",
    ],
}


def _valid(receipt: Any) -> bool:
    return bool(
        isinstance(receipt, Mapping)
        and receipt.get("schema_version") == SCHEMA
        and receipt.get("mode") == _RECEIPT["mode"]
        and receipt.get("calendar_hard_gates") is False
        and receipt.get("calendar_utility_priors") is False
        and receipt.get("calendar_overtime_weekend_costs") is False
        and receipt.get("work_rhythm_enabled") is False
        and receipt.get("model_call_limit_per_member") == MODEL_CALL_LIMIT_PER_MEMBER
        and receipt.get("model_call_semantics") == _RECEIPT["model_call_semantics"]
        and list(receipt.get("preserved_constraints") or [])
        == _RECEIPT["preserved_constraints"]
    )


def compressed_schedule_receipt(value: Any) -> dict[str, Any] | None:
    """Return a validated copy from a world, TimeSystem, or OrgClock."""

    namespace = getattr(value, "__dict__", {}) if value is not None else {}
    receipt = namespace.get(_STORE) if isinstance(namespace, dict) else None
    if _valid(receipt):
        return {**receipt, "preserved_constraints": list(receipt["preserved_constraints"])}
    return None


def compressed_schedule_enabled(value: Any) -> bool:
    return compressed_schedule_receipt(value) is not None


def install_compressed_schedule(world: Any) -> dict[str, Any]:
    """Install the same explicit receipt on the world, time system and clock."""

    if world is None:
        raise ValueError("cooperbench_compressed_schedule_requires_world")
    time_system = getattr(world, "time", None)
    clock = getattr(time_system, "clock", None) if time_system is not None else None
    if time_system is not None:
        time_system.rhythm_enabled = False
    # Lightweight lifecycle/evidence fixtures may intentionally omit a clock;
    # an actual OrgWorld always has both additional targets.
    targets = tuple(target for target in (world, time_system, clock) if target is not None)
    for target in targets:
        existing = getattr(target, "__dict__", {}).get(_STORE)
        if existing is not None and not _valid(existing):
            raise ValueError("cooperbench_compressed_schedule_conflict")
        target.__dict__[_STORE] = {
            **_RECEIPT,
            "preserved_constraints": list(_RECEIPT["preserved_constraints"]),
        }
    return compressed_schedule_receipt(world) or {}


def install_model_call_budget(
    world: Any, member_ids: tuple[str, ...] | list[str]
) -> dict[str, Any]:
    """Install the treatment's 500-query allowance for each solver.

    The transport-integrity probe runs before feature owners and this budget
    are bound, so it is intentionally excluded. Every later organization
    model query must carry the acting member id and consumes one slot even if
    the provider or response parser fails, matching the upstream agent's
    pre-query ``n_calls`` increment.
    """

    members = tuple(str(member) for member in member_ids)
    if len(members) != 2 or len(set(members)) != 2 or any(not item for item in members):
        raise ValueError("cooperbench_model_call_budget_requires_two_members")
    existing = getattr(world, "__dict__", {}).get(_CALL_BUDGET_STORE)
    if existing is not None:
        receipt = model_call_budget_receipt(world)
        if receipt is None or tuple(receipt["members"]) != members:
            raise ValueError("cooperbench_model_call_budget_conflict")
        return receipt
    world.__dict__[_CALL_BUDGET_STORE] = {
        "schema_version": "cooperbench_team_model_call_budget_v2",
        "limit_per_member": MODEL_CALL_LIMIT_PER_MEMBER,
        "members": list(members),
        "logical_calls_by_member": {member: 0 for member in members},
        "denied_calls_by_member": {member: 0 for member in members},
        "organization_logical_calls": 0,
        "organization_denied_calls": 0,
        "worker_transport_preflight_excluded": True,
    }
    return model_call_budget_receipt(world) or {}


def model_call_budget_receipt(world: Any) -> dict[str, Any] | None:
    raw = getattr(world, "__dict__", {}).get(_CALL_BUDGET_STORE)
    if not isinstance(raw, Mapping):
        return None
    members = raw.get("members")
    calls = raw.get("logical_calls_by_member")
    denied = raw.get("denied_calls_by_member")
    organization_calls = raw.get("organization_logical_calls")
    organization_denied = raw.get("organization_denied_calls")
    if (
        raw.get("schema_version") != "cooperbench_team_model_call_budget_v2"
        or raw.get("limit_per_member") != MODEL_CALL_LIMIT_PER_MEMBER
        or raw.get("worker_transport_preflight_excluded") is not True
        or not isinstance(members, list)
        or len(members) != 2
        or any(type(member) is not str or not member for member in members)
        or len(set(members)) != 2
        or not isinstance(calls, Mapping)
        or not isinstance(denied, Mapping)
        or set(calls) != set(members)
        or set(denied) != set(members)
        or type(organization_calls) is not int
        or organization_calls < 0
        or type(organization_denied) is not int
        or organization_denied < 0
    ):
        return None
    normalized_calls: dict[str, int] = {}
    normalized_denied: dict[str, int] = {}
    for member in members:
        count = calls[member]
        denied_count = denied[member]
        if type(count) is not int or type(denied_count) is not int:
            return None
        if not 0 <= count <= MODEL_CALL_LIMIT_PER_MEMBER or denied_count < 0:
            return None
        normalized_calls[member] = count
        normalized_denied[member] = denied_count
    team_calls = sum(normalized_calls.values()) + organization_calls
    team_limit = 2 * MODEL_CALL_LIMIT_PER_MEMBER
    if team_calls > team_limit:
        return None
    return {
        "schema_version": raw["schema_version"],
        "limit_per_member": MODEL_CALL_LIMIT_PER_MEMBER,
        "members": list(members),
        "logical_calls_by_member": normalized_calls,
        "denied_calls_by_member": normalized_denied,
        "organization_logical_calls": organization_calls,
        "organization_denied_calls": organization_denied,
        "worker_transport_preflight_excluded": True,
        "team_logical_calls": team_calls,
        "team_limit": team_limit,
    }


def model_call_budget_enabled(world: Any) -> bool:
    return model_call_budget_receipt(world) is not None


def model_call_budget_installed(world: Any) -> bool:
    return _CALL_BUDGET_STORE in getattr(world, "__dict__", {})


def model_call_budget_exhausted(world: Any, member_id: str) -> bool:
    receipt = model_call_budget_receipt(world)
    if receipt is None:
        # A compressed Cooper world with a missing or malformed budget must
        # stop offering work, never silently regain unlimited queries.
        return compressed_schedule_enabled(world)
    if member_id not in receipt["logical_calls_by_member"]:
        return False
    return (
        receipt["team_logical_calls"] >= receipt["team_limit"]
        or receipt["logical_calls_by_member"][member_id]
        >= MODEL_CALL_LIMIT_PER_MEMBER
    )


def reserve_model_call(world: Any, member_id: str | None) -> dict[str, Any]:
    """Consume exactly one logical solver query, failing closed at the limit."""

    raw = getattr(world, "__dict__", {}).get(_CALL_BUDGET_STORE)
    if raw is None:
        return {}
    receipt = model_call_budget_receipt(world)
    if receipt is None:
        raise RuntimeError("cooperbench_model_call_budget_invalid")
    member = str(member_id or "")
    if member and member not in receipt["logical_calls_by_member"]:
        raise RuntimeError("cooperbench_model_call_actor_unknown")
    if receipt["team_logical_calls"] >= receipt["team_limit"]:
        if member:
            raw["denied_calls_by_member"][member] = int(
                raw["denied_calls_by_member"][member]
            ) + 1
        else:
            raw["organization_denied_calls"] = int(
                raw["organization_denied_calls"]
            ) + 1
        raise RuntimeError("cooperbench_model_call_team_budget_exhausted")
    if not member:
        raw["organization_logical_calls"] = int(
            raw["organization_logical_calls"]
        ) + 1
        return model_call_budget_receipt(world) or {}
    calls = raw["logical_calls_by_member"]
    if int(calls[member]) >= MODEL_CALL_LIMIT_PER_MEMBER:
        raw["denied_calls_by_member"][member] = int(
            raw["denied_calls_by_member"][member]
        ) + 1
        raise RuntimeError(f"cooperbench_model_call_budget_exhausted:{member}")
    calls[member] = int(calls[member]) + 1
    return model_call_budget_receipt(world) or {}


def member_model_call_budget_context(
    world: Any, member_id: str
) -> dict[str, Any] | None:
    """Small actor-local budget view for a decision prompt."""

    receipt = model_call_budget_receipt(world)
    if receipt is None or member_id not in receipt["logical_calls_by_member"]:
        return None
    used = int(receipt["logical_calls_by_member"][member_id])
    team_remaining = receipt["team_limit"] - receipt["team_logical_calls"]
    return {
        "limit": MODEL_CALL_LIMIT_PER_MEMBER,
        "used_before_current_query": used,
        "remaining_before_current_query": min(
            MODEL_CALL_LIMIT_PER_MEMBER - used,
            team_remaining,
        ),
        "team_remaining_before_current_query": team_remaining,
        "organization_logical_calls": receipt["organization_logical_calls"],
        "failed_queries_count_toward_limit": True,
        "worker_transport_preflight_excluded": True,
    }


__all__ = [
    "SCHEMA",
    "compressed_schedule_enabled",
    "compressed_schedule_receipt",
    "install_compressed_schedule",
    "MODEL_CALL_LIMIT_PER_MEMBER",
    "install_model_call_budget",
    "model_call_budget_enabled",
    "model_call_budget_installed",
    "model_call_budget_exhausted",
    "model_call_budget_receipt",
    "member_model_call_budget_context",
    "reserve_model_call",
]
