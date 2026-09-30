"""Runtime validators for Society-Core non-negotiable axioms."""

from __future__ import annotations

from .schemas import (
    FORBIDDEN_INITIAL_OBJECT_TYPES,
    FORBIDDEN_MACRO_ACTIONS,
    Action,
    ActionKind,
    SocietyState,
)


class SocietyCoreValidationError(ValueError):
    """Raised when a Society-Core invariant is violated."""


def validate_no_initial_institutions(state: SocietyState) -> None:
    forbidden = set(state.forbidden_initial_objects_seen)
    if forbidden:
        raise SocietyCoreValidationError(
            f"Forbidden initial institution objects present: {sorted(forbidden)}"
        )
    for artifact in state.artifacts.values():
        if artifact.artifact_kind in FORBIDDEN_INITIAL_OBJECT_TYPES:
            raise SocietyCoreValidationError(
                f"Forbidden initial artifact kind: {artifact.artifact_kind}"
            )


def validate_action(action: Action) -> None:
    if action.kind.value in FORBIDDEN_MACRO_ACTIONS:
        raise SocietyCoreValidationError(f"Forbidden macro action: {action.kind.value}")


def validate_action_name(action_name: str) -> None:
    if action_name in FORBIDDEN_MACRO_ACTIONS:
        raise SocietyCoreValidationError(f"Forbidden macro action: {action_name}")
    try:
        ActionKind(action_name)
    except ValueError as exc:
        raise SocietyCoreValidationError(f"Unknown primitive action: {action_name}") from exc


def validate_type_separation(state: SocietyState) -> None:
    for agent in state.agents.values():
        if hasattr(agent.material_resources, "attention_budget"):
            raise SocietyCoreValidationError("attention_budget leaked into MaterialResources")
        if hasattr(agent.material_resources, "social_trust_as_private_belief"):
            raise SocietyCoreValidationError("social trust leaked into MaterialResources")
        if hasattr(agent.body, "attention_budget"):
            raise SocietyCoreValidationError("attention_budget leaked into BodyState")
        if hasattr(agent.body, "food_or_budget_token"):
            raise SocietyCoreValidationError("material resource leaked into BodyState")


def validate_initial_state(state: SocietyState) -> None:
    validate_no_initial_institutions(state)
    validate_type_separation(state)
