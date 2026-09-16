"""The door a human action goes through.

Everything a seat submits is checked here before it reaches the world: that the
action exists, that its parameters are complete, that the member's role allows
it, and that the object it targets is one this member can actually see. Only
then does it go to ``HumanModeRuntime.submit_action``, which runs the same
pipeline the autonomous loop runs.

The visibility check matters as much as the role check. Object ids are
guessable, and without it a seat could act on a private task it was never shown
by typing its id — reading the organization through the write path.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from agent_sdk.lived.core.contracts import ActionCandidate
from environments.org_env.human import visibility as vis
from environments.org_env.human.affordances import (
    CONTEXT_GATES,
    GLOBAL_ACTIONS,
    REGISTRY_SURFACE_ACTIONS,
    ROLE_GATES,
    ActionSpec,
    find_spec,
    human_action_executable,
    role_denial,
    specs_for_object,
)

#: Parameters that name a person or a channel rather than an object.
_NON_OBJECT_PARAMS = frozenset({
    "channel_id", "owner_id", "to_agent", "assignee_id", "target_agent", "dm_id",
})


class ActionRefused(Exception):
    """An action a seat may not take. The message is the machine-readable reason."""


# --------------------------------------------------------------------------- #
# Objects
# --------------------------------------------------------------------------- #
def resolve_object(world: Any, object_id: str) -> Tuple[str, Any]:
    """Find an object by id and say what kind it is. Looks it up in the world's
    own collections rather than parsing the id, so a renamed prefix cannot
    quietly turn into an unchecked object."""
    repo = world.repo_system.repo
    pm = getattr(world, "proposal_manager", None)
    lookups = (
        ("task", world.tasks),
        ("document", world.documents),
        ("issue", getattr(world, "issues", {})),
        ("experiment", world.experiments),
        ("pull_request", repo.pull_requests),
        ("ci_run", getattr(repo, "ci_runs", {})),
        ("branch", repo.branches),
        ("release_candidate", getattr(repo, "release_candidates", {})),
        ("release", getattr(repo, "releases", {})),
        ("meeting", world.meeting_system.meetings),
        ("protocol", world.protocol_registry.protocols),
        ("proposal", (getattr(pm, "proposals", {}) or {}) if pm else {}),
        ("message", world.comm.messages),
        ("result", getattr(world.sandbox_system, "results", {})),
    )
    for kind, table in lookups:
        if table and object_id in table:
            return kind, table[object_id]
    raise ActionRefused(f"unknown_object:{object_id}")


def object_visible_to(world: Any, kind: str, obj: Any, agent_id: str) -> bool:
    checks = {
        "task": vis.task_visible_to,
        "document": vis.document_visible_to,
        "branch": vis.branch_visible_to,
        "pull_request": vis.pull_request_visible_to,
        "ci_run": lambda ci, member_id: vis.ci_run_visible_to(world, ci, member_id),
        "experiment": vis.experiment_visible_to,
        "meeting": vis.meeting_visible_to,
        "proposal": vis.proposal_visible_to,
        "protocol": vis.protocol_visible_to,
        "release": vis.release_visible_to,
        "release_candidate": vis.release_visible_to,
        "issue": vis.issue_visible_to,
    }
    check = checks.get(kind)
    if check is not None:
        return check(obj, agent_id)
    if kind == "message":
        return any(m.message_id == obj.message_id
                   for m in vis.visible_messages(world, agent_id))
    if kind == "result":
        return any(r.result_id == obj.result_id
                   for r in vis.visible_results(world, agent_id))
    return False


# --------------------------------------------------------------------------- #
# Offers
# --------------------------------------------------------------------------- #
def _offer(world, agent, spec: ActionSpec, object_id: str = "") -> Dict[str, Any]:
    params = {spec.target_param: object_id} if (spec.target_param and object_id) else {}
    denial = role_denial(world, agent, spec.action_type, params)
    if denial is None and not human_action_executable(spec.action_type):
        denial = f"unimplemented_action_handler:{spec.action_type}"
    offer = spec.schema()
    offer["allowed"] = denial is None
    if denial is not None:
        offer["denied_because"] = denial
    if object_id:
        offer["target"] = object_id
    return offer


def offers_for_object(world: Any, agent_id: str, object_id: str) -> Dict[str, Any]:
    """The context menu for one object: every action of its kind, each marked
    allowed or not. Forbidden ones are still listed with the reason, so the
    interface explains the organization's rules instead of hiding them."""
    agent = world.agents[agent_id]
    kind, obj = resolve_object(world, object_id)
    if not object_visible_to(world, kind, obj, agent_id):
        raise ActionRefused(f"object_not_visible:{object_id}")
    return {"object_id": object_id, "kind": kind,
            "actions": [_offer(world, agent, s, object_id)
                        for s in specs_for_object(kind)]}


def global_offers(world: Any, agent_id: str) -> List[Dict[str, Any]]:
    agent = world.agents[agent_id]
    return [_offer(world, agent, s) for s in (*GLOBAL_ACTIONS, *REGISTRY_SURFACE_ACTIONS)]


# --------------------------------------------------------------------------- #
# Submission
# --------------------------------------------------------------------------- #
def validate(world: Any, agent_id: str, action_type: str,
             params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Check an action and return the parameters to execute with.

    Raises ``ActionRefused`` with a machine-readable reason, which the UI shows
    and the tests assert on.
    """
    params = dict(params or {})
    if agent_id not in world.agents:
        raise ActionRefused(f"unknown_member:{agent_id}")
    agent = world.agents[agent_id]

    spec = find_spec(action_type, params)
    if spec is None:
        raise ActionRefused(f"unknown_action:{action_type}")

    # Role first: "you may not" is a more fundamental answer than "you left a
    # field blank", and it avoids telling a member which fields an action they
    # cannot take would have wanted.
    allowed_roles = ROLE_GATES.get(action_type)
    if allowed_roles is not None and agent.role not in allowed_roles:
        raise ActionRefused(f"role_not_permitted:{agent.role}")

    missing = [p for p in spec.required if not params.get(p)]
    if missing:
        raise ActionRefused(f"missing_parameters:{','.join(missing)}")

    # Objects this action names must be ones the member can already see —
    # otherwise a guessed id would be a way to read or touch private work
    # through the write path.
    #
    # The action's own target is exempt when the action has an explicit gate,
    # because there the gate IS the authority: a lead may merge any pull
    # request, including ones they are neither author nor reviewer of, which is
    # exactly what the autonomous path allows.
    gated = action_type in ROLE_GATES or action_type in CONTEXT_GATES
    for key, value in params.items():
        if key.endswith("_ids") and isinstance(value, list):
            for object_id in value:
                if not isinstance(object_id, str):
                    raise ActionRefused(f"invalid_object_reference:{key}")
                try:
                    kind, obj = resolve_object(world, object_id)
                except ActionRefused:
                    raise ActionRefused(f"unknown_object:{object_id}")
                if not object_visible_to(world, kind, obj, agent_id):
                    raise ActionRefused(f"object_not_visible:{object_id}")
            continue
        if not isinstance(value, str) or key in _NON_OBJECT_PARAMS:
            continue
        if not key.endswith("_id"):
            continue
        if gated and key == spec.target_param:
            continue
        try:
            kind, obj = resolve_object(world, value)
        except ActionRefused:
            continue          # not an object we track; the handler will judge it
        if not object_visible_to(world, kind, obj, agent_id):
            raise ActionRefused(f"object_not_visible:{value}")

    denial = role_denial(world, agent, action_type, params)
    if denial is not None:
        raise ActionRefused(denial)

    channel = params.get("channel_id")
    if channel and channel not in {c.channel_id
                                   for c in vis.visible_channels(world, agent_id)}:
        raise ActionRefused(f"not_a_channel_member:{channel}")

    return params


def submit(runtime: Any, agent_id: str, action_type: str,
           params: Optional[Dict[str, Any]] = None, *,
           execution_mode: str = "direct") -> Any:
    """Validate under the world lock, then execute through the shared pipeline."""
    with runtime.lock:
        checked = validate(runtime.world, agent_id, action_type, params)
        if not human_action_executable(action_type):
            raise ActionRefused(f"unimplemented_action_handler:{action_type}")
        action = ActionCandidate(action_type=action_type, parameters=checked)
        return runtime.submit_action(agent_id, action, execution_mode=execution_mode)


__all__ = [
    "ActionRefused", "resolve_object", "object_visible_to",
    "offers_for_object", "global_offers", "validate", "submit",
]
