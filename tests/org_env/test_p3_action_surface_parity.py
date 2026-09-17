"""Machine-check every registered organization action through P1/P2/P3.

``ORG_ACTION_CATEGORIES`` is the ordinary-member action vocabulary exposed by
``OrgActionMapper.available_actions``.  Every verb remains visible to the
secretary, but only verbs with action-specific runtime semantics may cross the
human confirmation boundary.  A registry-only audit event must never be
reported as the requested effect.

Run: PYTHONPATH="." python tests/org_env/test_p3_action_surface_parity.py
"""
import sys
import ast
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from environments.org_env.backend.actions import ORG_ACTION_CATEGORIES
from environments.org_env.human import gateway
from environments.org_env.human.affordances import (
    all_offered_action_specs,
    all_offered_action_types,
    REGISTRY_SURFACE_ACTIONS,
    registry_action_parity_matrix,
)
from environments.org_env.human.execution_team import ExecutionTeam
from environments.org_env.human.working_agent import (
    WorkingAgentSession,
    _DECISION_SCHEMA,
)
from environments.org_env.runtime_adapter.execution import OrgExecutionAdapter
from environments.org_env.runtime_adapter.live import OrgInspectorSession


SEAT = "victor"


def _runtime():
    session = OrgInspectorSession(seed=42)
    session.step(1)
    from environments.org_env.human.runtime import HumanModeRuntime
    runtime = HumanModeRuntime(session.world, seconds_per_tick=60.0)
    runtime.claim_seat(SEAT)
    return runtime


def _sample_value(parameter):
    """Schema-shaped data for a structural compiler/worker contract test.

    These are intentionally not live object ids.  The test verifies that every
    P1/P2 schema reaches the two model compilers; actual visibility, role, and
    current-state authorization remains the live gateway's responsibility.
    """
    if parameter in {"participants", "mentions", "required_steps"}:
        return ["visible_member"]
    if parameter == "due_tick":
        return 1
    if parameter == "status":
        return "available"
    if parameter in {"severity", "urgency", "importance"}:
        return "normal"
    if parameter.endswith("_id"):
        return f"visible_{parameter}"
    if parameter == "file_path":
        return "tools/visible_file.py"
    return f"value for {parameter}"


def _schema_key(spec):
    return (spec.action_type, tuple(spec.required), tuple(spec.optional), spec.target_param)


def _model_params(spec):
    return {parameter: _sample_value(parameter)
            for parameter in (*spec.required, *spec.optional)}


def _direct_handler_parameter_names():
    """Extract the declared ``p.get('field')`` contract from runtime handlers.

    This is not intent parsing: it is a static parity check that prevents the
    model compiler from rejecting a parameter that the shared executor itself
    reads.  Alias assignments intentionally need no row here because the
    aliased handler's declaration is checked under its canonical definition.
    """
    # Use the checked-in source rather than relying on the process working
    # directory or an installed copy of the package.
    source = Path(__file__).resolve().parents[2] / "environments/org_env/runtime_adapter/execution.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    result = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not node.name.startswith("_h_"):
            continue
        names = set()
        for call in ast.walk(node):
            if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                    and call.func.attr in {"get", "pop", "setdefault"}
                    and isinstance(call.func.value, ast.Name)
                    and call.func.value.id in {"p", "params"}
                    and call.args and isinstance(call.args[0], ast.Constant)
                    and isinstance(call.args[0].value, str)):
                continue
            names.add(call.args[0].value)
        result[node.name.removeprefix("_h_")] = names
    return result


def test_p3_p1_p2_action_parity_matrix_is_complete_and_confirmation_gated():
    """Build and assert the P3 parity matrix rather than a hand-picked sample.

    Per executable schema row this proves four links:

    ``affordance -> liaison model catalog -> liaison draft compiler ->
    execution-worker validator -> confirmed gateway call``.

    The final link is intercepted at the gateway boundary, so this test cannot
    mutate the organization while still proving that no side channel bypasses
    confirmation or changes the requested schema.
    """
    specs = all_offered_action_specs()
    catalog = WorkingAgentSession.action_catalog()
    catalog_by_key = {
        (row["action_type"], tuple(row["required"]), tuple(row["optional"]),
         row["target_param"]): row
        for row in catalog
    }
    matrix = []
    for spec in specs:
        key = _schema_key(spec)
        params = _model_params(spec)
        action_type = spec.action_type
        specific_handler = hasattr(OrgExecutionAdapter, f"_h_{action_type}")
        matrix.append({
            "schema": key,
            "action_type": action_type,
            "params": params,
            "registry": action_type in ORG_ACTION_CATEGORIES,
            "liaison_catalog": key in catalog_by_key,
            "runtime_route": "handler" if specific_handler else "generic_registry",
            "worker_action": (ExecutionTeam._validated_action({
                "action_type": action_type, "params": params,
                "rationale": "contract fixture",
            }) if specific_handler else None),
        })

    # Do not hard-code the old 82 forms.  The source of truth is the real
    # ordinary-member registry, and run_ci deliberately has two target forms.
    assert len(matrix) == len(specs)
    assert len({row["schema"] for row in matrix}) == len(specs)
    assert all(row["registry"] for row in matrix)
    assert all(row["liaison_catalog"] for row in matrix)
    assert all(row["runtime_route"] in {"handler", "generic_registry"}
               for row in matrix)
    assert all(catalog_by_key[row["schema"]]["human_executable"] ==
               (row["runtime_route"] == "handler") for row in matrix)
    assert all(row["worker_action"]["action_type"] == row["action_type"]
               and row["worker_action"]["params"] == row["params"]
               for row in matrix if row["runtime_route"] == "handler")
    for row in matrix:
        if row["runtime_route"] == "handler":
            continue
        try:
            ExecutionTeam._validated_action({
                "action_type": row["action_type"], "params": row["params"],
                "rationale": "contract fixture",
            })
        except ValueError as exc:
            assert str(exc) == f"unimplemented_action_handler:{row['action_type']}"
        else:
            raise AssertionError(f"Worker accepted registry-only action {row['action_type']}")

    # The model protocol has a structured draft channel, but deliberately no
    # independent execution action field.  The compiler below is the boundary
    # that rejects invented/out-of-schema action types and parameter names.
    assert "drafts" in _DECISION_SCHEMA["properties"]
    assert "action_type" in _DECISION_SCHEMA["properties"]["drafts"]["items"]["properties"]
    assert "params" in _DECISION_SCHEMA["properties"]["drafts"]["items"]["properties"]

    runtime = _runtime()
    try:
        liaison = WorkingAgentSession(runtime, SEAT)
        action_log_before = len(runtime.world.action_log)
        # The production compiler intentionally limits one model response to
        # twelve actions.  Chunking here checks every schema without weakening
        # that safety limit.
        executable = [row for row in matrix if row["runtime_route"] == "handler"]
        unsupported = [row for row in matrix if row["runtime_route"] == "generic_registry"]
        prepared = []
        for start in range(0, len(executable), 12):
            raw = [{
                "action_type": row["action_type"], "params": row["params"],
                "rationale": "contract fixture",
            } for row in executable[start:start + 12]]
            drafts, error = liaison._prepare_compiled_drafts(raw, request="contract fixture")
            assert error == ""
            prepared.extend(drafts)
        assert len(prepared) == len(executable)
        assert len(runtime.world.action_log) == action_log_before, \
            "draft compilation must be zero-mutation"

        for row in unsupported:
            drafts, error = liaison._prepare_compiled_drafts([{
                "action_type": row["action_type"], "params": row["params"],
                "rationale": "contract fixture",
            }], request="contract fixture")
            assert drafts == []
            assert error == (
                f"{catalog_by_key[row['schema']]['label']} is registered but has no "
                "action-specific runtime handler yet; no confirmation draft was created.")
        assert len(runtime.world.action_log) == action_log_before

        calls = []
        original_submit = gateway.submit

        def capture_submit(runtime_arg, agent_id, action_type, params, *, execution_mode):
            calls.append((runtime_arg, agent_id, action_type, dict(params), execution_mode))
            return SimpleNamespace(
                success=True, action_id=f"action_{len(calls)}", failure_reason="",
                created_objects=[], modified_objects=[],
            )

        gateway.submit = capture_submit
        try:
            for draft in prepared:
                confirmed = liaison.confirm(draft.draft_id)
                assert confirmed["ok"] is True
        finally:
            gateway.submit = original_submit

        assert len(calls) == len(executable)
        expected = [(row["action_type"], row["params"]) for row in executable]
        actual = [(action_type, params) for _runtime, _agent, action_type, params, _mode in calls]
        assert actual == expected
        assert {agent_id for _runtime, agent_id, _action, _params, _mode in calls} == {SEAT}
        assert {mode for _runtime, _agent, _action, _params, mode in calls} == {"liaison_assisted"}
        assert len(runtime.world.action_log) == action_log_before, \
            "only the real gateway may mutate the world"
    finally:
        runtime.shutdown()


def test_every_registered_action_has_an_auditable_secretary_parity_row():
    """No registry-only action may be silently classified away as internal.

    The matrix records the code-evidenced runtime route for *every* original
    verb, including legacy aliases and registry-only verbs. Thus a newly
    registered action cannot silently look implemented merely because the
    shared adapter can append a generic event.
    """
    offered = set(all_offered_action_types())
    registered = set(ORG_ACTION_CATEGORIES)
    matrix = registry_action_parity_matrix()
    rows = {row["action_type"]: row for row in matrix}

    assert offered == registered
    assert set(rows) == registered
    assert all(row["classification"] == "exposed_shared_pipeline"
               for row in matrix)
    assert all(row["schemas"] for row in matrix)
    assert all(row["runtime_route"] in {"handler", "generic_registry"}
               for row in matrix)
    assert all("seat-visible gateway" in row["authority"] for row in matrix)

    # Coverage is by the actual registry taxonomy, not a cherry-picked sample.
    assert {row["category"] for row in matrix} == set(ORG_ACTION_CATEGORIES.values())


def test_registry_generated_global_actions_never_bypass_human_confirmation():
    """Long-tail verbs are reachable, but cannot execute from one form click."""
    assert REGISTRY_SURFACE_ACTIONS
    assert all(spec.confirm for spec in REGISTRY_SURFACE_ACTIONS)


def test_p3_schema_does_not_hide_any_direct_runtime_handler_parameter():
    """A handler parameter is part of its action's structured language."""
    declared = {}
    for spec in all_offered_action_specs():
        declared.setdefault(spec.action_type, set()).update(spec.required + spec.optional)
    missing = {
        action_type: sorted(fields - declared[action_type])
        for action_type, fields in _direct_handler_parameter_names().items()
        if action_type in declared and fields - declared[action_type]
    }
    assert missing == {}


def test_p1_p2_global_offers_expose_the_surface_and_disable_generic_events():
    """The P3 catalog and human UI share one honest capability surface."""
    runtime = _runtime()
    try:
        offers = gateway.global_offers(runtime.world, SEAT)
        offers_by_type = {str(offer["action_type"]): offer for offer in offers}
        assert {spec.action_type for spec in REGISTRY_SURFACE_ACTIONS} <= set(offers_by_type)
        for row in registry_action_parity_matrix():
            if row["action_type"] not in offers_by_type:
                continue
            offer = offers_by_type[row["action_type"]]
            if row["runtime_route"] == "generic_registry":
                assert offer["allowed"] is False
                assert offer["denied_because"] == (
                    f"unimplemented_action_handler:{row['action_type']}")
    finally:
        runtime.shutdown()


def test_each_registry_category_uses_the_live_gateway_for_allow_and_refusal():
    """Exercise a real gateway decision for every registry category.

    The accepted calls prove the new verbs are not merely model-catalog JSON.
    The paired unknown-seat refusal is intentionally made at the same gateway
    boundary for each category, proving the route does not fall through to a
    secretary-only executor before membership, role/context, and target checks
    can run.  Existing focused gateway tests cover the stronger role, channel,
    and invisible-object refusals; this is the category-completeness harness.
    """
    runtime = _runtime()
    try:
        from environments.org_env.proposals.objects import Proposal

        world = runtime.world
        proposal_id = "proposal_p3_gateway_parity"
        world.proposal_manager.proposals[proposal_id] = Proposal(
            proposal_id=proposal_id,
            proposal_type="task_proposal",
            title="P3 live gateway parity",
            status="under_review",
            approval_required_from=[SEAT],
        )
        by_category = {
            "work": ("inspect_task_board", {}),
            "comm": ("defer_until_work_hours", {}),
            "meeting": ("schedule_meeting", {"title": "P3 parity check"}),
            "repo": ("create_branch", {}),
            "sandbox": ("run_cheap_pilot", {}),
            "search": ("external_community_search", {"query": "known issue"}),
            "doc": ("create_onboarding_doc", {}),
            "artifact": ("create_customer_triage_sheet", {}),
            "protocol": ("propose_protocol", {"title": "P3 parity protocol"}),
            "time": ("rest_offline", {}),
            "payroll": ("ask_about_payroll", {}),
            "bridge": ("read_external_feed", {}),
            "governance": ("approve_proposal", {"proposal_id": proposal_id}),
            "release": ("create_release_candidate", {}),
        }
        assert set(by_category) == set(ORG_ACTION_CATEGORIES.values())
        for category, (action_type, params) in by_category.items():
            assert gateway.validate(world, SEAT, action_type, params) == params, category
            try:
                gateway.validate(world, "not_a_member", action_type, params)
            except gateway.ActionRefused as exc:
                assert str(exc) == "unknown_member:not_a_member", category
            else:
                raise AssertionError(f"{category} bypassed the live gateway")
    finally:
        runtime.shutdown()


def test_newly_exposed_registry_verbs_keep_live_target_and_channel_boundaries():
    """Registry completion must not make guessed targets or channels writable."""
    from environments.org_env.backend.entities import Task

    runtime = _runtime()
    try:
        world = runtime.world
        world.tasks["task_p3_private"] = Task(
            task_id="task_p3_private", title="not Victor-visible",
            owner_id="calvin", visibility="private",
        )
        try:
            gateway.validate(world, SEAT, "debug_code", {"task_id": "task_p3_private"})
        except gateway.ActionRefused as exc:
            assert str(exc) == "object_not_visible:task_p3_private"
        else:
            raise AssertionError("registry action bypassed target visibility")

        world.comm.create_channel("channel_p3_private", members={"paul"})
        try:
            gateway.validate(world, SEAT, "share_external_post", {
                "post_id": "untracked_post", "channel_id": "channel_p3_private",
            })
        except gateway.ActionRefused as exc:
            assert str(exc) == "not_a_channel_member:channel_p3_private"
        else:
            raise AssertionError("registry action bypassed channel membership")
    finally:
        runtime.shutdown()


if __name__ == "__main__":
    test_p3_p1_p2_action_parity_matrix_is_complete_and_confirmation_gated()
    test_every_registered_action_has_an_auditable_secretary_parity_row()
    test_p3_schema_does_not_hide_any_direct_runtime_handler_parameter()
    test_p1_p2_global_offers_expose_the_surface_and_disable_generic_events()
    test_each_registry_category_uses_the_live_gateway_for_allow_and_refusal()
    test_newly_exposed_registry_verbs_keep_live_target_and_channel_boundaries()
    print("All P3 action-surface parity tests passed!")
