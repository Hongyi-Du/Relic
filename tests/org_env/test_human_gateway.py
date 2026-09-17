"""What a human seat is allowed to do (HCI P2).

The action space a human gets has to be the one their role actually has —
neither smaller (they would be playing a weaker game than the agent they
replaced) nor larger (they could do things no agent in that seat could, and the
comparison would be meaningless).

That second half is new ground. The execution handlers check almost nothing:
``merge_pr`` and ``publish_product_release`` never look at who called them,
because for autonomous agents the gates live upstream in candidate generation.
A human seat is a second way to construct an action, so the gateway restates
those gates, and these tests pin them against the originals.

Run:  PYTHONPATH="." python tests/org_env/test_human_gateway.py
"""
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent_sdk.lived.core.contracts import ActionCandidate
from environments.org_env.backend.actions import ORG_ACTION_CATEGORIES
from environments.org_env.human import gateway
from environments.org_env.human.affordances import (
    GLOBAL_ACTIONS,
    OBJECT_ACTIONS,
    ROLE_GATES,
    all_offered_action_specs,
    all_offered_action_types,
    human_action_executable,
)
from environments.org_env.human.runtime import HumanModeRuntime
from environments.org_env.runtime_adapter.live import OrgInspectorSession
from environments.org_env.backend.repo.workflow import record_patch
from environments.org_env.product.patch_objects import CodePatch

ENGINEER = "sean"        # fast_engineer: no governance rights
COFOUNDER = "victor"     # cofounder: may propose protocols and cut releases
RELIABILITY = "calvin"


def _runtime(ticks=24):
    s = OrgInspectorSession(seed=42)
    s.step(ticks)
    return HumanModeRuntime(s.world, seconds_per_tick=60.0)


# ---- the table is real --------------------------------------------------- #
def test_every_offered_action_exists_in_the_action_registry():
    """A typo here would not fail loudly: an unknown action_type falls through
    to the generic handler and quietly does nothing recognisable."""
    unknown = [a for a in all_offered_action_types() if a not in ORG_ACTION_CATEGORIES]
    assert not unknown, f"offered actions missing from the registry: {unknown}"


def test_human_gateway_rejects_registry_verbs_without_runtime_semantics():
    """A generic action_event is not proof that the requested effect happened."""
    rt = _runtime(ticks=1)
    try:
        assert human_action_executable("approve_doc") is False
        before = (len(rt.world.action_log), len(rt.world.events))
        try:
            gateway.submit(rt, COFOUNDER, "approve_doc", {"doc_id": "doc_missing"})
        except gateway.ActionRefused as exc:
            assert str(exc) == "unimplemented_action_handler:approve_doc"
        else:
            raise AssertionError("a generic registry event was reported as document approval")
        assert (len(rt.world.action_log), len(rt.world.events)) == before
    finally:
        rt.shutdown()


def test_human_gateway_keeps_revalidation_and_execution_in_one_world_lock():
    """No organization tick may slip between the human check and shared action."""
    source = (Path(__file__).resolve().parents[2] /
              "environments/org_env/human/gateway.py").read_text(encoding="utf-8")
    submit = source.split("def submit", 1)[1].split("__all__", 1)[0]
    locked = submit.split("with runtime.lock:", 1)[1]
    assert "checked = validate" in locked
    assert "return runtime.submit_action" in locked


def test_overloaded_run_ci_keeps_both_pr_and_branch_parameter_contracts():
    variants = [spec for spec in all_offered_action_specs()
                if spec.action_type == "run_ci"]
    assert {(spec.required, spec.optional, spec.target_param) for spec in variants} == {
        (("pr_id",), (), "pr_id"),
        ((), ("branch_id",), "branch_id"),
    }
    rt = _runtime()
    try:
        # Parameter-aware lookup must not demand pr_id for the branch form.
        gateway.validate(rt.world, ENGINEER, "run_ci", {"branch_id": "branch_x"})
    finally:
        rt.shutdown()


def test_a_visible_prs_ci_result_is_a_readable_organization_object():
    rt = _runtime(ticks=1)
    try:
        repo = rt.world.repo_system.repo
        repo.pull_requests["pr_visible_ci"] = types.SimpleNamespace(
            pr_id="pr_visible_ci", author_id=COFOUNDER, reviewers=[],
            ci_run_ids=["ci_visible"])
        repo.ci_runs["ci_visible"] = types.SimpleNamespace(
            ci_id="ci_visible", pr_id="pr_visible_ci", status="failed",
            checks=[], failure_reasons=["public tests failed"])

        kind, ci = gateway.resolve_object(rt.world, "ci_visible")

        assert kind == "ci_run"
        assert gateway.object_visible_to(rt.world, kind, ci, COFOUNDER) is True
        assert gateway.object_visible_to(rt.world, kind, ci, ENGINEER) is False
    finally:
        rt.shutdown()


def test_branch_menu_open_pr_and_run_ci_execute_on_that_exact_branch():
    rt = _runtime(ticks=1)
    try:
        world = rt.world
        execution = world._loop["execution"]
        branch_result = execution.execute(
            ENGINEER,
            ActionCandidate(action_type="create_branch",
                            parameters={"linked_task": "task_00"}),
            world,
        )
        branch_id = branch_result.created_objects[0]
        execution.execute(
            ENGINEER,
            ActionCandidate(action_type="commit_changes",
                            parameters={"branch_id": branch_id}),
            world,
        )
        rt.claim_seat(ENGINEER)

        opened = gateway.submit(rt, ENGINEER, "open_pr", {"branch_id": branch_id})
        assert opened.success is True
        pr_id = opened.created_objects[0]
        assert world.repo_system.repo.pull_requests[pr_id].source_branch == branch_id

        checked = gateway.submit(rt, ENGINEER, "run_ci", {"branch_id": branch_id})
        assert checked.success is True
        ci = world.repo_system.repo.ci_runs[checked.created_objects[0]]
        assert ci.pr_id == pr_id
    finally:
        rt.shutdown()


def test_open_pr_can_carry_the_complete_visible_task_scope():
    rt = _runtime(ticks=1)
    try:
        world = rt.world
        branch = world.repo_system.create_branch(COFOUNDER, tick=0)
        world.repo_system.edit_file(COFOUNDER, branch.branch_id)
        world.repo_system.commit_changes(
            agent_id=COFOUNDER,
            branch_id=branch.branch_id,
            message="whole project implementation",
            changed_files=["project.py"],
            tick=0,
        )
        task_ids = list(world.tasks)[:3]
        rt.claim_seat(COFOUNDER)

        opened = gateway.submit(rt, COFOUNDER, "open_pr", {
            "branch_id": branch.branch_id,
            "linked_task_ids": task_ids,
        })

        assert opened.success is True
        pr = world.repo_system.repo.pull_requests[opened.created_objects[0]]
        assert pr.linked_task_ids == task_ids
    finally:
        rt.shutdown()


def test_commit_patch_targets_the_exact_pending_patch_across_multiple_branches():
    """A worker's patch_id must never commit an older unrelated branch."""
    rt = _runtime(ticks=1)
    try:
        world = rt.world
        rt.claim_seat(COFOUNDER)
        artifacts = [
            artifact for artifact in world.product_artifacts.values()
            if getattr(artifact, "artifact_type", "") == "repo_file"
        ][:2]
        assert len(artifacts) == 2
        branches = []
        for index, artifact in enumerate(artifacts, start=1):
            patch = CodePatch(
                patch_id=f"human_patch_{index}",
                target_object_id=artifact.artifact_id,
                actor_id=COFOUNDER,
                tick=world.world_tick,
                new_content=f"revision {index}",
            )
            world.patches[patch.patch_id] = patch
            branch_id = record_patch(world, COFOUNDER, patch, artifact, world.world_tick)
            assert branch_id
            branches.append(branch_id)

        result = gateway.submit(rt, COFOUNDER, "commit_patch", {
            "patch_id": "human_patch_2",
        })

        assert result.success is True
        commit = world.repo_system.repo.commits[result.created_objects[0]]
        assert commit.branch_id == branches[1]
        assert commit.patch_ids == ["human_patch_2"]
        assert world._pending_by_branch[branches[0]] == [
            ("human_patch_1", artifacts[0].artifact_id),
        ]
        assert world._pending_by_branch[branches[1]] == []
    finally:
        rt.shutdown()


def test_commit_patch_never_falls_back_from_an_explicit_empty_branch():
    rt = _runtime(ticks=1)
    try:
        world = rt.world
        rt.claim_seat(COFOUNDER)
        empty = world.repo_system.create_branch(COFOUNDER, tick=world.world_tick)
        other = world.repo_system.create_branch(COFOUNDER, tick=world.world_tick)
        artifact = next(
            artifact for artifact in world.product_artifacts.values()
            if getattr(artifact, "artifact_type", "") == "repo_file"
        )
        patch = CodePatch(
            patch_id="pending_elsewhere",
            target_object_id=artifact.artifact_id,
            actor_id=COFOUNDER,
            tick=world.world_tick,
            new_content="changed",
        )
        world.patches[patch.patch_id] = patch
        world._pending_by_branch = {
            empty.branch_id: [],
            other.branch_id: [(patch.patch_id, artifact.artifact_id)],
        }

        result = gateway.submit(rt, COFOUNDER, "commit_patch", {
            "branch_id": empty.branch_id,
        })

        assert result.success is False
        assert result.failure_reason == "no_uncommitted_patches_for_branch"
        assert not world.repo_system.repo.commits
        assert world._pending_by_branch[other.branch_id]
    finally:
        rt.shutdown()


def test_branch_targeted_ci_never_falls_back_to_another_open_request():
    """An explicit P1/P2 branch selection is a fail-closed target contract."""
    rt = _runtime(ticks=1)
    try:
        world = rt.world
        execution = world._loop["execution"]

        def committed_branch(task_id):
            created = execution.execute(
                ENGINEER,
                ActionCandidate(action_type="create_branch",
                                parameters={"linked_task": task_id}),
                world,
            )
            branch_id = created.created_objects[0]
            execution.execute(
                ENGINEER,
                ActionCandidate(action_type="commit_changes",
                                parameters={"branch_id": branch_id}),
                world,
            )
            return branch_id

        with_pr = committed_branch("task_00")
        selected_without_pr = committed_branch("task_01")
        rt.claim_seat(ENGINEER)
        opened = gateway.submit(rt, ENGINEER, "open_pr", {"branch_id": with_pr})
        assert opened.success is True
        prior_ci_ids = set(world.repo_system.repo.ci_runs)

        result = gateway.submit(rt, ENGINEER, "run_ci", {
            "branch_id": selected_without_pr,
        })

        assert result.success is False
        assert result.failure_reason == "no_open_pr_for_branch"
        assert set(world.repo_system.repo.ci_runs) == prior_ci_ids
        assert world.repo_system.repo.pull_requests[opened.created_objects[0]].source_branch == with_pr
    finally:
        rt.shutdown()


def test_the_menu_covers_every_category_the_brief_requires():
    """HCI V0 §3: communication, tasks, repo work, experiments and documents,
    and the full proposal/protocol lifecycle."""
    offered = set(all_offered_action_types())
    for required in ("send_message", "reply_thread",           # communication
                     "pick_task", "handoff_task", "update_task_status",
                     "edit_repo_file", "commit_patch", "open_pr", "review_pr",
                     "approve_pr", "merge_pr", "run_ci",       # repo
                     "run_experiment", "save_result", "create_doc", "review_doc",
                     "create_release_candidate", "approve_release_candidate",
                     "publish_product_release",                 # release
                     "propose_protocol", "support_protocol", "oppose_protocol",
                     "follow_protocol", "enforce_protocol", "violate_protocol",
                     "amend_protocol",                          # protocol
                     "approve_proposal", "reject_proposal",
                     "request_proposal_changes"):               # proposal
        assert required in offered, f"the brief requires {required}"


def test_every_object_kind_the_ui_opens_has_a_menu():
    for kind in ("task", "pull_request", "document", "experiment", "meeting",
                 "proposal", "protocol", "release_candidate", "message"):
        assert OBJECT_ACTIONS.get(kind), f"no context menu for {kind}"


def test_actions_that_change_the_organization_ask_for_confirmation():
    """HCI V0 §7: a working agent may prepare these, a human confirms them."""
    must_confirm = {"send_message", "approve_pr", "merge_pr", "approve_proposal",
                    "propose_protocol", "support_protocol", "publish_product_release",
                    "approve_release_candidate", "summarize_decision"}
    specs = {s.action_type: s for s in GLOBAL_ACTIONS}
    for group in OBJECT_ACTIONS.values():
        specs.update({s.action_type: s for s in group})
    unconfirmed = [a for a in must_confirm if a in specs and not specs[a].confirm]
    assert not unconfirmed, f"these should need confirming: {unconfirmed}"


# ---- role symmetry with the autonomous seat ------------------------------ #
def test_a_human_cannot_do_what_their_role_could_not():
    rt = _runtime()
    try:
        world = rt.world
        # Enough parameters that nothing is refused for being incomplete; the
        # role verdict is what this test is about.
        params = {"candidate_id": "rc_x", "title": "t", "protocol_id": "proto_x",
                  "release_id": "rel_x", "rationale": "r"}
        for action_type, allowed_roles in ROLE_GATES.items():
            for agent_id, agent in world.agents.items():
                try:
                    gateway.validate(world, agent_id, action_type, params)
                    refused = None
                except gateway.ActionRefused as exc:
                    refused = str(exc)
                if agent.role in allowed_roles:
                    assert refused != f"role_not_permitted:{agent.role}", (
                        f"{agent_id} ({agent.role}) should be able to {action_type}")
                else:
                    assert refused == f"role_not_permitted:{agent.role}", (
                        f"{agent_id} ({agent.role}) must not be able to {action_type}, "
                        f"got {refused!r}")
    finally:
        rt.shutdown()


def test_an_engineer_cannot_publish_a_release_but_a_founder_can():
    """The concrete case: nothing in the execution layer stops this, so if the
    gateway ever loses the gate a human seat quietly gains a power no agent in
    that seat has."""
    rt = _runtime()
    try:
        try:
            gateway.validate(rt.world, ENGINEER, "publish_product_release",
                             {"candidate_id": "rc_1"})
        except gateway.ActionRefused as exc:
            assert "role_not_permitted" in str(exc)
        else:
            raise AssertionError("a fast_engineer published a release")

        # paul is the founder; the same call must pass the role gate.
        gateway.validate(rt.world, "paul", "publish_product_release",
                         {"candidate_id": "rc_1"})
    finally:
        rt.shutdown()


def test_merge_is_open_to_the_author_or_a_lead():
    """Mirrors execution.py: `pr.author_id == aid or role in (cofounder, founder,
    reliability)`."""
    rt = _runtime()
    try:
        world = rt.world
        from environments.org_env.backend.repo.repo import PullRequest

        pr = PullRequest(pr_id="pr_test", author_id=ENGINEER, source_branch="b",
                         target_branch="main")
        world.repo_system.repo.pull_requests["pr_test"] = pr

        gateway.validate(world, ENGINEER, "merge_pr", {"pr_id": "pr_test"})    # author
        gateway.validate(world, RELIABILITY, "merge_pr", {"pr_id": "pr_test"})  # lead

        pr.reviewers = ["will"]
        try:
            gateway.validate(world, "will", "merge_pr", {"pr_id": "pr_test"})
        except gateway.ActionRefused as exc:
            assert "merge_requires_author_or_lead" in str(exc)
        else:
            raise AssertionError("a non-author non-lead merged")
    finally:
        rt.shutdown()


def test_only_a_designated_approver_can_rule_on_a_proposal():
    rt = _runtime()
    try:
        world = rt.world
        pm = world.proposal_manager
        proposal = next(iter(pm.proposals.values()), None)
        if proposal is None:
            print("  (no proposal in this run; gate covered by the unknown case)")
            try:
                gateway.validate(world, ENGINEER, "approve_proposal",
                                 {"proposal_id": "prop_missing"})
            except gateway.ActionRefused as exc:
                assert "unknown_proposal" in str(exc)
            return
        proposal.approval_required_from = [COFOUNDER]
        proposal.status = "under_review"

        gateway.validate(world, COFOUNDER, "approve_proposal",
                         {"proposal_id": proposal.proposal_id})
        try:
            gateway.validate(world, ENGINEER, "approve_proposal",
                             {"proposal_id": proposal.proposal_id})
        except gateway.ActionRefused as exc:
            assert "not_a_designated_approver" in str(exc)
        else:
            raise AssertionError("a non-approver approved a proposal")
    finally:
        rt.shutdown()


# ---- you cannot act on what you cannot see ------------------------------- #
def test_an_invisible_object_cannot_be_acted_on_by_guessing_its_id():
    from environments.org_env.backend.entities import Task

    rt = _runtime()
    try:
        rt.world.tasks["task_private"] = Task(task_id="task_private", title="secret",
                                              owner_id=RELIABILITY, visibility="private")
        try:
            gateway.validate(rt.world, ENGINEER, "pick_task", {"task_id": "task_private"})
        except gateway.ActionRefused as exc:
            assert "object_not_visible" in str(exc)
        else:
            raise AssertionError("acted on a task the member cannot see")

        # its owner can.
        gateway.validate(rt.world, RELIABILITY, "pick_task", {"task_id": "task_private"})
    finally:
        rt.shutdown()


def test_a_channel_the_member_is_not_in_cannot_be_posted_to():
    rt = _runtime()
    try:
        rt.world.comm.create_channel("ch_private", members={"paul"})
        try:
            gateway.validate(rt.world, ENGINEER, "send_message",
                             {"channel_id": "ch_private", "text": "hello"})
        except gateway.ActionRefused as exc:
            assert "not_a_channel_member" in str(exc)
        else:
            raise AssertionError("posted into a channel the member is not in")
    finally:
        rt.shutdown()


def test_offers_refuse_to_open_a_menu_on_an_invisible_object():
    from environments.org_env.backend.entities import Task

    rt = _runtime()
    try:
        rt.world.tasks["task_hidden"] = Task(task_id="task_hidden", title="secret",
                                             owner_id=RELIABILITY, visibility="private")
        try:
            gateway.offers_for_object(rt.world, ENGINEER, "task_hidden")
        except gateway.ActionRefused as exc:
            assert "object_not_visible" in str(exc)
        else:
            raise AssertionError("opened a menu on an invisible object")
    finally:
        rt.shutdown()


def test_a_menu_explains_a_refusal_rather_than_hiding_the_action():
    rt = _runtime()
    try:
        offers = {o["action_type"]: o for o in gateway.global_offers(rt.world, ENGINEER)}
        blocked = offers["propose_protocol"]
        assert blocked["allowed"] is False
        assert blocked["denied_because"] == "role_not_permitted:fast_engineer"
        assert offers["send_message"]["allowed"] is True
    finally:
        rt.shutdown()


# ---- the whole governance chain, driven by a human ----------------------- #
def test_a_human_can_drive_every_governance_step_their_role_allows():
    """HCI V0 §3: the seat must reach the whole protocol lifecycle — support,
    oppose, apply, enforce, amend — and each step must go through the shared
    pipeline as a human-attributed action.

    Whether a step *succeeds* is the organization's call, not the seat's: the
    handlers apply the same triggers and cooldowns they apply to agents. What
    is asserted here is reach and attribution, not privilege.
    """
    rt = _runtime()
    try:
        rt.claim_seat(COFOUNDER)
        rt.claim_seat(RELIABILITY)
        world = rt.world
        pid = next(iter(world.protocol_registry.protocols))

        steps = [
            (RELIABILITY, "support_protocol", {"protocol_id": pid}),
            (COFOUNDER, "follow_protocol", {"protocol_id": pid}),
            (RELIABILITY, "enforce_protocol", {"protocol_id": pid}),
            (COFOUNDER, "oppose_protocol", {"protocol_id": pid, "reason": "too strict"}),
            (COFOUNDER, "amend_protocol", {"protocol_id": pid,
                                           "rationale": "relax for hotfixes"}),
            (COFOUNDER, "propose_protocol",
             {"title": "declare ownership before a major change",
              "problem_evidence": "two members rebuilt the same module"}),
        ]
        for agent_id, action_type, params in steps:
            gateway.submit(rt, agent_id, action_type, params)

        logged = {c["action_type"] for c in world.controller_log}
        assert {a for _, a, _ in steps} == logged
        assert all(c["controller_type"] == "human" for c in world.controller_log)
        # The protocol ledger saw a human act on it, not just the action log.
        assert any(getattr(e, "actor_id", None) in (COFOUNDER, RELIABILITY)
                   for e in getattr(world.protocol_registry, "events", []))
    finally:
        rt.shutdown()


def test_a_human_message_reaches_the_organization():
    rt = _runtime()
    try:
        rt.claim_seat(ENGINEER)
        before = len(rt.world.comm.messages)
        out = gateway.submit(rt, ENGINEER, "send_message",
                             {"channel_id": "team_general",
                              "text": "picking up the sorting crash"})
        assert out.success, out.failure_reason
        assert len(rt.world.comm.messages) > before

        # ...and another member can see it, because it went through the real
        # communication system rather than a side channel.
        seen = rt.world.comm.perceivable_messages(RELIABILITY)
        assert any("sorting crash" in (m.full_text or m.text_summary) for m in seen)
    finally:
        rt.shutdown()


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print("  ok ", fn.__name__)
    print(f"All {len(fns)} gateway tests passed!")
