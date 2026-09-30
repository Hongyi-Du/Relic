"""Synthetic public programs and real RepoLite/source/export/probe execution."""
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace

import pytest

from environments.org_env.backend.comm.system import CommunicationSystem
from environments.org_env.backend.repo.repo import PRStatus, BranchStatus
from environments.org_env.cooperbench import behavior_probes as bp
from environments.org_env.cooperbench import source_views as sv
from environments.org_env.cooperbench import semantic_review as sr
from environments.org_env.cooperbench.joint_probe_replay import (
    replay_accepted_probe_plans as replay, current_accepted_probe_replay_matches as current,
    guard_actor_patch_against_merged_features as guard_patch,
    _compose_cross_feature_probe_definitions,
)
from environments.org_env.product.patch_objects import CodePatch
from environments.org_env.cooperbench.visibility import (
    initialize_private_briefs, share_feature_brief, read_shared_feature_brief,
)
from environments.org_env.product import materialize
from test_cooperbench_source_views import _world, _artifact, _branch, _commit, _merge_ledger, OWNER, PEER


F1, F2 = "cooper_feature_1", "cooper_feature_2"
BRIEFS = {F1: "Add a public function `alpha.value()` that returns 1.",
          F2: "Add a public function `beta.value()` that returns 2."}


def test_cross_feature_constructor_addition_relaxes_only_standalone_exact_field_assertion():
    probes = [{
        "probe_id": F1 + ":6",
        "code": '''
from package import StreamListener, StreamResponse
import inspect
sig = inspect.signature(StreamListener.__init__)
params = list(sig.parameters.keys())
original_fields = {"signature_field_name", "predict", "predict_name", "allow_reuse", "on_chunk"}
actual_init_params = set(params) - {"self"}
assert original_fields == actual_init_params, "Only on_chunk should be added"
expected_response_fields = {"chunk", "signature_field_name", "predict_name", "is_last_chunk"}
actual_response_fields = set(StreamResponse.__annotations__.keys())
assert expected_response_fields == actual_response_fields, "Response schema must stay exact"
''',
    }, {
        "probe_id": F2 + ":8",
        "code": '''
from package import StreamListener
listener = StreamListener(signature_field_name="answer", idle_timeout_s=0.5)
assert listener.idle_timeout_s == 0.5
''',
    }]
    original = deepcopy(probes)

    composed, adjustments = _compose_cross_feature_probe_definitions(probes)

    assert probes == original
    assert adjustments == [{
        "probe_id": F1 + ":6",
        "constructor": "StreamListener",
        "added_keywords": ["idle_timeout_s"],
        "source_features": [F2],
        "transformation": "exact_constructor_fields_to_required_subset",
    }]
    assert "original_fields <= actual_init_params" in composed[0]["code"]
    assert "expected_response_fields == actual_response_fields" in composed[0]["code"]
    assert composed[1] == probes[1]


@pytest.mark.parametrize("other_id,constructor", [
    (F1 + ":8", "StreamListener"),
    (F2 + ":8", "OtherListener"),
])
def test_constructor_exclusivity_is_not_relaxed_without_another_feature_for_same_constructor(
    other_id, constructor
):
    probes = [{
        "probe_id": F1 + ":6",
        "code": '''
from package import StreamListener
import inspect
sig = inspect.signature(StreamListener.__init__)
params = list(sig.parameters.keys())
expected = {"signature_field_name", "on_chunk"}
actual = set(params) - {"self"}
assert expected == actual
''',
    }, {
        "probe_id": other_id,
        "code": (
            f"from package import {constructor}\n"
            f"value = {constructor}(signature_field_name='answer', idle_timeout_s=0.5)\n"
        ),
    }]

    composed, adjustments = _compose_cross_feature_probe_definitions(probes)

    assert composed == probes
    assert adjustments == []


class ForbiddenProvider:
    def __init__(self):
        self.calls = 0

    def generate_json(self, *args, **kwargs):
        self.calls += 1
        raise AssertionError("An accepted-plan replay must not call a provider")


def _runtime(monkeypatch, tmp_path, *, on_run=None, unavailable=False):
    calls = []

    def run(*, root, **kwargs):
        probes = json.loads((root / "probe_input.json").read_text(encoding="utf-8"))["probes"]
        files = {path.relative_to(root / "candidate").as_posix(): path.read_text(encoding="utf-8")
                 for path in (root / "candidate").rglob("*") if path.is_file()}
        calls.append({"files": files, "probes": deepcopy(probes)})
        if unavailable:
            raise RuntimeError("synthetic isolated executor unavailable")
        # Only the tiny public programs authored in this test fixture execute
        # here. Production uses the pinned networkless executor boundary.
        from environments.org_env.cooperbench.probe_runner import run_plan
        rows = run_plan(root, probes)
        raw = json.dumps(rows).encode()
        (root / "probe_results.json").write_bytes(raw)
        if on_run:
            on_run()
        return SimpleNamespace(status="ok", exit_code=0, stderr_tail="", blocked_reason="",
                               stdout_tail="COOPER_BEHAVIOR_RESULTS_SHA256=" + hashlib.sha256(raw).hexdigest())

    executor = SimpleNamespace(run=run, policy=SimpleNamespace(container_image="synthetic-test-runtime", container_platform="test"))
    monkeypatch.setattr(materialize, "_formal_product_executor", lambda: executor)
    monkeypatch.setattr(materialize, "_product_smoke_root", lambda: tmp_path)
    return calls


def _approval(world, feature, pr, path, expected):
    reviewer = PEER if pr.author_id == OWNER else OWNER
    assert world.repo_system.approve_pr(reviewer_id=reviewer, pr_id=pr.pr_id, tick=3)
    contract = bp.visible_review_contract(world, reviewer, feature)
    probes = bp._validate_plan({"probes": [{
        "requirement_ids": list(contract["requirements"]), "paths": [path],
        "compare_baseline": False, "baseline_setup": "", "kind": "runtime",
        "code": f"import {path[:-3]}\nassert {path[:-3]}.value() == {expected}\n",
    }]}, feature_id=feature, requirements=contract["requirements"], compatibility=contract["compatibility"], paths=[path])
    baseline = sv.freeze_baseline(world)
    snapshot = sv.pr_head_snapshot(world, pr)
    feature_base = sv.pr_base_snapshot(world, pr)
    binding = bp._contract_binding(contract)
    world._cooperbench_behavior_plans[feature] = {
        "reviewer_id": reviewer, "paths": [path], "probes": deepcopy(probes), **binding,
        "identity": bp._plan_identity(contract["description"], contract["requirements"],
                                       contract["compatibility"], [path], dict(baseline.files), reviewer, contract),
    }
    # Historical actual-peer approval is fixture state; the replay under test
    # must independently execute these definitions on its selected NEW tree.
    behavior = {"available": True, "ok": True, "error": "", "probes": deepcopy(probes),
                "source_snapshot": snapshot.receipt(), "baseline_snapshot": baseline.receipt(),
                "plan_hash": bp._digest(probes),
                "results": [{"probe_id": probe["probe_id"], "candidate": {"status": "pass"}, "baseline": None}
                            for probe in probes]}
    world._cooperbench_semantic_reviews[feature] = {
        "approved": True, "pr_id": pr.pr_id, "reviewer_id": reviewer, "feature_id": feature,
        "reviewed_pr_revision": sr.pull_request_revision(pr), "source_snapshot": snapshot.receipt(),
        "baseline_snapshot": baseline.receipt(), "feature_base_snapshot": feature_base.receipt(),
        "behavior_evidence": behavior, **binding,
    }


def fixture_world(*, merge_first=True, merge_both=False, share=True):
    world = _world(freeze=False)
    world._cooperbench_sdl_state["feature_owners"] = {F1: OWNER, F2: PEER}
    world.tasks = {}
    world.world_tick = 5
    world.llm_client = ForbiddenProvider()
    world.comm = CommunicationSystem()
    world.comm.seed_default_channels({OWNER, PEER})
    initialize_private_briefs(world, BRIEFS, {F1: OWNER, F2: PEER})
    if share:
        for feature, owner, reader in ((F1, OWNER, PEER), (F2, PEER, OWNER)):
            msg = world.comm.send_message(sender_id=owner, channel_id="team_general", text="Share public feature brief", tick=1)
            attachment = share_feature_brief(world, owner, feature, reader, msg.message_id, 1)
            read_shared_feature_brief(world, reader, attachment["attachment_id"], msg.message_id, 2)
    for name in ("alpha", "beta"):
        path, text = name + ".py", "def value():\n    return 0\n"
        _artifact(world, "art_" + name, path, text)
        world._cooperbench_public_baseline_files[path] = text
    sv.freeze_baseline(world)
    sv.initialize_actor_desks(world)
    world._cooperbench_integrated_probe_workflow = {"schema_version": "cooperbench_integrated_probe_workflow_v1"}
    # Both branches fork the old main BEFORE either feature lands.
    branches = {F1: _branch(world, OWNER), F2: _branch(world, PEER)}
    prs = {}
    world._cooperbench_behavior_plans = {}
    world._cooperbench_semantic_reviews = {}
    for feature, module, value in ((F1, "alpha", 1), (F2, "beta", 2)):
        branch = branches[feature]
        _commit(world, branch, f"def value():\n    return {value}\n", aid="art_" + module)
        pr = world.repo_system.open_pr(agent_id=branch.owner_id, source_branch=branch.branch_id,
                                       reviewers=[PEER if branch.owner_id == OWNER else OWNER])
        pr.linked_issue_ids = [feature]
        sv.freeze_pr_head(world, pr)
        prs[feature] = pr
        if share:
            _approval(world, feature, pr, module + ".py", value)
    if merge_first or merge_both:
        _merge_ledger(world, prs[F1])
    if merge_both:
        _merge_ledger(world, prs[F2])
    return world, prs


def test_old_base_ownhead_and_actual_integration_have_distinct_probe_scopes(monkeypatch, tmp_path):
    world, prs = fixture_world()
    calls = _runtime(monkeypatch, tmp_path)
    contract = bp.visible_review_contract(world, OWNER, F2)
    own = bp.prepare_public_behavior_evidence(
        world, reviewer_id=OWNER, feature_id=F2, description=contract["description"],
        requirements=contract["requirements"], compatibility=contract["compatibility"],
        paths=["beta.py"], pull_request=prs[F2])
    assert own["ok"] and own["evaluation_scope"] == "feature_pr_head"
    assert [probe["probe_id"] for probe in own["probes"]] == [F2 + ":0"]
    assert calls[0]["files"]["alpha.py"].endswith("return 0\n")
    integrated = replay(world, actor_id=OWNER, pull_request=prs[F2])
    assert integrated["available"] and integrated["ok"], integrated
    assert integrated["eligible_features"] == [F1, F2]
    assert calls[1]["files"]["alpha.py"].endswith("return 1\n")
    assert len(calls[1]["probes"]) == 2
    assert integrated["source_snapshot"]["source_kind"] == "merge_candidate"
    assert current(world, actor_id=OWNER, receipt=integrated, pull_request=prs[F2])
    assert world.llm_client.calls == 0


def test_patch_guard_names_probe_with_incomplete_execution_receipt(monkeypatch):
    world, _ = fixture_world()
    before = sv.actor_desk_snapshot(world, PEER)
    synced = sv.sync_actor_desk(world, PEER, expected_snapshot_id=before.snapshot_id)
    patch = CodePatch(
        "candidate_with_incomplete_probe", "art_beta", PEER, world.world_tick,
        new_content="def value():\n    return 2\n",
    )
    patch.actor_source_snapshot = synced.receipt()
    ticket = sv.prepare_actor_patch(
        world, PEER, patch, expected_snapshot_id=synced.snapshot_id
    )

    def incomplete(_world, probes, *, source_snapshot, baseline_snapshot):
        return {
            "source_snapshot": source_snapshot.receipt(),
            "baseline_snapshot": baseline_snapshot.receipt(),
            "plan_hash": bp._digest(probes),
            "results": [{
                "probe_id": probe["probe_id"],
                "candidate": {
                    "status": "unavailable",
                    "error": "candidate_import_failed",
                    "output": "Traceback: NameError: dependency is not defined",
                },
                "baseline": None,
            } for probe in probes],
        }

    monkeypatch.setattr(bp, "_execute_actor_patch_candidate_plan", incomplete)
    result = guard_patch(world, actor_id=PEER, prepared_patch=ticket)
    assert result["available"] is False
    assert result["error"] == "actor_patch_guard_execution_incomplete"
    assert result["incomplete_probe_ids"] == [F1 + ":0"]
    assert result["behavior_evidence"]["results"][0]["candidate"]["error"] == (
        "candidate_import_failed"
    )


def test_approved_but_unmerged_peer_is_not_a_regression_obligation(monkeypatch, tmp_path):
    world, prs = fixture_world(merge_first=False)
    calls = _runtime(monkeypatch, tmp_path)
    # Its approval cache can be completely unavailable: it is not on main.
    world._cooperbench_behavior_plans.pop(F1)
    result = replay(world, actor_id=OWNER, pull_request=prs[F2])
    assert result["ok"], result
    assert result["eligible_features"] == [F2]
    assert [row["probe_id"] for row in calls[0]["probes"]] == [F2 + ":0"]


def test_proposed_private_patch_replays_merged_peer_plan_before_publication(monkeypatch, tmp_path):
    world, _ = fixture_world()
    before = sv.actor_desk_snapshot(world, PEER)
    synced = sv.sync_actor_desk(world, PEER, expected_snapshot_id=before.snapshot_id)
    repo_before = deepcopy(world.repo_system.repo)
    native_before = {aid: deepcopy(artifact.__dict__)
                     for aid, artifact in world.product_artifacts.items()}
    calls = _runtime(monkeypatch, tmp_path)

    regression = CodePatch("candidate_regression", "art_alpha", PEER, world.world_tick,
                           new_content="def value():\n    return 0\n")
    regression.actor_source_snapshot = synced.receipt()
    ticket = sv.prepare_actor_patch(world, PEER, regression,
                                    expected_snapshot_id=synced.snapshot_id)
    rejected = guard_patch(world, actor_id=PEER, prepared_patch=ticket)
    assert rejected["applicable"] and rejected["available"] and not rejected["ok"], rejected
    assert rejected["error"] == "merged_feature_probe_regression"
    assert rejected["protected_features"] == [F1]
    assert calls[0]["files"]["alpha.py"].endswith("return 0\n")
    assert [row["probe_id"] for row in calls[0]["probes"]] == [F1 + ":0"]

    compatible = CodePatch("candidate_compatible", "art_beta", PEER, world.world_tick,
                           new_content="def value():\n    return 2\n")
    compatible.actor_source_snapshot = synced.receipt()
    ticket = sv.prepare_actor_patch(world, PEER, compatible,
                                    expected_snapshot_id=synced.snapshot_id)
    accepted = guard_patch(world, actor_id=PEER, prepared_patch=ticket)
    assert accepted["applicable"] and accepted["available"] and accepted["ok"], accepted
    assert accepted["protected_features"] == [F1]
    assert calls[1]["files"]["alpha.py"].endswith("return 1\n")
    assert world.llm_client.calls == 0
    assert sv.actor_desk_snapshot(world, PEER) == synced
    assert world.repo_system.repo == repo_before
    assert not any(pid.startswith("candidate_") for pid in world.patches)
    assert {aid: artifact.__dict__ for aid, artifact in world.product_artifacts.items()} == native_before


@pytest.mark.parametrize("defective_status", ["pass", "fail"])
@pytest.mark.parametrize("legacy_checkpoint", [False, True])
def test_owner_repair_patch_must_make_peer_validated_failure_green_before_publication(
    monkeypatch, tmp_path, defective_status, legacy_checkpoint
):
    """Reproduce r80: preserving the peer feature cannot stand in for fixing mine."""

    from test_cooperbench_actor_workspace import _result
    from test_cooperbench_source_ci import _public_runtime
    from test_cooperbench_visibility_profile_loop import (
        F1 as LIVE_F1,
        _profile_select,
        _world as live_world,
    )

    world, _ = live_world()
    world.llm_client = ForbiddenProvider()
    sv.freeze_baseline(world)
    sv.initialize_actor_desks(world)
    message = world.comm.send_message(
        sender_id=OWNER, channel_id="team_general", text="share feature", tick=1
    )
    attachment = share_feature_brief(
        world, OWNER, LIVE_F1, PEER, message.message_id, 1
    )
    read_shared_feature_brief(
        world, PEER, attachment["attachment_id"], message.message_id, 2
    )

    # Publish an intentionally incomplete first head through the actual private
    # desk -> public tests -> commit -> PR workflow.
    initial = CodePatch(
        "repair_target_initial", "art_alpha_py", OWNER, world.world_tick,
        new_content="def value():\n    return 0\n", change_summary="initial implementation",
        related_issue_ids=[LIVE_F1], related_task_ids=["task_oss_" + LIVE_F1],
    )
    initial.actor_source_snapshot = sv.actor_desk_snapshot(world, OWNER).receipt()
    applied = _result()
    assert world.apply_product_patch(initial, applied, OWNER), applied.failure_reason
    _public_runtime(monkeypatch, tmp_path, world)
    for action_type in ("run_public_tests", "commit_patch", "open_pr"):
        result = world._loop["execution"].execute(
            OWNER, _profile_select(world, OWNER, action_type), world
        )
        assert result.success, (action_type, result.failure_reason)
        world.world_tick += 1
    pr = next(iter(world.repo_system.repo.pull_requests.values()))
    head = sv.pr_head_snapshot(world, pr)
    baseline = sv.freeze_baseline(world)
    feature_base = sv.pr_base_snapshot(world, pr)
    assert head.tree_digest == sv.actor_desk_snapshot(world, OWNER).tree_digest

    contract = bp.visible_review_contract(world, PEER, LIVE_F1)
    probes = bp._validate_plan(
        {"probes": [{
            "requirement_ids": list(contract["requirements"]),
            "paths": ["alpha.py"], "compare_baseline": False,
            "baseline_setup": "", "kind": "runtime",
            "code": "import alpha\nassert alpha.value() == 2\n",
        }, {
            "requirement_ids": list(contract["requirements"]),
            "paths": ["alpha.py"], "compare_baseline": False,
            "baseline_setup": "", "kind": "runtime",
            "code": "import alpha\nvalue = alpha.value()\nassert isinstance(value, int)\n",
        }, {
            "requirement_ids": list(contract["requirements"]),
            "paths": ["alpha.py"], "compare_baseline": False,
            "baseline_setup": "", "kind": "runtime",
            "code": "import alpha\nassert alpha.value() != 3\n",
        }]},
        feature_id=LIVE_F1, requirements=contract["requirements"],
        compatibility=contract["compatibility"], paths=["alpha.py"],
    )
    binding = bp._contract_binding(contract)
    plan = {
        "reviewer_id": PEER, "paths": ["alpha.py"], "probes": deepcopy(probes),
        "identity": bp._plan_identity(
            contract["description"], contract["requirements"],
            contract["compatibility"], ["alpha.py"], dict(baseline.files),
            PEER, contract,
            policy="peer_public_behavior_probes_v37_implementation_owned_calls" if legacy_checkpoint else None,
        ),
        **binding,
    }
    world.__dict__.setdefault("_cooperbench_behavior_plans", {})[LIVE_F1] = plan
    calls = _runtime(monkeypatch, tmp_path)
    behavior = bp._execute_plan(
        world, probes, source_snapshot=head, baseline_snapshot=baseline
    )
    behavior["available"] = True
    if legacy_checkpoint:
        behavior["policy"] = "peer_public_behavior_probes_v37_implementation_owned_calls"
    behavior["probes"] = deepcopy(probes)
    failed = behavior["results"][0]
    assert failed["candidate"]["status"] == "fail"
    material = sr._probe_validity_material(
        world, reviewer_id=PEER, feature_id=LIVE_F1,
        description=contract["description"],
        obligations=list(contract["requirements"].values()),
        probe=probes[0], result=failed,
    )
    validity = {
        key: deepcopy(material[key])
        for key in (
            "schema_version", "identity", "feature_id", "reviewer_id",
            "probe_id", "brief_identity", "failure_site",
        )
    }
    validity.update({
        "origin": "focused_actual_peer_adjudication", "verdict": "valid",
        "reason": "The requested public value is still absent.",
        "public_basis": contract["description"],
    })
    defective_result = behavior["results"][2]
    assert defective_result["candidate"]["status"] == "pass"
    if defective_status == "fail":
        defective_result["candidate"] = {
            "status": "fail", "output": "Traceback: synthetic probe scaffolding failure",
        }
    defective_material = sr._probe_validity_material(
        world, reviewer_id=PEER, feature_id=LIVE_F1,
        description=contract["description"],
        obligations=list(contract["requirements"].values()),
        probe=probes[2], result=defective_result,
    )
    defective_validity = {
        key: deepcopy(defective_material[key])
        for key in (
            "schema_version", "identity", "feature_id", "reviewer_id",
            "probe_id", "brief_identity", "failure_site",
        )
    }
    defective_validity.update({
        "origin": "focused_actual_peer_adjudication", "verdict": "defective",
        "reason": "This check does not express a required public behavior.", "public_basis": "",
    })
    world._cooperbench_probe_validity_receipts = {
        material["identity"]: validity,
        defective_material["identity"]: defective_validity,
    }
    review = {
        "approved": False, "pr_id": pr.pr_id, "reviewer_id": PEER,
        "feature_id": LIVE_F1, "reviewed_pr_revision": sr.pull_request_revision(pr),
        "source_snapshot": head.receipt(), "baseline_snapshot": baseline.receipt(),
        "feature_base_snapshot": feature_base.receipt(),
        "behavior_evidence": behavior,
        "probe_validity": {
            "schema_version": sr.PROBE_VALIDITY_SCHEMA,
            "receipts": [deepcopy(validity), deepcopy(defective_validity)],
            "unresolved_probe_ids": [],
        },
        "quarantined_review_evidence": {
            "probe_ids": [probes[2]["probe_id"]],
        },
        **binding,
    }
    world.__dict__.setdefault("_cooperbench_semantic_reviews", {})[LIVE_F1] = review
    assert sr.semantic_review_repair_evidence_matches(
        world, PEER, LIVE_F1, review
    )
    bp.invalidate_probe_rows(
        world, LIVE_F1, [probes[2]["probe_id"]],
        defects=[{
            "probe_id": probes[2]["probe_id"],
            "adjudication_status": "confirmed",
        }],
    )
    assert LIVE_F1 not in world._cooperbench_behavior_plans
    assert world._cooperbench_probe_evidence_gaps[LIVE_F1][
        "invalidated_plan"
    ]["probes"] == probes

    desk_before = sv.actor_desk_snapshot(world, OWNER)
    repo_before = deepcopy(world.repo_system.repo)
    unresolved = CodePatch(
        "repair_still_wrong", "art_alpha_py", OWNER, world.world_tick,
        new_content="def value():\n    return 1\n", change_summary="claimed repair",
        related_issue_ids=[LIVE_F1], related_task_ids=["task_oss_" + LIVE_F1],
    )
    unresolved.actor_source_snapshot = desk_before.receipt()
    pending = _result()
    assert world.apply_product_patch(unresolved, pending, OWNER)
    assert pending.failure_reason is None
    guard = pending.state_delta["current_feature_repair_guard"]
    assert guard["repair_feature"] == LIVE_F1
    assert guard["failed_repair_probe_ids"] == [LIVE_F1 + ":0"]
    assert [row["probe_id"] for row in calls[1]["probes"]] == [
        LIVE_F1 + ":0", LIVE_F1 + ":1",
    ]
    assert sv.actor_desk_snapshot(world, OWNER).files["alpha.py"].endswith("return 1\n")
    assert world.repo_system.repo.main_branch == repo_before.main_branch
    assert world.patches[unresolved.patch_id].validation_status == "accepted"
    failure = world._cooperbench_actor_patch_guard_failures[OWNER]
    assert LIVE_F1 in failure["brief"] and LIVE_F1 + ":0" in failure["brief"]

    repaired = CodePatch(
        "repair_actually_green", "art_alpha_py", OWNER, world.world_tick,
        new_content="def value():\n    return 2\n", change_summary="actual repair",
        related_issue_ids=[LIVE_F1], related_task_ids=["task_oss_" + LIVE_F1],
    )
    repaired.actor_source_snapshot = sv.actor_desk_snapshot(world, OWNER).receipt()
    accepted = _result()
    assert world.apply_product_patch(repaired, accepted, OWNER), accepted.failure_reason
    passed_guard = accepted.state_delta["current_feature_repair_guard"]
    assert passed_guard["ok"] is True
    assert passed_guard["repair_probe_ids"] == [LIVE_F1 + ":0"]
    assert passed_guard["preserve_probe_ids"] == [LIVE_F1 + ":1"]
    assert sv.actor_desk_snapshot(world, OWNER).files["alpha.py"].endswith("return 2\n")
    assert not world._cooperbench_actor_patch_guard_failures.get(OWNER)
    history = world._cooperbench_actor_patch_guard_failure_history[OWNER]
    assert len(history) == 1
    assert history[0]["feature_ids"] == [LIVE_F1]

    # The accepted repair remains pending on the same branch.  A second edit
    # cannot exploit the changed desk identity to undo the repair before commit.
    repaired_desk = sv.actor_desk_snapshot(world, OWNER)
    regressed = CodePatch(
        "repair_regressed_before_commit", "art_alpha_py", OWNER, world.world_tick,
        new_content="def value():\n    return 0\n", change_summary="regression before commit",
        related_issue_ids=[LIVE_F1], related_task_ids=["task_oss_" + LIVE_F1],
    )
    regressed.actor_source_snapshot = repaired_desk.receipt()
    regressed_result = _result()
    assert not world.apply_product_patch(regressed, regressed_result, OWNER)
    assert regressed_result.failure_reason == (
        "current_feature_repair_guard_rejected:current_feature_repair_unresolved"
    )
    assert sv.actor_desk_snapshot(world, OWNER) == repaired_desk
    assert world.llm_client.calls == 0
    # Historical review, private pending repair, green repair, rejected regression.
    assert len(calls) == 4


def test_old_dirty_branch_defers_peer_probe_to_integration_replay(monkeypatch, tmp_path):
    world, _ = fixture_world()
    calls = _runtime(monkeypatch, tmp_path)
    before = sv.actor_desk_snapshot(world, PEER)
    patch = CodePatch("old_branch_candidate", "art_beta", PEER, world.world_tick,
                      new_content="def value():\n    return 2\n")
    patch.actor_source_snapshot = before.receipt()
    ticket = sv.prepare_actor_patch(world, PEER, patch,
                                    expected_snapshot_id=before.snapshot_id)
    result = guard_patch(world, actor_id=PEER, prepared_patch=ticket)
    assert result["applicable"] is False and result["ok"] is True
    assert result["classification"] == "not_applicable_actor_base_precedes_mainline"
    assert calls == [] and world.llm_client.calls == 0


def test_final_replays_both_plans_on_exact_committed_main_and_each_call_is_fresh(monkeypatch, tmp_path):
    world, prs = fixture_world(merge_both=True)
    calls = _runtime(monkeypatch, tmp_path)
    for artifact in world.product_artifacts.values():
        artifact.content = "PEER_GLOBAL_PENDING_CANARY"
        artifact.mainline_content = "WRONG_GLOBAL_MAINLINE_CANARY"
    first = replay(world, actor_id=OWNER, require_all_features=True)
    second = replay(world, actor_id=OWNER, require_all_features=True)
    assert first["ok"] and second["ok"], (first, second)
    assert first["plan_set_digest"] == second["plan_set_digest"]
    assert first["replay_id"] != second["replay_id"]
    assert first["execution_sequence"] < second["execution_sequence"]
    assert len(calls) == 2 and calls[0] == calls[1]
    assert first["source_snapshot"] == sv.mainline_snapshot(world).receipt()
    assert current(world, actor_id=OWNER, receipt=first, require_all_features=True)
    assert not current(world, actor_id=PEER, receipt=first, require_all_features=True)
    assert world.llm_client.calls == 0


def test_final_feature_regression_is_executed_and_remains_unadjudicated(monkeypatch, tmp_path):
    world, prs = fixture_world(merge_both=True)
    # Publish a subsequent actual mainline change that regresses F1; neither
    # of the historical feature-head approvals can prove this new tree green.
    branch = _branch(world, PEER)
    _commit(world, branch, "def value():\n    return 99\n", aid="art_alpha")
    extra = world.repo_system.open_pr(agent_id=PEER, source_branch=branch.branch_id, reviewers=[OWNER])
    # This fixture merge is deliberately outside the feature protocol, like a
    # later integration mistake. The source selector still preserves its bytes.
    extra.linked_issue_ids = [F2]
    sv.freeze_pr_head(world, extra)
    _merge_ledger(world, extra)
    _runtime(monkeypatch, tmp_path)
    result = replay(world, actor_id=OWNER, require_all_features=True)
    assert result["available"] and not result["ok"], result
    assert result["classification"] == "unadjudicated_probe_failure"
    assert result["repair_paths"] == []
    assert {row["probe_id"]: row["candidate"]["status"] for row in result["behavior_evidence"]["results"]} == {
        F1 + ":0": "fail", F2 + ":0": "pass"}
    assert current(world, actor_id=OWNER, receipt=result, require_all_features=True, require_passed=False)
    assert not current(world, actor_id=OWNER, receipt=result, require_all_features=True)
    assert not getattr(world, "_cooperbench_probe_validity_receipts", {})


@pytest.mark.parametrize("mutation", ["missing_plan", "changed_code", "missing_definition", "changed_paths", "changed_requirements", "false_approval", "missing_behavior", "missing_hash"])
def test_definitions_must_exactly_match_current_approval_before_execution(monkeypatch, tmp_path, mutation):
    world, prs = fixture_world()
    calls = _runtime(monkeypatch, tmp_path)
    plan, review = world._cooperbench_behavior_plans[F1], world._cooperbench_semantic_reviews[F1]
    if mutation == "missing_plan":
        world._cooperbench_behavior_plans.pop(F1)
    elif mutation == "changed_code":
        plan["probes"][0]["code"] += "# changed definition\n"
    elif mutation == "missing_definition":
        plan["probes"] = []
    elif mutation == "changed_paths":
        plan["probes"][0]["paths"] = ["beta.py"]
    elif mutation == "changed_requirements":
        plan["probes"][0]["requirement_ids"] = ["wrong_req"]
    elif mutation == "false_approval":
        review["approved"] = False
    elif mutation == "missing_behavior":
        review.pop("behavior_evidence")
    else:
        review["behavior_evidence"].pop("plan_hash")
    result = replay(world, actor_id=OWNER, pull_request=prs[F2])
    assert not result["available"] and not result["ok"]
    assert not calls and world.llm_client.calls == 0


@pytest.mark.parametrize("mutation", ["replace_pr", "replace_peer_pr", "revoke_approval", "plan_code", "brief_receipt", "same_bytes_main"])
def test_source_or_plan_changes_during_execution_cannot_preserve_green(monkeypatch, tmp_path, mutation):
    world, prs = fixture_world()

    def mutate():
        if mutation == "replace_pr":
            world.repo_system.repo.pull_requests[prs[F2].pr_id] = deepcopy(prs[F2])
        elif mutation == "replace_peer_pr":
            world.repo_system.repo.pull_requests[prs[F1].pr_id] = deepcopy(prs[F1])
        elif mutation == "revoke_approval":
            prs[F2].approved_by.clear()
        elif mutation == "plan_code":
            world._cooperbench_behavior_plans[F1]["probes"][0]["code"] += "# changed\n"
        elif mutation == "brief_receipt":
            world._cooperbench_private_briefs["reads"][OWNER][F2]["read_tick"] += 1
        else:
            branch = _branch(world, OWNER)
            _commit(world, branch, "def value():\n    return 1\n", aid="art_alpha")
            added = world.repo_system.open_pr(agent_id=OWNER, source_branch=branch.branch_id, reviewers=[PEER])
            added.linked_issue_ids = [F1]
            sv.freeze_pr_head(world, added)
            _merge_ledger(world, added)

    calls = _runtime(monkeypatch, tmp_path, on_run=mutate)
    result = replay(world, actor_id=OWNER, pull_request=prs[F2])
    assert len(calls) == 1
    assert not result["available"] and not result["ok"], result
    assert result["behavior_evidence"]["results"]  # old execution retained only as audit
    assert not current(world, actor_id=OWNER, receipt=result, pull_request=prs[F2])


def test_final_requires_both_actual_merged_features_not_just_approved(monkeypatch, tmp_path):
    world, prs = fixture_world()
    calls = _runtime(monkeypatch, tmp_path)
    result = replay(world, actor_id=OWNER, require_all_features=True)
    assert not result["available"] and result["error"] == "accepted_probe_feature_not_merged"
    prs[F2].status = PRStatus.MERGED
    world.repo_system.repo.branches[prs[F2].source_branch].status = BranchStatus.MERGED
    forged = replay(world, actor_id=OWNER, require_all_features=True)
    assert not forged["available"] and forged["error"] == "accepted_probe_pr_not_in_actual_mainline"
    assert not calls


def test_unread_brief_and_source_snapshot_substitution_never_execute(monkeypatch, tmp_path):
    world, prs = fixture_world()
    calls = _runtime(monkeypatch, tmp_path)
    wrong = replay(world, actor_id=OWNER, pull_request=prs[F2], source_snapshot=sv.pr_head_snapshot(world, prs[F2]))
    assert not wrong["available"]
    world._cooperbench_private_briefs["reads"][OWNER].pop(F2)
    unread = replay(world, actor_id=OWNER, pull_request=prs[F2])
    assert unread["error"] == "accepted_probe_brief_unshared_or_unread"
    assert not calls


def test_executor_outage_is_not_product_failure_or_a_cached_green(monkeypatch, tmp_path):
    world, prs = fixture_world()
    calls = _runtime(monkeypatch, tmp_path, unavailable=True)
    result = replay(world, actor_id=OWNER, pull_request=prs[F2])
    assert len(calls) == 1 and not result["available"] and result["classification"] == "evidence_unavailable"
    assert not result["repair_paths"] and not result.get("replay_id")
    calls = _runtime(monkeypatch, tmp_path)
    fixed = replay(world, actor_id=OWNER, pull_request=prs[F2])
    assert len(calls) == 1 and fixed["ok"]
    forged = deepcopy(fixed)
    forged["behavior_evidence"]["results"][0]["candidate"]["output"] = "changed"
    assert not current(world, actor_id=OWNER, receipt=forged, pull_request=prs[F2])


def test_patch_guard_executor_exception_retains_detail_in_actor_feedback():
    from environments.org_env.cooperbench import joint_probe_replay as jpr
    from environments.org_env.cooperbench.actor_workspace import _patch_guard_failure_brief

    error = RuntimeError("public_behavior_executor_failed:timeout_after_seconds:240")
    receipt = jpr._patch_guard_failure(OWNER, error)
    assert receipt["error"] == "actor_patch_guard_execution_unavailable:RuntimeError"
    assert receipt["error_detail"] == str(error)
    assert str(error) in _patch_guard_failure_brief(receipt)
    assert not receipt["available"] and not receipt["ok"]


@pytest.mark.parametrize("mutation", ["source", "plan", "approval"])
def test_issued_green_receipt_is_invalidated_by_later_source_or_plan_changes(monkeypatch, tmp_path, mutation):
    world, prs = fixture_world()
    calls = _runtime(monkeypatch, tmp_path)
    receipt = replay(world, actor_id=OWNER, pull_request=prs[F2])
    assert receipt["ok"]
    if mutation == "source":
        branch = _branch(world, OWNER)
        _commit(world, branch, "def value():\n    return 1\n", aid="art_alpha")
        extra = world.repo_system.open_pr(agent_id=OWNER, source_branch=branch.branch_id, reviewers=[PEER])
        extra.linked_issue_ids = [F1]
        sv.freeze_pr_head(world, extra)
        _merge_ledger(world, extra)
    elif mutation == "plan":
        world._cooperbench_behavior_plans[F1]["probes"][0]["code"] += "# new definition\n"
    else:
        prs[F1].approved_by.clear()
    assert not current(world, actor_id=OWNER, receipt=receipt, pull_request=prs[F2])
    assert len(calls) == 1  # The currentness guard cannot execute anything.


def test_malformed_or_missing_replay_registration_is_not_a_pass(monkeypatch, tmp_path):
    world, prs = fixture_world()
    monkeypatch.setattr(bp, "_execute_integrated_plan", lambda *args, **kwargs: None)
    missing = replay(world, actor_id=OWNER, pull_request=prs[F2])
    assert missing["error"] == "accepted_probe_execution_receipt_mismatch"
    assert not missing["available"]
    calls = _runtime(monkeypatch, tmp_path)
    world._cooperbench_accepted_probe_replay_receipts = None
    invalid = replay(world, actor_id=OWNER, pull_request=prs[F2])
    assert invalid["error"] == "accepted_probe_receipt_store_invalid"
    assert not calls


def test_conservative_conflict_is_coordination_not_a_failing_product_probe(monkeypatch, tmp_path):
    world, prs = fixture_world()
    branch = world.repo_system.repo.branches[prs[F2].source_branch]
    _commit(world, branch, "def value():\n    return 8\n", aid="art_alpha")
    world.repo_system.sync_pr_commits(prs[F2].pr_id)
    sv.freeze_pr_head(world, prs[F2])
    calls = _runtime(monkeypatch, tmp_path)
    result = replay(world, actor_id=OWNER, pull_request=prs[F2])
    assert result["classification"] == "coordination_conflict", result
    assert not result["available"] and result["conflict_paths"] == ["alpha.py"]
    assert not calls and not result["repair_paths"]
