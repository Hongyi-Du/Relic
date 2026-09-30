"""Committed-head/integration CI and delivery reachability, without providers."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from environments.org_env.cooperbench import source_ci, source_views as sv
from environments.org_env.cooperbench.worker import _export_joint_mainline_patch
from environments.org_env.llm.client import OrgLLMClient
from environments.org_env.product import contracts, materialize
from environments.org_env.product.patch_objects import CodePatch
from environments.org_env.runtime_adapter.execution import (
    _cooperbench_action_block_reason, _cooperbench_current_ci_failure,
)
from test_cooperbench_source_views import (
    BASE, DESK, HEAD, OWNER, PEER, _artifact, _branch, _commit, _merge_ledger, _pr, _published,
)
from test_cooperbench_visibility_profile_loop import (
    F1, F2, _native_pool, _profile_select, _world as _profile_world,
)
from test_cooperbench_worker import _git


@pytest.fixture(autouse=True)
def _no_provider(monkeypatch):
    monkeypatch.setattr(OrgLLMClient, "generate_json", lambda *args, **kwargs: pytest.fail("CI tests cannot call a provider"))


def _public_runtime(monkeypatch, tmp_path, world, response=None):
    """Keep real public-test hashing/export/cache; replace only process launch."""
    calls = []

    def launch(*, repo_dir, **kwargs):
        files = {p.relative_to(repo_dir).as_posix(): p.read_text(encoding="utf-8")
                 for p in Path(repo_dir).rglob("*") if p.is_file()}
        calls.append(files)
        return response(files, len(calls)) if response else (0, "1 passed in 0.01s", "", None)

    world._cooperbench_public_runtime_preflight = {"passed": True}
    monkeypatch.setattr(materialize, "declared_public_test_command", lambda w: ["python", "-m", "pytest"])
    monkeypatch.setattr(materialize, "_product_smoke_root", lambda: str(tmp_path))
    monkeypatch.setattr(materialize, "_formal_product_executor", lambda: SimpleNamespace())
    monkeypatch.setattr(materialize, "public_test_sandbox_readiness", lambda *args: "")
    monkeypatch.setattr(materialize, "_execute_smoke_command", launch)
    return calls


def _unrelated_main_commit(world):
    peer = _branch(world, PEER)
    _commit(world, peer, "MERGED_README_CANARY\n", aid="art_readme")
    peer_pr = _pr(world, peer)
    _merge_ledger(world, peer_pr)
    return peer_pr


def _attest(pr, result):
    pr.ci_passed = bool(result["ok"])
    pr.ci_source_snapshot = result["source_snapshot"]
    pr.ci_merge_snapshot = result["merge_snapshot"]
    pr.ci_tree_hash = result["merge_snapshot"]["tree_digest"]
    pr.ci_validation_strength = result.get(
        "validation_strength", "public_regression"
    )


def test_ci_tests_two_explicit_trees_with_real_export_not_global_pending(monkeypatch, tmp_path):
    world, _, _, _, pr, head = _published()
    _unrelated_main_commit(world)
    world.product_artifacts["art_widget"].content = DESK
    world.product_artifacts["art_readme"].content = "PENDING_README_CANARY"
    _artifact(world, "pending_new", "pending.py", "PENDING_NEW_CANARY", new=True)
    calls = _public_runtime(monkeypatch, tmp_path, world)
    result = source_ci.run_pr_source_ci(world, pr)
    merged = sv.conservative_merge_candidate(world, pr)
    assert result["ok"]
    assert calls == [dict(head.files), dict(merged.files)]
    assert result["checks"]["pr_head"]["source_snapshot"] == head.receipt()
    assert result["checks"]["merge_candidate"]["source_snapshot"] == merged.receipt()
    assert result["source_snapshot"]["tree_digest"] != result["merge_snapshot"]["tree_digest"]
    assert "PEER_UNCOMMITTED" not in str(calls) and "PENDING_" not in str(calls)
    _attest(pr, result)
    assert source_ci.current_source_ci_matches(world, pr)


def test_source_integrity_ci_is_current_but_never_labeled_public_regression(
    monkeypatch, tmp_path
):
    world, _, _, _, pr, _ = _published()
    world._cooperbench_public_runtime_preflight = {
        "passed": True,
        "public_validation_strength": "source_snapshot_integrity",
        "functional_public_regression_available": False,
    }
    _public_runtime(monkeypatch, tmp_path, world)
    # The helper sets the ordinary preflight shape; restore the degraded
    # capability after its process-launch fixture is installed.
    world._cooperbench_public_runtime_preflight.update({
        "public_validation_strength": "source_snapshot_integrity",
        "functional_public_regression_available": False,
    })

    result = source_ci.run_pr_source_ci(world, pr)
    _attest(pr, result)

    assert result["ok"] is True
    assert result["validation_strength"] == "source_snapshot_integrity"
    assert result["functional_public_regression_available"] is False
    assert source_ci.current_source_ci_matches(world, pr)


def test_same_head_launcher_outage_reexecutes_only_failed_exact_tree(monkeypatch, tmp_path):
    world, _, _, _, pr, _ = _published()
    _unrelated_main_commit(world)
    unrelated = {"error": "unrelated old launch outage"}
    world._public_test_cache = {"unrelated-tree": unrelated}

    def response(files, count):
        if count == 2:
            return None, "", "", "container temporarily unavailable"
        return 0, "1 passed in 0.01s", "", None

    calls = _public_runtime(monkeypatch, tmp_path, world, response)
    first = source_ci.run_pr_source_ci(world, pr)
    assert not first["ok"] and first["kind"] == "infrastructure_error"
    assert first["checks"]["pr_head"]["ok"] and len(calls) == 2
    first_head_result = first["checks"]["pr_head"]
    second = source_ci.run_pr_source_ci(world, pr)
    assert second["ok"] and len(calls) == 3
    assert calls[1] == calls[2] and calls[0] != calls[1]
    assert second["checks"]["pr_head"] == first_head_result
    assert world._public_test_cache["unrelated-tree"] is unrelated


def test_integration_failure_is_not_misreported_as_head_failure(monkeypatch, tmp_path):
    world, _, _, _, pr, _ = _published()
    _unrelated_main_commit(world)

    def response(files, count):
        if files["README.md"]:
            return 1, "FAILED tests/test_public.py::test_integrated\n1 failed in 0.01s", "AssertionError", None
        return 0, "1 passed in 0.01s", "", None

    calls = _public_runtime(monkeypatch, tmp_path, world, response)
    result = source_ci.run_pr_source_ci(world, pr)
    assert len(calls) == 2 and result["checks"]["pr_head"]["ok"]
    assert not result["checks"]["merge_candidate"]["ok"]
    assert result["kind"] == "contract_break"
    assert result["boundary"] == "cooperbench_merge_candidate_public_tests"


@pytest.mark.parametrize("change", ["new_main", "new_pr_head"])
def test_old_green_ci_is_invalid_after_either_source_revision_changes(monkeypatch, tmp_path, change):
    world, branch, _, _, pr, _ = _published()
    _public_runtime(monkeypatch, tmp_path, world)
    _attest(pr, source_ci.run_pr_source_ci(world, pr))
    assert source_ci.current_source_ci_matches(world, pr)
    if change == "new_main":
        _unrelated_main_commit(world)
    else:
        _commit(world, branch, "NEW_PR_HEAD = 3\n")
        world.repo_system.sync_pr_commits(pr.pr_id)
        sv.freeze_pr_head(world, pr)
    assert not source_ci.current_source_ci_matches(world, pr)


def test_peer_pending_does_not_block_review_of_fixed_head_with_current_ci(monkeypatch, tmp_path):
    world, _, _, patch, pr, _ = _published()
    world._cooperbench_sdl_state["feature_owners"] = {F1: OWNER, F2: PEER}
    world._cooperbench_feature_issue_by_agent = {OWNER: F1, PEER: F2}
    pr.linked_issue_ids = [F1]
    patch.related_issue_ids = [F1]
    _public_runtime(monkeypatch, tmp_path, world)
    _attest(pr, source_ci.run_pr_source_ci(world, pr))
    world._public_tests_last_hash = "unrelated-working-tree"
    world._public_tests_last = {"passed": False, "failed_tests": ["peer pending broke working tree"]}
    world.product_artifacts["art_widget"].content = DESK
    assert source_ci.current_source_ci_matches(world, pr)
    for action in ("review_pr", "formal_pr_review", "approve_pr"):
        assert _cooperbench_action_block_reason(world, PEER, action, {"pr_id": pr.pr_id}) is None


def test_conservative_source_conflict_is_coordination_not_product_or_provider(monkeypatch, tmp_path):
    world, _, _, _, pr, _ = _published()
    peer = _branch(world, PEER)
    _commit(world, peer, "DIFFERENT_PEER_MAIN = 7\n")
    _merge_ledger(world, _pr(world, peer))
    calls = _public_runtime(monkeypatch, tmp_path, world)
    result = source_ci.run_pr_source_ci(world, pr)
    assert not result["ok"] and result["kind"] == "coordination_conflict"
    assert result["boundary"] == "cooperbench_source_identity" and calls == []
    ci = SimpleNamespace(status="passed", failure_reasons=[])
    pr.test_status = "passed"
    status = contracts.record_integration_verdict(ci, pr, result, world=world)
    assert status != "failed" and ci.status != "failed"
    assert pr.ci_passed is False and pr.test_status != "failed"
    assert not getattr(world, "_gate_stall", {})


def test_final_joint_patch_uses_merged_ledger_even_when_global_text_is_unreadable(tmp_path):
    world, _, _, _, pr, _ = _published()
    _merge_ledger(world, pr)
    _artifact(world, "pending_new", "pending.py", "PENDING_NEW_CANARY", new=True)

    class MetadataOnly:
        artifact_id = "art_widget"
        artifact_type = "repo_file"
        linked_file_path = "src/widget.py"
        created_as_new_file = False
        mainline_revision = 0  # Mutable revision counters are not the merged ledger.

        @property
        def content(self):
            raise AssertionError("final export cannot read global pending")

        @property
        def mainline_content(self):
            raise AssertionError("final export cannot read mutable mainline text")

    world.product_artifacts["art_widget"] = MetadataOnly()
    repo = tmp_path / "image_base"
    (repo / "src").mkdir(parents=True)
    (repo / "src/widget.py").write_text(BASE, encoding="utf-8")
    (repo / "README.md").write_text("", encoding="utf-8")
    for args in [("init",), ("config", "user.email", "test@example.com"), ("config", "user.name", "Test"),
                 ("add", "."), ("commit", "-m", "public fixture")]:
        _git(repo, *args)
    patch, paths = _export_joint_mainline_patch(world, repo, tmp_path / "overlay")
    assert paths == ["src/widget.py"] and "+COMMITTED_OWNER_CANARY = 2" in patch
    assert "PENDING_" not in patch and "pending.py" not in patch
    assert (repo / "src/widget.py").read_text(encoding="utf-8") == BASE


def _native_red_ci_world():
    world, _ = _profile_world()
    sv.freeze_baseline(world)
    branch = _branch(world, OWNER)
    commit, patch = _commit(world, branch, "VALUE = 1\n", aid="art_alpha_py")
    patch.related_issue_ids = [F1]
    patch.related_task_ids = ["task_oss_" + F1]
    pr = _pr(world, branch)
    pr.linked_issue_ids = [F1]
    sv.freeze_pr_head(world, pr)
    ci = world.repo_system.run_ci(pr_id=pr.pr_id, tick=10)
    ci.status = "failed"
    ci.failure_reasons = ["alpha.py:1: AssertionError public behavior differs"]
    pr.ci_passed = False
    pr.test_status = "failed"
    pr.ci_brief = ci.failure_reasons[0]
    pr.ci_base_main_commit_ids = tuple(world.repo_system.repo.main_commit_ids)
    pr.ci_source_snapshot = sv.pr_head_snapshot(world, pr).receipt()
    pr.ci_merge_snapshot = sv.conservative_merge_candidate(world, pr).receipt()
    pr.ci_tree_hash = pr.ci_merge_snapshot["tree_digest"]
    world.product_artifacts["art_alpha_py"].content = "VALUE = 1\n"
    world.world_tick = 11
    return world, branch, pr, ci


def test_native_profile_loop_can_test_and_commit_owner_pending_repair_without_greening_old_ci(monkeypatch, tmp_path):
    world, branch, pr, ci = _native_red_ci_world()
    old_snapshot_id = pr.ci_source_snapshot["snapshot_id"]
    _public_runtime(monkeypatch, tmp_path, world)
    assert _cooperbench_current_ci_failure(world, OWNER, F1)
    _profile_select(world, OWNER, "edit_repo_file")
    repair = CodePatch("after_ci_repair", "art_alpha_py", OWNER, 11, new_content="VALUE = 2\n",
        validation_status="accepted", related_issue_ids=[F1], related_task_ids=["task_oss_" + F1])
    world.patches[repair.patch_id] = repair
    world.__dict__.setdefault("_pending_by_branch", {})[branch.branch_id] = [(repair.patch_id, "art_alpha_py")]
    world.product_artifacts["art_alpha_py"].content = repair.new_content
    world.world_tick = 12
    candidate = _profile_select(world, OWNER, "run_public_tests")
    assert candidate.action_type == "run_public_tests"
    assert pr.ci_passed is False and ci.status == "failed"
    assert not source_ci.current_source_ci_matches(world, pr)
    result = world._loop["execution"].execute(OWNER, candidate, world)
    assert result.success, result.failure_reason
    world.world_tick += 1
    candidate = _profile_select(world, OWNER, "commit_patch")
    assert candidate.action_type == "commit_patch"
    assert pr.ci_passed is False and ci.status == "failed"
    result = world._loop["execution"].execute(OWNER, candidate, world)
    assert result.success, result.failure_reason
    assert repair.patch_id in world.repo_system.repo.commits[branch.commit_ids[-1]].patch_ids
    assert pr.ci_passed is False and not source_ci.current_source_ci_matches(world, pr)
    world.world_tick += 1
    candidate = _profile_select(world, OWNER, "run_ci")
    result = world._loop["execution"].execute(OWNER, candidate, world)
    assert result.success, result.failure_reason
    fresh_ci = world.repo_system.repo.ci_runs[pr.ci_run_ids[-1]]
    assert fresh_ci.ci_id != ci.ci_id and fresh_ci.commit_id == branch.commit_ids[-1]
    assert ci.status == "failed" and fresh_ci.status == "passed"
    head = sv.pr_head_snapshot(world, pr)
    merged = sv.conservative_merge_candidate(world, pr)
    assert head.snapshot_id != old_snapshot_id and head.files["alpha.py"] == repair.new_content
    assert pr.ci_source_snapshot == head.receipt()
    assert pr.ci_merge_snapshot == merged.receipt()
    assert result.state_delta["source_ci"]["checks"]["pr_head"]["source_snapshot"] == head.receipt()
    assert result.state_delta["source_ci"]["checks"]["merge_candidate"]["source_snapshot"] == merged.receipt()
    assert source_ci.current_source_ci_matches(world, pr)


@pytest.mark.parametrize("invalid", ["before_ci", "other_branch", "comment_only", "peer_authored", "rejected"])
def test_native_red_ci_not_released_by_unrelated_or_nonsemantic_pending_rows(monkeypatch, tmp_path, invalid):
    world, branch, pr, ci = _native_red_ci_world()
    _public_runtime(monkeypatch, tmp_path, world)
    text = "VALUE = 1\n# comment only\n" if invalid == "comment_only" else "VALUE = 2\n"
    repair = CodePatch("not_a_current_repair", "art_alpha_py", PEER if invalid == "peer_authored" else OWNER,
        9 if invalid == "before_ci" else 11, new_content=text,
        validation_status="rejected" if invalid == "rejected" else "accepted", related_issue_ids=[F1])
    world.patches[repair.patch_id] = repair
    pending_branch = _branch(world, OWNER).branch_id if invalid == "other_branch" else branch.branch_id
    world.__dict__.setdefault("_pending_by_branch", {})[pending_branch] = [(repair.patch_id, "art_alpha_py")]
    world.product_artifacts["art_alpha_py"].content = text
    world.world_tick = 12
    assert _cooperbench_current_ci_failure(world, OWNER, F1)
    _, candidates = _native_pool(world, OWNER)
    assert not {"run_public_tests", "commit_patch"}.intersection(item.action_type for item in candidates)
    assert pr.ci_passed is False and ci.status == "failed"
