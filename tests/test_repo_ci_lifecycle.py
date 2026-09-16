from __future__ import annotations

from types import SimpleNamespace

from environments.org_env.backend.repo.system import RepoLiteSystem
from environments.org_env.runtime_adapter.execution import _pr_needing_ci


def _commit(
    repo: RepoLiteSystem,
    *,
    maker: str,
    branch_id: str,
    message: str,
    path: str,
    tick: int,
) -> str:
    commit = repo.commit_changes(
        agent_id=maker,
        branch_id=branch_id,
        message=message,
        changed_files=[path],
        tick=tick,
        patch_ids=[f"patch_{message}"],
        artifact_ids=[f"artifact_{path}"],
        test_status="pass",
    )
    assert commit is not None
    return commit.commit_id


def _approved_request(repo: RepoLiteSystem, *, maker: str, path: str):
    branch = repo.create_branch(maker, tick=1)
    first_commit = _commit(
        repo,
        maker=maker,
        branch_id=branch.branch_id,
        message="first",
        path=path,
        tick=2,
    )
    pr = repo.open_pr(
        agent_id=maker,
        source_branch=branch.branch_id,
        reviewers=["reviewer"],
    )
    assert repo.approve_pr(
        reviewer_id="reviewer",
        pr_id=pr.pr_id,
        tick=3,
        comment="looks good",
    )
    return branch, pr, first_commit


def test_follow_up_commit_invalidates_ci_and_requires_current_head() -> None:
    repo = RepoLiteSystem()
    branch, pr, first_commit = _approved_request(
        repo,
        maker="maker",
        path="src/cli.py",
    )

    first_ci = repo.run_ci(pr_id=pr.pr_id, tick=4)
    assert first_ci is not None and first_ci.status == "passed"
    assert pr.ci_base_main_commit_ids == ()
    assert pr.review_comments == [
        {
            "review_id": "review_1",
            "reviewer": "reviewer",
            "approve": True,
            "comment": "looks good",
            "tick": 3,
        }
    ]

    second_commit = _commit(
        repo,
        maker="maker",
        branch_id=branch.branch_id,
        message="follow_up",
        path="src/cli.py",
        tick=5,
    )
    assert pr.ci_passed is False
    assert pr.ci_base_main_commit_ids is None

    # A stale boolean restored from a checkpoint cannot promote an untested head.
    pr.ci_passed = True
    assert repo.merge_pr(pr_id=pr.pr_id, tick=6) is False
    assert pr.commit_ids == [first_commit, second_commit]

    current_ci = repo.run_ci(pr_id=pr.pr_id, tick=7)
    assert current_ci is not None and current_ci.commit_id == second_commit
    assert repo.merge_pr(pr_id=pr.pr_id, tick=8) is True
    assert repo.merged_commit_patches(pr) == [
        ("patch_first", "artifact_src/cli.py"),
        ("patch_follow_up", "artifact_src/cli.py"),
    ]
    assert repo.run_ci(pr_id=pr.pr_id, tick=9) is None
    assert repo.merge_pr(pr_id=pr.pr_id, tick=10) is False


def test_missing_ci_base_is_not_an_empty_mainline_attestation() -> None:
    repo = RepoLiteSystem()
    _branch, pr, _first_commit = _approved_request(
        repo,
        maker="maker",
        path="src/cli.py",
    )

    ci = repo.run_ci(pr_id=pr.pr_id, tick=4)
    assert ci is not None and ci.status == "passed"
    assert pr.ci_base_main_commit_ids == ()

    # Pickles from before this field existed must re-run CI rather than treating
    # absent evidence as proof of an empty mainline.
    pr.__dict__.pop("ci_base_main_commit_ids")
    assert repo.merge_pr(pr_id=pr.pr_id, tick=5) is False
    assert pr.ci_passed is False

    rerun = repo.run_ci(pr_id=pr.pr_id, tick=6)
    assert rerun is not None and pr.ci_base_main_commit_ids == ()
    assert repo.merge_pr(pr_id=pr.pr_id, tick=7) is True


def test_mainline_advance_invalidates_another_pr_ci_verdict() -> None:
    repo = RepoLiteSystem()
    requests = []
    for maker, path in (("maker_a", "a.py"), ("maker_b", "b.py")):
        _branch, pr, _first_commit = _approved_request(
            repo,
            maker=maker,
            path=path,
        )
        ci = repo.run_ci(pr_id=pr.pr_id, tick=4)
        assert ci is not None and ci.status == "passed"
        requests.append(pr)

    assert repo.merge_pr(pr_id=requests[0].pr_id, tick=5) is True
    assert repo.merge_pr(pr_id=requests[1].pr_id, tick=6) is False
    assert requests[1].ci_passed is False


def test_sync_pr_commits_carries_late_work_item_links() -> None:
    repo = RepoLiteSystem()
    branch = repo.create_branch("maker", tick=1)
    first = repo.commit_changes(
        agent_id="maker",
        branch_id=branch.branch_id,
        message="first",
        changed_files=["pipeline.py"],
        tick=2,
    )
    assert first is not None
    first.linked_task_ids = ["task_pipeline"]
    first.linked_issue_ids = ["issue_pipeline"]
    pr = repo.open_pr(
        agent_id="maker",
        source_branch=branch.branch_id,
        reviewers=["reviewer"],
    )
    pr.linked_task_ids = list(first.linked_task_ids)
    pr.linked_issue_ids = list(first.linked_issue_ids)

    late = repo.commit_changes(
        agent_id="maker",
        branch_id=branch.branch_id,
        message="dashboard",
        changed_files=["dashboard.py"],
        tick=3,
    )
    assert late is not None
    late.linked_task_ids = ["task_dashboard"]
    late.linked_issue_ids = ["issue_dashboard"]

    assert repo.sync_pr_commits(pr.pr_id) is True
    assert pr.linked_task_ids == ["task_pipeline", "task_dashboard"]
    assert pr.linked_issue_ids == ["issue_pipeline", "issue_dashboard"]


def test_ci_offer_is_bound_to_the_pr_head_and_mainline_base() -> None:
    """Keep the native CI retry guard independent of a global product tree."""

    repo = RepoLiteSystem()
    branch, pr, _first_commit = _approved_request(
        repo,
        maker="maker",
        path="src/cli.py",
    )
    ci = repo.run_ci(pr_id=pr.pr_id, tick=4)
    assert ci is not None and ci.status == "passed"
    world = SimpleNamespace(repo_system=repo)

    # A red/unknown PR whose latest verdict is nevertheless for its current
    # head and current mainline is not repeatedly offered CI. This exercises
    # the per-PR identity without requiring a materialized global tree.
    pr.ci_passed = False
    pr.ci_tree_hash = "unrelated-global-tree-marker"
    assert _pr_needing_ci(world) is None

    # Mainline drift re-offers the request even if an unrelated tree marker is
    # unchanged. Restore the CI base before checking a new branch head too.
    repo.repo.main_commit_ids.append("other_pr_commit")
    assert _pr_needing_ci(world) == pr.pr_id
    repo.repo.main_commit_ids.clear()

    _commit(
        repo,
        maker="maker",
        branch_id=branch.branch_id,
        message="follow_up",
        path="src/cli.py",
        tick=5,
    )
    assert _pr_needing_ci(world) == pr.pr_id

    # Conflicted requests are intentionally not offered a CI action.
    pr.merge_conflict = True
    assert _pr_needing_ci(world) is None
