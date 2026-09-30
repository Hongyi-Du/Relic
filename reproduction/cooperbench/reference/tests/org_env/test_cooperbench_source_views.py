"""Public canaries over real RepoLite objects; no benchmark answers/providers."""
import pickle
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace

import pytest

from environments.org_env.backend.repo.repo import BranchStatus, PRStatus
from environments.org_env.backend.repo.system import RepoLiteSystem
from environments.org_env.cooperbench import source_views as sv
from environments.org_env.cooperbench.source_views import (
    SourceViewError,
    actor_artifact_catalog,
    actor_desk_projection,
    actor_desk_snapshot,
    actor_desks_enabled,
    actor_file_text,
    actor_patch_candidate_snapshot,
    assert_actor_desk_current,
    assert_pr_head_current,
    bind_actor_branch,
    commit_actor_patch,
    conservative_merge_candidate,
    freeze_baseline,
    freeze_branch_base,
    freeze_pr_head,
    initialize_actor_desks,
    mainline_snapshot,
    pr_base_snapshot,
    pr_head_snapshot,
    prepare_actor_patch,
    restart_actor_desk_after_conflict,
    snapshot_projection,
    source_views_enabled,
    sync_actor_desk,
)
from environments.org_env.product.materialize import export_product_repo
from environments.org_env.product.objects import ProductArtifact
from environments.org_env.product.patch_objects import CodePatch

OWNER = "victor"
PEER = "calvin"
BASE = "PUBLIC_BASELINE_CANARY = 1\n"
HEAD = "COMMITTED_OWNER_CANARY = 2\n"
DESK = "PEER_UNCOMMITTED_CANARY = 99\n"


def _artifact(world, aid, path, content="", *, new=False):
    artifact = ProductArtifact(aid, "repo_file", path, "active",
                               linked_file_path=path, content=content,
                               mainline_content=content, created_as_new_file=new)
    world.product_artifacts[aid] = artifact
    return artifact


def _world(*, freeze=True):
    world = SimpleNamespace(repo_system=RepoLiteSystem(), product_artifacts={},
                            patches={}, _pending_by_branch={},
                            _cooperbench_sdl_state={"feature_owners": {"f1": OWNER, "f2": PEER}},
                            _cooperbench_public_baseline_files={"src/widget.py": BASE, "README.md": ""})
    _artifact(world, "art_widget", "src/widget.py", BASE)
    _artifact(world, "art_readme", "README.md", "")
    if freeze:
        freeze_baseline(world)
    return world


def _branch(world, owner=OWNER, *, freeze=True):
    branch = world.repo_system.create_branch(owner, tick=1)
    if freeze:
        freeze_branch_base(world, branch.branch_id)
    return branch


def _commit(world, branch, text=HEAD, aid="art_widget", *, creates=False,
            deletes=False, pid=None):
    pid = pid or f"patch_{len(world.patches)}"
    patch = CodePatch(pid, aid, branch.owner_id, 2, new_content=text,
                       creates_file=creates, validation_status="accepted")
    if deletes:
        # Current native schema has no deletion field. Only an explicit trusted
        # ledger tombstone is supported; empty text never implies deletion.
        patch.deletes_file = True
    world.patches[pid] = patch
    commit = world.repo_system.commit_changes(
        agent_id=branch.owner_id, branch_id=branch.branch_id, message="public canary",
        changed_files=[world.product_artifacts[aid].linked_file_path],
        patch_ids=[pid], artifact_ids=[aid], tick=2)
    return commit, patch


def _pr(world, branch):
    return world.repo_system.open_pr(agent_id=branch.owner_id,
                                    source_branch=branch.branch_id, reviewers=[PEER])


def _published():
    world = _world()
    branch = _branch(world)
    commit, patch = _commit(world, branch)
    pr = _pr(world, branch)
    snapshot = freeze_pr_head(world, pr)
    return world, branch, commit, patch, pr, snapshot


def _merge_ledger(world, pr):
    # The selector does not run CI or approve. This fixture sets the actual
    # post-merge ledger fields without manufacturing a provider/evaluator claim.
    repo = world.repo_system.repo
    repo.main_commit_ids.extend(pr.commit_ids)
    for cid in pr.commit_ids:
        repo.commits[cid].status = "merged"
    pr.status = PRStatus.MERGED
    repo.branches[pr.source_branch].status = BranchStatus.MERGED


def test_baseline_is_immutable_and_pickle_safe_including_empty_file():
    world = _world()
    baseline = freeze_baseline(world)
    assert baseline.files == {"src/widget.py": BASE, "README.md": ""}
    with pytest.raises(TypeError):
        baseline.files["src/widget.py"] = DESK
    with pytest.raises(FrozenInstanceError):
        baseline.tree_digest = "wrong"
    receipt = baseline.receipt()
    receipt["commit_ids"].append("fake")
    assert baseline.commit_ids == ()
    for restored in (pickle.loads(pickle.dumps(world)), deepcopy(world)):
        assert freeze_baseline(restored) == baseline


def test_unrelated_native_world_is_unchanged_and_not_opted_in():
    world = SimpleNamespace(product_artifacts={}, patches={})
    before = dict(world.__dict__)
    assert not source_views_enabled(world)
    with pytest.raises(SourceViewError, match="not_cooperbench"):
        freeze_baseline(world)
    assert world.__dict__ == before
    world._cooperbench_source_views = None
    assert source_views_enabled(world)  # A corrupt opt-in cannot select legacy.


def test_missing_baseline_never_reads_desk_or_mutates_world():
    world = _world(freeze=False)
    del world._cooperbench_public_baseline_files
    with pytest.raises(SourceViewError, match="public_baseline_missing"):
        freeze_baseline(world)
    assert not source_views_enabled(world)


def test_baseline_map_cannot_be_replaced_after_freeze():
    world = _world()
    world._cooperbench_public_baseline_files["src/widget.py"] = DESK
    with pytest.raises(SourceViewError, match="public_baseline_changed"):
        freeze_baseline(world)


def test_pr_view_excludes_all_own_pending_peer_pending_and_peer_committed_heads(tmp_path):
    world, branch, _commit_obj, _patch, pr, snapshot = _published()
    peer = _branch(world, PEER)
    _commit(world, peer, DESK)
    _pr(world, peer)  # Even an unmerged published peer head is not our base.
    own_pending = CodePatch("own_pending", "art_widget", OWNER, 3,
                            new_content="OWN_PENDING_CANARY", validation_status="accepted")
    peer_pending = CodePatch("peer_pending", "art_readme", PEER, 3,
                             new_content=DESK, validation_status="accepted")
    world.patches.update(own_pending=own_pending, peer_pending=peer_pending)
    world._pending_by_branch = {
        branch.branch_id: [("own_pending", "art_widget")],
        peer.branch_id: [("peer_pending", "art_readme")],
    }
    world.product_artifacts["art_widget"].content = DESK
    world.product_artifacts["art_widget"].mainline_content = "WRONG_GLOBAL_MAIN_CANARY"
    world.product_artifacts["art_readme"].content = DESK
    _artifact(world, "art_peer_new", "src/peer.py", DESK, new=True)
    projection = snapshot_projection(world, snapshot)
    assert projection == {"overrides": {"art_widget": HEAD, "art_readme": ""},
                          "exclude_artifact_ids": frozenset({"art_peer_new"})}
    assert pr_head_snapshot(world, pr) == snapshot
    for prefer_mainline in (False, True):
        export_product_repo(world, str(tmp_path), prefer_mainline=prefer_mainline, **projection)
        assert (tmp_path / "src/widget.py").read_text() == HEAD
        assert (tmp_path / "README.md").read_bytes() == b""
        assert not (tmp_path / "src/peer.py").exists()


def test_real_export_does_not_read_global_content_properties(tmp_path):
    world, _, _, _, _, snapshot = _published()

    class MetadataOnly:
        artifact_id = "art_widget"
        artifact_type = "repo_file"
        linked_file_path = "src/widget.py"
        created_as_new_file = False

        @property
        def content(self):
            raise AssertionError("global desk must not be read")

        @property
        def mainline_content(self):
            raise AssertionError("global mainline must not be read")

    world.product_artifacts["art_widget"] = MetadataOnly()
    export_product_repo(world, str(tmp_path), **snapshot_projection(world, snapshot))
    assert (tmp_path / "src/widget.py").read_text() == HEAD


def test_old_head_survives_new_commit_and_new_publication():
    world, branch, _, _, pr, old = _published()
    _commit(world, branch, text="NEW_HEAD_CANARY\n")
    with pytest.raises(SourceViewError, match="pr_not_synced_to_branch"):
        assert_pr_head_current(world, pr, old)
    assert pr_head_snapshot(world, pr, require_current=False) == old
    world.repo_system.sync_pr_commits(pr.pr_id)
    with pytest.raises(SourceViewError, match="pr_head_changed"):
        pr_head_snapshot(world, pr)
    new = freeze_pr_head(world, pr)
    assert new.snapshot_id != old.snapshot_id
    assert snapshot_projection(world, old)["overrides"]["art_widget"] == HEAD
    assert new.files["src/widget.py"] == "NEW_HEAD_CANARY\n"


def test_same_tree_new_commit_still_changes_revision_identity():
    world, branch, _, _, pr, old = _published()
    _commit(world, branch, HEAD)
    world.repo_system.sync_pr_commits(pr.pr_id)
    new = freeze_pr_head(world, pr)
    assert old.tree_digest == new.tree_digest
    assert old.revision_identity != new.revision_identity


def test_same_ids_but_mutated_patch_bytes_cannot_reuse_review():
    world, _, _, patch, pr, snapshot = _published()
    patch.new_content = DESK
    with pytest.raises(SourceViewError, match="pr_head_changed"):
        assert_pr_head_current(world, pr, snapshot)
    assert snapshot.files["src/widget.py"] == HEAD


def test_replaced_pr_object_is_not_accepted_through_stale_pointer():
    world, _, _, _, pr, snapshot = _published()
    world.repo_system.repo.pull_requests[pr.pr_id] = deepcopy(pr)
    with pytest.raises(SourceViewError, match="pr_record_not_registered"):
        assert_pr_head_current(world, pr, snapshot)
    assert pr_head_snapshot(world, pr.pr_id) == snapshot


def test_mutating_commit_lifecycle_status_does_not_change_source_identity():
    world, _, commit, _, pr, snapshot = _published()
    _merge_ledger(world, pr)
    assert commit.status == "merged"
    assert_pr_head_current(world, pr, snapshot)


def test_branch_base_is_then_current_main_not_latest_global_main_or_tick_guess():
    world, _, _, _, first_pr, _ = _published()
    old_branch = _branch(world, PEER)
    old_base = freeze_branch_base(world, old_branch.branch_id)
    _merge_ledger(world, first_pr)
    world.product_artifacts["art_widget"].mainline_content = "UNTRUSTED_GLOBAL_MAIN"
    new_branch = _branch(world, PEER)
    new_base = freeze_branch_base(world, new_branch.branch_id)
    assert old_base.files["src/widget.py"] == BASE
    assert new_base.files["src/widget.py"] == HEAD
    assert new_base.base_main_commit_ids == tuple(first_pr.commit_ids)
    _commit(world, new_branch, text="CURRENT_FEATURE_ONLY\n", aid="art_readme")
    new_pr = _pr(world, new_branch)
    freeze_pr_head(world, new_pr)
    assert pr_base_snapshot(world, new_pr) == new_base
    assert pr_base_snapshot(world, new_pr).files["src/widget.py"] == HEAD
    assert pr_head_snapshot(world, new_pr).files["README.md"] == "CURRENT_FEATURE_ONLY\n"
    _commit(world, old_branch, text="old_branch_readme", aid="art_readme")
    old_head = freeze_pr_head(world, _pr(world, old_branch))
    assert old_head.files["src/widget.py"] == BASE


def test_existing_committed_branch_without_frozen_base_cannot_be_reconstructed():
    world = _world()
    branch = _branch(world, freeze=False)
    _commit(world, branch)
    with pytest.raises(SourceViewError, match="branch_base_unfrozen"):
        freeze_branch_base(world, branch.branch_id)
    with pytest.raises(SourceViewError, match="branch_base_unfrozen"):
        freeze_pr_head(world, _pr(world, branch))


def test_unpublished_head_is_not_implicitly_frozen_by_reader():
    world = _world()
    branch = _branch(world)
    _commit(world, branch)
    pr = _pr(world, branch)
    with pytest.raises(SourceViewError, match="pr_head_unfrozen"):
        pr_head_snapshot(world, pr)
    assert world._cooperbench_source_views["pr_heads"] == {}


def test_new_empty_file_updates_and_explicit_deletion_are_distinct(tmp_path):
    world = _world()
    branch = _branch(world)
    _artifact(world, "new_art", "src/new.py", "WRONG_GLOBAL_NEW", new=True)
    _commit(world, branch, text="", aid="new_art", creates=True)
    pr = _pr(world, branch)
    empty_new = freeze_pr_head(world, pr)
    assert empty_new.files["src/new.py"] == ""
    export_product_repo(world, str(tmp_path), prefer_mainline=True,
                        **snapshot_projection(world, empty_new))
    assert (tmp_path / "src/new.py").exists()
    assert (tmp_path / "src/new.py").read_bytes() == b""
    # Native edits of an unmerged newly created file repeat creates_file=True.
    _commit(world, branch, text="NEW_COMMITTED_CANARY\n", aid="new_art", creates=True)
    world.repo_system.sync_pr_commits(pr.pr_id)
    updated = freeze_pr_head(world, pr)
    assert updated.files["src/new.py"] == "NEW_COMMITTED_CANARY\n"
    _commit(world, branch, text="", aid="new_art", deletes=True)
    _commit(world, branch, text="", aid="art_widget")
    world.repo_system.sync_pr_commits(pr.pr_id)
    deleted = freeze_pr_head(world, pr)
    assert "src/new.py" not in deleted.files
    assert deleted.files["src/widget.py"] == ""
    export_product_repo(world, str(tmp_path), **snapshot_projection(world, deleted))
    assert not (tmp_path / "src/new.py").exists()
    assert (tmp_path / "src/widget.py").exists()
    assert deleted.tree_digest != empty_new.tree_digest


@pytest.mark.parametrize("fault, expected", [
    ("missing_commit", "commit_ledger_missing"),
    ("missing_patch", "patch_ledger_missing"),
    ("missing_artifact_alignment", "commit_patch_artifact_alignment_missing"),
    ("wrong_artifact_alignment", "patch_target_or_author_mismatch"),
    ("wrong_patch_author", "patch_target_or_author_mismatch"),
    ("wrong_commit_author", "commit_owner_or_branch_mismatch"),
    ("wrong_pr_author", "pr_owner_or_target_mismatch"),
    ("rejected_patch", "patch_not_accepted"),
    ("reverted_commit", "commit_status_invalid"),
    ("non_text_patch", "patch_content_or_operation_invalid"),
    ("unsynced_patches", "pr_patch_ledger_mismatch"),
    ("duplicate_commits", "pr_commits_invalid"),
    ("duplicate_patches", "commit_patches_invalid"),
    ("invalid_create", "create_path_already_exists"),
    ("invalid_tombstone", "delete_tombstone_invalid"),
])
def test_incomplete_or_inconsistent_ledgers_fail_closed(fault, expected):
    world, branch, commit, patch, pr, _ = _published()
    if fault == "missing_commit":
        del world.repo_system.repo.commits[commit.commit_id]
    elif fault == "missing_patch":
        del world.patches[patch.patch_id]
    elif fault == "missing_artifact_alignment":
        commit.artifact_ids = []
    elif fault == "wrong_artifact_alignment":
        commit.artifact_ids = ["art_readme"]
    elif fault == "wrong_patch_author":
        patch.actor_id = PEER
    elif fault == "wrong_commit_author":
        commit.author_id = PEER
    elif fault == "wrong_pr_author":
        pr.author_id = PEER
    elif fault == "rejected_patch":
        patch.validation_status = "rejected"
    elif fault == "reverted_commit":
        commit.status = "reverted"
    elif fault == "non_text_patch":
        patch.new_content = None
    elif fault == "unsynced_patches":
        pr.patch_ids = []
    elif fault == "duplicate_commits":
        pr.commit_ids *= 2
    elif fault == "duplicate_patches":
        commit.patch_ids *= 2
        commit.artifact_ids *= 2
    elif fault == "invalid_create":
        patch.creates_file = True
    elif fault == "invalid_tombstone":
        patch.deletes_file = True
    with pytest.raises(SourceViewError, match=expected):
        freeze_pr_head(world, pr)


def test_missing_early_branch_history_is_not_a_valid_complete_pr():
    world, branch, first, _, pr, _ = _published()
    second, second_patch = _commit(world, branch, text="SECOND")
    pr.commit_ids = [second.commit_id]
    pr.patch_ids = [second_patch.patch_id]
    with pytest.raises(SourceViewError, match="pr_not_synced_to_branch"):
        freeze_pr_head(world, pr)
    world.repo_system.repo.main_commit_ids = [second.commit_id]
    second.status = "merged"
    another = _branch(world, PEER, freeze=False)
    with pytest.raises(SourceViewError, match="branch_commit_history_incomplete"):
        freeze_branch_base(world, another.branch_id)


def test_missing_new_file_create_record_cannot_fall_back_to_global_artifact():
    world = _world()
    branch = _branch(world)
    _artifact(world, "new", "src/new.py", DESK, new=True)
    _commit(world, branch, "looks valid", aid="new", creates=False)
    with pytest.raises(SourceViewError, match="edit_path_missing_from_base"):
        freeze_pr_head(world, _pr(world, branch))


def test_materialization_requires_surviving_metadata_for_each_frozen_file():
    world, _, _, _, _, snapshot = _published()
    del world.product_artifacts["art_widget"]
    with pytest.raises(SourceViewError, match="snapshot_artifact_missing_for_projection"):
        snapshot_projection(world, snapshot)


@pytest.mark.parametrize("path, expected", [
    ("../escape.py", "artifact_path_invalid"),
    ("src/renamed.py", "artifact_path_changed"),
])
def test_path_mutations_are_rejected_before_export(path, expected):
    world, _, _, _, _, snapshot = _published()
    world.product_artifacts["art_widget"].linked_file_path = path
    with pytest.raises(SourceViewError, match=expected):
        snapshot_projection(world, snapshot)


def test_case_alias_is_rejected_before_export():
    world, _, _, _, _, snapshot = _published()
    _artifact(world, "duplicate", "SRC/WIDGET.PY", DESK)
    with pytest.raises(SourceViewError, match="artifact_path_collision"):
        snapshot_projection(world, snapshot)


def test_corrupted_snapshot_registry_cannot_be_read_or_materialized():
    world, _, _, _, pr, snapshot = _published()
    corrupt = replace(snapshot, _file_entries=(("src/widget.py", DESK),))
    world._cooperbench_source_views["snapshots"][snapshot.snapshot_id] = corrupt
    with pytest.raises(SourceViewError, match="snapshot_integrity_invalid"):
        pr_head_snapshot(world, pr, require_current=False)
    with pytest.raises(SourceViewError, match="snapshot_integrity_invalid"):
        snapshot_projection(world, snapshot)


def test_repeated_snapshot_reads_hash_once_but_replacements_revalidate(monkeypatch):
    world, _, _, _, _, snapshot = _published()
    real_tree_digest = sv._tree_digest
    calls = 0

    def counted_tree_digest(entries, runtime_assets_digest):
        nonlocal calls
        calls += 1
        return real_tree_digest(entries, runtime_assets_digest)

    monkeypatch.setattr(sv, "_tree_digest", counted_tree_digest)
    assert sv._get(world, snapshot.snapshot_id) == snapshot
    assert calls == 1
    for _ in range(10):
        assert sv._get(world, snapshot.snapshot_id) == snapshot
    assert calls == 1

    corrupt = replace(snapshot, _file_entries=(("src/widget.py", DESK),))
    world._cooperbench_source_views["snapshots"][snapshot.snapshot_id] = corrupt
    with pytest.raises(SourceViewError, match="snapshot_integrity_invalid"):
        sv._get(world, snapshot.snapshot_id)
    assert calls == 2

    world._cooperbench_source_views["snapshots"][snapshot.snapshot_id] = snapshot
    assert sv._get(world, snapshot.snapshot_id) == snapshot
    object.__setattr__(snapshot, "_file_entries", (("src/widget.py", "forged"),))
    with pytest.raises(SourceViewError, match="snapshot_integrity_invalid"):
        sv._get(world, snapshot.snapshot_id)
    assert calls == 3


def test_tree_digest_is_order_independent_but_distinguishes_empty_from_missing():
    first, second = _world(), _world(freeze=False)
    second._cooperbench_public_baseline_files = dict(reversed(
        list(second._cooperbench_public_baseline_files.items())))
    assert freeze_baseline(first).tree_digest == freeze_baseline(second).tree_digest
    third = _world(freeze=False)
    del third._cooperbench_public_baseline_files["README.md"]
    assert freeze_baseline(first).tree_digest != freeze_baseline(third).tree_digest


def test_committed_patch_also_marked_pending_fails_closed():
    world, branch, _, patch, pr, _ = _published()
    world._pending_by_branch[branch.branch_id] = [(patch.patch_id, "art_widget")]
    with pytest.raises(SourceViewError, match="committed_patch_still_pending"):
        freeze_pr_head(world, pr)


def test_dropped_branch_commit_in_both_pr_and_branch_still_fails_closed():
    world, branch, _, _, pr, _ = _published()
    second, patch = _commit(world, branch, "SECOND")
    branch.commit_ids = pr.commit_ids = [second.commit_id]
    pr.patch_ids = [patch.patch_id]
    with pytest.raises(SourceViewError, match="branch_commit_history_incomplete"):
        freeze_pr_head(world, pr)


def test_missing_entire_merged_branch_in_main_ledger_fails_closed():
    world, _, _, _, pr, _ = _published()
    _merge_ledger(world, pr)
    world.repo_system.repo.main_commit_ids = []
    next_branch = _branch(world, PEER, freeze=False)
    with pytest.raises(SourceViewError, match="main_commit_history_incomplete"):
        freeze_branch_base(world, next_branch.branch_id)
    with pytest.raises(SourceViewError, match="main_commit_history_incomplete"):
        conservative_merge_candidate(world, pr)


def test_merge_candidate_retains_nonoverlapping_peer_main_and_exact_pr(tmp_path):
    world, _, _, _, pr, head = _published()
    peer = _branch(world, PEER)
    _commit(world, peer, "PEER_MERGED_README\n", aid="art_readme")
    peer_pr = _pr(world, peer)
    freeze_pr_head(world, peer_pr)
    _merge_ledger(world, peer_pr)
    world.product_artifacts["art_widget"].content = DESK
    world.product_artifacts["art_readme"].mainline_content = "WRONG_GLOBAL_MAIN"
    _artifact(world, "uncommitted_new", "pending.py", DESK, new=True)
    candidate = conservative_merge_candidate(world, pr)
    assert candidate.source_kind == "merge_candidate"
    assert candidate.base_snapshot_id == head.snapshot_id
    assert candidate.base_main_commit_ids == tuple(peer_pr.commit_ids)
    assert candidate.files == {"src/widget.py": HEAD, "README.md": "PEER_MERGED_README\n"}
    export_product_repo(world, str(tmp_path), **snapshot_projection(world, candidate))
    assert (tmp_path / "README.md").read_text() == "PEER_MERGED_README\n"
    assert not (tmp_path / "pending.py").exists()


def test_merge_candidate_builds_each_large_source_projection_once(monkeypatch):
    world = _world(freeze=False)
    for index in range(64):
        path = f"src/generated_{index:03d}.py"
        artifact_id = f"art_generated_{index:03d}"
        content = f"VALUE_{index} = {index}\n"
        world._cooperbench_public_baseline_files[path] = content
        _artifact(world, artifact_id, path, content)
    freeze_baseline(world)
    branch = _branch(world)
    _commit(world, branch)
    pr = _pr(world, branch)
    head = freeze_pr_head(world, pr)

    snapshot_type = type(head)
    original_getter = snapshot_type.files.fget
    accesses = 0

    def counted_files(snapshot):
        nonlocal accesses
        accesses += 1
        return original_getter(snapshot)

    monkeypatch.setattr(snapshot_type, "files", property(counted_files))
    candidate = conservative_merge_candidate(world, pr)

    assert candidate.files["src/widget.py"] == HEAD
    # This count is intentionally independent of the number of files.  A
    # property access inside the path loop regresses to at least 2 * 66 here.
    assert accesses <= 8


@pytest.mark.parametrize("head_text, peer_text, head_delete, peer_delete", [
    (HEAD, "PEER_DIFFERENT_EDIT", False, False),
    ("", "PEER_DIFFERENT_EDIT", False, False),
    ("", "PEER_DIFFERENT_EDIT", True, False),
    (HEAD, "", False, True),
])
def test_merge_candidate_refuses_different_overlap_without_modifying_main(
        head_text, peer_text, head_delete, peer_delete):
    world = _world()
    own, peer = _branch(world), _branch(world, PEER)
    _commit(world, own, head_text, deletes=head_delete)
    own_pr = _pr(world, own)
    own_head = freeze_pr_head(world, own_pr)
    _commit(world, peer, peer_text, deletes=peer_delete)
    peer_pr = _pr(world, peer)
    freeze_pr_head(world, peer_pr)
    _merge_ledger(world, peer_pr)
    before = deepcopy(world.__dict__)
    with pytest.raises(SourceViewError, match="source_merge_conflict") as error:
        conservative_merge_candidate(world, own_pr)
    assert error.value.conflict_paths == ("src/widget.py",)
    assert world._cooperbench_source_views == before["_cooperbench_source_views"]
    assert world.repo_system.repo.main_commit_ids == before["repo_system"].repo.main_commit_ids
    assert pr_head_snapshot(world, own_pr) == own_head


@pytest.mark.parametrize("text, deletes", [(HEAD, False), ("", False)])
def test_identical_overlap_is_safe_including_empty_and_delete(text, deletes):
    world = _world()
    own, peer = _branch(world), _branch(world, PEER)
    _commit(world, own, text, deletes=deletes)
    own_pr = _pr(world, own)
    head = freeze_pr_head(world, own_pr)
    _commit(world, peer, text, deletes=deletes)
    peer_pr = _pr(world, peer)
    freeze_pr_head(world, peer_pr)
    _merge_ledger(world, peer_pr)
    candidate = conservative_merge_candidate(world, own_pr)
    assert candidate.files == head.files


def test_merge_candidate_preserves_own_new_file_and_peer_deletion_and_empty(tmp_path):
    world = _world()
    own, peer = _branch(world), _branch(world, PEER)
    _artifact(world, "new", "src/new.py", DESK, new=True)
    _commit(world, own, "", aid="new", creates=True)
    own_pr = _pr(world, own)
    freeze_pr_head(world, own_pr)
    _commit(world, peer, "", deletes=True)
    _commit(world, peer, "", aid="art_readme")
    peer_pr = _pr(world, peer)
    freeze_pr_head(world, peer_pr)
    _merge_ledger(world, peer_pr)
    candidate = conservative_merge_candidate(world, own_pr)
    assert candidate.files == {"src/new.py": "", "README.md": ""}
    export_product_repo(world, str(tmp_path), **snapshot_projection(world, candidate))
    assert not (tmp_path / "src/widget.py").exists()
    assert (tmp_path / "src/new.py").read_bytes() == b""


def test_merge_candidate_identity_changes_on_main_advance_even_same_tree():
    world, _, _, _, pr, _ = _published()
    before = conservative_merge_candidate(world, pr)
    peer = _branch(world, PEER)
    _commit(world, peer, "", aid="art_readme")
    peer_pr = _pr(world, peer)
    freeze_pr_head(world, peer_pr)
    _merge_ledger(world, peer_pr)
    after = conservative_merge_candidate(world, pr)
    assert before.tree_digest == after.tree_digest
    assert before.revision_identity != after.revision_identity


def test_merge_candidate_requires_a_published_exact_current_head():
    world, own, _, _, pr, _ = _published()
    _commit(world, own, "AFTER_REVIEW")
    with pytest.raises(SourceViewError, match="pr_not_synced_to_branch"):
        conservative_merge_candidate(world, pr)


def test_mainline_snapshot_exports_only_committed_main_not_unmerged_pr_or_desk(tmp_path):
    world, _, _, _, pr, head = _published()
    world.product_artifacts["art_widget"].content = DESK
    world.product_artifacts["art_widget"].mainline_content = DESK
    _artifact(world, "pending_new", "src/pending.py", DESK, new=True)
    initial = mainline_snapshot(world)
    assert initial.source_kind == "committed_mainline"
    assert initial.files["src/widget.py"] == BASE
    assert initial.base_main_commit_ids == ()
    _merge_ledger(world, pr)
    committed = mainline_snapshot(world)
    assert committed.files == head.files
    assert committed.commit_ids == tuple(pr.commit_ids)
    assert committed.base_main_commit_ids == tuple(pr.commit_ids)
    export_product_repo(world, str(tmp_path), **snapshot_projection(world, committed))
    assert (tmp_path / "src/widget.py").read_text() == HEAD
    assert not (tmp_path / "src/pending.py").exists()
    assert initial.files["src/widget.py"] == BASE


def test_mainline_snapshot_retains_new_empty_file_and_explicit_delete(tmp_path):
    world = _world()
    branch = _branch(world)
    _artifact(world, "new", "src/new.py", DESK, new=True)
    _commit(world, branch, "", aid="new", creates=True)
    _commit(world, branch, "", deletes=True)
    pr = _pr(world, branch)
    freeze_pr_head(world, pr)
    _merge_ledger(world, pr)
    committed = mainline_snapshot(world)
    assert committed.files == {"README.md": "", "src/new.py": ""}
    export_product_repo(world, str(tmp_path), **snapshot_projection(world, committed))
    assert (tmp_path / "src/new.py").read_bytes() == b""
    assert not (tmp_path / "src/widget.py").exists()


def test_mainline_snapshot_never_uses_global_content_when_patch_is_missing():
    world, _, _, patch, pr, _ = _published()
    _merge_ledger(world, pr)
    del world.patches[patch.patch_id]
    with pytest.raises(SourceViewError, match="patch_ledger_missing"):
        mainline_snapshot(world)


def test_explicit_delete_then_recreate_same_artifact_is_replayable():
    world = _world()
    branch = _branch(world)
    _commit(world, branch, "", deletes=True)
    _commit(world, branch, "RECREATED\n", creates=True)
    snapshot = freeze_pr_head(world, _pr(world, branch))
    assert snapshot.files["src/widget.py"] == "RECREATED\n"


def test_net_noop_pr_must_not_erase_newer_peer_main_when_native_commits_append():
    world = _world()
    own, peer = _branch(world), _branch(world, PEER)
    _commit(world, own, "TEMPORARY_OWNER_EDIT\n")
    _commit(world, own, BASE)  # No net change against the frozen branch base.
    own_pr = _pr(world, own)
    freeze_pr_head(world, own_pr)
    _commit(world, peer, "PEER_MERGED_MUST_SURVIVE\n")
    peer_pr = _pr(world, peer)
    freeze_pr_head(world, peer_pr)
    _merge_ledger(world, peer_pr)
    original_main = mainline_snapshot(world)
    registry = deepcopy(world._cooperbench_source_views)
    with pytest.raises(SourceViewError, match="source_merge_conflict") as error:
        conservative_merge_candidate(world, own_pr)
    assert error.value.conflict_paths == ("src/widget.py",)
    assert error.value.native_append_error == "native_append_tree_differs_from_merge_candidate"
    assert world._cooperbench_source_views == registry
    assert mainline_snapshot(world) == original_main


@pytest.mark.parametrize("peer_deletes", [False, True])
def test_native_append_rejects_same_net_delete_or_noop_recreation_conflicts(peer_deletes):
    world = _world()
    own, peer = _branch(world), _branch(world, PEER)
    _commit(world, own, "", deletes=True)
    if not peer_deletes:
        _commit(world, own, BASE, creates=True)  # Delete/recreate is a net noop.
    own_pr = _pr(world, own)
    freeze_pr_head(world, own_pr)
    _commit(world, peer, "" if peer_deletes else "PEER_NEW_CONTENT", deletes=peer_deletes)
    peer_pr = _pr(world, peer)
    freeze_pr_head(world, peer_pr)
    _merge_ledger(world, peer_pr)
    with pytest.raises(SourceViewError, match="source_merge_conflict") as error:
        conservative_merge_candidate(world, own_pr)
    assert error.value.conflict_paths == ("src/widget.py",)


def test_successful_conservative_candidate_equals_native_appended_main_snapshot():
    world, _, _, _, own_pr, _ = _published()
    peer = _branch(world, PEER)
    _commit(world, peer, "PEER_README\n", aid="art_readme")
    peer_pr = _pr(world, peer)
    freeze_pr_head(world, peer_pr)
    _merge_ledger(world, peer_pr)
    candidate = conservative_merge_candidate(world, own_pr)
    _merge_ledger(world, own_pr)
    actual_main = mainline_snapshot(world)
    assert candidate.tree_digest == actual_main.tree_digest
    assert candidate.files == actual_main.files


def _private_world():
    world = _world()
    initialize_actor_desks(world)
    return world


def _prepare_private(world, actor=OWNER, text=HEAD, aid="art_widget", *,
                     creates=False, deletes=False, pid=None, snapshot=None):
    snapshot = snapshot or actor_desk_snapshot(world, actor)
    patch = CodePatch(pid or f"private_patch_{len(world.patches)}", aid, actor, 3,
                      new_content=text, creates_file=creates)
    if deletes:
        patch.deletes_file = True
    ticket = prepare_actor_patch(world, actor, patch, expected_snapshot_id=snapshot.snapshot_id)
    return patch, ticket


def _native_accept_private(world, patch, ticket):
    state = world._cooperbench_actor_desks["actors"][patch.actor_id]
    bid = state["branch_id"]
    if bid is None:
        bid = world.repo_system.create_branch(patch.actor_id, tick=3).branch_id
        bind_actor_branch(world, patch.actor_id, bid)
    patch.validation_status = "accepted"
    world.patches[patch.patch_id] = patch
    world._pending_by_branch.setdefault(bid, []).append((patch.patch_id, patch.target_object_id))
    world.repo_system.repo.branches[bid].uncommitted_changes += 1
    return commit_actor_patch(world, ticket, branch_id=bid)


def _accept_private(world, actor=OWNER, text=HEAD, aid="art_widget", **kwargs):
    patch, ticket = _prepare_private(world, actor, text, aid, **kwargs)
    return _native_accept_private(world, patch, ticket)


def _commit_private(world, actor=OWNER):
    bid = world._cooperbench_actor_desks["actors"][actor]["branch_id"]
    entries = world._pending_by_branch[bid]
    commit = world.repo_system.commit_changes(
        agent_id=actor, branch_id=bid, message="private public canary",
        changed_files=[world.product_artifacts[aid].linked_file_path for _, aid in entries],
        patch_ids=[pid for pid, _ in entries], artifact_ids=[aid for _, aid in entries], tick=4)
    world._pending_by_branch[bid] = []
    return commit


def _publish_private(world, actor=OWNER):
    _commit_private(world, actor)
    bid = world._cooperbench_actor_desks["actors"][actor]["branch_id"]
    pr = _pr(world, world.repo_system.repo.branches[bid])
    freeze_pr_head(world, pr)
    return pr


def test_private_desks_are_explicit_and_both_bootstrap_bases_are_pinned():
    world = _world()
    assert not actor_desks_enabled(world)
    initial = initialize_actor_desks(world)
    assert actor_desks_enabled(world)
    assert initial[OWNER].files == initial[PEER].files == freeze_baseline(world).files
    assert initial[OWNER].snapshot_id != initial[PEER].snapshot_id
    assert initial[OWNER].owner_id == OWNER
    assert initialize_actor_desks(world) == initial
    with pytest.raises(SourceViewError, match="actor_not_registered"):
        actor_desk_snapshot(world, "stranger")
    with pytest.raises(SourceViewError, match="actor_snapshot_identity_mismatch"):
        actor_artifact_catalog(world, PEER, snapshot=initial[OWNER])


def test_private_desks_refuse_shared_history_migration_without_changing_v50():
    world, _, _, _, pr, head = _published()
    with pytest.raises(SourceViewError, match="actor_desks_require_fresh_world"):
        initialize_actor_desks(world)
    assert not actor_desks_enabled(world)
    assert pr_head_snapshot(world, pr) == head


def test_private_prepare_is_pure_and_ticket_is_immutable():
    world = _private_world()
    view = actor_desk_snapshot(world, OWNER)
    before = pickle.dumps(world)
    _, ticket = _prepare_private(world, snapshot=view)
    assert pickle.dumps(world) == before
    with pytest.raises(FrozenInstanceError):
        ticket.new_content = DESK


def test_candidate_preview_freezes_proposed_bytes_without_publishing_private_or_native_state():
    world = _private_world()
    before = actor_desk_snapshot(world, OWNER)
    repo_before = deepcopy(world.repo_system.repo)
    patches_before = deepcopy(world.patches)
    native_before = deepcopy(world.product_artifacts["art_widget"].__dict__)
    patch, ticket = _prepare_private(world, text="PROPOSED_ONLY\n", pid="preview")
    candidate = actor_patch_candidate_snapshot(world, ticket)
    assert candidate.source_kind == "actor_patch_candidate"
    assert candidate.files["src/widget.py"] == "PROPOSED_ONLY\n"
    assert candidate.patch_ids == (*before.patch_ids, patch.patch_id)
    assert actor_patch_candidate_snapshot(world, ticket) == candidate
    assert actor_desk_snapshot(world, OWNER) == before
    assert world.repo_system.repo == repo_before
    assert world.patches == patches_before
    assert world.product_artifacts["art_widget"].__dict__ == native_before


def test_two_authors_same_path_pending_are_independent_and_global_text_unchanged(tmp_path):
    world = _private_world()
    own_before, peer_before = actor_desk_snapshot(world, OWNER), actor_desk_snapshot(world, PEER)
    own_patch, own_ticket = _prepare_private(world, OWNER, "OWNER_PRIVATE\n", pid="owner_ticket")
    peer_patch, peer_ticket = _prepare_private(world, PEER, "PEER_PRIVATE\n", pid="peer_ticket")
    _native_accept_private(world, own_patch, own_ticket)
    assert_actor_desk_current(world, PEER, peer_before)
    _native_accept_private(world, peer_patch, peer_ticket)
    assert actor_file_text(world, OWNER, "art_widget") == "OWNER_PRIVATE\n"
    assert actor_file_text(world, PEER, "art_widget") == "PEER_PRIVATE\n"
    assert world.product_artifacts["art_widget"].content == BASE
    assert world.product_artifacts["art_widget"].mainline_content == BASE
    for actor, expected in [(OWNER, "OWNER_PRIVATE\n"), (PEER, "PEER_PRIVATE\n")]:
        export_product_repo(world, str(tmp_path / actor), **actor_desk_projection(world, actor))
        assert (tmp_path / actor / "src/widget.py").read_text() == expected
    assert actor_file_text(world, OWNER, "art_widget", snapshot=own_before) == BASE
    with pytest.raises(SourceViewError, match="actor_desk_changed"):
        assert_actor_desk_current(world, OWNER, own_before)


def test_peer_new_file_absent_from_actor_catalog_but_independent_create_allowed():
    world = _private_world()
    _artifact(world, "shared_path_metadata", "src/new.py", "DO_NOT_READ_PEER_METADATA_TEXT", new=True)
    own = _accept_private(world, OWNER, "OWNER_NEW", aid="shared_path_metadata", creates=True)
    assert actor_file_text(world, PEER, "shared_path_metadata") is None
    assert "shared_path_metadata" not in actor_artifact_catalog(world, PEER)
    peer = _accept_private(world, PEER, "PEER_NEW", aid="shared_path_metadata", creates=True)
    assert own.files["src/new.py"] == "OWNER_NEW"
    assert peer.files["src/new.py"] == "PEER_NEW"
    own_pr = _publish_private(world, OWNER)
    peer_pr = _publish_private(world, PEER)
    _merge_ledger(world, own_pr)
    with pytest.raises(SourceViewError, match="source_merge_conflict"):
        conservative_merge_candidate(world, peer_pr)


def test_private_catalog_accepts_fixed_snapshot_without_resampling_new_paths():
    world = _private_world()
    before = actor_desk_snapshot(world, OWNER)
    _artifact(world, "new", "created.py", "GLOBAL", new=True)
    after = _accept_private(world, OWNER, "", aid="new", creates=True)
    assert "new" not in actor_artifact_catalog(world, OWNER, snapshot=before)
    assert actor_artifact_catalog(world, OWNER, snapshot=after)["new"] == "created.py"
    assert actor_file_text(world, OWNER, "new", snapshot=before) is None
    assert actor_file_text(world, OWNER, "new", snapshot=after) == ""


def test_private_empty_update_and_delete_are_distinct_from_peer_view(tmp_path):
    world = _private_world()
    empty = _accept_private(world, text="")
    assert empty.files["src/widget.py"] == ""
    deleted = _accept_private(world, text="", deletes=True)
    assert "src/widget.py" not in deleted.files
    assert actor_file_text(world, OWNER, "art_widget") is None
    assert actor_file_text(world, PEER, "art_widget") == BASE
    assert "art_widget" not in actor_artifact_catalog(world, OWNER)
    export_product_repo(world, str(tmp_path), **actor_desk_projection(world, OWNER))
    assert not (tmp_path / "src/widget.py").exists()


def test_own_commit_moves_pending_without_changing_private_bytes_or_peer_view():
    world = _private_world()
    pending = _accept_private(world)
    peer_before = actor_desk_snapshot(world, PEER)
    commit = _commit_private(world)
    committed = actor_desk_snapshot(world, OWNER)
    assert pending.files == committed.files
    assert committed.commit_ids == (commit.commit_id,)
    assert pending.revision_identity != committed.revision_identity
    assert_actor_desk_current(world, PEER, peer_before)
    _accept_private(world, text="AFTER_OWN_COMMIT")
    assert actor_file_text(world, OWNER, "art_widget") == "AFTER_OWN_COMMIT"
    assert actor_file_text(world, PEER, "art_widget") == BASE


def test_private_pr_still_uses_v50_committed_head_and_excludes_later_pending():
    world = _private_world()
    _accept_private(world)
    pr = _publish_private(world)
    head = pr_head_snapshot(world, pr)
    _accept_private(world, text="LATER_PRIVATE_PENDING")
    assert pr_head_snapshot(world, pr) == head
    assert head.files["src/widget.py"] == HEAD
    assert actor_file_text(world, OWNER, "art_widget") == "LATER_PRIVATE_PENDING"


def test_new_branch_after_peer_merge_uses_pinned_actor_base_without_implicit_sync():
    world = _private_world()
    peer_before = actor_desk_snapshot(world, PEER)
    _accept_private(world)
    pr = _publish_private(world)
    _merge_ledger(world, pr)
    assert_actor_desk_current(world, PEER, peer_before)
    _accept_private(world, PEER, "PEER_README", aid="art_readme")
    bid = world._cooperbench_actor_desks["actors"][PEER]["branch_id"]
    base = bind_actor_branch(world, PEER, bid)
    assert base.base_main_commit_ids == ()
    assert base.files["src/widget.py"] == BASE
    assert actor_file_text(world, PEER, "art_widget") == BASE


def test_clean_actor_requires_explicit_sync_to_import_peer_committed_main():
    world = _private_world()
    before = actor_desk_snapshot(world, PEER)
    _accept_private(world)
    pr = _publish_private(world)
    _merge_ledger(world, pr)
    assert actor_desk_snapshot(world, PEER) == before
    after = sync_actor_desk(world, PEER, expected_snapshot_id=before.snapshot_id)
    assert after.files["src/widget.py"] == HEAD
    assert after.base_main_commit_ids == tuple(pr.commit_ids)
    assert world._cooperbench_actor_desks["sync_receipts"][-1]["explicit"] is True
    _accept_private(world, PEER, "PEER_EDIT_AFTER_SYNC")
    peer_pr = _publish_private(world, PEER)
    assert conservative_merge_candidate(world, peer_pr).files["src/widget.py"] == "PEER_EDIT_AFTER_SYNC"


@pytest.mark.parametrize("peer_aid, error_code", [
    ("art_widget", "actor_sync_conflict"),
    ("art_readme", "actor_sync_requires_clean_branch"),
])
def test_dirty_sync_refuses_both_conflicts_and_nonoverlap_without_any_mutation(peer_aid, error_code):
    world = _private_world()
    _accept_private(world, OWNER, "OWNER_UNDELIVERED")
    _accept_private(world, PEER, "PEER_COMMITTED", aid=peer_aid)
    pr = _publish_private(world, PEER)
    _merge_ledger(world, pr)
    own = actor_desk_snapshot(world, OWNER)
    before = pickle.dumps(world)
    with pytest.raises(SourceViewError, match=error_code) as error:
        sync_actor_desk(world, OWNER, expected_snapshot_id=own.snapshot_id)
    assert pickle.dumps(world) == before
    assert error.value.conflict_paths == (("src/widget.py",) if peer_aid == "art_widget" else ())


def test_unmerged_own_commits_also_make_sync_dirty():
    world = _private_world()
    _accept_private(world)
    _publish_private(world)
    own = actor_desk_snapshot(world, OWNER)
    with pytest.raises(SourceViewError, match="actor_sync_requires_clean_branch"):
        sync_actor_desk(world, OWNER, expected_snapshot_id=own.snapshot_id)


def test_conflicted_pr_can_restart_on_current_main_without_erasing_history():
    world = _private_world()
    _accept_private(world, OWNER, "OWNER_STALE_EDIT\n")
    own_pr = _publish_private(world, OWNER)
    old_branch_id = own_pr.source_branch
    old_commit_ids = tuple(own_pr.commit_ids)
    old_patch_ids = tuple(own_pr.patch_ids)
    _accept_private(world, OWNER, "OWNER_LATE_PENDING_EDIT\n")
    pending_patch_id = world._pending_by_branch[old_branch_id][-1][0]

    _accept_private(world, PEER, "PEER_MERGED_EDIT\n")
    peer_pr = _publish_private(world, PEER)
    _merge_ledger(world, peer_pr)
    before = actor_desk_snapshot(world, OWNER)
    current_main = mainline_snapshot(world)

    restarted = restart_actor_desk_after_conflict(
        world,
        OWNER,
        pr_id=own_pr.pr_id,
        expected_snapshot_id=before.snapshot_id,
        mainline_snapshot_id=current_main.snapshot_id,
        tick=11,
    )

    assert own_pr.status == PRStatus.STALE
    assert world.repo_system.repo.branches[own_pr.source_branch].status == (
        BranchStatus.ABANDONED
    )
    assert tuple(own_pr.commit_ids) == old_commit_ids
    assert tuple(own_pr.patch_ids) == old_patch_ids
    assert pending_patch_id in world.patches
    assert world.patches[pending_patch_id].validation_status == "rejected"
    assert world.patches[pending_patch_id].rejection_reason == (
        "stale_base_coordination_conflict"
    )
    assert world._pending_by_branch[old_branch_id] == []
    assert world.repo_system.repo.branches[old_branch_id].uncommitted_changes == 0
    assert restarted.files["src/widget.py"] == "PEER_MERGED_EDIT\n"
    assert restarted.branch_id is None and restarted.patch_ids == ()
    assert world._cooperbench_actor_desks["sync_receipts"][-1]["reason"] == (
        "coordination_conflict_restart"
    )
    assert world._cooperbench_actor_desks["sync_receipts"][-1][
        "abandoned_pending_patch_ids"
    ] == [pending_patch_id]

    _accept_private(
        world,
        OWNER,
        "PEER_MERGED_EDIT\nOWNER_REIMPLEMENTED\n",
    )
    replacement_pr = _publish_private(world, OWNER)
    candidate = conservative_merge_candidate(world, replacement_pr)
    assert "PEER_MERGED_EDIT" in candidate.files["src/widget.py"]
    assert "OWNER_REIMPLEMENTED" in candidate.files["src/widget.py"]


def test_after_own_merge_sync_can_advance_then_bind_a_fresh_feature_branch():
    world = _private_world()
    _accept_private(world)
    own_pr = _publish_private(world)
    _merge_ledger(world, own_pr)
    peer = actor_desk_snapshot(world, PEER)
    sync_actor_desk(world, PEER, expected_snapshot_id=peer.snapshot_id)
    _accept_private(world, PEER, "LATEST_PUBLIC_MAIN")
    peer_pr = _publish_private(world, PEER)
    _merge_ledger(world, peer_pr)
    own = actor_desk_snapshot(world, OWNER)
    assert own.files["src/widget.py"] == HEAD
    advanced = sync_actor_desk(world, OWNER, expected_snapshot_id=own.snapshot_id)
    assert advanced.files["src/widget.py"] == "LATEST_PUBLIC_MAIN"
    assert advanced.branch_id is None
    _accept_private(world, OWNER, "OWNER_NEXT_BRANCH")
    assert actor_file_text(world, OWNER, "art_widget") == "OWNER_NEXT_BRANCH"


def test_competing_same_actor_ticket_is_rejected_without_private_publication():
    world = _private_world()
    first, ticket1 = _prepare_private(world, text="FIRST", pid="first")
    second, ticket2 = _prepare_private(world, text="SECOND", pid="second")
    _native_accept_private(world, first, ticket1)
    private_state = deepcopy(world._cooperbench_actor_desks)
    with pytest.raises(SourceViewError, match="actor_patch_transaction_changed"):
        _native_accept_private(world, second, ticket2)
    assert world._cooperbench_actor_desks == private_state
    # Native acceptance belongs to the caller's transaction; rollback its failed
    # pending insertion before reading the private desk again.
    bid = private_state["actors"][OWNER]["branch_id"]
    world._pending_by_branch[bid].pop()
    del world.patches["second"]
    assert actor_file_text(world, OWNER, "art_widget") == "FIRST"


def test_changed_provider_payload_cannot_be_committed_from_a_prepared_ticket():
    world = _private_world()
    patch, ticket = _prepare_private(world)
    branch = world.repo_system.create_branch(OWNER)
    bind_actor_branch(world, OWNER, branch.branch_id)
    patch.new_content = "MUTATED_AFTER_PREPARE"
    patch.validation_status = "accepted"
    world.patches[patch.patch_id] = patch
    world._pending_by_branch[branch.branch_id] = [(patch.patch_id, patch.target_object_id)]
    before = deepcopy(world._cooperbench_actor_desks)
    with pytest.raises(SourceViewError, match="actor_patch_payload_changed"):
        commit_actor_patch(world, ticket, branch_id=branch.branch_id)
    assert world._cooperbench_actor_desks == before


def test_stale_prompt_snapshot_cannot_prepare_after_own_view_or_sync_changes():
    world = _private_world()
    before = actor_desk_snapshot(world, OWNER)
    _accept_private(world)
    with pytest.raises(SourceViewError, match="actor_desk_changed"):
        _prepare_private(world, text="STALE_REPLY", snapshot=before)


def test_unregistered_native_pending_patch_cannot_enter_private_desk():
    world = _private_world()
    branch = world.repo_system.create_branch(OWNER)
    bind_actor_branch(world, OWNER, branch.branch_id)
    patch = CodePatch("bare", "art_widget", OWNER, 3, new_content=DESK, validation_status="accepted")
    world.patches[patch.patch_id] = patch
    world._pending_by_branch[branch.branch_id] = [(patch.patch_id, patch.target_object_id)]
    with pytest.raises(SourceViewError, match="actor_patch_ledger_incomplete"):
        actor_desk_snapshot(world, OWNER)
    assert actor_file_text(world, PEER, "art_widget") == BASE


def test_private_accepted_patch_bytes_cannot_be_mutated_after_publication():
    world = _private_world()
    desk = _accept_private(world)
    world.patches[desk.patch_ids[-1]].new_content = DESK
    with pytest.raises(SourceViewError, match="actor_patch_ledger_changed"):
        actor_desk_snapshot(world, OWNER)


def test_private_branch_binding_refuses_wrong_owner_and_existing_v50_base():
    world = _private_world()
    wrong = world.repo_system.create_branch(PEER)
    with pytest.raises(SourceViewError, match="actor_branch_owner_mismatch"):
        bind_actor_branch(world, OWNER, wrong.branch_id)
    branch = world.repo_system.create_branch(OWNER)
    freeze_branch_base(world, branch.branch_id)  # Caller used the wrong seam.
    with pytest.raises(SourceViewError, match="actor_branch_base_mismatch"):
        bind_actor_branch(world, OWNER, branch.branch_id)


def test_private_desk_checkpoint_roundtrip_and_missing_vs_empty(tmp_path):
    world = _private_world()
    own = _accept_private(world, text="")
    for restored in (deepcopy(world), pickle.loads(pickle.dumps(world))):
        assert actor_desk_snapshot(restored, OWNER) == own
        assert actor_file_text(restored, OWNER, "art_widget") == ""
        assert actor_file_text(restored, OWNER, "missing") is None
        export_product_repo(restored, str(tmp_path), **actor_desk_projection(restored, OWNER))
        assert (tmp_path / "src/widget.py").read_bytes() == b""
