"""Frozen public-source views reconstructed from RepoLite's explicit ledgers.

This module never reads artifact ``content`` or ``mainline_content``. Bootstrap
must freeze the public starter tree, branch creation must freeze its main base
before its first commit, and PR publication/CI must explicitly freeze the synced
head. Old checkpoints without those records cannot reconstruct historical bases.

Full-text patches faithfully reproduce the committed ledger, not independent
actor desks: a committed patch may itself have copied peer code before commit.
Actor-private editing is a separate integration stage. Explicit ``deletes_file``
patch tombstones are supported; current CodePatch has no native delete field, so
empty text or a missing artifact must never be interpreted as a deletion.
"""
from __future__ import annotations

import hashlib
import json
import weakref
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from environments.org_env.product.repo_paths import normalize_repo_relative_path

_STORE = "_cooperbench_source_views"
_SCHEMA = "cooperbench_source_views_v1"
_DESK_STORE = "_cooperbench_actor_desks"
_DESK_SCHEMA = "cooperbench_actor_desks_v1"

# A frozen source snapshot can contain the entire public repository.  Integrity
# validation therefore hashes tens of megabytes for a non-trivial pack.  Some
# lifecycle predicates read the same snapshot every tick; re-hashing the same
# immutable object there turned an O(1) coverage lookup into repeated O(tree)
# work.  Keep this cache process-local rather than on ``world`` so it never
# becomes checkpoint state or an authority-bearing receipt.  The entry is
# reusable only while the exact world, snapshot object, immutable entry tuples,
# and every identity field are unchanged.  A restored world or replaced/forged
# snapshot must pay for validation once again.
_INTEGRITY_CACHE: dict[
    int,
    tuple[Callable[[], Any], dict[str, tuple[Any, ...]]],
] = {}
_INTEGRITY_CACHE_MAX_WORLDS = 64


class SourceViewError(ValueError):
    """An explicit source-provenance failure; callers must not fall back to desk."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(f"cooperbench_source_view:{code}")


def _hash(value: Any) -> str:
    try:
        raw = json.dumps(value, sort_keys=True, ensure_ascii=False,
                         separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise SourceViewError("invalid_serializable_source") from exc
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class SourceSnapshot:
    """Pickle/deepcopy-safe immutable tree and exact ledger-revision identity."""

    source_kind: str
    snapshot_id: str
    tree_digest: str
    revision_identity: str
    branch_id: str | None
    pr_id: str | None
    owner_id: str | None
    base_snapshot_id: str | None
    base_main_commit_ids: tuple[str, ...]
    commit_ids: tuple[str, ...]
    patch_ids: tuple[str, ...]
    _file_entries: tuple[tuple[str, str], ...]
    _artifact_entries: tuple[tuple[str, str], ...]
    _ledger_digest: str
    runtime_assets_digest: str | None = None

    @property
    def files(self) -> Mapping[str, str]:
        return MappingProxyType(dict(self._file_entries))

    def receipt(self) -> dict[str, Any]:
        """Return metadata only, not source text; callers may safely annotate it."""
        receipt = {
            "schema_version": _SCHEMA, "source_kind": self.source_kind,
            "snapshot_id": self.snapshot_id, "tree_digest": self.tree_digest,
            "revision_identity": self.revision_identity,
            "branch_id": self.branch_id, "pr_id": self.pr_id,
            "owner_id": self.owner_id, "base_snapshot_id": self.base_snapshot_id,
            "base_main_commit_ids": list(self.base_main_commit_ids),
            "commit_ids": list(self.commit_ids), "patch_ids": list(self.patch_ids),
        }
        # Keep legacy worlds/receipts byte-for-byte compatible when bootstrap
        # did not opt into runtime assets. Only identity, never resource bytes,
        # belongs in model-facing source snapshots.
        if self.runtime_assets_digest is not None:
            receipt["runtime_assets_digest"] = self.runtime_assets_digest
        return receipt


def source_views_enabled(world: Any) -> bool:
    # Presence, not validity: a malformed opted-in store must fail closed instead
    # of selecting the legacy global-working-tree path in a consumer.
    return _STORE in getattr(world, "__dict__", {})


def _store(world: Any) -> dict[str, Any]:
    state = getattr(world, "__dict__", {}).get(_STORE)
    if not isinstance(state, dict) or state.get("schema_version") != _SCHEMA:
        raise SourceViewError("store_uninitialized_or_invalid")
    for name in ("snapshots", "branch_bases", "pr_heads"):
        if not isinstance(state.get(name), dict):
            raise SourceViewError("store_invalid")
    return state


def _repo(world: Any) -> Any:
    repo = getattr(getattr(world, "repo_system", None), "repo", None)
    if repo is None or any(not isinstance(getattr(repo, field, None), dict)
                           for field in ("branches", "commits", "pull_requests")):
        raise SourceViewError("repo_ledger_missing")
    return repo


def _ids(value: Any, field: str, *, unique: bool = True) -> tuple[str, ...]:
    if (not isinstance(value, (list, tuple))
            or any(not isinstance(item, str) or not item for item in value)
            or (unique and len(set(value)) != len(value))):
        raise SourceViewError(f"{field}_invalid")
    return tuple(value)


def _path(raw: Any) -> str:
    if not isinstance(raw, str) or not raw:
        raise SourceViewError("artifact_path_missing")
    try:
        return normalize_repo_relative_path(raw)
    except ValueError as exc:
        raise SourceViewError("artifact_path_invalid") from exc


def _catalog(world: Any) -> dict[str, str]:
    """Read repository metadata only; fail on aliases like the real exporter."""
    artifacts = getattr(world, "product_artifacts", None)
    if not isinstance(artifacts, dict):
        raise SourceViewError("artifact_registry_missing")
    result: dict[str, str] = {}
    paths: set[str] = set()
    for key, artifact in artifacts.items():
        if getattr(artifact, "artifact_type", None) == "issue":
            continue
        raw = getattr(artifact, "linked_file_path", None)
        if raw is None or raw == "":
            continue
        aid = getattr(artifact, "artifact_id", None)
        path = _path(raw)
        if not isinstance(aid, str) or not aid or key != aid or aid in result:
            raise SourceViewError("artifact_identity_invalid")
        if path.casefold() in paths:
            raise SourceViewError("artifact_path_collision")
        result[aid] = path
        paths.add(path.casefold())
    return result


def _runtime_assets_digest(world: Any) -> str | None:
    from .runtime_assets import RuntimeAssetError, runtime_assets_digest
    try:
        return runtime_assets_digest(world)
    except RuntimeAssetError as error:
        raise SourceViewError("runtime_assets_invalid:" + error.code) from error


def _assert_runtime_assets_current(world: Any, snapshot: SourceSnapshot) -> None:
    if _runtime_assets_digest(world) != snapshot.runtime_assets_digest:
        # Absence is pinned too: adding assets after bootstrap cannot silently
        # reinterpret an already approved asset-free tree.
        raise SourceViewError("runtime_assets_changed")


def _tree_digest(entries: tuple, runtime_assets_digest: str | None) -> str:
    return (_hash(entries) if runtime_assets_digest is None else _hash({
        "source_files": entries, "runtime_assets_digest": runtime_assets_digest,
    }))


def _integrity_stamp(snapshot: SourceSnapshot) -> tuple[Any, ...]:
    """Cheap identity of one already-verified frozen snapshot object."""
    return (
        id(snapshot), id(snapshot._file_entries), id(snapshot._artifact_entries),
        snapshot.source_kind, snapshot.snapshot_id, snapshot.tree_digest,
        snapshot.revision_identity, snapshot.branch_id, snapshot.pr_id,
        snapshot.owner_id, snapshot.base_snapshot_id,
        snapshot.base_main_commit_ids, snapshot.commit_ids, snapshot.patch_ids,
        snapshot._ledger_digest, getattr(snapshot, "runtime_assets_digest", None),
    )


def _integrity_cache_for(world: Any) -> dict[str, tuple[Any, ...]] | None:
    """Return an identity-safe, bounded, process-local verification cache."""
    world_id = id(world)
    existing = _INTEGRITY_CACHE.get(world_id)
    if existing is not None and existing[0]() is world:
        # Dict insertion order supplies a tiny dependency-free LRU.
        _INTEGRITY_CACHE.pop(world_id)
        _INTEGRITY_CACHE[world_id] = existing
        return existing[1]
    try:
        def discard(reference: weakref.ReferenceType[Any], *, key: int = world_id) -> None:
            current = _INTEGRITY_CACHE.get(key)
            if current is not None and current[0] is reference:
                _INTEGRITY_CACHE.pop(key, None)

        reference: Callable[[], Any] = weakref.ref(world, discard)
    except TypeError:
        # SimpleNamespace and some slot-only worlds cannot be weak-referenced.
        # A bounded strong closure still avoids checkpoint mutation and id reuse.
        reference = lambda bound_world=world: bound_world
    while len(_INTEGRITY_CACHE) >= _INTEGRITY_CACHE_MAX_WORLDS:
        _INTEGRITY_CACHE.pop(next(iter(_INTEGRITY_CACHE)))
    cache: dict[str, tuple[Any, ...]] = {}
    _INTEGRITY_CACHE[world_id] = (reference, cache)
    return cache


def _identity(snapshot: SourceSnapshot) -> str:
    identity = {
        "schema": _SCHEMA, "kind": snapshot.source_kind,
        "tree": snapshot.tree_digest, "branch": snapshot.branch_id,
        "pr": snapshot.pr_id, "owner": snapshot.owner_id,
        "base": snapshot.base_snapshot_id,
        "main": snapshot.base_main_commit_ids,
        "commits": snapshot.commit_ids, "patches": snapshot.patch_ids,
        "artifacts": snapshot._artifact_entries, "ledger": snapshot._ledger_digest,
    }
    if snapshot.runtime_assets_digest is not None:
        identity["runtime_assets_digest"] = snapshot.runtime_assets_digest
    return _hash(identity)


def _make(kind: str, files: Mapping[str, str], catalog: Mapping[str, str], *,
          branch_id: str | None = None, pr_id: str | None = None,
          owner_id: str | None = None, base: SourceSnapshot | None = None,
          main_ids: tuple[str, ...] = (), commit_ids: tuple[str, ...] = (),
          patch_ids: tuple[str, ...] = (), ledger: Any = (),
          runtime_assets_digest: str | None = None) -> SourceSnapshot:
    if base is not None:
        if runtime_assets_digest is not None and runtime_assets_digest != base.runtime_assets_digest:
            raise SourceViewError("runtime_assets_base_mismatch")
        runtime_assets_digest = base.runtime_assets_digest
    entries = tuple(sorted(files.items()))
    fields = dict(source_kind=kind, snapshot_id="", tree_digest=_tree_digest(entries, runtime_assets_digest),
                  revision_identity="", branch_id=branch_id, pr_id=pr_id,
                  owner_id=owner_id, base_snapshot_id=base.snapshot_id if base else None,
                  base_main_commit_ids=main_ids, commit_ids=commit_ids,
                  patch_ids=patch_ids, _file_entries=entries,
                  _artifact_entries=tuple(sorted(catalog.items())),
                  _ledger_digest=_hash(ledger), runtime_assets_digest=runtime_assets_digest)
    identity = _identity(SourceSnapshot(**fields))
    fields.update(snapshot_id=identity, revision_identity=identity)
    return SourceSnapshot(**fields)


def _get(world: Any, snapshot_id: Any) -> SourceSnapshot:
    if not isinstance(snapshot_id, str) or not snapshot_id:
        raise SourceViewError("snapshot_missing")
    snapshot = _store(world)["snapshots"].get(snapshot_id)
    if not isinstance(snapshot, SourceSnapshot):
        raise SourceViewError("snapshot_missing")
    _assert_runtime_assets_current(world, snapshot)
    stamp = _integrity_stamp(snapshot)
    cache = _integrity_cache_for(world)
    if cache is None or cache.get(snapshot_id) != stamp:
        if (snapshot.snapshot_id != snapshot_id
                or snapshot.tree_digest != _tree_digest(
                    snapshot._file_entries,
                    getattr(snapshot, "runtime_assets_digest", None),
                )
                or snapshot.revision_identity != _identity(snapshot)
                or snapshot.snapshot_id != snapshot.revision_identity):
            raise SourceViewError("snapshot_integrity_invalid")
        if cache is not None:
            cache[snapshot_id] = stamp
    return snapshot


def _save(world: Any, snapshot: SourceSnapshot) -> SourceSnapshot:
    _assert_runtime_assets_current(world, snapshot)
    snapshots = _store(world)["snapshots"]
    existing = snapshots.get(snapshot.snapshot_id)
    if existing is not None and existing != snapshot:
        raise SourceViewError("snapshot_identity_collision")
    snapshots[snapshot.snapshot_id] = snapshot
    return snapshot


def freeze_baseline(world: Any) -> SourceSnapshot:
    """Opt in at bootstrap from the explicit public starter-files mapping only."""
    if not isinstance(getattr(world, "__dict__", {}).get("_cooperbench_sdl_state"), dict):
        raise SourceViewError("not_cooperbench")
    raw = getattr(world, "__dict__", {}).get("_cooperbench_public_baseline_files")
    if not isinstance(raw, Mapping):
        raise SourceViewError("public_baseline_missing")
    files: dict[str, str] = {}
    seen: set[str] = set()
    for raw_path, content in raw.items():
        path = _path(raw_path)
        if path.casefold() in seen or not isinstance(content, str):
            raise SourceViewError("public_baseline_invalid")
        files[path] = content
        seen.add(path.casefold())
    if source_views_enabled(world):
        snapshot = _get(world, _store(world).get("baseline_id"))
        if snapshot.source_kind != "baseline" or dict(snapshot.files) != files:
            raise SourceViewError("public_baseline_changed")
        return snapshot
    catalog = {aid: path for aid, path in _catalog(world).items() if path in files}
    if set(catalog.values()) != set(files):
        raise SourceViewError("baseline_artifact_missing")
    _repo(world)  # Refuse an incomplete world before creating the opt-in marker.
    snapshot = _make("baseline", files, catalog, runtime_assets_digest=_runtime_assets_digest(world))
    world.__dict__[_STORE] = {
        "schema_version": _SCHEMA, "baseline_id": snapshot.snapshot_id,
        "snapshots": {snapshot.snapshot_id: snapshot}, "branch_bases": {}, "pr_heads": {},
    }
    return snapshot


def _branch(world: Any, branch_id: str) -> Any:
    branch = _repo(world).branches.get(branch_id)
    if (branch is None or getattr(branch, "branch_id", None) != branch_id
            or not isinstance(getattr(branch, "owner_id", None), str)
            or not branch.owner_id):
        raise SourceViewError("branch_ledger_missing")
    if getattr(branch, "base_branch", None) != _repo(world).main_branch:
        raise SourceViewError("unsupported_branch_base")
    carried = _ids(getattr(branch, "commit_ids", None), "branch_commits")
    recorded = {cid for cid, commit in _repo(world).commits.items()
                if getattr(commit, "branch_id", None) == branch_id
                and getattr(commit, "status", None) != "reverted"}
    if recorded.difference(carried):
        raise SourceViewError("branch_commit_history_incomplete")
    return branch


def _main_commits(world: Any) -> tuple[str, ...]:
    repo = _repo(world)
    main_ids = _ids(getattr(repo, "main_commit_ids", None), "main_commits")
    merged_ids = {cid for cid, commit in repo.commits.items()
                  if getattr(commit, "status", None) == "merged"}
    if merged_ids != set(main_ids):
        raise SourceViewError("main_commit_history_incomplete")
    return main_ids


def _replay(world: Any, base: SourceSnapshot, commit_ids: tuple[str, ...], *,
            branch_id: str | None = None, owner_id: str | None = None,
            merged: bool = False) -> tuple[dict, dict, list, tuple[str, ...]]:
    """Only committed patches participate, never pending or unrelated records."""
    repo = _repo(world)
    patches = getattr(world, "patches", None)
    if not isinstance(patches, dict):
        raise SourceViewError("patch_ledger_missing")
    files, catalog = dict(base.files), dict(base._artifact_entries)
    current_catalog = _catalog(world)
    for aid, path in catalog.items():
        if aid in current_catalog and current_catalog[aid] != path:
            raise SourceViewError("artifact_path_changed")
    ledger, all_patches = [], []
    created_here: set[str] = set()
    pending = getattr(world, "__dict__", {}).get("_pending_by_branch", {})
    if not isinstance(pending, dict):
        raise SourceViewError("pending_ledger_invalid")
    pending_ids: set[str] = set()
    for entries in pending.values():
        if not isinstance(entries, (list, tuple)):
            raise SourceViewError("pending_ledger_invalid")
        for entry in entries:
            if (not isinstance(entry, (list, tuple)) or len(entry) != 2
                    or any(not isinstance(item, str) or not item for item in entry)):
                raise SourceViewError("pending_ledger_invalid")
            pending_ids.add(entry[0])
    branch_commits: dict[str, list[str]] = {}
    for cid in commit_ids:
        commit = repo.commits.get(cid)
        if commit is None or getattr(commit, "commit_id", None) != cid:
            raise SourceViewError("commit_ledger_missing")
        bid = getattr(commit, "branch_id", None)
        branch = _branch(world, bid)
        author = getattr(commit, "author_id", None)
        if ((branch_id is not None and bid != branch_id)
                or author != branch.owner_id
                or (owner_id is not None and author != owner_id)):
            raise SourceViewError("commit_owner_or_branch_mismatch")
        carried = _ids(getattr(branch, "commit_ids", None), "branch_commits")
        if cid not in carried:
            raise SourceViewError("commit_not_on_branch")
        branch_commits.setdefault(bid, []).append(cid)
        status = getattr(commit, "status", None)
        if status not in {"local", "pushed", "included_in_pr", "merged"} or (merged and status != "merged"):
            raise SourceViewError("commit_status_invalid")
        pids = _ids(getattr(commit, "patch_ids", None), "commit_patches")
        aids = _ids(getattr(commit, "artifact_ids", None), "commit_artifacts", unique=False)
        if not pids or len(pids) != len(aids):
            raise SourceViewError("commit_patch_artifact_alignment_missing")
        patch_rows = []
        for pid, aid in zip(pids, aids):
            if pid in pending_ids:
                raise SourceViewError("committed_patch_still_pending")
            if pid in all_patches:
                raise SourceViewError("duplicate_committed_patch")
            patch = patches.get(pid)
            if patch is None or getattr(patch, "patch_id", None) != pid:
                raise SourceViewError("patch_ledger_missing")
            if (getattr(patch, "target_object_id", None) != aid
                    or getattr(patch, "actor_id", None) != author):
                raise SourceViewError("patch_target_or_author_mismatch")
            if getattr(patch, "validation_status", None) != "accepted":
                raise SourceViewError("patch_not_accepted")
            text = getattr(patch, "new_content", None)
            creates = getattr(patch, "creates_file", False)
            deletes = getattr(patch, "deletes_file", False)
            if (not isinstance(text, str) or type(creates) is not bool
                    or type(deletes) is not bool or (creates and deletes)):
                raise SourceViewError("patch_content_or_operation_invalid")
            path = catalog.get(aid, current_catalog.get(aid))
            if path is None:
                raise SourceViewError("patch_artifact_path_missing")
            if any(other != aid and known.casefold() == path.casefold()
                   for other, known in catalog.items()):
                raise SourceViewError("artifact_path_collision")
            catalog[aid] = path
            if creates:
                # Native creates_file marks an unmerged file's origin, so a
                # follow-up edit before its first merge retains that flag.
                if path in files and aid not in created_here:
                    raise SourceViewError("create_path_already_exists")
                files[path] = text
                created_here.add(aid)
            elif deletes:
                if path not in files or text != "":
                    raise SourceViewError("delete_tombstone_invalid")
                del files[path]
            else:
                if path not in files:
                    raise SourceViewError("edit_path_missing_from_base")
                files[path] = text
            all_patches.append(pid)
            patch_rows.append((pid, aid, path, creates, deletes, _hash(text)))
        ledger.append((cid, bid, author, tuple(patch_rows)))
    # A partial or permuted branch history cannot silently become a full tree.
    for bid, seen in branch_commits.items():
        carried = _ids(repo.branches[bid].commit_ids, "branch_commits")
        if tuple(seen) != carried[:len(seen)]:
            raise SourceViewError("branch_commit_history_incomplete")
    return files, catalog, ledger, tuple(all_patches)


def freeze_branch_base(world: Any, branch_id: str) -> SourceSnapshot:
    """Freeze then-current committed main before a branch's first commit."""
    state = _store(world)
    branch = _branch(world, branch_id)
    if branch_id in state["branch_bases"]:
        snapshot = _get(world, state["branch_bases"][branch_id])
        if snapshot.branch_id != branch_id or snapshot.owner_id != branch.owner_id:
            raise SourceViewError("branch_base_identity_changed")
        return snapshot
    if _ids(getattr(branch, "commit_ids", None), "branch_commits"):
        raise SourceViewError("branch_base_unfrozen")
    base = _get(world, state.get("baseline_id"))
    main_ids = _main_commits(world)
    files, catalog, ledger, pids = _replay(world, base, main_ids, merged=True)
    snapshot = _make("branch_base", files, catalog, branch_id=branch_id,
                     owner_id=branch.owner_id, base=base, main_ids=main_ids,
                     patch_ids=pids, ledger=ledger)
    _save(world, snapshot)
    state["branch_bases"][branch_id] = snapshot.snapshot_id
    return snapshot


def _pr(world: Any, pr: Any) -> Any:
    pid = pr if isinstance(pr, str) else getattr(pr, "pr_id", None)
    registered = _repo(world).pull_requests.get(pid)
    if registered is None or getattr(registered, "pr_id", None) != pid:
        raise SourceViewError("pr_ledger_missing")
    if not isinstance(pr, str) and pr is not registered:
        raise SourceViewError("pr_record_not_registered")
    return registered


def _build_pr_head(world: Any, pr: Any, *, allow_branch_descendant: bool = False) -> SourceSnapshot:
    pr = _pr(world, pr)
    branch = _branch(world, pr.source_branch)
    if pr.author_id != branch.owner_id or pr.target_branch != _repo(world).main_branch:
        raise SourceViewError("pr_owner_or_target_mismatch")
    base_id = _store(world)["branch_bases"].get(branch.branch_id)
    if base_id is None:
        raise SourceViewError("branch_base_unfrozen")
    base = _get(world, base_id)
    if base.branch_id != branch.branch_id or base.owner_id != branch.owner_id:
        raise SourceViewError("branch_base_identity_changed")
    commits = _ids(getattr(pr, "commit_ids", None), "pr_commits")
    branch_commits = _ids(branch.commit_ids, "branch_commits")
    expected_commits = branch_commits[:len(commits)] if allow_branch_descendant else branch_commits
    if not commits or commits != expected_commits:
        raise SourceViewError("pr_not_synced_to_branch")
    if set(commits).intersection(base.base_main_commit_ids):
        raise SourceViewError("pr_commit_already_in_base")
    files, catalog, ledger, pids = _replay(
        world, base, commits, branch_id=branch.branch_id, owner_id=pr.author_id)
    if pids != _ids(getattr(pr, "patch_ids", None), "pr_patches"):
        raise SourceViewError("pr_patch_ledger_mismatch")
    return _make("committed_pr_head", files, catalog, branch_id=branch.branch_id,
                 pr_id=pr.pr_id, owner_id=pr.author_id, base=base,
                 main_ids=base.base_main_commit_ids, commit_ids=commits,
                 patch_ids=pids, ledger=ledger)


def freeze_pr_head(world: Any, pr: Any) -> SourceSnapshot:
    """Publish exactly the synchronized committed PR head, retaining old heads."""
    snapshot = _build_pr_head(world, pr)
    _save(world, snapshot)
    _store(world)["pr_heads"][snapshot.pr_id] = snapshot.snapshot_id
    return snapshot


def pr_head_snapshot(world: Any, pr: Any, *, require_current: bool = True) -> SourceSnapshot:
    pr = _pr(world, pr)
    sid = _store(world)["pr_heads"].get(pr.pr_id)
    if sid is None:
        raise SourceViewError("pr_head_unfrozen")
    snapshot = _get(world, sid)
    if snapshot.pr_id != pr.pr_id or snapshot.source_kind != "committed_pr_head":
        raise SourceViewError("pr_snapshot_identity_mismatch")
    if require_current:
        assert_pr_head_current(world, pr, snapshot)
    return snapshot


def pr_base_snapshot(world: Any, pr: Any, *, require_current: bool = True) -> SourceSnapshot:
    """Return the exact committed-main snapshot from which this PR branch forked.

    A later feature may be based on already merged peer work.  Its semantic
    review therefore needs both the untouched public baseline and this protected
    feature base: only the delta from this snapshot belongs to the current PR.
    """
    head = pr_head_snapshot(world, pr, require_current=require_current)
    return _pr_base_from_validated_head(world, head)


def _pr_base_from_validated_head(world: Any, head: SourceSnapshot) -> SourceSnapshot:
    """Resolve one registered head's base without rebuilding that PR a second time."""
    if _get(world, head.snapshot_id) != head:
        raise SourceViewError("snapshot_not_registered")
    base = _get(world, head.base_snapshot_id)
    if (base.source_kind != "branch_base"
            or base.branch_id != head.branch_id
            or base.owner_id != head.owner_id
            or base.base_main_commit_ids != head.base_main_commit_ids
            or base.snapshot_id != head.base_snapshot_id):
        raise SourceViewError("pr_branch_base_identity_mismatch")
    return base


def assert_pr_head_current(world: Any, pr: Any, snapshot: SourceSnapshot) -> None:
    """Pure post-provider guard: both revision IDs and actual ledger bytes match."""
    if _get(world, snapshot.snapshot_id) != snapshot:
        raise SourceViewError("snapshot_not_registered")
    current = _build_pr_head(world, pr)
    if current != snapshot:
        raise SourceViewError("pr_head_changed")


def snapshot_projection(world: Any, snapshot: SourceSnapshot) -> dict[str, Any]:
    """Give the existing exporter an explicit *whole-tree* no-fallback view.

    Pass both returned kwargs to export_product_repo/_repo_hash. In particular,
    using overrides alone could export a new pending file from the shared desk.
    A historical source path with no surviving artifact metadata cannot be
    represented by that exporter and is refused, never silently omitted.
    """
    if _get(world, snapshot.snapshot_id) != snapshot:
        raise SourceViewError("snapshot_not_registered")
    current = _catalog(world)
    frozen = dict(snapshot._artifact_entries)
    for aid, path in frozen.items():
        if aid in current and current[aid] != path:
            raise SourceViewError("artifact_path_changed")
    by_path = {path: aid for aid, path in frozen.items()}
    overrides: dict[str, str] = {}
    for path, content in snapshot.files.items():
        aid = by_path.get(path)
        if aid is None or current.get(aid) != path:
            raise SourceViewError("snapshot_artifact_missing_for_projection")
        overrides[aid] = content
    return {"overrides": overrides,
            "exclude_artifact_ids": frozenset(set(current).difference(overrides))}


def _build_mainline(world: Any) -> SourceSnapshot:
    baseline = _get(world, _store(world).get("baseline_id"))
    main_ids = _main_commits(world)
    files, catalog, ledger, pids = _replay(world, baseline, main_ids, merged=True)
    return _make("committed_mainline", files, catalog, base=baseline,
                 main_ids=main_ids, commit_ids=main_ids, patch_ids=pids, ledger=ledger)


def mainline_snapshot(world: Any) -> SourceSnapshot:
    """Freeze exact committed main for final export, never sampling the desk."""
    return _save(world, _build_mainline(world))


def conservative_merge_candidate(world: Any, pr: Any) -> SourceSnapshot:
    """Freeze current committed main plus the PR's changes, refusing overlap.

    This is a conservative path-level three-way *check*, not an automatic merge:
    a path changed differently on both sides is a conflict even when a text merge
    might succeed. No pending, working-tree or current mainline-content bytes are
    sampled. The receipt's base_snapshot_id identifies the exact published PR
    head; base_main_commit_ids identifies the exact integration main revision.
    """
    head = pr_head_snapshot(world, pr)
    branch_base = _get(world, head.base_snapshot_id)
    committed_main = _build_mainline(world)
    main_ids = committed_main.base_main_commit_ids
    main, catalog = dict(committed_main.files), dict(committed_main._artifact_entries)
    catalog_by_casefold = {path.casefold(): aid for aid, path in catalog.items()}
    for aid, path in head._artifact_entries:
        if aid in catalog and catalog[aid] != path:
            raise SourceViewError("artifact_path_changed")
        folded_path = path.casefold()
        if folded_path in catalog_by_casefold and catalog_by_casefold[folded_path] != aid:
            raise SourceViewError("artifact_path_collision")
        catalog[aid] = path
        catalog_by_casefold[folded_path] = aid
    missing = object()
    conflicts = []
    # Include deletions by comparing the union, preserving empty-file identity.
    # ``SourceSnapshot.files`` intentionally returns a fresh immutable
    # projection.  Keep one projection per snapshot here: rebuilding both
    # full source dictionaries for every path turns this merge check into
    # O(number_of_files ** 2) work on large CooperBench repositories.
    branch_base_files = branch_base.files
    head_files = head.files
    for path in sorted(set(branch_base_files).union(head_files)):
        before = branch_base_files.get(path, missing)
        after = head_files.get(path, missing)
        current = main.get(path, missing)
        if before == after:
            continue
        if current != before and current != after:
            conflicts.append(path)
        elif after is missing:
            main.pop(path, None)
        else:
            main[path] = after
    if conflicts:
        error = SourceViewError("source_merge_conflict")
        error.conflict_paths = tuple(conflicts)
        raise error
    # RepoLite merges by appending the branch's original full-text commits; it
    # does not write a three-way merge commit. A branch which changed a file and
    # then restored its base has no net delta above, but appending those patches
    # would still overwrite a newer peer change. Attest only trees that this
    # actual native promotion/replay path can produce, not a hypothetical merge.
    if set(main_ids).intersection(head.commit_ids):
        raise SourceViewError("pr_commit_already_on_main")
    baseline = _get(world, _store(world).get("baseline_id"))
    touched_paths = tuple(sorted({
        dict(head._artifact_entries)[getattr(world.patches[pid], "target_object_id")]
        for pid in head.patch_ids
    }))
    try:
        native_files, _, _, _ = _replay(world, baseline, (*main_ids, *head.commit_ids))
    except SourceViewError as cause:
        # Both individual trees already passed exact ledger validation. These
        # operation errors therefore describe a cross-branch integration that
        # the current append-only ledger cannot represent (e.g. double delete).
        if cause.code not in {"create_path_already_exists", "delete_tombstone_invalid",
                              "edit_path_missing_from_base", "artifact_path_collision"}:
            raise
        error = SourceViewError("source_merge_conflict")
        error.conflict_paths = touched_paths
        error.native_append_error = cause.code
        raise error from cause
    if native_files != main:
        error = SourceViewError("source_merge_conflict")
        absent = object()
        error.conflict_paths = tuple(sorted(
            path for path in set(native_files).union(main)
            if native_files.get(path, absent) != main.get(path, absent)))
        error.native_append_error = "native_append_tree_differs_from_merge_candidate"
        raise error
    snapshot = _make(
        "merge_candidate", main, catalog, branch_id=head.branch_id,
        pr_id=head.pr_id, owner_id=head.owner_id, base=head, main_ids=main_ids,
        commit_ids=head.commit_ids, patch_ids=head.patch_ids,
        ledger={"main": committed_main.revision_identity,
                "published_head": head.revision_identity})
    return _save(world, snapshot)


@dataclass(frozen=True)
class _ActorPatch:
    actor_id: str
    patch_id: str
    artifact_id: str
    file_path: str
    new_content: str
    creates_file: bool
    deletes_file: bool

    def fingerprint(self) -> tuple:
        return (self.actor_id, self.patch_id, self.artifact_id, self.file_path,
                _hash(self.new_content), self.creates_file, self.deletes_file)


@dataclass(frozen=True)
class PreparedActorPatch:
    """Immutable, uncommitted ticket; preparing one has no world-side effects."""

    actor_id: str
    patch_id: str
    artifact_id: str
    file_path: str
    new_content: str
    creates_file: bool
    deletes_file: bool
    before_snapshot_id: str
    _base_id: str
    _generation: int
    _branch_id: str | None
    _accepted_ids: tuple[str, ...]
    _commit_ids: tuple[str, ...]
    _pending_entries: tuple[tuple[str, str], ...]

    def _record(self) -> _ActorPatch:
        return _ActorPatch(self.actor_id, self.patch_id, self.artifact_id,
                           self.file_path, self.new_content,
                           self.creates_file, self.deletes_file)


def actor_desks_enabled(world: Any) -> bool:
    """Only an explicit opt-in enables private desks; corrupt opt-ins fail shut."""
    return _DESK_STORE in getattr(world, "__dict__", {})


def _desks(world: Any) -> dict:
    _store(world)
    state = getattr(world, "__dict__", {}).get(_DESK_STORE)
    if (not isinstance(state, dict) or state.get("schema_version") != _DESK_SCHEMA
            or not isinstance(state.get("actors"), dict)
            or not isinstance(state.get("sync_receipts"), list)):
        raise SourceViewError("actor_desks_uninitialized_or_invalid")
    return state


def _actor_state(world: Any, actor_id: str) -> dict:
    if not isinstance(actor_id, str) or not actor_id:
        raise SourceViewError("actor_not_registered")
    state = _desks(world)["actors"].get(actor_id)
    if not isinstance(state, dict):
        raise SourceViewError("actor_not_registered")
    if (not isinstance(state.get("base_id"), str)
            or type(state.get("generation")) is not int
            or state["generation"] < 0
            or not isinstance(state.get("patch_records"), tuple)
            or any(not isinstance(record, _ActorPatch) or record.actor_id != actor_id
                   for record in state["patch_records"])
            or (state.get("branch_id") is not None
                and not isinstance(state.get("branch_id"), str))):
        raise SourceViewError("actor_desk_state_invalid")
    return state


def initialize_actor_desks(world: Any, actor_ids: list[str] | tuple[str, ...] | None = None
                           ) -> dict[str, SourceSnapshot]:
    """Pin both owners to bootstrap, before any commits or accepted pending work.

    This does not change v50 source-view behavior unless the caller opts in. It
    deliberately refuses to migrate old shared-desk histories into private ones.
    """
    baseline = _get(world, _store(world).get("baseline_id"))
    owners = (getattr(world, "_cooperbench_sdl_state", {}) or {}).get("feature_owners")
    if not isinstance(owners, dict):
        raise SourceViewError("actor_roster_missing")
    if any(not isinstance(owner, str) or not owner for owner in owners.values()):
        raise SourceViewError("actor_roster_invalid")
    roster = tuple(sorted(set(owners.values())))
    selected = roster if actor_ids is None else _ids(actor_ids, "actor_ids")
    if (len(roster) != 2 or len(selected) != 2 or set(selected) != set(roster)
            or any(not isinstance(aid, str) or not aid for aid in roster)):
        raise SourceViewError("actor_roster_invalid")
    if actor_desks_enabled(world):
        if set(_desks(world)["actors"]) != set(selected):
            raise SourceViewError("actor_roster_changed")
        return {aid: actor_desk_snapshot(world, aid) for aid in selected}
    repo = _repo(world)
    pending = getattr(world, "__dict__", {}).get("_pending_by_branch", {})
    if not isinstance(pending, dict) or any(pending.values()) or repo.commits or repo.main_commit_ids:
        raise SourceViewError("actor_desks_require_fresh_world")
    states = {aid: {"base_id": baseline.snapshot_id, "branch_id": None,
                    "patch_records": (), "generation": 0} for aid in selected}
    snapshots = {aid: _make("actor_desk", baseline.files, dict(baseline._artifact_entries),
                            owner_id=aid, base=baseline,
                            ledger={"schema": _DESK_SCHEMA, "generation": 0,
                                    "committed": [], "pending": []}) for aid in selected}
    for snapshot in snapshots.values():
        _save(world, snapshot)
    world.__dict__[_DESK_STORE] = {
        "schema_version": _DESK_SCHEMA, "actors": states, "sync_receipts": [],
    }
    return snapshots


def _pending_entries(world: Any, branch_id: str | None) -> tuple[tuple[str, str], ...]:
    if branch_id is None:
        return ()
    pending = getattr(world, "__dict__", {}).get("_pending_by_branch", {})
    entries = pending.get(branch_id, ()) if isinstance(pending, dict) else None
    if (not isinstance(entries, (list, tuple))
            or any(not isinstance(entry, (list, tuple)) or len(entry) != 2
                   or any(not isinstance(item, str) or not item for item in entry)
                   for entry in entries)):
        raise SourceViewError("actor_pending_ledger_invalid")
    result = tuple(tuple(entry) for entry in entries)
    if len({pid for pid, _ in result}) != len(result):
        raise SourceViewError("actor_pending_ledger_invalid")
    return result


def _actor_patch(world: Any, actor_id: str, patch: Any, *, accepted: bool) -> _ActorPatch:
    aid = getattr(patch, "target_object_id", None)
    pid = getattr(patch, "patch_id", None)
    catalog = _catalog(world)
    if (getattr(patch, "actor_id", None) != actor_id
            or not isinstance(pid, str) or not pid or aid not in catalog):
        raise SourceViewError("actor_patch_identity_invalid")
    status = getattr(patch, "validation_status", None)
    if status not in {"pending", "accepted"} or (accepted and status != "accepted"):
        raise SourceViewError("actor_patch_not_accepted")
    text = getattr(patch, "new_content", None)
    creates, deletes = getattr(patch, "creates_file", False), getattr(patch, "deletes_file", False)
    if (not isinstance(text, str) or type(creates) is not bool
            or type(deletes) is not bool or (creates and deletes)):
        raise SourceViewError("actor_patch_operation_invalid")
    return _ActorPatch(actor_id, pid, aid, catalog[aid], text, creates, deletes)


def _apply_actor_patch(files: dict, catalog: dict, record: _ActorPatch,
                       created_here: set[str]) -> None:
    path, aid = record.file_path, record.artifact_id
    if aid in catalog and catalog[aid] != path:
        raise SourceViewError("artifact_path_changed")
    if any(other != aid and known.casefold() == path.casefold()
           for other, known in catalog.items()):
        raise SourceViewError("artifact_path_collision")
    if record.creates_file:
        if path in files and aid not in created_here:
            raise SourceViewError("actor_create_path_already_exists")
        files[path] = record.new_content
        created_here.add(aid)
    elif record.deletes_file:
        if path not in files or record.new_content != "":
            raise SourceViewError("actor_delete_tombstone_invalid")
        del files[path]
    else:
        if path not in files:
            raise SourceViewError("actor_edit_path_not_visible")
        files[path] = record.new_content
    catalog[aid] = path


def _build_actor_desk(world: Any, actor_id: str, *, state_override: dict | None = None
                      ) -> SourceSnapshot:
    state = _actor_state(world, actor_id) if state_override is None else state_override
    base = _get(world, state["base_id"])
    bid, records = state["branch_id"], state["patch_records"]
    files, catalog = dict(base.files), dict(base._artifact_entries)
    commits, committed_pids, ledger = (), (), []
    if bid is not None:
        branch = _branch(world, bid)
        if branch.owner_id != actor_id:
            raise SourceViewError("actor_branch_owner_mismatch")
        branch_base = _get(world, _store(world)["branch_bases"].get(bid))
        if (branch_base.source_kind != "branch_base" or branch_base.owner_id != actor_id
                or branch_base.branch_id != bid or branch_base.base_snapshot_id != base.snapshot_id
                or branch_base.files != base.files
                or branch_base.base_main_commit_ids != base.base_main_commit_ids):
            raise SourceViewError("actor_branch_base_mismatch")
        commits = _ids(branch.commit_ids, "branch_commits")
        files, catalog, ledger, committed_pids = _replay(
            world, branch_base, commits, branch_id=bid, owner_id=actor_id)
    elif records:
        raise SourceViewError("actor_branch_missing")
    pending = _pending_entries(world, bid)
    record_ids = tuple(record.patch_id for record in records)
    if len(set(record_ids)) != len(record_ids) or (*committed_pids, *(pid for pid, _ in pending)) != record_ids:
        raise SourceViewError("actor_patch_ledger_incomplete")
    actual_patches = getattr(world, "patches", {})
    for record in records:
        actual = actual_patches.get(record.patch_id)
        if actual is None or _actor_patch(world, actor_id, actual, accepted=True) != record:
            raise SourceViewError("actor_patch_ledger_changed")
    by_id = {record.patch_id: record for record in records}
    created = {by_id[pid].artifact_id for pid in committed_pids if by_id[pid].creates_file}
    pending_fingerprints = []
    for pid, aid in pending:
        record = by_id[pid]
        if record.artifact_id != aid:
            raise SourceViewError("actor_pending_target_mismatch")
        _apply_actor_patch(files, catalog, record, created)
        pending_fingerprints.append(record.fingerprint())
    return _make("actor_desk", files, catalog, owner_id=actor_id, branch_id=bid,
                 base=base, main_ids=base.base_main_commit_ids, commit_ids=commits,
                 patch_ids=record_ids,
                 ledger={"schema": _DESK_SCHEMA, "generation": state["generation"],
                         "committed": ledger, "pending": pending_fingerprints})


def actor_desk_snapshot(world: Any, actor_id: str) -> SourceSnapshot:
    """Freeze only this owner's base + own committed ledger + own accepted pending."""
    return _save(world, _build_actor_desk(world, actor_id))


def _actor_snapshot(world: Any, actor_id: str, snapshot: SourceSnapshot | None) -> SourceSnapshot:
    _actor_state(world, actor_id)
    if snapshot is None:
        return actor_desk_snapshot(world, actor_id)
    if (not isinstance(snapshot, SourceSnapshot) or snapshot.owner_id != actor_id
            or snapshot.source_kind != "actor_desk"
            or _get(world, snapshot.snapshot_id) != snapshot):
        raise SourceViewError("actor_snapshot_identity_mismatch")
    return snapshot


def assert_actor_desk_current(world: Any, actor_id: str, snapshot: SourceSnapshot) -> None:
    """Pure post-provider guard; reading an old frozen view never refreshes it."""
    snapshot = _actor_snapshot(world, actor_id, snapshot)
    if _build_actor_desk(world, actor_id) != snapshot:
        raise SourceViewError("actor_desk_changed")


def actor_artifact_catalog(world: Any, actor_id: str, *, snapshot: SourceSnapshot | None = None
                           ) -> dict[str, str]:
    """Actor-visible path metadata only; peer pending new paths remain absent."""
    snapshot = _actor_snapshot(world, actor_id, snapshot)
    source_files = snapshot.files
    return {aid: path for aid, path in snapshot._artifact_entries if path in source_files}


def actor_file_text(world: Any, actor_id: str, artifact_id: str, *,
                    snapshot: SourceSnapshot | None = None) -> str | None:
    snapshot = _actor_snapshot(world, actor_id, snapshot)
    path = dict(snapshot._artifact_entries).get(artifact_id)
    return snapshot.files.get(path) if path is not None else None


def actor_desk_projection(world: Any, actor_id: str) -> dict[str, Any]:
    return snapshot_projection(world, actor_desk_snapshot(world, actor_id))


def bind_actor_branch(world: Any, actor_id: str, branch_id: str) -> SourceSnapshot:
    """Bind one new feature branch to the owner's pinned base, not current main.

    Call at the record_patch seam before its first pending entry. Existing v50
    branch snapshots are never silently rewritten; a wrong base is a refusal.
    """
    state = _actor_state(world, actor_id)
    branch = _branch(world, branch_id)
    if branch.owner_id != actor_id:
        raise SourceViewError("actor_branch_owner_mismatch")
    if state["branch_id"] not in {None, branch_id}:
        raise SourceViewError("actor_branch_already_bound")
    base = _get(world, state["base_id"])
    expected = _make("branch_base", base.files, dict(base._artifact_entries),
                     owner_id=actor_id, branch_id=branch_id, base=base,
                     main_ids=base.base_main_commit_ids,
                     ledger={"schema": _DESK_SCHEMA, "actor_base": base.snapshot_id})
    existing_id = _store(world)["branch_bases"].get(branch_id)
    if existing_id is not None and _get(world, existing_id) != expected:
        raise SourceViewError("actor_branch_base_mismatch")
    if state["branch_id"] is None:
        if branch.commit_ids or _pending_entries(world, branch_id) or state["patch_records"]:
            raise SourceViewError("actor_branch_requires_fresh_ledger")
    _save(world, expected)
    _store(world)["branch_bases"][branch_id] = expected.snapshot_id
    _desks(world)["actors"][actor_id] = {**state, "branch_id": branch_id}
    return expected


def prepare_actor_patch(world: Any, actor_id: str, patch: Any, *,
                        expected_snapshot_id: str) -> PreparedActorPatch:
    """Validate a provider patch against exactly the private source it received.

    No native artifact, patch ledger, pending entry or private store is changed.
    The caller must encompass native application and commit_actor_patch in one
    rollback transaction, including both Cooper stores and native patch ledgers.
    """
    state = _actor_state(world, actor_id)
    before = _build_actor_desk(world, actor_id)
    if before.snapshot_id != expected_snapshot_id or _get(world, expected_snapshot_id) != before:
        raise SourceViewError("actor_desk_changed")
    record = _actor_patch(world, actor_id, patch, accepted=False)
    if record.patch_id in (getattr(world, "patches", {}) or {}):
        raise SourceViewError("actor_patch_id_already_used")
    files, catalog = dict(before.files), dict(before._artifact_entries)
    created = {item.artifact_id for item in state["patch_records"] if item.creates_file}
    _apply_actor_patch(files, catalog, record, created)
    return PreparedActorPatch(
        record.actor_id, record.patch_id, record.artifact_id, record.file_path,
        record.new_content, record.creates_file, record.deletes_file,
        before.snapshot_id, state["base_id"], state["generation"], state["branch_id"],
        tuple(item.patch_id for item in state["patch_records"]), before.commit_ids,
        _pending_entries(world, state["branch_id"]))


def actor_patch_candidate_snapshot(world: Any, prepared: PreparedActorPatch) -> SourceSnapshot:
    """Freeze the exact private tree that would result from one prepared patch.

    This is a preview only: it does not publish the patch to the actor ledger,
    create a branch, or mutate native product state.  The registered immutable
    snapshot exists so provider-free public probes can inspect the proposed tree
    before the native patch transaction is allowed to begin.
    """
    if not isinstance(prepared, PreparedActorPatch):
        raise SourceViewError("actor_patch_ticket_invalid")
    actor_id = prepared.actor_id
    state = _actor_state(world, actor_id)
    before = _get(world, prepared.before_snapshot_id)
    if (before.source_kind != "actor_desk" or before.owner_id != actor_id
            or state["base_id"] != prepared._base_id
            or state["generation"] != prepared._generation
            or state["branch_id"] != prepared._branch_id
            or tuple(item.patch_id for item in state["patch_records"]) != prepared._accepted_ids
            or before.commit_ids != prepared._commit_ids
            or _pending_entries(world, state["branch_id"]) != prepared._pending_entries
            or _build_actor_desk(world, actor_id) != before):
        raise SourceViewError("actor_patch_transaction_changed")
    files, catalog = dict(before.files), dict(before._artifact_entries)
    created = {item.artifact_id for item in state["patch_records"] if item.creates_file}
    record = prepared._record()
    _apply_actor_patch(files, catalog, record, created)
    snapshot = _make(
        "actor_patch_candidate", files, catalog,
        owner_id=actor_id, branch_id=state["branch_id"], base=before,
        main_ids=before.base_main_commit_ids, commit_ids=before.commit_ids,
        patch_ids=(*before.patch_ids, prepared.patch_id),
        ledger={"schema": _DESK_SCHEMA, "before": before.revision_identity,
                "candidate_patch": record.fingerprint()},
    )
    return _save(world, snapshot)


def commit_actor_patch(world: Any, prepared: PreparedActorPatch, *, branch_id: str
                       ) -> SourceSnapshot:
    """Publish one private pending patch only after exact native acceptance.

    Another owner's concurrent edit is harmless. A competing local edit, commit,
    sync, mismatched patch, or unrelated pending mutation fails the compare/swap.
    First-branch binding between prepare and commit is the one allowed change.
    """
    if not isinstance(prepared, PreparedActorPatch):
        raise SourceViewError("actor_patch_ticket_invalid")
    aid = prepared.actor_id
    state = _actor_state(world, aid)
    branch = _branch(world, branch_id)
    if (state["base_id"] != prepared._base_id or state["generation"] != prepared._generation
            or state["branch_id"] != branch_id or branch.owner_id != aid
            or prepared._branch_id not in {None, branch_id}
            or tuple(record.patch_id for record in state["patch_records"]) != prepared._accepted_ids
            or tuple(branch.commit_ids) != prepared._commit_ids
            or _pending_entries(world, branch_id) != (
                *prepared._pending_entries, (prepared.patch_id, prepared.artifact_id))):
        raise SourceViewError("actor_patch_transaction_changed")
    before = _get(world, prepared.before_snapshot_id)
    if before.owner_id != aid or before.source_kind != "actor_desk":
        raise SourceViewError("actor_patch_ticket_invalid")
    patch = (getattr(world, "patches", {}) or {}).get(prepared.patch_id)
    record = prepared._record()
    if patch is None or _actor_patch(world, aid, patch, accepted=True) != record:
        raise SourceViewError("actor_patch_payload_changed")
    next_state = {**state, "patch_records": (*state["patch_records"], record)}
    snapshot = _build_actor_desk(world, aid, state_override=next_state)
    # Independently apply the prepared patch to its exact old tree, protecting
    # the permitted first-branch-binding transition from changing source bytes.
    expected_files, expected_catalog = dict(before.files), dict(before._artifact_entries)
    created = {item.artifact_id for item in state["patch_records"] if item.creates_file}
    _apply_actor_patch(expected_files, expected_catalog, record, created)
    if snapshot.files != expected_files:
        raise SourceViewError("actor_patch_transaction_changed")
    _save(world, snapshot)
    _desks(world)["actors"][aid] = next_state
    return snapshot


def sync_actor_desk(world: Any, actor_id: str, *, expected_snapshot_id: str) -> SourceSnapshot:
    """Explicitly import committed main, only with no unmerged/private work.

    This stage intentionally implements neither rebase nor dirty-tree merging.
    Divergent edits report paths; even non-overlapping dirty work must first be
    delivered. Refusals preserve the desk, baseline, pending and native history.
    """
    state = _actor_state(world, actor_id)
    before = _build_actor_desk(world, actor_id)
    if before.snapshot_id != expected_snapshot_id or _get(world, expected_snapshot_id) != before:
        raise SourceViewError("actor_desk_changed")
    current_main = _build_mainline(world)
    bid = state["branch_id"]
    pending = _pending_entries(world, bid)
    own_unmerged = tuple(cid for cid in before.commit_ids
                         if getattr(_repo(world).commits[cid], "status", None) != "merged")
    if pending or own_unmerged:
        base = _get(world, state["base_id"])
        absent = object()
        base_files = base.files
        before_files = before.files
        current_main_files = current_main.files
        conflicts = tuple(sorted(
            path
            for path in set(base_files).union(before_files, current_main_files)
            if before_files.get(path, absent) != base_files.get(path, absent)
            and current_main_files.get(path, absent) != base_files.get(path, absent)
            and before_files.get(path, absent) != current_main_files.get(path, absent)
        ))
        error = SourceViewError("actor_sync_conflict" if conflicts else "actor_sync_requires_clean_branch")
        error.conflict_paths = conflicts
        raise error
    if bid is not None and not before.commit_ids:
        # An empty, uncommitted branch can be retained only when the base did not
        # move; updating its frozen base would violate v50 PR history semantics.
        current_main_files = current_main.files
        before_files = before.files
        if (current_main_files != before_files
                or current_main.base_main_commit_ids != before.base_main_commit_ids):
            raise SourceViewError("actor_sync_requires_unbound_or_merged_branch")
        return _save(world, before)
    if bid is None:
        current_main_files = current_main.files
        before_files = before.files
    if (bid is None and current_main.base_main_commit_ids == before.base_main_commit_ids
            and current_main_files == before_files):
        return _save(world, before)
    next_state = {"base_id": current_main.snapshot_id, "branch_id": None,
                  "patch_records": (), "generation": state["generation"] + 1}
    _save(world, current_main)
    after = _build_actor_desk(world, actor_id, state_override=next_state)
    _save(world, after)
    desks = _desks(world)
    desks["actors"][actor_id] = next_state
    desks["sync_receipts"].append({"actor_id": actor_id, "explicit": True,
                                    "before_snapshot_id": before.snapshot_id,
                                    "after_snapshot_id": after.snapshot_id,
                                    "main_commit_ids": list(current_main.base_main_commit_ids)})
    return after


def restart_actor_desk_after_conflict(
    world: Any,
    actor_id: str,
    *,
    pr_id: str,
    expected_snapshot_id: str,
    mainline_snapshot_id: str,
    tick: int,
) -> SourceSnapshot:
    """Abandon one conflicted PR head and restart its owner on current main.

    The stale commits and patches remain in the repository ledger for audit.
    Only the actor's private desk is rebound, so the owner must reimplement and
    deliver the feature through a fresh branch and PR.
    """
    state = _actor_state(world, actor_id)
    before = _build_actor_desk(world, actor_id)
    if (
        before.snapshot_id != expected_snapshot_id
        or _get(world, expected_snapshot_id) != before
    ):
        raise SourceViewError("actor_desk_changed")
    current_main = _build_mainline(world)
    if current_main.snapshot_id != mainline_snapshot_id:
        raise SourceViewError("actor_sync_mainline_changed")
    branch_id = state["branch_id"]
    repo = _repo(world)
    pr = repo.pull_requests.get(pr_id)
    branch = repo.branches.get(branch_id) if branch_id else None
    if (
        pr is None
        or branch is None
        or str(getattr(pr, "author_id", "") or "") != actor_id
        or str(getattr(pr, "source_branch", "") or "") != branch_id
        or str(getattr(branch, "owner_id", "") or "") != actor_id
    ):
        raise SourceViewError("actor_conflict_pr_mismatch")
    status = str(
        getattr(getattr(pr, "status", ""), "value", getattr(pr, "status", ""))
        or ""
    ).casefold()
    if status in {"merged", "closed", "stale"}:
        raise SourceViewError("actor_conflict_pr_not_live")
    pending = _pending_entries(world, branch_id)
    # Restart is explicitly abandoning a stale PR, not reviewing/merging it.
    # Validate its frozen prefix without requiring subsequent desk commits to
    # have already been published to that PR (the condition being repaired).
    head = pr_head_snapshot(world, pr, require_current=False)
    if _build_pr_head(world, pr, allow_branch_descendant=True) != head:
        raise SourceViewError("actor_conflict_pr_head_changed")
    pending_patch_ids = tuple(patch_id for patch_id, _ in pending)
    if (
        head.owner_id != actor_id
        or head.branch_id != branch_id
        or before.commit_ids[:len(head.commit_ids)] != head.commit_ids
        or before.patch_ids[:len(head.patch_ids)] != head.patch_ids
        or (pending_patch_ids and before.patch_ids[-len(pending_patch_ids):] != pending_patch_ids)
        or (not pending and head.commit_ids == before.commit_ids and head.tree_digest != before.tree_digest)
    ):
        raise SourceViewError("actor_conflict_pr_head_changed")

    from environments.org_env.backend.repo.repo import BranchStatus, PRStatus

    pr.status = PRStatus.STALE
    pr.merge_conflict = True
    branch.status = BranchStatus.ABANDONED
    for patch_id in pending_patch_ids:
        patch = world.patches[patch_id]
        patch.validation_status = "rejected"
        patch.rejection_reason = "stale_base_coordination_conflict"
    world.__dict__["_pending_by_branch"][branch_id] = []
    branch.uncommitted_changes = 0
    next_state = {
        "base_id": current_main.snapshot_id,
        "branch_id": None,
        "patch_records": (),
        "generation": state["generation"] + 1,
    }
    _save(world, current_main)
    after = _build_actor_desk(world, actor_id, state_override=next_state)
    _save(world, after)
    desks = _desks(world)
    desks["actors"][actor_id] = next_state
    desks["sync_receipts"].append(
        {
            "actor_id": actor_id,
            "explicit": True,
            "reason": "coordination_conflict_restart",
            "conflicted_pr_id": pr_id,
            "restart_tick": int(tick),
            "abandoned_pending_patch_ids": list(pending_patch_ids),
            "before_snapshot_id": before.snapshot_id,
            "after_snapshot_id": after.snapshot_id,
            "main_commit_ids": list(current_main.base_main_commit_ids),
        }
    )
    return after


__all__ = ["SourceSnapshot", "SourceViewError", "source_views_enabled",
           "freeze_baseline", "freeze_branch_base", "freeze_pr_head",
           "pr_head_snapshot", "pr_base_snapshot", "assert_pr_head_current", "snapshot_projection",
           "conservative_merge_candidate", "mainline_snapshot",
           "PreparedActorPatch", "actor_desks_enabled", "initialize_actor_desks",
           "actor_desk_snapshot", "assert_actor_desk_current", "actor_file_text", "actor_artifact_catalog",
           "actor_desk_projection", "bind_actor_branch", "prepare_actor_patch",
           "actor_patch_candidate_snapshot", "commit_actor_patch", "sync_actor_desk",
           "restart_actor_desk_after_conflict"]
