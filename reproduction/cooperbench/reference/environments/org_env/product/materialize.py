"""Product materialization (2026-06-19 spec) — turn the in-world product state into a REAL,
runnable directory and run its smoke test.

`export_product_repo` writes each non-issue artifact's `content` to `dest/<linked_file_path>`,
producing an actual repo on disk (README + code + corpus + sample task + smoke_check). The
release gate (and the inspector's Export button) call `export_and_smoke`, which then runs
`python smoke_check.py` in a subprocess and reports returncode + the metrics JSON line. This is
how "release_ready" becomes grounded in real files + a real run instead of pure state.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from environments.org_env.product.repo_paths import (
    InvalidRepoPath,
    normalize_repo_relative_path,
)


PROGRAMBENCH_CANDIDATE_VIEW_SCHEMA_VERSION = (
    "programbench_mainline_plus_integration_candidate_probe_isolated_v2"
)


def _is_unmerged_new_file(artifact: Any) -> bool:
    """Whether ``artifact`` has a working-tree path but no mainline path yet.

    ``created_as_new_file`` states this explicitly, but only two creation paths
    ever set it: ``create_new_repo_file`` and the typed ProgramBench contract.
    Every other file the organization writes during a run — design notes,
    onboarding docs, report checklists — was minted with the flag off, so a doc
    first written at tick 15 was indistinguishable from a file the pack shipped.
    The strict ProgramBench mainline view then handed it an explicit
    empty-bytes override and materialized a 0-byte path the starter never had,
    which moved ``programbench_mainline_digest`` off the attested starter digest
    and made ``run_final_evaluation`` refuse to produce any artifact at all.

    Creation provenance answers the question without the flag. Only a merge
    increments ``mainline_revision``, and only pack-shipped artifacts are seeded
    at tick 0, so a path that carries no mainline bytes and that the
    organization itself introduced is absent from mainline whether or not the
    creating action bothered to say so. The two orderings below matter: a merged
    file stays in the view even when its merged bytes are legitimately empty,
    and a pack-shipped file stays in the view even when the pack shipped it
    empty.
    """

    try:
        mainline_revision = int(getattr(artifact, "mainline_revision", 0) or 0)
    except (TypeError, ValueError):
        # A counter we cannot read is not evidence that no merge happened.
        return False
    if mainline_revision != 0:
        return False
    if bool(getattr(artifact, "created_as_new_file", False)):
        return True
    if getattr(artifact, "mainline_content", ""):
        return False
    # A tick we cannot read as a genuine positive integer is treated as pack
    # provenance, which keeps the artifact in the mainline view exactly as
    # before rather than silently dropping bytes on malformed state.
    created_at_tick = getattr(artifact, "created_at_tick", 0)
    return (
        isinstance(created_at_tick, int)
        and not isinstance(created_at_tick, bool)
        and created_at_tick > 0
    )


def _file_text(a: Any, prefer_mainline: bool) -> str:
    if prefer_mainline:
        # A created file has an explicit mainline existence boundary. Never
        # fall through to later working-tree edits, including when the merged
        # mainline file is intentionally empty.
        if getattr(a, "created_as_new_file", False):
            return getattr(a, "mainline_content", "") or ""
        return getattr(a, "mainline_content", "") or getattr(a, "content", "") or ""
    return getattr(a, "content", "") or ""


def _safe_export_path(root: Path, raw: Any) -> tuple[str, Path]:
    try:
        normalized = normalize_repo_relative_path(raw)
    except InvalidRepoPath as exc:
        # Preserve the materializer's public error family while using the same
        # path grammar as execution-time file creation.
        raise ValueError(f"unsafe_product_artifact_path:{raw}") from exc
    relative = Path(*normalized.split("/"))
    target = root.joinpath(relative)
    resolved_parent = target.parent.resolve()
    try:
        resolved_parent.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"product_artifact_path_escapes_export:{normalized}"
        ) from exc
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        is_junction = getattr(cursor, "is_junction", None)
        if cursor.is_symlink() or (
            callable(is_junction) and is_junction()
        ):
            raise ValueError(
                f"product_artifact_path_contains_symlink:{normalized}"
            )
    try:
        target.resolve(strict=False).relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"product_artifact_path_escapes_export:{normalized}"
        ) from exc
    if target.exists():
        metadata = target.lstat()
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise ValueError(
                f"product_artifact_path_is_not_regular_file:{normalized}"
            )
    return relative.as_posix(), target


def _repo_artifact_entries(artifacts: Dict[str, Any]) -> List[tuple[str, str, Any]]:
    """Return path-bearing artifacts after a cross-platform uniqueness check."""
    entries: List[tuple[str, str, Any]] = []
    seen: Dict[str, tuple[str, str]] = {}
    for artifact in artifacts.values():
        if getattr(artifact, "artifact_type", "") == "issue":
            continue
        raw_path = getattr(artifact, "linked_file_path", None)
        if not raw_path:
            continue
        canonical = normalize_repo_relative_path(raw_path)
        artifact_id = str(getattr(artifact, "artifact_id", "") or "")
        collision_key = canonical.casefold()
        previous = seen.get(collision_key)
        if previous is not None:
            previous_path, previous_id = previous
            raise ValueError(
                "product_artifact_path_collision:"
                f"{previous_path}:{previous_id}:{canonical}:{artifact_id}"
            )
        seen[collision_key] = (canonical, artifact_id)
        entries.append((canonical, artifact_id, artifact))
    return sorted(entries, key=lambda row: (row[0], row[1]))


def _atomic_export_text(
    root: Path, path: str, content: str, *, durable: bool = True
) -> Path:
    """Replace one exported file without ever truncating an existing hardlink."""
    if type(durable) is not bool:
        raise ValueError("product_artifact_export_durability_invalid")
    _normalized, target = _safe_export_path(root, path)
    target.parent.mkdir(parents=True, exist_ok=True)
    # Revalidate after mkdir: a pre-existing parent must not have been swapped
    # for a symlink or junction while the directory chain was created.
    _normalized, target = _safe_export_path(root, path)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".product-export-", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(content.encode("utf-8"))
            handle.flush()
            if durable:
                os.fsync(handle.fileno())
        # Close the check-to-write hardlink race. Even if the destination is
        # changed after this check, os.replace swaps the directory entry rather
        # than opening/truncating the possibly hardlinked inode.
        _normalized, checked_target = _safe_export_path(root, path)
        if checked_target != target:
            raise ValueError(f"product_artifact_path_changed:{path}")
        os.replace(temporary, target)
        metadata = target.lstat()
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise ValueError(f"product_artifact_export_not_regular:{path}")
        return target
    finally:
        temporary.unlink(missing_ok=True)


def _runtime_asset_export_context(world: Any, entries, *, verify: bool = False):
    """Capture immutable resources and registered source/deletion paths.

    Resource conflict checks intentionally precede candidate filtering: an
    excluded/deleted/peer-private source path must not reappear as an asset.
    No source contents are selected by this metadata-only helper.

    Native deletion keeps artifact metadata. Baseline paths and source-mode
    metadata remain protected even if their live artifact metadata is absent;
    this does not reconstruct arbitrary lost historical new-file registries.
    """
    from environments.org_env.cooperbench.runtime_assets import (
        RuntimeAssetError, _verify_set, assert_no_runtime_asset_conflicts,
        runtime_assets_for_world,
    )
    assets = runtime_assets_for_world(world)
    from environments.org_env.cooperbench.source_views import freeze_baseline, source_views_enabled
    frozen_baseline = freeze_baseline(world) if source_views_enabled(world) else None
    if assets is None:
        return None, ()
    if verify:
        _verify_set(assets)
    if frozen_baseline is not None:
        # This also rejects post-freeze asset replacement/removal instead of
        # producing a tree under a receipt bound to another resource set.
        baseline = frozen_baseline.files
    else:
        baseline = getattr(world, "_cooperbench_public_baseline_files", {}) or {}
        if not hasattr(baseline, "keys"):
            raise RuntimeAssetError("source_registry_invalid")
    paths = {path for path, _aid, _artifact in entries}
    paths.update(normalize_repo_relative_path(path) for path in baseline)
    paths.update(assets.source_modes)
    protected = tuple(sorted(paths))
    assert_no_runtime_asset_conflicts(assets, protected)
    return assets, protected


def _export_runtime_asset_overlay(
    world: Any,
    root: Path,
    assets,
    protected,
    written,
    *,
    container_asset_root: str | None = None,
):
    from environments.org_env.cooperbench.runtime_assets import (
        RuntimeAssetError, apply_runtime_source_modes, export_runtime_assets,
        link_runtime_assets_for_container, runtime_assets_for_world,
    )
    if runtime_assets_for_world(world) is not assets:
        raise RuntimeAssetError("asset_set_changed_during_export")
    # Missing files in a fresh/incomplete destination are materialized from the
    # frozen set. Missing/corrupt frozen blobs or modified *existing* exported
    # bytes are errors, never repaired from ambient repository contents.
    if container_asset_root is None:
        receipt = export_runtime_assets(
            assets, root, source_paths=protected,
            expected_digest=assets.asset_set_digest,
        )
    else:
        receipt = link_runtime_assets_for_container(
            assets, root, container_root=container_asset_root
        )
    mode_receipt = apply_runtime_source_modes(assets, root, source_paths=written,
                                             expected_digest=assets.asset_set_digest)
    if runtime_assets_for_world(world) is not assets:
        raise RuntimeAssetError("asset_set_changed_during_export")
    return {"runtime_assets": receipt, "source_modes": mode_receipt}


def export_product_repo(
    world: Any,
    dest: str,
    prefer_mainline: bool = False,
    overrides: Dict[str, str] | None = None,
    *,
    exclude_artifact_ids: set[str] | frozenset[str] | None = None,
    _runtime_asset_container_root: str | None = None,
    _durable_writes: bool = True,
) -> Dict[str, Any]:
    """Write each non-issue artifact's real file text to `dest`. `prefer_mainline` exports the
    merged mainline tree (for releases); otherwise the working tree (artifact.content).

    ``overrides`` replaces individual artifacts' text by artifact id, which is how
    a merge candidate is built: the mainline tree plus one pull request's own
    changes and nothing else.
    """
    if type(_durable_writes) is not bool:
        raise ValueError("product_artifact_export_durability_invalid")
    arts = getattr(world, "product_artifacts", {}) or {}
    replacements = overrides or {}
    excluded = set(exclude_artifact_ids or ())
    written = []
    empties = []
    try:
        entries = _repo_artifact_entries(arts)
    except InvalidRepoPath as exc:
        raise ValueError(f"unsafe_product_artifact_path:{exc}") from exc
    assets, protected = _runtime_asset_export_context(world, entries, verify=True)
    if assets is not None:
        # Validate the original destination before resolve can hide a linked
        # root; only the assets-enabled path changes legacy export semantics.
        from environments.org_env.cooperbench.runtime_assets import _root
        root = _root(Path(dest).expanduser(), create=True)
    else:
        root = Path(dest).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
    for path, artifact_id, a in entries:
        _path, full = _safe_export_path(root, path)
        if artifact_id in excluded:
            # Reused export roots must not retain a definition from an older
            # full-tree export.  Probe definitions are captured in their own
            # sandbox before candidate compilation and are never present in
            # the executable build/evaluator workspace.
            if full.exists():
                if not full.is_file():
                    raise ValueError(
                        f"product_artifact_path_is_not_file:{path}"
                    )
                full.unlink()
            continue
        has_override = artifact_id in replacements
        if prefer_mainline and not has_override and _is_unmerged_new_file(a):
            # Reusing an export directory must not make a previously exported
            # working file leak into the mainline view.
            if full.exists():
                if not full.is_file():
                    raise ValueError(f"product_artifact_path_is_not_file:{path}")
                full.unlink()
            continue
        content = replacements.get(artifact_id) if has_override else None
        if not has_override:
            content = _file_text(a, prefer_mainline)
        _atomic_export_text(root, path, content, durable=_durable_writes)
        written.append(path)
        if not content.strip():
            empties.append(path)
    result = {"dir": str(root), "files": sorted(written),
              "file_count": len(written), "empty_files": sorted(empties)}
    if assets is not None:
        result.update(
            _export_runtime_asset_overlay(
                world,
                root,
                assets,
                protected,
                written,
                container_asset_root=_runtime_asset_container_root,
            )
        )
        result["files"] = sorted([*written, *assets.paths])
        result["file_count"] = len(result["files"])
    return result


def _programbench_candidate_view_profile_state(world: Any) -> Dict[str, Any]:
    """Return the active adapted-profile state or fail closed."""

    from environments.org_env.programbench import (
        get_programbench_profile_state,
        programbench_profile_active,
        validate_programbench_profile_state_for_step,
    )

    if not programbench_profile_active(world):
        raise ValueError("programbench_candidate_view_profile_not_active")
    state = get_programbench_profile_state(world)
    if not isinstance(state, dict):
        raise ValueError("programbench_candidate_view_profile_state_invalid")
    try:
        validate_programbench_profile_state_for_step(world)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "programbench_candidate_view_profile_state_invalid"
        ) from error
    return state


def _programbench_candidate_view_branch(world: Any, state: Dict[str, Any]):
    """Resolve the unique live integration branch and validate its owner."""

    from environments.org_env.backend.repo.workflow import (
        programbench_integration_branches,
    )

    branches = programbench_integration_branches(world)
    if len(branches) > 1:
        raise ValueError("programbench_candidate_view_multiple_integration_branches")
    if not branches:
        return None
    owners = [
        str(row.get("agent_id") or "")
        for row in (state.get("role_assignments") or [])
        if isinstance(row, dict) and row.get("work_role") == "integration_owner"
    ]
    if (
        len(owners) != 1
        or not owners[0]
        or str(getattr(branches[0], "owner_id", "") or "") != owners[0]
    ):
        raise ValueError("programbench_candidate_view_integration_owner_invalid")
    return branches[0]


def _programbench_candidate_patch_override(
    world: Any,
    *,
    patch_id: Any,
    artifact_id: Any,
    seen_patch_ids: set[str],
    introduced_artifact_ids: set[str],
) -> tuple[str, str]:
    """Validate one full-text integration patch and return its override."""

    normalized_patch_id = str(patch_id or "")
    normalized_artifact_id = str(artifact_id or "")
    if not normalized_patch_id or not normalized_artifact_id:
        raise ValueError("programbench_candidate_view_patch_identity_missing")
    if normalized_patch_id in seen_patch_ids:
        raise ValueError(
            "programbench_candidate_view_patch_duplicate:" + normalized_patch_id
        )
    patch = (getattr(world, "patches", {}) or {}).get(normalized_patch_id)
    artifact = (getattr(world, "product_artifacts", {}) or {}).get(
        normalized_artifact_id
    )
    if patch is None:
        raise ValueError(
            "programbench_candidate_view_patch_missing:" + normalized_patch_id
        )
    if artifact is None:
        raise ValueError(
            "programbench_candidate_view_artifact_missing:" + normalized_artifact_id
        )
    if str(getattr(patch, "target_object_id", "") or "") != normalized_artifact_id:
        raise ValueError(
            "programbench_candidate_view_patch_artifact_mismatch:"
            + normalized_patch_id
        )
    if str(getattr(patch, "validation_status", "") or "") != "accepted":
        raise ValueError(
            "programbench_candidate_view_patch_not_accepted:" + normalized_patch_id
        )
    # Routing classification is deliberately not an inclusion filter here.
    # Once the unique integration ledger carries a patch, that full-text patch
    # is part of the official candidate even if it is documentation/data.  The
    # full view must not silently discard bytes the integration owner chose to
    # submit.
    content = getattr(patch, "new_content", None)
    if not isinstance(content, str):
        raise ValueError(
            "programbench_candidate_view_patch_full_text_missing:"
            + normalized_patch_id
        )
    base_revision = getattr(patch, "base_mainline_revision", None)
    current_mainline_revision = getattr(artifact, "mainline_revision", 0)
    if (
        isinstance(base_revision, bool)
        or not isinstance(base_revision, int)
        or base_revision < 0
        or isinstance(current_mainline_revision, bool)
        or not isinstance(current_mainline_revision, int)
        or current_mainline_revision < 0
        or base_revision != current_mainline_revision
    ):
        raise ValueError(
            "programbench_candidate_view_patch_base_stale:" + normalized_patch_id
        )
    creates_file = getattr(patch, "creates_file", None)
    if not isinstance(creates_file, bool):
        raise ValueError(
            "programbench_candidate_view_patch_create_semantics_missing:"
            + normalized_patch_id
        )
    unmerged_new_file = _is_unmerged_new_file(artifact)
    if creates_file:
        if not unmerged_new_file or normalized_artifact_id in introduced_artifact_ids:
            raise ValueError(
                "programbench_candidate_view_invalid_create_patch:"
                + normalized_patch_id
            )
        introduced_artifact_ids.add(normalized_artifact_id)
    elif unmerged_new_file and normalized_artifact_id not in introduced_artifact_ids:
        raise ValueError(
            "programbench_candidate_view_new_file_create_missing:"
            + normalized_patch_id
        )
    seen_patch_ids.add(normalized_patch_id)
    return normalized_artifact_id, content


def programbench_integration_candidate_overrides(
    world: Any,
    *,
    include_pending: bool = True,
) -> Dict[str, str]:
    """Return the exact adapted candidate view's full-text overrides.

    The official view is the complete mainline tree plus the one designated
    integration branch, replayed in branch/commit/patch order.  Pending patches
    are included because agents run public differential probes against their
    latest candidate before committing it.  Collateral working-tree branches
    are deliberately absent; their files remain present at mainline bytes.

    Every ledger edge is validated.  Missing patches, mismatched targets,
    rejected/non-full-text patches, stale create bases, duplicate integration
    branches, or candidate-bearing working bytes outside this branch fail
    closed instead of being inferred from ambient ``artifact.content``.
    """

    state = _programbench_candidate_view_profile_state(world)
    branch = _programbench_candidate_view_branch(world, state)
    repo = getattr(getattr(world, "repo_system", None), "repo", None)
    commits = getattr(repo, "commits", {}) if repo is not None else {}
    overrides: Dict[str, str] = {}
    seen_patch_ids: set[str] = set()
    introduced_artifact_ids: set[str] = set()
    if branch is not None:
        commit_ids = list(getattr(branch, "commit_ids", None) or [])
        if len(commit_ids) != len(set(commit_ids)):
            raise ValueError("programbench_candidate_view_commit_duplicate")
        for commit_id in commit_ids:
            commit = commits.get(commit_id)
            if (
                commit is None
                or str(getattr(commit, "branch_id", "") or "")
                != str(getattr(branch, "branch_id", "") or "")
            ):
                raise ValueError(
                    "programbench_candidate_view_commit_invalid:" + str(commit_id)
                )
            patch_ids = list(getattr(commit, "patch_ids", None) or [])
            artifact_ids = list(getattr(commit, "artifact_ids", None) or [])
            if len(patch_ids) != len(artifact_ids):
                raise ValueError(
                    "programbench_candidate_view_commit_patch_alignment_invalid:"
                    + str(commit_id)
                )
            for patch_id, artifact_id in zip(patch_ids, artifact_ids):
                normalized_artifact_id, content = (
                    _programbench_candidate_patch_override(
                        world,
                        patch_id=patch_id,
                        artifact_id=artifact_id,
                        seen_patch_ids=seen_patch_ids,
                        introduced_artifact_ids=introduced_artifact_ids,
                    )
                )
                overrides[normalized_artifact_id] = content

        pending_book = getattr(world, "__dict__", {}).get("_pending_by_branch", {})
        if not isinstance(pending_book, dict):
            raise ValueError("programbench_candidate_view_pending_ledger_invalid")
        raw_pending = pending_book.get(str(getattr(branch, "branch_id", "") or ""), [])
        if not isinstance(raw_pending, list) or any(
            not isinstance(row, (list, tuple)) or len(row) != 2
            for row in raw_pending
        ):
            raise ValueError("programbench_candidate_view_pending_rows_invalid")
        pending = list(raw_pending)
        uncommitted = getattr(branch, "uncommitted_changes", None)
        if (
            isinstance(uncommitted, bool)
            or not isinstance(uncommitted, int)
            or uncommitted != len(pending)
        ):
            raise ValueError("programbench_candidate_view_pending_count_mismatch")
        if include_pending:
            for patch_id, artifact_id in pending:
                normalized_artifact_id, content = (
                    _programbench_candidate_patch_override(
                        world,
                        patch_id=patch_id,
                        artifact_id=artifact_id,
                        seen_patch_ids=seen_patch_ids,
                        introduced_artifact_ids=introduced_artifact_ids,
                    )
                )
                overrides[normalized_artifact_id] = content

    from environments.org_env.backend.repo.workflow import (
        programbench_candidate_bearing_artifact,
    )

    for _path, artifact_id, artifact in _repo_artifact_entries(
        getattr(world, "product_artifacts", {}) or {}
    ):
        if not programbench_candidate_bearing_artifact(world, artifact):
            continue
        working = _file_text(artifact, prefer_mainline=False)
        if artifact_id in overrides:
            if overrides[artifact_id] != working:
                raise ValueError(
                    "programbench_candidate_view_working_text_untracked:" + artifact_id
                )
            continue
        if _is_unmerged_new_file(artifact):
            if working:
                raise ValueError(
                    "programbench_candidate_view_unrouted_new_file:" + artifact_id
                )
            continue
        mainline = getattr(artifact, "mainline_content", "")
        if not isinstance(mainline, str):
            raise ValueError(
                "programbench_candidate_view_mainline_text_invalid:" + artifact_id
            )
        if working != mainline:
            raise ValueError(
                "programbench_candidate_view_unrouted_working_change:" + artifact_id
            )
    return overrides


def programbench_integration_candidate_has_pending(world: Any) -> bool:
    """Whether the validated integration candidate still has uncommitted work."""

    state = _programbench_candidate_view_profile_state(world)
    branch = _programbench_candidate_view_branch(world, state)
    if branch is None:
        # Validate that there is no unrouted candidate work as well.
        programbench_integration_candidate_overrides(world)
        return False
    programbench_integration_candidate_overrides(world)
    pending = getattr(world, "__dict__", {}).get("_pending_by_branch", {}) or {}
    return bool(pending.get(str(getattr(branch, "branch_id", "") or ""), []))


def programbench_strict_mainline_overrides(
    world: Any,
    *,
    candidate_overrides: Dict[str, str] | None = None,
) -> Dict[str, str]:
    """Materialize explicit mainline bytes, including legitimate empty files.

    Generic legacy materialization treats an empty ``mainline_content`` as
    missing and falls back to ambient working content.  That compatibility
    behavior is intentionally preserved for native runs, but it is not a safe
    base for the ProgramBench official view: a collateral edit to an empty seed
    would leak into the candidate.  Here every existing mainline artifact gets
    an explicit override, while unmerged new files remain absent unless the
    validated integration branch supplies them.
    """

    replacements: Dict[str, str] = {}
    for _path, artifact_id, artifact in _repo_artifact_entries(
        getattr(world, "product_artifacts", {}) or {}
    ):
        if _is_unmerged_new_file(artifact):
            continue
        content = getattr(artifact, "mainline_content", "")
        if not isinstance(content, str):
            raise ValueError(
                "programbench_candidate_view_mainline_text_invalid:" + artifact_id
            )
        replacements[artifact_id] = content
    for artifact_id, content in (candidate_overrides or {}).items():
        if not isinstance(content, str):
            raise ValueError(
                "programbench_candidate_view_override_text_invalid:"
                + str(artifact_id)
            )
        replacements[str(artifact_id)] = content
    return replacements


def _programbench_probe_definition_artifact_ids(world: Any) -> frozenset[str]:
    """Return the frozen public-probe artifacts excluded from build views.

    Definitions remain ordinary candidate-bearing repository work: they are
    patched, committed, reviewed and merged on the integration branch.  Their
    code is executed once in the dedicated definition sandbox to capture the
    public input corpus.  Candidate compilation and hidden evaluation then use
    a physically separate view without those executable definitions, making
    indirect shell references (variables, ``find``, helper scripts) incapable
    of turning EXPLORE-authored probe bytes into the product executable.
    """

    from environments.org_env.programbench import (
        programbench_frozen_public_probe_contract,
        programbench_frozen_public_probe_path_status,
    )

    frozen = programbench_frozen_public_probe_contract(world)
    surface = frozen["definition_surface"]
    exact = set(str(path) for path in surface["exact_paths"])
    patterns = tuple(re.compile(str(value)) for value in surface["path_patterns"])
    selected: set[str] = set()
    for artifact in (getattr(world, "product_artifacts", {}) or {}).values():
        raw_path = getattr(artifact, "linked_file_path", None)
        if not raw_path:
            continue
        status = programbench_frozen_public_probe_path_status(world, raw_path)
        if status.startswith("invalid"):
            raise ValueError("programbench_public_probe_path_alias_invalid")
        if status != "trusted":
            continue
        path = normalize_repo_relative_path(raw_path)
        artifact_id = str(getattr(artifact, "artifact_id", "") or "")
        if not artifact_id:
            raise ValueError("programbench_public_probe_artifact_id_missing")
        selected.add(artifact_id)
    return frozenset(selected)


def programbench_integration_candidate_digest(world: Any) -> str:
    """SHA-1 identity of full mainline plus the validated integration branch."""

    candidate_overrides = programbench_integration_candidate_overrides(world)
    return _repo_hash(
        world,
        prefer_mainline=True,
        overrides=programbench_strict_mainline_overrides(
            world,
            candidate_overrides=candidate_overrides,
        ),
        exclude_artifact_ids=_programbench_probe_definition_artifact_ids(world),
    )


def programbench_mainline_digest(world: Any) -> str:
    """SHA-1 identity of the strict full ProgramBench mainline view."""

    _programbench_candidate_view_profile_state(world)
    return _repo_hash(
        world,
        prefer_mainline=True,
        overrides=programbench_strict_mainline_overrides(world),
        exclude_artifact_ids=_programbench_probe_definition_artifact_ids(world),
    )


def programbench_pr_candidate_overrides(world: Any, pr: Any) -> Dict[str, str]:
    """Replay the exact synchronized integration PR ledger, fail closed.

    This deliberately does not call the native ``merge_candidate_text`` helper:
    that helper retains compatibility fallbacks for legacy patches without
    full-text content and treats an empty string as missing for existing files.
    ProgramBench patches are accepted full-text records, so ``""`` is an exact,
    valid override and every ledger edge must be attributable.
    """

    state = _programbench_candidate_view_profile_state(world)
    branch = _programbench_candidate_view_branch(world, state)
    if branch is None:
        raise ValueError("programbench_pr_candidate_integration_branch_missing")
    if str(getattr(pr, "source_branch", "") or "") != str(
        getattr(branch, "branch_id", "") or ""
    ):
        raise ValueError("programbench_pr_candidate_source_branch_invalid")
    commit_ids = list(getattr(branch, "commit_ids", None) or [])
    pr_commit_ids = list(getattr(pr, "commit_ids", None) or [])
    if not commit_ids or pr_commit_ids != commit_ids:
        raise ValueError("programbench_pr_candidate_head_not_synchronized")
    if len(commit_ids) != len(set(commit_ids)):
        raise ValueError("programbench_pr_candidate_commit_duplicate")
    repo = getattr(getattr(world, "repo_system", None), "repo", None)
    commits = getattr(repo, "commits", {}) if repo is not None else {}
    expected_pr_patch_ids: List[str] = []
    overrides: Dict[str, str] = {}
    seen_patch_ids: set[str] = set()
    introduced_artifact_ids: set[str] = set()
    for commit_id in commit_ids:
        commit = commits.get(commit_id)
        if (
            commit is None
            or str(getattr(commit, "branch_id", "") or "")
            != str(getattr(branch, "branch_id", "") or "")
        ):
            raise ValueError(
                "programbench_pr_candidate_commit_invalid:" + str(commit_id)
            )
        patch_ids = list(getattr(commit, "patch_ids", None) or [])
        artifact_ids = list(getattr(commit, "artifact_ids", None) or [])
        if len(patch_ids) != len(artifact_ids):
            raise ValueError(
                "programbench_pr_candidate_patch_alignment_invalid:" + str(commit_id)
            )
        expected_pr_patch_ids.extend(str(patch_id) for patch_id in patch_ids)
        for patch_id, artifact_id in zip(patch_ids, artifact_ids):
            normalized_artifact_id, content = _programbench_candidate_patch_override(
                world,
                patch_id=patch_id,
                artifact_id=artifact_id,
                seen_patch_ids=seen_patch_ids,
                introduced_artifact_ids=introduced_artifact_ids,
            )
            overrides[normalized_artifact_id] = content
    if list(getattr(pr, "patch_ids", None) or []) != expected_pr_patch_ids:
        raise ValueError("programbench_pr_candidate_patch_ledger_not_synchronized")
    return overrides


def programbench_pr_candidate_digest(world: Any, pr: Any) -> str:
    """Strict full-mainline identity after applying exactly one PR."""

    _programbench_candidate_view_profile_state(world)
    return _repo_hash(
        world,
        prefer_mainline=True,
        overrides=programbench_strict_mainline_overrides(
            world,
            candidate_overrides=programbench_pr_candidate_overrides(world, pr),
        ),
        exclude_artifact_ids=_programbench_probe_definition_artifact_ids(world),
    )


def export_programbench_integration_candidate(
    world: Any,
    dest: str,
) -> Dict[str, Any]:
    """Export the full adapted candidate view without ambient branch leakage."""

    candidate_overrides = programbench_integration_candidate_overrides(world)
    return export_product_repo(
        world,
        dest,
        prefer_mainline=True,
        overrides=programbench_strict_mainline_overrides(
            world,
            candidate_overrides=candidate_overrides,
        ),
        exclude_artifact_ids=_programbench_probe_definition_artifact_ids(world),
    )


def export_programbench_mainline(world: Any, dest: str) -> Dict[str, Any]:
    """Export strict full mainline bytes for adapted delivery provenance."""

    _programbench_candidate_view_profile_state(world)
    return export_product_repo(
        world,
        dest,
        prefer_mainline=True,
        overrides=programbench_strict_mainline_overrides(world),
        exclude_artifact_ids=_programbench_probe_definition_artifact_ids(world),
    )


def _product_llm_env() -> Dict[str, str]:
    """LLM creds for the materialized product's OWN calls (a cheap model), so the product can
    really call an LLM during a real org run. Gated on ORG_LLM (off in tests -> offline fallback,
    deterministic + free). Reads config/llm(.local).yaml."""
    if (os.environ.get("ORG_LLM", "") or "").lower() not in ("1", "true", "yes", "on"):
        return {}
    try:
        import yaml
    except Exception:
        return {}
    for path in ("config/llm.local.yaml", "config/llm.yaml"):
        try:
            with open(path, encoding="utf-8") as fh:
                cfg = yaml.safe_load(fh) or {}
        except Exception:
            continue
        org = cfg.get("org_env", {}) or {}
        key = org.get("api_key")
        base = org.get("base_url") or "https://api.openai.com/v1"
        if not key:
            ag = ((cfg.get("agent", {}) or {}).get("openai", {}) or {})
            keys = ag.get("api_keys") or []
            key = keys[0] if keys else None
            base = ag.get("base_url") or base
        if key:
            # the product uses a CHEAP model for its own synthesis (org cognition stays separate)
            return {"LANTERN_LLM_KEY": str(key), "LANTERN_LLM_BASE": str(base),
                    "LANTERN_LLM_MODEL": "gpt-4o-mini"}
    return {}


# the question the user-facing CLI is exercised with (mirrors smoke_check.py's QUERY)
_CLI_SMOKE_QUERY = "What are the leading open-source vector databases and their tradeoffs?"
# the real user entry points that MUST run (not just smoke_check.py, which can bypass them)
_USER_ENTRY_FILES = ("agent.py",)


def _last_error_line(stderr: str) -> str:
    lines = [ln.rstrip() for ln in (stderr or "").splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def _safe_product_environment(
    repo_dir: str,
    extra: Optional[Dict[str, str]],
) -> Dict[str, str]:
    allowed = (
        "PATH",
        "TMPDIR",
        "TEMP",
        "TMP",
        "LANG",
        "LC_ALL",
        "SYSTEMROOT",
    )
    environment = {
        key: os.environ[key]
        for key in allowed
        if os.environ.get(key)
    }
    environment.update(
        {
            "HOME": os.path.join(repo_dir, ".product-home"),
            "PYTHONNOUSERSITE": "1",
            "PYTHONPATH": os.pathsep.join(
                (
                    repo_dir,
                    os.path.join(repo_dir, "src"),
                )
            ),
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_TERMINAL_PROMPT": "0",
            "HTTP_PROXY": "http://127.0.0.1:9",
            "HTTPS_PROXY": "http://127.0.0.1:9",
            "ALL_PROXY": "http://127.0.0.1:9",
            "NO_PROXY": "",
            "no_proxy": "",
        }
    )
    os.makedirs(environment["HOME"], exist_ok=True)
    if extra:
        permitted = {
            "LANTERN_LLM_KEY",
            "LANTERN_LLM_BASE",
            "LANTERN_LLM_MODEL",
        }
        unexpected = set(extra) - permitted
        if unexpected:
            raise ValueError(
                "unsupported_product_environment:"
                + ",".join(sorted(unexpected))
            )
        environment.update(
            {key: str(value) for key, value in extra.items()}
        )
    return environment


def _formal_product_executor():
    mode = str(os.environ.get("ORG_OSS_MODE") or "").lower()
    if mode not in {"formal", "pilot"}:
        return None
    from society_core.execution import ExecutionPolicy, build_command_executor

    backend = str(os.environ.get("ORG_EVALUATOR_BACKEND") or "")
    if backend == "bubblewrap":
        if mode != "pilot":
            raise RuntimeError("formal_product_runtime_requires_container")
        virtualenv_path = os.environ.get("ORG_EVALUATOR_VENV")
        if not virtualenv_path:
            raise RuntimeError(
                "pilot_product_runtime_binding_missing:virtualenv_path"
            )
        return build_command_executor(
            ExecutionPolicy(
                trust_level="untrusted",
                backend="bubblewrap",
                virtualenv_path=virtualenv_path,
                network_enabled=False,
                memory_limit_mb=2048,
                cpu_limit=2.0,
                pids_limit=128,
            )
        )
    values = {
        "backend": backend,
        "container_image": os.environ.get(
            "ORG_EVALUATOR_CONTAINER_IMAGE"
        ),
        "container_platform": os.environ.get(
            "ORG_EVALUATOR_CONTAINER_PLATFORM"
        ),
    }
    missing = sorted(key for key, value in values.items() if not value)
    if missing:
        raise RuntimeError(
            "formal_product_runtime_binding_missing:"
            + ",".join(missing)
        )
    return build_command_executor(
        ExecutionPolicy(
            trust_level="untrusted",
            backend=str(values["backend"]),
            container_image=str(values["container_image"]),
            container_platform=str(values["container_platform"]),
            clear_container_entrypoint=(
                str(
                    os.environ.get("ORG_EVALUATOR_CLEAR_ENTRYPOINT") or ""
                ).strip()
                == "1"
            ),
            network_enabled=False,
            memory_limit_mb=2048,
            cpu_limit=2.0,
            pids_limit=128,
        )
    )


def _container_command(command: List[str]) -> tuple[str, ...]:
    normalized = list(command)
    if normalized and (
        normalized[0] in {"python", "python3"}
        or Path(normalized[0]).name.startswith("python")
    ):
        normalized[0] = "python"
    return tuple(normalized)


_WORKSPACE_PROBE_SCRIPT = (
    "import os,sys;sys.stdout.write(str(len(os.listdir('.'))))"
)
_WORKSPACE_PROBE_INTERPRETERS = ("python3", "python")


def _workspace_probe_failure_detail(outcome: Any) -> str:
    """Bounded public diagnostic for a probe process that actually launched.

    The old ``workspace_probe_exit_1`` code erased the only useful evidence.
    CooperBench images commonly wrap commands in their own ENTRYPOINT, so the
    omitted stderr was the difference between "candidate code is broken" and
    "the task runner swallowed our command".  Keep one compact line; this text
    is attached to the CI record and therefore reaches the PR perception.
    """

    text = "\n".join(
        str(value or "")
        for value in (
            getattr(outcome, "stderr_tail", ""),
            getattr(outcome, "stdout_tail", ""),
            getattr(outcome, "blocked_reason", ""),
        )
    )
    compact = " ".join(text.split())[:360]
    return compact


def _workspace_visibility_error(executor: Any, repo_dir: str) -> Optional[str]:
    """Return an infrastructure error when the executor cannot see the exported tree.

    A containerized executor mounts ``repo_dir`` into the sandbox. When the host
    path lies outside the container runtime's shared-filesystem set (on macOS,
    Docker Desktop shares neither the system TMPDIR under ``/var/folders`` nor
    ``/tmp`` by default), the bind mount SUCCEEDS and yields an EMPTY directory.
    Every command then fails with an import/file error that looks exactly like a
    broken product, which is how a 336-tick formal run recorded 172/172
    ``ci_contract_break`` while the product was fine.

    The invariant checked here is substrate-independent: the sandbox must see the
    same non-empty workspace the host just wrote.
    """
    host_entries = 0
    try:
        host_entries = len(os.listdir(repo_dir))
    except OSError as exc:
        return f"workspace_unreadable_on_host:{exc}"
    if host_entries == 0:
        return "workspace_export_empty"
    try:
        outcome = None
        for interpreter in _WORKSPACE_PROBE_INTERPRETERS:
            outcome = executor.run(
                root=Path(repo_dir),
                argv=(interpreter, "-c", _WORKSPACE_PROBE_SCRIPT),
                timeout_seconds=60.0,
            )
            # ProgramBench cleanroom images expose ``python3`` but not the
            # optional ``python`` alias; some older OSS images do the reverse.
            # Exit 126/127 is unambiguously a launch failure for this fixed
            # probe, so try the other conventional Python executable before
            # declaring an infrastructure fault.
            if outcome.exit_code not in {126, 127}:
                break
    except Exception as exc:  # executor/runtime unavailable is infra, not product
        return f"workspace_probe_failed:{exc!r}"
    if outcome is None or outcome.exit_code in {126, 127}:
        return "workspace_probe_interpreter_unavailable:python3,python"
    if outcome.status in {"blocked", "timeout", "infra_error"}:
        return f"workspace_probe_{outcome.status}:{outcome.blocked_reason or ''}"
    if outcome.exit_code != 0:
        detail = _workspace_probe_failure_detail(outcome)
        return f"workspace_probe_exit_{outcome.exit_code}" + (
            f":{detail}" if detail else ""
        )
    seen = (outcome.stdout_tail or "").strip()
    if seen.isdigit() and int(seen) == 0:
        return (
            "workspace_mount_empty:host_has_"
            f"{host_entries}_entries_sandbox_sees_0 "
            f"(host path {repo_dir!r} is not shared with the container runtime; "
            "export to a shared location or configure ORG_PRODUCT_SMOKE_ROOT)"
        )
    return None


def _execute_smoke_command(
    *,
    command: List[str],
    repo_dir: str,
    timeout: int,
    environment: Dict[str, str],
    executor: Any,
    read_only_mounts: tuple[tuple[Path, str], ...] = (),
    image_workspace: str | None = None,
) -> tuple[int | None, str, str, str | None]:
    if executor is not None:
        extra: dict[str, Any] = {}
        if image_workspace:
            runner = getattr(executor, "run_in_image_workspace", None)
            if runner is None:
                return None, "", "", "executor_image_workspace_unsupported"
            extra["image_workspace"] = image_workspace
            extra["read_only_mounts"] = read_only_mounts
        else:
            visibility_error = _workspace_visibility_error(executor, repo_dir)
            if visibility_error:
                return None, "", "", visibility_error
            runner = executor.run
        if read_only_mounts and not image_workspace:
            runner = getattr(executor, "run_with_read_only_mounts", None)
            if runner is None:
                return None, "", "", "executor_read_only_mounts_unsupported"
            extra["read_only_mounts"] = read_only_mounts
        outcome = runner(
            root=Path(repo_dir),
            argv=_container_command(command),
            timeout_seconds=float(timeout),
            **extra,
        )
        error = (
            outcome.blocked_reason
            if outcome.status in {"blocked", "timeout", "infra_error"}
            else None
        )
        return (
            outcome.exit_code,
            outcome.stdout_tail,
            outcome.stderr_tail,
            error,
        )
    # A manifest says `python` because that is what the command means inside the
    # evaluator image. Run locally there may be no bare `python` on PATH (a venv
    # commonly exposes only `python3`), and OSError then reads as "the suite is
    # broken" when nothing about the product was ever executed. The container
    # branch above keeps the manifest spelling; only the host run is rebound.
    local_command = list(command)
    if local_command and local_command[0] in ("python", "python3"):
        local_command[0] = sys.executable
    try:
        process = subprocess.run(
            local_command,
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=environment,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return None, "", "", f"smoke timed out after {timeout}s"
    except OSError as exc:
        return None, "", "", str(exc)
    return (
        process.returncode,
        process.stdout or "",
        process.stderr or "",
        None,
    )


def run_product_smoke(
    repo_dir: str,
    timeout: int = 25,
    env: Dict[str, str] = None,
    command: Optional[List[str]] = None,
    executor: Any = None,
    read_only_mounts: tuple[tuple[Path, str], ...] = (),
    image_workspace: str | None = None,
) -> Dict[str, Any]:
    """Run the product's declared smoke command inside the exported repo.

    OSS time-machine manifests can provide a language-appropriate command (for example a Node or
    source-contract smoke). The legacy LanternScout path still defaults to ``python smoke_check.py``
    and additionally exercises ``agent.py`` so the user-facing CLI cannot silently regress."""
    declared_command = command is not None
    smoke_command = list(command or [sys.executable, "smoke_check.py"])
    if smoke_command and smoke_command[0] in ("python", "python3"):
        smoke_command[0] = sys.executable
    if not declared_command and not os.path.isfile(os.path.join(repo_dir, "smoke_check.py")):
        return {"ok": False, "error": "no smoke_check.py in exported repo"}
    formal_executor = executor if executor is not None else _formal_product_executor()
    run_env = _safe_product_environment(repo_dir, env)
    execute_kwargs = dict(
        command=smoke_command,
        repo_dir=repo_dir,
        timeout=timeout,
        environment=run_env,
        executor=formal_executor,
        read_only_mounts=read_only_mounts,
    )
    if image_workspace:
        execute_kwargs["image_workspace"] = image_workspace
    returncode, stdout, stderr, launch_error = _execute_smoke_command(
        **execute_kwargs
    )
    if launch_error:
        return {"ok": False, "error": launch_error}
    metrics, summary = None, None
    for line in stdout.splitlines():
        s = line.strip()
        if s.startswith("{") and "metrics" in s:
            try:
                summary = json.loads(s)
                metrics = summary.get("metrics")
                break
            except Exception:
                pass
    result = {"ok": returncode == 0, "returncode": returncode, "metrics": metrics,
              "summary": summary, "stdout_tail": stdout[-2000:],
              "stderr_tail": stderr[-1500:]}
    if declared_command:
        result["command"] = smoke_command
        return result
    # v14c: exercise the REAL user entry point(s) — a regression here must fail the smoke even when
    # smoke_check.py is green, because this is what the user actually runs.
    for entry in _USER_ENTRY_FILES:
        if not os.path.isfile(os.path.join(repo_dir, entry)):
            continue
        cli_kwargs = dict(
            command=[sys.executable, entry, _CLI_SMOKE_QUERY],
            repo_dir=repo_dir,
            timeout=timeout,
            environment=run_env,
            executor=formal_executor,
            read_only_mounts=read_only_mounts,
        )
        if image_workspace:
            cli_kwargs["image_workspace"] = image_workspace
        cli_returncode, _cli_stdout, cli_stderr, cli_error = (
            _execute_smoke_command(**cli_kwargs)
        )
        if cli_error:
            result["ok"] = False
            result["cli_ok"] = False
            result["error"] = f"user entry {entry} failed: {cli_error}"
            break
        if cli_returncode != 0:
            result["ok"] = False
            result["cli_ok"] = False
            result["cli_returncode"] = cli_returncode
            result["cli_stderr_tail"] = cli_stderr[-1500:]
            # surface the CLI break as the PRIMARY error so the brief/localizer target it
            result["error"] = f"user entry {entry} failed: {_last_error_line(cli_stderr) or ('exit ' + str(cli_returncode))}"
            break
        result["cli_ok"] = True
    return result


def declared_public_test_command(world: Any) -> Optional[List[str]]:
    """The substrate's OWN public test command, if it declares one.

    These are the tests that ship inside the agent-visible starter repo (the
    manifest lists hidden tests separately under private_evaluator_only), so
    running them leaks nothing: an agent can already read the files. Exposing
    them is what lets the organization verify its own fix, which is the whole
    difference between patching blindly and converging on a correct fix.
    """
    verified = getattr(world, "__dict__", {}).get(
        "_cooperbench_verified_public_test_command"
    )
    if isinstance(verified, list) and verified and all(
        isinstance(item, str) and item for item in verified
    ):
        return list(verified)

    from environments.org_env.product.substrates.eval_assets import oss_eval_assets

    assets = oss_eval_assets(world) or {}
    manifest = assets.get("manifest") or {}
    command = (manifest.get("public_tests") or {}).get("command")
    return list(command) if isinstance(command, list) and command else None


def declared_public_test_files(world: Any) -> List[str]:
    """Repo paths the declared public test command actually runs.

    Only these files can produce evidence: a test written anywhere else is never
    executed, so the organization would author it and learn nothing. Derived
    from the command itself rather than a separate manifest key, so the two can
    never disagree about what the suite covers.

    Options and the runner invocation are skipped; what remains are path-shaped
    arguments. Substrate-independent.
    """
    command = declared_public_test_command(world) or []
    artifact_paths = {
        str(getattr(artifact, "linked_file_path", "") or "").replace("\\", "/")
        for artifact in (getattr(world, "product_artifacts", {}) or {}).values()
        if getattr(artifact, "linked_file_path", None)
    }
    files: List[str] = []
    for token in command:
        text = str(token)
        normalized = text.replace("\\", "/")
        if text.startswith("-"):
            continue
        # Bare runner names (``sh``, ``python``) are not repo paths, but a bare
        # filename such as ProgramBench's compile.sh is when it is an actual
        # agent-visible artifact.
        if "/" not in normalized and normalized not in artifact_paths:
            continue
        files.append(normalized)
    return files


def public_test_target_files(world: Any) -> List[str]:
    """Concrete repo files the agent can actually edit to add a public test.

    declared_public_test_files reports what the command names, and for most
    packs that is a DIRECTORY — `pytest tests/public` runs a tree, not a file.
    Nothing in the world is an artifact called "tests/public", so a candidate
    keyed on that path found no artifact and was dropped: of thirteen packs only
    one names a file directly, so writing a regression test was unreachable in
    twelve of them. That is the discipline the whole red-then-green contract
    depends on, silently unavailable wherever the command happens to be written
    against a directory.

    Directory tokens expand to the test files that exist under them; file tokens
    pass through. Only paths that exist as artifacts are returned, because a
    path the agent cannot open is not a place it can write a test.
    """
    declared = declared_public_test_files(world)
    if not declared:
        return []
    artifact_paths = {
        path
        for artifact in (getattr(world, "product_artifacts", {}) or {}).values()
        if (path := str(getattr(artifact, "linked_file_path", "") or ""))
    }
    out: List[str] = []
    for token in declared:
        if token in artifact_paths:
            out.append(token)
            continue
        prefix = token.rstrip("/") + "/"
        out.extend(sorted(p for p in artifact_paths if p.startswith(prefix)))
    # dict.fromkeys: a file named both directly and via its directory is one
    # place to write, not two.
    return list(dict.fromkeys(out))


def declared_public_test_dependencies(world: Any) -> List[str]:
    """Extra modules the public test command needs (pytest plugins, fixtures).

    Separate from runtime_dependencies (what the PRODUCT imports) and from
    evaluator_dependencies (what the hidden oracles need): a test plugin is
    required only to RUN the suite. Declared per substrate so nothing is
    hardcoded.
    """
    from environments.org_env.product.substrates.eval_assets import oss_eval_assets

    assets = oss_eval_assets(world) or {}
    manifest = assets.get("manifest") or {}
    deps = (manifest.get("public_tests") or {}).get("dependencies")
    return [str(d) for d in deps] if isinstance(deps, list) else []


def public_test_sandbox_readiness(world: Any, repo_dir: str, executor: Any) -> Optional[str]:
    """Verify the declared test dependencies are importable WHERE THE SUITE RUNS.

    The formal preflight checks dependencies with importlib on the HOST, but the
    public suite executes inside the sandbox: a host that has pytest_asyncio
    proves nothing about the container. Checking in the wrong place is why a
    missing test plugin surfaced only as an unexplained red suite.
    """
    deps = declared_public_test_dependencies(world)
    if not deps or executor is None:
        return None
    # A requirement names a distribution; `find_spec` wants a module. They differ
    # for almost every hyphenated package -- typing-extensions imports as
    # typing_extensions, pytest-xdist as xdist -- so asking find_spec for the
    # distribution name reports installed packages as missing. cattrs declares
    # four such, two of which were present, and its agent-visible suite was
    # refused fifty-three times across a 336-tick run: the organization could
    # never once see its own tests.
    #
    # A marker is honoured too. cattrs asks for exceptiongroup only below Python
    # 3.11; demanding it on 3.12 blocks a suite over a dependency the suite does
    # not want.
    requirements = [str(d).strip() for d in deps if str(d).strip()]
    if not requirements:
        return None
    probe = (
        "import sys\n"
        "from importlib.metadata import distribution, PackageNotFoundError\n"
        "try:\n"
        "    from packaging.requirements import Requirement\n"
        "except Exception:\n"
        "    Requirement = None\n"
        f"missing = []\n"
        f"for raw in {requirements!r}:\n"
        "    name, marker = raw, None\n"
        "    if Requirement is not None:\n"
        "        try:\n"
        "            req = Requirement(raw)\n"
        "            name, marker = req.name, req.marker\n"
        "        except Exception:\n"
        "            pass\n"
        "    else:\n"
        "        name = raw.split(';')[0].split('[')[0]\n"
        "        for sep in ('>=', '<=', '==', '!=', '~=', '>', '<'):\n"
        "            name = name.split(sep)[0]\n"
        "        name = name.strip()\n"
        "    if marker is not None and not marker.evaluate():\n"
        "        continue\n"
        "    try:\n"
        "        distribution(name)\n"
        "    except PackageNotFoundError:\n"
        "        import importlib.util as u\n"
        "        if u.find_spec(name.replace('-', '_')) is None:\n"
        "            missing.append(name)\n"
        "    except Exception:\n"
        "        pass\n"
        "sys.stdout.write(','.join(missing))\n"
    )
    try:
        outcome = executor.run(
            root=Path(repo_dir), argv=("python", "-c", probe), timeout_seconds=60.0
        )
    except Exception as exc:
        return f"public_test_dependency_probe_failed:{exc!r}"
    missing = (outcome.stdout_tail or "").strip()
    if outcome.exit_code == 0 and missing:
        return (
            "public_test_dependencies_missing_in_sandbox:" + missing
            + " (declare them under public_tests.dependencies and install them in"
            " the evaluator image; the host having them proves nothing)"
        )
    return None


_PROGRAMBENCH_DEFINITION_EXACT_PATHS = ("eval/eval_stub.py",)
_PROGRAMBENCH_DEFINITION_PATTERNS = (r"^eval/eval_[0-9]+\.py$",)
_PROGRAMBENCH_MAX_DEFINITION_FILES = 32
_PROGRAMBENCH_PUBLIC_PREVIEW_BYTES = 2048
_PROGRAMBENCH_PUBLIC_EFFECT_ENTRIES = 32
# Go's standard logger prefixes every line with its own wall clock, and the
# reference and candidate run in separate sandboxes, so any behaviour that
# reaches `log` mismatches on the clock alone and can never be repaired by the
# organization. The normalization for that lives in the probe comparison policy.
#
# This used to be a table keyed by pack id, which made the correct policy an
# opt-in that a pack could lose by being renamed. Deriving quota variants of the
# audited Go pack did exactly that: the derived ids missed the table, silently
# fell back to exact_v1, and one self-authored case (`stdin='invalid json'`,
# which takes the logged error path) became permanently unsatisfiable. That
# single case blocked the transactional merge for the rest of the run, so the
# mainline stayed at the starter and the arm could not be scored at all.
#
# Keying on the language makes the policy a property of the reference's runtime
# rather than of a mutable name, so a new Go pack is correct on arrival instead
# of correct only if someone remembers to add a row. `yj` was already in the
# repository with the same exposure and no entry.
_PROGRAMBENCH_COMPARISON_POLICY_BY_LANGUAGE = {
    "go": "go_standard_log_clock_v1",
    "rs": "exact_v1",
    "c": "exact_v1",
    "cpp": "exact_v1",
}
# Reserved for an instance whose reference contradicts its language default.
# Empty is the correct state; an entry here is a claim about one binary.
_PROGRAMBENCH_COMPARISON_POLICY_BY_INSTANCE: Dict[str, str] = {}


def _programbench_comparison_policy(manifest: Any) -> str:
    """Resolve the public differential comparison policy, or fail closed.

    An unrecognised language raises rather than defaulting: a wrong policy is
    invisible while it runs and only shows up as an unfixable mismatch dozens
    of ticks later.
    """

    if not isinstance(manifest, dict):
        raise ValueError("programbench_comparison_policy_manifest_invalid")
    programbench = manifest.get("programbench")
    task = programbench.get("task") if isinstance(programbench, dict) else None
    if not isinstance(task, dict):
        raise ValueError("programbench_comparison_policy_task_missing")
    instance_id = str(task.get("instance_id") or "")
    override = _PROGRAMBENCH_COMPARISON_POLICY_BY_INSTANCE.get(instance_id)
    if override is not None:
        return override
    language = str(task.get("language") or "").strip().casefold()
    policy = _PROGRAMBENCH_COMPARISON_POLICY_BY_LANGUAGE.get(language)
    if policy is None:
        raise ValueError(
            "programbench_comparison_policy_language_unsupported:" + (language or "?")
        )
    return policy


def _programbench_public_probe_assets(world: Any) -> Optional[Dict[str, Any]]:
    """Select ProgramBench before the legacy public-test/cache path.

    A hidden ``programbench.json`` selects this path even when the public
    manifest is malformed. That is intentional: a broken formal declaration
    must fail closed instead of falling back to host execution.
    """
    from types import SimpleNamespace

    from environments.org_env.product.substrates.eval_assets import oss_eval_assets
    from society_core.programbench_evaluation import is_programbench_spec

    assets = oss_eval_assets(world) or {}
    if not isinstance(assets, dict):
        return None
    spec = SimpleNamespace(
        manifest=assets.get("manifest") or {},
        hidden_tests_dir=assets.get("hidden_tests_dir") or "",
    )
    return assets if is_programbench_spec(spec) else None


def _programbench_probe_contract(manifest: Any):
    """Return the strict public-probe contract, or raise a stable error."""
    from society_core.programbench_probes import (
        PUBLIC_PROBE_ENV_ALLOWLIST,
        PUBLIC_PROBE_RUNNER,
        PUBLIC_PROBE_SCHEMA_VERSION,
        ProbeLimits,
    )

    if not isinstance(manifest, dict):
        raise ValueError("programbench_manifest_invalid")
    public = manifest.get("public_probes")
    if not isinstance(public, dict):
        raise ValueError("programbench_public_probe_contract_missing")
    required = {
        "schema_version",
        "schema_path",
        "definition_command",
        "definition_surface",
        "limits",
        "compile",
        "executor",
        "runner",
        "env_allowlist",
        "definition_command_boundary",
        "execution_backend",
        "status",
    }
    # `exploration_case_quota` is the cumulative floor; `limits.max_cases` is
    # the per-receipt technical ceiling. They were one field, so raising the
    # floor also raised the ceiling and vice versa. The floor cannot live inside
    # `limits`, whose key set must stay exactly `ProbeLimits`, so it is a
    # sibling key.
    #
    # It is optional rather than required because this key-set check is an
    # equality: making it mandatory would refuse every pack built before the
    # split. Those packs declare the floor in `limits.max_cases` and the profile
    # falls back to it, so they attach with their identity unchanged. A pack that
    # does declare it would be refused outright without this entry, which is why
    # the reader and the manifest have to land together.
    optional = {"exploration_case_quota"}
    if set(public) - optional != required:
        raise ValueError("programbench_public_probe_contract_keys_invalid")
    if "exploration_case_quota" in public:
        floor = public["exploration_case_quota"]
        if isinstance(floor, bool) or not isinstance(floor, int) or floor < 1:
            raise ValueError("programbench_public_probe_exploration_quota_invalid")
    if public.get("schema_version") != PUBLIC_PROBE_SCHEMA_VERSION:
        raise ValueError("programbench_public_probe_schema_invalid")
    if public.get("runner") != PUBLIC_PROBE_RUNNER:
        raise ValueError("programbench_public_probe_runner_invalid")
    if public.get("schema_path") != "tests/public/schema.json":
        raise ValueError("programbench_public_probe_schema_path_invalid")
    if public.get("definition_command") != ["python3", "{definition_path}"]:
        raise ValueError("programbench_public_probe_definition_command_invalid")
    if public.get("env_allowlist") != sorted(PUBLIC_PROBE_ENV_ALLOWLIST):
        raise ValueError("programbench_public_probe_env_allowlist_invalid")
    if public.get("definition_command_boundary") != "candidate_sandbox_only":
        raise ValueError("programbench_public_probe_definition_boundary_invalid")
    if public.get("execution_backend") != (
        "separate_pinned_sandbox_executors_required"
    ):
        raise ValueError("programbench_public_probe_execution_boundary_invalid")

    surface = public.get("definition_surface")
    if not isinstance(surface, dict) or set(surface) != {
        "exact_paths",
        "path_patterns",
        "max_definitions",
    }:
        raise ValueError("programbench_public_probe_surface_invalid")
    if surface.get("exact_paths") != list(_PROGRAMBENCH_DEFINITION_EXACT_PATHS):
        raise ValueError("programbench_public_probe_exact_paths_invalid")
    if surface.get("path_patterns") != list(_PROGRAMBENCH_DEFINITION_PATTERNS):
        raise ValueError("programbench_public_probe_patterns_invalid")
    maximum = surface.get("max_definitions")
    if isinstance(maximum, bool) or not isinstance(maximum, int) or not 1 <= maximum <= 32:
        raise ValueError("programbench_public_probe_definition_limit_invalid")

    raw_limits = public.get("limits")
    expected_limit_fields = set(ProbeLimits.__dataclass_fields__)
    if not isinstance(raw_limits, dict) or set(raw_limits) != expected_limit_fields:
        raise ValueError("programbench_public_probe_limits_invalid")
    try:
        limits = ProbeLimits(**raw_limits)
    except (TypeError, ValueError) as error:
        raise ValueError("programbench_public_probe_limits_invalid") from error

    compile_contract = public.get("compile")
    if not isinstance(compile_contract, dict) or set(compile_contract) != {
        "command",
        "output_path",
    }:
        raise ValueError("programbench_public_probe_compile_contract_invalid")
    if compile_contract.get("command") != ["sh", "compile.sh"]:
        raise ValueError("programbench_public_probe_compile_command_invalid")
    if compile_contract.get("output_path") != "executable":
        raise ValueError("programbench_public_probe_output_path_invalid")

    executor_contract = public.get("executor")
    if not isinstance(executor_contract, dict) or set(executor_contract) != {
        "max_output_bytes",
        "output_tail_bytes",
    }:
        raise ValueError("programbench_public_probe_executor_contract_invalid")
    maximum_output = executor_contract.get("max_output_bytes")
    tail = executor_contract.get("output_tail_bytes")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in (maximum_output, tail)
    ) or tail > maximum_output:
        raise ValueError("programbench_public_probe_executor_limits_invalid")
    return public, limits


def _programbench_probe_executor(manifest: Dict[str, Any], public: Dict[str, Any]):
    """Build the digest-pinned, network-disabled ProgramBench command boundary."""
    if str(os.environ.get("ORG_OSS_MODE") or "").lower() != "formal":
        raise RuntimeError("programbench_public_probe_formal_executor_required")
    backend = str(os.environ.get("ORG_EVALUATOR_BACKEND") or "")
    if backend not in {"docker", "apptainer"}:
        raise RuntimeError("programbench_public_probe_container_backend_required")
    programbench = manifest.get("programbench")
    cleanroom = programbench.get("cleanroom") if isinstance(programbench, dict) else None
    image = cleanroom.get("image") if isinstance(cleanroom, dict) else None
    resources = cleanroom.get("resources") if isinstance(cleanroom, dict) else None
    if not isinstance(image, dict) or not isinstance(resources, dict):
        raise ValueError("programbench_public_probe_cleanroom_invalid")
    reference = image.get("reference")
    platform = image.get("platform")
    memory = resources.get("memory_limit_mb")
    cpus = resources.get("docker_cpus")
    pids = resources.get("pids_limit")
    executor_limits = public["executor"]
    if not isinstance(reference, str) or not isinstance(platform, str):
        raise ValueError("programbench_public_probe_image_invalid")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or value <= 0
        for value in (memory, cpus, pids)
    ):
        raise ValueError("programbench_public_probe_resources_invalid")

    from society_core.execution import ExecutionPolicy, build_command_executor

    return build_command_executor(
        ExecutionPolicy(
            trust_level="untrusted",
            backend=backend,
            container_image=reference,
            container_platform=platform,
            network_enabled=False,
            memory_limit_mb=int(memory),
            cpu_limit=float(cpus),
            pids_limit=int(pids),
            max_output_bytes=int(executor_limits["max_output_bytes"]),
            output_tail_bytes=int(executor_limits["output_tail_bytes"]),
        )
    )


def _programbench_active_candidate_overrides(
    world: Any,
) -> Dict[str, str] | None:
    """Resolve the adapted view only when the opt-in profile is active."""

    try:
        from environments.org_env.programbench import programbench_profile_active

        if programbench_profile_active(world):
            return programbench_integration_candidate_overrides(world)
    except ImportError:
        pass
    return None


def _programbench_definition_text(
    artifact: Any,
    overrides: Dict[str, str] | None,
) -> str:
    artifact_id = str(getattr(artifact, "artifact_id", "") or "")
    if overrides is None:
        return _file_text(artifact, prefer_mainline=False)
    if artifact_id in overrides:
        return overrides[artifact_id]
    content = getattr(artifact, "mainline_content", "")
    if not isinstance(content, str):
        raise ValueError(
            "programbench_candidate_view_mainline_text_invalid:" + artifact_id
        )
    return content


def _programbench_definition_artifacts(
    world: Any,
    *,
    max_definitions: int,
    candidate_overrides: Dict[str, str] | None = None,
) -> List[tuple[str, Any]]:
    """Enumerate changed probe definitions present in the selected view."""
    patterns = [re.compile(item) for item in _PROGRAMBENCH_DEFINITION_PATTERNS]
    claimed_profile = "programbench_profile_state" in getattr(
        world, "__dict__", {}
    )
    selected: Dict[str, Any] = {}
    for artifact in (getattr(world, "product_artifacts", {}) or {}).values():
        raw_path = getattr(artifact, "linked_file_path", None)
        if not raw_path:
            continue
        if claimed_profile:
            from environments.org_env.programbench import (
                programbench_frozen_public_probe_path_status,
            )

            status = programbench_frozen_public_probe_path_status(
                world, raw_path
            )
            if status.startswith("invalid"):
                raise ValueError(
                    "programbench_public_probe_path_alias_invalid"
                )
            if status != "trusted":
                continue
            path = normalize_repo_relative_path(raw_path)
        else:
            try:
                path = normalize_repo_relative_path(raw_path)
            except InvalidRepoPath:
                continue
        if not claimed_profile and (
            path not in _PROGRAMBENCH_DEFINITION_EXACT_PATHS
            and not any(pattern.fullmatch(path) for pattern in patterns)
        ):
            continue
        revision = getattr(artifact, "revision", 0)
        changed = (
            isinstance(revision, int)
            and not isinstance(revision, bool)
            and revision > 0
            and bool(
                getattr(artifact, "patch_history_ids", None)
                or getattr(artifact, "linked_action_ids", None)
            )
        )
        if not changed:
            continue
        artifact_id = str(getattr(artifact, "artifact_id", "") or "")
        if (
            candidate_overrides is not None
            and _is_unmerged_new_file(artifact)
            and artifact_id not in candidate_overrides
        ):
            # A working-tree definition on a collateral/unrouted branch is not
            # present in full-mainline-plus-integration-candidate.
            continue
        if path in selected:
            raise ValueError("programbench_public_probe_definition_path_duplicate")
        selected[path] = artifact
    if len(selected) > max_definitions:
        raise ValueError("programbench_public_probe_definition_count_exceeded")
    return sorted(selected.items())


def programbench_probe_corpus_digest(world: Any) -> str:
    """Hash only the executable public-probe definition corpus.

    Reference-only observations do not depend on candidate source, build files,
    documentation, or repository metadata.  Keying their cache on the global
    working-tree hash therefore reruns an unchanged corpus after every unrelated
    edit.  This digest intentionally includes only normalized definition paths
    and their current bytes; a no-op revision does not invalidate it.
    """

    maximum = _PROGRAMBENCH_MAX_DEFINITION_FILES
    claimed_profile = "programbench_profile_state" in getattr(
        world, "__dict__", {}
    )
    if claimed_profile:
        # v5 freezes the public definition surface at t0.  In particular, a
        # final-primary decision at t336 must not open evaluator assets merely
        # to discover the public file-count cap.
        from environments.org_env.programbench import (
            programbench_frozen_public_probe_contract,
        )

        frozen = programbench_frozen_public_probe_contract(world)
        maximum = int(frozen["definition_surface"]["max_definitions"])
    else:
        assets = _programbench_public_probe_assets(world)
        if assets is not None:
            public, _limits = _programbench_probe_contract(
                assets.get("manifest") or {}
            )
            maximum = int(public["definition_surface"]["max_definitions"])
        manifest = getattr(world, "programbench_public_manifest", None)
        if assets is None and isinstance(manifest, dict) and isinstance(
            manifest.get("public_probes"), dict
        ):
            public, _limits = _programbench_probe_contract(manifest)
            maximum = int(public["definition_surface"]["max_definitions"])
    # Probe identity is an independent public surface.  Hash only the trusted
    # manifest-declared definition artifacts and their live full text; do not
    # traverse/validate every implementation artifact merely to decide whether
    # an unchanged reference corpus is current.  Delivery still routes these
    # files on the one integration branch, and differential candidate export
    # below continues to use the strict full candidate view.
    candidate_overrides = (
        None
        if claimed_profile
        else _programbench_active_candidate_overrides(world)
    )
    definitions = _programbench_definition_artifacts(
        world,
        max_definitions=maximum,
        candidate_overrides=candidate_overrides,
    )
    payload = [
        {
            "path": path,
            "size": len(
                _programbench_definition_text(artifact, candidate_overrides).encode()
            ),
            "sha256": hashlib.sha256(
                _programbench_definition_text(artifact, candidate_overrides).encode()
            ).hexdigest(),
        }
        for path, artifact in definitions
    ]
    encoded = json.dumps(
        {"schema_version": "programbench_probe_corpus_digest_v1", "files": payload},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _programbench_export_definition_surface(
    definitions: List[tuple[str, Any]],
    destination: Path,
    *,
    candidate_overrides: Dict[str, str] | None = None,
) -> None:
    """Stage only selected probe definitions, never the candidate workspace."""

    destination.mkdir(parents=False, exist_ok=False)
    root = destination.resolve()
    for source, artifact in definitions:
        _normalized, target = _safe_export_path(root, source)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            _programbench_definition_text(artifact, candidate_overrides),
            encoding="utf-8",
        )


def preflight_programbench_probe_definition_patch(
    world: Any,
    artifact: Any,
    new_content: str,
) -> None:
    """Validate proposed probe bytes in the definition-only sandbox.

    This gate deliberately uses only the attachment-time frozen path contract,
    agent-visible authoring limits and the pinned formal command executor.  It
    never opens evaluator assets and never stages, mounts, compiles or executes
    a candidate, reference implementation or hidden test.
    """

    if "programbench_profile_state" not in getattr(world, "__dict__", {}):
        return
    from environments.org_env.backend.repo.workflow import (
        programbench_public_probe_artifact,
    )
    from environments.org_env.programbench import (
        programbench_frozen_public_probe_contract,
    )
    from society_core.programbench_probes import (
        PUBLIC_PROBE_ENV_ALLOWLIST,
        ProbeLimits,
        capture_probe_document_in_sandbox,
    )

    if not programbench_public_probe_artifact(world, artifact):
        return
    if not isinstance(new_content, str):
        raise ValueError("programbench_public_probe_patch_content_invalid")
    frozen = programbench_frozen_public_probe_contract(world)
    state = world.__dict__.get("programbench_profile_state") or {}
    product = getattr(world, "product", None)
    authoring = (getattr(product, "substrate_meta", {}) or {}).get(
        "public_probe_authoring"
    )
    if not isinstance(authoring, dict):
        raise ValueError("programbench_public_probe_authoring_contract_invalid")
    if (
        authoring.get("schema_version") != frozen["probe_schema_version"]
        or authoring.get("definition_surface") != frozen["definition_surface"]
        or authoring.get("case_keys") != ["argv", "stdin", "input_files", "env"]
        or authoring.get("input_file_keys") != ["path", "content_base64"]
        or authoring.get("env_allowlist") != sorted(PUBLIC_PROBE_ENV_ALLOWLIST)
    ):
        raise ValueError("programbench_public_probe_authoring_contract_invalid")
    raw_limits = authoring.get("limits")
    if not isinstance(raw_limits, dict):
        raise ValueError("programbench_public_probe_authoring_contract_invalid")
    expected = set(ProbeLimits.__dataclass_fields__)
    if set(raw_limits) != expected:
        raise ValueError("programbench_public_probe_authoring_contract_invalid")
    try:
        limits = ProbeLimits(**raw_limits)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "programbench_public_probe_authoring_contract_invalid"
        ) from error
    # The authoring surface an editor is shown must advertise the same
    # per-receipt ceiling the preflight enforces, or a definition is refused
    # after it has been written. This used to compare against the exploration
    # quota because the two numbers were the same field; they are now separate,
    # and it is the ceiling that governs one document.
    ceiling = state.get("public_probe_receipt_case_ceiling")
    if (
        isinstance(ceiling, bool)
        or not isinstance(ceiling, int)
        or ceiling != limits.max_cases
    ):
        raise ValueError("programbench_public_probe_quota_state_invalid")
    source = normalize_repo_relative_path(
        getattr(artifact, "linked_file_path", None)
    )
    definitions: list[tuple[str, Any]] = []
    for candidate in (getattr(world, "product_artifacts", {}) or {}).values():
        if not programbench_public_probe_artifact(world, candidate):
            continue
        candidate_source = normalize_repo_relative_path(
            getattr(candidate, "linked_file_path", None)
        )
        authored = bool(
            candidate is artifact
            or (
                int(getattr(candidate, "created_at_tick", 0) or 0) > 0
                and str(getattr(candidate, "content", "") or "").strip()
            )
        )
        if authored:
            definitions.append((candidate_source, candidate))
    if len({path for path, _candidate in definitions}) != len(definitions):
        raise ValueError("programbench_public_probe_definition_path_duplicate")
    maximum = int(frozen["definition_surface"]["max_definitions"])
    if len(definitions) > maximum:
        raise ValueError("programbench_public_probe_definition_count_exceeded")
    scratch_parent = Path(_product_smoke_root())
    executor = _formal_product_executor()
    if executor is None:
        raise ValueError("programbench_public_probe_formal_executor_required")
    with tempfile.TemporaryDirectory(
        prefix="programbench_probe_preflight_", dir=scratch_parent
    ) as raw_scratch:
        definition_root = Path(raw_scratch) / "definition_surface"
        definition_root.mkdir(parents=False, exist_ok=False)
        for candidate_source, candidate in definitions:
            _normalized, target = _safe_export_path(
                definition_root.resolve(), candidate_source
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                new_content
                if candidate is artifact
                else str(getattr(candidate, "content", "") or ""),
                encoding="utf-8",
            )
        case_count = 0
        for candidate_source, _candidate in definitions:
            payload = capture_probe_document_in_sandbox(
                command_executor=executor,
                candidate_repo=definition_root,
                script_path=candidate_source,
                limits=limits,
            )
            document = json.loads(payload.decode("utf-8"))
            case_count += len(document["cases"])
            if case_count > ceiling:
                raise ValueError("probe_case_limit_exceeded")
        # A definition no longer has to hold exactly the exploration quota. The
        # quota is met by accumulating distinct observations across probe runs,
        # so a small targeted batch is a legitimate corpus and refusing it was
        # what made the corpus un-editable once it had been observed.
        if case_count < 1:
            raise ValueError("programbench_public_probe_zero_cases")


def programbench_static_probe_definition_failure(
    world: Any,
) -> str | None:
    """Return a proven static definition violation without executing scripts.

    This intentionally recognizes only literal ``json.dump(s)`` payloads.  A
    dynamic definition is unknown, not invalid, and remains the definition
    sandbox's responsibility.  The witness is useful when infrastructure is
    unavailable but current bytes plainly exceed the frozen aggregate quota.
    """

    if "programbench_profile_state" not in getattr(world, "__dict__", {}):
        return None
    from environments.org_env.backend.repo.workflow import (
        programbench_public_probe_artifact,
    )

    ceiling = int(
        (world.__dict__.get("programbench_profile_state") or {}).get(
            "public_probe_receipt_case_ceiling"
        )
        or 0
    )
    total = 0
    live_definitions = 0
    resolved_definitions = 0
    for artifact in (getattr(world, "product_artifacts", {}) or {}).values():
        if not programbench_public_probe_artifact(world, artifact):
            continue
        content = str(getattr(artifact, "content", "") or "")
        if not content.strip() or int(getattr(artifact, "created_at_tick", 0) or 0) <= 0:
            continue
        live_definitions += 1
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return "probe_definition_python_syntax_invalid"
        literals: dict[str, Any] = {}
        mutated_names = {
            node.func.value.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
        }
        mutated_names.update(
            node.target.id
            for node in ast.walk(tree)
            if isinstance(node, ast.AugAssign)
            and isinstance(node.target, ast.Name)
        )

        def resolve_literal(node: ast.AST) -> Any:
            if isinstance(node, ast.Name):
                if node.id in mutated_names or node.id not in literals:
                    raise ValueError("unbound_name")
                return literals[node.id]
            if isinstance(node, ast.Constant):
                return node.value
            if isinstance(node, (ast.List, ast.Tuple)):
                return [resolve_literal(item) for item in node.elts]
            if isinstance(node, ast.Dict):
                return {
                    resolve_literal(key): resolve_literal(value)
                    for key, value in zip(node.keys, node.values)
                }
            raise ValueError("not_static_literal")

        for statement in tree.body:
            if (
                not isinstance(statement, ast.Assign)
                or len(statement.targets) != 1
                or not isinstance(statement.targets[0], ast.Name)
            ):
                continue
            try:
                value = resolve_literal(statement.value)
            except (ValueError, TypeError):
                continue
            literals[statement.targets[0].id] = value
        payload = None
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not node.args:
                continue
            func = node.func
            name = (
                func.attr
                if isinstance(func, ast.Attribute)
                else func.id if isinstance(func, ast.Name) else ""
            )
            if name not in {"dump", "dumps"}:
                continue
            try:
                argument = node.args[0]
                value = resolve_literal(argument)
            except (ValueError, TypeError):
                continue
            if isinstance(value, dict):
                payload = value
        if payload is None:
            continue
        resolved_definitions += 1
        if set(payload) != {"schema_version", "cases"}:
            return "probe_json_keys_invalid"
        if payload.get("schema_version") != "programbench_public_probe_cases_v1":
            return "probe_schema_unsupported"
        cases = payload.get("cases")
        if not isinstance(cases, list):
            return "probe_cases_invalid"
        for case in cases:
            if not isinstance(case, dict) or set(case) != {
                "argv", "stdin", "input_files", "env"
            }:
                return "probe_case_keys_invalid"
        total += len(cases)
        if ceiling and total > ceiling:
            return "probe_case_limit_exceeded"
    # The old aggregate-equals-quota witness is gone with the exact quota. A
    # statically resolvable corpus that is simply smaller than the exploration
    # floor is not a defective definition: the floor is met cumulatively, and
    # calling it broken is what forced the whole corpus into one document.
    if live_definitions and resolved_definitions == live_definitions and not total:
        return "programbench_public_probe_zero_cases"
    return None


def programbench_exported_repo_digest(root: Path | str) -> str:
    """Return the public content identity of an exported ProgramBench tree.

    Public differential evidence uses this exact path/content SHA-1.  Sealed
    evaluation separately uses its 64-character workspace-execution identity;
    callers must not compare those two intentionally different digest domains.
    """

    resolved = Path(root).resolve(strict=True)
    rows: list[tuple[str, bytes]] = []
    for item in resolved.rglob("*"):
        relative = item.relative_to(resolved).as_posix()
        junction = getattr(item, "is_junction", None)
        if item.is_symlink() or (callable(junction) and junction()):
            raise ValueError("programbench_candidate_compile_view_link_invalid")
        if item.is_dir():
            continue
        if not item.is_file():
            raise ValueError("programbench_candidate_compile_view_entry_invalid")
        canonical = normalize_repo_relative_path(relative)
        if canonical != relative:
            raise ValueError("programbench_candidate_compile_view_path_invalid")
        rows.append((canonical, item.read_bytes()))
    digest = hashlib.sha1()
    for path, content in sorted(rows):
        digest.update(path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(content)
        digest.update(b"\0")
    return digest.hexdigest()


def _programbench_failure(
    code: str,
    *,
    sources: List[str] | None = None,
    probe_mode: str = "differential",
) -> Dict[str, Any]:
    stable = re.sub(r"[^A-Za-z0-9_.:-]+", "_", str(code))[:240]
    return {
        "ok": False,
        "available": True,
        "returncode": None,
        "failed_tests": [],
        "summary": "ProgramBench public probe failed closed",
        "error": stable,
        "collected": 0,
        "programbench_public_probes": {
            "schema_version": "programbench_public_probe_materialization_v2",
            "mode": probe_mode,
            "comparison_policy": "exact_v1",
            "status": "failed",
            "failure": stable,
            "definition_sources": list(sources or []),
            "cases": [],
        },
    }


def _programbench_compile(executor: Any, root: Path, command: List[str], timeout: int) -> str | None:
    try:
        outcome = executor.run(
            root=root,
            argv=tuple(command),
            timeout_seconds=float(timeout),
        )
    except Exception:
        return "executor_exception"
    if getattr(outcome, "status", None) != "passed":
        return "status_" + str(getattr(outcome, "status", "invalid"))
    if getattr(outcome, "exit_code", None) != 0:
        return "nonzero_exit"
    return None


def _programbench_runtime_root(
    compile_root: Path, runtime_root: Path, output_path: str
) -> Path:
    """Stage only an attested compile output for one public-probe role.

    The compile tree can contain the private reference payload, source files,
    and evaluator metadata.  None of that belongs in the agent-controlled argv
    namespace.  Resolve the declared output inside the compile root, copy its
    bytes into a fresh one-file runtime root, and make it execute-only where the
    host filesystem has POSIX permission semantics.
    """
    try:
        normalized = normalize_repo_relative_path(output_path)
    except InvalidRepoPath as error:
        raise ValueError("programbench_probe_runtime_output_path_invalid") from error
    try:
        resolved_compile_root = compile_root.resolve(strict=True)
        source = resolved_compile_root.joinpath(*normalized.split("/"))
        cursor = resolved_compile_root
        for part in normalized.split("/"):
            cursor = cursor / part
            junction = getattr(cursor, "is_junction", None)
            if callable(junction) and junction():
                raise ValueError("programbench_probe_runtime_output_junction")
        resolved_source = source.resolve(strict=True)
        try:
            resolved_source.relative_to(resolved_compile_root)
        except ValueError as error:
            raise ValueError("programbench_probe_runtime_output_escape") from error
    except ValueError:
        raise
    except (OSError, RuntimeError) as error:
        raise ValueError("programbench_probe_runtime_output_missing") from error

    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(resolved_source, flags)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("programbench_probe_runtime_output_not_regular")
        runtime_root.mkdir(parents=False, exist_ok=False)
        target = runtime_root / "executable"
        with os.fdopen(descriptor, "rb", closefd=False) as source_handle:
            with target.open("xb") as target_handle:
                shutil.copyfileobj(source_handle, target_handle)
                target_handle.flush()
                os.fsync(target_handle.fileno())
    finally:
        os.close(descriptor)
    if os.name != "nt":
        target.chmod(0o111)
    target_metadata = target.lstat()
    if (
        not stat.S_ISREG(target_metadata.st_mode)
        or target_metadata.st_nlink != 1
        or sorted(item.name for item in runtime_root.iterdir()) != ["executable"]
    ):
        raise ValueError("programbench_probe_runtime_root_invalid")
    return runtime_root


def _programbench_redact_text(value: Any, private_roots: List[Path]) -> str:
    text = str(value or "")
    for root in private_roots:
        forms = {str(root), root.as_posix(), str(root).replace("\\", "/")}
        for form in sorted(forms, key=len, reverse=True):
            if form:
                text = text.replace(form, "[redacted-path]")
    return text[:_PROGRAMBENCH_PUBLIC_PREVIEW_BYTES]


def _programbench_public_side(value: Any, *, private_roots: List[Path]) -> Any:
    if value is None:
        return None
    if value.get("status") == "skipped":
        return {
            "status": "skipped",
            "reason": _programbench_redact_text(value.get("reason"), private_roots),
        }
    if value.get("status") == "infra_error":
        return {
            "status": "infra_error",
            "reason": _programbench_redact_text(value.get("reason"), private_roots),
        }
    outputs = {}
    for name in ("stdout", "stderr"):
        item = value[name]
        raw_text = str(item["text"])
        outputs[name] = {
            "bytes": item["bytes"],
            "sha256": item["sha256"],
            "truncated": bool(item["truncated"]),
            "text_preview": _programbench_redact_text(raw_text, private_roots),
            "preview_truncated": len(raw_text) > _PROGRAMBENCH_PUBLIC_PREVIEW_BYTES,
        }
    effects = value["filesystem_effects"]
    flattened = [
        (category, item)
        for category in ("created", "modified", "deleted")
        for item in effects[category]
    ]

    def public_effect(item: Dict[str, Any]) -> Dict[str, Any]:
        copied = dict(item)
        if "target" in copied:
            copied["target"] = _programbench_redact_text(
                copied["target"], private_roots
            )
        for side in ("before", "after"):
            metadata = copied.get(side)
            if isinstance(metadata, dict) and "target" in metadata:
                copied[side] = {
                    **metadata,
                    "target": _programbench_redact_text(
                        metadata["target"], private_roots
                    ),
                }
        return copied

    return {
        "status": "completed",
        **outputs,
        "exit": value["exit"],
        "filesystem_effects": {
            "entries": [
                {"effect": category, **public_effect(item)}
                for category, item in flattened[:_PROGRAMBENCH_PUBLIC_EFFECT_ENTRIES]
            ],
            "entry_count": len(flattened),
            "entries_omitted": max(
                0, len(flattened) - _PROGRAMBENCH_PUBLIC_EFFECT_ENTRIES
            ),
        },
    }


def _programbench_public_input(case: Any) -> Dict[str, Any]:
    """Bounded, content-addressed provenance for one declarative probe input."""
    argv = [str(item) for item in getattr(case, "argv", ())]
    stdin = str(getattr(case, "stdin", "") or "")
    input_files = [
        {
            "path": str(item.path),
            "bytes": len(item.content),
            "sha256": hashlib.sha256(item.content).hexdigest(),
        }
        for item in getattr(case, "input_files", ())
    ]
    env = {str(key): str(value) for key, value in getattr(case, "env", ())}
    canonical = json.dumps(
        {
            "argv": argv,
            "stdin_sha256": hashlib.sha256(stdin.encode("utf-8")).hexdigest(),
            "input_files": input_files,
            "env": env,
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return {
        "sha256": hashlib.sha256(canonical).hexdigest(),
        "argv": [item[:256] for item in argv[:16]],
        "argv_count": len(argv),
        "stdin": {
            "bytes": len(stdin.encode("utf-8")),
            "sha256": hashlib.sha256(stdin.encode("utf-8")).hexdigest(),
            "text_preview": stdin[:512],
            "preview_truncated": len(stdin) > 512,
        },
        "input_files": input_files[:16],
        "input_file_count": len(input_files),
        "env": {key: value[:256] for key, value in sorted(env.items())},
    }


def _programbench_public_report(
    report: Dict[str, Any],
    *,
    private_roots: List[Path],
    source_by_case: List[tuple[str, int]],
    cases: Any,
    definitions: List[tuple[str, Any]],
) -> Dict[str, Any]:
    if len(source_by_case) != len(report["cases"]):
        raise ValueError("programbench_public_probe_source_map_invalid")
    definition_counts: Dict[str, int] = {}
    for source, _local_index in source_by_case:
        definition_counts[source] = definition_counts.get(source, 0) + 1
    definition_metadata = {
        source: {
            "artifact_id": str(getattr(artifact, "artifact_id", "") or ""),
            "revision": int(getattr(artifact, "revision", 0) or 0),
            "content_sha256": hashlib.sha256(
                str(getattr(artifact, "content", "") or "").encode("utf-8")
            ).hexdigest(),
        }
        for source, artifact in definitions
    }
    def public_case(row: Dict[str, Any]) -> Dict[str, Any]:
        public_row = {
            "case_index": row["case_index"],
            "probe_id": (
                source_by_case[row["case_index"]][0]
                + "#"
                + str(source_by_case[row["case_index"]][1])
            ),
            "definition_source": source_by_case[row["case_index"]][0],
            "definition": definition_metadata[
                source_by_case[row["case_index"]][0]
            ],
            "input": _programbench_public_input(cases[row["case_index"]]),
            "status": row["status"],
            "infra_side": row["infra_side"],
            "matched": row["matched"],
            "reference": _programbench_public_side(
                row["reference"], private_roots=private_roots
            ),
            "candidate": _programbench_public_side(
                row["candidate"], private_roots=private_roots
            ),
        }
        if "reference_repeat" in row:
            # Only a non-reproducible case carries these, so an ordinary
            # receipt canonicalizes to the same bytes and the same evidence
            # digest as before this evidence channel existed.
            public_row["reference_repeat"] = _programbench_public_side(
                row["reference_repeat"], private_roots=private_roots
            )
            public_row["reference_divergent_fields"] = list(
                row.get("reference_divergent_fields") or []
            )
        return public_row

    return {"report_schema_version": report["schema_version"]} | {
        key: report[key]
        for key in (
            "probe_schema_version",
            "mode",
            "comparison_policy",
            "case_count",
            "compared_case_count",
            "reference_observed_case_count",
            "matched_case_count",
            "mismatched_case_count",
            "infra_error_count",
        )
    } | {
        "cases": [public_case(row) for row in report["cases"]],
        "definition_case_counts": [
            {"definition_source": source, "case_count": count}
            for source, count in sorted(definition_counts.items())
        ],
    }


def _run_programbench_public_tests(
    world: Any,
    *,
    assets: Dict[str, Any],
    timeout: int,
    probe_mode: Literal["differential", "reference_only"] = "differential",
) -> Dict[str, Any]:
    from society_core.programbench_probes import (
        ProgramBenchProbeError,
        SandboxedCommandProbeExecutor,
        capture_probe_document_in_sandbox,
        evaluate_public_probes,
        load_probe_cases,
    )

    if probe_mode not in {"differential", "reference_only"}:
        return _programbench_failure(
            "programbench_public_probe_mode_invalid", probe_mode=str(probe_mode)
        )

    manifest = assets.get("manifest") or {}
    try:
        comparison_policy = _programbench_comparison_policy(manifest)
    except ValueError as error:
        return _programbench_failure(str(error))
    try:
        public, limits = _programbench_probe_contract(manifest)
        independent_reference_surface = bool(
            probe_mode == "reference_only"
            and "programbench_profile_state" in getattr(world, "__dict__", {})
        )
        candidate_overrides = (
            None
            if independent_reference_surface
            else _programbench_active_candidate_overrides(world)
        )
        definition_overrides = candidate_overrides
        definitions = _programbench_definition_artifacts(
            world,
            max_definitions=public["definition_surface"]["max_definitions"],
            candidate_overrides=definition_overrides,
        )
    except (TypeError, ValueError) as error:
        return _programbench_failure(str(error), probe_mode=probe_mode)
    sources = [path for path, _artifact in definitions]
    if not definitions:
        return _programbench_failure(
            "programbench_public_probe_definition_missing_or_seed_only",
            probe_mode=probe_mode,
        )
    try:
        executor = _programbench_probe_executor(manifest, public)
    except Exception as error:
        code = str(error) if isinstance(error, (ValueError, RuntimeError)) else (
            "programbench_public_probe_executor_initialization_failed"
        )
        return _programbench_failure(
            code, sources=sources, probe_mode=probe_mode
        )

    reference_source = Path(str(assets.get("reference_repo_dir") or ""))
    if not reference_source.is_dir():
        return _programbench_failure(
            "programbench_private_reference_repo_missing",
            sources=sources,
            probe_mode=probe_mode,
        )
    scratch_parent = Path(_product_smoke_root())
    try:
        with tempfile.TemporaryDirectory(
            prefix="programbench_public_", dir=scratch_parent
        ) as raw_scratch:
            scratch = Path(raw_scratch)
            definition_repo = scratch / "definition_surface"
            _programbench_export_definition_surface(
                definitions,
                definition_repo,
                candidate_overrides=definition_overrides,
            )

            captured_cases: List[Any] = []
            source_by_case: List[tuple[str, int]] = []
            for source in sources:
                payload = capture_probe_document_in_sandbox(
                    command_executor=executor,
                    candidate_repo=definition_repo,
                    script_path=source,
                    limits=limits,
                )
                document = json.loads(payload.decode("utf-8"))
                local_cases = document["cases"]
                captured_cases.extend(local_cases)
                source_by_case.extend(
                    (source, local_index)
                    for local_index in range(len(local_cases))
                )
                if len(captured_cases) > limits.max_cases:
                    raise ProgramBenchProbeError("probe_case_limit_exceeded")
            merged_document = json.dumps(
                {
                    "schema_version": "programbench_public_probe_cases_v1",
                    "cases": captured_cases,
                },
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
            cases = load_probe_cases(merged_document, limits=limits)
            if not cases:
                return _programbench_failure(
                    "programbench_public_probe_zero_cases",
                    sources=sources,
                    probe_mode=probe_mode,
                )
            # ``create_eval_stub`` intentionally starts from one structurally
            # valid, input-free case. Executing that seed against both sides
            # produces a comparison, but it exercises no caller-controlled
            # behavior and is not evidence of a materialized public probe.
            # One real input surface on any case is enough to be substantive.
            if all(
                not case.argv
                and not case.stdin
                and not case.input_files
                and not case.env
                for case in cases
            ):
                return _programbench_failure(
                    "programbench_public_probe_seed_only",
                    sources=sources,
                    probe_mode=probe_mode,
                )

            reference_root = scratch / "reference_program"
            candidate_root: Path | None = None
            tested_candidate_repo_digest: str | None = None
            candidate_surface_schema_version: str | None = None
            if probe_mode == "differential":
                candidate_root = scratch / "candidate_program"
                # Definition code has already produced the immutable public
                # input document in its own sandbox.  Re-export the official
                # probe-isolated candidate view for compilation instead of
                # copying that executable definition surface into the build
                # tree.  This is a physical boundary, so variable aliases,
                # ``find`` and child shell scripts cannot recover a probe file.
                if "programbench_profile_state" in getattr(
                    world, "__dict__", {}
                ):
                    export_programbench_integration_candidate(
                        world, str(candidate_root)
                    )
                    tested_candidate_repo_digest = (
                        programbench_exported_repo_digest(candidate_root)
                    )
                    expected_candidate_digest = (
                        programbench_integration_candidate_digest(world)
                    )
                    if tested_candidate_repo_digest != expected_candidate_digest:
                        return _programbench_failure(
                            "programbench_candidate_compile_view_digest_mismatch",
                            sources=sources,
                            probe_mode=probe_mode,
                        )
                    candidate_surface_schema_version = (
                        PROGRAMBENCH_CANDIDATE_VIEW_SCHEMA_VERSION
                    )
                else:
                    # Preserve the native ProgramBench probe lifecycle.  The
                    # probe-isolated official candidate is an opt-in adapted
                    # profile invariant; native callers still compare the
                    # ordinary working tree exactly as before.
                    export_product_repo(
                        world,
                        str(candidate_root),
                        prefer_mainline=False,
                    )
            # Preserve private-vault links as links. Dereferencing here could
            # make a trusted pack link copy bytes from outside its declared
            # reference root on the host before the sandbox boundary exists.
            shutil.copytree(
                reference_source,
                reference_root,
                copy_function=shutil.copy2,
                symlinks=True,
            )
            compile_command = list(public["compile"]["command"])
            if candidate_root is not None:
                candidate_compile_error = _programbench_compile(
                    executor, candidate_root, compile_command, timeout
                )
                if candidate_compile_error:
                    return _programbench_failure(
                        "programbench_candidate_compile_" + candidate_compile_error,
                        sources=sources,
                        probe_mode=probe_mode,
                    )
            reference_compile_error = _programbench_compile(
                executor, reference_root, compile_command, timeout
            )
            if reference_compile_error:
                return _programbench_failure(
                    "programbench_reference_compile_" + reference_compile_error,
                    sources=sources,
                    probe_mode=probe_mode,
                )

            output_path = public["compile"]["output_path"]
            reference_runtime = _programbench_runtime_root(
                reference_root, scratch / "reference_runtime", output_path
            )
            candidate_probe_executor = None
            if candidate_root is not None:
                candidate_runtime = _programbench_runtime_root(
                    candidate_root, scratch / "candidate_runtime", output_path
                )
                candidate_probe_executor = SandboxedCommandProbeExecutor(
                    command_executor=executor, program_root=candidate_runtime
                )
            reference_probe_executor = SandboxedCommandProbeExecutor(
                command_executor=executor,
                program_root=reference_runtime,
                execute_only=True,
            )
            report = evaluate_public_probes(
                probe_document=merged_document,
                reference_command=("executable",),
                candidate_command=(
                    ("executable",) if probe_mode == "differential" else None
                ),
                reference_executor=reference_probe_executor,
                candidate_executor=candidate_probe_executor,
                probe_mode=probe_mode,
                comparison_policy=comparison_policy,
                limits=limits,
            )
            public_report = _programbench_public_report(
                report,
                private_roots=[scratch, reference_source.resolve()],
                source_by_case=source_by_case,
                cases=cases,
                definitions=definitions,
            )
            if "programbench_profile_state" in getattr(world, "__dict__", {}):
                public_report["candidate_surface_schema_version"] = (
                    candidate_surface_schema_version
                )
                public_report["tested_candidate_repo_digest"] = (
                    tested_candidate_repo_digest
                )
    except (OSError, ProgramBenchProbeError, ValueError, json.JSONDecodeError) as error:
        code = str(error) if isinstance(error, (ProgramBenchProbeError, ValueError)) else (
            "programbench_public_probe_materialization_failed"
        )
        return _programbench_failure(
            code, sources=sources, probe_mode=probe_mode
        )

    if public_report["infra_error_count"]:
        result = _programbench_failure(
            "programbench_public_probe_infrastructure_failure",
            sources=sources,
            probe_mode=probe_mode,
        )
        result["collected"] = public_report["case_count"]
        result["programbench_public_probes"].update(public_report)
        result["programbench_public_probes"]["status"] = "failed"
        result["programbench_public_probes"]["failure"] = result["error"]
        result["programbench_public_probes"]["definition_sources"] = sources
        return result

    mismatches = public_report["mismatched_case_count"]
    if probe_mode == "reference_only":
        volatile = sum(
            row.get("status") == "reference_nondeterministic"
            for row in public_report["cases"]
        )
        summary = (
            f"{public_report['reference_observed_case_count']} of "
            f"{public_report['case_count']} ProgramBench public probes "
            "reproduced on reference; candidate skipped"
        )
        if volatile:
            summary += (
                f"; {volatile} case(s) gave two different reference "
                "observations and cannot be satisfied"
            )
    else:
        summary = (
            f"{public_report['case_count']} ProgramBench public probes compared; "
            f"{mismatches} behavior mismatches (informational)"
        )
    return {
        "ok": True,
        "available": True,
        "returncode": 0,
        "failed_tests": [],
        "summary": summary[:300],
        "error": None,
        "collected": public_report["case_count"],
        "programbench_public_probes": {
            "schema_version": "programbench_public_probe_materialization_v2",
            "status": "completed",
            "compile": {
                "candidate": (
                    "passed" if probe_mode == "differential" else "skipped"
                ),
                "reference": "passed",
            },
            "definition_sources": sources,
            **public_report,
        },
    }


def run_public_tests(
    world: Any,
    timeout: int = 120,
    *,
    probe_mode: Literal["differential", "reference_only"] = "differential",
    source_snapshot: Any = None,
) -> Dict[str, Any]:
    """Run the substrate's declared public test suite against the WORKING tree.

    Returns ``{ok, available, returncode, failed_tests, summary, error}``.
    ``available`` is False when the substrate declares no public tests, so a
    caller can distinguish "no suite here" from "suite failed" — the two must
    never be conflated into a single falsy verdict.
    """
    projection = {}
    if source_snapshot is not None:
        from environments.org_env.cooperbench.source_views import (
            SourceViewError, snapshot_projection, source_views_enabled,
        )
        if not source_views_enabled(world):
            raise SourceViewError("source_snapshot_not_enabled")
        projection = snapshot_projection(world, source_snapshot)
    programbench_assets = _programbench_public_probe_assets(world)
    if programbench_assets is not None:
        # Do not admit ProgramBench to the generic repo-hash cache. Definitions,
        # evaluator config, and pinned runtime identity all affect this result.
        return _run_programbench_public_tests(
            world,
            assets=programbench_assets,
            timeout=timeout,
            probe_mode=probe_mode,
        )
    command = declared_public_test_command(world)
    if not command:
        return {"ok": False, "available": False, "summary": "substrate declares no public tests"}
    cache = world.__dict__.setdefault("_public_test_cache", {})
    key = _repo_hash(world, prefer_mainline=False, **projection)
    if key in cache:
        if source_snapshot is not None and cache[key].get("error"):
            # A launcher outage is not a property of immutable source bytes.
            # Re-run only this failed snapshot; preserve all completed results.
            cache.pop(key, None)
        else:
            return cache[key]
    dest = os.path.join(_product_smoke_root(), "public_tests_" + key[:12])
    executor = _formal_product_executor()
    image_workspace = str(
        getattr(world, "__dict__", {}).get("_cooperbench_image_workspace") or ""
    ).strip() or None
    container_asset_root, read_only_mounts = _runtime_asset_container_context(
        world, executor
    )
    export_product_repo(
        world,
        dest,
        prefer_mainline=False,
        _runtime_asset_container_root=container_asset_root,
        **projection,
    )
    readiness_error = public_test_sandbox_readiness(world, dest, executor)
    if readiness_error:
        result = {"ok": False, "available": True, "returncode": None,
                  "failed_tests": [], "summary": "", "error": readiness_error}
        cache[key] = result
        return result
    environment = _safe_product_environment(dest, {})
    returncode, stdout, stderr, launch_error = _execute_smoke_command(
        command=list(command),
        repo_dir=dest,
        timeout=timeout,
        environment=environment,
        executor=executor,
        read_only_mounts=read_only_mounts,
        image_workspace=image_workspace,
    )
    if launch_error:
        result = {"ok": False, "available": True, "returncode": None,
                  "failed_tests": [], "summary": str(launch_error)[:300],
                  "error": str(launch_error),
                  "failure_brief": _bounded_public_test_failure_brief(f"{stdout}\n{stderr}")}
        cache[key] = result
        return result
    combined = f"{stdout}\n{stderr}"
    lines = combined.splitlines()
    failed = sorted({
        line.split("::")[0].strip() + "::" + line.split("::")[1].split()[0].strip()
        for line in lines
        if line.startswith("FAILED ") and "::" in line
    })
    go_failure = _go_test_failure_diagnostic(command, combined)
    go_test_failures = _go_test_failed_tests(command, combined)
    if not failed:
        failed = [go_failure] if go_failure else go_test_failures
    summary = ""
    for line in reversed(lines):
        stripped = line.strip()
        if any(tok in stripped for tok in (" passed", " failed", " error", "no tests ran")):
            summary = stripped
            break
    if go_failure:
        summary = go_failure
    elif go_test_failures:
        summary = f"go test failed: {go_test_failures[0]}"
    harness_error = (
        None
        if go_failure or go_test_failures
        else _test_harness_launch_error(returncode, combined, summary)
    )
    if harness_error:
        # The suite never ran: reporting this as "tests failed" would send the org
        # after a code defect that does not exist - the same misattribution that
        # turned an empty container mount into 172 phantom contract breaks.
        result = {"ok": False, "available": True, "returncode": returncode,
                  "failed_tests": [], "summary": summary[:300], "error": harness_error,
                  "failure_brief": _bounded_public_test_failure_brief(combined)}
        cache[key] = result
        return result
    result = {"ok": returncode == 0, "available": True, "returncode": returncode,
              "failed_tests": failed[:20], "summary": summary[:300], "error": None,
              "collected": _collected_test_count(summary),
              "failure_brief": (
                  _bounded_public_test_failure_brief(combined)
                  if returncode != 0
                  else ""
              )}
    cache[key] = result
    return result


def _bounded_public_test_failure_brief(output: str) -> str:
    """Return prompt-safe public traceback context for an implementation repair."""

    from society_core.code_landing.event_store import redact_sensitive_payload

    redacted = str(redact_sensitive_payload(str(output or "")))
    # The exception and innermost candidate frames are at the tail of pytest
    # output.  Keeping only that tail bounds prompt/log growth while preserving
    # the concrete error that a repair agent needs (for example a missing name
    # in the edited module rather than merely three opaque test node ids).
    return redacted[-2000:]


_OUTCOME_COUNT_RE = re.compile(
    r"(\d+)\s+(passed|failed|error|errors|skipped|xfailed|xpassed)\b")


def _collected_test_count(summary: str) -> int:
    """How many tests the runner actually reported on, from its summary line.

    The suite is agent-editable, so the organization can make a red suite green
    by deleting or weakening tests instead of fixing the product. A pass rate
    cannot distinguish those; the size of the suite can. Counting every reported
    outcome - not just passes - is what makes a shrunk suite visible.

    Returns 0 when no outcome counts appear, which callers must read as "no
    measurement", never as "the suite is empty".
    """
    return sum(int(n) for n, _ in _OUTCOME_COUNT_RE.findall(str(summary or "")))


_HARNESS_LAUNCH_MARKERS = (
    "error importing plugin",
    "no module named",
    "importerror",
    "modulenotfounderror",
    "error: unrecognized arguments",
    "usage error",
    "internal error",
)


_GO_TEST_DIAGNOSTIC_RE = re.compile(
    r"^(?:\./)?[^:\r\n]+\.go:\d+:\d+:\s+\S.*$"
)
_GO_TEST_FAILURE_RE = re.compile(
    r"^\s*--- FAIL:\s+(\S+)(?:\s+\([^)]*\))?\s*$"
)


def _go_test_failure_diagnostic(command: List[str], output: str) -> str:
    """Return the concrete compiler diagnostic from a failed ``go test``."""

    if (
        len(command) < 2
        or os.path.basename(str(command[0])) != "go"
        or command[1] != "test"
    ):
        return ""
    for line in reversed(str(output or "").splitlines()):
        diagnostic = line.strip()
        if _GO_TEST_DIAGNOSTIC_RE.match(diagnostic):
            return diagnostic
    return ""


def _go_test_failed_tests(command: List[str], output: str) -> List[str]:
    """Return test names reported by a failed ``go test`` run."""

    if (
        len(command) < 2
        or os.path.basename(str(command[0])) != "go"
        or command[1] != "test"
    ):
        return []
    failed = []
    for line in str(output or "").splitlines():
        match = _GO_TEST_FAILURE_RE.match(line)
        if match and match.group(1) not in failed:
            failed.append(match.group(1))
    return failed


def public_test_evidence_from_smoke(
    outcome: Dict[str, Any], command: List[str]
) -> Dict[str, Any]:
    """Attach bounded test identities to an already-executed smoke result.

    ``release_smoke`` predates committed-source Cooper verification and keeps
    only bounded stdout/stderr. A guarded public-contract compatibility receipt
    must compare exact failing node IDs and the ordinary-test count. Normalize
    those fields without launching a second test process or originating a new
    compatibility exception.
    """

    normalized = dict(outcome or {})
    combined = "\n".join(
        str(normalized.get(key) or "")
        for key in ("stdout_tail", "stderr_tail", "error")
    )
    lines = combined.splitlines()
    failed = sorted({
        line.split("::")[0].strip() + "::" + line.split("::")[1].split()[0].strip()
        for line in lines
        if line.startswith("FAILED ") and "::" in line
    })
    go_failure = _go_test_failure_diagnostic(list(command), combined)
    go_test_failures = _go_test_failed_tests(list(command), combined)
    if not failed:
        failed = [go_failure] if go_failure else go_test_failures
    summary = ""
    for line in reversed(lines):
        stripped = line.strip()
        if any(tok in stripped for tok in (" passed", " failed", " error", "no tests ran")):
            summary = stripped
            break
    if go_failure:
        summary = go_failure
    elif go_test_failures:
        summary = f"go test failed: {go_test_failures[0]}"
    normalized.update(
        available=True,
        failed_tests=failed[:20],
        collected=_collected_test_count(summary),
        summary=summary[:300] or str(normalized.get("summary") or "")[:300],
    )
    return normalized


def _test_harness_launch_error(
    returncode: int | None,
    output: str,
    summary: str,
) -> Optional[str]:
    """Distinguish "the test runner could not start" from "tests failed".

    A missing test plugin, an unimportable conftest or a usage error means the
    suite never executed, so there is no evidence about the product at all.
    Substrate-independent signal: no per-test outcome summary was produced AND
    the output carries a harness-level import/usage failure.
    """
    if returncode == 0:
        return None
    produced_outcomes = bool(summary) and any(
        token in summary for token in (" passed", " failed", " error")
    )
    if produced_outcomes:
        return None
    lowered = output.lower()
    for marker in _HARNESS_LAUNCH_MARKERS:
        if marker in lowered:
            for line in reversed(output.splitlines()):
                if marker in line.lower():
                    return f"public_test_harness_failed:{line.strip()[:180]}"
            return f"public_test_harness_failed:{marker}"
    # Fundamental criterion, independent of how the runner phrases its error: a
    # non-zero exit that produced NO per-test outcome at all means the suite never
    # ran. Enumerating error shapes is endless - a duplicate-plugin registration
    # surfaces as a bare ValueError with empty stdout and matches no marker.
    if not produced_outcomes:
        detail = ""
        for line in reversed(output.splitlines()):
            stripped = line.strip()
            if stripped and not stripped.startswith(("File \"", "  ", "{")):
                detail = stripped
                break
        return (
            "public_test_harness_produced_no_outcomes:"
            + (detail[:180] or f"exit={returncode}")
        )
    return None


def _declared_smoke_command(world: Any) -> Optional[List[str]]:
    """Return an OSS manifest smoke command without exposing evaluator-only paths or content."""
    from environments.org_env.product.substrates.eval_assets import (
        oss_eval_assets,
    )

    assets = oss_eval_assets(world) or {}
    manifest = assets.get("manifest") or {}
    command = ((manifest.get("entrypoints") or {}).get("smoke") or {}).get("command")
    return list(command) if isinstance(command, list) and command else None


def pr_public_test_command(world: Any, pr: Any) -> Optional[List[str]]:
    """Narrow a PR's public CI to the issues that request actually carries.

    A pack may expose one acceptance module per public issue.  Running the whole
    directory on every PR makes independent work impossible: a correct fix for
    issue A remains red until B..N are implemented too.  Running only a generic
    smoke is the opposite failure -- any edit turns CI green.  The merge
    candidate therefore runs the pack's top-level smoke modules plus the
    acceptance modules for ``pr.linked_issue_ids``.

    Packs without issue-level acceptance modules retain their declared command.
    No evaluator-only path or hidden test is consulted here.
    """
    command = _declared_smoke_command(world)
    if not command or pr is None:
        return command

    linked = list(getattr(pr, "linked_issue_ids", None) or [])
    fallback = str(getattr(pr, "linked_issue", "") or "")
    if fallback and fallback not in linked:
        linked.append(fallback)
    if not linked:
        return command

    artifact_paths = {
        str(getattr(artifact, "linked_file_path", "") or "").replace("\\", "/")
        for artifact in (getattr(world, "product_artifacts", {}) or {}).values()
    }
    selected = []
    for issue_id in linked:
        safe = re.sub(r"[^A-Za-z0-9_]+", "_", str(issue_id)).strip("_")
        path = f"tests/public/acceptance/test_{safe}.py"
        if path in artifact_paths:
            selected.append(path)
    if not selected:
        return command

    generic = sorted(
        path
        for path in artifact_paths
        if path.startswith("tests/public/")
        and path.endswith(".py")
        and "/acceptance/" not in path
        and path.count("/") == 2
    )
    targets = [*generic, *dict.fromkeys(selected)]

    narrowed: List[str] = []
    replaced = False
    for token in command:
        normalized = str(token).replace("\\", "/").rstrip("/")
        if normalized == "tests/public":
            narrowed.extend(targets)
            replaced = True
        else:
            narrowed.append(str(token))
    return narrowed if replaced else command


def smoke_error_brief(sm: Dict[str, Any]) -> str:
    """A short, actionable error from a failed smoke run: '<file>:<line> | <ErrorType: msg>'.
    Fed into the release-blocker issue + the code editor so agents fix the real break."""
    import re
    if not sm or sm.get("ok"):
        return ""
    if sm.get("error"):
        return str(sm["error"])
    stdout = str(sm.get("stdout_tail") or "")
    stderr = str(sm.get("stderr_tail") or "")
    # Pytest writes assertion failures and source locations to stdout, while
    # plugin warnings go to stderr.  Reading stderr alone made every red TG CI
    # say only "PytestDeprecationWarning", so agents re-ran the same check
    # without learning which product file was broken.
    for stream in (stdout, stderr):
        locations = []
        for line in stream.splitlines():
            match = re.match(
                r"^([A-Za-z0-9_./\\-]+\.py):(\d+):\s*(.+)$",
                line.strip(),
            )
            if not match:
                continue
            path = match.group(1).replace("\\", "/")
            if path.startswith("tests/") or "site-packages/" in path or "/.venv/" in path:
                continue
            locations.append((path, match.group(2), match.group(3)))
        if locations:
            path, line_number, detail = locations[-1]
            return f"{path}:{line_number} | {detail}"[:300]

    lines = [ln.rstrip() for ln in (stdout + "\n" + stderr).splitlines() if ln.strip()]
    failures = [line.strip()[7:] for line in lines if line.strip().startswith("FAILED ")]
    if failures:
        return failures[0][:300]

    useful = [
        line
        for line in lines
        if "PytestDeprecationWarning" not in line
        and not line.lstrip().startswith("warnings.warn(")
    ]
    if not useful:
        return f"smoke exited rc={sm.get('returncode')}"
    frame = ""
    for ln in useful:
        if ln.lstrip().startswith("File "):
            frame = ln.strip()
    final = useful[-1]

    # Where the steps stack, the earliest failing one is the one to fix: the
    # rest fail because it does. Taking the last line named the last step
    # instead, so a run whose step 3 returned a malformed manifest was told
    # about step 5, and eight people spread their edits over three modules for
    # two hundred ticks without one of them going back to step 3.
    steps = []
    for ln in lines:
        m = re.search(r"\bstep (\d+)\b.*\bfail", ln, re.IGNORECASE)
        if m:
            steps.append((int(m.group(1)), ln.strip()))
    if steps:
        first, said = min(steps, key=lambda s: s[0])
        rest = sorted({n for n, _ in steps} - {first})
        trailing = (f"  (steps {', '.join(str(n) for n in rest)} fail behind it)"
                    if rest else "")
        return (said + trailing)[:300]


    m = re.search(r'File ".*[\\/]([^"\\/]+)", line (\d+)', frame)
    loc = f"{m.group(1)}:{m.group(2)} | " if m else ""
    return (loc + final)[:300]


# v14b: tolerance for evaluator self-consistency. A correct eval has
# `unsupported_claim_rate ≈ 1 - claim_evidence_coverage`. If the two disagree by more than this,
# the EVALUATOR is internally inconsistent (not the product) — see smoke_eval_inconsistency.
_EVAL_CONSISTENCY_TOL = 0.34


def _smoke_metrics(sm: Dict[str, Any]):
    """Return (summary, metrics) dicts from a smoke result, or (summary, None)."""
    summary = sm.get("summary") if isinstance(sm.get("summary"), dict) else {}
    metrics = sm.get("metrics")
    if not isinstance(metrics, dict):
        metrics = summary.get("metrics") if isinstance(summary.get("metrics"), dict) else {}
    return summary, (metrics if isinstance(metrics, dict) else None)


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def smoke_quality_issue(sm: Dict[str, Any]) -> str:
    """v11 §8 red light: a smoke run that DOESN'T CRASH can still be semantically broken
    (the v10/v11 finding: `credibility_score=0` + `eval_failed` both "passed"). Enforce —
    ONLY when the smoke itself reports them — that the product actually WORKS:
      * eval runs cleanly        (summary.eval_result has no 'error')
      * claims are grounded       (metrics.credibility_score / claim_evidence_coverage > 0)
    Returns a brief that LEADS with the suspected '<file>.py' so the localizer + debugging
    loop target the right module; '' when clean or when the run doesn't report these signals
    (we never block the early prototype before the feature exists).

    v14b (anti false-negative): this gate must judge the PRODUCT, not trust a possibly-buggy
    evaluator. If claims are source-matched (coverage / credibility > 0) yet the eval ALSO
    reports `unsupported_claim_rate=1.0`, that contradiction is an evaluator bug — we do NOT
    block a working product on it (and `smoke_eval_inconsistency` localizes it to eval_stub).
    This is exactly what stalled v14: a t232 eval patch set unsupported=1.0 while coverage
    stayed 1.0, the old gate believed it, blocked every release, and mislocalized to
    claim_tracker.py — so the org optimized the wrong module for two weeks."""
    if not sm or not sm.get("ok"):
        return ""   # an outright crash is already covered by smoke_error_brief
    summary, metrics = _smoke_metrics(sm)
    ev = summary.get("eval_result")
    if isinstance(ev, dict) and ev.get("error"):
        return (f"eval/eval_stub.py | eval did not run cleanly (eval_result.error={ev.get('error')}): "
                "run_eval raised on the real claims/sources — fix eval/eval_stub.py")
    if metrics is None:
        return ""

    # primary grounding signal: agent-added credibility_score (semantic) if present, else the
    # seed's structural claim_evidence_coverage. This is the TRUTH about whether claims are tied
    # to sources; the all-unsupported red light only fires when it does NOT contradict this.
    cred = _num(metrics.get("credibility_score"))
    cov = _num(metrics.get("claim_evidence_coverage"))
    grounded = cred if cred is not None else cov

    if grounded is not None and grounded <= 0:
        return ("tools/claim_tracker.py | grounding=0 (no claim is linked to a source) — make "
                "research_loop.py/claim_tracker.py attach source_ids so claims are supported")
    unsup = _num(metrics.get("unsupported_claim_rate"))
    # only treat "every claim unsupported" as a PRODUCT failure when grounding doesn't say
    # otherwise — a high unsupported_rate alongside high coverage/credibility is an evaluator
    # contradiction (surfaced non-blocking via smoke_eval_inconsistency), not a product break.
    if unsup is not None and unsup >= 1.0 and not (grounded is not None and grounded > 0):
        return ("tools/claim_tracker.py | unsupported_claim_rate=1.0 (every claim is unsupported) — "
                "link claims to sources in research_loop.py/claim_tracker.py")
    # v14 P2 (anti spec-gaming): synthetic / fallback source attachment does NOT count as real
    # grounding. v13b passed coverage=1.0 with fallback_attached_count=4 (every claim auto-attached
    # to a generated source) — structurally grounded, semantically empty. If the REAL grounding
    # (claims tied to a non-fallback source) is <= 0, the gate fails.
    n = _num(metrics.get("n_claims"))
    fb = _num(metrics.get("fallback_attached_count"))
    if n and n > 0 and fb is not None and fb > 0:
        real = (cov if cov is not None else (grounded or 0)) * n - fb
        if real <= 0:
            return ("tools/claim_tracker.py | grounding is FALLBACK-ONLY (" + str(int(fb)) + " synthetic "
                    "source attachments) — link claims to REAL retrieved sources, not auto-generated ones")
    return ""


def smoke_eval_inconsistency(sm: Dict[str, Any]) -> str:
    """v14b: detect a SELF-CONTRADICTORY evaluator and localize it to eval/eval_stub.py.

    A correct eval satisfies `unsupported_claim_rate ≈ 1 - claim_evidence_coverage`. When claims
    are source-matched (coverage / credibility high) yet the eval reports them all-unsupported, the
    metric is internally inconsistent — the bug is in `run_eval`, NOT in claim tracking. Returns an
    eval_stub-localized brief (so CI / the debugging loop fix the evaluator); '' when consistent.
    This is informational: it must NOT by itself block a product that is actually grounded."""
    if not sm or not sm.get("ok"):
        return ""
    _summary, metrics = _smoke_metrics(sm)
    if metrics is None:
        return ""
    cred = _num(metrics.get("credibility_score"))
    cov = _num(metrics.get("claim_evidence_coverage"))
    unsup = _num(metrics.get("unsupported_claim_rate"))
    gr = _num(metrics.get("grounding_rate"))
    if unsup is not None:
        if cov is not None and (cov - (1.0 - unsup)) > _EVAL_CONSISTENCY_TOL:
            return (f"eval/eval_stub.py | metric self-contradiction: claim_evidence_coverage={cov} but "
                    f"unsupported_claim_rate={unsup} (should be ~{round(1.0 - cov, 2)}) — fix the "
                    "supported/unsupported logic in run_eval (eval/eval_stub.py), not claim_tracker.py")
        if cred is not None and cred > 0 and unsup >= 1.0:
            return (f"eval/eval_stub.py | metric self-contradiction: credibility_score={cred} but "
                    "unsupported_claim_rate=1.0 — fix the supported/unsupported logic in run_eval "
                    "(eval/eval_stub.py), not claim_tracker.py")
    # v14c: a dedicated SEMANTIC grounding metric must not flatly contradict the STRUCTURAL one.
    # v14b shipped with claim_evidence_coverage=1.0 yet grounding_rate=0.0 — incoherent quality
    # signals that let "0 real grounding" sail through. Force the eval to reconcile them.
    base = cov if cov is not None else cred
    if gr is not None and base is not None and (base - gr) > _EVAL_CONSISTENCY_TOL:
        return (f"eval/eval_stub.py | metric self-contradiction: grounding_rate={gr} but "
                f"claim_evidence_coverage/credibility={base} — reconcile the structural vs semantic "
                "grounding metrics in run_eval (eval/eval_stub.py); don't report both 0 and 1")
    return ""


def export_and_smoke(world: Any, dest: str, timeout: int = 25) -> Dict[str, Any]:
    exp = export_product_repo(world, dest)
    smoke = run_product_smoke(
        dest,
        timeout=timeout,
        command=_declared_smoke_command(world),
    )
    return {"export": exp, "smoke": smoke}


def _repo_hash(
    world: Any,
    prefer_mainline: bool = True,
    *,
    overrides: Optional[Dict[str, str]] = None,
    exclude_artifact_ids: set[str] | frozenset[str] | None = None,
) -> str:
    arts = getattr(world, "product_artifacts", {}) or {}
    replacements = overrides or {}
    excluded = set(exclude_artifact_ids or ())
    h = hashlib.sha1()
    entries = []
    registry_entries = _repo_artifact_entries(arts)
    assets, _protected = _runtime_asset_export_context(world, registry_entries)
    if assets is not None:
        h.update(b"cooperbench_repo_with_runtime_assets_v1\0")
        h.update(assets.asset_set_digest.encode("ascii"))
        h.update(b"\0")
    for p, artifact_id, a in registry_entries:
        if artifact_id in excluded:
            continue
        if (
            prefer_mainline
            and artifact_id not in replacements
            and _is_unmerged_new_file(a)
        ):
            continue
        entries.append((p, artifact_id, a))
    for p, artifact_id, a in sorted(entries):
        h.update(p.encode("utf-8")); h.update(b"\0")
        content = replacements.get(artifact_id)
        if artifact_id not in replacements:
            content = _file_text(a, prefer_mainline)
        h.update(content.encode("utf-8")); h.update(b"\0")
    return h.hexdigest()


def _product_smoke_root() -> str:
    """Directory that exported product trees are written to before a smoke run.

    Defaults to the system temp dir, which is correct for a plain subprocess
    smoke. Containerized (formal) runs need a path the container runtime is
    allowed to bind-mount: on macOS, Docker Desktop shares neither the system
    TMPDIR (``/var/folders/...``) nor ``/tmp`` by default, so a temp-dir export
    mounts as an EMPTY workspace and every command fails as if the product were
    broken. Set ``ORG_PRODUCT_SMOKE_ROOT`` to a shared path, or rely on the
    repo-local fallback used whenever a container executor is configured.
    """
    configured = (os.environ.get("ORG_PRODUCT_SMOKE_ROOT") or "").strip()
    if configured:
        root = os.path.abspath(os.path.expanduser(configured))
        os.makedirs(root, exist_ok=True)
        return root
    containerized = (
        str(os.environ.get("ORG_OSS_MODE") or "").lower() == "formal"
        and (os.environ.get("ORG_EVALUATOR_BACKEND") or "").strip()
    )
    if containerized:
        root = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))))),
            ".product_smoke_workspaces",
        )
        os.makedirs(root, exist_ok=True)
        return root
    return tempfile.gettempdir()


def merge_candidate_text(world: Any, pr: Any) -> Dict[str, str]:
    """The tree a merge of ``pr`` would produce: mainline, plus this PR's patches.

    CI used to judge the whole working tree, which fails a pull request for code
    it does not carry. On a pack whose steps stack that is fatal rather than
    merely strict: an organization that started step 3 could never land its
    correct step 1, because the same working tree held the unfinished step 3. The
    mainline then stays empty however much correct work exists, and the
    evaluation — which scores the mainline — reports nothing. A real CI builds
    the merge candidate, so this does too.

    Legacy seeded artifacts with no mainline text retain their historical
    working-copy fallback. A file explicitly marked ``created_as_new_file`` has
    stronger semantics: it is absent from mainline until this PR supplies an
    override and remains absent from every other PR candidate.
    """
    arts = getattr(world, "product_artifacts", {}) or {}
    patches = getattr(world, "patches", {}) or {}
    out: Dict[str, str] = {}
    try:
        carried = world.repo_system.merged_commit_patches(pr)
    except Exception:  # noqa: BLE001  a PR with no readable commits carries nothing
        return out
    for patch_id, artifact_id in carried:
        art = arts.get(artifact_id) if artifact_id else None
        patch = patches.get(patch_id)
        if art is None:
            raise ValueError(f"merge_candidate_artifact_missing:{artifact_id}")
        if patch is None:
            raise ValueError(f"merge_candidate_patch_missing:{patch_id}")
        if str(getattr(patch, "target_object_id", "") or "") != str(artifact_id):
            raise ValueError(f"merge_candidate_patch_artifact_mismatch:{patch_id}")
        if patch is not None and getattr(patch, "creates_file", False):
            base = int(getattr(patch, "base_mainline_revision", 0) or 0)
            current = int(getattr(art, "mainline_revision", 0) or 0)
            if base != 0 or current != base:
                path = str(getattr(art, "linked_file_path", "") or artifact_id)
                raise ValueError(f"new_file_create_conflict:{path}")
        promoted = getattr(patch, "new_content", "")
        if getattr(art, "created_as_new_file", False) and not isinstance(promoted, str):
            raise ValueError(f"merge_candidate_create_text_missing:{patch_id}")
        # Legacy non-create patches may still carry no full text. New files may
        # never fall back to the global working copy: that would merge another
        # branch's unaudited content when a checkpoint loses its patch ledger.
        out[artifact_id] = (
            promoted
            if getattr(art, "created_as_new_file", False)
            else promoted or (getattr(art, "content", "") or "")
        )
    return out


def _cacheable_smoke_result(result: Any) -> bool:
    """Whether a smoke result is determined by candidate content.

    ``run_product_smoke`` uses ``error`` both for recoverable launch failures
    and for two deterministic candidate failures. Keep the latter cacheable;
    every other error envelope must be retried when infrastructure recovers.
    """

    if not isinstance(result, dict):
        return False
    error = str(result.get("error") or "")
    return (
        not error
        or error == "no smoke_check.py in exported repo"
        or result.get("cli_returncode") is not None
    )


def _runtime_asset_container_context(
    world: Any, executor: Any
) -> tuple[str | None, tuple[tuple[Path, str], ...]]:
    """Bind a large verified asset pack read-only instead of copying it."""

    if executor is None or not callable(
        getattr(executor, "run_with_read_only_mounts", None)
    ):
        return None, ()
    from environments.org_env.cooperbench.runtime_assets import (
        CONTAINER_ASSET_ROOT,
        runtime_asset_backing_root,
        runtime_assets_for_world,
    )

    assets = runtime_assets_for_world(world)
    backing = runtime_asset_backing_root(assets) if assets is not None else None
    if backing is None:
        return None, ()
    return CONTAINER_ASSET_ROOT, ((backing, CONTAINER_ASSET_ROOT),)


def _run_executor_with_read_only_mounts(
    executor: Any,
    *,
    root: Path,
    argv: tuple[str, ...],
    timeout_seconds: float,
    read_only_mounts: tuple[tuple[Path, str], ...] = (),
):
    if not read_only_mounts:
        return executor.run(root=root, argv=argv, timeout_seconds=timeout_seconds)
    runner = getattr(executor, "run_with_read_only_mounts", None)
    if not callable(runner):
        raise RuntimeError("executor_read_only_mounts_unsupported")
    return runner(
        root=root,
        argv=argv,
        timeout_seconds=timeout_seconds,
        read_only_mounts=read_only_mounts,
    )


def release_smoke(
    world: Any,
    prefer_mainline: bool = True,
    timeout: int = 25,
    overrides: Dict[str, str] | None = None,
    command: Optional[List[str]] = None,
    exclude_artifact_ids: set[str] | frozenset[str] | None = None,
) -> Dict[str, Any]:
    """Export the (mainline) tree to a temp dir and run smoke_check.py, CACHED by content hash
    so the release gate can call it every sweep without re-spawning a subprocess each tick."""
    cache = world.__dict__.setdefault("_smoke_cache", {})
    if exclude_artifact_ids is not None:
        key = _repo_hash(world, prefer_mainline, overrides=overrides, exclude_artifact_ids=exclude_artifact_ids)
    else:
        key = _repo_hash(world, prefer_mainline)
    if overrides and exclude_artifact_ids is None:
        # A merge candidate is a different tree from the mainline it is based on,
        # so it must not answer from the mainline's cache entry.
        key = hashlib.sha1(
            (key + "|" + "|".join(f"{k}:{hashlib.sha1(v.encode()).hexdigest()}"
                                  for k, v in sorted(overrides.items()))).encode()
        ).hexdigest()
    smoke_command = list(command) if command is not None else _declared_smoke_command(world)
    image_workspace = str(
        getattr(world, "__dict__", {}).get("_cooperbench_image_workspace") or ""
    ).strip() or None
    if smoke_command:
        # The same candidate may be judged against different issue-level tests;
        # a result cached for issue A cannot answer issue B.
        key = hashlib.sha1(
            (key + "|command:" + json.dumps(smoke_command, separators=(",", ":"))).encode()
        ).hexdigest()
    if image_workspace:
        key = hashlib.sha1(
            (key + "|image_workspace:" + image_workspace).encode()
        ).hexdigest()
    cached = cache.get(key)
    if cached is not None:
        # ``error`` is the public smoke-result envelope for launch and other
        # infrastructure failures.  Those outcomes are not properties of the
        # candidate tree: a stopped container daemon or a temporarily missing
        # mount can recover while the content hash remains unchanged.  Older
        # checkpoints may already contain such an entry, so evict it on read
        # and retry instead of making the failure permanent.
        if _cacheable_smoke_result(cached):
            return cached
        # Legacy checkpoints did not explicitly tag infrastructure errors.
        # Anything outside the two deterministic candidate-error shapes is
        # therefore evicted and retried under current infrastructure state.
        cache.pop(key, None)
    dest = os.path.join(_product_smoke_root(), "product_smoke_" + key[:12])
    executor = _formal_product_executor()
    container_asset_root, read_only_mounts = _runtime_asset_container_context(
        world, executor
    )
    export_product_repo(
        world,
        dest,
        prefer_mainline=prefer_mainline,
        overrides=overrides,
        exclude_artifact_ids=exclude_artifact_ids,
        _runtime_asset_container_root=container_asset_root,
    )
    llm_env = _product_llm_env()
    smoke_kwargs = {
        "timeout": (60 if llm_env else timeout),
        "env": llm_env,
        "command": smoke_command,
    }
    # Keep the established runner surface unchanged for ordinary exports.  The
    # executor/mount extension is needed only by directory-backed CooperBench
    # assets; passing it unconditionally breaks compatible injected runners
    # that implement the long-standing four-argument protocol.
    if read_only_mounts or image_workspace:
        smoke_kwargs.update(
            executor=executor,
            read_only_mounts=read_only_mounts,
        )
    if image_workspace:
        smoke_kwargs["image_workspace"] = image_workspace
    res = run_product_smoke(dest, **smoke_kwargs)
    res["repo_dir"] = os.path.abspath(dest)
    # Cache successful checks and deterministic candidate failures (which have
    # a return code), but never cache the infrastructure/error envelope.
    if _cacheable_smoke_result(res):
        cache[key] = res
    return res


def export_release_snapshot(world: Any, version: str, base: str = None) -> Dict[str, Any]:
    """Materialize the MAINLINE tree as a real, persistent release snapshot directory."""
    base = base or os.path.join("docs", "product_exports", "releases")
    dest = os.path.join(base, "v" + str(version))
    exp = export_product_repo(world, dest, prefer_mainline=True)
    smoke = run_product_smoke(dest, command=_declared_smoke_command(world))
    return {"export": exp, "smoke": smoke, "version": str(version)}


__all__ = ["PROGRAMBENCH_CANDIDATE_VIEW_SCHEMA_VERSION",
           "export_product_repo", "export_programbench_integration_candidate",
           "export_programbench_mainline",
           "programbench_exported_repo_digest",
           "programbench_integration_candidate_digest",
           "programbench_integration_candidate_has_pending",
           "programbench_integration_candidate_overrides",
           "programbench_mainline_digest",
           "programbench_pr_candidate_digest",
           "programbench_strict_mainline_overrides",
           "programbench_probe_corpus_digest",
           "run_product_smoke", "export_and_smoke",
           "declared_public_test_command", "declared_public_test_dependencies",
           "pr_public_test_command",
           "public_test_sandbox_readiness", "run_public_tests",
           "release_smoke", "export_release_snapshot", "smoke_error_brief", "smoke_quality_issue",
           "smoke_eval_inconsistency", "merge_candidate_text"]
