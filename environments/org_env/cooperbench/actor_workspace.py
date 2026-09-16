"""Native action transactions over explicit, owner-private Cooper source desks.

This module never changes the shared artifact content as a way of selecting an
actor's view. RepoLite patch/commit ledgers are the source; descriptors remain
path identities, and public-test receipts are per actor and exact source tree.
"""
from __future__ import annotations

import copy
import difflib
from typing import Any

from . import source_views as sv


_PATCH_GUARD_FAILURES = "_cooperbench_actor_patch_guard_failures"
_PATCH_GUARD_FAILURE_HISTORY = "_cooperbench_actor_patch_guard_failure_history"


def _patch_guard_failure_brief(receipt: dict) -> str:
    behavior = receipt.get("behavior_evidence") or {}
    results = {
        str(row.get("probe_id") or ""): row
        for row in behavior.get("results") or []
        if isinstance(row, dict) and row.get("probe_id")
    }
    probes = {
        str(row.get("probe_id") or ""): row
        for row in behavior.get("probes") or []
        if isinstance(row, dict) and row.get("probe_id")
    }
    failed_protected = list(receipt.get("failed_protected_probe_ids") or [])
    failed_repair = list(receipt.get("failed_repair_probe_ids") or [])
    failed_preservation = list(receipt.get("failed_preservation_probe_ids") or [])
    incomplete = list(receipt.get("incomplete_probe_ids") or [])
    failed_ids = list(dict.fromkeys(
        failed_protected + failed_repair + failed_preservation
    ))
    if not failed_ids:
        failed_ids = [
            probe_id for probe_id, row in results.items()
            if (row.get("candidate") or {}).get("status") == "fail"
        ]

    def bounded(value: object, limit: int) -> str:
        text = str(value or "").strip()
        if len(text) <= limit:
            return text
        half = max(1, (limit - 45) // 2)
        return text[:half] + "\n...[bounded middle omitted]...\n" + text[-half:]

    def result_detail(value: dict, status: str) -> str:
        parts = []
        if value.get("error"):
            parts.append("ERROR: " + bounded(value["error"], 220))
        if value.get("output"):
            parts.append("OUTPUT:\n" + bounded(value["output"], 550))
        return "\n".join(parts) or status

    evidence_ids = list(dict.fromkeys(failed_ids + incomplete))
    rows = []
    for probe_id in evidence_ids[:3]:
        row = results.get(str(probe_id)) or {}
        candidate = row.get("candidate") or {}
        probe = probes.get(str(probe_id)) or {}
        requirement_ids = [str(value) for value in probe.get("requirement_ids") or []]
        code = bounded(probe.get("code"), 1500)
        candidate_status = str(candidate.get("status") or "missing")
        candidate_detail = result_detail(candidate, candidate_status)
        if candidate_status == "fail" and code:
            rows.append(
                f"EXACT HOST-EXECUTED PUBLIC CHECK {probe_id} "
                f"(requirement_ids={requirement_ids}):\n{code}\n"
                f"FRESH FAILURE OUTPUT:\n{candidate_detail}"
            )
        elif code:
            side_details = [
                f"CANDIDATE STATUS: {candidate_status}\n{candidate_detail}"
            ]
            baseline = row.get("baseline") or {}
            if probe.get("compare_baseline") and baseline.get("status") != "pass":
                baseline_status = str(baseline.get("status") or "missing")
                baseline_detail = result_detail(baseline, baseline_status)
                side_details.append(
                    f"BASELINE STATUS: {baseline_status}\n{baseline_detail}"
                )
            rows.append(
                f"EXACT HOST-EXECUTED PUBLIC CHECK {probe_id} "
                f"(requirement_ids={requirement_ids}):\n{code}\n"
                + "\n".join(side_details)
            )
        else:
            rows.append(f"{probe_id}: {candidate_status}: {candidate_detail}")
    detail = "\n\n".join(rows) or str(receipt.get("error") or "evidence unavailable")
    protected = ", ".join(receipt.get("protected_features") or [])
    repair_feature = str(receipt.get("repair_feature") or "")
    clauses = [
        "HOST PRE-ACCEPTANCE EXECUTION GUARD: the previous proposed full-text patch was not accepted."
    ]
    if repair_feature and (failed_repair or failed_preservation):
        clauses.append(
            f"It did not make the peer-validated repair checks for {repair_feature} green "
            f"(unresolved={failed_repair}, regressed_prior_passes={failed_preservation})."
        )
    if protected:
        clauses.append(
            f"It must also preserve already merged feature(s) {protected} and their public entry points."
        )
    if failed_protected:
        clauses.append(
            f"The rejected patch regressed merged-feature checks {failed_protected}."
        )
    clauses.append(
        "Do not weaken, delete, bypass, or move the protected behavior across an "
        "incompatible operation order. Preserve the exact public check while repairing "
        "the current feature."
    )
    clauses.append(f"Fresh execution evidence:\n{detail}")
    return " ".join(clauses)[:3000]


def actor_patch_guard_failure(world: Any, actor_id: str) -> dict:
    """Return bounded evidence from this actor's latest refused proposed patch."""
    value = (world.__dict__.get(_PATCH_GUARD_FAILURES) or {}).get(actor_id)
    return copy.deepcopy(value) if isinstance(value, dict) else {}


def actor_patch_guard_failure_history(
    world: Any, actor_id: str, feature_id: str = ""
) -> list[dict]:
    """Return recent rejected repair approaches, optionally for one feature.

    The current failure is cleared after a later patch passes the execution
    guard so normal commit/test actions remain reachable.  The short history is
    deliberately separate: a still-red semantic review can remind the editor
    not to oscillate back to an already executed, rejected implementation.
    """
    values = (world.__dict__.get(_PATCH_GUARD_FAILURE_HISTORY) or {}).get(
        actor_id, []
    )
    rows = [copy.deepcopy(row) for row in values if isinstance(row, dict)]
    if feature_id:
        rows = [
            row
            for row in rows
            if feature_id in (row.get("feature_ids") or [])
        ]
    return rows


def _restore(target: Any, before: Any) -> Any:
    """Restore transaction-owned state without replacing existing repo objects."""
    if isinstance(target, dict) and isinstance(before, dict):
        for key in set(target) - set(before):
            del target[key]
        for key, value in before.items():
            target[key] = _restore(target[key], value) if key in target else copy.deepcopy(value)
        return target
    if isinstance(target, list) and isinstance(before, list):
        target[:] = copy.deepcopy(before)
        return target
    if hasattr(target, "__dict__") and hasattr(before, "__dict__"):
        _restore(target.__dict__, before.__dict__)
        return target
    return copy.deepcopy(before)


def apply_actor_product_patch(world: Any, patch: Any, result: Any, actor_id: str) -> bool:
    """Atomically accept exact actor bytes into the native pending ledger."""
    from environments.org_env.backend.repo.workflow import record_patch

    artifact = world.product_artifacts.get(patch.target_object_id)
    receipt = getattr(patch, "actor_source_snapshot", None)
    if artifact is None or not isinstance(receipt, dict):
        result.failure_reason = "actor_source_snapshot_required"
        return False
    try:
        prepared = sv.prepare_actor_patch(
            world, actor_id, patch, expected_snapshot_id=str(receipt.get("snapshot_id") or "")
        )
        before_source = sv.actor_desk_snapshot(world, actor_id)
    except sv.SourceViewError as error:
        result.failure_reason = str(error)
        return False
    from .joint_probe_replay import guard_actor_patch_against_merged_features
    guard = guard_actor_patch_against_merged_features(
        world, actor_id=actor_id, prepared_patch=prepared)
    if guard.get("applicable") is True:
        guard_name = (
            "current_feature_repair_guard"
            if guard.get("repair_feature")
            else "merged_feature_patch_guard"
        )
        repair_paths = {
            str(path).replace("\\", "/")
            for path in (guard.get("repair_paths") or [])
            if str(path)
        }
        stage_multi_file_repair = bool(
            guard_name == "current_feature_repair_guard"
            and guard.get("available") is True
            and guard.get("ok") is not True
            and guard.get("error") == "current_feature_repair_unresolved"
            and not guard.get("failed_protected_probe_ids")
            and len(repair_paths) > 1
            and prepared.file_path.replace("\\", "/") in repair_paths
        )
        result.state_delta[guard_name] = copy.deepcopy(guard)
        result.events.append({
            "type": "repo_event",
            "subtype": (
                f"{guard_name}_passed"
                if guard.get("ok") is True
                else "current_feature_repair_guard_pending"
                if stage_multi_file_repair
                else f"{guard_name}_rejected"
            ),
            "agent_id": actor_id,
            "patch_id": patch.patch_id,
            "tick": int(getattr(world, "world_tick", 0) or 0),
            "protected_features": list(guard.get("protected_features") or []),
            "repair_feature": guard.get("repair_feature"),
            "repair_probe_ids": list(guard.get("repair_probe_ids") or []),
            "preserve_probe_ids": list(guard.get("preserve_probe_ids") or []),
            "failed_protected_probe_ids": list(
                guard.get("failed_protected_probe_ids") or []
            ),
            "failed_repair_probe_ids": list(guard.get("failed_repair_probe_ids") or []),
            "failed_preservation_probe_ids": list(
                guard.get("failed_preservation_probe_ids") or []
            ),
            "incomplete_probe_ids": list(guard.get("incomplete_probe_ids") or []),
            "available": guard.get("available") is True,
            "ok": guard.get("ok") is True,
            "error": str(guard.get("error") or ""),
            "source_snapshot": copy.deepcopy(guard.get("source_snapshot") or {}),
        })
        failures = world.__dict__.setdefault(_PATCH_GUARD_FAILURES, {})
        if not isinstance(failures, dict):
            result.failure_reason = "actor_patch_guard_failure_store_invalid"
            return False
        if guard.get("available") is not True or guard.get("ok") is not True:
            failure_record = {
                "schema_version": "cooperbench_actor_patch_guard_failure_v5",
                "actor_id": actor_id,
                "patch_id": patch.patch_id,
                "tick": int(getattr(world, "world_tick", 0) or 0),
                "protected_features": list(guard.get("protected_features") or []),
                "repair_feature": guard.get("repair_feature"),
                "repair_probe_ids": list(guard.get("repair_probe_ids") or []),
                "preserve_probe_ids": list(guard.get("preserve_probe_ids") or []),
                "failed_protected_probe_ids": list(
                    guard.get("failed_protected_probe_ids") or []
                ),
                "failed_repair_probe_ids": list(guard.get("failed_repair_probe_ids") or []),
                "failed_preservation_probe_ids": list(
                    guard.get("failed_preservation_probe_ids") or []
                ),
                "incomplete_probe_ids": list(
                    guard.get("incomplete_probe_ids") or []
                ),
                "source_snapshot": copy.deepcopy(guard.get("source_snapshot") or {}),
                "brief": _patch_guard_failure_brief(guard),
                "receipt": copy.deepcopy(guard),
            }
            feature_ids = [
                str(item)
                for item in (getattr(patch, "related_issue_ids", []) or [])
                if str(item)
            ]
            failure_record["feature_ids"] = feature_ids
            failures[actor_id] = failure_record
            histories = world.__dict__.setdefault(
                _PATCH_GUARD_FAILURE_HISTORY, {}
            )
            if not isinstance(histories, dict):
                result.failure_reason = "actor_patch_guard_failure_history_store_invalid"
                return False
            history = histories.setdefault(actor_id, [])
            if not isinstance(history, list):
                result.failure_reason = "actor_patch_guard_failure_history_invalid"
                return False
            snapshot = failure_record.get("source_snapshot") or {}
            fingerprint = str(
                snapshot.get("tree_digest")
                or snapshot.get("snapshot_id")
                or patch.patch_id
            )
            historical = {
                key: copy.deepcopy(value)
                for key, value in failure_record.items()
                if key != "receipt"
            }
            historical.update({
                "schema_version": "cooperbench_actor_patch_guard_failure_history_v1",
                "fingerprint": fingerprint,
            })
            history[:] = [
                row
                for row in history
                if str((row or {}).get("fingerprint") or "") != fingerprint
            ]
            history.append(historical)
            del history[:-4]
            if not stage_multi_file_repair:
                patch.validation_status = "rejected"
                patch.rejection_reason = (
                    f"{guard_name}:"
                    + str(guard.get("error") or "evidence_unavailable")
                )[:500]
                world.patches[patch.patch_id] = patch
                result.success = False
                result.failure_reason = (
                    f"{guard_name}_rejected:"
                    + str(guard.get("error") or "evidence_unavailable")
                )
                return False
            result.state_delta["current_feature_repair_pending"] = {
                "feature_id": guard.get("repair_feature"),
                "failed_repair_probe_ids": list(
                    guard.get("failed_repair_probe_ids") or []
                ),
                "failed_preservation_probe_ids": list(
                    guard.get("failed_preservation_probe_ids") or []
                ),
            }
        else:
            failures.pop(actor_id, None)
    keys = ("_cooperbench_source_views", "_cooperbench_actor_desks", "_pending_by_branch")
    # This transaction only appends frozen source snapshots, updates one actor
    # desk, and appends one pending tuple.  Snapshot those mutable containers;
    # copying every immutable source projection is unnecessary and can fail on
    # its deliberately read-only mapping views.
    source_store = world.__dict__.get("_cooperbench_source_views")
    source_before = (
        {
            **source_store,
            "snapshots": dict(source_store["snapshots"]),
            "branch_bases": dict(source_store["branch_bases"]),
            "pr_heads": dict(source_store["pr_heads"]),
        }
        if isinstance(source_store, dict)
        else None
    )
    desks_store = world.__dict__.get("_cooperbench_actor_desks")
    desks_before = (
        {
            **desks_store,
            "actors": {
                key: {
                    **value,
                    "patch_records": tuple(value.get("patch_records", ())),
                }
                for key, value in desks_store["actors"].items()
            },
            "sync_receipts": [dict(row) for row in desks_store["sync_receipts"]],
        }
        if isinstance(desks_store, dict)
        else None
    )
    pending_before = {
        key: list(value)
        for key, value in (world.__dict__.get("_pending_by_branch", {}) or {}).items()
    }
    before = {
        "_cooperbench_source_views": source_before,
        "_cooperbench_actor_desks": desks_before,
        "_pending_by_branch": pending_before,
    }
    repo_system = world.repo_system
    # record_patch mutates only the branch registry.  Historical commits, PRs,
    # and CI evidence are read-only for this action and must not be pulled into
    # the edit rollback boundary.
    branches_before = copy.deepcopy(repo_system.repo.branches)
    seq_before = copy.deepcopy(getattr(repo_system, "_seq", None))
    artifact_before, patch_before = copy.deepcopy(artifact), copy.deepcopy(patch)
    patches_before = dict(world.patches)
    result_before = copy.deepcopy(result)
    personal_before = {
        key: list(getattr(personal, "local_branch_ids", []) or [])
        for key, personal in (getattr(world, "personal", {}) or {}).items()
    }
    try:
        tick = int(world.world_tick)
        patch.validation_status = "accepted"
        patch.applied_tick = tick
        world.patches[patch.patch_id] = patch
        # Lifecycle metadata is attributable; it is not the reader's source.
        artifact.revision += 1
        artifact.updated_at_tick = tick
        artifact.patch_history_ids.append(patch.patch_id)
        if patch.change_summary:
            artifact.change_summaries.append(patch.change_summary)
        artifact.awaiting_review = True
        artifact.status = "needs_review"
        path = prepared.file_path
        old = before_source.files.get(path, "")
        patch.unified_diff = "\n".join(difflib.unified_diff(
            old.splitlines(), prepared.new_content.splitlines(),
            fromfile=f"a/{path}", tofile=f"b/{path}", lineterm="",
        ))
        branch_id = record_patch(world, actor_id, patch, artifact, tick)
        if not branch_id:
            raise sv.SourceViewError("actor_patch_delivery_route_unavailable")
        after_source = sv.commit_actor_patch(world, prepared, branch_id=branch_id)
        result.modified_objects.append(artifact.artifact_id)
        result.state_delta.update({"patch_id": patch.patch_id,
                                   "actor_source_snapshot": after_source.receipt()})
        result.events.append({"type": "product_event", "subtype": "patch_applied",
                              "artifact_id": artifact.artifact_id, "patch_id": patch.patch_id,
                              "agent_id": actor_id, "tick": tick,
                              "change_summary": patch.change_summary,
                              "patch_type": getattr(patch, "patch_type", ""),
                              "source_snapshot": after_source.receipt()})
        result.graph_edges.append((actor_id, "patched", artifact.artifact_id))
        return True
    except BaseException as error:
        _restore(artifact, artifact_before)
        _restore(patch, patch_before)
        world.patches.clear()
        world.patches.update(patches_before)
        _restore(repo_system.repo.branches, branches_before)
        repo_system._seq = seq_before
        for key in keys:
            if key in before:
                world.__dict__[key] = _restore(world.__dict__.get(key), before[key])
            else:
                world.__dict__.pop(key, None)
        for key, branches in personal_before.items():
            personal = world.personal[key]
            if hasattr(personal, "local_branch_ids"):
                personal.local_branch_ids[:] = branches
        _restore(result, result_before)
        result.failure_reason = str(error) if isinstance(error, sv.SourceViewError) else "actor_patch_transaction_failed"
        if not isinstance(error, Exception):
            raise
        return False


def actor_public_test_record(world: Any, actor_id: str, *, require_current: bool = True) -> dict:
    record = (world.__dict__.get("_cooperbench_actor_public_tests") or {}).get(actor_id)
    if not isinstance(record, dict):
        return {}
    receipt = record.get("source_snapshot")
    if (record.get("actor_id") != actor_id or not isinstance(receipt, dict)
            or receipt.get("owner_id") != actor_id or receipt.get("source_kind") != "actor_desk"):
        return {}
    if require_current:
        current = sv.actor_desk_snapshot(world, actor_id)
        # A pending->committed transition does not alter executable bytes. The
        # snapshot receipt records attribution; test validity follows source
        # and runtime bytes rather than a second fallible materialization.
        if (
            receipt.get("tree_digest") != current.tree_digest
            or receipt.get("runtime_assets_digest")
            != current.runtime_assets_digest
        ):
            return {}
    return copy.deepcopy(record)


def run_actor_public_tests(world: Any, actor_id: str, result: Any, tick: int) -> None:
    from environments.org_env.product.materialize import _repo_hash, run_public_tests

    snapshot = sv.actor_desk_snapshot(world, actor_id)
    tree_hash = _repo_hash(world, prefer_mainline=False, **sv.snapshot_projection(world, snapshot))
    records = world.__dict__.setdefault("_cooperbench_actor_public_tests", {})
    preflight = (
        world.__dict__.get("_cooperbench_public_runtime_preflight") or {}
    )
    validation_strength = str(
        preflight.get("public_validation_strength") or "public_regression"
    )
    record = {"actor_id": actor_id, "source_snapshot": snapshot.receipt(), "repo_hash": tree_hash,
              "tick": int(tick), "status": "running", "failed_tests": [], "summary": "",
              "validation_strength": validation_strength,
              "command": list(world.__dict__.get("_cooperbench_verified_public_test_command") or []),
              "coverage_scope": "repository_smoke_not_feature_acceptance",
              "functional_public_regression_available": bool(
                  preflight.get("functional_public_regression_available", True)
              )}
    records[actor_id] = record
    outcome = run_public_tests(world, source_snapshot=snapshot)
    if sv.actor_desk_snapshot(world, actor_id).snapshot_id != snapshot.snapshot_id:
        record["status"] = "source_changed_during_tests"
        result.success = False
        result.failure_reason = "actor_source_changed_during_tests"
        return
    launched_failure = bool(outcome.get("error") and outcome.get("returncode") is not None
                            and (world.__dict__.get("_cooperbench_public_runtime_preflight") or {}).get("passed") is True)
    if not outcome.get("available") or (outcome.get("error") and not launched_failure):
        record.update(status="infrastructure_error", summary=str(outcome.get("error") or "no_public_tests_declared")[:300])
        result.success = False
        result.failure_reason = "public_tests_infrastructure_error"
    else:
        passed = bool(outcome.get("ok")) and not launched_failure
        record.update(status="passed" if passed else "failed", passed=passed,
                      summary=str(outcome.get("summary") or outcome.get("error") or "")[:300],
                      failed_tests=list(outcome.get("failed_tests") or []),
                      failure_brief=str(outcome.get("failure_brief") or outcome.get("error") or "")[:2000])
        result.success = True
        result.state_delta.update(public_tests_passed=passed, public_tests_summary=record["summary"])
        if not passed:
            result.state_delta["public_tests_failed"] = record["failed_tests"]
    result.state_delta["actor_public_test_evidence"] = copy.deepcopy(record)
    result.events.append({"type": "repo_event", "subtype": f"public_tests_{record['status']}",
                          "agent_id": actor_id, "tick": tick, "summary": record["summary"],
                          "source_snapshot": snapshot.receipt(), "repo_hash": tree_hash,
                          "failure_brief": record.get("failure_brief", ""),
                          "failed_tests": record["failed_tests"],
                          "command": record["command"], "coverage_scope": record["coverage_scope"],
                          "failed_count": len(record["failed_tests"])})


def actor_sync_needed(world: Any, actor_id: str) -> bool:
    """Only offer clean-desk synchronization, never perform it while selecting."""
    if not sv.actor_desks_enabled(world):
        return False
    source = sv.actor_desk_snapshot(world, actor_id)
    main = sv.mainline_snapshot(world)
    if source.branch_id and any(cid not in main.commit_ids for cid in source.commit_ids):
        return False
    if (world.__dict__.get("_pending_by_branch") or {}).get(source.branch_id):
        return False
    if source.branch_id and not source.commit_ids:
        return False
    known_main = tuple(dict.fromkeys((*source.base_main_commit_ids, *source.commit_ids)))
    return source.tree_digest != main.tree_digest or known_main != main.commit_ids
