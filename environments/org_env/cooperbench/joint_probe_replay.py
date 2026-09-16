"""Replay accepted public checks on an actual integration tree, without an LLM.

A feature's own PR-head review is not a regression verdict for a different
integration tree. This consumer runs the exact definitions an actual peer
approved, checks both source and plan-set identities after execution, and never
turns a newly failing check into an instruction to repair product code.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping

from . import behavior_probes as bp
from . import semantic_review as sr
from . import source_views as sv
from .semantic_review import current_semantic_approval
from .visibility import is_strict_coop


REPLAY_SCHEMA = "cooperbench_integrated_probe_replay_v1"
PATCH_GUARD_SCHEMA = "cooperbench_actor_patch_execution_guard_v2"
_RECEIPTS = "_cooperbench_accepted_probe_replay_receipts"
_SEQUENCE = "_cooperbench_accepted_probe_replay_sequence"


class ProbeReplayError(ValueError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def _status(value: Any) -> str:
    return str(getattr(value, "value", value) or "").lower()


def _feature(pr: Any, owners: Mapping[str, str]) -> str:
    linked = list(getattr(pr, "linked_issue_ids", []) or [])
    linked += [getattr(pr, "linked_issue", None)]
    features = {str(item) for item in linked if item in owners}
    if len(features) != 1:
        raise ProbeReplayError("accepted_probe_pr_feature_ambiguous")
    feature = next(iter(features))
    if getattr(pr, "author_id", None) != owners[feature]:
        raise ProbeReplayError("accepted_probe_pr_owner_mismatch")
    return feature


def _actually_merged(world: Any, pr: Any, main: Any) -> bool:
    if _status(getattr(pr, "status", "")) != "merged":
        return False
    repo = world.repo_system.repo
    commits = list(getattr(pr, "commit_ids", []) or [])
    branch = repo.branches.get(getattr(pr, "source_branch", None))
    if (not commits or any(cid not in main.commit_ids for cid in commits)
            or any(_status(getattr(repo.commits.get(cid), "status", "")) != "merged" for cid in commits)
            or branch is None or _status(getattr(branch, "status", "")) != "merged"):
        raise ProbeReplayError("accepted_probe_pr_not_in_actual_mainline")
    # The exact PR ledger must still be registered, synced and frozen, rather
    # than trusting a forged terminal status or a subset of mainline IDs.
    sv.pr_head_snapshot(world, pr)
    return True


def _eligible_prs(world: Any, actor_id: str, target: Any, main: Any,
                  require_all_features: bool) -> dict[str, Any]:
    state = getattr(world, "_cooperbench_sdl_state", {}) or {}
    owners = state.get("feature_owners") or {}
    if (not isinstance(owners, dict) or len(owners) != 2 or len(set(owners.values())) != 2
            or actor_id not in owners.values()):
        raise ProbeReplayError("accepted_probe_pair_actor_required")
    repo = world.repo_system.repo
    target_feature = _feature(target, owners) if target is not None else None
    if target is not None and _status(getattr(target, "status", "")) != "approved":
        raise ProbeReplayError("accepted_probe_current_pr_not_approved")
    merged: dict[str, list[Any]] = {}
    for pr in repo.pull_requests.values():
        if _status(getattr(pr, "status", "")) != "merged":
            continue
        feature = _feature(pr, owners)
        if _actually_merged(world, pr, main):
            merged.setdefault(feature, []).append(pr)
    required = set(owners) if require_all_features else (set(merged) | ({target_feature} if target_feature else set()))
    if not required:
        raise ProbeReplayError("accepted_probe_no_eligible_features")
    selected = {}
    for feature in sorted(required):
        # Visibility is checked before reading this feature's cached approval,
        # plan or definitions. An unmerged peer is not made eligible merely
        # because its approval exists somewhere in the shared registry.
        if bp.visible_review_contract(world, actor_id, feature) is None:
            raise ProbeReplayError("accepted_probe_brief_unshared_or_unread")
        if feature == target_feature:
            selected[feature] = target
            continue
        candidates = merged.get(feature, [])
        if not candidates:
            raise ProbeReplayError("accepted_probe_feature_not_merged")
        review = (getattr(world, "_cooperbench_semantic_reviews", {}) or {}).get(feature) or {}
        matching = [pr for pr in candidates if getattr(pr, "pr_id", None) == review.get("pr_id")]
        if len(matching) != 1:
            raise ProbeReplayError("accepted_probe_merged_approval_missing")
        selected[feature] = matching[0]
    return selected


def _accepted_plan(world: Any, actor_id: str, feature: str, pr: Any, baseline: Any):
    visible = bp.visible_review_contract(world, actor_id, feature)
    if visible is None:
        raise ProbeReplayError("accepted_probe_brief_unshared_or_unread")
    review = (getattr(world, "_cooperbench_semantic_reviews", {}) or {}).get(feature) or {}
    reviewer = str(review.get("reviewer_id") or "")
    origin = bp.visible_review_contract(world, reviewer, feature) if reviewer else None
    if origin is None or origin["description"] != visible["description"]:
        raise ProbeReplayError("accepted_probe_origin_brief_unavailable")
    if not current_semantic_approval(world, feature, pr):
        raise ProbeReplayError("accepted_probe_approval_not_current")
    saved = (getattr(world, "_cooperbench_behavior_plans", {}) or {}).get(feature)
    if not isinstance(saved, Mapping):
        raise ProbeReplayError("accepted_probe_plan_missing")
    paths = saved.get("paths")
    if (not isinstance(paths, list) or not paths or not all(isinstance(path, str) and path for path in paths)
            or saved.get("reviewer_id") != reviewer
            or any(saved.get(key) != origin[key] or review.get(key) != origin[key]
                   for key in ("brief_visibility_receipt", "review_brief_identity"))):
        raise ProbeReplayError("accepted_probe_plan_visibility_mismatch")
    integration_contract = saved.get("integration_contract") or {}
    if not isinstance(integration_contract, Mapping):
        raise ProbeReplayError("accepted_probe_plan_integration_contract_invalid")
    expected = bp._plan_identity(origin["description"], origin["requirements"], origin["compatibility"],
                                 paths, dict(baseline.files), reviewer, origin,
                                 integration_contract)
    if saved.get("identity") != expected:
        raise ProbeReplayError("accepted_probe_plan_identity_mismatch")
    try:
        probes = bp._validate_plan(saved, feature_id=feature, requirements=origin["requirements"],
                                   compatibility=origin["compatibility"], paths=paths,
                                   integration_contract=integration_contract,
                                   _preserve_ids=True)
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        raise ProbeReplayError("accepted_probe_plan_invalid") from exc
    if probes != saved.get("probes"):
        raise ProbeReplayError("accepted_probe_plan_not_canonical")
    behavior = review.get("behavior_evidence")
    if (not isinstance(behavior, Mapping) or behavior.get("available") is not True
            or behavior.get("ok") is not True or behavior.get("error")
            or (behavior.get("integration_contract") or {}) != integration_contract
            or behavior.get("source_snapshot") != review.get("source_snapshot")
            or behavior.get("baseline_snapshot") != baseline.receipt()):
        raise ProbeReplayError("accepted_probe_approval_execution_missing")
    approved_probes = behavior.get("probes")
    approved_results = behavior.get("results")
    if (not isinstance(approved_probes, list) or not all(isinstance(row, Mapping) for row in approved_probes)
            or not isinstance(approved_results, list) or not all(isinstance(row, Mapping) for row in approved_results)):
        raise ProbeReplayError("accepted_probe_approval_definitions_missing")
    approved_ids = [row.get("probe_id") for row in approved_probes]
    if (not all(isinstance(pid, str) for pid in approved_ids)
            or len(set(approved_ids)) != len(approved_ids)
            or behavior.get("plan_hash") != bp._digest(approved_probes)):
        raise ProbeReplayError("accepted_probe_approval_definitions_mismatch")
    own_approved = [dict(row) for row in approved_probes if str(row.get("probe_id") or "").split(":", 1)[0] == feature]
    if own_approved != probes:
        raise ProbeReplayError("accepted_probe_approval_definitions_mismatch")
    result_ids = [row.get("probe_id") for row in approved_results]
    if not all(isinstance(pid, str) for pid in result_ids) or len(set(result_ids)) != len(result_ids):
        raise ProbeReplayError("accepted_probe_approval_execution_missing")
    by_id = {row.get("probe_id"): row for row in approved_results}
    for probe in probes:
        row = by_id.get(probe["probe_id"]) or {}
        if ((row.get("candidate") or {}).get("status") != "pass"
                or (probe["compare_baseline"] and (row.get("baseline") or {}).get("status") != "pass")):
            raise ProbeReplayError("accepted_probe_approval_execution_missing")
    metadata = {
        "feature_id": feature, "origin_reviewer": reviewer, "plan_identity": expected,
        "approved_pr_id": pr.pr_id, "approved_source_snapshot": deepcopy(review["source_snapshot"]),
        "review_brief_identity": origin["review_brief_identity"],
        "brief_visibility_receipt": deepcopy(origin["brief_visibility_receipt"]),
        "probe_ids": [row["probe_id"] for row in probes], "definitions_digest": _digest(probes),
        "integration_contract": deepcopy(dict(integration_contract)),
    }
    return metadata, deepcopy(probes), visible["review_brief_identity"]


@dataclass(frozen=True)
class _PreparedReplay:
    binding: dict
    source: Any
    baseline: Any
    probes: list[dict]
    approved_prs: tuple[tuple[str, Any], ...]


@dataclass(frozen=True)
class _PreparedPatchGuard:
    binding: dict
    source: Any
    baseline: Any
    probes: list[dict]
    approved_prs: tuple[tuple[str, Any], ...]


@dataclass(frozen=True)
class _SkippedPatchGuard:
    classification: str


def _actor_desk_descends_from_reviewed_head(before: Any, receipt: Mapping[str, Any]) -> bool:
    """Whether the owner's mutable desk is the rejected head or its pending descendant.

    ``base_snapshot_id`` differs legitimately between an actor desk and its
    committed-PR projection.  The first repair candidate is therefore bound by
    equal tree and ledger identity.  Once one repair is accepted but not yet
    committed, subsequent pending patches remain descendants of that exact
    reviewed head and must keep the same executable debt green too.
    """

    reviewed_commits = tuple(receipt.get("commit_ids") or ())
    reviewed_patches = tuple(receipt.get("patch_ids") or ())
    return bool(
        receipt.get("source_kind") == "committed_pr_head"
        and receipt.get("owner_id") == before.owner_id
        and receipt.get("branch_id") == before.branch_id
        and reviewed_commits == before.commit_ids
        and reviewed_patches == before.patch_ids[:len(reviewed_patches)]
        and (
            reviewed_patches != before.patch_ids
            or receipt.get("tree_digest") == before.tree_digest
        )
    )


def _current_feature_repair_plan(world: Any, actor_id: str, before: Any,
                                 baseline: Any) -> dict | None:
    """Return peer-observed checks that a repair candidate must make green.

    A rejected semantic review is not by itself executable authority.  The
    failed row must have a current focused ``valid`` receipt, the whole plan
    must still be the canonical actual-peer plan for the visible public brief,
    and the review must name the exact actor desk being edited.  Checks that
    passed on that same head are replayed too, so fixing one clause cannot
    silently regress another clause of the current feature.
    """

    state = getattr(world, "_cooperbench_sdl_state", {}) or {}
    owners = state.get("feature_owners") or {}
    owned = [feature for feature, owner in owners.items() if owner == actor_id]
    if len(owned) != 1:
        raise ProbeReplayError("actor_patch_guard_owner_feature_ambiguous")
    feature = str(owned[0])
    reviews = getattr(world, "_cooperbench_semantic_reviews", {}) or {}
    review = reviews.get(feature)
    if not isinstance(review, Mapping) or review.get("approved") is not False:
        return None
    reviewer = str(review.get("reviewer_id") or "")
    reviewed_source = review.get("source_snapshot")
    # A stale rejection belongs to an older head and must neither authorize nor
    # block an unrelated new edit.
    if (not reviewer or not isinstance(reviewed_source, Mapping)
            or not _actor_desk_descends_from_reviewed_head(before, reviewed_source)):
        return None
    if not sr.semantic_review_repair_evidence_matches(
        world, reviewer, feature, review
    ):
        raise ProbeReplayError("actor_patch_guard_current_failure_evidence_invalid")
    contract = bp.visible_review_contract(world, reviewer, feature)
    saved = (getattr(world, "_cooperbench_behavior_plans", {}) or {}).get(feature)
    pending_defect_ids: set[str] = set()
    if not isinstance(saved, Mapping):
        gap = (
            getattr(world, "_cooperbench_probe_evidence_gaps", {}) or {}
        ).get(feature)
        invalidated = gap.get("invalidated_plan") if isinstance(gap, Mapping) else None
        repair_ids = {
            str(item)
            for item in (gap.get("repair_probe_ids") or [])
            if str(item)
        } if isinstance(gap, Mapping) else set()
        confirmed_defects = {
            str(row.get("probe_id") or "")
            for row in (gap.get("defects") or [])
            if isinstance(row, Mapping)
            and row.get("adjudication_status") == "confirmed"
        } if isinstance(gap, Mapping) else set()
        if (
            isinstance(invalidated, Mapping)
            and repair_ids
            and repair_ids == confirmed_defects
            and gap.get("identity") == invalidated.get("identity")
            and gap.get("previous_probes") == invalidated.get("probes")
            and gap.get("retained_probes") == [
                row for row in (invalidated.get("probes") or [])
                if row.get("probe_id") not in repair_ids
            ]
        ):
            saved = invalidated
            pending_defect_ids = repair_ids
    if contract is None or not isinstance(saved, Mapping):
        raise ProbeReplayError("actor_patch_guard_current_plan_missing")
    paths = saved.get("paths")
    if (not isinstance(paths, list) or not paths
            or not all(isinstance(path, str) and path for path in paths)
            or saved.get("reviewer_id") != reviewer
            or any(saved.get(key) != contract[key] or review.get(key) != contract[key]
                   for key in ("brief_visibility_receipt", "review_brief_identity"))):
        raise ProbeReplayError("actor_patch_guard_current_plan_visibility_mismatch")
    if not isinstance(saved.get("integration_contract") or {}, Mapping):
        raise ProbeReplayError("actor_patch_guard_current_integration_contract_invalid")
    expected = bp._plan_identity(
        contract["description"], contract["requirements"], contract["compatibility"],
        paths, dict(baseline.files), reviewer, contract,
        saved.get("integration_contract") or {},
    )
    if saved.get("identity") != expected:
        raise ProbeReplayError("actor_patch_guard_current_plan_identity_mismatch")
    try:
        canonical = bp._validate_plan(
            saved, feature_id=feature, requirements=contract["requirements"],
            compatibility=contract["compatibility"], paths=paths,
            integration_contract=saved.get("integration_contract") or {},
            _preserve_ids=True,
        )
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        raise ProbeReplayError("actor_patch_guard_current_plan_invalid") from exc
    if canonical != saved.get("probes"):
        raise ProbeReplayError("actor_patch_guard_current_plan_not_canonical")

    behavior = review.get("behavior_evidence")
    if (not isinstance(behavior, Mapping) or behavior.get("available") is not True
            or (behavior.get("integration_contract") or {})
            != (saved.get("integration_contract") or {})
            or behavior.get("source_snapshot") != reviewed_source
            or behavior.get("baseline_snapshot") != baseline.receipt()):
        raise ProbeReplayError("actor_patch_guard_current_execution_missing")
    observed_probes = behavior.get("probes")
    observed_results = behavior.get("results")
    if (not isinstance(observed_probes, list)
            or not all(isinstance(row, Mapping) for row in observed_probes)
            or not isinstance(observed_results, list)
            or not all(isinstance(row, Mapping) for row in observed_results)
            or behavior.get("plan_hash") != bp._digest(observed_probes)):
        raise ProbeReplayError("actor_patch_guard_current_execution_mismatch")
    own_observed = [dict(row) for row in observed_probes
                    if str(row.get("probe_id") or "").split(":", 1)[0] == feature]
    if own_observed != canonical:
        raise ProbeReplayError("actor_patch_guard_current_execution_mismatch")
    result_by_id = {str(row.get("probe_id") or ""): row for row in observed_results}
    if len(result_by_id) != len(observed_results):
        raise ProbeReplayError("actor_patch_guard_current_execution_mismatch")
    audit = review.get("probe_validity") or {}
    recorded_rows = audit.get("receipts") if isinstance(audit, Mapping) else None
    if not isinstance(recorded_rows, list) or not all(
        isinstance(row, Mapping) for row in recorded_rows
    ):
        raise ProbeReplayError("actor_patch_guard_current_validity_missing")
    recorded = {str(row.get("probe_id") or ""): row for row in recorded_rows}
    if len(recorded) != len(recorded_rows):
        raise ProbeReplayError("actor_patch_guard_current_validity_mismatch")

    selected: list[dict] = []
    preserve_ids: list[str] = []
    repair_ids: list[str] = []
    validity: list[dict] = []
    for probe in canonical:
        probe_id = str(probe["probe_id"])
        result = result_by_id.get(probe_id)
        if not isinstance(result, Mapping):
            raise ProbeReplayError("actor_patch_guard_current_execution_missing")
        status = (result.get("candidate") or {}).get("status")
        if probe_id in pending_defect_ids:
            material = sr._probe_validity_material(
                world, reviewer_id=reviewer, feature_id=feature,
                description=contract["description"],
                obligations=list(contract["requirements"].values()),
                probe=probe, result=result,
            )
            trusted = sr._cached_probe_validity(world, material)
            row = recorded.get(probe_id)
            matches = bool(trusted is not None and row is not None and all(
                row.get(key) == value
                for key, value in trusted.items()
                if key != "cached"
            ))
            if status != "fail" or not matches or trusted.get("verdict") != "defective":
                raise ProbeReplayError("actor_patch_guard_current_validity_mismatch")
            continue
        if status == "pass":
            if (probe.get("compare_baseline")
                    and (result.get("baseline") or {}).get("status") != "pass"):
                raise ProbeReplayError("actor_patch_guard_current_execution_incomplete")
            selected.append(deepcopy(probe))
            preserve_ids.append(probe_id)
            continue
        if status != "fail":
            raise ProbeReplayError("actor_patch_guard_current_execution_incomplete")
        material = sr._probe_validity_material(
            world, reviewer_id=reviewer, feature_id=feature,
            description=contract["description"],
            obligations=list(contract["requirements"].values()),
            probe=probe, result=result,
        )
        trusted = sr._cached_probe_validity(world, material)
        row = recorded.get(probe_id)
        matches = bool(trusted is not None and row is not None and all(
            row.get(key) == value for key, value in trusted.items() if key != "cached"
        ))
        if matches and trusted.get("verdict") == "valid":
            selected.append(deepcopy(probe))
            repair_ids.append(probe_id)
            validity.append(deepcopy(dict(row)))
    if not repair_ids:
        # Source-only findings still route to the editor, but there is no
        # executable current-feature assertion for this gate to claim.
        return None
    metadata = {
        "feature_id": feature,
        "owner_id": actor_id,
        "paths": list(paths),
        "origin_reviewer": reviewer,
        "reviewed_pr_id": review.get("pr_id"),
        "reviewed_source_snapshot": deepcopy(dict(reviewed_source)),
        "reviewed_pr_revision": deepcopy(review.get("reviewed_pr_revision") or {}),
        "plan_identity": expected,
        "review_brief_identity": contract["review_brief_identity"],
        "brief_visibility_receipt": deepcopy(contract["brief_visibility_receipt"]),
        "preserve_probe_ids": preserve_ids,
        "repair_probe_ids": repair_ids,
        "probe_ids": [row["probe_id"] for row in selected],
        "definitions_digest": _digest(selected),
        "validity_receipts": validity,
    }
    return {"metadata": metadata, "probes": selected}


def _prepare(world: Any, actor_id: str, pull_request: Any, source_snapshot: Any,
             require_all_features: bool) -> _PreparedReplay:
    if not sv.source_views_enabled(world) or not is_strict_coop(world):
        raise ProbeReplayError("integrated_probe_strict_source_views_required")
    baseline = sv.freeze_baseline(world)
    main = sv.mainline_snapshot(world)
    if pull_request is None:
        if not require_all_features:
            raise ProbeReplayError("joint_probe_requires_all_features")
        source, stage = main, "joint_mainline"
    else:
        source, stage = sv.conservative_merge_candidate(world, pull_request), "pr_integration"
    if source_snapshot is not None and source != source_snapshot:
        raise ProbeReplayError("accepted_probe_source_snapshot_mismatch")
    selected = _eligible_prs(world, actor_id, pull_request, main, require_all_features)
    plans, probes, readers = [], [], {}
    for feature, pr in selected.items():
        metadata, definitions, reader_identity = _accepted_plan(world, actor_id, feature, pr, baseline)
        plans.append(metadata)
        probes.extend(definitions)
        readers[feature] = reader_identity
    if not probes or len({row["probe_id"] for row in probes}) != len(probes):
        raise ProbeReplayError("accepted_probe_definitions_empty_or_ambiguous")
    binding = {
        "schema_version": REPLAY_SCHEMA, "stage": stage, "actor_id": actor_id,
        "source_snapshot": source.receipt(), "baseline_snapshot": baseline.receipt(),
        "integration_pr_id": getattr(pull_request, "pr_id", None),
        "integration_owner_id": getattr(pull_request, "author_id", None),
        "require_all_features": bool(require_all_features), "eligible_features": list(selected),
        "plans": plans, "plan_set_digest": _digest({"plans": plans, "probes": probes}),
        "reader_brief_identities": readers,
    }
    return _PreparedReplay(binding, source, baseline, probes, tuple(selected.items()))


def _prepare_actor_patch_guard(world: Any, actor_id: str, prepared_patch: Any,
                               source_snapshot: Any = None) -> _PreparedPatchGuard | _SkippedPatchGuard:
    """Bind a private patch to merged features and its validated repair debt.

    Peer-approved green plans protect already-merged behavior.  Separately, an
    exact rejected PR head can contribute its passing rows plus focused-valid
    failing rows: those are the executable obligations the proposed repair is
    supposed to preserve and fix.  Neither source is inferred from prose.
    """
    if not sv.source_views_enabled(world) or not is_strict_coop(world):
        raise ProbeReplayError("actor_patch_guard_strict_source_views_required")
    if not isinstance(prepared_patch, sv.PreparedActorPatch):
        raise ProbeReplayError("actor_patch_guard_ticket_invalid")
    if prepared_patch.actor_id != actor_id:
        raise ProbeReplayError("actor_patch_guard_actor_mismatch")
    probe_capability = getattr(world, "__dict__", {}).get(
        "_cooperbench_behavior_probe_capability"
    )
    if (
        isinstance(probe_capability, Mapping)
        and probe_capability.get("available") is False
    ):
        return _SkippedPatchGuard(
            "not_applicable_executable_probe_capability_unavailable"
        )
    repo = world.repo_system.repo
    has_merged = any(_status(getattr(pr, "status", "")) == "merged"
                     for pr in repo.pull_requests.values())
    state = getattr(world, "_cooperbench_sdl_state", {}) or {}
    owners = state.get("feature_owners") or {}
    owned = [feature for feature, owner in owners.items() if owner == actor_id]
    reviews = getattr(world, "_cooperbench_semantic_reviews", {}) or {}
    has_rejected_current = bool(
        len(owned) == 1
        and isinstance(reviews.get(owned[0]), Mapping)
        and reviews[owned[0]].get("approved") is False
    )
    # Preserve the old observational no-op: an ordinary first patch with no
    # peer rejection must not even create source snapshot audit entries.
    if not has_merged and not has_rejected_current:
        return _SkippedPatchGuard("not_applicable_no_merged_feature")
    baseline = sv.freeze_baseline(world)
    main = sv.mainline_snapshot(world)
    before = sv.actor_desk_snapshot(world, actor_id)
    if before.snapshot_id != prepared_patch.before_snapshot_id:
        raise ProbeReplayError("actor_patch_guard_actor_desk_changed")
    selected: dict[str, Any] = {}
    deferred_merged = False
    if before.base_main_commit_ids != main.commit_ids:
        # A dirty branch may legitimately have forked before a peer landed.
        # Its eventual conservative integration replay owns that case; running
        # peer probes directly on this intentionally old private tree would
        # manufacture a regression before the peer bytes are even present.
        # Mainline history is append-only, so any non-prefix identity is still
        # corruption and must fail closed.
        if tuple(main.commit_ids[:len(before.base_main_commit_ids)]) != before.base_main_commit_ids:
            raise ProbeReplayError("actor_patch_guard_actor_base_not_mainline_prefix")
        deferred_merged = has_merged
    elif has_merged:
        selected = _eligible_prs(world, actor_id, None, main, False)
    repair = _current_feature_repair_plan(world, actor_id, before, baseline)
    if not selected and repair is None:
        if deferred_merged:
            return _SkippedPatchGuard("not_applicable_actor_base_precedes_mainline")
        return _SkippedPatchGuard("not_applicable_no_merged_or_current_repair_probes")
    source = sv.actor_patch_candidate_snapshot(world, prepared_patch)
    if source_snapshot is not None and source != source_snapshot:
        raise ProbeReplayError("actor_patch_guard_source_snapshot_mismatch")
    plans, probes, readers = [], [], {}
    for feature, pr in selected.items():
        metadata, definitions, reader_identity = _accepted_plan(
            world, actor_id, feature, pr, baseline)
        plans.append(metadata)
        probes.extend(definitions)
        readers[feature] = reader_identity
    repair_metadata = None
    if repair is not None:
        repair_metadata = repair["metadata"]
        probes.extend(repair["probes"])
        readers[repair_metadata["feature_id"]] = repair_metadata["review_brief_identity"]
    if not probes or len({row["probe_id"] for row in probes}) != len(probes):
        raise ProbeReplayError("actor_patch_guard_definitions_empty_or_ambiguous")
    binding = {
        "schema_version": PATCH_GUARD_SCHEMA,
        "stage": "actor_patch_preaccept",
        "applicable": True,
        "actor_id": actor_id,
        "patch_id": prepared_patch.patch_id,
        "before_snapshot": before.receipt(),
        "source_snapshot": source.receipt(),
        "baseline_snapshot": baseline.receipt(),
        "mainline_snapshot": main.receipt(),
        "protected_features": list(selected),
        "deferred_merged_features": bool(deferred_merged),
        "plans": plans,
        "repair_plan": deepcopy(repair_metadata),
        "repair_feature": repair_metadata["feature_id"] if repair_metadata else None,
        "repair_paths": list(repair_metadata["paths"] if repair_metadata else []),
        "repair_probe_ids": list(repair_metadata["repair_probe_ids"] if repair_metadata else []),
        "preserve_probe_ids": list(repair_metadata["preserve_probe_ids"] if repair_metadata else []),
        "plan_set_digest": _digest({"plans": plans, "probes": probes}),
        "reader_brief_identities": readers,
    }
    return _PreparedPatchGuard(
        binding, source, baseline, probes, tuple(selected.items()))


def _patch_guard_failure(actor_id: str, error: Exception, *, binding: dict | None = None,
                         behavior: dict | None = None) -> dict:
    code = getattr(error, "code", "actor_patch_guard_execution_unavailable:" + type(error).__name__)
    return {
        **(binding or {"schema_version": PATCH_GUARD_SCHEMA, "stage": "actor_patch_preaccept",
                       "applicable": True, "actor_id": actor_id}),
        "available": False,
        "ok": False,
        "error": str(code),
        "classification": "evidence_unavailable",
        "behavior_evidence": behavior or {},
        "repair_paths": list((binding or {}).get("repair_paths") or []),
    }


def _failure(actor_id: str, error: Exception, *, binding: dict | None = None,
             behavior: dict | None = None) -> dict:
    code = getattr(error, "code", "accepted_probe_execution_unavailable:" + type(error).__name__)
    conflict = code == "source_merge_conflict"
    return {**(binding or {"schema_version": REPLAY_SCHEMA, "actor_id": actor_id}),
            "available": False, "ok": False, "error": str(code),
            "classification": "coordination_conflict" if conflict else "evidence_unavailable",
            "conflict_paths": list(getattr(error, "conflict_paths", ())) if conflict else [],
            "behavior_evidence": behavior or {}, "repair_paths": []}


def replay_accepted_probe_plans(world: Any, *, actor_id: str, pull_request: Any = None,
                                source_snapshot: Any = None, require_all_features: bool = False) -> dict:
    """Execute every eligible accepted definition afresh; never call a provider."""
    try:
        prepared = _prepare(world, actor_id, pull_request, source_snapshot, require_all_features)
    except (ProbeReplayError, sv.SourceViewError, ValueError, TypeError, KeyError, AttributeError) as exc:
        return _failure(actor_id, exc)
    store = world.__dict__.setdefault(_RECEIPTS, {})
    if not isinstance(store, dict):
        return _failure(actor_id, ProbeReplayError("accepted_probe_receipt_store_invalid"), binding=prepared.binding)
    sequence = world.__dict__.get(_SEQUENCE, 0)
    if type(sequence) is not int or sequence < 0:
        return _failure(actor_id, ProbeReplayError("accepted_probe_execution_sequence_invalid"), binding=prepared.binding)
    sequence += 1
    world.__dict__[_SEQUENCE] = sequence
    executed_at_tick = int(getattr(world, "world_tick", 0) or 0)
    try:
        evidence = bp._execute_integrated_plan(
            world, prepared.probes, source_snapshot=prepared.source, baseline_snapshot=prepared.baseline)
    except Exception as exc:
        return _failure(actor_id, exc, binding=prepared.binding)
    if not isinstance(evidence, Mapping):
        return _failure(actor_id, ProbeReplayError("accepted_probe_execution_receipt_mismatch"), binding=prepared.binding)
    behavior = {**evidence, "probes": deepcopy(prepared.probes)}
    try:
        current = _prepare(world, actor_id, pull_request, None, require_all_features)
        if (current.binding != prepared.binding
                or any(dict(current.approved_prs).get(feature) is not pr for feature, pr in prepared.approved_prs)):
            raise ProbeReplayError("accepted_probe_source_or_plans_changed_during_execution")
    except (ProbeReplayError, sv.SourceViewError, ValueError, TypeError, KeyError, AttributeError) as exc:
        return _failure(actor_id, exc, binding=prepared.binding, behavior=behavior)
    rows = behavior.get("results")
    ids = [probe["probe_id"] for probe in prepared.probes]
    if (not isinstance(rows, list) or not all(isinstance(row, Mapping) for row in rows)
            or [row.get("probe_id") for row in rows] != ids
            or behavior.get("plan_hash") != bp._digest(prepared.probes)
            or behavior.get("source_snapshot") != prepared.source.receipt()
            or behavior.get("baseline_snapshot") != prepared.baseline.receipt()):
        return _failure(actor_id, ProbeReplayError("accepted_probe_execution_receipt_mismatch"),
                        binding=prepared.binding, behavior=behavior)
    for probe, row in zip(prepared.probes, rows):
        if ((row.get("candidate") or {}).get("status") not in {"pass", "fail"}
                or (probe["compare_baseline"] and (row.get("baseline") or {}).get("status") != "pass")):
            return _failure(actor_id, ProbeReplayError("accepted_probe_execution_incomplete"),
                            binding=prepared.binding, behavior=behavior)
    passed = all(row["candidate"]["status"] == "pass" for row in rows)
    result = {**prepared.binding, "available": True, "ok": passed,
              "execution_sequence": sequence, "executed_at_tick": executed_at_tick,
              "classification": "passed" if passed else "unadjudicated_probe_failure",
              "error": "" if passed else "accepted_probe_regression_unadjudicated",
              "behavior_evidence": behavior, "repair_paths": []}
    result["replay_id"] = "probe_replay_" + _digest(result)
    store[result["replay_id"]] = _digest(result)
    return result


def guard_actor_patch_against_merged_features(world: Any, *, actor_id: str,
                                              prepared_patch: Any) -> dict:
    """Execute trusted merged and current-repair probes on a private patch.

    This gate never calls a provider and never interprets a failure as an owner
    repair diagnosis.  Its current-feature definitions come only from the
    actual peer's already-adjudicated review.  A failed or unavailable replay
    simply prevents the proposed full-text patch from entering actor/native
    ledgers.
    """
    try:
        prepared = _prepare_actor_patch_guard(world, actor_id, prepared_patch)
    except (ProbeReplayError, sv.SourceViewError, ValueError, TypeError,
            KeyError, AttributeError) as exc:
        return _patch_guard_failure(actor_id, exc)
    if isinstance(prepared, _SkippedPatchGuard):
        return {
            "schema_version": PATCH_GUARD_SCHEMA,
            "stage": "actor_patch_preaccept",
            "applicable": False,
            "actor_id": actor_id,
            "patch_id": getattr(prepared_patch, "patch_id", None),
            "available": True,
            "ok": True,
            "error": "",
            "classification": prepared.classification,
            "protected_features": [],
            "repair_feature": None,
            "repair_probe_ids": [],
            "preserve_probe_ids": [],
            "behavior_evidence": {},
            "repair_paths": [],
        }
    try:
        evidence = bp._execute_actor_patch_candidate_plan(
            world,
            prepared.probes,
            source_snapshot=prepared.source,
            baseline_snapshot=prepared.baseline,
        )
    except Exception as exc:
        return _patch_guard_failure(actor_id, exc, binding=prepared.binding)
    if not isinstance(evidence, Mapping):
        return _patch_guard_failure(
            actor_id,
            ProbeReplayError("actor_patch_guard_execution_receipt_mismatch"),
            binding=prepared.binding,
        )
    behavior = {**evidence, "probes": deepcopy(prepared.probes)}
    try:
        current = _prepare_actor_patch_guard(
            world, actor_id, prepared_patch, source_snapshot=prepared.source)
        if (isinstance(current, _SkippedPatchGuard) or current.binding != prepared.binding
                or any(dict(current.approved_prs).get(feature) is not pr
                       for feature, pr in prepared.approved_prs)):
            raise ProbeReplayError("actor_patch_guard_source_or_plans_changed_during_execution")
    except (ProbeReplayError, sv.SourceViewError, ValueError, TypeError,
            KeyError, AttributeError) as exc:
        return _patch_guard_failure(
            actor_id, exc, binding=prepared.binding, behavior=behavior)
    rows = behavior.get("results")
    ids = [probe["probe_id"] for probe in prepared.probes]
    if (not isinstance(rows, list) or not all(isinstance(row, Mapping) for row in rows)
            or [row.get("probe_id") for row in rows] != ids
            or behavior.get("plan_hash") != bp._digest(prepared.probes)
            or behavior.get("source_snapshot") != prepared.source.receipt()
            or behavior.get("baseline_snapshot") != prepared.baseline.receipt()):
        return _patch_guard_failure(
            actor_id,
            ProbeReplayError("actor_patch_guard_execution_receipt_mismatch"),
            binding=prepared.binding,
            behavior=behavior,
        )
    incomplete_ids = [
        str(row.get("probe_id") or "")
        for probe, row in zip(prepared.probes, rows)
        if ((row.get("candidate") or {}).get("status") not in {"pass", "fail"}
            or (probe["compare_baseline"]
                and (row.get("baseline") or {}).get("status") != "pass"))
    ]
    if incomplete_ids:
        failure = _patch_guard_failure(
            actor_id,
            ProbeReplayError("actor_patch_guard_execution_incomplete"),
            binding=prepared.binding,
            behavior=behavior,
        )
        failure["incomplete_probe_ids"] = incomplete_ids
        return failure
    failed_ids = {
        str(row.get("probe_id") or "")
        for row in rows
        if (row.get("candidate") or {}).get("status") == "fail"
    }
    protected_ids = {
        str(probe_id)
        for plan in prepared.binding.get("plans") or []
        for probe_id in plan.get("probe_ids") or []
    }
    repair_ids = set(prepared.binding.get("repair_probe_ids") or [])
    preserve_ids = set(prepared.binding.get("preserve_probe_ids") or [])
    failed_protected = sorted(failed_ids & protected_ids)
    failed_repair = sorted(failed_ids & repair_ids)
    failed_preservation = sorted(failed_ids & preserve_ids)
    passed = not failed_ids
    if failed_protected:
        error = "merged_feature_probe_regression"
        classification = "rejected_merged_feature_regression"
    elif failed_repair or failed_preservation:
        error = "current_feature_repair_unresolved"
        classification = "rejected_unresolved_current_feature_repair"
    else:
        error = ""
        classification = "passed"
    result = {
        **prepared.binding,
        "available": True,
        "ok": passed,
        "error": error,
        "classification": classification,
        "failed_protected_probe_ids": failed_protected,
        "failed_repair_probe_ids": failed_repair,
        "failed_preservation_probe_ids": failed_preservation,
        "behavior_evidence": behavior,
        "repair_paths": list(prepared.binding.get("repair_paths") or []),
    }
    result["replay_id"] = "patch_guard_" + _digest(result)
    store = world.__dict__.setdefault(_RECEIPTS, {})
    if not isinstance(store, dict):
        return _patch_guard_failure(
            actor_id,
            ProbeReplayError("accepted_probe_receipt_store_invalid"),
            binding=prepared.binding,
            behavior=behavior,
        )
    store[result["replay_id"]] = _digest(result)
    return result


def current_accepted_probe_replay_matches(world: Any, *, actor_id: str, receipt: Mapping[str, Any],
                                         pull_request: Any = None, require_all_features: bool = False,
                                         require_passed: bool = True) -> bool:
    """Validate an issued receipt against today's exact tree and approved plans."""
    if (not isinstance(receipt, Mapping) or receipt.get("available") is not True
            or (require_passed and receipt.get("ok") is not True)):
        return False
    try:
        trusted = (world.__dict__.get(_RECEIPTS) or {}).get(receipt.get("replay_id"))
        if trusted != _digest(dict(receipt)):
            return False
        prepared = _prepare(world, actor_id, pull_request, None, require_all_features)
        return all(receipt.get(key) == value for key, value in prepared.binding.items())
    except (ProbeReplayError, sv.SourceViewError, ValueError, TypeError, KeyError, AttributeError):
        return False


__all__ = [
    "REPLAY_SCHEMA",
    "PATCH_GUARD_SCHEMA",
    "replay_accepted_probe_plans",
    "guard_actor_patch_against_merged_features",
    "current_accepted_probe_replay_matches",
]
