"""Actual-peer adjudication and repair debt for issued integration replays.

An executed failed check is an observation, not an owner repair instruction.
This module neither runs CI nor changes source, accepted plans, approvals or
merged PR status. Full evidence remains in the internal sealed ledger; query
helpers expose bounded, source-bound feedback rather than raw probe programs.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from typing import Any, Mapping

from . import behavior_probes as bp, joint_probe_replay as replay_api, semantic_review as sr
from . import source_views as sv


ADJUDICATION_SCHEMA = "cooperbench_replay_adjudication_v1"
_STORE = "_cooperbench_replay_adjudications"


class ReplayAdjudicationError(ValueError):
    def __init__(self, code: str):
        self.code = code
        super().__init__("cooperbench_replay_adjudication:" + code)


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def _safe(value: Any, limit: int = 2400) -> str:
    from environments.org_env.programbench.public_evidence import redact_public_text
    return redact_public_text(str(value or ""), max_chars=limit)


def _public_projection(value):
    if isinstance(value, str):
        return _safe(value, 1600)
    if isinstance(value, Mapping):
        return {str(key): _public_projection(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_public_projection(item) for item in value[:64]]
    return value


def _store(world, *, create=False):
    data = world.__dict__.get(_STORE)
    if _STORE not in world.__dict__ and create:
        data = {"schema_version": ADJUDICATION_SCHEMA, "records": {}, "seals": {}}
        world.__dict__[_STORE] = data
    if _STORE not in world.__dict__:
        return {"records": {}, "seals": {}}
    if (not isinstance(data, dict) or data.get("schema_version") != ADJUDICATION_SCHEMA
            or not isinstance(data.get("records"), dict) or not isinstance(data.get("seals"), dict)):
        raise ReplayAdjudicationError("debt_store_invalid")
    return data


def _save(world, row):
    data = _store(world, create=True)
    saved = deepcopy(row)
    data["records"][row["failure_id"]] = saved
    data["seals"][row["failure_id"]] = _digest(saved)


def _load(world, failure_id):
    data = _store(world)
    row = data["records"].get(failure_id)
    if not isinstance(row, dict) or data["seals"].get(failure_id) != _digest(row):
        raise ReplayAdjudicationError("debt_missing_or_modified")
    return deepcopy(row)


def _pr_for_replay(world, replay):
    pr_id = replay.get("integration_pr_id")
    if pr_id is None:
        return None
    repo = getattr(getattr(world, "repo_system", None), "repo", None)
    pr = (getattr(repo, "pull_requests", {}) or {}).get(pr_id)
    if pr is None or getattr(pr, "pr_id", None) != pr_id:
        raise ReplayAdjudicationError("integration_pr_not_registered")
    return pr


def _replay_current(world, replay, *, passed=False):
    try:
        return replay_api.current_accepted_probe_replay_matches(
            world, actor_id=str(replay.get("actor_id") or ""), receipt=replay,
            pull_request=_pr_for_replay(world, replay),
            require_all_features=replay.get("require_all_features") is True,
            require_passed=passed,
        )
    except (ValueError, TypeError, KeyError, AttributeError):
        return False


def _source_current(world, row):
    try:
        pr = _pr_for_replay(world, row["replay"])
        snapshot = sv.conservative_merge_candidate(world, pr) if pr is not None else sv.mainline_snapshot(world)
        return snapshot.receipt() == row["source_snapshot"]
    except (ValueError, TypeError, KeyError, AttributeError):
        return False


def _pair(world):
    owners = (getattr(world, "_cooperbench_sdl_state", {}) or {}).get("feature_owners")
    if (not isinstance(owners, dict) or len(owners) != 2 or len(set(owners.values())) != 2
            or any(not isinstance(key, str) or not isinstance(value, str) or not value for key, value in owners.items())):
        raise ReplayAdjudicationError("actual_pair_required")
    return owners


def _responsibility(world, replay, final_feature_id=None):
    owners = _pair(world)
    pr = _pr_for_replay(world, replay)
    if pr is None:
        # Final checks already carry accepted feature/actual-reviewer provenance.
        # Assign diagnosis from that provenance, never from a guessed code path.
        plans = [p for p in replay.get("plans", []) if p.get("feature_id") == final_feature_id]
        if len(plans) != 1 or final_feature_id not in owners:
            return None, None, None, None
        author = owners[final_feature_id]
        peer = next(actor for actor in owners.values() if actor != author)
        if plans[0].get("origin_reviewer") != peer:
            raise ReplayAdjudicationError("final_actual_peer_provenance_missing")
        return author, peer, final_feature_id, None
    author = getattr(pr, "author_id", None)
    features = set(getattr(pr, "linked_issue_ids", []) or []) | {getattr(pr, "linked_issue", None)}
    feature = [fid for fid, owner in owners.items() if owner == author and fid in features]
    if len(feature) != 1 or author != replay.get("integration_owner_id"):
        raise ReplayAdjudicationError("integration_owner_or_feature_mismatch")
    peer = next(actor for actor in owners.values() if actor != author)
    if peer not in (getattr(pr, "approved_by", []) or []):
        raise ReplayAdjudicationError("integration_actual_peer_approval_missing")
    return author, peer, feature[0], pr


def _failed_probes(replay, final_feature_id=None):
    behavior = replay.get("behavior_evidence") or {}
    probes, results = behavior.get("probes"), behavior.get("results")
    if not isinstance(probes, list) or not isinstance(results, list):
        raise ReplayAdjudicationError("executed_failure_missing")
    by_id = {row["probe_id"]: row for row in probes}
    failed = [row for row in results if (row.get("candidate") or {}).get("status") == "fail"]
    if final_feature_id is not None:
        ids = {pid for plan in replay.get("plans", []) if plan.get("feature_id") == final_feature_id
               for pid in plan.get("probe_ids", [])}
        failed = [row for row in failed if row.get("probe_id") in ids]
    if not failed or any(row.get("probe_id") not in by_id for row in failed):
        raise ReplayAdjudicationError("executed_failure_missing")
    return by_id, failed


def record_replay_failure(world, *, actor_id: str, replay: Mapping[str, Any], tick: int,
                          _final_feature_id=None) -> dict:
    """Record only a currently valid, helper-issued executed failure."""
    if (not isinstance(replay, Mapping) or actor_id != replay.get("actor_id")
            or replay.get("ok") is not False or not _replay_current(world, replay)):
        raise ReplayAdjudicationError("current_issued_failed_replay_required")
    if replay.get("integration_pr_id") is None and _final_feature_id is None:
        _, failures = _failed_probes(replay)
        ids = {row["probe_id"] for row in failures}
        features = sorted({p["feature_id"] for p in replay.get("plans", []) if ids & set(p.get("probe_ids", []))})
        if features:
            debts = [record_replay_failure(world, actor_id=actor_id, replay=replay, tick=tick,
                                          _final_feature_id=feature) for feature in features]
            return debts[0]
    definitions, failed = _failed_probes(replay, _final_feature_id)
    author, peer, feature, pr = _responsibility(world, replay, _final_feature_id)
    identity = {"schema": ADJUDICATION_SCHEMA, "replay_id": replay["replay_id"]}
    if _final_feature_id is not None:
        identity["final_feature_id"] = _final_feature_id
    failure_id = "replay_failure_" + _digest(identity)
    if failure_id in _store(world)["records"]:
        return replay_failure_status(world, failure_id)
    owner_head = sv.pr_head_snapshot(world, pr) if pr is not None else sv.mainline_snapshot(world) if author else None
    row = {
        "schema_version": ADJUDICATION_SCHEMA, "failure_id": failure_id, "revision": 1,
        "replay": deepcopy(dict(replay)), "replay_id": replay["replay_id"],
        "source_snapshot": deepcopy(replay["source_snapshot"]),
        "baseline_snapshot": deepcopy(replay["baseline_snapshot"]),
        "plan_set_digest": replay["plan_set_digest"], "stage": replay["stage"],
        "integration_pr_id": replay.get("integration_pr_id"), "repair_owner_id": author,
        "final_feature_id": _final_feature_id,
        "adjudicator_id": peer, "owner_feature_id": feature,
        "owner_head_snapshot": owner_head.receipt() if owner_head else None,
        "owner_head_fingerprints": {path: sr.implementation_fingerprint(path, text)
                                    for path, text in owner_head.files.items()} if owner_head else {},
        "created_tick": int(tick), "adjudicated_tick": None, "adjudication_attempts": 0,
        "status": "needs_peer_adjudication" if author is not None else "needs_coordination",
        "reason": "actual_peer_failure_validity_required" if author is not None else "final_mainline_has_no_implicit_repair_owner",
        "failed_probe_ids": [item["probe_id"] for item in failed],
        "failure_observations": [{"probe_id": item["probe_id"], "paths": list(item.get("paths") or []),
                                  "failure_site": _public_projection(sr._probe_failure_site(definitions[item["probe_id"]], item)),
                                  "output": _safe((item.get("candidate") or {}).get("output"), 1800)} for item in failed],
        "failed_probe_definitions": {item["probe_id"]: deepcopy(definitions[item["probe_id"]]) for item in failed},
        "probe_validity": {}, "valid_probe_ids": [], "defective_probe_ids": [], "unresolved_probe_ids": [],
        "repair_paths": [], "repair_path_groups": [], "owner_desk_fingerprints": {},
        "repair_path_adjudicated_ticks": {},
        "probe_corrections": [], "corrected_probe_ids": [], "brief": "",
    }
    _save(world, row)
    return replay_failure_status(world, failure_id)


def _validate_focused_receipts(world, row):
    """Recompute contract/definition/reached-site materials, not just a schema."""
    if row.get("adjudicated_tick") is None:
        return
    actor, feature = row["adjudicator_id"], row["owner_feature_id"]
    contract = bp.visible_review_contract(world, actor, feature)
    if contract is None:
        raise ReplayAdjudicationError("adjudicator_brief_not_visible")
    definitions, failed = _failed_probes(row["replay"], row.get("final_feature_id"))
    receipts = {item["probe_id"]: item for item in row["probe_validity"].get("receipts", [])}
    required = set(row["valid_probe_ids"]) | set(row["defective_probe_ids"])
    for observed in failed:
        pid = observed["probe_id"]
        if pid not in required:
            continue
        material = sr._probe_validity_material(
            world, reviewer_id=actor, feature_id=feature, description=contract["description"],
            obligations=list(contract["requirements"].values()), probe=definitions[pid], result=observed,
        )
        trusted = sr._cached_probe_validity(world, material)
        recorded = receipts.get(pid)
        if (trusted is None or recorded is None
                or any(recorded.get(key) != value for key, value in trusted.items() if key != "cached")
                or trusted["verdict"] != ("valid" if pid in row["valid_probe_ids"] else "defective")):
            raise ReplayAdjudicationError("focused_failure_receipt_no_longer_matches")


def trusted_replay_failure(world, failure_id: str, *, require_current=True) -> dict:
    """Internal full evidence. Do not put this payload directly in perception."""
    row = _load(world, failure_id)
    if require_current and not _replay_current(world, row["replay"]):
        raise ReplayAdjudicationError("replay_source_or_plans_not_current")
    _validate_focused_receipts(world, row)
    return row


def _repair_scope(world, row, valid_ids):
    pr = _pr_for_replay(world, row["replay"])
    head = sv.pr_head_snapshot(world, pr) if pr is not None else sv.mainline_snapshot(world)
    if head.receipt() != row["owner_head_snapshot"]:
        raise ReplayAdjudicationError("owner_head_changed")
    state = world._cooperbench_sdl_state
    allowed = {str(path).replace("\\", "/") for path in (state.get("feature_paths") or {}).get(row["owner_feature_id"], [])}
    changed = set()
    artifacts = getattr(world, "product_artifacts", {}) or {}
    for pid in head.patch_ids:
        patch = world.patches[pid]
        if getattr(patch, "actor_id", None) == row["repair_owner_id"]:
            artifact = artifacts.get(getattr(patch, "target_object_id", ""))
            changed.add(str(getattr(artifact, "linked_file_path", "") or "").replace("\\", "/"))
    scope = allowed & changed & set(head.files)
    groups = []
    for result in row["replay"]["behavior_evidence"]["results"]:
        if result["probe_id"] not in valid_ids:
            continue
        # Paths here were part of the accepted definition and freshly executed
        # observation. No arbitrary path from the focused model is authoritative.
        group = sorted(scope & set(result.get("paths") or row["failed_probe_definitions"][result["probe_id"]]["paths"]))
        if not group:
            return [], []
        groups.append(group)
    return sorted({path for group in groups for path in group}), groups


def adjudicate_replay_failure(world, *, actor_id: str, failure_id: str, tick: int) -> dict:
    """One explicit actual-peer action, with current source/plan CAS around it."""
    row = trusted_replay_failure(world, failure_id)
    author, peer, feature, _ = _responsibility(world, row["replay"], row.get("final_feature_id"))
    if not author or actor_id != peer or row["adjudicator_id"] != peer:
        raise ReplayAdjudicationError("actual_integration_peer_required")
    if (row["status"] not in {"needs_peer_adjudication", "validity_unresolved"}
            and not (row.get("unresolved_probe_ids") and row["status"] in {
                "confirmed_product_failure", "needs_probe_correction", "needs_coordination"})):
        raise ReplayAdjudicationError("failure_not_awaiting_adjudication")
    contract = bp.visible_review_contract(world, actor_id, feature)
    definitions, failed = _failed_probes(row["replay"], row.get("final_feature_id"))
    materials = {}
    for observed in failed:
        pid = observed["probe_id"]
        material = sr._probe_validity_material(
            world, reviewer_id=actor_id, feature_id=feature, description=contract["description"] if contract else "",
            obligations=list(contract["requirements"].values()) if contract else [], probe=definitions[pid], result=observed,
        )
        if material is None:
            raise ReplayAdjudicationError("adjudicator_required_brief_unshared_or_unread")
        materials[pid] = material
    if contract is None:
        raise ReplayAdjudicationError("adjudicator_required_brief_unshared_or_unread")
    before_revision = row["revision"]
    _, _, error, audit = sr._adjudicate_probe_defects(
        world, reviewer_id=actor_id, feature_id=feature, description=contract["description"],
        obligations=list(contract["requirements"].values()), behavior=deepcopy(row["replay"]["behavior_evidence"]),
        defects=[{"probe_id": item["probe_id"], "origin": "unclassified_executed_failure",
                  "issue": "Determine whether this exact integrated execution failure is authorized by its public contract."}
                 for item in failed],
    )
    if (_load(world, failure_id)["revision"] != before_revision
            or not _replay_current(world, row["replay"])
            or _responsibility(world, row["replay"], row.get("final_feature_id"))[:3] != (author, peer, feature)):
        raise ReplayAdjudicationError("adjudication_source_plans_or_debt_changed")
    for observed in failed:
        pid = observed["probe_id"]
        if materials[pid] != sr._probe_validity_material(
                world, reviewer_id=actor_id, feature_id=feature, description=contract["description"],
                obligations=list(contract["requirements"].values()), probe=definitions[pid], result=observed):
            raise ReplayAdjudicationError("adjudication_brief_or_failure_changed")
    valid, defective = [], []
    for receipt in audit.get("receipts", []):
        pid = receipt.get("probe_id")
        if pid not in materials:
            raise ReplayAdjudicationError("unrequested_focused_receipt")
        trusted = sr._cached_probe_validity(world, materials[pid])
        if trusted is None or any(receipt.get(key) != value for key, value in trusted.items() if key != "cached"):
            raise ReplayAdjudicationError("focused_receipt_not_issued")
        (valid if trusted["verdict"] == "valid" else defective).append(pid)
    paths, groups = _repair_scope(world, row, valid) if valid else ([], [])
    status = ("confirmed_product_failure" if valid and paths else "needs_coordination" if valid
              else "needs_probe_correction" if defective else "validity_unresolved")
    unresolved = sorted(set(row["failed_probe_ids"]) - set(valid) - set(defective))
    current_desk = sv.actor_desk_snapshot(world, author)
    brief = "\n".join(
        f"[{item['probe_id']}] {item['verdict']}: {item.get('reason', '')}\n"
        f"Public basis: {item.get('public_basis', '')}\n"
        + next((str((result.get('candidate') or {}).get('output') or '')[-1800:]
                for result in failed if result['probe_id'] == item['probe_id']), "")
        for item in audit.get("receipts", []) if item["probe_id"] in valid
    )
    current_desk_files = current_desk.files
    row.update(
        revision=before_revision + 1, adjudicated_tick=int(tick),
        adjudication_attempts=row["adjudication_attempts"] + 1, status=status,
        reason=("valid_failure_outside_integration_owner_repair_scope" if valid and not paths else error or status),
        probe_validity=deepcopy(audit), valid_probe_ids=sorted(valid), defective_probe_ids=sorted(defective),
        unresolved_probe_ids=unresolved, repair_paths=paths, repair_path_groups=groups,
        owner_desk_snapshot=current_desk.receipt(),
        owner_desk_fingerprints={path: row["owner_desk_fingerprints"].get(
            path, sr.implementation_fingerprint(path, current_desk_files.get(path, ""))) for path in paths},
        repair_path_adjudicated_ticks={path: row.get("repair_path_adjudicated_ticks", {}).get(path, int(tick)) for path in paths},
        brief=_safe(brief), last_adjudication_error=_safe(error, 600),
        # Replays that already existed before this adjudication cannot settle it.
        adjudicated_after_execution_sequence=int(world.__dict__.get("_cooperbench_accepted_probe_replay_sequence", row["replay"].get("execution_sequence", 0)) or 0),
    )
    _save(world, row)
    return replay_failure_status(world, failure_id)


def _repair_progress(world, row):
    if row["status"] != "confirmed_product_failure":
        return "", []
    try:
        current = sv.actor_desk_snapshot(world, row["repair_owner_id"])
        head = row["owner_head_snapshot"]
        if row.get("final_feature_id"):
            if tuple(current.base_main_commit_ids[:len(head["commit_ids"])]) != tuple(head["commit_ids"]):
                return "needs_sync", []
        elif (current.branch_id != head["branch_id"] or current.owner_id != row["repair_owner_id"]
                or tuple(current.commit_ids[:len(head["commit_ids"])]) != tuple(head["commit_ids"])):
            return "needs_coordination", []
        changed, patch_ids = set(), []
        current_files = current.files
        for pid in current.patch_ids:
            patch = world.patches[pid]
            if (getattr(patch, "actor_id", None) != row["repair_owner_id"]
                    or getattr(patch, "validation_status", None) != "accepted"
                    or row["owner_feature_id"] not in (getattr(patch, "related_issue_ids", []) or [])):
                continue
            artifact = world.product_artifacts.get(getattr(patch, "target_object_id", ""))
            path = str(getattr(artifact, "linked_file_path", "") or "").replace("\\", "/")
            if (path not in row["repair_paths"] or path not in current_files
                    or int(getattr(patch, "tick", 0)) <= row.get("repair_path_adjudicated_ticks", {}).get(path, row["adjudicated_tick"])
                    or sr.implementation_fingerprint(path, current_files[path]) == row["owner_desk_fingerprints"][path]
                    or sr.implementation_fingerprint(path, current_files[path]) != sr.implementation_fingerprint(path, patch.new_content)):
                continue
            changed.add(path)
            patch_ids.append(pid)
        ready = all(set(group) & changed for group in row["repair_path_groups"])
        return "ready_for_replay" if ready else "needs_edit", patch_ids
    except (ValueError, KeyError, TypeError, AttributeError):
        return "needs_coordination", []


def replay_failure_status(world, failure_id: str) -> dict:
    """Read-only bounded UI/action projection; no complete brief/probe bodies."""
    row = _load(world, failure_id)
    current = _replay_current(world, row["replay"])
    validity_current = True
    try:
        _validate_focused_receipts(world, row)
    except (ValueError, KeyError, TypeError, AttributeError):
        validity_current = False
    phase, patches = _repair_progress(world, row)
    if not validity_current:
        phase = "needs_coordination"
    elif not current and phase == "needs_edit":
        phase = "stale"
    keys = ("schema_version", "failure_id", "revision", "replay_id", "stage", "status", "reason",
            "integration_pr_id", "repair_owner_id", "adjudicator_id", "owner_feature_id",
            "source_snapshot", "baseline_snapshot", "plan_set_digest", "owner_head_snapshot",
            "owner_head_fingerprints", "owner_desk_fingerprints", "created_tick", "adjudicated_tick",
            "adjudication_attempts", "failed_probe_ids", "failure_observations",
            "valid_probe_ids", "defective_probe_ids", "unresolved_probe_ids",
            "repair_paths", "repair_path_groups", "corrected_probe_ids", "probe_corrections", "brief",
            "last_adjudication_error", "resolved_tick", "resolved_replay_id")
    result = {key: deepcopy(row[key]) for key in keys if key in row}
    result.update(current=current, validity_current=validity_current, repair_phase=phase,
                  repair_patch_ids=patches, source_current=_source_current(world, row))
    # The focused audit has bounded quote/reason fields but may contain raw
    # provider text; redact its serialized projection before it reaches prompts.
    result["probe_validity"] = _public_projection(row.get("probe_validity") or {})
    return result


def pending_replay_adjudications(world, actor_id: str) -> list[dict]:
    rows = []
    for fid in _store(world)["records"]:
        row = replay_failure_status(world, fid)
        if (row["current"] and row["adjudicator_id"] == actor_id
                and (row["status"] in {"needs_peer_adjudication", "validity_unresolved"}
                     or (row["unresolved_probe_ids"] and row["status"] in {
                         "confirmed_product_failure", "needs_probe_correction", "needs_coordination"}))):
            rows.append(row)
    return rows


def active_replay_failures(world) -> list[dict]:
    """Trusted bounded nonresolved failures still current or awaiting replay.

    Consumers must distinguish ``current`` (do not re-execute this unchanged
    failure) from ``ready_for_replay`` after a changed head (permit a fresh
    replay, which is the only operation that can resolve the repair debt).
    """
    return [row for fid in _store(world)["records"]
            if (row := replay_failure_status(world, fid))["status"] != "resolved"
            and (row["current"] or row["repair_phase"] == "ready_for_replay")]


def confirmed_replay_repair_debts(world, actor_id: str) -> list[dict]:
    return [row for fid in _store(world)["records"]
            if (row := replay_failure_status(world, fid))["repair_owner_id"] == actor_id
            and row["status"] == "confirmed_product_failure" and row["validity_current"]
            and (row["current"] or row["repair_phase"] == "ready_for_replay")]


def pending_confirmed_replay_repair_paths(world, actor_id: str) -> list[str]:
    return sorted({path for row in confirmed_replay_repair_debts(world, actor_id)
                   if row["repair_phase"] == "needs_edit" for path in row["repair_paths"]})


def pending_probe_corrections(world, actor_id: str) -> list[dict]:
    rows = []
    for fid in _store(world)["records"]:
        try:
            debt = trusted_replay_failure(world, fid)
        except (ValueError, KeyError, TypeError, AttributeError):
            continue
        pending = set(debt["defective_probe_ids"]) - set(debt["corrected_probe_ids"])
        for plan in debt["replay"]["plans"]:
            ids = sorted(pending & set(plan["probe_ids"]))
            if ids and plan["origin_reviewer"] == actor_id:
                rows.append({**replay_failure_status(world, fid), "feature_id": plan["feature_id"],
                             "probe_ids": ids, "origin_reviewer": actor_id,
                             "approved_pr_id": plan["approved_pr_id"]})
    return rows


def record_probe_correction(world, *, actor_id: str, failure_id: str, feature_id: str,
                            probe_ids: list[str], previous_revision: int, tick: int) -> dict:
    """CAS immediately before a caller atomically publishes corrected evidence."""
    row = trusted_replay_failure(world, failure_id)
    plans = [plan for plan in row["replay"]["plans"] if plan["feature_id"] == feature_id]
    pending = set(row["defective_probe_ids"]) - set(row["corrected_probe_ids"])
    if (row["revision"] != previous_revision or len(plans) != 1
            or plans[0]["origin_reviewer"] != actor_id
            or not isinstance(probe_ids, list) or not probe_ids or len(set(probe_ids)) != len(probe_ids)
            or set(probe_ids) != pending & set(plans[0]["probe_ids"])):
        raise ReplayAdjudicationError("probe_correction_identity_or_author_invalid")
    row["corrected_probe_ids"] = sorted(set(row["corrected_probe_ids"]) | set(probe_ids))
    row["probe_corrections"].append({"actor_id": actor_id, "feature_id": feature_id,
                                      "probe_ids": sorted(probe_ids), "tick": int(tick)})
    row["revision"] += 1
    if row["status"] == "needs_probe_correction":
        row["status"] = "corrected_requires_fresh_replay"
    _save(world, row)
    return replay_failure_status(world, failure_id)


def resolve_replay_failures(world, *, actor_id: str, replay: Mapping[str, Any], tick: int) -> list[str]:
    """Only a later issued current passing execution settles compatible debts."""
    if (actor_id != replay.get("actor_id") or replay.get("ok") is not True
            or not _replay_current(world, replay, passed=True)):
        raise ReplayAdjudicationError("current_issued_passing_replay_required")
    resolved = []
    new_probes = {item["probe_id"]: item for item in replay["behavior_evidence"]["probes"]}
    for fid in _store(world)["records"]:
        row = _load(world, fid)
        if (row["status"] == "resolved" or row["stage"] != replay["stage"]
                or row["integration_pr_id"] != replay.get("integration_pr_id")
                or (not row.get("final_feature_id") and row["repair_owner_id"] != replay.get("integration_owner_id"))
                or int(replay.get("execution_sequence", 0)) <= max(
                    int(row["replay"].get("execution_sequence", 0)),
                    int(row.get("adjudicated_after_execution_sequence", 0)))):
            continue
        if not set(row["failed_probe_ids"]) <= set(new_probes):
            continue
        # A valid failure cannot disappear by silently weakening its probe.
        unchanged = set(row["failed_probe_ids"]) - set(row["corrected_probe_ids"])
        if any(new_probes[pid] != row["failed_probe_definitions"][pid] for pid in unchanged):
            continue
        if row["status"] == "confirmed_product_failure":
            phase, pids = _repair_progress(world, row)
            if phase != "ready_for_replay" or not set(pids).intersection(replay["source_snapshot"]["patch_ids"]):
                continue
        row.update(status="resolved", resolved_tick=int(tick), resolved_replay_id=replay["replay_id"],
                   revision=row["revision"] + 1, resolved_source_snapshot=deepcopy(replay["source_snapshot"]))
        _save(world, row)
        resolved.append(fid)
    return resolved


__all__ = ["ADJUDICATION_SCHEMA", "ReplayAdjudicationError", "record_replay_failure",
           "adjudicate_replay_failure", "trusted_replay_failure", "replay_failure_status",
           "pending_replay_adjudications", "active_replay_failures", "confirmed_replay_repair_debts",
           "pending_confirmed_replay_repair_paths", "pending_probe_corrections",
           "record_probe_correction", "resolve_replay_failures"]
