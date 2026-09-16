"""Explicit original-peer repair of accepted checks, never a repository review.

Failed drafts live separately from accepted plans. Only a new executed and
source-reviewed plan is published, while the old integration debt still
requires a fresh replay. Repository/branch/merge state is never changed here.
"""
from __future__ import annotations

from copy import copy, deepcopy
import hashlib
import json
from typing import Any, Mapping

from . import behavior_probes as bp
from . import joint_probe_replay as replay
from . import semantic_review as sr
from . import source_views as sv


_DRAFTS = "_cooperbench_accepted_probe_correction_drafts"
_ISOLATED = ("_cooperbench_behavior_plans", "_cooperbench_probe_evidence_gaps",
             "_cooperbench_semantic_reviews", "_cooperbench_probe_validity_receipts")


class AcceptedProbeRepairError(ValueError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def _context(world: Any, actor_id: str, failure_id: str, feature_id: str) -> dict:
    from .replay_adjudication import trusted_replay_failure

    debt = trusted_replay_failure(world, failure_id, require_current=True)
    evidence = debt.get("replay") or {}
    plans = evidence.get("plans") or []
    selected = [plan for plan in plans if isinstance(plan, Mapping) and plan.get("feature_id") == feature_id]
    if len(selected) != 1 or selected[0].get("origin_reviewer") != actor_id:
        raise AcceptedProbeRepairError("accepted_probe_correction_requires_original_peer")
    metadata = selected[0]
    corrected = set(debt.get("corrected_probe_ids") or [])
    probe_ids = sorted({pid for pid in debt.get("defective_probe_ids", [])
                        if isinstance(pid, str) and pid.split(":", 1)[0] == feature_id and pid not in corrected})
    if not probe_ids:
        raise AcceptedProbeRepairError("accepted_probe_correction_has_no_adjudicated_defect")
    repo = world.repo_system.repo
    pr = repo.pull_requests.get(metadata.get("approved_pr_id"))
    if pr is None or replay._status(getattr(pr, "status", "")) not in {"approved", "merged"}:
        raise AcceptedProbeRepairError("accepted_probe_correction_approved_pr_missing")
    baseline = sv.freeze_baseline(world)
    current_metadata, definitions, _ = replay._accepted_plan(world, actor_id, feature_id, pr, baseline)
    if current_metadata != metadata:
        raise AcceptedProbeRepairError("accepted_probe_correction_accepted_plan_changed")
    contract = bp.visible_review_contract(world, actor_id, feature_id)
    behavior = evidence.get("behavior_evidence") or {}
    by_id = {row.get("probe_id"): row for row in behavior.get("results", []) if isinstance(row, Mapping)}
    original = {row["probe_id"]: row for row in definitions}
    verdicts = (debt.get("probe_validity") or {}).get("receipts") or []
    adjudicator = str(debt.get("adjudicator_id") or "")
    adjudicator_contract = bp.visible_review_contract(world, adjudicator, feature_id)
    if adjudicator_contract is None:
        raise AcceptedProbeRepairError("accepted_probe_correction_adjudication_brief_unavailable")
    defects = []
    for pid in probe_ids:
        matching = [row for row in verdicts if isinstance(row, Mapping) and row.get("probe_id") == pid]
        if len(matching) != 1 or pid not in original or pid not in by_id:
            raise AcceptedProbeRepairError("accepted_probe_correction_defect_receipt_missing")
        material = sr._probe_validity_material(
            world, reviewer_id=adjudicator, feature_id=feature_id, description=adjudicator_contract["description"],
            obligations=list(adjudicator_contract["requirements"].values()), probe=original[pid], result=by_id[pid])
        cached = sr._cached_probe_validity(world, material)
        receipt = matching[0]
        if (cached is None or cached.get("verdict") != "defective" or receipt.get("verdict") != "defective"
                or any(receipt.get(key) != cached.get(key) for key in (
                    "identity", "brief_identity", "feature_id", "reviewer_id", "failure_site", "reason"))):
            raise AcceptedProbeRepairError("accepted_probe_correction_defect_receipt_mismatch")
        defects.append({"probe_id": pid, "adjudication_status": "confirmed",
                        "adjudication_reason": receipt["reason"], "validity_identity": receipt["identity"]})
    snapshot = sv.pr_head_snapshot(world, pr)
    if snapshot.receipt() != metadata.get("approved_source_snapshot"):
        raise AcceptedProbeRepairError("accepted_probe_correction_approved_source_changed")
    binding = {
        "failure_id": failure_id, "failure_revision": debt.get("revision"),
        "replay_id": evidence.get("replay_id"), "actor_id": actor_id, "feature_id": feature_id,
        "probe_ids": probe_ids, "source_snapshot": snapshot.receipt(), "baseline_snapshot": baseline.receipt(),
        "plan": deepcopy(world._cooperbench_behavior_plans[feature_id]),
        "approval": deepcopy(world._cooperbench_semantic_reviews[feature_id]),
        "contract": deepcopy(contract), "defects": defects,
    }
    return {"binding": binding, "pr": pr, "snapshot": snapshot, "baseline": baseline,
            "metadata": current_metadata, "debt": debt}


def _staged_world(world: Any, feature_id: str, binding: dict) -> tuple[Any, dict]:
    staged = copy(world)
    for name in _ISOLATED:
        setattr(staged, name, deepcopy(getattr(world, name, {}) or {}))
    key = binding["failure_id"] + ":" + feature_id
    saved = (getattr(world, _DRAFTS, {}) or {}).get(key)
    if saved is not None:
        payload = saved.get("payload") if isinstance(saved, Mapping) else None
        if not isinstance(payload, Mapping) or saved.get("digest") != _digest(payload):
            raise AcceptedProbeRepairError("accepted_probe_correction_draft_invalid")
        if payload.get("binding") == binding:
            for name in ("_cooperbench_behavior_plans", "_cooperbench_probe_evidence_gaps"):
                value = payload.get(name)
                target = getattr(staged, name)
                if value is None:
                    target.pop(feature_id, None)
                else:
                    target[feature_id] = deepcopy(value)
            staged._cooperbench_probe_validity_receipts.update(deepcopy(payload.get("validity_receipts") or {}))
            return staged, dict(payload)
    bp.invalidate_probe_rows(staged, feature_id, binding["probe_ids"],
                             defects=deepcopy(binding["defects"]), integration_failure_id=binding["failure_id"])
    return staged, {}


def _save_draft(world: Any, staged: Any, binding: dict, semantic: dict, *, previous: dict,
                coordination: dict | None = None) -> dict:
    from relic.research.public_redaction import redact_public_text

    feature_id = binding["feature_id"]
    payload = {"binding": deepcopy(binding), "last_error": redact_public_text(
                   semantic.get("error") or "source_review_not_approved", max_chars=1200),
               "validity_receipts": deepcopy({key: value for key, value in staged._cooperbench_probe_validity_receipts.items()
                                               if value.get("feature_id") == feature_id
                                               and value.get("reviewer_id") == binding["actor_id"]})}
    for name in ("_cooperbench_behavior_plans", "_cooperbench_probe_evidence_gaps"):
        payload[name] = deepcopy(getattr(staged, name).get(feature_id))
    gap = payload["_cooperbench_probe_evidence_gaps"] or {}
    payload["progress_digest"] = _digest({
        "plan": payload["_cooperbench_behavior_plans"],
        "gap": {key: gap.get(key) for key in ("previous_probes", "previous_response", "retained_probes",
                                               "repair_probe_ids", "missing_requirements", "missing_interactions")},
    })
    payload["stalled_attempts"] = (int(previous.get("stalled_attempts") or 0) + 1
                                   if previous.get("progress_digest") == payload["progress_digest"] else 1)
    if coordination is None and payload["stalled_attempts"] >= 3:
        coordination = {"status": "needs_coordination", "reason": "accepted_probe_correction_no_progress",
                        "blocked_source": deepcopy(binding["source_snapshot"]),
                        "stalled_attempts": payload["stalled_attempts"]}
    payload["coordination"] = deepcopy(coordination)
    world.__dict__.setdefault(_DRAFTS, {})[binding["failure_id"] + ":" + feature_id] = {
        "payload": payload, "digest": _digest(payload)}
    return payload


def _check_targeted_change(binding: dict, plan: Mapping[str, Any]) -> None:
    old = {row["probe_id"]: row for row in binding["plan"]["probes"]}
    new = {row["probe_id"]: row for row in plan.get("probes", [])}
    repaired = set(binding["probe_ids"])
    if not set(old) <= set(new) or any(new[pid] != row for pid, row in old.items() if pid not in repaired):
        raise AcceptedProbeRepairError("accepted_probe_correction_changed_unaffected_definitions")
    if any(bp._probe_content(new[pid]) == bp._probe_content(old[pid]) for pid in repaired):
        raise AcceptedProbeRepairError("accepted_probe_correction_definition_unchanged")


def repair_accepted_feature_probes(world: Any, *, actor_id: str, failure_id: str,
                                    feature_id: str, tick: int) -> dict:
    """Run the original peer's targeted correction without re-reviewing RepoLite."""
    from .replay_adjudication import ReplayAdjudicationError, record_probe_correction

    result = {"available": False, "ok": False, "failure_id": failure_id, "feature_id": feature_id,
              "actor_id": actor_id, "tick": tick, "correction_pending": True,
              "requires_fresh_replay": False, "repair_paths": []}
    try:
        context = _context(world, actor_id, failure_id, feature_id)
        binding, pr = context["binding"], context["pr"]
        result.update(pr_id=pr.pr_id, probe_ids=binding["probe_ids"],
                      source_snapshot=deepcopy(binding["source_snapshot"]),
                      plan_set_digest=context["debt"]["replay"]["plan_set_digest"])
        original_validity = deepcopy(getattr(world, "_cooperbench_probe_validity_receipts", {}) or {})
        staged, previous_draft = _staged_world(world, feature_id, binding)
        if previous_draft.get("coordination"):
            result.update(previous_draft["coordination"])
            result["error"] = result["reason"]
            return result
        # This helper performs real peer generation, real frozen-head probe
        # execution, and real source review, but never touches RepoLite status.
        semantic = sr.review_feature_pull_request(staged, reviewer_id=actor_id, pull_request=pr)
        result["semantic_evidence"] = semantic
        # Validate against the REAL world after all model/executor work. A
        # staging copy can never authorize overwriting a concurrently new plan.
        current = _context(world, actor_id, failure_id, feature_id)
        if current["binding"] != binding:
            raise AcceptedProbeRepairError("accepted_probe_correction_context_changed")
        if semantic.get("available") is not True or semantic.get("approved") is not True:
            actual_failures = {row.get("probe_id") for row in (semantic.get("behavior_evidence") or {}).get("results", [])
                               if (row.get("candidate") or {}).get("status") == "fail"}
            valid_failures = [row for row in (semantic.get("probe_validity") or {}).get("receipts", [])
                              if row.get("verdict") == "valid" and row.get("probe_id") in actual_failures]
            coordination = None
            if valid_failures and sr.semantic_review_repair_evidence_matches(staged, actor_id, feature_id, semantic):
                from relic.research.public_redaction import redact_public_text
                coordination = {
                    "status": "needs_coordination",
                    "reason": "accepted_probe_correction_revealed_product_failure",
                    "blocked_source": deepcopy(binding["source_snapshot"]),
                    "failed_probe_ids": sorted(row["probe_id"] for row in valid_failures),
                    "product_failure_summary": redact_public_text(
                        "\n".join(str(row.get("reason") or "") for row in valid_failures), max_chars=1200),
                }
            draft = _save_draft(world, staged, binding, semantic, previous=previous_draft,
                                coordination=coordination)
            result["error"] = str(semantic.get("error") or "accepted_probe_correction_source_review_not_approved")
            if draft.get("coordination"):
                result.update(draft["coordination"])
                result["error"] = result["reason"]
            return result
        plan = staged._cooperbench_behavior_plans.get(feature_id) or {}
        _check_targeted_change(binding, plan)
        receipt = {**deepcopy(semantic), "pr_id": pr.pr_id, "reviewer_id": actor_id,
                   "tick": int(tick), "blocking_rounds": int(binding["approval"].get("blocking_rounds") or 0),
                   "accepted_probe_correction_failure_id": failure_id}
        staged._cooperbench_semantic_reviews[feature_id] = receipt
        # Verify actual execution rows and exact new definitions, not merely
        # approved=True returned by source review or an old green receipt.
        replay._accepted_plan(staged, actor_id, feature_id, pr, context["baseline"])
        published_plan, published_receipt = deepcopy(plan), deepcopy(receipt)
        validity_updates = {key: deepcopy(value) for key, value in staged._cooperbench_probe_validity_receipts.items()
                            if value != original_validity.get(key)}
        current_validity = getattr(world, "_cooperbench_probe_validity_receipts", {}) or {}
        if any(current_validity.get(key) not in (original_validity.get(key), value)
               for key, value in validity_updates.items()):
            raise AcceptedProbeRepairError("accepted_probe_correction_validity_changed")
        # The debt helper validates its host-issued seal/revision/source again.
        # There are no awaits or model calls between this CAS and publication.
        debt = record_probe_correction(world, actor_id=actor_id, failure_id=failure_id,
                                       feature_id=feature_id, probe_ids=binding["probe_ids"],
                                       previous_revision=binding["failure_revision"], tick=int(tick))
        world._cooperbench_behavior_plans[feature_id] = published_plan
        world._cooperbench_semantic_reviews[feature_id] = published_receipt
        world.__dict__.setdefault("_cooperbench_probe_evidence_gaps", {}).pop(feature_id, None)
        world.__dict__.setdefault("_cooperbench_probe_validity_receipts", {}).update(validity_updates)
        world.__dict__.setdefault(_DRAFTS, {}).pop(failure_id + ":" + feature_id, None)
        state = world.__dict__.get("_cooperbench_sdl_state") or {}
        state.update(attestations={}, frozen_digest=None, frozen_tick=None, phase="delivery")
        # Historical replay receipts remain auditable. Their original plan-set
        # binding no longer matches, so none can stand in for a fresh replay.
        result.update(available=True, ok=True, error="", correction_pending=False,
                      requires_fresh_replay=True, debt_revision=debt.get("revision"),
                      source_snapshot=receipt["source_snapshot"],
                      corrected_plan_hash=receipt["behavior_evidence"]["plan_hash"])
        return result
    except (AcceptedProbeRepairError, ReplayAdjudicationError, replay.ProbeReplayError, sv.SourceViewError) as error:
        result["error"] = error.code
        return result


__all__ = ["repair_accepted_feature_probes", "AcceptedProbeRepairError"]
