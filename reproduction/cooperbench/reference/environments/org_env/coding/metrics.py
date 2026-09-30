"""Coding capability metrics (v11 §8): make 'they can code' measurable as 'they CLOSE
technical problems'. All derived from the event stream + debugging episodes + repo, so they
are event-grounded and inspectable (no free-form scoring).
"""
from __future__ import annotations

from typing import Any, Dict

_CI_FAIL = {"failed", "failing", "error", "red", "broken", "fail"}


def _readiness_layers(world: Any, merged_pr: int) -> Dict[str, Any]:
    """v13 P1: separate the four kinds of 'ready' so workflow activity can't masquerade as a
    shippable product. artifact (gaps cleared) / workflow (PR+CI) / runtime (smoke really
    passes) / market (released + external feedback)."""
    prod = getattr(world, "product", None)
    arts = getattr(world, "product_artifacts", {}) or {}
    gaps = getattr(world, "known_gaps", {}) or {}
    critical_open = sum(1 for g in gaps.values()
                        if getattr(g, "status", "active") in ("active", "regressed")
                        and str(getattr(g, "severity", "")).lower() in ("critical", "high"))
    artifact_ready = critical_open == 0
    workflow_ready = merged_pr >= 1
    # runtime: only trust a CACHED smoke for the CURRENT mainline (no new subprocess here)
    runtime_ready = False
    smoke_ok = False
    try:
        from environments.org_env.product.materialize import _repo_hash, smoke_quality_issue
        sm = (getattr(world, "_smoke_cache", {}) or {}).get(_repo_hash(world, True))
        smoke_ok = bool(sm and sm.get("ok"))                      # CLI runs end-to-end (exit 0)
        runtime_ready = bool(smoke_ok and not smoke_quality_issue(sm))  # ...AND grounded
    except Exception:
        runtime_ready = smoke_ok = not (getattr(world, "_build_error", "") or "")
    rs = getattr(world, "repo_system", None)
    released = len(getattr(getattr(rs, "repo", None), "releases", {}) or {}) if rs else 0
    has_feedback = bool(getattr(world, "tickets", {}) or {})
    market_ready = released >= 1 and has_feedback
    try:
        from environments.org_env.product.milestone import current_stage, in_shipping_mode
        stage, shipping = current_stage(world), in_shipping_mode(world)
    except Exception:
        stage, shipping = None, False
    # #4: full open-gap count + OSS objective split, so release/market readiness never implies
    # "all gaps cleared" (release_ready can be true with open non-critical / untested issues).
    open_gaps_all = sum(1 for g in gaps.values() if getattr(g, "status", "active") in ("active", "regressed"))
    _oss_hidden = getattr(world, "_oss_release_hidden", None) or {}
    return {"artifact_ready": artifact_ready, "workflow_ready": workflow_ready,
            "runtime_ready": runtime_ready, "market_ready": market_ready,
            "critical_gaps_open": critical_open, "open_systemic_gaps": open_gaps_all,
            "gaps_cleared": open_gaps_all == 0,
            "tested_issue_fix_rate": _oss_hidden.get("issue_fix_rate"),
            "hidden_test_pass_rate": _oss_hidden.get("pass_rate"),
            "released": released, "stage": stage, "shipping_mode": shipping,
            "v01": _v01_criteria(world, smoke_ok, runtime_ready, market_ready)}


def _v01_criteria(world: Any, smoke_ok: bool, runtime_ready: bool, market_ready: bool) -> Dict[str, Any]:
    """v13 §G: the v0.1 done-criteria as a measurable checklist (enforcement is via the existing
    release gates + the grounding red light; this makes 'how close to v0.1' inspectable)."""
    arts = getattr(world, "product_artifacts", {}) or {}
    def has(sub):
        return any(sub in (getattr(a, "linked_file_path", "") or "").lower() for a in arts.values())
    readme_safe = False
    try:
        from environments.org_env.backend.repo.release import readme_overclaim
        rm = next((a for a in arts.values() if (getattr(a, "linked_file_path", "") or "").lower() == "readme.md"), None)
        readme_safe = rm is not None and not readme_overclaim(getattr(rm, "content", "") or "")
    except Exception:
        pass
    crit = {
        "cli_runnable": smoke_ok,             # CLI runs end-to-end (smoke exit 0)
        "claims_grounded": runtime_ready,     # ...AND grounded (smoke grounding red light passes)
        "readme_safe": readme_safe,
        "onboarding": has("onboarding"),
        "demo_transcript": has("sample") or has("example"),
        "market_validation": market_ready,
    }
    crit["done"] = sum(1 for k, v in crit.items() if v)
    crit["total"] = len(crit) - 1
    return crit


def coding_metrics(world: Any) -> Dict[str, Any]:
    events = getattr(world, "events", []) or []
    em = getattr(world, "episode_manager", None)

    def cnt(t: str, sub: str) -> int:
        return sum(1 for e in events if e.get("type") == t and e.get("subtype") == sub)

    ci = [e for e in events if e.get("type") == "repo_event" and e.get("subtype") == "ci"]
    ci_pass = sum(1 for e in ci if str(e.get("status", "")).lower() == "passed")
    ci_fail = sum(1 for e in ci if str(e.get("status", "")).lower() in _CI_FAIL)
    merged = cnt("repo_event", "pr_merged")
    changes_req = cnt("repo_event", "changes_requested")
    patch_applied = cnt("product_event", "patch_applied")
    patch_rej = cnt("product_event", "patch_rejected")
    total_patch = patch_applied + patch_rej

    dogfood = [e for e in events if e.get("type") == "product_event" and e.get("subtype") == "dogfood"]
    dogfood_broken = sum(1 for e in dogfood if str(e.get("outcome", "")) in ("crash", "broken_output"))
    dbg = [ep for ep in em.episodes.values() if ep.episode_type == "debugging_episode"] if em else []
    dbg_res = [ep for ep in dbg if ep.status == "resolved"]
    durations = [(ep.end_tick or ep.updated_at_tick) - ep.start_tick
                 for ep in dbg_res if (ep.end_tick or ep.updated_at_tick)]
    tokens = getattr(world, "llm_token_total", None)

    def rate(a: int, b: int):
        return round(a / b, 3) if b else None

    # #9: split test/CI telemetry so there is no single misleading "test_pass_rate" (the old one was
    # ci_pass / ALL ci events incl auto re-checks -> read ~0.06 while public+hidden actually passed).
    _oss_hidden = getattr(world, "_oss_release_hidden", None) or {}
    hidden_rate = _oss_hidden.get("pass_rate")
    hidden_issue_fix = _oss_hidden.get("issue_fix_rate")
    if hidden_rate is None:
        _hevs = [e for e in events if e.get("subtype") == "oss_evaluation"]
        if _hevs:
            hidden_rate = _hevs[-1].get("hidden_pass_rate")
            hidden_issue_fix = _hevs[-1].get("issue_fix_rate")
    smoke_pass_rate = 0.0 if (getattr(world, "_mainline_smoke_error", "") or "") else 1.0

    # v13 P1: blocker_resolution_rate is grounded in REAL release-blocker issues (closed /
    # total), so it can't read 1.0 while the release candidate is still blocked.
    arts = getattr(world, "product_artifacts", {}) or {}
    blockers = [a for a in arts.values() if str(getattr(a, "artifact_id", "")).startswith("rel_blocker_")]
    blk_closed = sum(1 for a in blockers if getattr(a, "status", "open") not in ("open", "in_progress"))

    # v14b: surface evaluator self-consistency (non-blocking) — a contradictory eval
    # (coverage high but unsupported_claim_rate high) is an eval_stub bug, now visible here.
    eval_consistent = True
    try:
        from environments.org_env.product.materialize import _repo_hash, smoke_eval_inconsistency
        _sm = (getattr(world, "_smoke_cache", {}) or {}).get(_repo_hash(world, True))
        eval_consistent = not (_sm and smoke_eval_inconsistency(_sm))
    except Exception:
        pass

    return {
        "eval_consistent": eval_consistent,
        "merged_pr_count": merged,
        # split test/CI rates (#9) — do NOT read test_pass_rate as "did the product pass tests":
        "pr_ci_pass_rate": rate(ci_pass, len(ci)),          # PR-CI events that passed (incl re-checks)
        "hidden_test_pass_rate": hidden_rate,               # OBJECTIVE OSS behavior tests (the real one)
        "hidden_issue_fix_rate": hidden_issue_fix,          # issue-level hidden pass rate
        "smoke_pass_rate": smoke_pass_rate,                 # mainline smoke ok?
        "test_pass_rate": rate(ci_pass, len(ci)),           # DEPRECATED alias of pr_ci_pass_rate
        "blocker_resolution_rate": rate(blk_closed, len(blockers)),
        "debugging_resolution_rate": rate(len(dbg_res), len(dbg)),
        "readiness": _readiness_layers(world, merged),
        "patch_rejection_rate": rate(patch_rej, total_patch),
        "ci_iteration_count": round(len(ci) / merged, 2) if merged else None,
        "code_review_request_changes": changes_req,
        "regression_rate": (rate(ci_fail, merged) if merged else rate(ci_fail, len(ci))),
        "token_per_merged_patch": round(tokens / merged, 1) if (tokens and merged) else None,
        "issue_to_merge_time": round(sum(durations) / len(durations), 1) if durations else None,
        # supporting counts (denominators) for transparency
        "ci_events": len(ci),                               # #9: ci EVENTS incl auto re-checks (not distinct PRs)
        "ci_runs": len(ci),                                 # kept for back-compat (== ci_events)
        "patches_applied": patch_applied,
        "debugging_episodes": len(dbg),
        "debugging_resolved": len(dbg_res),
        "dogfood_runs": len(dogfood),
        "dogfood_broken": dogfood_broken,
    }


__all__ = ["coding_metrics"]
