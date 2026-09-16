"""Product milestone state machine + shipping mode (v13 P3).

v12 plateaued: days 14–30 produced lots of organization activity but no product closure,
because the org stayed in one work mode the whole month. A stage machine + "shipping mode"
make the org switch from open-ended polishing to converging on a shippable v0.1: once a release
smoke/CI/contract blocker is open, generic off-task actions are down-weighted so the team closes
the blocker instead of staying busy.
"""
from __future__ import annotations

from typing import Any

STAGES = ("discovery", "stabilization", "integration", "market_validation")

# blocker-issue id substrings that mean "the product doesn't run / isn't shippable yet"
_SHIPPING_BLOCKER_KEYS = ("smoke", "ci", "eval", "test", "build", "contract")


def in_shipping_mode(world: Any) -> bool:
    """True when a release smoke / CI / contract blocker is OPEN (or a build error is live) — the
    org should converge on closing it rather than doing generic busywork."""
    arts = getattr(world, "product_artifacts", {}) or {}
    for a in arts.values():
        if getattr(a, "artifact_type", "") != "issue" or getattr(a, "status", "") != "open":
            continue
        aid = str(getattr(a, "artifact_id", "")).lower()
        if aid.startswith("rel_blocker_") and any(k in aid for k in _SHIPPING_BLOCKER_KEYS):
            return True
    return bool(getattr(world, "_build_error", "") or getattr(world, "_mainline_smoke_error", "") or "")


def has_shippable_work(world: Any) -> bool:
    """True when a MERGED fix is sitting UNSHIPPED — no released version includes it yet — so the org
    should cut/publish a release rather than idle (the "fixed but never shipped" failure mode)."""
    rs = getattr(world, "repo_system", None)
    if rs is None:
        return False
    repo = getattr(rs, "repo", None)
    if repo is None:
        return False
    rcs = getattr(repo, "release_candidates", {}) or {}
    shipped_prs = {p for r in rcs.values() if getattr(r, "status", "") == "released"
                   for p in (getattr(r, "included_pr_ids", []) or [])}
    for pr in (getattr(repo, "pull_requests", {}) or {}).values():
        st = getattr(pr.status, "value", str(pr.status))
        if st == "merged" and getattr(pr, "patch_ids", None) and pr.pr_id not in shipped_prs:
            return True
    return False


def current_stage(world: Any) -> str:
    """Derive the product stage from world state (read-only)."""
    rs = getattr(world, "repo_system", None)
    released = len(getattr(getattr(rs, "repo", None), "releases", {}) or {}) if rs else 0
    if released >= 1:
        return "market_validation"
    if in_shipping_mode(world):
        return "integration"
    merged = sum(1 for e in (getattr(world, "events", []) or [])
                 if e.get("type") == "repo_event" and e.get("subtype") == "pr_merged")
    if merged >= 1:
        return "stabilization"
    return "discovery"


__all__ = ["STAGES", "in_shipping_mode", "has_shippable_work", "current_stage"]
