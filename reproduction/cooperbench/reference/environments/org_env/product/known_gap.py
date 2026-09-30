"""KnownGap — a status-bearing product gap (Internal Pipeline Completion spec #1).

v6 carried gaps as bare strings on artifacts (`art.known_gaps`) and on the product
(`known_systemic_issues`), so a built/merged capability could leave a contradictory
"still active" gap. A KnownGap is the reconcilable object: it links to the artifact
that would close it, the capability that resolves it, and accumulates resolution
evidence (patch / task / PR ids). The StateReconciler updates its `status`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

GAP_STATUS = ("active", "mitigated", "resolved", "regressed")


@dataclass
class KnownGap:
    gap_id: str
    description: str
    scope: str = "systemic"                 # systemic | artifact
    status: str = "active"                   # active | mitigated | resolved | regressed
    linked_artifact_id: Optional[str] = None
    required_capability: str = ""            # keyword that, once built, closes the gap ("" = N/A)
    linked_issue_ids: List[str] = field(default_factory=list)
    resolved_by_patch_ids: List[str] = field(default_factory=list)
    resolved_by_task_ids: List[str] = field(default_factory=list)
    resolved_by_pr_ids: List[str] = field(default_factory=list)
    critical: bool = False
    first_seen_tick: int = 0
    last_update_tick: int = 0
    evidence: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {k: (list(v) if isinstance(v, list) else v) for k, v in self.__dict__.items()}


# gap-text -> (artifact purpose that closes it, capability keyword, is-critical)
# matched in order; the first keyword hit wins. Keeps the reconciler's resolution
# rules in one inspectable place (spec examples 1-3).
_GAP_RULES: Tuple[Tuple[Tuple[str, ...], str, str, bool], ...] = (
    (("credibility", "source credibility", "score credibility"), "source_tracker", "credibility", False),
    (("evidence link", "enforce evidence", "claim-evidence", "claim evidence", "evidence enforcement"),
     "claim_tracker", "evidence", True),
    (("uncertainty",), "claim_tracker", "uncertainty", False),
    (("unsupported claim", "report writer", "evidence gate"), "report_writer", "supported", True),
    (("release quality gate", "quality gate", "report release quality"), "report_quality", "", True),
    (("validated metric", "validated metrics", "eval", "benchmark", "metric"), "eval", "metric", True),
    (("overpromis", "overclaim", "readme"), "readme", "", True),
    (("reproducible", "expected trace", "reproducib"), "research_loop", "trace", False),
    (("design doc", "mixes vision", "coherent spec", "mixes"), "product_design", "", False),
    (("inconsistent with", "docs are inconsistent", "do not match code"), "readme", "", False),
    (("state model", "not connected to trackers", "research loop"), "research_loop", "state_model", False),
)


def infer_gap_meta(text: str) -> Tuple[Optional[str], str, bool]:
    """(artifact purpose, capability keyword, critical) for a gap description."""
    s = (text or "").lower()
    for keys, purpose, cap, critical in _GAP_RULES:
        if any(k in s for k in keys):
            return purpose, cap, critical
    return None, "", False


def _artifact_for_purpose(world: Any, purpose: Optional[str]):
    if not purpose:
        return None
    from environments.org_env.product.objects import artifact_purpose
    for a in (getattr(world, "product_artifacts", {}) or {}).values():
        if getattr(a, "artifact_type", "") == "issue":
            continue
        key = getattr(a, "linked_file_path", "") or getattr(a, "artifact_id", "")
        if artifact_purpose(key) == purpose:
            return a
    return None


def build_known_gaps(world: Any) -> Dict[str, KnownGap]:
    """Seed the gap registry from the product's systemic issues + per-artifact gaps,
    linking each to the artifact/capability that would close it."""
    gaps: Dict[str, KnownGap] = {}
    ps = getattr(world, "product", None)
    tick = int(getattr(world, "world_tick", 0) or 0)
    seen_desc: set = set()

    def _add(gid: str, desc: str, scope: str, art):
        key = (desc or "").strip().lower()
        if not key or key in seen_desc:
            return
        seen_desc.add(key)
        purpose, cap, critical = infer_gap_meta(desc)
        a = art or _artifact_for_purpose(world, purpose)
        gaps[gid] = KnownGap(
            gap_id=gid, description=desc, scope=scope, required_capability=cap, critical=critical,
            linked_artifact_id=(getattr(a, "artifact_id", None) if a else None),
            first_seen_tick=tick, last_update_tick=tick)

    for i, desc in enumerate(list(getattr(ps, "known_systemic_issues", []) or [])):
        _add(f"gap_sys_{i}", desc, "systemic", None)
    for a in (getattr(world, "product_artifacts", {}) or {}).values():
        if getattr(a, "artifact_type", "") == "issue":
            continue
        for j, desc in enumerate(list(getattr(a, "known_gaps", []) or [])):
            _add(f"gap_{a.artifact_id}_{j}", desc, "artifact", a)
    return gaps


__all__ = ["KnownGap", "GAP_STATUS", "build_known_gaps", "infer_gap_meta"]
