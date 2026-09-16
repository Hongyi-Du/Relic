"""Workload-based action duration (Internal Pipeline Completion spec #6).

A long, multi-file change should take more ticks than a one-line edit. We derive a
workload score from the concrete patch (pseudo-diff lines, files touched, added
fields/checks) or the action's generated content, map it to a tick cost with a log
curve + a superlinear penalty for oversized work, and cap a single action so the
remainder splits into pending work units.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional

HARD_WORKLOAD_THRESHOLD = 6.0
MAX_TICK_PER_ACTION = 8


def estimate_workload(action_type: str, params: Dict[str, Any], patch: Optional[Any]) -> float:
    """A unit-less workload score from whatever signal is available."""
    diff_lines = 0
    files = 0
    fields = 0
    if patch is not None:
        diff_lines = len((getattr(patch, "pseudo_diff", "") or "").splitlines())
        sections = getattr(patch, "changed_sections", None) or []
        diff_lines += 4 * len(sections)
        files = len(getattr(patch, "files_changed", []) or [])
        fields = (len(getattr(patch, "added_fields", []) or [])
                  + len(getattr(patch, "added_checks", []) or [])
                  + len(getattr(patch, "added_requirements", []) or [])
                  + len(getattr(patch, "checklist_items", []) or [])
                  + len(getattr(patch, "workflow_sections", []) or []))
    files = files or 1
    gen_tokens = int(params.get("generated_tokens", 0) or 0)
    test = 1.5 if params.get("test_required") else 0.0
    review = 1.0 if params.get("review_required") else 0.0
    return (gen_tokens / 2000.0 + diff_lines / 80.0 + files / 2.0 + fields / 3.0 + test + review)


def workload_tick_cost(workload: float) -> int:
    """1 + log2 growth for ordinary work; superlinear penalty past the hard threshold."""
    workload = max(0.0, float(workload))
    cost = 1 + math.ceil(math.log2(1.0 + workload))
    if workload > HARD_WORKLOAD_THRESHOLD:
        cost += math.ceil((workload - HARD_WORKLOAD_THRESHOLD) ** 1.5)
    return max(1, cost)


__all__ = ["estimate_workload", "workload_tick_cost", "HARD_WORKLOAD_THRESHOLD", "MAX_TICK_PER_ACTION"]
