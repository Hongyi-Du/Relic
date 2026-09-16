"""Proposal / tool FAMILY fingerprint + active-tool cap (v11 P2: stop proposal/tool sprawl).

The v10 finding was not "too few proposals" but "too many shallow, fast, repetitive" ones —
e.g. an evidence gate, a claim-evidence checker, a source-credibility gate and a minimal-
evidence protocol all minted separately. A *family* groups proposals/tools that regulate the
SAME organizational concern. The manager caps ACTIVE tools per family and folds a same-family
near-duplicate into support/amend of an existing tool instead of minting another, so the org
converges on a few strong mechanisms rather than a long tail of one-off tools.
"""
from __future__ import annotations

from typing import Any

FAMILIES = (
    "evidence_governance", "release_engineering", "debugging", "budget_governance",
    "customer", "ownership", "experiment", "review_merge", "docs", "other",
)

# keyword -> family. Multi-word keys are stronger (count x2) so a specific phrase
# ("source credibility") beats an incidental single word ("source").
_FAMILY_KEYWORDS = {
    "evidence_governance": ("evidence", "claim", "source credib", "credibility", "citation",
                            "traceab", "verif", "fact-check", "provenance", "unsupported"),
    "release_engineering": ("release", "launch", "readiness", "rollback", "ship it",
                            "deploy", "publish", "release gate"),
    "debugging": ("debug", "smoke", "ci fail", "stack trace", "traceback", "localize",
                  "bug fix", "regression", "integration error", "broken build", "blocker"),
    "budget_governance": ("budget", "token", "cost", "spend", "expensive", "quota", "pricing"),
    "customer": ("customer", "user feedback", "complaint", "triage", "support ticket", "churn"),
    "ownership": ("owner", "ownership", "assign owner", "accountab", "responsib", "raci"),
    "experiment": ("experiment", "pilot", "baseline", "ablation", "benchmark", "reproduc"),
    "review_merge": ("review", "approve", "sign-off", "signoff", "merge", "pull request",
                     "before merge"),
    "docs": ("readme", "documentation", "report quality", "changelog", "writeup", "doc clarity"),
}


def _text(p: Any) -> str:
    """Flatten a Proposal OR a ToolSpec into searchable text (handles both field sets)."""
    parts = [
        getattr(p, "title", "") or getattr(p, "name", ""),
        getattr(p, "summary", "") or getattr(p, "description", ""),
        getattr(p, "target_problem", ""),
        getattr(p, "proposed_solution", "") or getattr(p, "enforcement_rule", ""),
        " ".join(getattr(p, "required_actions", []) or []),
        " ".join(getattr(p, "required_artifacts", []) or getattr(p, "required_fields", []) or []),
    ]
    return " ".join(str(x) for x in parts if x).lower()


def classify_family(p: Any) -> str:
    """Best-matching family by weighted keyword hits; 'other' when nothing matches."""
    text = _text(p)
    best, best_score = "other", 0
    for fam, kws in _FAMILY_KEYWORDS.items():
        score = sum(2 if " " in kw else 1 for kw in kws if kw in text)
        if score > best_score:
            best, best_score = fam, score
    return best


__all__ = ["FAMILIES", "classify_family"]
