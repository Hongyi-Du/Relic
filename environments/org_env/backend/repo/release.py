"""Release gates (v5 §5) — a ReleaseCandidate must pass these (or have them waived)
before it can be published. Gates read live world state (product artifacts, CI,
issues, protocols), so they live here rather than on the world-agnostic repo system.
"""
from __future__ import annotations

from typing import Any, Dict, List

RELEASE_GATES = (
    "gate_no_critical_gaps_remaining",
    "gate_no_unresolved_high_risk_issue",
    "gate_readme_claims_audited",
    "gate_claim_evidence_protocol_active_or_pending",
    "gate_source_credibility_supported",
    "gate_report_quality_checklist_exists",
    "gate_eval_metrics_defined_or_marked",
    "gate_ci_passed_for_included_prs",
    "gate_known_limitations_documented",
    "gate_smoke_test_passes",
)

# OSS time-machine ships a real third-party tool, so product-specific claim/report
# gates do not apply — and requiring "no open high-risk issue / no critical gap"
# would forbid ever releasing a v0.1.x product that legitimately ships WITH open historical issues.
# A real OSS release just has to BUILD + pass CI; objective quality is judged separately by the
# hidden behavior tests feeding product_quality -> the market. So the OSS publish gate is minimal;
# this is what lets a release actually ship -> trigger the external market/community loop.
OSS_RELEASE_GATES = (
    "gate_smoke_test_passes",
    "gate_ci_passed_for_included_prs",
    "gate_oss_hidden_tests_passed",
)


RELEASE_APPROVER_MIN = 2
_LEAD_ROLES = ("founder", "cofounder")
_EVIDENCE_ROLE = "reliability"


def release_approval_met(world: Any, candidate: Any) -> bool:
    """Whether a candidate has the sign-off its organization can actually give.

    The rule is two approvers including a lead and an evidence owner, which is
    the right rule for a staffed organization and unreachable for a one-person
    one: a solo founder can only ever be one approver and can never be the
    reliability role. A release candidate then sits in `under_review` forever,
    `_open_rc` keeps returning it, and every later create_release_candidate
    fails as `rc_in_flight` — so release_count for that condition is zero by
    arithmetic, not by conduct, and reads as "a single agent cannot ship".

    Both requirements are therefore capped by the roster: a role that does not
    exist cannot be demanded, and more approvers than there are people cannot be
    required. Where the roles do exist nothing changes.
    """
    agents = getattr(world, "agents", {}) or {}
    approvals = [a for a in (getattr(candidate, "approvals", []) or []) if a in agents]
    roster_roles = {str(getattr(agent, "role", "")) for agent in agents.values()}
    approver_roles = {str(getattr(agents[a], "role", "")) for a in approvals}

    if len(approvals) < min(RELEASE_APPROVER_MIN, max(1, len(agents))):
        return False
    if roster_roles & set(_LEAD_ROLES) and not (approver_roles & set(_LEAD_ROLES)):
        return False
    if _EVIDENCE_ROLE in roster_roles and _EVIDENCE_ROLE not in approver_roles:
        return False
    return True


def _oss_hidden_gate_result(world: Any) -> dict:
    """Cached (per mainline tree hash) hidden-behavior-test result for the release gate — so the
    OSS objective metric is a FIRST-CLASS release signal, not something only measured out-of-band."""
    from environments.org_env.product.substrates.eval_assets import materialize_and_run_oss_hidden_tests
    cache = world.__dict__.setdefault("_oss_hidden_gate_cache", {})
    key = None
    try:
        from environments.org_env.product.materialize import _repo_hash
        key = _repo_hash(world, prefer_mainline=True)
    except Exception:
        key = None
    if key is not None and key in cache:
        return cache[key]
    res = materialize_and_run_oss_hidden_tests(world, prefer_mainline=True)
    if key is not None:
        cache.clear()
        cache[key] = res
    return res


def release_gates_for(world: Any):
    """Pick the release gate set appropriate to the product substrate (see OSS_RELEASE_GATES)."""
    try:
        from environments.org_env.product.substrates.eval_assets import is_oss_substrate
        if is_oss_substrate(world):
            return OSS_RELEASE_GATES
    except Exception:
        pass
    return RELEASE_GATES

# what to do about a failing gate -> so a blocker can become a concrete issue/task (spec #3)
_GATE_ACTION = {
    "gate_no_critical_gaps_remaining": "resolve the remaining critical product gaps before release",
    "gate_no_unresolved_high_risk_issue": "close or downgrade the open high-risk issues",
    "gate_readme_claims_audited": "audit README and soften/limit any overclaim",
    "gate_claim_evidence_protocol_active_or_pending": "adopt a claim-evidence protocol",
    "gate_source_credibility_supported": "add source-credibility scoring to the source tracker",
    "gate_report_quality_checklist_exists": "create a report quality checklist",
    "gate_eval_metrics_defined_or_marked": "define eval metrics or explicitly mark eval incomplete",
    "gate_ci_passed_for_included_prs": "fix CI on the included PRs",
    "gate_known_limitations_documented": "document known limitations in the release notes",
    "gate_smoke_test_passes": "make the real product run: `python smoke_check.py` must exit 0",
    "gate_oss_hidden_tests_passed": "fix the behavior behind the failing hidden tests (the covered issues)",
}

# README overclaim detection — v13 P1: two layers, negation-aware, so legitimate limitation
# language ("not guaranteed to be correct", "requires human review") is NOT flagged as an
# overclaim (the old bare-substring matcher false-positived on "guaranteed" inside a disclaimer
# and pushed agents to DELETE honest disclaimers to pass the gate).
_HARD_OVERCLAIMS = (
    "production ready", "production-ready", "fully implemented", "fully verified",
    "fully validated", "100% accurate", "100% correct", "always correct",
    "guaranteed correct", "guaranteed accurate",
    "all claims are supported", "no known limitations",
)
# NOTE: "evidence-grounded reports" is the product's PURPOSE/domain term, not an overclaim — the
# v13 README uses it responsibly ("not guaranteed ... fully evidence-grounded" + vision framing),
# so the blunt substring was a false positive that blocked release + pushed deleting honest text.
# explicit safe-disclaimer phrases (never overclaims even if they contain a trigger word)
_SAFE_DISCLAIMERS = (
    "not guaranteed", "no guarantee", "cannot guarantee", "not production-ready",
    "not production ready", "requires human review", "draft-level", "draft level",
    "may be incomplete", "not fully", "may be incorrect",
)
# negation / limitation markers — if any appears IN THE SAME SENTENCE as a bare "guarantee"
# (before OR after it), the word is part of a disclaimer, not a positive promise. v14b: the old
# matcher only looked 14 chars BEFORE the word, so "implied a guarantee the claim tracker does
# NOT enforce" (trailing negation) was a false positive — it blocked release for 14 days and
# pushed agents to delete honest limitation text.
_OVERCLAIM_NEG = (
    "not ", "no ", "cannot ", "n't", "without ", "never ", "nor ",
    "does not", "do not", "is not", "are not", "cannot be", "not be",
    "implied", "imply", "limitation", "disclaim", "no warranty", "without warranty",
)


def _sentences(text: str):
    """Split README text into sentence/line-level clauses for context-aware scanning."""
    import re
    return [s for s in re.split(r"[.!?;\n]+", text) if s.strip()]


def readme_overclaim(content: str) -> str:
    """Return the offending overclaim phrase, or '' if the README is honestly scoped.

    v14b — two layers, context-aware:
      1. hard overclaims (e.g. "100% accurate", "guaranteed correct") are flagged unconditionally;
      2. a bare "guarantee(d)" is only an overclaim when its SENTENCE has no negation/limitation
         marker (before or after). Fenced code blocks and HTML comments are ignored (they are
         examples/notes, not product claims)."""
    import re
    raw = content or ""
    # drop fenced code blocks and HTML comments — examples / notes, not product promises
    stripped = re.sub(r"```.*?```", " ", raw, flags=re.S)
    stripped = re.sub(r"<!--.*?-->", " ", stripped, flags=re.S)
    c = stripped.lower()
    for ph in _HARD_OVERCLAIMS:
        if ph in c:
            return ph
    # bare "guarantee(d)" only counts when its sentence reads as a POSITIVE promise
    for sent in _sentences(c):
        if "guarantee" not in sent:
            continue
        if any(neg in sent for neg in _OVERCLAIM_NEG):
            continue
        if any(s in sent for s in _SAFE_DISCLAIMERS):
            continue
        return "guaranteed"
    return ""


def evaluate_release_gates(world: Any, rc) -> List[Dict[str, Any]]:
    """Return [{gate, passed, detail, recommended_action}] for each required gate."""
    from environments.org_env.product.objects import artifact_purpose
    arts = getattr(world, "product_artifacts", {}) or {}
    repo = world.repo_system.repo
    out: List[Dict[str, Any]] = []

    def add(gate: str, ok: bool, detail: str = "") -> None:
        out.append({"gate": gate, "passed": bool(ok), "detail": detail,
                    "recommended_action": "" if ok else _GATE_ACTION.get(gate, "")})

    def _cap(purpose: str, *keywords: str) -> bool:
        for a in arts.values():
            if getattr(a, "artifact_type", "") == "issue":
                continue
            if artifact_purpose(getattr(a, "linked_file_path", "") or a.artifact_id) == purpose:
                caps = [str(c).lower() for c in (getattr(a, "capabilities", []) or [])]
                if any(k in c for k in keywords for c in caps):
                    return True
        return False

    def _content_has(purpose: str, *keywords: str) -> bool:
        """Grounded check: the REAL file text of the artifact contains a keyword."""
        for a in arts.values():
            if getattr(a, "artifact_type", "") == "issue":
                continue
            if artifact_purpose(getattr(a, "linked_file_path", "") or a.artifact_id) == purpose:
                c = (getattr(a, "content", "") or "").lower()
                if any(k in c for k in keywords):
                    return True
        return False

    gates = list(rc.required_gates or [])   # explicit [] = no gates (don't fall back to all)
    for g in gates:
        if g == "gate_no_critical_gaps_remaining":
            remaining = int((getattr(world, "product_readiness", {}) or {}).get("critical_gaps_remaining", 0))
            add(g, remaining == 0, "" if remaining == 0 else f"{remaining} critical gap(s) still active")
        elif g == "gate_source_credibility_supported":
            # grounded: the real source_tracker file must mention credibility (tag is a fallback)
            ok = _content_has("source_tracker", "credibility") or _cap("source_tracker", "credibility")
            add(g, ok, "" if ok else "source_tracker.py has no credibility scoring")
        elif g == "gate_eval_metrics_defined_or_marked":
            built = (_content_has("eval", "coverage", "metric", "grounding", "unsupported")
                     or _cap("eval", "metric", "grounding", "coverage"))
            marked = any("eval" in str(l).lower() for p in (getattr(world, "patches", {}) or {}).values()
                         for l in ((getattr(p, "added_limitations", []) or [])
                                   + (getattr(p, "known_limitations", []) or [])))
            add(g, built or marked, "" if (built or marked) else "eval metrics neither defined nor marked incomplete")
        elif g == "gate_readme_claims_audited":
            readme = next((a for a in arts.values() if (a.linked_file_path or "") == "README.md"), None)
            audited = bool(readme and any(k in (s or "").lower()
                                          for s in (getattr(readme, "change_summaries", []) or [])
                                          for k in ("overclaim", "softened", "limitation")))
            overclaim = readme_overclaim(getattr(readme, "content", "") or "") if readme else ""
            no_overclaim = readme is not None and not overclaim
            ok = audited and no_overclaim
            add(g, ok, "" if ok else ("README claims not audited" if not audited
                                      else f"README overclaim phrase: '{overclaim}'"))
        elif g == "gate_report_quality_checklist_exists":
            ok = any(artifact_purpose(a.linked_file_path or a.artifact_id) == "report_quality"
                     for a in arts.values())
            add(g, ok, "" if ok else "no report quality checklist")
        elif g == "gate_ci_passed_for_included_prs":
            prs = [repo.pull_requests[p] for p in rc.included_pr_ids if p in repo.pull_requests]
            ok = bool(prs) and all(pr.ci_passed for pr in prs)
            add(g, ok, "" if ok else "an included PR has no passing CI")
        elif g == "gate_known_limitations_documented":
            ok = any(getattr(p, "added_limitations", None) or getattr(p, "known_limitations", None)
                     for p in (getattr(world, "patches", {}) or {}).values())
            add(g, ok, "" if ok else "no documented limitations")
        elif g == "gate_no_unresolved_high_risk_issue":
            # v8f P0a: exclude the gate's OWN rel_blocker_* meta-issues — counting them made
            # the gate self-referential (it created high-risk issues that then failed it, so
            # it could never pass). Real product issues + escalated disputes still count.
            high = [a for a in arts.values() if a.artifact_type == "issue" and a.status == "open"
                    and str(getattr(a, "priority", "")).lower() in ("high", "critical")
                    and not str(a.artifact_id).startswith("rel_blocker_")]
            add(g, not high, "" if not high else f"{len(high)} open high-risk issue(s)")
        elif g == "gate_claim_evidence_protocol_active_or_pending":
            specs = getattr(world.proposal_manager, "protocol_specs", {}) or {}
            live = getattr(world.protocol_registry, "protocols", {}) or {}
            ok = bool(specs) or any(("evidence" in p.protocol_type.lower()
                                     or "claim" in p.protocol_type.lower()
                                     or "review" in p.protocol_type.lower()) for p in live.values())
            add(g, ok, "" if ok else "no claim-evidence/review protocol active or pending")
        elif g == "gate_smoke_test_passes":
            # grounded: export the real (mainline) tree and run `python smoke_check.py`.
            # v11 §8 red light: a non-crashing smoke is NOT enough — it must also be
            # semantically sound (claims grounded in sources, eval runs cleanly), else a
            # "green" run hides credibility_score=0 / eval_failed and the org keeps marking
            # the gap resolved without actually closing it.
            from environments.org_env.product.materialize import (
                release_smoke, smoke_error_brief, smoke_quality_issue)
            sm = release_smoke(world)
            quality = smoke_quality_issue(sm)
            ok = bool(sm.get("ok")) and not quality
            brief = smoke_error_brief(sm) or quality
            # this gate runs the MAINLINE smoke — stash it on a DEDICATED field so it never clobbers the
            # WORKING-tree CI error (_build_error) that the code editor's debug loop needs. Previously a
            # clean mainline here wiped the live working-tree ImportError, so agents never saw the break.
            world.__dict__["_mainline_smoke_error"] = "" if ok else brief
            add(g, ok, "" if ok else ("smoke failed: " + brief))
        elif g == "gate_oss_hidden_tests_passed":
            # OSS objective hard metric as a first-class release gate (#1). Records the hidden pass rate
            # + tested-issue-fix-rate on the world so readiness/telemetry/market read ONE source of
            # truth. Default is SOFT (a <100% release ships as beta-with-known-limitations, so early
            # betas still reach the market); ORG_OSS_HIDDEN_GATE_STRICT makes it a HARD block.
            import os
            from environments.org_env.product.substrates.eval_assets import (
                is_oss_substrate, oss_eval_enabled)
            if not is_oss_substrate(world) or not oss_eval_enabled(
                    world, "run_oss_hidden_tests", "ORG_OSS_HIDDEN_TESTS"):
                add(g, True, "n/a (hidden tests off / non-OSS)")
            else:
                hr = _oss_hidden_gate_result(world)
                total, passed = int(hr.get("total", 0) or 0), int(hr.get("passed", 0) or 0)
                world.__dict__["_oss_release_hidden"] = {
                    "passed": passed, "total": total, "pass_rate": hr.get("pass_rate", 0.0),
                    "issue_fix_rate": hr.get("issue_fix_rate", 0.0), "issue_fix": hr.get("issue_fix", {})}
                all_pass = total > 0 and passed == total
                strict = (os.environ.get("ORG_OSS_HIDDEN_GATE_STRICT", "") or "").lower() in ("1", "true", "yes", "on")
                if all_pass:
                    add(g, True, f"hidden behavior tests {passed}/{total}")
                elif strict:
                    add(g, False, f"hidden behavior tests {passed}/{total} — fix the covered issues")
                else:
                    add(g, True, f"beta: hidden behavior tests {passed}/{total} (known limitation)")
        else:
            add(g, True, "unknown gate (auto-pass)")
    return out


__all__ = ["RELEASE_GATES", "OSS_RELEASE_GATES", "release_gates_for", "evaluate_release_gates"]
