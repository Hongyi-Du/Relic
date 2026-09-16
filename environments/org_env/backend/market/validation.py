"""Market-validation loop (v14 P5).

``run_market_trials`` is called after a release is published (and when an agent
collects post-launch feedback). It spawns a small burst of simulated external users
who evaluate the *current published product* and leave a market signal:

    real product quality  ->  trial satisfaction  ->  conversion + willingness-to-pay
                                                   ->  CustomerTicket (+ cust_issue if they churn)
                                                   ->  external_signal_event/customer_trial
                                                   ->  feedback episode  ->  (org fixes it) -> product change

The satisfaction/WTP are DERIVED FROM the four-layer readiness (runtime_ready is the
dominant factor) so the market cannot be won by a product that does not actually run
and ground its claims. ``market_summary`` turns the accumulated trials into the
``customers`` funding-milestone score (paying conversions + WTP, NOT raw ticket count,
so churned/complaint tickets can't game the milestone).
"""
from __future__ import annotations

import random
from typing import Any, Dict, List

from environments.org_env.backend.entities.economy import CustomerTrial, CustomerTicket

# (customer_type, persona, query, monthly_budget_in_tokens)
_PERSONAS = [
    ("smb_analyst", "SMB market analyst evaluating research tooling",
     "Summarize the 2026 outlook for our sector, with sources", 40.0),
    ("researcher", "Academic researcher checking citation grounding",
     "What does recent literature say about retrieval evaluation, with citations?", 26.0),
    ("enterprise_eval", "Enterprise buyer running a procurement trial",
     "Produce a sourced competitive-landscape brief on eval infra", 95.0),
    ("indie_writer", "Independent writer needing quick grounded drafts",
     "Draft a short evidence-backed explainer on agent benchmarks", 16.0),
]

# #5: for an OSS substrate (e.g. gitingest, a repo->text ingestion CLI) the trial is a real
# repo-ingestion job — NOT a research report — so personas/queries/feedback speak the product's
# language (submodules, ignore files, token budgets, offline, Docker/CI, cross-platform).
_OSS_PERSONAS = [
    ("platform_eng", "Platform engineer feeding repo context into an LLM pipeline",
     "Ingest our monorepo (submodules + .gitignore) into one clean text digest", 40.0),
    ("ml_engineer", "ML engineer turning codebases into model input",
     "Turn a large repo into a token-bounded digest with include/exclude patterns", 26.0),
    ("devtools_buyer", "Enterprise devtools buyer running a procurement trial",
     "Ingest big repos offline (token estimate, max-file-size) without crashing", 95.0),
    ("oss_maintainer", "OSS maintainer automating release context in CI",
     "Produce a reproducible repo digest in Docker/CI across Windows/macOS/Linux", 16.0),
]


def _latest_release(world: Any):
    rs = getattr(world, "repo_system", None)
    rels = getattr(getattr(rs, "repo", None), "releases", {}) or {} if rs else {}
    return next(reversed(list(rels.values())), None) if rels else None


def _top_gap(world: Any) -> str:
    gaps = getattr(world, "known_gaps", {}) or {}
    active = [g for g in gaps.values()
              if str(getattr(g, "status", "active")) in ("active", "regressed")]
    if not active:
        return ""
    active.sort(key=lambda g: 0 if str(getattr(g, "severity", "")).lower()
                in ("critical", "high") else 1)
    g = active[0]
    return (getattr(g, "description", "") or getattr(g, "title", "") or "").strip()


def product_quality(world: Any) -> Dict[str, Any]:
    """0..1 quality the market reacts to, dominated by runtime_ready (does it actually
    run end-to-end AND ground its claims). A product that does not even run is hard-capped
    low so the market can't be won on paperwork."""
    events = getattr(world, "events", []) or []
    merged = sum(1 for e in events
                 if e.get("type") == "repo_event" and e.get("subtype") == "pr_merged")
    try:
        from environments.org_env.coding.metrics import _readiness_layers
        r = _readiness_layers(world, merged)
    except Exception:
        r = {}
    v01 = r.get("v01", {}) if isinstance(r, dict) else {}

    def f(x):
        return 1.0 if x else 0.0
    q = (0.45 * f(r.get("runtime_ready"))
         + 0.18 * f(r.get("artifact_ready"))
         + 0.12 * f(r.get("workflow_ready"))
         + 0.10 * f(v01.get("readme_safe"))
         + 0.10 * f(v01.get("onboarding") or v01.get("demo_transcript"))
         + 0.05 * f(r.get("released")))
    if not v01.get("cli_runnable"):
        q = min(q, 0.30)            # doesn't even run -> the market won't pay
    out = {"quality": round(max(0.0, min(1.0, q)), 3),
           "runtime_ready": bool(r.get("runtime_ready")),
           "cli_runnable": bool(v01.get("cli_runnable")),
           "top_gap": _top_gap(world)}
    # OSS time-machine outcome metrics (brief §8): hidden/public behavior-test summary. These are
    # PRODUCT OUTCOMES; cached so the gate can call cheaply. They must NOT feed the
    # capability-formation detector (brief §11) — but they SHOULD drive the MARKET (below).
    out["quality_basis"] = "readiness"
    oss = _oss_product_metrics(world)
    if oss is not None:
        out.update(oss)
        # Wire the REAL behavior quality into the market signal: when the hidden/public behavior
        # tests actually ran, let the market react to whether the org FIXED real behaviors — not
        # just whether the product "runs + looks ready". Runnability stays a hard floor (a product
        # that doesn't run can't win the market). This closes the objective<->market loop so
        # fixing gitingest behaviors raises satisfaction/WTP/conversion (and is what makes the
        # customers milestone reachable for an OSS substrate).
        if "oss_hidden_pass_rate" in oss:
            hp = float(oss.get("oss_hidden_pass_rate") or 0.0)
            pub = float(oss.get("oss_public_test_pass_rate", 1.0) or 0.0)
            q_real = 0.65 * hp + 0.15 * pub + 0.20 * out["quality"]   # behavior-dominated
            if not out["cli_runnable"]:
                q_real = min(q_real, 0.30)
            # #5: an OPEN high-severity issue the hidden tests DON'T cover (e.g. Docker launch broken)
            # should still dent the market — real users churn on it even if covered behaviors pass.
            _hi = 0
            for _a in (getattr(world, "product_artifacts", {}) or {}).values():
                if getattr(_a, "artifact_type", "") != "issue":
                    continue
                if getattr(_a, "status", "open") not in ("open", "in_progress", "reopened"):
                    continue
                if str(getattr(_a, "artifact_id", "")).startswith(("rel_blocker_", "proto_violation_")):
                    continue
                if (str(getattr(_a, "priority", "")).lower() in ("high", "critical")
                        or str(getattr(_a, "severity", "")).lower() in ("high", "major", "critical")):
                    _hi += 1
            if _hi:
                q_real *= max(0.6, 1.0 - 0.08 * _hi)
                out["open_high_sev_issues"] = _hi
            out["readiness_quality"] = out["quality"]                # keep the readiness term visible
            out["quality"] = round(max(0.0, min(1.0, q_real)), 3)
            out["quality_basis"] = "oss_real_behavior"
    return out


def _oss_product_metrics(world: Any):
    """Hidden/public test summary for an OSS substrate, or None for the synthetic substrate.
    Gated on ORG_OSS_HIDDEN_TESTS (run real subprocess tests) so unit tests stay fast/offline;
    cached on the world by product-tree hash so it isn't recomputed every sweep."""
    try:
        from environments.org_env.product.substrates.eval_assets import (
            is_oss_substrate, materialize_and_run_oss_hidden_tests, run_oss_public_tests,
            oss_eval_enabled)
        from environments.org_env.product.materialize import (export_product_repo, _repo_hash)
    except Exception:
        return None
    if not is_oss_substrate(world):
        return None
    # evaluator_config (formal default) OR ORG_OSS_HIDDEN_TESTS override (brief review §4): the
    # objective metrics are not gated behind a manual env var in a formal run.
    if not oss_eval_enabled(world, "run_oss_hidden_tests", "ORG_OSS_HIDDEN_TESTS"):
        # cheap/offline default: report only that the substrate is OSS (no subprocess run)
        return {"oss_substrate": True}
    cache = world.__dict__.setdefault("_oss_metrics_cache", {})
    try:
        key = _repo_hash(world, prefer_mainline=True)
    except Exception:
        key = None
    if key and key in cache:
        return cache[key]
    import tempfile
    hid = materialize_and_run_oss_hidden_tests(world, prefer_mainline=True)
    dest = tempfile.mkdtemp(prefix="oss_public_")
    try:
        export_product_repo(world, dest, prefer_mainline=True)
        pub = run_oss_public_tests(world, dest)
    finally:
        import shutil
        shutil.rmtree(dest, ignore_errors=True)
    res = {
        "oss_substrate": True,
        "oss_hidden_pass_rate": hid.get("pass_rate", 0.0),
        "oss_issue_fix_rate": hid.get("issue_fix_rate", 0.0),     # TESTED issues only (back-compat)
        "oss_regression_pass_rate": pub.get("pass_rate", 1.0),
        "oss_public_test_pass_rate": pub.get("pass_rate", 1.0),
        "oss_cli_ok": bool(pub.get("ok", True)),
        "oss_hidden_total": hid.get("total", 0),
        "oss_hidden_passed": hid.get("passed", 0),
    }
    # brief review §5/§6: separate OBJECTIVELY-VERIFIED fixes (issues with a hidden behavior test)
    # from mere activity on untested issues + held-out transfer — so the headline metric can't be
    # mistaken for "all public issues objectively evaluated".
    res.update(_oss_issue_breakdown(world, hid))
    if key:
        cache[key] = res
    return res


def _oss_issue_breakdown(world: Any, hid: Dict[str, Any]) -> Dict[str, Any]:
    """Split public issues into TESTED (have a hidden behavior test → objective fix) vs UNTESTED
    (only task/patch/review activity → unverified progress), plus the held-out count. Prevents a
    reviewer from reading ``oss_issue_fix_rate`` as 'all 10 issues objectively evaluated' (review §5/§6)."""
    from environments.org_env.product.substrates.eval_assets import oss_eval_assets
    a = oss_eval_assets(world) or {}
    tested = {iid for s in (a.get("hidden_test_specs") or [])
              for iid in (getattr(s, "issue_ids", None) or [])}
    arts = getattr(world, "product_artifacts", {}) or {}
    # review fix §4: scope to the REAL OSS historical issues (the issue stream) — do NOT count
    # internally-generated issues (issue_N / rel_blocker_* / proto_violation_*) as public issues.
    stream_ids = {e.get("issue_id") for e in (world.__dict__.get("_oss_issue_stream") or [])}
    public = {aid for aid, art in arts.items()
              if getattr(art, "artifact_type", "") == "issue"
              and (aid in stream_ids if stream_ids else True)}
    tested_public = sorted(public & tested)
    untested = sorted(public - tested)
    issue_fix = hid.get("issue_fix") or {}
    tested_fixed = sum(1 for iid in tested_public if issue_fix.get(iid) is True)
    try:
        from environments.org_env.product.substrates.issue_stream import _has_post_issue_work
        progressed = sum(1 for iid in untested if _has_post_issue_work(world, iid))
    except Exception:
        progressed = 0
    return {
        "oss_tested_issue_ids": tested_public,
        "oss_tested_issue_count": len(tested_public),
        "oss_tested_issue_fix_rate": (round(tested_fixed / len(tested_public), 3)
                                      if tested_public else 0.0),
        "oss_untested_issue_ids": untested,
        "oss_untested_issue_count": len(untested),
        # activity-only signal on issues WITHOUT an objective test — NOT a verified fix
        "oss_unverified_issue_progress": (round(progressed / len(untested), 3) if untested else 0.0),
        "oss_heldout_issue_count": len(a.get("heldout_issues") or []),
    }


def run_market_trials(world: Any, tick: int, n: int = 3,
                      trigger: str = "release") -> List[CustomerTrial]:
    """Spawn ``n`` external trials against the latest *published* release. Returns the
    new CustomerTrials (also stored on ``world.trials``); writes one CustomerTicket per
    trial and a ``cust_issue`` product artifact for each churned user, and appends a
    ``customer_trial`` external_signal_event (observed into a feedback episode)."""
    # The HCI project workspace uses the organization's real task/repo/review
    # loops. It must not fabricate customers or churn work merely because a
    # release was published. The simulation profile retains this mechanism.
    if getattr(world, "customer_market_enabled", True) is False:
        return []
    rel = _latest_release(world)
    if rel is None:
        return []                  # no published product -> no market yet
    version = getattr(rel, "version", "") or ""
    pq = product_quality(world)
    quality, top_gap = pq["quality"], pq["top_gap"]

    trials: Dict[str, Any] = world.__dict__.setdefault("trials", {})
    tickets: Dict[str, Any] = world.__dict__.setdefault("tickets", {})
    if getattr(world, "events", None) is None:
        world.events = []
    # (no world class defines ``.tick`` — the old ``getattr(world, "tick", tick) or tick``
    # always resolved to the passed tick; keep exactly that, without the dead lookup)
    rng = random.Random(int(tick) * 1009 + len(trials) * 31 + 7)

    try:
        from environments.org_env.product.substrates.eval_assets import is_oss_substrate
        _oss = is_oss_substrate(world)
    except Exception:
        _oss = False
    personas = _OSS_PERSONAS if _oss else _PERSONAS

    out: List[CustomerTrial] = []
    for _ in range(max(1, int(n))):
        ct_type, persona, query, budget = personas[len(trials) % len(personas)]
        sat = max(0.0, min(1.0, quality + rng.uniform(-0.12, 0.12)))
        if sat >= 0.6:
            outcome, converted = "converted", True
            wtp = round(budget * (0.5 + 0.5 * sat), 1)
            fb = (("ingested our repo end-to-end and the digest was correct and usable enough to pay for; "
                   if _oss else "ran end-to-end and the sourced output was trustworthy enough to pay for; ")
                  + f"would adopt for {ct_type.replace('_', ' ')} work")
            sev, status, resp = "minor", "resolved", "resolved"
        elif sat >= 0.4:
            outcome, converted = "interested", False
            wtp = round(budget * 0.3 * sat, 1)
            fb = ("promising but not yet worth paying for — "
                  + (top_gap or ("some repos still fail before we commit" if _oss
                                 else "needs more reliable grounding before we commit")))
            sev, status, resp = "minor", "open", "pending"
        else:
            outcome, converted = "rejected", False
            wtp = 0.0
            fb = ("could not rely on it: "
                  + (top_gap or ("the CLI crashed / produced a wrong digest" if _oss
                                 else "output was not grounded or the CLI did not run cleanly"))
                  + " — churned")
            sev, status, resp = "major", "open", "pending"

        tid = f"ticket_{len(tickets) + 1}"
        while tid in tickets:
            tid = f"ticket_{len(tickets) + 1}_{rng.randint(0, 9999)}"
        trial_id = f"trial_{len(trials) + 1}"
        while trial_id in trials:
            trial_id = f"trial_{len(trials) + 1}_{rng.randint(0, 9999)}"

        ticket = CustomerTicket(
            ticket_id=tid, customer_type=ct_type, complaint_or_request=fb,
            severity=sev, topic="market_validation", status=status, response_status=resp)
        ticket.__dict__.update({"created_tick": tick, "trial_id": trial_id,
                                "willingness_to_pay": wtp, "converted": converted,
                                "satisfaction": round(sat, 3)})
        tickets[tid] = ticket

        trial = CustomerTrial(
            trial_id=trial_id, customer_type=ct_type, persona=persona, query=query,
            release_version=version, outcome=outcome, satisfaction=round(sat, 3),
            willingness_to_pay=wtp, converted=converted, feedback=fb,
            linked_ticket_id=tid, created_tick=tick)
        trials[trial_id] = trial
        out.append(trial)

        world.events.append({
            "type": "external_signal_event", "subtype": "customer_trial", "tick": tick,
            "trial_id": trial_id, "ticket_id": tid, "customer_type": ct_type,
            "outcome": outcome, "satisfaction": round(sat, 3), "willingness_to_pay": wtp,
            "converted": converted, "release_version": version, "object_ids": [tid],
            "summary": f"{persona}: {fb}"[:200]})

        if outcome == "rejected":
            _spawn_customer_issue(world, tick, rel, ct_type, top_gap or fb)
    return out


def _spawn_customer_issue(world: Any, tick: int, rel: Any, ct_type: str, problem: str) -> None:
    """A churned customer leaves a concrete, fixable product issue (mirrors
    collect_post_launch_feedback's cust_issue), so the feedback loop has a real
    product-change target instead of evaporating."""
    try:
        from environments.org_env.product.objects import ProductArtifact
    except Exception:
        return
    arts = getattr(world, "product_artifacts", None)
    if arts is None:
        return
    seq = world.__dict__.setdefault("_cust_issue_seq", 0) + 1
    world._cust_issue_seq = seq
    iid = f"cust_issue_{seq}"
    while iid in arts:
        seq += 1
        world._cust_issue_seq = seq
        iid = f"cust_issue_{seq}"
    arts[iid] = ProductArtifact(
        artifact_id=iid, artifact_type="issue",
        title=f"churn: {ct_type} — {problem[:48]}", status="open",
        problem=problem, summary=problem, priority="high",
        created_at_tick=tick, updated_at_tick=tick)
    prod = getattr(world, "product", None)
    if prod is not None:
        prod.artifact_ids.append(iid)
        prod.open_issue_ids.append(iid)
    if rel is not None and hasattr(rel, "post_launch_feedback_ids"):
        rel.post_launch_feedback_ids.append(iid)


def market_summary(world: Any) -> Dict[str, Any]:
    """Aggregate the accumulated market signal. ``customers_score`` (0..1) drives the
    funding ``customers`` milestone: it rewards PAYING conversions + willingness-to-pay,
    not raw ticket count, so churned/complaint tickets cannot game the milestone."""
    trials = list((getattr(world, "trials", {}) or {}).values())
    tickets = list((getattr(world, "tickets", {}) or {}).values())
    n = len(trials)
    paying = sum(1 for t in trials if getattr(t, "converted", False))
    # positive human evaluations (rating >= 4 via the product/feedback API) count too
    human_pos = sum(1 for tk in tickets
                    if float(getattr(tk, "rating", 0) or tk.__dict__.get("rating", 0) or 0) >= 4.0)
    conversions = paying + human_pos
    wtp = sum(float(getattr(t, "willingness_to_pay", 0) or 0) for t in trials)
    avg_sat = round(sum(float(getattr(t, "satisfaction", 0) or 0) for t in trials) / n, 3) if n else 0.0
    score = min(1.0, 0.30 * conversions + min(0.25, wtp / 120.0) + min(0.15, n / 12.0))
    return {
        "trials": n, "conversions": conversions, "paying_conversions": paying,
        "human_positive": human_pos, "total_wtp": round(wtp, 1),
        "avg_satisfaction": avg_sat,
        "conversion_rate": round(paying / n, 3) if n else 0.0,
        "customers_score": round(score, 3),
    }


__all__ = ["product_quality", "run_market_trials", "market_summary"]
