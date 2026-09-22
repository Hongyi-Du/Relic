#!/usr/bin/env python3
"""Build the per-tick timeline JSON from OrgEnv run logs (cattrs B3 seed 4013).

The checked-in `data/cattrs_b3_4013.json` is what the demo loads; this script is
only needed to regenerate it, and it reads raw run directories that live outside
the repository. Point `--runs` at wherever yours are.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_RUNS = REPO / "scratch/log"
OUT = Path(__file__).resolve().parents[1] / "data/cattrs_b3_4013.json"

RUN_NAMES = {
    "b0": "20260913-110642_org_mx4013_cattrs_b0_seed4013_llm",
    "b1": "20260913-115845_org_mx4013_cattrs_b1_seed4013_llm",
    "b2": "20260913-115856_org_mx4013_cattrs_b2_seed4013_llm",
    "b3": "20260913-161353_org_mx4013_cattrs_b3_seed4013_llm",
}


TICK_RE = re.compile(
    r"\[t(\d+)\s+(\d{2}:\d{2})\s+([^\]]+)\]\s+.*?LLM calls=(\d+)\s+\|\s*(.*)$"
)

HIGHLIGHT_VERBS = {
    "edit_repo_file",
    "commit_patch",
    "merge_pr",
    "open_pr",
    "review_pr",
    "approve_pr",
    "run_ci",
    "run_public_tests",
    "approve_proposal",
    "reject_proposal",
    "propose_protocol",
    "amend_protocol",
    "publish_product_release",
    "create_release_candidate",
}

# Which room of the virtual office a verb belongs to.
ZONE_BY_VERB = {
    "edit_repo_file": "code",
    "commit_patch": "code",
    "work_on_task": "code",
    "write_design_note": "code",
    "open_pr": "review",
    "review_pr": "review",
    "formal_pr_review": "review",
    "approve_pr": "review",
    "request_changes": "review",
    "ask_for_review": "review",
    "merge_pr": "review",
    "review_doc": "review",
    "run_ci": "ci",
    "run_public_tests": "ci",
    "run_eval_stub": "ci",
    "run_cheap_pilot": "ci",
    "run_launch_readiness_check": "ci",
    "create_eval_stub": "ci",
    "approve_proposal": "protocol",
    "reject_proposal": "protocol",
    "propose_protocol": "protocol",
    "amend_protocol": "protocol",
    "request_proposal_changes": "protocol",
    "summarize_decision": "protocol",
    "record_meeting_notes": "protocol",
    # Meetings are not governance: keep them out of the protocol room so that
    # room stays a clean signal for the institutional layer B2 does not have.
    "attend_meeting": "desk",
    "schedule_meeting": "desk",
    "publish_product_release": "ship",
    "create_release_candidate": "ship",
    "approve_release_candidate": "ship",
    "share_external_post": "ship",
    "respond_to_public_comment": "ship",
    "monitor_customer_feedback": "ship",
    "dogfood_product": "ship",
    "read_feed": "ship",
}

# Counters that tick up live so the HUD is not frozen between checkpoints.
COUNTER_BY_VERB = {
    "merge_pr": "merges",
    "publish_product_release": "releases",
    "commit_patch": "commits",
    "edit_repo_file": "edits",
    "run_ci": "ci_runs",
    "run_public_tests": "ci_runs",
    "approve_proposal": "protocol_votes",
    "reject_proposal": "protocol_votes",
    "propose_protocol": "protocols_proposed",
    "amend_protocol": "protocols_amended",
    "review_pr": "reviews",
    "formal_pr_review": "reviews",
    "approve_pr": "reviews",
}

COUNTER_KEYS = (
    "merges",
    "releases",
    "commits",
    "edits",
    "ci_runs",
    "protocol_votes",
    "protocols_proposed",
    "protocols_amended",
    "reviews",
)


def parse_summary(path: Path) -> list[dict]:
    ticks: list[dict] = []
    totals = dict.fromkeys(COUNTER_KEYS, 0)
    text = path.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        m = TICK_RE.search(line.strip())
        if not m:
            continue
        tick, clock, phase, llm_calls, rest = m.groups()
        actions = []
        for part in rest.split(","):
            part = part.strip()
            if ":" not in part:
                continue
            agent, verb = part.split(":", 1)
            agent, verb = agent.strip(), verb.strip()
            counter = COUNTER_BY_VERB.get(verb)
            if counter:
                totals[counter] += 1
            actions.append(
                {
                    "agent": agent,
                    "verb": verb,
                    "zone": ZONE_BY_VERB.get(verb, "desk"),
                    "highlight": verb in HIGHLIGHT_VERBS,
                }
            )
        hour = int(clock.split(":")[0])
        ticks.append(
            {
                "tick": int(tick),
                "clock": clock,
                "hour": hour,
                "phase": phase.strip(),
                "day": (int(tick) - 1) // 24 + 1,
                "night": hour < 7 or hour >= 20,
                "llm_calls": int(llm_calls),
                "actions": actions,
                "totals": dict(totals),
            }
        )
    return ticks


def load_checkpoints(path: Path) -> list[dict]:
    rows: list[dict] = []
    metrics_path = path / "figure_metrics.jsonl"
    if not metrics_path.is_file():
        return rows
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        p = row.get("protocols") or {}
        d = row.get("delivery") or {}
        b = row.get("board") or {}
        rows.append(
            {
                "tick": row["tick"],
                "merged_prs": d.get("merged_prs", 0),
                "releases": d.get("releases", 0),
                "opened_prs": d.get("opened_prs", 0),
                "file_edits": d.get("file_edits", 0),
                "seeded_done": b.get("seeded_declared_completed", 0),
                "seeded_total": b.get("seeded_total", 0),
                "protocol_uses": p.get("uses", 0),
                "protocol_enforcements": p.get("enforcements", 0),
                "protocol_adopted": p.get("adopted", 0),
                "protocol_proposed": p.get("proposed", 0),
            }
        )
    return rows


def load_final_eval(path: Path) -> dict:
    ev_dir = path / "evaluations"
    if not ev_dir.is_dir():
        return {}
    for fn in sorted(ev_dir.glob("final_evaluation_*.json")):
        data = json.loads(fn.read_text(encoding="utf-8"))
        ev = data.get("evaluation") or data
        passed, failed = [], []
        for rec in ev.get("evidence_records") or []:
            if rec.get("kind") != "executable_behavioral_contract":
                continue
            tid = rec.get("task_id") or rec.get("command") or ""
            short = tid.split(":")[-1] if ":" in tid else tid
            if rec.get("candidate_status") == "passed":
                passed.append(short)
            else:
                failed.append(short)
        return {
            "candidate_pass_rate": ev.get("candidate_pass_rate"),
            "passed_contracts": passed,
            "failed_count": len(failed),
        }
    return {}


def arm_summary(path: Path) -> dict:
    rec_path = path / "experiment_run_record.json"
    if not rec_path.is_file():
        return {"path": str(path.name), "missing": True}
    rec = json.loads(rec_path.read_text(encoding="utf-8"))
    m = rec.get("metrics") or {}
    fe = rec.get("final_evaluation") or {}
    rate = fe.get("candidate_pass_rate")
    if rate is None:
        rate = m.get("oss_hidden_pass_rate")
    return {
        "arm": rec.get("arm_id"),
        "pass_rate": rate,
        "causal_fixes": m.get("causal_fix_count") or fe.get("causal_fix_count"),
        "merged_prs": m.get("merged_pr_count"),
        "protocol_uses": m.get("protocol_use_count", 0),
    }


def hud_at_tick(checkpoints: list[dict], tick: int) -> dict:
    hud = checkpoints[0] if checkpoints else {}
    for row in checkpoints:
        if row["tick"] <= tick:
            hud = row
        else:
            break
    return hud


def first_divergence(a: list[dict], b: list[dict]) -> int | None:
    """First tick where the two arms stop taking identical actions."""
    for ra, rb in zip(a, b):
        sig_a = [(x["agent"], x["verb"]) for x in ra["actions"]]
        sig_b = [(x["agent"], x["verb"]) for x in rb["actions"]]
        if sig_a != sig_b:
            return ra["tick"]
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--runs",
        type=Path,
        default=DEFAULT_RUNS,
        help=f"directory holding the four arm run directories (default: {DEFAULT_RUNS})",
    )
    ap.add_argument("--out", type=Path, default=OUT, help=f"output JSON (default: {OUT})")
    args = ap.parse_args()

    arms = {arm: args.runs / name for arm, name in RUN_NAMES.items()}
    b3_dir = arms["b3"]
    missing = [str(p) for p in arms.values() if not p.is_dir()]
    if missing:
        raise SystemExit("Run directories not found:\n  " + "\n  ".join(missing))

    ticks = parse_summary(b3_dir / "summary.txt")
    b2_ticks = parse_summary(arms["b2"] / "summary.txt")
    checkpoints = load_checkpoints(b3_dir)
    b2_checkpoints = load_checkpoints(arms["b2"])
    split_tick = first_divergence(b2_ticks, ticks)
    milestones = [
        {"tick": 12, "label": "First merge PR", "tag": "merge"},
        {"tick": 13, "label": "Protocols enforced", "tag": "protocol"},
        {"tick": 169, "label": "New protocol proposed", "tag": "protocol"},
        {"tick": 179, "label": "Protocol amended", "tag": "protocol"},
        {"tick": 336, "label": "Run complete", "tag": "end"},
    ]
    compare = {arm: arm_summary(p) for arm, p in arms.items()}

    payload = {
        "schema_version": "relic_visual_timeline_v1",
        "company": "LanternForge",
        "pack": "cattrs_v2510_to_v2610",
        "condition": "b3_full_sociogenesis",
        "seed": 4013,
        "max_tick": 336,
        "agents": ["paul", "victor", "calvin", "scarlett", "sean", "skitty", "iris", "will"],
        "agent_roles": {
            "paul": "Eng lead",
            "victor": "Release & protocols",
            "calvin": "CI / merge",
            "scarlett": "Product",
            "sean": "Backend",
            "skitty": "Feature dev",
            "iris": "Docs & tasks",
            "will": "QA",
        },
        "zones": {
            "desk": {"label": "Desks", "emoji": "🪑"},
            "code": {"label": "Code", "emoji": "⌨️"},
            "review": {"label": "Review & merge", "emoji": "🔀"},
            "ci": {"label": "CI lab", "emoji": "⚙️"},
            "protocol": {"label": "Protocol room", "emoji": "📜"},
            "ship": {"label": "Shipping", "emoji": "🚀"},
        },
        "compare_arms": compare,
        "final_eval_b3": load_final_eval(b3_dir),
        "final_eval_b2": load_final_eval(arms["b2"]),
        "milestones": milestones,
        "checkpoints": checkpoints,
        "ticks": ticks,
        # Same seed, same model, same task — only the organization differs.
        "compare_track": {
            "arm": "b2",
            "label": "B2 · roles only",
            "b3_label": "B3 · institutions",
            "split_tick": split_tick,
            "checkpoints": b2_checkpoints,
            "ticks": b2_ticks,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    size_mb = args.out.stat().st_size / 1e6
    print(
        f"Wrote {args.out} ({size_mb:.1f} MB): "
        f"B3 {len(ticks)} ticks, B2 {len(b2_ticks)} ticks, "
        f"diverge at t{split_tick}"
    )


if __name__ == "__main__":
    main()
