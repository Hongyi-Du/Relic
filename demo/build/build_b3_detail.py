"""Extract the *content* of the B3 cattrs run for the workbench view.

`build_timeline.py` gives per-tick actions ("who did what verb"). This script
answers the other three questions: what the team is working on, what rules they
argued into existence, and what they said to each other.

Sources (all from the run folder, none of them the 217 MB replay):
  final_snapshot.json                    tasks, proposals, messages, protocols
  organizational_capability_evidence.json  every protocol event, with ticks
  evaluations/final_evaluation_*.json     which hidden contracts ended up passing
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUN_NAME = "20260913-161353_org_mx4013_cattrs_b3_seed4013_llm"
DEFAULT_RUN = REPO / "scratch/log" / RUN_NAME
OUT = Path(__file__).resolve().parents[1] / "data/b3_detail_4013.json"

MAX_TICK = 336

# Events that are worth a card in the governance feed; `use` is far too frequent
# (935 on one protocol alone) and is only ever shown as a counter.
NOTABLE_EVENTS = {"proposal", "support", "adoption", "amendment", "impact"}


def clip(text: str | None, n: int) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def load(run: Path, name: str) -> dict:
    return json.loads((run / name).read_text(encoding="utf-8"))


def build_tasks(snap: dict) -> list[dict]:
    """The cattrs issues the team is actually paid to close."""
    out = []
    for t in snap["internal"]["tasks"]:
        history = [
            {
                "tick": h["tick"],
                "agent": h.get("agent_id") or "",
                "to": h.get("to") or "",
                "via": h.get("via") or "",
            }
            for h in (t.get("history") or [])
            if h.get("tick") is not None
        ]
        history.sort(key=lambda h: h["tick"])
        title = t["title"]
        for prefix in ("Implement: ", "Fix: "):
            if title.startswith(prefix):
                title = title[len(prefix) :]
                break
        out.append(
            {
                "id": t["task_id"],
                "title": clip(title, 120),
                "detail": clip(t.get("description"), 400),
                "oss": t["task_id"].startswith("task_oss_issue_"),
                "owner": t.get("owner_id") or "",
                "final_status": t.get("status") or "open",
                "history": history,
            }
        )
    # Work the run was graded on first, then everything else, oldest first.
    out.sort(key=lambda x: (not x["oss"], x["history"][0]["tick"] if x["history"] else 999))
    return out


def build_protocols(snap: dict, evidence: dict) -> list[dict]:
    """Each rule, the problem it was written against, and its event timeline."""
    by_id = {p["protocol_id"]: p for p in snap["internal"]["protocols"]}
    # Specs carry the human-readable name but id themselves `protospec_1`
    # against the protocol's `proto_spec_1`.
    specs = {
        s["protocol_id"].replace("protospec_", "proto_spec_"): s
        for s in snap["protocol_specs"]["items"]
    }
    # Proposals carry a decent title for the protocols that have no spec.
    prop_titles = {
        p.get("object_created_id"): p.get("title")
        for p in snap["proposals"]["items"]
        if p.get("object_created_id")
    }

    out = []
    for ev in evidence["protocols"]:
        pid = ev["protocol_id"]
        base = by_id.get(pid)
        if not base:
            continue  # doc_* carriers, not real protocols
        per_tick: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        notable = []
        for e in ev["events"]:
            tick = e.get("tick")
            if tick is None:
                continue
            per_tick[tick][e["event_type"]] += 1
            if e["event_type"] in NOTABLE_EVENTS:
                notable.append(
                    {"tick": tick, "kind": e["event_type"], "actor": e.get("actor_id") or ""}
                )
            elif e["event_type"] == "enforcement" and e.get("data", {}).get("blocked"):
                notable.append(
                    {
                        "tick": tick,
                        "kind": "blocked",
                        "actor": e.get("actor_id") or "",
                        "context": e.get("data", {}).get("context_id") or "",
                    }
                )
        notable.sort(key=lambda x: x["tick"])
        counts = defaultdict(int)
        for row in per_tick.values():
            for k, v in row.items():
                counts[k] += v

        out.append(
            {
                "id": pid,
                "name": clip(
                    specs.get(pid, {}).get("name")
                    or prop_titles.get(pid)
                    or base["protocol_type"].replace("_", " "),
                    90,
                ),
                "rule": clip(base.get("rule_summary"), 420),
                "problem": clip(
                    base.get("target_process")
                    or ev.get("target_process")
                    or specs.get(pid, {}).get("trigger_condition"),
                    360,
                ),
                "proposer": base.get("proposer_id") or "",
                "supporters": [s for s in (base.get("supporters") or []) if s != "system"],
                "status": base.get("adoption_status") or "",
                "emergence": ev.get("emergence_level") or "",
                "first_tick": base.get("first_tick"),
                "totals": dict(counts),
                "per_tick": {str(k): dict(v) for k, v in sorted(per_tick.items())},
                "notable": notable[:400],
            }
        )
    out.sort(key=lambda p: p["first_tick"] or 999)
    return out


def build_proposals(snap: dict) -> list[dict]:
    """77 proposals, 37 of them vetoed — this is where the conflict lives."""
    out = []
    for p in snap["proposals"]["items"]:
        tick = p.get("created_at_tick")
        if tick is None:
            continue
        out.append(
            {
                "id": p["proposal_id"],
                "tick": tick,
                "proposer": p.get("proposer_agent_id") or "",
                "title": clip(p.get("title"), 110),
                "problem": clip(p.get("target_problem"), 300),
                "solution": clip(p.get("proposed_solution"), 300),
                "status": p.get("status") or "",
                "vetoed_by": p.get("rejected_by") or [],
                "backed_by": p.get("supporters") or [],
                "kind": p.get("proposal_type") or "",
            }
        )
    out.sort(key=lambda x: x["tick"])
    return out


def build_messages(snap: dict) -> list[dict]:
    out = []
    for m in snap["internal"]["messages"]:
        text = m.get("full_text") or m.get("text_summary") or ""
        if not text.strip():
            continue
        out.append(
            {
                "tick": m["created_tick"],
                "from": m["sender_id"],
                "channel": m.get("channel_id") or "dm",
                "text": clip(text, 460),
            }
        )
    out.sort(key=lambda x: x["tick"])
    return out


def build_cast(snap: dict) -> list[dict]:
    """Who these eight people are, in their own configuration's words."""
    out = []
    for aid, a in snap["agents"].items():
        identity = a.get("initial_identity") or ""
        # "Founder / Visionary Catalyst — strong vision...; volatile, scope-prone."
        title, _, traits = identity.partition("—")
        skills = sorted((a.get("skills") or {}).items(), key=lambda kv: -kv[1])[:3]
        reflections = ((a.get("memory") or {}).get("reflections")) or []
        out.append(
            {
                "id": aid,
                "name": a.get("name") or aid,
                "codename": a.get("codename") or "",
                "role": (a.get("role") or "").replace("_", " "),
                "title": clip(title.strip(" /"), 70),
                "traits": clip(traits.strip(), 170),
                "tone": (a.get("communication_style") or {}).get("tone", "").replace("_", " "),
                "skills": [{"name": k.replace("_", " "), "score": round(v, 2)} for k, v in skills],
                "flaws": [f.replace("_", " ") for f in (a.get("failure_modes") or [])[:3]],
                "dominance": round((a.get("profile") or {}).get("dominance", 0), 2),
                "conformity": round((a.get("profile") or {}).get("conformity", 0), 2),
                "reflection": clip(reflections[0] if reflections else "", 420),
            }
        )
    return out


def build_first_cycle(snap: dict, protocols: list[dict], proposals: list[dict]) -> dict:
    """
    The first complete institutional lifecycle, beat by beat.

    Everything here is looked up rather than hardcoded, so a different run
    produces a different (but still complete) opening arc.
    """
    # The rule that ended up doing the most work is the one worth narrating.
    main = max(protocols, key=lambda p: p["totals"].get("enforcement", 0))
    firsts: dict[str, dict] = {}
    for e in main["notable"]:
        firsts.setdefault(e["kind"], e)
    # `use` and `blocked` are not in `notable` for every kind; recover from per_tick.
    first_use = None
    for tick_str in sorted(main["per_tick"], key=int):
        if main["per_tick"][tick_str].get("use"):
            first_use = int(tick_str)
            break
    first_block = next((e for e in main["notable"] if e["kind"] == "blocked"), None)

    # The attempt that failed before this one landed.
    earlier_veto = next(
        (p for p in proposals if p["status"] == "rejected" and p["tick"] < (main["first_tick"] or 0)),
        None,
    )
    return {
        "protocol_id": main["id"],
        "veto": earlier_veto,
        "proposal_tick": firsts.get("proposal", {}).get("tick"),
        "proposer": firsts.get("proposal", {}).get("actor"),
        "support": firsts.get("support"),
        "adoption": firsts.get("adoption"),
        "first_use_tick": first_use,
        "first_block": first_block,
        "amendment": firsts.get("amendment"),
        "impact": firsts.get("impact"),
    }


def build_eval(run: Path) -> dict:
    ev_dir = run / "evaluations"
    for fn in sorted(ev_dir.glob("final_evaluation_*.json")):
        data = json.loads(fn.read_text(encoding="utf-8"))
        ev = data.get("evaluation") or data
        passed, failed = [], []
        for rec in ev.get("evidence_records") or []:
            if rec.get("kind") != "executable_behavioral_contract":
                continue
            name = (rec.get("task_id") or "").split(":")[-1]
            (passed if rec.get("candidate_status") == "passed" else failed).append(name)
        return {
            "pass_rate": ev.get("candidate_pass_rate"),
            "passed": passed,
            "failed": failed,
        }
    return {}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--run",
        type=Path,
        default=DEFAULT_RUN,
        help=f"B3 run directory (default: {DEFAULT_RUN})",
    )
    ap.add_argument("--out", type=Path, default=OUT, help=f"output JSON (default: {OUT})")
    args = ap.parse_args()
    if not args.run.is_dir():
        raise SystemExit(f"Run directory not found: {args.run}")

    snap = load(args.run, "final_snapshot.json")
    evidence = load(args.run, "organizational_capability_evidence.json")
    product = snap["product"]
    protocols = build_protocols(snap, evidence)
    proposals = build_proposals(snap)

    payload = {
        "schema_version": "relic_b3_detail_v1",
        "max_tick": MAX_TICK,
        "product": {
            "name": product["name"],
            "summary": clip(product.get("summary"), 300),
            "dataset": product["oss_eval"]["dataset_id"],
            # cattrs_v2510_to_v2610 -> "cattrs v25.1.0 → v26.1.0"
            "label": "cattrs v25.1.0 → v26.1.0",
            "hidden_tests": product["oss_eval"]["hidden_tests_count"],
            "gaps": [clip(g, 140) for g in product.get("known_gaps") or []],
        },
        "agents": ["paul", "victor", "calvin", "scarlett", "sean", "skitty", "iris", "will"],
        "channels": [
            {"id": c["channel_id"], "messages": c["message_count"]}
            for c in snap["internal"]["channels"]
            if c["message_count"]
        ],
        "cast": build_cast(snap),
        "tasks": build_tasks(snap),
        "protocols": protocols,
        "proposals": proposals,
        "first_cycle": build_first_cycle(snap, protocols, proposals),
        "messages": build_messages(snap),
        "evaluation": build_eval(args.run),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    vetoed = sum(1 for p in payload["proposals"] if p["status"] == "rejected")
    print(
        f"Wrote {args.out} ({args.out.stat().st_size / 1e6:.2f} MB)\n"
        f"  {len(payload['tasks'])} tasks ({sum(t['oss'] for t in payload['tasks'])} graded OSS issues)\n"
        f"  {len(payload['protocols'])} protocols, "
        f"{sum(p['totals'].get('enforcement', 0) for p in payload['protocols'])} enforcements\n"
        f"  {len(payload['proposals'])} proposals ({vetoed} vetoed)\n"
        f"  {len(payload['messages'])} messages across {len(payload['channels'])} channels"
    )


if __name__ == "__main__":
    main()
