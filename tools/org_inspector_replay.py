"""Generate an OrgEnv inspector replay (frontend spec §8 / §53 log chain).

Runs an OrgInspectorSession for N ticks and saves the captured deep frames to
docs/replays/<name>.json — the replay is the FULL documented log chain (every
tick is a complete org_lived_full_snapshot: action_decisions / messages /
episodes / reflections / wishes / proposals / governance / event-graph / logs).

Modes:
  * mock (default): deterministic, no LLM — for CI / quick content.
  * real LLM: set ORG_LLM=1 to wire config/llm(.local).yaml + semi_auto approvals.
  * real LLM actions: also set ORG_LLM_ACTIONS=1 so the LLM drives action decisions
    (not just cognition). Many more calls; use for full experiments.

Optionally fast-forward to a start tick (skip the empty night so a short run
lands in work hours) and emit a flat final snapshot for quick reading.

Run:  PYTHONPATH="." python tools/org_inspector_replay.py [name] [ticks] [seed] [start_tick]
      PYTHONPATH="." ORG_LLM=1 python tools/org_inspector_replay.py org_llm 16 42 8
      PYTHONPATH="." ORG_LLM=1 ORG_LLM_ACTIONS=1 python tools/org_inspector_replay.py org_real_72t 72
"""
import datetime
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from environments.org_env.runtime_adapter.live import OrgInspectorSession
from environments.org_env.product.substrates.final_evaluation import (
    run_final_evaluation,
    write_experiment_run_record,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _truthy(v: str) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def _acceptance(world) -> dict:
    """Compact acceptance metrics (no repeated README/onboarding churn, real task
    evidence, no orphan product changes, explainable policy masks)."""
    from environments.org_env.product.objects import artifact_purpose
    arts = world.product_artifacts

    def active(p):
        return sum(1 for a in arts.values()
                   if artifact_purpose(a.linked_file_path or a.artifact_id) == p
                   and a.status not in ("deprecated", "closed"))
    readme = next((a for a in arts.values() if (a.linked_file_path or "") == "README.md"), None)
    done = [t for t in world.tasks.values()
            if getattr(t.status, "value", str(t.status)) in ("done", "merged", "released")]
    done_ev = [t for t in done if any(e.get("evidence_type") != "weak_signal"
                                      for e in t.progress_evidence)]
    return {
        "readme_revision": getattr(readme, "revision", None),
        "onboarding_active": active("onboarding"),
        "report_quality_active": active("report_quality"),
        "tasks_done": len(done), "tasks_done_with_evidence": len(done_ev),
        "orphan_artifacts": sum(1 for a in arts.values() if int(a.revision or 0) > 0
                                and not (a.linked_action_ids or a.linked_episode_ids
                                         or a.patch_history_ids)),
        "policy_trace_with_masks": sum(1 for r in getattr(world, "policy_trace", [])
                                       if r.get("masked_actions")),
        "patch_applied": sum(1 for e in world.events if e.get("type") == "product_event"
                             and e.get("subtype") == "patch_applied"),
        "patch_rejected": sum(1 for e in world.events if e.get("type") == "product_event"
                              and e.get("subtype") == "patch_rejected"),
    }


def _opt(flag: str, env: str, default: str = "") -> str:
    """Read a CLI option from `--flag=value` (anywhere in argv) or env `ORG_<ENV>`."""
    pref = f"--{flag}="
    for a in sys.argv[1:]:
        if a.startswith(pref):
            return a[len(pref):]
    return os.environ.get(f"ORG_{env}", default)


_METRICS_MANIFEST: dict = {}


def _transfer_receipt_with_exposure(world) -> dict | None:
    """The receipt, plus how often the inherited prose was actually in front of
    anyone.

    The text-only arm rests on a claim that both arms had the rules in view.
    That is checkable rather than assertable, and the numbers to check it with
    exist only at the end of the run.
    """
    receipt = world.__dict__.get("_capability_transfer_receipt")
    if not receipt:
        return None
    try:
        from environments.org_env.experiments.capability_transfer import (
            inherited_prose_exposure,
        )

        return {**receipt, "prose_exposure": inherited_prose_exposure(world)}
    except Exception:
        return receipt


def _emit_figure_metrics(session, run_dir, tick, *, llm_usage, run_id, emit) -> None:
    """Score this checkpoint now and write the row the figure will read.

    Doing it here rather than afterwards is the whole point: a run that scores
    nothing should say so while it is running, not months later when somebody
    opens the checkpoints and finds the evaluator never worked. It is also the
    slow part, so a failure is reported and the run carries on -- losing a row
    is recoverable, losing the run is not.

    Nothing written here re-enters the world. The hidden suite stays
    evaluator-only; these rows sit beside the run.
    """
    if str(os.environ.get("ORG_FIGURE_METRICS", "1")).strip().lower() in {"0", "off", "false"}:
        return
    dataset = (os.environ.get("ORG_OSS_DATASET") or "").strip()
    if not dataset:
        return
    try:
        from environments.org_env.experiments import figure_metrics as fm

        global _METRICS_MANIFEST
        if not _METRICS_MANIFEST:
            _METRICS_MANIFEST = fm.load_pack_manifest(dataset)
        # The frame carries a call count and no tokens; the client carries both,
        # and panels e and f are about tokens.
        client = getattr(session.world, "llm_client", None)
        totals = dict(getattr(client, "usage_totals", None) or {})
        usage = {**totals, "calls": int(getattr(client, "calls", 0) or 0)} \
            if client is not None else dict(llm_usage or {})
        # Scoring is the half that runs containers over a reconstructed tree,
        # and it is the half that has taken the machine down: agent code that
        # does not terminate spins there until the bound kills it, twice in one
        # morning with every arm lost each time. The cheap half is worth keeping
        # regardless -- token totals come off the live client and cannot be
        # recovered from a checkpoint afterwards, while the oracle can.
        #
        # Off unless asked for, because the default is what an unattended sweep
        # gets. A pack whose merged code does not terminate can spend an hour
        # per checkpoint here and a day per arm, and nothing about that is
        # visible until the sweep is over. Set ORG_FIGURE_METRICS_SCORE=1 for a
        # run being watched; otherwise score from the checkpoints afterwards
        # with tools/sociogenesis_organizational_ladder.py.
        score = str(os.environ.get("ORG_FIGURE_METRICS_SCORE", "0")).strip().lower() \
            not in {"0", "off", "false"}
        record = fm.snapshot(
            session.world, _METRICS_MANIFEST, tick,
            llm_usage=usage,
            condition=os.environ.get("ORG_EXPERIMENT_CONDITION"),
            run_id=run_id,
            score=score,
        )
        fm.append(run_dir, record)
        oracle = record.get("oracle") or {}
        scored = (f"{oracle.get('cases_passed')}/"
                  f"{record['denominators']['cases_total']} "
                  f"{record['denominators']['unit']}" if score else "scored later")
        emit(f"    · metrics @t{tick}: {scored}, "
             f"{record['delivery']['merged_prs']} merged, "
             f"{record['board']['seeded_declared_completed']} declared")
    except Exception as error:  # noqa: BLE001 - a lost row must not end the run
        emit(f"    · metrics @t{tick} not written: {type(error).__name__}: {error}")


def main() -> None:
    started_at = datetime.datetime.now(datetime.timezone.utc)
    # Windows consoles default to GBK/cp936 and crash on non-encodable glyphs (e.g. '↔')
    # in LLM rationales — which used to abort the post-run summary AFTER the heavy save.
    # Force UTF-8 with replacement so printing can never kill the run.
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    pos = [a for a in sys.argv[1:] if not a.startswith("--")]   # positionals (ignore --flags)
    name = pos[0] if len(pos) > 0 else "org_run"
    ticks = int(pos[1]) if len(pos) > 1 else 72
    seed = int(pos[2]) if len(pos) > 2 else 42
    start_tick = int(pos[3]) if len(pos) > 3 else 0
    use_llm = _truthy(os.environ.get("ORG_LLM", "0"))
    decide_actions = _truthy(os.environ.get("ORG_LLM_ACTIONS", "0"))
    from_checkpoint = _opt("from-checkpoint", "FROM_CHECKPOINT", "")
    checkpoint_every = int(_opt("checkpoint-every", "CHECKPOINT_EVERY", "0") or 0)
    target_tick_raw = _opt(
        "target-tick",
        "EXPERIMENT_TARGET_TICK",
        "",
    )
    target_tick = int(target_tick_raw) if target_tick_raw else None
    execution_profile = str(
        os.environ.get("ORG_EXECUTION_PROFILE", "native") or "native"
    ).strip()
    if execution_profile != "native":
        raise RuntimeError(
            "programbench_execution_profile_not_available_in_relic_release"
        )
    if not name.startswith("org_"):
        name = "org_" + name

    # Each run gets its own timestamped folder under <repo>/log/ holding ALL its
    # materials (replay / final snapshot / human summary / meta).
    run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = REPO_ROOT / "log" / f"{run_id}_{name}_seed{seed}_{'llm' if use_llm else 'mock'}"
    replay_path = run_dir / "replay.json"
    snapshot_path = run_dir / "final_snapshot.json"
    summary_path = run_dir / "summary.txt"
    meta_path = run_dir / "meta.json"
    record_path = run_dir / "experiment_run_record.json"

    summary_lines: list[str] = []

    def emit(line: str = "") -> None:
        print(line, flush=True)
        summary_lines.append(line)

    # Every captured frame is a WHOLE world snapshot, and the buffer holds them
    # all until the run ends: one snapshot is the size of final_snapshot.json
    # (6.8 MB on a finished cattrs arm), the world grows as it goes, and 336 of
    # them is gigabytes of resident memory. Six arms at the horizon exhausted a
    # 16 GB box -- resident memory climbed about 6 MB per tick, reaching 1.4 GB
    # by t211, which reads as a leak and is really the replay buffer.
    #
    # ORG_REPLAY_MAX_FRAMES bounds what is kept, so memory is flat in the run
    # length instead of quadratic in it. replay.json then covers a trailing
    # window rather than the whole run; nothing in the figure or scoring path
    # reads it -- the metrics come from the world and its checkpoints -- so the
    # cost is a shorter human transcript. Unset keeps the previous behaviour.
    replay_frames = os.environ.get("ORG_REPLAY_MAX_FRAMES", "").strip()
    session_kwargs = {}
    if replay_frames.isdigit() and int(replay_frames) > 0:
        session_kwargs["max_frames"] = int(replay_frames)

    # Resume candidates are authenticated before any output directory or LLM
    # is created. The loader reattaches the configured client only after its
    # strict v5 state/profile/remaining validator succeeds. That ordering is
    # why the frame bound is read above rather than beside the session: the
    # session must not exist until the checkpoint identity has been checked.
    session = OrgInspectorSession(
        seed=seed, load_llm=(use_llm and not from_checkpoint),
        approval_mode="semi_auto" if use_llm else "auto",
        defer_initial_readiness_check=bool(from_checkpoint),
        **session_kwargs)
    if from_checkpoint:
        info = session.load_checkpoint(
            from_checkpoint,
            reattach_llm=use_llm,
            expected_target_tick=target_tick,
            remaining_ticks=ticks,
        )
        if use_llm and decide_actions and getattr(session.world, "llm_client", None) is not None:
            session.world.llm_decides_actions = True
        if target_tick is not None and session.world.world_tick + ticks != target_tick:
            raise RuntimeError(
                "checkpoint continuation does not terminate at the frozen "
                f"target tick: checkpoint={session.world.world_tick} "
                f"remaining={ticks} target={target_tick}"
            )
    elif start_tick > 0:
        session.world.time.clock.current_tick = start_tick
        session.world.world_tick = start_tick
        session.buffer.frames.clear()
        session._capture()
    if (
        not from_checkpoint
        and target_tick is not None
        and session.world.world_tick + ticks != target_tick
    ):
        raise RuntimeError(
            "fresh execution does not terminate at the frozen target tick: "
            f"start={session.world.world_tick} ticks={ticks} "
            f"target={target_tick}"
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    mode = "REAL LLM (config/llm.local.yaml)" if use_llm else "mock (no LLM)"
    emit(f"running OrgEnv {ticks} ticks (seed={seed}, {mode}"
         + (f", start_tick={start_tick}" if start_tick else "") + ")")
    emit(f"run folder: {run_dir}")
    lc = getattr(session.world, "llm_client", None)
    if use_llm and decide_actions and lc is not None:
        session.world.llm_decides_actions = True
    emit(f"  LLM cognitive layer: {'ON (' + lc.provider + ')' if lc else 'off (rule/template)'}"
         f" | actions: {'LLM' if (lc and decide_actions) else 'rule'}")
    if from_checkpoint:
        emit(f"  resumed from checkpoint: {from_checkpoint} (saved at tick {info.get('checkpoint_tick')}, "
             f"now at tick {session.world.world_tick}) -> running {ticks} more ticks")

    ckpt_dir = run_dir / "checkpoints"
    if checkpoint_every:
        emit(f"  checkpointing every {checkpoint_every} ticks -> {ckpt_dir}")

    # step one tick at a time and STREAM progress (flush so it shows live even
    # when stdout is piped — no more silent multi-minute waits). Per-tick deltas
    # are computed from GLOBAL counters (preflight v3 §9), not the capped `recent`
    # list, so the count is reliable for the whole run.
    def _totals(fr, world):
        # obs7: per-tick deltas must come from UNCAPPED global counters, not the
        # snapshot's capped `logs.actions` / `internal.messages` lists (which stop
        # growing once capped and make late ticks read "+0 events" falsely).
        comm = getattr(world, "comm", None)
        return {
            "decisions": fr["action_decisions"]["total"],
            "events": len(getattr(world, "action_log", []) or []),
            "messages": len(getattr(comm, "messages", {}) or {}),
            "episodes": fr["episodes"]["total"],
            "product": sum(int(a.get("revision", 0) or 0) for a in fr["product"]["artifacts"]),
        }
    prev = _totals(session.full(), session.world)
    for _ in range(ticks):
        session.step(1)
        f = session.full()
        cur = _totals(f, session.world)
        d = {k: cur[k] - prev[k] for k in cur}
        prev = cur
        clk = f"t{f['tick']} {f['hour']:02d}:00 {f['phase']}"
        acted = [f"{x['agent_id']}:{x['candidate_action']}"
                 for x in f["action_decisions"]["recent"] if x.get("tick") == f["tick"]]
        if not acted:
            # action_decisions is the LLM-direct channel, so it is empty for
            # every tick of a B3 run — the profile policy records what it chose
            # in policy_trace instead. Keyed only on the first, the live log
            # showed no actions at all for the one condition the experiment is
            # about, and an operator watching a 336-tick run could not tell work
            # from a stall.
            acted = [
                f"{x.get('agent_id')}:{x.get('chosen_action')}"
                for x in (f.get("logs", {}).get("policy_trace") or [])
                if x.get("tick") == f["tick"] and x.get("chosen_action")
            ]
        emit(f"  [{clk}] +{d['decisions']} decisions +{d['events']} events "
             f"+{d['messages']} msg +{d['episodes']} ep +{d['product']} prod "
             f"| LLM calls={f['llm']['calls']} | {', '.join(acted[:8])}")
        # periodic checkpoint so the run is resumable from any saved tick
        if checkpoint_every and int(f["tick"]) % checkpoint_every == 0:
            ck = session.save_checkpoint(str(ckpt_dir / f"t{int(f['tick'])}.pkl"),
                                         meta={"run_id": run_id, "name": name,
                                               "experiment_run_record": record_path.name})
            emit(f"    · checkpoint @t{int(f['tick'])} -> {ck['path']} ({ck['size_mb']} MB)")
            _emit_figure_metrics(session, run_dir, int(f["tick"]),
                                 llm_usage=f.get("llm"), run_id=run_id, emit=emit)

    session.save_replay(str(replay_path), name=name)
    f = session.full()
    if target_tick is not None and int(f["tick"]) != target_tick:
        raise RuntimeError(
            "execution ended at the wrong tick: "
            f"observed={f['tick']} target={target_tick}"
        )
    snapshot_path.write_text(json.dumps(f, ensure_ascii=False, indent=1), encoding="utf-8")
    # always save a FINAL checkpoint so this run can itself be resumed later
    final_ckpt = session.save_checkpoint(
        str(run_dir / f"checkpoint_t{int(session.world.world_tick)}.pkl"),
        meta={"run_id": run_id, "name": name, "final": True,
              "experiment_run_record": record_path.name})

    # -- summary of the exported log chain (printed AND saved to summary.txt) --
    llm, ad = f["llm"], f["action_decisions"]
    emit("")
    emit("=" * 74)
    emit(f"final: tick={f['tick']} day={f['day']} | LLM {llm['provider']} "
         f"calls={llm['calls']} failures={llm['failures']} fallbacks={llm['fallbacks']}")
    _world_actions = len(getattr(session.world, "action_log", []) or [])
    _llm_drives = bool(getattr(session.world, "llm_decides_actions", False))
    emit(f"llm_action_decisions total={ad['total']} accepted={ad['accepted_count']} "
         f"rejected={ad['rejected_count']} fallback={ad['fallback_count']}"
         + ("" if _llm_drives else "  (policy drives actions -> LLM action-decisions are 0 BY DESIGN)"))
    emit(f"world_actions_total={_world_actions} | llm_cognitive_calls={llm['calls']} "
         f"| llm_drives_actions={_llm_drives}")
    emit(f"messages={len(f['internal'].get('messages', []))} "
         f"episodes={f['episodes']['total']} reflections={f['reflections']['total']} "
         f"wishes={f['wishes']['total']} proposals={f['proposals']['total']} "
         f"event_graph_edges={len(f['graphs']['event'].get('edges', []))}")
    emit("-" * 74)
    emit("WHO DID WHAT (recent LLM decisions):")
    by = defaultdict(list)
    for d in ad["recent"]:
        by[d["agent_id"]].append(d)
    for aid in f["agents"]:
        rows = by.get(aid, [])
        if not rows:
            continue
        emit(f"  [{f['agents'][aid].get('name', aid)}]")
        for d in rows[-4:]:
            tgt = d.get("target_object_id") or (d.get("params") or {}).get("task_id") or ""
            emit(f"    t{d['tick']} {d['candidate_action']}"
                 f"{('[' + tgt + ']') if tgt else ''}: {(d.get('rationale') or '')[:110]}")
    emit("-" * 74)
    # P0-6: action distribution from the UNCAPPED world.action_log (not the 200-capped
    # snapshot log) so 100t+ global metrics are correct.
    emit("action distribution: " + json.dumps(dict(Counter(
        a["action_type"] for a in getattr(session.world, "action_log", [])))))
    if f["episodes"]["items"]:
        emit("episodes: " + json.dumps([(e.get("episode_type"), e.get("status"))
                                        for e in f["episodes"]["items"]]))
    acceptance = _acceptance(session.world)
    emit("-" * 74)
    emit("acceptance: " + json.dumps(acceptance))

    # -- persist materials into the run folder -----------------------------
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    # A formation run's capability bundle is what a transfer arm inherits, so it
    # is written unconditionally: which runs become sources is decided by the
    # DAG afterwards, and a run that finished without leaving one cannot be
    # named as a source later.
    bundle_path = run_dir / "capability_bundle.json"
    try:
        from environments.org_env.experiments.capability_transfer import (
            export_capability_bundle,
        )

        bundle_path.write_text(
            json.dumps(
                export_capability_bundle(
                    session.world,
                    source_repository_id=os.environ.get(
                        "ORG_OSS_REPOSITORY_ID", ""
                    ) or os.environ.get("ORG_OSS_DATASET", ""),
                    source_seed=seed,
                ),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    except Exception as exc:  # noqa: BLE001
        # Not fatal to the run, but it must be visible: a source without a
        # bundle silently removes four transfer arms from the matrix.
        emit(f"  WARNING: capability bundle not written: {type(exc).__name__}: {exc}")
        bundle_path = None
    final_artifact = None
    evaluation_error: Exception | None = None
    try:
        final_artifact = run_final_evaluation(
            session.world,
            output_dir=run_dir / "evaluations",
            run_tag=os.environ.get("ORG_RUN_TAG", name),
        )
    except Exception as exc:
        evaluation_error = exc
    if final_artifact is None and evaluation_error is None:
        # run_final_evaluation returns None rather than raising when
        # run_oss_final_evaluation is off. Formal and host-controlled pilot
        # modes both require it. A run without it finishes with an empty
        # final_evaluation, no evaluations/ directory, and no
        # oss_hidden_pass_rate — the experiment's headline outcome — while still
        # writing status "completed". Two 336-tick B3 runs were lost to that
        # before anyone looked at the record closely enough to notice the
        # missing_fields caveat. A run that produced no outcome measurement is
        # not a completed run.
        emit("")
        emit("  WARNING: no final evaluation was produced, so this run has no "
             "oss_hidden_pass_rate.")
        emit("  run_oss_final_evaluation requires ORG_OSS_MODE=formal with a "
             "container binding, or ORG_OSS_MODE=pilot with a Bubblewrap venv.")
        evaluation_missing = "final_evaluation_not_produced:run_oss_final_evaluation_disabled"
    else:
        evaluation_missing = None
    ended_at = datetime.datetime.now(datetime.timezone.utc)
    run_record = write_experiment_run_record(
        session.world,
        record_path,
        final_evaluator=final_artifact,
        started_at=started_at,
        ended_at=ended_at,
        status=(
            "failed" if (evaluation_error is not None or evaluation_missing)
            else "completed"
        ),
        failure_reason=(
            str(evaluation_error) if evaluation_error is not None
            else evaluation_missing
        ),
        checkpoint=final_ckpt,
        provenance={
            "orchestrator": "tools.org_inspector_replay",
            "run_folder": str(run_dir),
            "resumed_from_checkpoint": from_checkpoint or None,
            "resume_source_tick": (
                info.get("checkpoint_tick") if from_checkpoint else None
            ),
            "case_plan_fingerprint": os.environ.get(
                "ORG_CASE_PLAN_FINGERPRINT"
            ),
            "target_tick": target_tick,
        },
    )
    meta = {
        "run_id": run_id, "name": name, "seed": seed, "ticks": ticks,
        "start_tick": start_tick, "mode": "llm" if use_llm else "mock",
        "status": run_record["status"],
        "started_at": run_record["started_at"],
        "ended_at": run_record["ended_at"],
        "duration_seconds": run_record["duration_seconds"],
        "provider": run_record["provider"],
        "model": run_record["model"],
        "replication_id": run_record["replication_id"],
        "randomization_block": run_record["randomization_block"],
        "randomization_order": run_record["randomization_order"],
        "experiment_run_record": record_path.name,
        "llm_drives_actions": bool(use_llm and decide_actions),
        "acceptance": acceptance,
        "llm": llm, "decisions": {k: ad.get(k) for k in
                                  ("total", "accepted_count", "rejected_count", "fallback_count")},
        "final_tick": f["tick"], "frames": len(session.buffer.frames),
        "counts": {"messages": len(f["internal"].get("messages", [])),
                   "episodes": f["episodes"]["total"], "reflections": f["reflections"]["total"],
                   "wishes": f["wishes"]["total"], "proposals": f["proposals"]["total"],
                   "event_graph_edges": len(f["graphs"]["event"].get("edges", []))},
        "files": {
            "replay": replay_path.name,
            "final_snapshot": snapshot_path.name,
            "summary": summary_path.name,
            "checkpoint": Path(final_ckpt["path"]).name,
            "experiment_run_record": record_path.name,
            "final_evaluation": (
                str(final_artifact.path.relative_to(run_dir))
                if final_artifact is not None
                else None
            ),
            "capability_bundle": (
                bundle_path.name if bundle_path is not None else None
            ),
        },
        "capability_transfer": _transfer_receipt_with_exposure(session.world),
        "resumed_from": from_checkpoint or None,
        "resume_source_tick": (
            info.get("checkpoint_tick") if from_checkpoint else None
        ),
        "case_plan_fingerprint": os.environ.get(
            "ORG_CASE_PLAN_FINGERPRINT"
        ),
        "target_tick": target_tick,
        "checkpoint_every": checkpoint_every or None,
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    emit("")
    emit(f"materials saved to: {run_dir}")
    emit(f"  replay.json ({len(session.buffer.frames)} frames) · final_snapshot.json · "
         "summary.txt · meta.json · experiment_run_record.json")
    if final_artifact is not None:
        emit(f"  final evidence: {final_artifact.path}")
    emit(f"  {run_record['schema_version']}: {record_path.name}")
    emit(f"  checkpoint: {Path(final_ckpt['path']).name} ({final_ckpt['size_mb']} MB)  "
         f"resume with:  ORG_LLM=1 ORG_LLM_ACTIONS=1 python tools/org_inspector_replay.py "
         f"{name} <more_ticks> {seed} --from-checkpoint=\"{final_ckpt['path']}\"")

    if evaluation_error is not None:
        raise RuntimeError("formal final evaluation failed") from evaluation_error

    # -- also write a portable .zip of the run folder (for easy sending/sharing);
    #    a 100t replay is ~270MB -> ~20MB zipped. The uncompressed folder is KEPT
    #    by default (the .zip sits alongside it). Env knobs:
    #      ORG_LOG_ZIP=0           -> skip the .zip entirely (raw folder only)
    #      ORG_LOG_PRUNE_REPLAY=1  -> after a verified zip, delete the big
    #                                 uncompressed replay.json to reclaim disk
    #                                 (restore any time with `unzip <archive>`)
    if _truthy(os.environ.get("ORG_LOG_ZIP", "1")):
        import shutil
        import zipfile

        archive = Path(shutil.make_archive(
            str(run_dir), "zip", root_dir=run_dir.parent, base_dir=run_dir.name))
        member = f"{run_dir.name}/{replay_path.name}"
        with zipfile.ZipFile(archive) as zf:
            archived_ok = member in zf.namelist()
        size_mb = archive.stat().st_size / 1e6
        print(f"compressed -> {archive.name} ({size_mb:.1f} MB; raw folder kept)", flush=True)
        if archived_ok and replay_path.exists() \
                and _truthy(os.environ.get("ORG_LOG_PRUNE_REPLAY", "0")):
            freed_mb = replay_path.stat().st_size / 1e6
            replay_path.unlink()
            print(f"  pruned uncompressed {replay_path.name} (-{freed_mb:.0f} MB; "
                  f"restore with: unzip {archive.name})", flush=True)


if __name__ == "__main__":
    main()
