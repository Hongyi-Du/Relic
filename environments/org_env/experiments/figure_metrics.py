"""Everything the nine panels need, measured where the run can measure it.

The figure used to be assembled after the fact: a tool reopened every
checkpoint, rebuilt the mainline at each one and ran the hidden suite against
it. That works, and it is where every published number has come from, but it
puts the expensive half of the measurement outside the run -- so a run can
finish, look healthy, and only later turn out to carry no score at all.
mini_blobstore did exactly that thirteen times.

So the scoring lives here now, and a run calls it as each checkpoint lands. The
tool imports matplotlib at module level and a run must not, which is why these
functions moved out of it rather than being imported from it.

Two rules the records keep:

A field that does not apply to a pack is null, and `na` says why. A pack that
holds nothing back has no held-out count; a desk that is not one tree has no
workspace score. Neither is a zero, and writing zero would put a measurement
where there is none.

Nothing here is visible to the organization. The hidden suite is evaluator-only:
these records are written beside the run, never into the world the agents read.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "sociogenesis_figure_metrics_v1"
METRICS_FILENAME = "figure_metrics.jsonl"

# How long one contract may take when scoring a checkpoint.
#
# Qualification runs the starter and the reference, which are known-good trees,
# and can afford to wait. Scoring a checkpoint runs agent code, and agent code
# can fail to terminate: a tenacity run wrote its t96 checkpoint and then sat
# for twenty minutes inside one retry contract, because a patch had made a
# retry loop that never gave up. At the inherited 900 seconds a pack with
# sixteen contracts, scored against both the mainline and the desk, can spend
# eight hours on a single checkpoint and the run makes no progress at all.
#
# A normal contract here takes about five seconds. Two minutes leaves ample
# room for one that is merely slow, and treats one that hangs as what it is --
# a result about the code, recorded as a failure, not something to wait out.
SCORING_TIMEOUT = int(os.environ.get("ORG_CHECKPOINT_SCORING_TIMEOUT", "120"))


def load_pack_manifest(pack: str) -> dict:
    """Denominators and identities the figure must not assume.

    A contract is *exposed* when it names an issue the agents were shown, and
    *held out* when it names none. Some packs hold nothing back; that is a fact
    about the pack and the held-out series is then NA rather than zero.
    """
    from society_core.time_machine_evaluation import build_time_machine_evaluation_plan

    # Qualification runs the same potentially non-terminating contracts as
    # checkpoint scoring.  A hard-coded 900 seconds here bypassed the advertised
    # ORG_CHECKPOINT_SCORING_TIMEOUT and made a figure hang before its first
    # checkpoint was even read.
    plan = build_time_machine_evaluation_plan(
        dataset_id=pack,
        timeout_seconds=SCORING_TIMEOUT,
    )
    spec = plan._spec
    root = Path(spec.starter_repo_dir).parent
    public = sorted(p.stem for p in (root / "issues" / "public").glob("*.json")) \
        if (root / "issues" / "public").is_dir() else []
    specs_path = root / "tests" / "hidden" / "specs.json"
    contracts = json.loads(specs_path.read_text(encoding="utf-8")) \
        if specs_path.exists() else []
    exposed, heldout = [], []
    for c in contracts:
        ids = list(c.get("issue_ids") or [])
        (exposed if any(i in public for i in ids) else heldout).append(c["test_id"])

    # A pack may score itself in behavioural cases rather than in files: the
    # blobstore's status line reads "1-content-addressing:0of7->7". At file
    # granularity such a step is all-or-nothing, and four arms that had reached
    # 0, 3, 0 and 6 cases all showed up as a flat zero, so where the pack
    # declares cases they are the denominator.
    #
    # Most packs declare none. anyio's contracts are whole-file: one test module
    # passes or it does not. Reading a missing declaration as zero emptied the
    # three panels that carry the result — B3 had five contracts and the figure
    # showed nothing — so the contract itself is the unit when no finer one is
    # published, and the axis says which is in force.
    declared_cases = sum(int(c.get("starter_total") or 0) for c in contracts)
    scores_in_cases = declared_cases > 0
    cases_total = declared_cases if scores_in_cases else len(contracts)
    module_to_contract = {Path(str(c.get("rel_path") or "")).stem: c["test_id"]
                          for c in contracts}

    # issue -> the files that issue is about, through the component the issue
    # names. This is what makes "did this patch address an issue" answerable
    # from provenance rather than from a patch's own description.
    manifest_path = root / "manifest.yaml"
    components: dict[str, list[str]] = {}
    smoke: list[str] = []
    if manifest_path.exists():
        import yaml

        raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
        components = {str(k): [str(v) for v in (vs or [])]
                      for k, vs in (raw.get("component_map") or {}).items()}
        smoke = list(((raw.get("entrypoints") or {}).get("smoke")
                      or {}).get("command") or [])
    # The component was read off the contract's `introduced_in`, which holds a
    # component name on some packs and a release tag on others: all sixteen of
    # urllib3's contracts say "2.7.0", so every lookup missed, no patch could be
    # linked to an issue, and the delivery ratio was NA for all four arms of a
    # round in which eleven contracts were fixed. Each issue states its own
    # component, so that is asked first and the contract is the fallback.
    issue_files: dict[str, list[str]] = {}
    for issue in public + [p.stem for p in (root / "issues" / "heldout").glob("*.json")
                           if (root / "issues" / "heldout").is_dir()]:
        for sub in ("public", "heldout"):
            f = root / "issues" / sub / f"{issue}.json"
            if not f.is_file():
                continue
            named = str((json.loads(f.read_text(encoding="utf-8"))
                         or {}).get("component") or "")
            if named in components:
                issue_files.setdefault(issue, []).extend(components[named])
    for c in contracts:
        for issue in (c.get("issue_ids") or []):
            if issue in issue_files:
                continue
            issue_files.setdefault(issue, []).extend(
                components.get(str(c.get("introduced_in") or ""), []))

    return {
        "pack": pack, "plan": plan, "starter": Path(spec.starter_repo_dir),
        "hidden_dir": root / "tests" / "hidden",
        "seeded_issues": public, "seeded_total": len(public),
        "hidden_total": len(contracts),
        "exposed": exposed, "heldout": heldout,
        "has_heldout": bool(heldout),
        "cases_total": cases_total, "scores_in_cases": scores_in_cases,
        "unit": "behavioural cases" if scores_in_cases else "frozen contracts",
        "module_to_contract": module_to_contract,
        "components": components, "issue_files": issue_files,
        "smoke": smoke,
    }
def exposed_denominator(manifest: dict) -> int:
    """How many the exposed count is out of.

    Kept in one place because it was written twice and once wrongly: "all
    exposed fixed" compared a count that tops out at the twelve exposed
    contracts against all sixteen, so it could never be true and the column was
    empty for every arm of every round.
    """
    return (manifest["cases_total"] if manifest["scores_in_cases"]
            else len(manifest["exposed"]))
# ---------------------------------------------------- mainline reconstruction --
def _merge_timeline(world: Any) -> dict[str, list[tuple[int, str]]]:
    patches = getattr(world, "patches", {}) or {}
    per: dict[str, list[tuple[int, str]]] = {}
    for pr in world.repo_system.repo.pull_requests.values():
        tick = getattr(pr, "merged_tick", None)
        if tick is None:
            continue
        for patch_id, art_id in world.repo_system.merged_commit_patches(pr):
            patch = patches.get(patch_id)
            per.setdefault(art_id, []).append(
                (int(tick), getattr(patch, "new_content", "") if patch else ""))
    for rows in per.values():
        rows.sort(key=lambda r: r[0])
    return per
def mainline_patch_ids(world: Any) -> set[str]:
    """Patch ids reachable from the merge history, which is what "landed" means."""
    out: set[str] = set()
    for pr in world.repo_system.repo.pull_requests.values():
        if getattr(pr, "merged_tick", None) is None:
            continue
        for patch_id, _art in world.repo_system.merged_commit_patches(pr):
            out.add(str(patch_id))
    return out
def reconstruct_mainline_at_tick(world: Any, starter: Path, dest: Path,
                                 tick: int) -> dict:
    """The mainline tree as it stood at `tick`, written to `dest`.

    Mirrors the artifact set the live exporter writes rather than copying the
    frozen directory, whose build leftovers the grader never sees.
    """
    timeline = _merge_timeline(world)
    _seed_from_starter(starter, dest)
    written, unresolved, deferred = 0, 0, 0
    for art_id, art in (getattr(world, "product_artifacts", {}) or {}).items():
        if getattr(art, "artifact_type", "") == "issue":
            continue
        path = getattr(art, "linked_file_path", "")
        if not path:
            continue
        merges = [(t, text) for t, text in timeline.get(art_id, []) if t <= tick]
        seeded = starter / path
        if merges:
            text = merges[-1][1]
            if not text:
                unresolved += 1
                text = (seeded.read_text(encoding="utf-8", errors="replace")
                        if seeded.is_file()
                        else (getattr(art, "mainline_content", "")
                              or getattr(art, "content", "") or ""))
        elif seeded.is_file():
            text = seeded.read_text(encoding="utf-8", errors="replace")
        elif timeline.get(art_id):
            deferred += 1
            continue
        else:
            text = (getattr(art, "mainline_content", "")
                    or getattr(art, "content", "") or "")
        target = dest / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        written += 1
    return {"written": written, "unresolved": unresolved, "not_yet_created": deferred}
def _seed_from_starter(starter: Path, dest: Path) -> None:
    """The pack's own files under `dest`, minus what a grader never sees.

    A tree assembled from artifacts alone is only whole where every source file
    is an artifact. On mini_blobstore that held; on traffic_watch it did not,
    and the suite scored zero for every arm on a desk that was missing files
    nobody had edited. The starter is the floor; the artifacts are laid over it.
    """
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(starter, dest, ignore=shutil.ignore_patterns(
        "tests", "__pycache__", ".pytest_cache", "*.pyc", ".git"))
def write_workspace_at_tick(world: Any, starter: Path, dest: Path) -> dict:
    """The working tree as the members left it, written to `dest`.

    The mainline is what the organization delivered; this is what its members
    wrote. Scoring only the first reports an arm that wrote nothing and an arm
    that wrote working code it could never land as the same zero, and the
    distance between the two is what an organization is for. Taken at the
    checkpoint's own tick, since the working text carries no history to walk.
    """
    _seed_from_starter(starter, dest)
    written = 0
    for art in (getattr(world, "product_artifacts", {}) or {}).values():
        if getattr(art, "artifact_type", "") == "issue":
            continue
        path = getattr(art, "linked_file_path", "")
        text = getattr(art, "content", "") or getattr(art, "mainline_content", "") or ""
        if not path or not text:
            continue
        target = dest / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        written += 1
    return {"written": written}
def workspace_stands_up(manifest: dict, tree: Path) -> tuple[bool, str]:
    """Whether the desk is one tree, judged by the pack's own smoke gate.

    Since work moved onto a branch per task, an artifact's live text is what its
    branch says, and laying every artifact over one directory mixes branches
    that were never meant to meet. On urllib3 one branch's `exceptions.py` lost
    a name the rest of the package imports, so all thirty-five modules failed to
    import and the suite returned zero -- reported beside a mainline passing
    eleven, and indistinguishable from members who wrote nothing. B1 read 0 from
    t216, B2 from t48, B3 from t72, each after reading above its own mainline
    earlier, which is the shape of a tree falling apart rather than of work
    stopping. A tree that cannot stand is not a score; it is the absence of one.
    """
    command = [str(p) for p in (manifest.get("smoke") or [])]
    if not command and (manifest["starter"] / "smoke_check.py").is_file():
        command = [sys.executable, "smoke_check.py"]
    if not command:
        return True, ""
    from environments.org_env.product.substrates.eval_assets import run_bounded

    if command[0] == "python":
        command = [sys.executable, *command[1:]]
    tests = manifest["starter"] / "tests"
    if tests.is_dir():
        shutil.copytree(tests, tree / "tests", dirs_exist_ok=True)
    try:
        code, out, err = run_bounded(command, cwd=str(tree), timeout=900)
    except Exception as exc:
        return False, f"the pack's smoke gate could not be run: {exc}"
    if code == 0:
        return True, ""
    # The reason has to be a line that failed. Taking the first non-empty line
    # of a gate that reports step by step gave "the desk stopped being one tree
    # (step 1 passed)" on three arms out of four -- a success quoted as the
    # cause of a failure, which tells a reader nothing and looks like a bug in
    # the gate rather than in this function.
    lines = [ln.strip() for ln in (out + "\n" + err).splitlines() if ln.strip()]
    line = next((ln for ln in lines
                 if "fail" in ln.lower() or "error" in ln.lower()
                 or "Traceback" in ln), lines[0] if lines else "")
    return False, line[:160]
def _score_cases(manifest: dict, tree: Path) -> dict:
    """Behavioural cases passing, which is the grain the pack scores itself in.

    A contract file holds several cases and reports pass only when every one of
    them holds, so file granularity cannot show an arm that got most of the way.
    """
    import xml.etree.ElementTree as ET

    from environments.org_env.product.substrates.eval_assets import run_bounded

    shutil.copytree(manifest["hidden_dir"], tree / "tests" / "hidden",
                    dirs_exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        report = Path(tmp) / "cases.xml"
        run_bounded(
            [sys.executable, "-m", "pytest", "tests/hidden", "-q", "--tb=no",
             "-p", "no:cacheprovider", "--confcutdir=tests/hidden",
             f"--junit-xml={report}"],
            cwd=str(tree), timeout=1800)
        if not report.exists():
            return {"cases_passed": None, "cases_by_contract": {}}
        root = ET.parse(report).getroot()

    by_contract: dict[str, int] = {}
    passed = 0
    for case in root.iter("testcase"):
        # Every test in the suite counts. This once kept to names beginning
        # `test_case`, to exclude a source-shape check mini_blobstore carried;
        # that check was removed from the pack outright, and the filter went on
        # to blind the counter on every pack whose hidden tests are named
        # anything else -- traffic_watch scored zero cases where its arms were
        # passing twenty-two.
        if not str(case.get("name", "")):
            continue
        if any(child.tag in ("failure", "error", "skipped") for child in case):
            continue
        module = str(case.get("classname", "")).split(".")[-1]
        contract = manifest["module_to_contract"].get(module, module)
        by_contract[contract] = by_contract.get(contract, 0) + 1
        passed += 1
    return {"cases_passed": passed, "cases_by_contract": by_contract}
def run_frozen_evaluator(manifest: dict, world: Any, tick: int) -> dict:
    """Score the frozen suite against the mainline as it stood at `tick`."""
    from society_core.time_machine_evaluation import evaluate_time_machine_candidate

    with tempfile.TemporaryDirectory() as tmp:
        candidate = Path(tmp) / "candidate"
        reconstruct_mainline_at_tick(world, manifest["starter"], candidate, tick)
        result = evaluate_time_machine_candidate(manifest["plan"], candidate,
                                                 timeout_seconds=SCORING_TIMEOUT)
        # Only worth a second suite run when the pack publishes case counts to
        # compare against; otherwise the contract verdict above is the unit.
        cases = (_score_cases(manifest, candidate) if manifest["scores_in_cases"]
                 else {"cases_passed": None, "cases_by_contract": {}})
        # The same suite against the desk, so the figure can show what was
        # written beside what was delivered.
        # A desk carries work in progress; that is what a desk is. Judging it by
        # the pack's merge gate and withholding the score whenever the gate is
        # red withheld it always: on mini_blobstore every arm read "not one
        # tree" from t24, and the whole workspace series -- the only measure of
        # what was written but not delivered, which is precisely what separates
        # these arms -- was empty for all four.
        #
        # So the suite runs against the desk either way and the score is
        # reported. Whether the tree was coherent rides along as a note, for the
        # case the check was written for: a desk assembled from branches that
        # were never meant to meet can read zero for a reason that has nothing
        # to do with how much was written.
        desk = Path(tmp) / "workspace"
        write_workspace_at_tick(world, manifest["starter"], desk)
        coherent, desk_note = workspace_stands_up(manifest, desk)
        try:
            desk_result = evaluate_time_machine_candidate(manifest["plan"], desk,
                                                          timeout_seconds=SCORING_TIMEOUT)
            desk_passed = {o.test_id for o in desk_result.outcomes
                           if o.candidate_status == "passed"}
            desk_cases = (_score_cases(manifest, desk) if manifest["scores_in_cases"]
                          else {"cases_passed": None, "cases_by_contract": {}})
        except Exception as error:
            desk_passed = None
            desk_cases = {"cases_passed": None, "cases_by_contract": {}}
            desk_note = f"the desk could not be scored at all: {error}"
    outcomes = result.outcomes
    passed = {o.test_id for o in outcomes if o.candidate_status == "passed"}
    by_contract = cases["cases_by_contract"]
    if not manifest["scores_in_cases"]:
        # No finer unit was published, so a passing contract is one unit. Taking
        # the case counter's zero instead would report that nothing was fixed on
        # every pack that scores whole files.
        by_contract = {test_id: 1 for test_id in passed}
    exposed_cases = sum(n for c, n in by_contract.items()
                        if c in set(manifest["exposed"]))
    heldout_cases = (sum(n for c, n in by_contract.items()
                         if c in set(manifest["heldout"]))
                     if manifest["has_heldout"] else None)
    return {
        "tick": tick,
        "full_hidden_passes": len(passed),
        "exposed_oracle_passes": len(passed & set(manifest["exposed"])),
        "heldout_oracle_passes": (len(passed & set(manifest["heldout"]))
                                  if manifest["has_heldout"] else None),
        "cases_passed": (cases["cases_passed"] if manifest["scores_in_cases"]
                         else len(passed)),
        "workspace_cases_passed": (
            None if desk_passed is None
            else (desk_cases["cases_passed"] if manifest["scores_in_cases"]
                  else len(desk_passed))),
        "workspace_hidden_passes": (None if desk_passed is None
                                    else len(desk_passed)),
        "workspace_coherent": bool(coherent),
        "workspace_note": desk_note,
        "exposed_cases_passed": exposed_cases,
        "heldout_cases_passed": heldout_cases,
        "cases_by_contract": by_contract,
        "causal_fixes": sum(1 for o in outcomes if o.causal_fix),
        "regressions": sum(1 for o in outcomes if o.regression),
        "passed_tests": sorted(passed),
        "fixed_issue_ids": sorted({i for o in outcomes if o.causal_fix
                                   for i in o.issue_ids}),
        "starter_passes": sum(1 for o in outcomes if o.baseline_status == "passed"),
        "reference_passes": sum(1 for o in outcomes if o.reference_status == "passed"),
    }


# ------------------------------------------------------------- the record --
def _action_rows(world: Any) -> list[dict]:
    return [r for r in (list(getattr(world, "baseline_archived_action_log", []) or [])
                        + list(getattr(world, "action_log", []) or []))
            if isinstance(r, dict)]


# A task's terminal states, as the board names them. `merged` belongs here and
# is easy to leave out: it is the state most delivered work ends in, and a set
# of {completed, done, closed} reported zero declared completions on a run that
# had merged ninety-seven pull requests.
COMPLETED = ("done", "merged", "released")


def _first_completion_ticks(world: Any) -> dict[str, int]:
    """First arrival at a completed state, so a task that flaps is counted once.

    Read from the task's own history rather than from `update_task_status`
    actions: those rows carry no parameters at all, so nothing can be recovered
    from them about which task reached which state.
    """
    out: dict[str, int] = {}
    for task_id, task in (getattr(world, "tasks", {}) or {}).items():
        for move in (getattr(task, "history", None) or []):
            if str(move.get("to")) in COMPLETED:
                out[task_id] = int(move.get("tick") or 0)
                break
    return out


def _protocol_detail(world: Any, tick: int) -> tuple[list[dict], dict[str, int]]:
    reg = getattr(world, "protocol_registry", None)
    events = [e for e in (getattr(reg, "events", []) or [])
              if int(getattr(e, "tick", 0) or 0) <= tick]
    totals = {kind: sum(1 for e in events
                        if getattr(e, "event_type", "") == kind)
              for kind in ("proposal", "adoption", "use", "enforcement",
                           "amendment")}
    rows = []
    for pid, protocol in (getattr(reg, "protocols", {}) or {}).items():
        mine = [e for e in events if str(getattr(e, "protocol_id", "")) == str(pid)]
        if not mine:
            continue

        def first(kind):
            ticks = [int(getattr(e, "tick", 0) or 0) for e in mine
                     if getattr(e, "event_type", "") == kind]
            return min(ticks) if ticks else None

        def count(kind):
            return sum(1 for e in mine if getattr(e, "event_type", "") == kind)

        rows.append({
            "protocol_id": str(pid),
            "name": str(getattr(protocol, "name", "") or getattr(protocol, "rule_text", ""))[:120],
            "proposed": first("proposal"),
            "adopted": first("adoption"),
            "first_enforcement": first("enforcement"),
            "uses": count("use"),
            "enforcements": count("enforcement"),
            "revisions": count("amendment"),
            "status": str(getattr(protocol, "status", "")),
        })
    rows.sort(key=lambda r: (r["proposed"] is None, r["proposed"] or 0))
    return rows, totals


def _delivery(world: Any, tick: int) -> dict:
    repo = world.repo_system.repo
    patches = (getattr(world, "patches", {}) or {}).values()
    code = [p for p in patches if getattr(p, "patch_type", "") == "code_patch"]

    def at(p):
        return int(getattr(p, "applied_tick", 0) or getattr(p, "tick", 0) or 0)

    prs = list(repo.pull_requests.values())
    merged = [p for p in prs if getattr(p, "merged_tick", None) is not None
              and int(p.merged_tick) <= tick]
    landed = mainline_patch_ids(world)
    rows = _action_rows(world)
    return {
        "generated_patches": sum(1 for p in code if at(p) <= tick),
        "accepted_patches": sum(1 for p in code if at(p) <= tick
                                and getattr(p, "validation_status", "") == "accepted"),
        "patches_reaching_mainline": len(landed),
        "opened_prs": sum(1 for p in prs
                          if int(getattr(p, "opened_tick", 0) or 0) <= tick),
        "merged_prs": len(merged),
        "releases": sum(1 for r in (repo.releases or {}).values()
                        if int(getattr(r, "released_at_tick", 0) or 0) <= tick),
        "file_edits": sum(1 for r in rows if r.get("action_type") == "edit_repo_file"
                          and int(r.get("tick") or 0) <= tick),
        "review_interactions": sum(
            1 for pr in prs for _c in (getattr(pr, "review_comments", None) or [])),
        "self_reviews": sum(
            1 for pr in prs for c in (getattr(pr, "review_comments", None) or [])
            if c.get("reviewer") == pr.author_id),
    }


def snapshot(world: Any, manifest: dict, tick: int, *,
             llm_usage: dict | None = None, condition: str | None = None,
             run_id: str | None = None, score: bool = True) -> dict:
    """One record: what the nine panels read, at this tick.

    `score` runs the hidden suite against the reconstructed mainline, which is
    the slow part. It is on by default because a record without it cannot
    answer the question the figure exists for -- what the organization actually
    delivered, as opposed to what it declared.
    """
    na: dict[str, str] = {}
    seeded = {t for t in (getattr(world, "tasks", {}) or {})
              if t.startswith("task_oss_issue_")}
    done = _first_completion_ticks(world)
    declared = sum(1 for t in seeded if done.get(t) is not None and done[t] <= tick)

    protocols, totals = _protocol_detail(world, tick)
    rows = _action_rows(world)
    by_type: dict[str, int] = {}
    for r in rows:
        if int(r.get("tick") or 0) > tick:
            continue
        by_type[str(r.get("action_type") or "")] = \
            by_type.get(str(r.get("action_type") or ""), 0) + 1

    oracle: dict[str, Any]
    if score:
        oracle = run_frozen_evaluator(manifest, world, tick)
        if oracle.get("workspace_cases_passed") is None:
            na["oracle.workspace_cases_passed"] = (
                oracle.get("workspace_note")
                or "the desk was not one tree at this tick")
        if not manifest["has_heldout"]:
            oracle["heldout_cases_passed"] = None
            oracle["heldout_oracle_passes"] = None
            na["oracle.heldout_cases_passed"] = "this pack holds nothing back"
    else:
        oracle = {}
        na["oracle"] = "not scored at this tick"

    usage = dict(llm_usage or {})
    if not usage:
        na["llm"] = "the run recorded no provider usage"

    return {
        "schema_version": SCHEMA_VERSION,
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tick": int(tick),
        "condition": condition,
        "run_id": run_id,
        "pack": manifest["pack"],
        "denominators": {
            "seeded_total": manifest["seeded_total"],
            "hidden_total": manifest["hidden_total"],
            "exposed_total": len(manifest["exposed"]),
            "heldout_total": len(manifest["heldout"]),
            "cases_total": manifest["cases_total"],
            "exposed_denominator": exposed_denominator(manifest),
            "scores_in_cases": manifest["scores_in_cases"],
            "unit": manifest["unit"],
        },
        "board": {
            "seeded_total": len(seeded),
            "seeded_declared_completed": declared,
        },
        "delivery": _delivery(world, tick),
        "protocols": {
            "proposed": totals.get("proposal", 0),
            "adopted": totals.get("adoption", 0),
            "uses": totals.get("use", 0),
            "enforcements": totals.get("enforcement", 0),
            "revisions": totals.get("amendment", 0),
            "active": sum(1 for p in protocols if p["adopted"] is not None),
            "detail": protocols,
        },
        "actions": {"total": sum(by_type.values()), "by_type": by_type},
        "llm": {
            "calls": usage.get("calls"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "cached_prompt_tokens": usage.get("cached_prompt_tokens"),
            "total_tokens": usage.get("total_tokens"),
        },
        "oracle": oracle,
        "na": na,
    }


def append(run_dir: Path, record: dict) -> Path:
    """One record per line, so a run that dies still leaves the ticks it reached."""
    path = Path(run_dir) / METRICS_FILENAME
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    return path


def read(run_dir: Path) -> list[dict]:
    path = Path(run_dir) / METRICS_FILENAME
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return sorted(out, key=lambda r: int(r.get("tick") or 0))
