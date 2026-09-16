"""What went wrong last time, asked of a checkpoint automatically.

Every probe here is a defect that was found by hand, on one run, after it had
already spent its ticks. Each one cost an evening of reading pickles, and each
was visible in the first checkpoint it appeared in. They are cheap -- a loaded
world and no subprocess -- so a run can be asked on every save instead of at the
post mortem.

A probe returns None when it has nothing to say. It never raises: a probe that
breaks must not stop the others, and a watcher that dies on a malformed world
tells you less than one that reports what it could.
"""
from __future__ import annotations

import collections
import re
from typing import Any, Callable, Dict, List, Optional

# A verdict line from the smoke gate: "blobstore/manifests.py: step 2 failed: ..."
_NAMES_A_FILE = re.compile(r"^([\w./-]+\.py):\s*step\b")

# How the public API document declares its surface:
#   - Class `ManifestConflictError()`
#     - `validate_name(name: str) -> None`
_PUBLISHED_CLASS = re.compile(r"^-\s*Class\s+`(\w+)\s*\(", re.M)
_PUBLISHED_METHOD = re.compile(r"^\s+-\s*`(\w+)\s*\(", re.M)


def _artifacts(world: Any) -> Dict[str, Any]:
    return getattr(world, "product_artifacts", {}) or {}


def _repo(world: Any):
    return getattr(getattr(world, "repo_system", None), "repo", None)


def _open_requests(world: Any) -> List[Any]:
    repo = _repo(world)
    return [p for p in (getattr(repo, "pull_requests", {}) or {}).values()
            if getattr(p, "merged_tick", None) is None]


def _files_a_request_carries(world: Any, pr: Any) -> set:
    """The paths this request's own commits touch."""
    repo, arts, out = _repo(world), _artifacts(world), set()
    for cid in (getattr(pr, "commit_ids", []) or []):
        commit = (getattr(repo, "commits", {}) or {}).get(cid)
        for aid in (getattr(commit, "artifact_ids", []) or []):
            art = arts.get(aid)
            path = getattr(art, "linked_file_path", "") if art else ""
            if path:
                out.add(path)
    return out


def _contract_lines(ci: Any) -> List[str]:
    return [str(r) for r in (getattr(ci, "failure_reasons", []) or [])
            if "step " in str(r) and "fail" in str(r)]


def the_code_would_fail_even_if_everything_merged(world: Any) -> Optional[Dict[str, Any]]:
    """The published surface the contract names, against what the desk defines.

    Ask this before anything about requests, branches or merges: if the code on
    the desk would fail the product's own gate with every gate removed, then
    nothing about how work reaches mainline is what stands in the way.

    Four rounds went into the merge pipeline before anyone ran the gate on the
    desk itself. It failed four of five steps, and had all along: UploadConflict
    written where the contract says UploadConflictError, ReplicaStatus.__init__
    taking *args where it names object_id and states, GCPlan never written. The
    pipeline was real and was never the binding constraint.

    Static, so it costs nothing: a name the contract publishes and no module
    defines cannot be satisfied however the merge machinery behaves.
    """
    arts = _artifacts(world)
    api = next((a for a in arts.values()
                if (getattr(a, "linked_file_path", "") or "").endswith("public_api.md")),
               None)
    if api is None:
        return None
    contract = (getattr(api, "mainline_content", "")
                or getattr(api, "content", "") or "")
    # The document declares its surface in a fixed shape, so read that rather
    # than guessing at capitalised words: a prose heading is not a symbol, and a
    # probe that says "Public is undefined" is one nobody will read twice.
    published = set(_PUBLISHED_CLASS.findall(contract))
    published |= set(_PUBLISHED_METHOD.findall(contract))
    if not published:
        return None
    written = "\n".join(
        getattr(a, "content", "") or "" for a in arts.values()
        if (getattr(a, "linked_file_path", "") or "").endswith(".py")
        and not (getattr(a, "linked_file_path", "") or "").startswith(
            ("public_contract_tests/", "tests/")))
    if len(written) < 2000:
        return None
    missing = sorted(n for n in published
                     if not re.search(rf"\b(class|def)\s+{re.escape(n)}\b", written)
                     and f"{n} =" not in written and f"{n}=" not in written)
    # A near miss is the telling shape: the contract's name absent while a prefix
    # of it is defined, which is what "UploadConflict for UploadConflictError"
    # looks like from here.
    near = [n for n in missing
            if any(re.search(rf"\b(class|def)\s+{re.escape(n[:k])}\b", written)
                   for k in range(len(n) - 1, max(len(n) - 6, 3), -1))]
    if not missing:
        return None
    return {"said": f"{len(missing)} name(s) the contract publishes are defined "
                    f"nowhere in the product"
                    + (f", {len(near)} of them written under a shorter name"
                       if near else ""),
            "evidence": [f"{n} (a shorter form of it is defined)" if n in near else n
                         for n in missing[:8]]}


def nothing_reaches_mainline(world: Any) -> Optional[Dict[str, Any]]:
    """Work piles up on the desk and the branch everything is scored on stays empty.

    Seen at t168: every module still held its 55-byte stub on mainline while the
    desk carried up to 16876 bytes, two requests had merged in 168 ticks and both
    were a README and an eval stub. The evaluation scores mainline, so the run
    was producing nothing measurable and said so nowhere.
    """
    arts, stubbed, live = _artifacts(world), [], 0
    for a in arts.values():
        path = getattr(a, "linked_file_path", "") or ""
        if not path.endswith(".py") or getattr(a, "artifact_type", "") == "issue":
            continue
        mainline = len(getattr(a, "mainline_content", "") or "")
        desk = len(getattr(a, "content", "") or "")
        if desk > 4 * max(mainline, 1) and desk > 1000:
            stubbed.append(f"{path} (mainline {mainline}B, desk {desk}B)")
        if desk > mainline:
            live += 1
    if len(stubbed) < 3:
        return None
    repo = _repo(world)
    merged = [p for p in (getattr(repo, "pull_requests", {}) or {}).values()
              if getattr(p, "merged_tick", None) is not None]
    return {"said": f"{len(stubbed)} modules carry real work on the desk that "
                    f"mainline has never seen; {len(merged)} requests have merged",
            "evidence": sorted(stubbed)[:8]}


def a_request_is_judged_on_work_it_does_not_carry(world: Any) -> Optional[Dict[str, Any]]:
    """A request refused for a module that is a stub in its own merge candidate.

    The signature of judging the shared desk instead of the candidate. pr_62
    carried blobstore/replicas.py alone and was refused for manifests.py step 2
    and objects.py step 1, modules its candidate holds as stubs, so those steps
    should not have run at all.
    """
    hits = []
    for pr in _open_requests(world):
        carried = _files_a_request_carries(world, pr)
        if not carried:
            continue
        said = str(getattr(pr, "ci_brief", "") or "")
        m = _NAMES_A_FILE.match(said.strip())
        if not m:
            continue
        blamed = m.group(1)
        if blamed in carried:
            continue
        art = next((a for a in _artifacts(world).values()
                    if (getattr(a, "linked_file_path", "") or "") == blamed), None)
        # Only a stub is conclusive: a module this request does import could
        # genuinely break through it.
        if art is not None and len(getattr(art, "mainline_content", "") or "") > 200:
            continue
        hits.append(f"{getattr(pr, 'pr_id', '?')} carries "
                    f"{sorted(carried)} and was refused for {blamed}")
    return None if not hits else {
        "said": f"{len(hits)} request(s) refused for a module their own merge "
                f"candidate holds as a stub",
        "evidence": hits[:6]}


def the_same_tree_drew_different_verdicts(world: Any,
                                          since: int = 0) -> Optional[Dict[str, Any]]:
    """Content that did not change was judged several different ways.

    Reads as the authors thrashing and is not: 16 of 21 (request, head commit)
    pairs drew more than one verdict, one of them five across twenty runs.
    """
    repo = _repo(world)
    runs = (getattr(repo, "ci_runs", {}) or {}).values()
    by_tree = collections.defaultdict(list)
    for ci in runs:
        if int(getattr(ci, "created_at_tick", 0) or 0) < since:
            continue
        by_tree[(getattr(ci, "pr_id", "?"), getattr(ci, "commit_id", "?"))].extend(
            _contract_lines(ci))
    wobbly = {k: sorted(set(v)) for k, v in by_tree.items() if len(set(v)) > 1}
    if len(wobbly) < 2:
        return None
    worst = max(wobbly.items(), key=lambda kv: len(kv[1]))
    return {"said": f"{len(wobbly)} of {len(by_tree)} unchanged trees drew more "
                    f"than one verdict; the worst drew {len(worst[1])}",
            "evidence": [f"{worst[0][0]}/{worst[0][1]}: {v[:110]}"
                         for v in worst[1][:5]]}


def one_rule_holds_all_the_credit(world: Any,
                                  since: int = 0) -> Optional[Dict[str, Any]]:
    """Enforcement concentrated on a rule that did not do the work.

    Keyword matching credited the first adopted spec whose text matched, and a
    generally worded rule matches nearly any moment: 193 of 193 enforcements
    landed on one rule while another had made 26 of the refusals.

    Counted from the events rather than the specs' own totals, because a spec's
    counter is cumulative and a resumed run would be judged forever on the
    history it inherited.
    """
    pm = getattr(world, "proposal_manager", None)
    adopted = {s.protocol_id for s in (getattr(pm, "protocol_specs", {}) or {}).values()
               if getattr(s, "status", "") == "adopted"}
    if len(adopted) < 2:
        return None
    counts = collections.Counter(
        e.get("protocol_id") for e in (getattr(world, "events", []) or [])
        if isinstance(e, dict) and e.get("type") == "protocol_enforcement_event"
        and int(e.get("tick") or 0) >= since)
    total = sum(counts.values())
    if total < 20:
        return None
    top, n = counts.most_common(1)[0]
    silent = [p for p in adopted if not counts.get(p)]
    if n < 0.9 * total or not silent:
        return None
    refusals = collections.Counter(
        e.get("protocol_id") for e in (getattr(world, "events", []) or [])
        if isinstance(e, dict) and e.get("subtype") == "patch_refused_by_protocol"
        and int(e.get("tick") or 0) >= since)
    return {"said": f"{top} holds {n} of {total} enforcements since t{since} while "
                    f"{len(silent)} adopted rule(s) hold none",
            "evidence": [f"enforcement: {dict(counts)}",
                         f"refusals actually made: {dict(refusals)}"]}


def a_rule_no_change_can_satisfy(world: Any,
                                 since: int = 0) -> Optional[Dict[str, Any]]:
    """A rule that refuses code and asks for something code is not.

    One run adopted a rule opening "Propose a protocol backed by one reusable
    review checklist" and held code patches to it. Every author rewrote to its
    limit and the change was dropped: fifteen edits lost to a demand no edit of
    that file could answer.
    """
    events = [e for e in (getattr(world, "events", []) or [])
              if isinstance(e, dict) and e.get("subtype") == "patch_refused_by_protocol"
              and int(e.get("tick") or 0) >= since]
    if not events:
        return None
    exhausted = collections.Counter(
        e.get("protocol_id") for e in events if int(e.get("attempt") or 0) >= 4)
    if not exhausted:
        return None
    worst, n = exhausted.most_common(1)[0]
    if n < 2:
        return None
    said = [str(e.get("reason", ""))[:150] for e in events
            if e.get("protocol_id") == worst][:3]
    return {"said": f"{worst} exhausted every rewrite {n} time(s); those changes "
                    f"were dropped rather than fixed",
            "evidence": said}


def approved_and_never_merged(world: Any) -> Optional[Dict[str, Any]]:
    """Requests the organization has said yes to and cannot land."""
    tick = int(getattr(world, "world_tick", 0) or 0)
    stuck = []
    for pr in _open_requests(world):
        approved = getattr(pr, "approved_tick", None)
        if approved is None or tick - int(approved) < 48:
            continue
        stuck.append(f"{getattr(pr, 'pr_id', '?')} approved at t{approved}, "
                     f"{tick - int(approved)} ticks ago, ci_passed="
                     f"{getattr(pr, 'ci_passed', None)}")
    return None if len(stuck) < 2 else {
        "said": f"{len(stuck)} request(s) approved and unmerged for over 48 ticks",
        "evidence": stuck[:6]}


def a_request_is_frozen_while_the_desk_moves_on(world: Any) -> Optional[Dict[str, Any]]:
    """A request judged on content its author has long since improved.

    pr_62 re-ran CI twelve times over 96 ticks and got the same verdict every
    time: replicas.py has no ReplicaStatus. It was right -- the candidate was the
    content of a commit made at t54. Another agent had written ReplicaStatus onto
    the desk and had nowhere to commit it, so the fix and the failure sat 114
    ticks apart inside the same organization.
    """
    from environments.org_env.product.materialize import merge_candidate_text

    arts, frozen = _artifacts(world), []
    for pr in _open_requests(world):
        try:
            candidate = merge_candidate_text(world, pr)
        except Exception:  # noqa: BLE001
            continue
        for aid, text in candidate.items():
            art = arts.get(aid)
            desk = getattr(art, "content", "") or ""
            if not desk or text == desk:
                continue
            drift = abs(len(desk) - len(text))
            if drift > 400:
                frozen.append(
                    f"{getattr(pr, 'pr_id', '?')} is judged on "
                    f"{getattr(art, 'linked_file_path', aid)} at {len(text)}B "
                    f"while the desk holds {len(desk)}B")
    return None if len(frozen) < 2 else {
        "said": f"{len(frozen)} request(s) judged on content the desk has moved "
                f"past; the fix cannot reach the failure",
        "evidence": frozen[:6]}


def one_request_swallows_the_commits(world: Any,
                                     since: int = 0) -> Optional[Dict[str, Any]]:
    """Every commit lands in one request while the others starve.

    Twice now, by two different routes. First an author had one branch and
    opening a request took it out of reach, so all later work joined the one red
    request. Then, once a fix was allowed to join somebody else's red request on
    the strength of touching something it touches, the request carrying all five
    modules matched everything: 15 of the next 17 commits, while four others sat
    untouched for 56 to 100 ticks. A request carrying every module is judged on
    every step, so it can only go green when everything is right.
    """
    repo, tick = _repo(world), int(getattr(world, "world_tick", 0) or 0)
    got = collections.Counter()
    for pr in _open_requests(world):
        for cid in (getattr(pr, "commit_ids", []) or []):
            commit = (getattr(repo, "commits", {}) or {}).get(cid)
            if int(getattr(commit, "timestamp", 0) or 0) >= since:
                got[getattr(pr, "pr_id", "?")] += 1
    total = sum(got.values())
    if total < 8 or len(_open_requests(world)) < 3:
        return None
    top, n = got.most_common(1)[0]
    if n < 0.7 * total:
        return None
    stale = []
    for pr in _open_requests(world):
        last = max([int(getattr((getattr(repo, "commits", {}) or {}).get(c),
                                "timestamp", 0) or 0)
                    for c in (getattr(pr, "commit_ids", []) or [])] or [0])
        if getattr(pr, "pr_id", "?") != top and tick - last > 48:
            stale.append(f"{getattr(pr, 'pr_id', '?')} last took a commit at "
                         f"t{last}, {tick - last} ticks ago")
    return {"said": f"{top} took {n} of {total} commits since t{since} while "
                    f"{len(stale)} other request(s) went untouched",
            "evidence": stale[:6]}


def work_that_can_never_be_committed(world: Any) -> Optional[Dict[str, Any]]:
    """Finished patches with nowhere to go.

    Two agents held 24 between them for 96 ticks: every request in flight was
    somebody else's, so the module ceiling refused them a branch and the owner
    rule refused them the branches that were failing.
    """
    from environments.org_env.backend.repo.workflow import pending_for_agent
    waiting = {aid: len(pending_for_agent(world, aid))
               for aid in (getattr(world, "agents", {}) or {})}
    waiting = {aid: n for aid, n in waiting.items() if n}
    stranded = {aid: n for aid, n in waiting.items() if n >= 4}
    if not stranded:
        return None
    deferred = [e for e in (getattr(world, "events", []) or [])
                if isinstance(e, dict) and e.get("subtype") == "commit_deferred"]
    return {"said": f"{sum(stranded.values())} finished patch(es) held by "
                    f"{len(stranded)} agent(s) have never reached a branch",
            "evidence": [f"{aid}: {n} waiting" for aid, n in stranded.items()]
                        + [f"commits deferred: {len(deferred)}"]
                        + [str(e.get("brief"))[:120] for e in deferred[-2:]]}


def a_task_is_done_without_meeting_its_gate(world: Any) -> Optional[Dict[str, Any]]:
    """Completion recorded where the requirements were never satisfied."""
    phantom = []
    for t in (getattr(world, "tasks", {}) or {}).values():
        if "COMPLETE" not in str(getattr(t, "status", "")).upper():
            continue
        need = list(getattr(t, "completion_requirements", []) or [])
        have = set(getattr(t, "linked_artifacts", []) or []) | set(
            getattr(t, "evidence", []) or [])
        missing = [r for r in need if r not in have]
        if need and missing:
            phantom.append(f"{getattr(t, 'task_id', '?')} needs {missing}")
    return None if not phantom else {
        "said": f"{len(phantom)} task(s) complete without meeting their own gate",
        "evidence": phantom[:6]}


PROBES: Dict[str, Callable[..., Optional[Dict[str, Any]]]] = {
    # First, because it is upstream of every other question here.
    "code_fails_whatever_the_pipeline_does": the_code_would_fail_even_if_everything_merged,
    "nothing_reaches_mainline": nothing_reaches_mainline,
    "judged_on_work_it_does_not_carry": a_request_is_judged_on_work_it_does_not_carry,
    "same_tree_different_verdicts": the_same_tree_drew_different_verdicts,
    "one_rule_holds_all_the_credit": one_rule_holds_all_the_credit,
    "a_rule_no_change_can_satisfy": a_rule_no_change_can_satisfy,
    "approved_and_never_merged": approved_and_never_merged,
    "frozen_while_the_desk_moves_on": a_request_is_frozen_while_the_desk_moves_on,
    "one_request_swallows_the_commits": one_request_swallows_the_commits,
    "work_that_can_never_be_committed": work_that_can_never_be_committed,
    "done_without_meeting_its_gate": a_task_is_done_without_meeting_its_gate,
}

# The ones that count things that happened, as against reading what is true now.
_TAKES_A_WINDOW = {
    "same_tree_different_verdicts": True,
    "one_rule_holds_all_the_credit": True,
    "a_rule_no_change_can_satisfy": True,
    "one_request_swallows_the_commits": True,
}


def run_probes(world: Any, since: int = 0) -> List[Dict[str, Any]]:
    """Every probe against one world. Never raises.

    ``since`` is the tick this run began at. A resumed run inherits the whole
    history of the one it continues, so counting over all of it means reporting
    forever on defects that were fixed before it started: the run that carried
    these fixes still showed 10 of 17 wobbly trees, all of them judged before
    t72. Probes that count events take the window; probes that read present
    state ignore it, because state is state whenever it arrived.
    """
    alarms = []
    for name, probe in PROBES.items():
        try:
            found = (probe(world, since) if _TAKES_A_WINDOW.get(name)
                     else probe(world))
        except Exception as exc:  # noqa: BLE001  one broken probe must not hide the rest
            alarms.append({"probe": name, "said": f"the probe itself failed: "
                                                  f"{type(exc).__name__}: {exc}",
                           "evidence": [], "broken": True})
            continue
        if found:
            alarms.append({"probe": name, **found})
    return alarms


__all__ = ["PROBES", "run_probes"]
