"""Product Contract Graph (v13 P2) — the cross-module interface contract as a FIRST-CLASS object.

v12's failure was not "no evidence protocol" but "no interface-contract governance": each
module got locally better, yet `Claim / Source / ResearchResult / Eval / Report / Smoke` drifted
out of agreement, so the product stopped running end-to-end (research_loop passed a dict where
ClaimTracker.add expected text+source_ids). This module makes the producer/consumer schema
explicit + classifiable, so an integration break is caught at CI/PR time (does the product still
RUN end-to-end) — distinct from the release gate's quality bar (is the product good enough).
"""
from __future__ import annotations

from typing import Any, Dict

CONTRACT_VERSION = 1

# canonical schema (first-class, inspectable, governable)
PRODUCT_CONTRACT: Dict[str, Any] = {
    "version": CONTRACT_VERSION,
    "objects": {
        "Claim": ["text", "source_ids", "evidence", "uncertainty_note"],
        "Source": ["source_id", "title", "path", "url", "credibility_score"],
        "ResearchResult": ["claims", "sources", "report"],
    },
    "interfaces": {
        "run_research": {"inputs": ["query"], "output": "ResearchResult", "file": "research_loop.py"},
        "ClaimTracker.add": {"inputs": ["text", "source_ids", "evidence", "uncertainty_note"],
                             "file": "tools/claim_tracker.py"},
        "SourceTracker.add": {"inputs": ["rec"], "output": "Source", "file": "tools/source_tracker.py"},
        "run_eval": {"inputs": ["claims", "sources"], "output": "metrics", "file": "eval/eval_stub.py"},
        "write_report": {"inputs": ["query", "claims", "sources"], "file": "tools/report_writer.py"},
        "smoke_check": {"role": "end_to_end_test", "file": "smoke_check.py"},
    },
    # (producer_file, consumer_file, shared_object) — edits to either side must keep the contract
    "boundaries": [
        ["research_loop.py", "tools/claim_tracker.py", "Claim"],
        ["research_loop.py", "eval/eval_stub.py", "Claim+Source"],
        ["research_loop.py", "tools/report_writer.py", "Claim+Source"],
        ["smoke_check.py", "research_loop.py", "ResearchResult"],
        ["agent.py", "eval/eval_stub.py", "run_eval signature"],
    ],
}

# the files whose edits touch a contract boundary (a PR touching these must pass the contract CI)
CONTRACT_FILES = {
    "research_loop.py", "tools/claim_tracker.py", "tools/source_tracker.py",
    "eval/eval_stub.py", "tools/report_writer.py", "smoke_check.py", "agent.py",
}


def touches_contract(file_path: str) -> bool:
    fp = (file_path or "").replace("\\", "/")
    return fp in CONTRACT_FILES or fp.split("/")[-1] in {f.split("/")[-1] for f in CONTRACT_FILES}


_BOUNDARY_HINTS = [
    # v14c: the Claim SHAPE contract (object vs dict) — agent.py expected dict-shaped claims while
    # research_loop returned Claim objects ("claim at index 0 must be a dict"). Keep this BEFORE the
    # generic "claim" hint so it localizes to the real producer/consumer boundary.
    (("must be a dict", "must be dict", "is not a dict", "expected dict", "claim at index",
      "object is not subscriptable", "not subscriptable"),
     "agent.py <-> research_loop.py (Claim shape: object vs dict — canonicalize the Claim schema)"),
    (("source_id", "source_ids"), "research_loop.py <-> tools/claim_tracker.py (Claim.source_ids)"),
    (("run_eval", "positional", "takes", "argument"), "agent.py/smoke_check.py <-> eval/eval_stub.py (run_eval signature)"),
    (("credibility", "credibility_score"), "tools/source_tracker.py <-> eval/eval_stub.py (Source.credibility_score)"),
    (("attribute", "has no attribute", ".all"), "research_loop.py <-> tools/source_tracker.py (SourceTracker API)"),
    (("claim", "evidence"), "research_loop.py <-> tools/report_writer.py (Claim.evidence)"),
]


def classify_contract_break(err: str) -> str:
    """Map a smoke/build error to the contract boundary that broke (best-effort, for routing)."""
    e = (err or "").lower()
    for keys, boundary in _BOUNDARY_HINTS:
        if any(k in e for k in keys):
            return boundary
    return ""


def product_local_modules(world: Any) -> set:
    """Import names the exported product tree provides itself.

    Read from the tree rather than listed, because a list can only ever describe
    one product. Written down, it named LanternScout's files, so on every OSS Pack
    the product's OWN package — blobstore, boltons, anyio — counted as absent from
    the sandbox: a tree that could not import itself was excused instead of
    refused, and would have merged.
    """
    names = set()
    for artifact in (getattr(world, "product_artifacts", {}) or {}).values():
        path = str(getattr(artifact, "linked_file_path", "") or "")
        if not path.endswith(".py"):
            continue
        parts = [p for p in path.replace("\\", "/").split("/") if p]
        if len(parts) == 1:
            names.add(parts[0][:-3])                       # a top-level module
            continue
        names.add(parts[0])
        # A src layout is imported by the package inside it, not by "src".
        if parts[0] in ("src", "lib") and len(parts) > 2:
            names.add(parts[1])
    return names


def declared_dependency_names(world: Any) -> set:
    """Import names the Pack declares its product and suites need.

    A dependency the Pack names is the operator's responsibility to install; a
    name it does not is the organization's own. Read from the manifest so the
    two can never disagree, and reduced to import names because a declaration
    reads ``typing_extensions >= 4.5; python_version < '3.13'``.
    """
    from environments.org_env.product.substrates.eval_assets import oss_eval_assets

    try:
        # Callers include probes that hand in a stand-in rather than a live world,
        # and a substrate with no Pack behind it declares nothing. Either way the
        # answer is "no declarations", not a crash in the middle of a CI verdict.
        manifest = (oss_eval_assets(world) or {}).get("manifest") or {}
    except Exception:  # noqa: BLE001
        return set()
    declared = list(manifest.get("runtime_dependencies") or [])
    for key in ("public_tests", "hidden_tests"):
        section = manifest.get(key)
        if isinstance(section, dict):
            declared.extend(section.get("dependencies") or [])
    names = set()
    for item in declared:
        head = str(item).split(";")[0].strip()
        head = head.split("[")[0]
        for sep in (">=", "<=", "==", "!=", "~=", ">", "<", " "):
            head = head.split(sep)[0]
        head = head.strip()
        if head:
            # a distribution is named with hyphens, imported with underscores
            names.add(head.replace("-", "_"))
            names.add(head)
    return names


def missing_sandbox_dependency(
    err: str, local: set | None = None, declared: set | None = None
) -> str:
    """The DECLARED third-party module the sandbox lacks, if that is what the smoke reports.

    A product that cannot import a library it declares is a statement about the
    environment, not about the code: anyio imports typing_extensions, the
    evaluator image ships only pytest, and every request on the pack was refused
    with ``ModuleNotFoundError: No module named 'typing_extensions'`` classified
    as a contract break. Nothing an organization can write repairs that, so the
    refusal is unanswerable and the delivery chain never reaches a merge.

    The excuse is bounded by the declaration, because otherwise it covers an
    invented name too. An arm wrote ``from context import Context`` into
    _tasks.py, no such distribution exists, the package stopped importing, and
    all sixteen oracles went from five passing to infra_error while CI raised no
    objection: an unimportable product had been read as an unprepared sandbox.
    A name the Pack never declared is the organization's to answer for.

    ``local`` covers the product's own modules for the same reason: deleting or
    renaming something its own code imports is a real break.
    """
    import re

    text = err or ""
    # A test-runner plugin the suite asks for BY NAME is a second shape of the
    # same fact. anyio's pyproject carries `addopts = ... -p pytest_mock`, so the
    # runner reports `ImportError: Error importing plugin "pytest_mock"` and
    # never reaches "No module named": the first shape alone left the pack
    # refused for a plugin its public suite does not even use.
    plugin = re.search(r"[Ee]rror importing plugin ['\"]([^'\"]+)['\"]", text)
    if plugin:
        return _absent_name(plugin.group(1), local, declared)
    match = re.search(
        r"(?:ModuleNotFoundError|ImportError): No module named ['\"]([^'\"]+)['\"]",
        text,
    )
    if not match:
        return ""
    return _absent_name(match.group(1), local, declared)


def reconstruction_build_contract_check(world: Any, pr: Any = None) -> Dict[str, Any]:
    """Fail closed when a reconstruction build replaces its selected source.

    A successful compiler exit is not evidence that the organization delivered
    its implementation: a build script can overwrite the implementation path
    with a tiny placeholder and then compile that different program.  Judge the
    exact merge candidate here, before smoke execution, so both action-driven
    CI and the world's automatic sweep enforce the same source/build identity.
    """
    product = getattr(world, "product", None)
    meta = getattr(product, "substrate_meta", {}) or {}
    if not bool(meta.get("allow_unbound_reconstruction_issue")):
        return {"ok": True, "brief": "", "boundary": "", "kind": ""}

    compile_path = str(meta.get("reconstruction_compile_path") or "").replace(
        "\\", "/").strip("/")
    if not compile_path:
        return {"ok": True, "brief": "", "boundary": "", "kind": ""}

    from environments.org_env.product.materialize import (
        _file_text,
        _is_unmerged_new_file,
        merge_candidate_text,
        programbench_integration_candidate_overrides,
        programbench_pr_candidate_overrides,
        programbench_strict_mainline_overrides,
    )
    from environments.org_env.runtime_adapter.execution import (
        _build_contract_mentions_source,
        _reconstruction_implementation_paths,
    )

    adapted = False
    try:
        from environments.org_env.programbench import programbench_profile_active

        adapted = programbench_profile_active(world)
    except (ImportError, AttributeError, TypeError, ValueError):
        adapted = False
    if adapted:
        candidate_overrides = (
            programbench_pr_candidate_overrides(world, pr)
            if pr is not None
            else programbench_integration_candidate_overrides(world)
        )
        overrides = programbench_strict_mainline_overrides(
            world,
            candidate_overrides=candidate_overrides,
        )
    else:
        overrides = merge_candidate_text(world, pr) if pr is not None else {}
    candidate_by_path: Dict[str, str] = {}
    for artifact_id, artifact in (
            getattr(world, "product_artifacts", {}) or {}).items():
        path = str(getattr(artifact, "linked_file_path", "") or "").replace(
            "\\", "/").strip("/")
        if not path:
            continue
        if artifact_id in overrides:
            candidate_by_path[path] = str(overrides[artifact_id] or "")
            continue
        if (pr is not None or adapted) and _is_unmerged_new_file(artifact):
            continue
        candidate_by_path[path] = _file_text(artifact, pr is not None)

    source_paths = _reconstruction_implementation_paths(world)
    if not source_paths:
        # Before the organization has selected an implementation path, the
        # ordinary compile smoke remains the only available contract.
        return {"ok": True, "brief": "", "boundary": "", "kind": ""}

    build_text = candidate_by_path.get(compile_path, "")
    if any(_build_contract_mentions_source(build_text, path)
           for path in source_paths):
        return {"ok": True, "brief": "", "boundary": "", "kind": ""}

    rendered_sources = ", ".join(source_paths[:4])
    brief = (
        f"{compile_path}: build contract must preserve and consume the selected "
        f"implementation source ({rendered_sources}). The live build, copy, or "
        "launch command must contain a selected repo-relative source path literally "
        "on that same command line; assigning the path to a shell variable and later "
        "passing only `$variable` does not satisfy this static contract"
    )
    return {
        "ok": False,
        "brief": brief,
        "boundary": compile_path,
        "kind": "contract_break",
        "detail": brief,
    }


def _absent_name(raw: str, local: set | None, declared: set | None) -> str:
    """The name, when it is the sandbox's to provide, and "" when it is not."""
    name = str(raw).split(".")[0]
    if name in (local or set()):
        return ""
    if declared is None:                    # caller has no manifest to check against
        return name
    return name if name in declared else ""


def run_contract_check(world: Any, pr: Any = None) -> Dict[str, Any]:
    """Does the product still RUN end-to-end (no interface break)? This is the INTEGRATION
    check — independent of grounding quality. ok == the materialized product's smoke exits 0
    (run_research -> claim/source -> eval -> report -> smoke all wired). A non-zero exit means
    a contract boundary broke; we classify which one for the debugging loop.

    An INFRASTRUCTURE failure (the smoke could not be launched or the sandbox could
    not see the exported workspace) is reported as ``kind="infrastructure_error"``
    and must never be attributed to the product: a 336-tick formal run once recorded
    172/172 ``ci_contract_break`` from an empty container mount, which both zeroed the
    product-outcome axis and inflated governance activity with enforcement against a
    defect that did not exist.
    """
    reconstruction = reconstruction_build_contract_check(world, pr)
    if not reconstruction.get("ok"):
        return reconstruction

    from environments.org_env.product.materialize import (
        merge_candidate_text,
        pr_public_test_command,
        programbench_integration_candidate_overrides,
        programbench_pr_candidate_overrides,
        programbench_strict_mainline_overrides,
        release_smoke,
        smoke_error_brief,
    )
    adapted = False
    try:
        from environments.org_env.programbench import programbench_profile_active

        adapted = programbench_profile_active(world)
    except (ImportError, AttributeError, TypeError, ValueError):
        adapted = False
    if pr is not None:
        # Judge what merging THIS pull request would produce: the mainline plus
        # the changes it carries. Judging the whole working tree instead failed a
        # request for code it does not touch, which on a pack whose steps stack
        # locks correct early work behind unfinished later work forever.
        overrides = (
            programbench_pr_candidate_overrides(world, pr)
            if adapted
            else merge_candidate_text(world, pr)
        )
        if adapted:
            overrides = programbench_strict_mainline_overrides(
                world,
                candidate_overrides=overrides,
            )
        command = pr_public_test_command(world, pr)
        if overrides:
            sm = release_smoke(
                world,
                prefer_mainline=True,
                overrides=overrides,
                command=command,
            )
        else:
            sm = release_smoke(world, prefer_mainline=True, command=command)
    else:
        if adapted:
            overrides = programbench_strict_mainline_overrides(
                world,
                candidate_overrides=programbench_integration_candidate_overrides(
                    world
                ),
            )
            sm = release_smoke(
                world,
                prefer_mainline=True,
                overrides=overrides,
            )
        else:
            sm = release_smoke(world, prefer_mainline=False)   # no PR named: the working tree
    if sm.get("ok"):
        return {"ok": True, "boundary": "", "brief": "", "rc": sm.get("returncode"),
                "kind": ""}
    launch_error = sm.get("error")
    if launch_error:
        return {"ok": False, "boundary": "", "brief": str(launch_error),
                "rc": sm.get("returncode"), "kind": "infrastructure_error"}
    brief = smoke_error_brief(sm)
    passed = _checks_the_product_passed(sm)
    local = product_local_modules(world)
    declared = declared_dependency_names(world)
    absent = (missing_sandbox_dependency(brief, local, declared)
              or missing_sandbox_dependency(str(sm.get("stderr_tail") or ""),
                                            local, declared))
    if absent:
        return {"ok": False, "boundary": "", "kind": "infrastructure_error",
                "rc": sm.get("returncode"), "passed": passed,
                "brief": (f"sandbox is missing {absent}, which the product imports; "
                          f"install it in the evaluator image and declare it under "
                          f"runtime_dependencies ({brief})")[:300]}
    return {"ok": False, "boundary": classify_contract_break(brief), "brief": brief,
            "rc": sm.get("returncode"), "kind": "contract_break", "passed": passed,
            # Everything the gate said, beside the one line `brief` keeps. A pack
            # whose steps stack fails several at once and reports each on its own
            # line; `brief` is the last of them, so four of five complaints had
            # nowhere to go. Recording them changes what nobody reads by default —
            # `brief` is untouched — and gives whoever does read them the whole
            # verdict instead of its last line.
            "detail": (
                f"{sm.get('stdout_tail') or ''}\n{sm.get('stderr_tail') or ''}"
            )[-4000:]}


def run_metric_consistency_check(world: Any, pr: Any = None) -> Dict[str, Any]:
    """v14b CI: does the eval report INTERNALLY CONSISTENT metrics? A correct `run_eval` keeps
    `unsupported_claim_rate ≈ 1 - claim_evidence_coverage`. A patch that breaks this (the t232
    regression: unsupported=1.0 while coverage stayed 1.0) must fail CI HERE so it never merges —
    otherwise the contradictory metric blocks every release and mislocalizes the org to
    claim_tracker.py for weeks. Localizes the break to eval/eval_stub.py."""
    from environments.org_env.product.materialize import (
        merge_candidate_text,
        programbench_integration_candidate_overrides,
        programbench_pr_candidate_overrides,
        programbench_strict_mainline_overrides,
        release_smoke,
        smoke_eval_inconsistency,
    )
    adapted = False
    try:
        from environments.org_env.programbench import programbench_profile_active

        adapted = programbench_profile_active(world)
    except (ImportError, AttributeError, TypeError, ValueError):
        adapted = False
    if adapted:
        candidate_overrides = (
            programbench_pr_candidate_overrides(world, pr)
            if pr is not None
            else programbench_integration_candidate_overrides(world)
        )
        sm = release_smoke(
            world,
            prefer_mainline=True,
            overrides=programbench_strict_mainline_overrides(
                world,
                candidate_overrides=candidate_overrides,
            ),
        )
    else:
        sm = release_smoke(world, prefer_mainline=False)
    brief = smoke_eval_inconsistency(sm)
    return {"ok": not brief, "boundary": ("eval/eval_stub.py" if brief else ""), "brief": brief}


def _checks_the_product_passed(sm: Dict[str, Any]) -> int | None:
    """How many of the product's own checks passed, if its gate says so.

    The count is the progress term: a refusal that reads the same as last time
    still means the work moved if more checks pass behind it.
    """
    import re

    text = f"{sm.get('stdout_tail') or ''}\n{sm.get('stderr_tail') or ''}"
    counts = [int(m.group(1)) for m in re.finditer(r"(\d+) passed", text)]
    return max(counts) if counts else None


def failure_signature(verdict: Dict[str, Any]) -> tuple:
    """A stable identity for a refusal, so "the same complaint" is a real question.

    Comparing whole messages could not answer it: a temporary path, a tick, or a
    reordered list changes the text while the situation is identical, and an
    organization that had genuinely moved on could look stuck. What identifies a
    refusal is the step it failed, the module it belongs to, and the kind of error
    — each read off the message the gate already leads with.

    Failed test identities would belong here too, but the CI path records only a
    message; the ids exist solely on the public-suite path. The count of checks
    that DO pass is tracked separately as the progress term rather than folded in,
    so more passing always resets the count even when the top failure is unchanged.
    """
    import re

    brief = str(verdict.get("brief") or "")
    module = ""
    match = re.match(r"\s*([\w./\\-]+\.py)", brief)
    if match:
        module = match.group(1).replace("\\", "/")
    elif verdict.get("boundary"):
        module = str(verdict["boundary"]).split()[0]
    step = ""
    match = re.search(r"\bstep (\d+)\b", brief)
    if match:
        step = f"step {match.group(1)}"
    kind = ""
    match = re.search(r"\b([A-Z]\w*(?:Error|Exception|Warning))\b", brief)
    if match:
        kind = match.group(1)
    return (step, module, kind or str(verdict.get("kind") or ""))


def _landed_on(world: Any, module: str) -> int:
    """How many accepted patches this module has received."""
    arts = getattr(world, "product_artifacts", {}) or {}
    targets = {aid for aid, a in arts.items()
               if str(getattr(a, "linked_file_path", "") or "") == module}
    if not targets:
        return 0
    return sum(1 for p in (getattr(world, "patches", {}) or {}).values()
               if str(getattr(p, "target_object_id", "")) in targets
               and getattr(p, "validation_status", "") in ("accepted", "applied", "merged"))


def note_gate_stall(world: Any, verdict: Dict[str, Any]) -> None:
    """Track, per module, how many REPAIRS have changed nothing the gate can see.

    What retires an issue is a stretch of work that got nowhere. Counting attempts
    instead ended work that was getting somewhere: on a five-step greenfield Pack
    every issue retired at exactly four attempts while the refusals over that
    stretch had been moving — a missing symbol, then a parameter name, then a
    KeyError from real logic.

    The unit has to be a repair, not a reading. Counting every verdict made the
    number say how often anyone had looked: with six agents and five open requests
    the sweep re-judges each one every tick, so one module reached twenty-seven
    while it had received five patches, and the limit was passed inside a single
    tick without anybody having tried anything.
    """
    if world is None:
        return
    signature = failure_signature(verdict)
    module = signature[1]
    if not module:
        return
    passed = verdict.get("passed")
    landed = _landed_on(world, module)
    book = world.__dict__.setdefault("_gate_stall", {})
    entry = book.get(module) or {"signature": None, "passed": -1, "stalled": 0, "landed": -1}
    moved = (signature != entry["signature"]
             or (passed is not None and passed > entry["passed"]))
    if moved:
        entry["stalled"] = 0
    elif landed > entry.get("landed", -1):
        # A repair landed and the gate says exactly what it said before.
        entry["stalled"] = entry["stalled"] + 1
    # else: the same verdict on the same code, which is a second reading of one
    # answer and says nothing new about whether the work is moving.
    entry["signature"] = signature
    entry["landed"] = max(landed, entry.get("landed", -1))
    if passed is not None:
        entry["passed"] = max(passed, entry["passed"])
    book[module] = entry


def record_integration_verdict(ci: Any, pr: Any, verdict: Dict[str, Any],
                               world: Any = None) -> str:
    """Write one integration verdict onto both the CI record and the request.

    The two callers — the run_ci action and the world's own sweep — each did this
    by hand and each did it differently, so the record and the flag disagreed and
    the record contradicted itself.

    The sweep set ``pr.ci_passed = False`` and wrote the status only to a local
    variable, so a request carried a refusal while its one CI run still read
    "passed". The action flipped ``ci.status`` to "failed" without adding a
    reason, so runs read "failed" with an empty ``failure_reasons`` — a verdict
    that denies itself. And the sweep refused a request for an infrastructure
    error while the action deliberately did not, so whether an outage blocked
    delivery depended on which code path happened to look.

    Returns the status written, so a caller that reports it stays in agreement.
    """
    from environments.org_env.cooperbench.source_views import source_views_enabled
    if ci is not None and world is not None and source_views_enabled(world):
        from copy import deepcopy
        # Keep the executed source evidence on the existing CI record. A short
        # PR headline must not be the owner's only account of a failed merge.
        ci.checks = [row for row in getattr(ci, "checks", [])
                     if row.get("name") != "cooperbench_source_ci"]
        if not verdict.get("ok") and verdict.get("kind") == "contract_break":
            ci.checks.append({"name": "cooperbench_source_ci", "ok": False,
                              "verdict": deepcopy(verdict)})
    if verdict.get("kind") == "coordination_conflict":
        if ci is not None:
            ci.status = "blocked"
            ci.failure_reasons = [str(verdict.get("brief") or "source merge conflict")]
        if pr is not None:
            pr.ci_passed = False
            pr.test_status = "blocked"
            pr.ci_brief = str(verdict.get("brief") or "source merge conflict")
        if world is not None:
            linked = list(getattr(pr, "linked_issue_ids", []) or [])
            if getattr(pr, "linked_issue", None):
                linked.append(getattr(pr, "linked_issue"))
            feature_ids = [
                str(item) for item in linked
                if str(item).startswith("cooper_feature_")
            ]
            world.__dict__["_cooperbench_coordination_required"] = {
                "pr_id": str(getattr(pr, "pr_id", "") or ""),
                "owner_id": str(getattr(pr, "author_id", "") or ""),
                "feature_id": feature_ids[0] if len(set(feature_ids)) == 1 else "",
                "kind": "coordination_conflict", "conflict_paths": list(verdict.get("conflict_paths") or []),
                "native_append_error": str(verdict.get("native_append_error") or ""),
                "main_commit_ids": list(verdict.get("main_commit_ids") or []),
                "source_snapshot": dict(verdict.get("source_snapshot") or {}),
                "merge_snapshot": dict(verdict.get("merge_snapshot") or {}),
            }
        return "blocked"
    infra = verdict.get("kind") == "infrastructure_error"
    if verdict.get("ok"):
        return getattr(ci, "status", "passed") if ci is not None else "passed"
    if infra:
        # An outage does not condemn the product and does not clear it either.
        #
        # Not condemn: a library missing from the image once put 172 of 172 runs
        # on a pack down as contract breaks, which both emptied the product axis
        # and filled the governance one with enforcement against a defect that
        # was not there. So no contract-break event, no stall, and test_status is
        # left saying whatever the lightweight tests actually found.
        #
        # Not clear: mainline is what the evaluation scores, and merging on the
        # strength of a check that did not run puts unverified code there. The
        # request waits instead, and the periodic re-check takes it green once
        # the outage passes.
        if ci is not None:
            ci.status = "not_run"
            reasons = getattr(ci, "failure_reasons", None)
            said = str(verdict.get("brief", ""))[:300]
            if isinstance(reasons, list) and said and said not in reasons:
                reasons.append(said)
        if pr is not None:
            pr.ci_passed = False
            pr.ci_brief = str(verdict.get("brief", ""))[:300]
        return "not_run"
    note_gate_stall(world, verdict)
    brief = str(verdict.get("brief", ""))[:300]
    if ci is not None:
        ci.status = "failed"
        reasons = getattr(ci, "failure_reasons", None)
        if isinstance(reasons, list) and brief and brief not in reasons:
            reasons.append(brief)
    if pr is not None:
        pr.ci_passed = False
        pr.test_status = "failed"
        pr.ci_brief = brief
    return "failed"


def record_working_tree_break(world: Any, verdict: Dict[str, Any]) -> None:
    """Keep the desk's own break current, set on fail and cleared on pass.

    The code editor edits the working tree, so the break its debug loop chases
    has to be the working tree's. Writing a request's merge-candidate verdict
    here instead pointed it at a different tree from the one it can edit: a
    request carrying one file of a stacked pack has a candidate that is mostly
    stubs, the gate skips the steps nobody has started, and the desk's real break
    was cleared by a candidate that was never going to show it.

    An infrastructure failure says nothing about the product and must not enter
    the loop: it would send agents hunting a defect that does not exist.
    """
    infra = verdict.get("kind") == "infrastructure_error"
    quiet = bool(verdict.get("ok")) or infra
    world.__dict__["_build_error"] = "" if quiet else str(verdict.get("brief", ""))
    # Beside it, everything the gate said. The brief keeps one line, and a pack
    # whose steps stack fails several at once: an organization reading only the
    # brief heard that Manifest.__init__ was wrong and never that
    # UploadConflictError, ReplicaStatus and GCPlan were missing too.
    world.__dict__["_build_error_detail"] = (
        "" if quiet else str(verdict.get("detail") or ""))


def run_integration_ci(world: Any, pr: Any = None) -> Dict[str, Any]:
    """Combined PR-time CI: the product must RUN end-to-end (contract) AND report self-consistent
    eval metrics (metric consistency). Either failure fails the PR's CI (and routes the right file
    into the debugging loop). Returns {ok, brief, boundary, kind}.

    Given a ``pr``, the contract check judges the merge candidate rather than the
    whole working tree, so a request is answerable for the change it carries."""
    from environments.org_env.cooperbench.source_views import source_views_enabled
    if pr is not None and source_views_enabled(world):
        from environments.org_env.cooperbench.source_ci import run_pr_source_ci
        return run_pr_source_ci(world, pr)
    cc = run_contract_check(world, pr)
    if not cc.get("ok"):
        # Preserve an infrastructure verdict: only a real product failure may be
        # classified (and later enforced against) as a contract break.
        return {"ok": False, "brief": cc.get("brief", ""), "boundary": cc.get("boundary", ""),
                "kind": cc.get("kind") or "contract_break",
                "detail": cc.get("detail", "")}
    mc = run_metric_consistency_check(world, pr)
    if not mc.get("ok"):
        return {"ok": False, "brief": mc.get("brief", ""), "boundary": mc.get("boundary", ""),
                "kind": "metric_inconsistency"}
    return {"ok": True, "brief": "", "boundary": "", "kind": ""}


__all__ = ["PRODUCT_CONTRACT", "CONTRACT_FILES", "CONTRACT_VERSION",
           "touches_contract", "classify_contract_break", "run_contract_check",
           "run_metric_consistency_check", "run_integration_ci",
           "reconstruction_build_contract_check",
           "missing_sandbox_dependency", "product_local_modules",
           "declared_dependency_names",
           "record_integration_verdict", "failure_signature", "note_gate_stall"]
