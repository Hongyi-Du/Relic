"""OSS time-machine product seeding (brief §6.3).

Seed an ``OrgWorld`` from a real OSS project's early runnable release: starter repo files become
agent-visible ``ProductArtifact``s, historical public issues become issue artifacts + backlog tasks,
and the reference repo / hidden tests / held-out issues are stashed as evaluator-only assets. The
agent organization then iterates the product under real friction; evaluation uses the withheld
behavior tests + issue-level acceptance (brief §0).
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from environments.org_env.product.objects import ProductArtifact, ProductState
from environments.org_env.product.seed import DEFAULT_COMPANY_CONFIG
from environments.org_env.product.substrates import anonymize, controls, loader
from environments.org_env.product.substrates.base import OSS_TIME_MACHINE
from environments.org_env.product.substrates.eval_assets import (
    attach_oss_eval_assets,
    qualify_oss_hidden_tests_for_spec,
)

# severity (historical) -> (issue-artifact priority string, backlog task priority int)
# Mirrors the unowned-task perception gate in
# environments/org_env/runtime_adapter/perception.py. A seeded backlog task below
# this priority can never be picked up, so the issue behind it is unreachable.
_MIN_SURFACEABLE_TASK_PRIORITY = 4

_SEVERITY = {
    "critical": ("high", 5),
    "major": ("high", 5),
    "medium": ("medium", 3),
    "minor": ("low", 2),
}
# issue_type -> a backlog task framing (brief §6.3)
_TASK_FRAME = {
    "bug": "Fix",
    "docs": "Improve docs for",
    "usability": "Improve CLI/onboarding for",
    "performance": "Improve performance of",
    "compatibility": "Improve compatibility of",
    "feature": "Implement",
}

_DOC_SUFFIXES = (".md", ".markdown", ".rst", ".txt")


def _slug(path: str) -> str:
    return "art_" + path.replace("/", "_").replace(".", "_").replace("-", "_")


def _function_parameter_signature(src: str) -> dict:
    """{function name: sorted parameter names} — the baseline a later edit is diffed against."""
    from environments.org_env.product.interface_guard import (
        _function_defs,
        _parameter_names,
    )

    return {name: sorted(_parameter_names(node))
            for name, node in _function_defs(src).items()}


def _component_map_or_derived(declared, public_issues):
    """The manifest's component map, or one built from the issues' own path hints.

    A component is only useful if it resolves to files. Hand-curated datasets
    declare the mapping; machine-generated ones name every component logically
    ("public_node_api") and put the real paths on each issue as
    ``candidate_path_hints`` instead, so the manifest map is absent and every
    component resolves to nothing.

    That failure is silent and total: ``_component_artifact_ids`` returns [],
    ``unpatched_coding_issues`` skips the issue, and no ``edit_repo_file``
    candidate is ever dealt - the organization can see the issues and has no way
    to act on any of them. The cells still run and still report, so the outcome
    reads as "fixed nothing" rather than "was never able to try".

    Deriving the map costs nothing where one is declared (it is returned
    untouched) and uses only data the dataset already carries.
    """
    if declared:
        return declared
    derived: dict = {}
    for issue in public_issues or ():
        component = str(getattr(issue, "component", "") or "").strip()
        hints = [str(h).strip() for h in (getattr(issue, "candidate_path_hints", None) or [])]
        hints = [h for h in hints if h]
        if not component or not hints:
            continue
        paths = derived.setdefault(component, [])
        for hint in hints:
            if hint not in paths:
                paths.append(hint)
    return derived


def _artifact_type_for(path: str) -> str:
    p = path.lower()
    if p.endswith(_DOC_SUFFIXES) or p.startswith("docs/") or p.startswith("examples/"):
        return "doc"
    return "repo_file"


def seed_oss_time_machine_product(world: Any, substrate_config: Dict[str, Any]) -> ProductState:
    substrate_config = substrate_config or {}
    dataset_id = str(substrate_config.get("dataset_id", "mini_cli_digest"))
    anonymize_on = bool(substrate_config.get("anonymize", True))

    spec = loader.load_oss_substrate_spec(dataset_id)
    manifest = spec.manifest
    # formal-experiment guard (requirement §9): a formal run must use a REAL OSS snapshot, never a
    # synthetic fixture. mini_cli_digest etc. live under fixtures/ and are dev/test only.
    mode = str(substrate_config.get("mode", "dev"))
    if mode == "formal" and loader.is_fixture_dir(spec.dataset_dir):
        raise ValueError(
            f"formal OSS experiment refuses a fixture substrate: {dataset_id!r} resolves to "
            f"{spec.dataset_dir} (under fixtures/). Use a frozen release benchmark pack "
            "benchmarks/relic-main-v1/packs (for example dataset_id='mini_blobstore_v1').")
    if mode == "formal":
        # dependency preflight (brief review §4): a formal run must have the product's runtime deps,
        # else readiness/smoke/hidden-tests are silently unstable across machines — fail clearly.
        import importlib.util
        deps = list(spec.manifest.get("runtime_dependencies") or [])
        deps.extend(spec.manifest.get("evaluator_dependencies") or [])
        miss = [str(d).split("[")[0].split(">")[0].split("=")[0].split("<")[0].strip() for d in deps]
        miss = [m for m in miss if m and importlib.util.find_spec(m) is None]
        if miss:
            raise RuntimeError(
                f"formal OSS experiment for {dataset_id!r} is missing required runtime dependencies: "
                f"{miss}. Install them (see {os.path.relpath(spec.starter_repo_dir)}/requirements.txt) "
                "or run in a prepared venv/Docker before the formal experiment.")
        qualification = qualify_oss_hidden_tests_for_spec(
            spec,
            timeout=int(os.environ.get("ORG_OSS_QUALIFICATION_TIMEOUT", "180")),
        )
        if not qualification["formal_ready"]:
            raise RuntimeError(
                f"formal OSS hidden suite qualification failed for {dataset_id!r}: "
                f"{qualification['blocking_reasons']}"
            )
        world.__dict__["_oss_hidden_qualification"] = qualification
    # Hidden tests remain evaluator-only during a formal rollout and run once at
    # the controller's final evaluation. Public tests and smoke stay available.
    ec = dict(substrate_config.get("evaluator_config") or {})
    if mode == "formal":
        from environments.org_env.product.substrates.final_evaluation import (
            manifest_requires_manual_release_checks,
        )

        ec["run_oss_hidden_tests"] = False
        ec["run_oss_final_evaluation"] = True
        ec["run_oss_manual_checks"] = manifest_requires_manual_release_checks(
            manifest
        )
        ec["hidden_feedback_forbidden"] = True
        ec["run_oss_public_tests"] = True
        ec["prewarm_smoke"] = True
    world.__dict__["_oss_evaluator_config"] = ec

    public_issues = loader.load_public_issues(spec)
    heldout_issues = loader.load_heldout_issues(spec)
    hidden_test_specs = loader.load_hidden_test_specs(spec)
    starter_files = loader.read_repo_files(spec.starter_repo_dir)
    if not starter_files:
        raise FileNotFoundError(f"OSS dataset {dataset_id!r} has no starter_repo files "
                                f"({spec.starter_repo_dir})")

    # anti-scripting control condition (brief §12): perturb friction / affordance / order so a
    # capability that forms in the main run should NOT form (or transfer) under the control.
    control = controls.resolve(substrate_config, manifest)
    public_issues = controls.apply_to_issues(control, public_issues)

    # Organization and repository identity come from the selected workload.
    cfg = dict(DEFAULT_COMPANY_CONFIG)
    cfg["product_name"] = spec.product_name
    cfg["company_name"] = str(manifest.get("company_name") or cfg.get("company_name"))
    cfg["product_stage"] = "early runnable OSS release (messy but shippable)"
    cfg["product_purpose"] = str(manifest.get("product_summary")
                                 or f"{spec.product_name}: a real OSS tool the org iterates from its "
                                    "historical issues (fix behavior, ship releases).")
    cfg["product_substrate"] = dict(substrate_config)
    world.company_config = cfg

    repo = world.repo_system.repo
    repo.repo_id = f"repo_{spec.project_id}"
    repo.name = spec.product_name
    repo.modules = sorted(starter_files)
    world.company.company_name = cfg["company_name"]

    repo_id = repo.repo_id

    ps = ProductState(
        product_id=f"product_{spec.project_id}",
        name=f"{spec.product_name} (OSS time-machine starter)",
        stage="messy_but_runnable_beta",
        summary=str(manifest.get("product_summary")
                    or f"A real OSS product ({spec.product_name}) at an early runnable release; "
                       "future code and release notes are withheld. Iterate it from historical issues."),
        repo_id=repo_id,
        substrate_type=OSS_TIME_MACHINE,
        substrate_meta={"dataset_id": spec.project_id, "product_name": spec.product_name,
                        "starter_ref": manifest.get("starter_ref", "")},
    )

    arts: Dict[str, ProductArtifact] = {}
    path_to_art: Dict[str, str] = {}
    for path in sorted(starter_files):
        text = starter_files[path]
        atype = _artifact_type_for(path)
        if anonymize_on:
            text = (anonymize.anonymize_text(text, manifest, strip_identity=True) if atype == "doc"
                    else anonymize.anonymize_code(text, manifest))
        aid = _slug(path)
        arts[aid] = ProductArtifact(
            artifact_id=aid, artifact_type=atype, title=path, status="active",
            linked_repo_id=repo_id, linked_file_path=path,
            summary=f"{spec.product_name} starter file: {path}",
            content=text, mainline_content=text,          # brief §6.3.5: never leave mainline empty
            known_gaps=[], created_at_tick=0, updated_at_tick=0)
        ps.artifact_ids.append(aid)
        path_to_art[path] = aid

    # public historical issues -> issue artifacts (+ open issues). Hidden test linkage is NOT
    # exposed on the agent-visible artifact (brief §9.1).
    for iss in public_issues:
        prio_str, _ = _SEVERITY.get(iss.severity, ("medium", 3))
        body = iss.body
        if anonymize_on:
            body = anonymize.anonymize_text(body, manifest, strip_identity=True)
        problem = body + (f"\n\nAcceptance: {iss.acceptance_hint}" if iss.acceptance_hint else "")
        arts[iss.issue_id] = ProductArtifact(
            artifact_id=iss.issue_id, artifact_type="issue", title=iss.title, status="open",
            summary=(f"[{iss.component}] {iss.title}" if iss.component else iss.title),
            problem=problem, priority=prio_str, created_at_tick=int(iss.created_tick or 0),
            updated_at_tick=int(iss.created_tick or 0), issue_status="open")
        ps.artifact_ids.append(iss.issue_id)
        ps.open_issue_ids.append(iss.issue_id)

    # agent-visible systemic gaps = public issue titles (NEVER hidden tests / reference notes)
    ps.known_systemic_issues = list(manifest.get("starter_known_gaps")
                                    or [iss.title for iss in public_issues])

    world.product = ps
    world.product_artifacts = arts
    # Parameter baseline for the propagation check. Only the signatures are kept:
    # the detector asks "which parameters did the organization ADD", and storing
    # whole starter files to answer that would carry the repo twice through every
    # checkpoint for a question about a few dozen names.
    world.__dict__["_starter_function_parameters"] = {
        aid: params
        for aid, params in (
            (aid, _function_parameter_signature(getattr(a, "content", "") or ""))
            for aid, a in arts.items()
            if str(getattr(a, "linked_file_path", "") or "").endswith(".py")
        )
        if params
    }

    # Backlog task specs consumed by OrgWorld.build — one per visible issue.
    # The control may withhold the backlog (affordance absent) — friction stays, means to act gone.
    seed_val = int(getattr(getattr(world, "scenario", None), "seed", 0) or 0)
    component_map = manifest.get("component_map") or {}
    component_map = _component_map_or_derived(component_map, public_issues)
    world.__dict__["_oss_component_map"] = component_map
    world.__dict__["_oss_seed_tasks"] = (
        _build_task_specs(public_issues, path_to_art, component_map)
        if controls.generates_backlog(control) else [])
    # historical-issue stream: release public issues as external pressure over time (brief §9).
    # Held-out issues are NOT placed here, so they can't be released during formation.
    stream = _build_issue_stream(public_issues, manifest, anonymize_on)
    stream = controls.apply_to_stream(control, stream, seed=seed_val)
    world.__dict__["_oss_issue_stream"] = stream
    world.__dict__["_oss_control"] = controls.control_meta(control)
    ps.substrate_meta["control"] = control
    ps.substrate_meta["provenance"] = {
        "repo_url": manifest.get("repo_url", ""),
        "license": manifest.get("license", ""),
        "starter_ref": manifest.get("starter_ref", ""),
        "reference_ref": manifest.get("reference_ref", ""),
        "is_real": not loader.is_fixture_dir(spec.dataset_dir),
    }

    # seed the external community with the dataset's field chatter (brief review §5): organic posts
    # beyond the historical-issue stream, so the external world is a coupled system, not just a bench.
    _seed_external_signals(world, spec)

    # evaluator-only assets (reference repo / hidden tests / held-out issues) — brief §7
    attach_oss_eval_assets(world, spec, heldout_issues, hidden_test_specs)
    # evaluator-side issue provenance (real GitHub linkage / original text / labels) — NOT exposed on
    # the agent-visible product artifacts (it can reveal identity or the future fix).
    world.__dict__["_oss_issue_provenance"] = {
        iss.issue_id: iss.provenance() for iss in public_issues
    }
    return ps


def _seed_external_signals(world: Any, spec: Any) -> int:
    """Inject the dataset's seed external posts into world.community (brief review §5)."""
    community = getattr(world, "community", None)
    if community is None or not hasattr(community, "add_post"):
        return 0
    posts = loader.load_external_signals(spec)
    if not posts:
        return 0
    try:
        from environments.org_env.backend.community.objects import Post
    except Exception:
        return 0
    n = 0
    for rec in posts:
        pid = str(rec.get("post_id") or f"ext_seed_{n}")
        community.add_post(Post(
            post_id=pid, author_id=str(rec.get("author_id", "ext_user")),
            topic=str(rec.get("topic", "")), content_summary=str(rec.get("content_summary", "")),
            stance=float(rec.get("stance", 0.0) or 0.0), credibility=float(rec.get("credibility", 0.5) or 0.5),
            reach=int(rec.get("reach", 10) or 10), created_tick=int(rec.get("created_tick", 0) or 0),
            visibility="public"))
        n += 1
    world.__dict__["_oss_seed_signal_ids"] = [str(r.get("post_id")) for r in posts if r.get("post_id")]
    return n


def _build_issue_stream(public_issues: List[Any], manifest: Dict[str, Any],
                        anonymize_on: bool) -> List[Dict[str, Any]]:
    first_tick = int(((manifest.get("issue_stream") or {}).get("first_tick", 0)) or 0)
    stream: List[Dict[str, Any]] = []
    for iss in public_issues:
        rt = int(iss.created_tick or 0)
        body = anonymize.anonymize_text(iss.body, manifest, strip_identity=True) if anonymize_on else iss.body
        stream.append({
            "issue_id": iss.issue_id, "title": iss.title, "body": body,
            "component": iss.component, "severity": iss.severity, "issue_type": iss.issue_type,
            "release_tick": rt if rt > 0 else first_tick, "released": False,
        })
    return stream


def resolve_component_artifacts(component: str, component_map: Dict[str, Any],
                                path_to_art: Dict[str, str]) -> List[str]:
    """Resolve an issue ``component`` to artifact ids: an exact file path, else a logical name via
    ``component_map`` (req §7). Only paths that exist as artifacts are linked (e.g. Dockerfile, not
    in the frozen subtree, yields none)."""
    comp = (component or "").strip()
    if not comp:
        return []
    if comp in path_to_art:
        return [path_to_art[comp]]
    paths = component_map.get(comp) or []
    return [path_to_art[p] for p in paths if p in path_to_art]


def _build_task_specs(public_issues: List[Any], path_to_art: Dict[str, str],
                      component_map: Dict[str, Any]) -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []
    for iss in public_issues:
        _, prio_int = _SEVERITY.get(iss.severity, ("medium", 3))
        # Severity is a domain judgement; task priority is also a VISIBILITY
        # threshold - perception only surfaces unowned tasks at priority >= 4
        # (runtime_adapter/perception.py). Mapping severity straight through left
        # every medium/minor historical issue invisible to the org: measured on a
        # real run, only 3 of 10 issue-linked tasks were surfaceable. The other
        # creation path (world._reconcile_issue_backlog) already floors at 4 for
        # exactly this reason; this one did not know the gate existed.
        # Keep the severity ORDERING (critical/major still outrank the rest) but
        # never fall below the visibility floor.
        prio_int = max(int(prio_int), _MIN_SURFACEABLE_TASK_PRIORITY)
        frame = _TASK_FRAME.get(iss.issue_type, "Address")
        linked_art = resolve_component_artifacts(iss.component, component_map, path_to_art)
        desc = iss.body.strip()
        if iss.acceptance_hint:
            desc = (desc + f"\nAcceptance: {iss.acceptance_hint}").strip()
        specs.append({
            "task_id": f"task_oss_{iss.issue_id}",
            "title": f"{frame}: {iss.title}",
            "description": desc[:400],
            "linked_issues": [iss.issue_id],
            "linked_artifacts": linked_art,
            "priority": prio_int,
        })
    return specs


def seed_substrate(world: Any, substrate_config: Dict[str, Any] = None) -> ProductState:
    return seed_oss_time_machine_product(world, substrate_config or {})


__all__ = ["seed_oss_time_machine_product", "seed_substrate"]
