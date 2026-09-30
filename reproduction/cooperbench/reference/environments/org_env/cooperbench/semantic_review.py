"""Public-spec semantic peer review for the two-person Cooper treatment."""
from __future__ import annotations

import ast
from copy import deepcopy
import difflib
import hashlib
import json
import re
from pathlib import PurePosixPath
from typing import Any, Mapping

from .behavior_probes import (
    PUBLIC_CONTRACT_BOUNDARY_GUIDANCE,
    _contract_binding,
    _protected_integration_contract,
    authorized_atomic_failure,
    prepare_public_behavior_evidence,
    visible_review_contract,
)
from .public_contract import acceptance_index
from .probe_prerequisites import PREREQUISITE_GUIDANCE, public_prerequisite_context
from .probe_preconditions import observer_evidence, grounded_observer_basis


_MAX_REVIEW_FILE_CHARS = 80_000
_MAX_REVIEW_TOTAL_CHARS = 240_000
_MAX_REVIEW_PROJECTED_FILE_CHARS = 22_000
_MAX_REVIEW_CHANGE_RANGE_CHARS = 1_000
_SOURCE_PROJECTION_CONTEXT_LINES = 12
_MAX_REVIEW_PROBE_OUTPUT_CHARS = 1_600
_REVIEW_INPUT_STRATEGY = "three_way_source_review_complete_requirements_v12"
SEMANTIC_REVIEW_REASONING_EFFORT = "medium"
PROBE_DEFECT_ADJUDICATION_REASONING_EFFORT = "medium"
PROBE_VALIDITY_SCHEMA = "cooperbench_probe_failure_validity_v11_cache_ownership"
_CONTRACT_RELATION_TO_VERDICT = {
    "explicitly_authorized": "valid",
    "unauthorized_assertion": "defective",
    "scaffolding_failure": "defective",
    "indeterminate": "unknown",
}
_LEGACY_VERDICT_TO_CONTRACT_RELATION = {
    "valid": "explicitly_authorized",
    "defective": "unauthorized_assertion",
    "unknown": "indeterminate",
}
# A semantic reviewer is useful as a repair signal, but it is not an oracle.
# This threshold is diagnostic only: after three rejected repaired heads the
# trajectory records that the ordinary budget was extended.  It must never be
# converted into approval because Cooper's public tests predate the requested
# feature and cannot prove the new public API.
MAX_BLOCKING_SEMANTIC_REVIEW_ROUNDS = 3


def _required_ordered_source_stages(requirement: Any) -> list[str]:
    """Return public transformation stages that need a source witness.

    This deliberately compiles only the *named* operation, not its hidden or
    underspecified algorithm.  For example, public prose that says a callback
    runs after trimming requires evidence that a trim stage exists before the
    callback; it does not let the host choose which bytes are padding.
    """

    text = str(requirement or "")
    has_observer = re.search(
        r"\b(?:callback|hook|observer|listener|handler|notif(?:y|ication)|metric)\w*\b",
        text,
        re.IGNORECASE,
    )
    has_order = re.search(
        r"\b(?:after|before|then|follow(?:ed|ing)|prior|subsequent)\b",
        text,
        re.IGNORECASE,
    )
    has_trim = re.search(
        r"\b(?:trim(?:s|med|ming)?|strip(?:s|ped|ping)?)\b",
        text,
        re.IGNORECASE,
    )
    return ["trim"] if has_observer and has_order and has_trim else []


def _source_operation_identifiers(content: str) -> set[str]:
    """Collect invoked operations while ignoring prose, comments and aliases.

    A local named ``trimmed_data`` is not a transformation.  Only a call such
    as ``data.rstrip(...)`` / ``trim_audio(...)`` or an explicitly enabled
    ``trim=...`` call argument can witness an executable trim stage.
    """

    try:
        tree = ast.parse(str(content or ""))
    except (SyntaxError, TypeError, ValueError):
        return set()
    identifiers: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            identifiers.add(node.func.id.lower())
        elif isinstance(node.func, ast.Attribute):
            identifiers.add(node.func.attr.lower())
        for keyword in node.keywords:
            if not isinstance(keyword.arg, str):
                continue
            if keyword.arg.lower() != "trim":
                continue
            if not (
                isinstance(keyword.value, ast.Constant)
                and keyword.value.value in {False, None}
            ):
                identifiers.add("trim")
    return identifiers


def _call_terminal(node: ast.Call) -> str:
    if isinstance(node.func, ast.Name):
        return node.func.id.lower()
    if isinstance(node.func, ast.Attribute):
        return node.func.attr.lower()
    return ""


def _enabled_trim_keyword(node: ast.Call) -> bool:
    for keyword in node.keywords:
        if not isinstance(keyword.arg, str) or keyword.arg.lower() != "trim":
            continue
        return not (
            isinstance(keyword.value, ast.Constant)
            and keyword.value.value in {False, None}
        )
    return False


def _public_entrypoint_names(requirements: Mapping[str, str]) -> set[str]:
    """Extract explicitly named public calls from the feature's own prose."""

    names: set[str] = set()
    for requirement in requirements.values():
        names.update(
            match.group(1).lower()
            for match in re.finditer(
                r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(\s*\)",
                str(requirement or ""),
            )
        )
    return names


def _source_stage_analysis(
    content: str,
    stage: str,
    entrypoints: set[str],
) -> tuple[set[str], set[str]]:
    """Return reachable calls and concrete witnesses for one public stage.

    A trim-named unresolved/library call is the strongest source evidence the
    host can obtain without prescribing an implementation.  If the same name
    resolves to a helper defined in the submitted public source, however, its
    body is available and must contain a concrete strip/trim operation.  This
    prevents an empty `_trim_*` wrapper made only of seek/tell/read operations
    from satisfying the stage by its name alone.
    """

    try:
        tree = ast.parse(str(content or ""))
    except (SyntaxError, TypeError, ValueError):
        return set(), set()
    definitions: dict[str, list[ast.AST]] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            definitions.setdefault(node.name.lower(), []).append(node)
    roots: list[ast.AST] = [
        node
        for name in sorted(entrypoints)
        for node in definitions.get(name, [])
    ]
    if not roots:
        roots = [tree]
    queue = list(roots)
    visited: set[int] = set()
    observed: set[str] = set()
    witnesses: set[str] = set()
    while queue:
        root = queue.pop(0)
        if id(root) in visited:
            continue
        visited.add(id(root))
        for node in ast.walk(root):
            if not isinstance(node, ast.Call):
                continue
            terminal = _call_terminal(node)
            if terminal:
                observed.add(terminal)
            local_targets = definitions.get(terminal, [])
            for target in local_targets:
                if id(target) not in visited:
                    queue.append(target)
            if stage != "trim":  # pragma: no cover - future stages define tokens
                continue
            if terminal in {"strip", "lstrip", "rstrip"}:
                witnesses.add(terminal)
            elif _enabled_trim_keyword(node):
                witnesses.add("trim=")
            elif "trim" in terminal and not local_targets:
                witnesses.add(terminal)
    return observed, witnesses


def _missing_ordered_source_stages(
    requirements: Mapping[str, str],
    current_files: Mapping[str, str],
) -> list[dict[str, Any]]:
    """Fail closed when a reviewer credits a publicly named but absent stage.

    The check is intentionally source-level and weakly semantic: it requires a
    recognizable implementation witness, but never prescribes the transform's
    boundary character, direction, expected bytes, or hidden evaluator value.
    A peer probe that observes only the downstream callback cannot substitute
    for an earlier stage that does not exist in the current source.
    """

    entrypoints = _public_entrypoint_names(requirements)
    gaps: list[dict[str, Any]] = []
    for identity, requirement in requirements.items():
        for stage in _required_ordered_source_stages(requirement):
            analyses = {
                str(path): _source_stage_analysis(content, stage, entrypoints)
                for path, content in current_files.items()
            }
            witnesses = {
                path: sorted(found)
                for path, (_, found) in analyses.items()
            }
            witnesses = {path: names for path, names in witnesses.items() if names}
            if not witnesses:
                gaps.append(
                    {
                        "requirement_id": str(identity),
                        "public_basis": str(requirement),
                        "stage": stage,
                        "paths": sorted(analyses),
                        "observed_calls_by_path": {
                            path: sorted(observed)[:32]
                            for path, (observed, _) in analyses.items()
                        },
                    }
                )
    return gaps


def _ordered_source_gap_evidence(gap: Mapping[str, Any]) -> str:
    """Render public-source call evidence without inventing stage semantics."""

    observed = gap.get("observed_calls_by_path") or {}
    rows = []
    if isinstance(observed, Mapping):
        for path, names in observed.items():
            rendered = ", ".join(str(name) for name in (names or [])[:32])
            rows.append(f"{path}: {rendered or '<no calls>'}")
    inventory = "; ".join(rows)[:1200] or "<no public-source calls observed>"
    return (
        f"Publicly ordered {gap['stage']} stage has no executable source witness. "
        f"Host AST observed these invoked operations in the cited public source: "
        f"{inventory}. None witnesses the publicly named {gap['stage']} stage; "
        "cursor movement, byte reads or rewrapping, length measurement, and the "
        "downstream observer do not by themselves prove a content transformation."
    )


def _probe_failure_site(probe: Mapping[str, Any], result: Mapping[str, Any]) -> dict[str, Any]:
    """Bind validity to reached operations, not volatile paths or actual values."""
    output = str((result.get("candidate") or {}).get("output") or "")
    code = str(probe.get("code") or "").splitlines()
    frames = []
    output_lines = output.splitlines()
    for index, line in enumerate(output_lines):
        match = re.match(r'^\s*File "([^\"]+)", line (\d+)(?:, in (.*))?$', line)
        if match is None:
            continue
        path, number, function = match.group(1).replace("\\", "/"), int(match.group(2)), match.group(3) or ""
        # These are the fixed probe runner's scratch layouts. Keep every
        # repository-relative path and line; strip only the volatile root.
        # The runner compiles its real check.py path so inspect.getsource works.
        # Only that top-level script is probe code, never check/<repo-file>.
        path = re.sub(r'^.*?/cooper-check-[^/]+/check\.py$', '<public-behavior-probe>', path)
        path = re.sub(r'^.*?/cooper-check-[^/]+/check/', '<candidate>/', path)
        path = re.sub(r'^.*?/cooper-check-[^/]+/seed/', '<baseline>/', path)
        path = re.sub(r'^.*?/cooper-check-[^/]+/', '<probe-sandbox>/', path)
        path = re.sub(r'^.*?/candidate/', '<candidate>/', path)
        path = re.sub(r'^.*?/baseline/', '<baseline>/', path)
        source = ""
        if path == "<public-behavior-probe>" and 0 < number <= len(code):
            source = code[number - 1].strip()
        elif index + 1 < len(output_lines):
            following = output_lines[index + 1]
            if following.startswith("    ") and not following.lstrip().startswith("File "):
                source = following.strip()
        frames.append({"path": path, "line": number, "function": function, "source": source})
    lines = [frame["line"] for frame in frames if frame["path"] == "<public-behavior-probe>"]
    exception = next((match.group(1) for line in reversed(output.splitlines())
                      if (match := re.match(r"^([\w.]+(?:Error|Exception|Exit|Interrupt))(?::|$)", line))), "")
    terminal = frames[-1] if frames else {}
    phase = ("probe_assertion" if terminal.get("path") == "<string>" and terminal.get("function") == "checked"
             else "probe_operation" if terminal.get("path") == "<public-behavior-probe>"
             else "candidate_operation" if str(terminal.get("path") or "").startswith("<candidate>/")
             else "external_operation")
    site = {"probe_lines": lines, "exception_type": exception,
            "frames": frames, "operation_phase": phase,
            "operations": [code[line - 1].strip() if 0 < line <= len(code) else ""
                           for line in lines]}
    if not lines or not exception:
        # Unknown trace formats are not normalized speculatively. A different
        # observation must be re-adjudicated rather than collide with a known
        # assertion. This also handles executor-level failures conservatively.
        site["unlocated_observation_digest"] = hashlib.sha256(output.encode()).hexdigest()
    return site


def _probe_validity_material(world: Any, *, reviewer_id: str, feature_id: str,
                             description: str, obligations: list[str], probe: Mapping[str, Any],
                             result: Mapping[str, Any]) -> dict[str, Any] | None:
    from .visibility import is_strict_coop, visible_feature_brief

    actual_feature = str(probe.get("probe_id") or "").split(":", 1)[0]
    contract = visible_review_contract(world, reviewer_id, actual_feature) if is_strict_coop(world) else None
    if is_strict_coop(world):
        if contract is None:
            return None
        brief = contract["description"]
        requirements = contract["requirements"]
        visibility = _contract_binding(contract)
    else:
        brief = description if actual_feature == feature_id else visible_feature_brief(world, reviewer_id, actual_feature)
        if not brief:
            return None
        requirements = ({requirement_id(item): item for item in obligations}
                        if actual_feature == feature_id else {})
        if actual_feature != feature_id:
            from .public_contract import indexed_obligations
            requirements = {requirement_id(item): item for item in indexed_obligations(brief)}
        visibility = {"source": "legacy_public_task", "reviewer_id": reviewer_id}
    mapped = {identity: requirements[identity] for identity in probe.get("requirement_ids", [])
              if identity in requirements}
    failure_site = _probe_failure_site(probe, result)
    material = {"schema_version": PROBE_VALIDITY_SCHEMA, "feature_id": actual_feature,
                "reviewer_id": str(reviewer_id), "probe_id": str(probe.get("probe_id") or ""),
                "probe_definition": dict(probe), "public_feature_request": brief,
                "mapped_public_requirements": mapped, "visibility": visibility,
                "failure_site": failure_site,
                "public_prerequisite_context": public_prerequisite_context(world, probe, failure_site)}
    material["observer_evidence"] = observer_evidence(probe, material["public_prerequisite_context"], failure_site)
    host_authorization = authorized_atomic_failure(probe, mapped, failure_site)
    if host_authorization is not None:
        material["host_authorized_atomic_failure"] = host_authorization
    material["brief_identity"] = hashlib.sha256(json.dumps(
        {"brief": brief, "visibility": visibility, "reviewer_id": reviewer_id}, sort_keys=True).encode()).hexdigest()
    material["identity"] = hashlib.sha256(json.dumps(material, sort_keys=True).encode()).hexdigest()
    return material


def _grounded_fixture_basis(material: Mapping[str, Any], value: Any) -> dict[str, str] | None:
    """Bind a peer's cross-boundary fixture finding to captured public evidence.

    A candidate frame locates an exception, not necessarily its cause. Still,
    an unsupported fixture must be positively identified by the independent
    peer, not inferred by the host from a product exception or a keyword.
    """
    if not isinstance(value, Mapping):
        return None
    fields = ("probe_quote", "prerequisite_path", "prerequisite_quote")
    if any(not isinstance(value.get(k), str) or not value[k].strip()
           or len(value[k]) > 1600 for k in fields):
        return None
    definition = material.get("probe_definition") or {}
    if not any(value["probe_quote"] in str(definition.get(k) or "")
               for k in ("code", "baseline_setup")):
        return None
    context = material.get("public_prerequisite_context") or {}
    if context.get("origin") != "untouched_public_starter_source":
        return None
    for file in context.get("files") or []:
        if file.get("path") != value["prerequisite_path"]:
            continue
        if any(value["prerequisite_quote"] in str(s.get("text") or "")
               for s in file.get("snippets") or []):
            return {k: value[k] for k in fields}
    return None


def _cached_probe_validity(world: Any, material: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if material is None:
        return None
    receipt = (getattr(world, "_cooperbench_probe_validity_receipts", {}) or {}).get(material["identity"])
    if (not isinstance(receipt, Mapping)
            or receipt.get("origin") not in {"focused_actual_peer_adjudication", "host_public_atomic_contract"}
            or receipt.get("verdict") not in ("valid", "defective")
            or not str(receipt.get("reason") or "").strip()
            or any(receipt.get(key) != material[key] for key in (
                "schema_version", "identity", "feature_id", "reviewer_id", "probe_id", "brief_identity", "failure_site"))):
        return None
    if receipt.get("origin") == "host_public_atomic_contract":
        authorization = material.get("host_authorized_atomic_failure")
        if (not isinstance(authorization, Mapping)
                or receipt.get("verdict") != "valid"
                or receipt.get("contract_relation") != "explicitly_authorized"
                or receipt.get("public_basis") != authorization.get("public_basis")
                or receipt.get("host_authorized_atomic_failure") != authorization):
            return None
    if receipt["verdict"] == "valid" and (
        not isinstance(receipt.get("public_basis"), str) or not receipt["public_basis"].strip()
        or receipt["public_basis"] not in material["public_feature_request"]
    ):
        return None
    if receipt["verdict"] == "valid" and _syntax_literals(material):
        if _grounded_syntax_basis(material, receipt.get("syntax_basis")) is None:
            return None
    if receipt["verdict"] == "valid" and receipt.get("origin") != "host_public_atomic_contract":
        if grounded_observer_basis(material, receipt.get("observer_basis")) is None:
            return None
    relation = receipt.get("contract_relation")
    if (relation == "scaffolding_failure"
            and material["failure_site"]["operation_phase"] == "candidate_operation"
            and _grounded_fixture_basis(material, receipt.get("fixture_basis")) is None):
        return None
    if relation is not None and (
        not isinstance(relation, str)
        or _CONTRACT_RELATION_TO_VERDICT.get(relation) != receipt["verdict"]
    ):
        return None
    return {**receipt, "cached": True}


def _syntax_literals(material: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Locate literal parser inputs when execution stopped at a syntax error.

    Decoding is evidence, not a host verdict about the nested language. The
    actual peer must establish why this exact input is supported (or why its
    rejection was intended) before a broad feature quote can authorize blame.
    """
    site = material.get("failure_site") or {}
    if not str(site.get("exception_type") or "").endswith("SyntaxError"):
        return []
    lines = set(site.get("probe_lines") or [])
    context = material.get("public_prerequisite_context") or {}
    return [row for row in context.get("literal_call_arguments") or []
            if row.get("line") in lines and row.get("complete") is True]


def _grounded_syntax_basis(material: Mapping[str, Any], value: Any) -> list[dict[str, Any]] | None:
    literals = _syntax_literals(material)
    if not literals:
        return []
    if not isinstance(value, list) or len(value) != len(literals):
        return None
    captured = []
    for literal in literals:
        matches = [row for row in value if isinstance(row, Mapping)
                   and all(row.get(k) == literal.get(k)
                           for k in ("line", "callee", "argument", "decoded_value"))]
        if len(matches) != 1:
            return None
        row = matches[0]
        if (row.get("input_kind") not in {"syntactically_valid", "intentional_invalid_input"}
                or not isinstance(row.get("reason"), str) or not row["reason"].strip()
                or len(row["reason"]) > 1600):
            return None
        captured.append({k: row[k] for k in (
            "line", "callee", "argument", "decoded_value", "input_kind", "reason")})
    return captured


def _adjudicate_probe_defects(
    world: Any,
    *,
    reviewer_id: str,
    feature_id: str,
    description: str,
    obligations: list[str],
    behavior: Mapping[str, Any],
    defects: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str, dict[str, Any]]:
    """Require a focused peer verdict before routing or discarding a failed check.

    The broad source review has a large context and can occasionally label an
    unusual-but-explicit API contract as impossible.  A single such label must
    not erase the pre-candidate executable check and turn a product failure
    into a test rewrite.  This bounded adjudication sees only the public
    contract, the challenged probes, and their observations.  Missing or
    invalid adjudication fails closed by retaining the original defect claim.
    """
    probes = {
        str(item.get("probe_id") or ""): item
        for item in (behavior.get("probes") or [])
        if isinstance(item, Mapping)
    }
    results = {
        str(item.get("probe_id") or ""): item
        for item in (behavior.get("results") or [])
        if isinstance(item, Mapping)
    }
    challenged = []
    materials, receipts, unresolved_verdicts = {}, {}, []
    basis_rejections = []
    for defect in defects:
        probe_id = str(defect.get("probe_id") or "")
        probe = probes.get(probe_id) or {}
        result = results.get(probe_id) or {}
        material = _probe_validity_material(world, reviewer_id=reviewer_id, feature_id=feature_id,
                                            description=description, obligations=obligations,
                                            probe=probe, result=result)
        if material is None:
            continue
        # Both focused sends must adjudicate the same captured evidence even
        # if a provider callback (or another worker) mutates a live container.
        material = json.loads(json.dumps(material))
        materials[probe_id] = material
        cached = _cached_probe_validity(world, material)
        host_authorization = material.get("host_authorized_atomic_failure")
        if isinstance(host_authorization, Mapping):
            if cached is not None and cached.get("origin") == "host_public_atomic_contract":
                receipts[probe_id] = cached
                continue
            receipt = {key: material[key] for key in (
                "schema_version", "identity", "feature_id", "reviewer_id", "probe_id",
                "brief_identity", "failure_site")}
            receipt.update(
                origin="host_public_atomic_contract",
                verdict="valid",
                public_basis=host_authorization["public_basis"],
                contract_relation="explicitly_authorized",
                reason=(
                    "The reached assertion is one of the exact assertions mechanically validated "
                    "as observing the host's conservative public atomic contract "
                    f"{host_authorization['observation']}."
                ),
                host_authorized_atomic_failure=dict(host_authorization),
                cached=False,
            )
            world.__dict__.setdefault("_cooperbench_probe_validity_receipts", {})[
                receipt["identity"]
            ] = dict(receipt)
            receipts[probe_id] = receipt
            continue
        # A fresh explicit defect claim can reveal a problem missed in an
        # earlier verdict, so do not dismiss it solely because of a cache hit.
        if cached is not None and (cached["verdict"] == "defective"
                                   or defect.get("origin") in {"unclassified_executed_failure", "first_executed_failure_validity"}):
            receipts[probe_id] = cached
            continue
        challenged.append({
            "probe_id": probe_id,
            "feature_id": material["feature_id"],
            "public_feature_request": material["public_feature_request"],
            "failure_site": material["failure_site"],
            "public_prerequisite_context": material["public_prerequisite_context"],
            "observer_evidence": material["observer_evidence"],
            "claimed_defect": str(defect.get("issue") or ""),
            "mapped_public_requirements": material["mapped_public_requirements"],
            "probe": {
                key: material["probe_definition"].get(key)
                for key in ("requirement_ids", "paths", "compare_baseline", "kind",
                            "baseline_setup", "code")
            },
            "observed_result": json.loads(json.dumps(result)),
        })
    schema = {"verdicts": [{
        "probe_id": "exact challenged probe id",
        "contract_relation": (
            "explicitly_authorized | unauthorized_assertion | "
            "scaffolding_failure | indeterminate"
        ),
        "reason": "specific public-contract reason",
        "public_basis": (
            "exact quote from this probe's public_feature_request authorizing the failed operation; "
            "required only for explicitly_authorized"
        ),
        "fixture_basis": {
            "probe_quote": "exact nonempty quote identifying the probe-owned fixture or invocation",
            "prerequisite_path": "exact path from captured public_prerequisite_context.files",
            "prerequisite_quote": "exact nonempty source quote establishing the violated fixture protocol",
        },
        "syntax_basis": [{
            "line": "exact captured literal line number",
            "callee": "exact captured callee",
            "argument": "exact captured argument index or name",
            "decoded_value": "exact complete decoded_value, not Python/JSON source spelling",
            "input_kind": "syntactically_valid | intentional_invalid_input",
            "reason": "Explain delimiter/escape validity in the nested language, or the public rejection requirement.",
        }],
        "observer_basis": [{"line": "exact evidence line", "kind": "exact evidence kind",
                            "probe_quote": "exact captured probe quote",
                            "reason": "Explain why this observer and its preconditions are supported, not a generic feature quote."}],
    }]}
    system = (
        "You are the focused second-pass adjudicator for peer-authored public behavior checks. "
        "Use only the public feature request, mapped public requirements and captured untouched "
        "public_prerequisite_context. The source establishes calling conventions and fixture setup, "
        "not an oracle for new behavior. "
        + PREREQUISITE_GUIDANCE + " " + PUBLIC_CONTRACT_BOUNDARY_GUIDANCE + " A probe defect "
        "is valid only when the check itself asserts behavior absent from, or contradictory to, "
        "the public request, or when its own import/setup/inspection scaffolding raises before it "
        "can check the requested behavior. A candidate failure is not evidence that the probe is defective. "
        "Constructing or loading a requested product component is candidate behavior, even before the "
        "probe's first assertion. If failure_site.operation_phase is candidate_operation, the exception "
        "was raised inside candidate code: do not classify it as scaffolding_failure merely because "
        "the probe was initializing an object. However, a probe-owned fake, dependency response or "
        "unsupported fixture can cause an exception inside an unchanged adapter. Location is not cause. "
        "For that specific scaffolding_failure finding, fixture_basis is mandatory: quote the actual "
        "probe fixture and the violated untouched public prerequisite, and explain their causal link "
        "to this observed failure in reason. Quote only captured text, not candidate changes or "
        "invented requirements. If that evidence is absent, use indeterminate or independently establish "
        "another relation; do not guess fixture blame. Ordinary product initialization errors remain "
        "candidate failures. Check whether the invocation is supported by the public "
        "request. A required component's ordinary initialization need not be separately specified as "
        "'must not raise' to exercise that component; quote the relevant component or behavior requirement. "
        "An unsupported invocation can still be unauthorized_assertion, and unresolved support is "
        "indeterminate. Do not replace the failing product entry point just to avoid its exception. "
        "Independently verify the actual reached failure operation, even when the broad source review "
        "confidently calls it a product bug. Read failure_site and the traceback: do not attribute a later "
        "assertion to an earlier operation that already passed, or invent an observed actual value. "
        "Trace the failed assertion's expected value to an explicit public literal or a reproducible "
        "derivation from the public rule. For hashes and other derived constants, do not calculate or "
        "guess a value mentally: if the probe substitutes an unsupported hard-coded value for a public "
        "formula, require its author to compute the expectation from that formula in the probe. "
        "Likewise do not infer object identity (`is None`) or concrete representation from a request "
        "that only specifies value retrieval and callable/context-manager behavior. "
        "Unusual combinations on one public attribute are not impossible: Python descriptors, "
        "callable accessors, and proxy values can support property-like get/set behavior together "
        "with call or context-manager syntax. Therefore a check that directly exercises each "
        "explicitly required observable call form is not defective merely because a plain method "
        "or plain scalar property cannot implement all forms. Do not weaken, rename, or replace an "
        "explicit public entry point. Return exactly one verdict for every challenged probe. "
        "Use verdict=valid only after positively establishing that the actual failed operation and "
        "its expected value are authorized by the public request; public_basis must quote that "
        "authorization exactly from this probe's public_feature_request and reason must explain "
        "its relation to the observed failure. Not proving a defect does not prove validity. "
        "When the reached exception is a SyntaxError (including TemplateSyntaxError), inspect the "
        "literal_call_arguments at failure_site.probe_lines. An explicitly_authorized verdict must "
        "include syntax_basis for every complete captured literal on those lines, copying the exact "
        "decoded_value. Explain the actual delimiter and escaping rules, not merely that the feature "
        "supports special characters. A backslash consumed by Python does not escape the nested "
        "language. If that input is malformed before the intended assertion, classify its actual "
        "fixture failure; do not authorize product blame with a generic feature quote. Deliberately "
        "invalid inputs remain valid tests when the public contract requires their rejection: identify "
        "that rejection requirement and the actual expected exception instead of demanding acceptance. "
        "Classify contract_relation before any verdict: explicitly_authorized means the reached operation "
        "and expected result are directly required; unauthorized_assertion means the probe adds a concrete "
        "expectation or calling convention absent from or contradictory to the request; scaffolding_failure "
        "means the probe's own setup/import/inspection or grounded invalid dependency fixture prevents "
        "the intended behavior check; indeterminate "
        "means the evidence establishes none of those. In particular, if a probe assumes unstated ambient or "
        "global propagation between separate calls--such as setting state on one object and expecting a "
        "standalone function that was not passed that state to observe it--classify unauthorized_assertion, "
        "not indeterminate. The host derives valid from explicitly_authorized, defective from "
        "unauthorized_assertion or scaffolding_failure, and unknown only from indeterminate. Return JSON only."
    )
    user = json.dumps({
        "feature_id": feature_id,
        "public_feature_request": description,
        "challenged_probes": challenged,
        "review_binding": {
            "reviewer_id": reviewer_id,
            "candidate_hash": behavior.get("candidate_hash"),
            "baseline_hash": behavior.get("baseline_hash"),
            "plan_hash": behavior.get("plan_hash"),
            "probe_identities": {row["probe_id"]: materials[row["probe_id"]]["identity"]
                                 for row in challenged},
        },
    }, ensure_ascii=False)
    provider_error = ""
    answer = {}
    try:
        from environments.org_env.llm.client import reasoning_effort_override

        if challenged:
            with reasoning_effort_override(PROBE_DEFECT_ADJUDICATION_REASONING_EFFORT):
                answer = world.llm_client.generate_json(
                    system, user, schema, max_tokens=min(12000, max(2200,
                    sum(550 + 250 * len(item.get("observer_evidence") or []) for item in challenged))))
    except Exception as error:
        provider_error = f"probe_defect_adjudication_provider_failed:{type(error).__name__}"
    def consume_verdicts(response: Any, requested: list[dict[str, Any]], *, attempt: int) -> list[dict[str, Any]]:
        from environments.org_env.programbench.public_evidence import redact_public_text

        rows = response.get("verdicts") if isinstance(response, Mapping) else None
        correctable = []
        for item in requested:
            probe_id = item["probe_id"]
            matches = [row for row in rows or [] if isinstance(row, Mapping) and row.get("probe_id") == probe_id] if isinstance(rows, list) else []
            if (len(matches) != 1 or not isinstance(matches[0].get("reason"), str)
                    or not matches[0]["reason"].strip()):
                continue
            verdict = dict(matches[0])
            relation = verdict.get("contract_relation")
            structured_relation = (
                isinstance(relation, str)
                and relation in _CONTRACT_RELATION_TO_VERDICT
            )
            if structured_relation:
                # The structured relation is the peer's actual finding. Derive
                # control flow from it rather than trusting a contradictory
                # label such as "unknown, but this assertion is unauthorized."
                verdict["verdict"] = _CONTRACT_RELATION_TO_VERDICT[relation]
            elif relation is not None:
                continue
            elif (
                isinstance(verdict.get("verdict"), str)
                and verdict["verdict"] in _LEGACY_VERDICT_TO_CONTRACT_RELATION
            ):
                # Compatibility for existing serialized/test clients. New
                # prompts request contract_relation and do not request verdict.
                relation = _LEGACY_VERDICT_TO_CONTRACT_RELATION[verdict["verdict"]]
            else:
                continue
            if verdict["verdict"] == "unknown":
                unresolved = {"probe_id": probe_id, "verdict": "unknown",
                              "reason": verdict["reason"][:1600]}
                if structured_relation:
                    unresolved["contract_relation"] = relation
                unresolved_verdicts.append(unresolved)
                continue
            public_basis = verdict.get("public_basis")
            basis_error = ""
            if (relation == "scaffolding_failure"
                    and materials[probe_id]["failure_site"]["operation_phase"] == "candidate_operation"):
                # A library frame alone cannot rule out a malformed peer fake.
                # Require captured evidence, otherwise retain the existing
                # one-correction/fail-closed path for real product failures.
                if _grounded_fixture_basis(materials[probe_id], verdict.get("fixture_basis")) is None:
                    basis_error = "candidate_operation_misclassified_as_scaffolding"
            elif verdict["verdict"] == "valid":
                if public_basis is None:
                    basis_error = "positive_public_basis_missing"
                elif not isinstance(public_basis, str):
                    basis_error = "positive_public_basis_invalid_type"
                elif not public_basis.strip():
                    basis_error = "positive_public_basis_blank"
                elif public_basis not in materials[probe_id]["public_feature_request"]:
                    basis_error = "positive_public_basis_not_in_public_request"
                elif _grounded_syntax_basis(materials[probe_id], verdict.get("syntax_basis")) is None:
                    basis_error = "decoded_syntax_basis_missing_or_mismatched"
                elif grounded_observer_basis(materials[probe_id], verdict.get("observer_basis")) is None:
                    basis_error = "observer_precondition_missing_or_contradicted"
            if basis_error:
                # Keep bounded, redacted evidence of what was actually rejected;
                # a missing field and a near-exact quote are different failures.
                rejection = {
                    "probe_id": probe_id, "verdict": "unknown", "reason": basis_error,
                    "attempt": attempt, "identity": materials[probe_id]["identity"],
                    "brief_identity": materials[probe_id]["brief_identity"],
                    "rejected_basis": redact_public_text(public_basis, max_chars=1600)
                                      if isinstance(public_basis, str) else None,
                    "rejected_basis_type": type(public_basis).__name__,
                    "rejected_basis_chars": len(public_basis) if isinstance(public_basis, str) else 0,
                    "rejected_basis_truncated": isinstance(public_basis, str) and len(public_basis) > 1600,
                    "peer_reason": redact_public_text(verdict["reason"], max_chars=1600),
                }
                basis_rejections.append(rejection)
                if attempt == 1 and basis_error != "positive_public_basis_invalid_type":
                    correctable.append(rejection)
                else:
                    unresolved_verdicts.append(rejection)
                continue
            if public_basis is not None and not isinstance(public_basis, str):
                continue
            receipt = {key: materials[probe_id][key] for key in (
                "schema_version", "identity", "feature_id", "reviewer_id", "probe_id", "brief_identity", "failure_site")}
            receipt.update(origin="focused_actual_peer_adjudication",
                           verdict=verdict["verdict"], public_basis=public_basis or "",
                           contract_relation=relation,
                           reason=verdict["reason"][:1600], cached=False)
            if verdict["verdict"] == "valid" and _syntax_literals(materials[probe_id]):
                receipt["syntax_basis"] = _grounded_syntax_basis(
                    materials[probe_id], verdict.get("syntax_basis"))
            if verdict["verdict"] == "valid" and materials[probe_id]["observer_evidence"]:
                receipt["observer_basis"] = grounded_observer_basis(materials[probe_id], verdict.get("observer_basis"))
            if (relation == "scaffolding_failure"
                    and materials[probe_id]["failure_site"]["operation_phase"] == "candidate_operation"):
                receipt["fixture_basis"] = _grounded_fixture_basis(materials[probe_id], verdict.get("fixture_basis"))
            world.__dict__.setdefault("_cooperbench_probe_validity_receipts", {})[receipt["identity"]] = dict(receipt)
            receipts[probe_id] = receipt
        return correctable

    corrections = consume_verdicts(answer, challenged, attempt=1)
    correction_attempted = bool(corrections) and not provider_error
    if correction_attempted:
        # This is a single focused verdict repair, not another source review or
        # an approval fallback. Already resolved/unknown probes are not resent.
        correction_payload = json.loads(user)
        correction_ids = {row["probe_id"] for row in corrections}
        correction_payload["challenged_probes"] = [
            row for row in correction_payload["challenged_probes"] if row["probe_id"] in correction_ids]
        correction_payload["correction_feedback"] = {
            "reason": "probe_validity_verdict_rejected",
            "remaining_corrections": 0,
            "rejected_verdicts": corrections,
        }
        correction_system = system + (
            " For observer_precondition_missing_or_contradicted, address every observer_evidence "
            "item with exact line/kind/probe_quote and a causal justification. A mechanical lexical "
            "contradiction cannot be authorized by prose. Reclassify the malformed fixture through "
            "the actual peer verdict; never prescribe accepting invalid input. "
            " Your previous verdicts were not accepted for the listed quote or failure-site errors. "
            "This is the only correction attempt for this same captured review evidence. "
            "For decoded_syntax_basis_missing_or_mismatched, read the captured decoded literal "
            "at the reached syntax failure. Supply its exact runtime value and a nested-language "
            "syntax explanation only if that actual input is supported or intentionally invalid "
            "under the public rejection requirement. Otherwise change the contract relation; "
            "do not repeat a broad special-character feature quote as syntax evidence. "
            "For candidate_operation_misclassified_as_scaffolding, the traceback ends inside candidate "
            "code. This does not by itself establish which side caused the error. If the probe-owned "
            "fixture violates an existing public adapter protocol, provide fixture_basis with the "
            "exact probe quote, captured prerequisite path and source quote, and explain the causal "
            "link. Without those grounded quotes, repeating scaffolding_failure is not acceptable. "
            "Reassess whether the invocation is publicly supported; "
            "do not turn this location check into automatic product-failure authorization. "
            "Copy an exact contiguous quote, preserving punctuation, backticks, Unicode, and whitespace, "
            "from the challenged probe's public_feature_request if it truly authorizes the reached failure. "
            "Do not paraphrase, concatenate separate passages, or add enclosing quotation marks. "
            "You may instead change contract_relation to unauthorized_assertion, scaffolding_failure, "
            "or indeterminate; supplying a quote alone does "
            "not establish semantic validity. Do not change the probe, failure, source, or contract. "
            "Return only the challenged probe verdicts, using the same schema."
        )
        try:
            with reasoning_effort_override(PROBE_DEFECT_ADJUDICATION_REASONING_EFFORT):
                corrected = world.llm_client.generate_json(
                    correction_system, json.dumps(correction_payload, ensure_ascii=False), schema,
                    max_tokens=min(12000, max(2200, sum(550 + 250 * len(item.get("observer_evidence") or [])
                               for item in correction_payload["challenged_probes"]))))
        except Exception as error:
            provider_error = f"probe_defect_basis_correction_provider_failed:{type(error).__name__}"
        else:
            consume_verdicts(corrected, correction_payload["challenged_probes"], attempt=2)
        unresolved_ids = {row["probe_id"] for row in unresolved_verdicts}
        for rejected in corrections:
            if rejected["probe_id"] not in receipts and rejected["probe_id"] not in unresolved_ids:
                unresolved_verdicts.append({**rejected, "correction_error": provider_error
                                            or "probe_defect_basis_correction_invalid_or_missing_verdict"})
    accepted, rejected = [], []
    for defect in defects:
        probe_id = str(defect.get("probe_id") or "")
        receipt = receipts.get(probe_id)
        if receipt is None:
            accepted.append({**defect, "adjudication_status": "unresolved"})
            continue
        enriched = {**defect, "adjudication_reason": receipt["reason"][:600],
                    "adjudication_status": "confirmed", "validity_identity": receipt["identity"]}
        (accepted if receipt["verdict"] == "defective" else rejected).append(enriched)
    missing = sorted({str(d.get("probe_id") or "") for d in defects} - set(receipts))
    error = provider_error or ("probe_defect_adjudication_incomplete:" + ",".join(missing) if missing else "")
    audit = {"schema_version": PROBE_VALIDITY_SCHEMA, "receipts": list(receipts.values()),
             "unresolved_probe_ids": missing, "unresolved_verdicts": unresolved_verdicts,
             "basis_rejections": basis_rejections, "basis_correction_attempted": correction_attempted}
    return accepted, rejected, error, audit


def _status(value: Any) -> str:
    return str(getattr(value, "value", value) or "").casefold()


def _quarantine_defective_probe_review(
    answer: Mapping[str, Any],
    behavior: Mapping[str, Any],
    defect_ids: set[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Withdraw conclusions that may depend on a check queued for correction.

    Removing only its execution failure is insufficient: the broad review can
    repeat that same invalid assertion in a requirement row, finding, or repair
    hint. Unattributed prose is not safe to send to the owner in a mixed verdict.
    Preserve it as audit evidence, and retain independently attributed findings
    plus the separately adjudicated executable failures. No withdrawn check is
    turned into a pass; the reviewer must check the requirement again.
    """
    probes = {
        str(row.get("probe_id") or ""): set(row.get("requirement_ids") or [])
        for row in behavior.get("probes", []) if isinstance(row, Mapping)
    }
    affected = set().union(*(probes.get(identity, set()) for identity in defect_ids))
    known_requirements = set().union(*probes.values()) if probes else set()
    sanitized = dict(answer)
    withdrawn: dict[str, Any] = {"probe_ids": sorted(defect_ids),
                                 "requirement_ids": sorted(affected)}
    for field in ("requirement_checks", "compatibility_checks"):
        rows, quarantined = [], []
        for row in answer.get(field) or []:
            if not isinstance(row, Mapping):
                continue
            if row.get("requirement_id") in affected:
                quarantined.append(dict(row))
                rows.append({"requirement_id": row.get("requirement_id"),
                             "status": "uncertain",
                             "evidence": "Recheck after independent validation or correction of the disputed public probe."})
            else:
                rows.append(dict(row))
        sanitized[field], withdrawn[field] = rows, quarantined
    sanitized["blocking_issues"] = []
    withdrawn["blocking_issues"] = list(answer.get("blocking_issues") or [])
    findings, quarantined_findings = [], []
    for row in answer.get("blocking_findings") or []:
        if not isinstance(row, Mapping):
            continue
        raw_reqs, raw_probes = row.get("requirement_ids", []), row.get("probe_ids", [])
        valid_shape = (isinstance(raw_reqs, list) and isinstance(raw_probes, list)
                       and all(isinstance(item, str) for item in raw_reqs + raw_probes))
        reqs = set(raw_reqs) if valid_shape else set()
        refs = set(raw_probes) if valid_shape else set()
        independent = bool(
            valid_shape and (reqs or refs) and reqs <= known_requirements
            and refs <= set(probes) and not reqs.intersection(affected)
            and not refs.intersection(defect_ids)
        )
        (findings if independent else quarantined_findings).append(dict(row))
    sanitized["blocking_findings"] = findings
    withdrawn["blocking_findings"] = quarantined_findings
    return sanitized, withdrawn


def pull_request_revision(pull_request: Any) -> dict[str, list[str]]:
    """Identify the committed revision actually covered by a semantic verdict."""
    return {key: [str(item) for item in (getattr(pull_request, key, []) or [])]
            for key in ("commit_ids", "patch_ids")}


def current_semantic_approval(world: Any, feature_id: str, pull_request: Any) -> bool:
    """A raw repo approval cannot replace a current, actual-peer semantic receipt."""
    owners = (getattr(world, "_cooperbench_sdl_state", {}) or {}).get("feature_owners") or {}
    author = str(getattr(pull_request, "author_id", "") or "")
    peers = {str(item) for item in owners.values()} - {author}
    from .visibility import is_strict_coop

    if is_strict_coop(world) and not any(visible_review_contract(world, peer, feature_id) for peer in peers):
        return False
    receipt = (getattr(world, "_cooperbench_semantic_reviews", {}) or {}).get(feature_id) or {}
    reviewer = str(receipt.get("reviewer_id") or "")
    revision = pull_request_revision(pull_request)
    return bool(
        receipt.get("approved") is True
        and receipt.get("pr_id") == str(getattr(pull_request, "pr_id", "") or "")
        and reviewer in peers
        and reviewer in (getattr(pull_request, "approved_by", []) or [])
        and revision["commit_ids"]
        and receipt.get("reviewed_pr_revision") == revision
        and semantic_review_visibility_matches(world, reviewer, feature_id, receipt)
        and semantic_review_source_snapshot_matches(world, receipt, pull_request)
    )


def semantic_review_visibility_matches(world: Any, reviewer_id: str, feature_id: str,
                                       receipt: Mapping[str, Any]) -> bool:
    """Reject omniscient, stale-read, or different-reviewer receipts in strict coop."""
    from .visibility import is_strict_coop

    if not is_strict_coop(world):
        return True
    contract = visible_review_contract(world, reviewer_id, feature_id)
    return bool(contract is not None and all(
        receipt.get(key) == contract[key]
        for key in ("brief_visibility_receipt", "review_brief_identity")
    ))


def semantic_review_repair_evidence_matches(world: Any, reviewer_id: str, feature_id: str,
                                           receipt: Mapping[str, Any]) -> bool:
    """Do not resurrect legacy or differently bound executable repair advice.

    Source-only reviews retain their existing provenance gate. A review with
    failing executable checks must additionally carry focused verdicts that
    still match the host cache and today's visible public contract. In mixed
    reviews, withdrawn checks may remain as audit evidence, never as the sole
    authorization for an owner repair.
    """
    if not semantic_review_visibility_matches(world, reviewer_id, feature_id, receipt):
        return False
    if not semantic_review_source_snapshot_matches(world, receipt):
        return False
    behavior = receipt.get("behavior_evidence") or {}
    if not isinstance(behavior, Mapping):
        return False
    failed = [row for row in behavior.get("results") or []
              if isinstance(row, Mapping) and (row.get("candidate") or {}).get("status") == "fail"]
    if not failed:
        return True
    audit = receipt.get("probe_validity")
    if not isinstance(audit, Mapping) or audit.get("schema_version") != PROBE_VALIDITY_SCHEMA:
        return False
    rows, unresolved = audit.get("receipts"), audit.get("unresolved_probe_ids")
    probes = behavior.get("probes")
    if (not isinstance(rows, list) or not all(isinstance(row, Mapping) for row in rows)
            or not isinstance(unresolved, list) or not all(isinstance(item, str) for item in unresolved)
            or not isinstance(probes, list) or not all(isinstance(row, Mapping) for row in probes)):
        return False
    verdicts = {str(row.get("probe_id") or ""): row for row in rows}
    definitions = {str(row.get("probe_id") or ""): row for row in probes}
    if len(verdicts) != len(rows) or len(definitions) != len(probes):
        return False
    quarantined = receipt.get("quarantined_review_evidence") or {}
    withdrawn = quarantined.get("probe_ids") if isinstance(quarantined, Mapping) else None
    withdrawn = set(withdrawn) if isinstance(withdrawn, list) and all(isinstance(item, str) for item in withdrawn) else set()
    from .visibility import is_strict_coop

    if is_strict_coop(world):
        contract = visible_review_contract(world, reviewer_id, feature_id)
        if contract is None:
            return False
        description, obligations = contract["description"], list(contract["requirements"].values())
    else:
        task = (getattr(world, "tasks", {}) or {}).get(f"task_oss_{feature_id}")
        description = str(getattr(task, "description", "") or "").strip()
        state = getattr(world, "_cooperbench_sdl_state", {}) or {}
        obligations = [str(item)[:500] for item in (state.get("acceptance_obligations") or {}).get(feature_id, [])
                       if str(item).strip()]
    valid_count = 0
    for result in failed:
        probe_id = str(result.get("probe_id") or "")
        if probe_id not in definitions:
            return False
        material = _probe_validity_material(
            world, reviewer_id=reviewer_id, feature_id=feature_id, description=description,
            obligations=obligations, probe=definitions[probe_id], result=result)
        trusted = _cached_probe_validity(world, material)
        recorded = verdicts.get(probe_id)
        matches = bool(trusted is not None and recorded is not None and all(
            recorded.get(key) == value for key, value in trusted.items() if key != "cached"))
        if matches and trusted["verdict"] == "valid":
            valid_count += 1
            continue
        if probe_id in withdrawn and (
            (matches and trusted["verdict"] == "defective")
            or (recorded is None and probe_id in unresolved and material is not None)
        ):
            continue
        return False
    return valid_count > 0


def implementation_fingerprint(path: str, content: str, *, source_sensitive: bool = False) -> str:
    """Compare Python implementation, or raw source for an explicit source clause.

    A changed fingerprint only permits another peer review; it never proves
    that the finding is resolved. The peer still judges the repaired code.
    """
    # Layout/docstring clauses are observable in source even when Python folds
    # them to the same AST. The prefix travels in the existing review receipt;
    # it permits re-review, never certifies that the public clause is satisfied.
    if source_sensitive:
        return "source:" + hashlib.sha256(content.encode()).hexdigest()
    normalized = content
    if path.endswith(".py"):
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    body = node.body
                    if (body and isinstance(body[0], ast.Expr)
                            and isinstance(body[0].value, ast.Constant)
                            and isinstance(body[0].value.value, str)):
                        del body[0]
            normalized = ast.dump(tree, include_attributes=False)
        except SyntaxError:
            # A broken repair remains subject to the existing public/CI gates.
            pass
    return hashlib.sha256(normalized.encode()).hexdigest()


def _source_sensitive_requirement(requirement: str) -> bool:
    """Identify explicit source presentation clauses, not reviewer preferences."""
    return re.search(
        r"\b(?:multi[ -]?line|single[ -]?line|line breaks?|indentation|"
        r"docstrings?|documentation|comments?|type hints?|type annotations?)\b|"
        r"\b(?:format|formatting|layout)\b.{0,50}\b(?:source|code)\b|"
        r"\b(?:source|code)\b.{0,50}\b(?:format|formatting|layout)\b",
        requirement, re.IGNORECASE,
    ) is not None


def _repair_paths(files: Mapping[str, str], findings: list[dict], issues: list[str]) -> list[str]:
    """Resolve public finding locations without trusting arbitrary model paths."""
    paths: set[str] = set()
    for finding in findings:
        path = str(finding.get("file_path") or "").replace("\\", "/")
        if path in files:
            paths.add(path)
    # Exact current-source findings are the reviewer's grounded localization.
    # Do not widen them by subsequently matching incidental API names in prose
    # summaries: a request_cache failure in cache.py can mention the separate
    # configure_cache entry point without requiring a no-op edit to __init__.py.
    if paths:
        return sorted(paths)
    for finding in findings:
        if not re.search(r"\btests?\b", str(finding.get("issue") or ""), re.I):
            continue
        hint = str(finding.get("repair_hint") or "")
        named_tests = {
            path
            for path in files
            if (path in hint or PurePosixPath(path).name in hint)
            and re.search(
                r"(^|[/_.-])tests?([/_.-]|$)|\.(?:test|spec)\.",
                path,
                re.I,
            )
        }
        if named_tests:
            return sorted(named_tests)
    location_texts = [*issues]
    for finding in findings:
        location_texts.extend(
            str(finding.get(key) or "")
            for key in ("issue", "evidence", "repair_hint")
        )
    for issue in location_texts:
        # Legacy reviews without valid findings fall back to an explicit file,
        # then a class/function named in the prose issue.
        named = {path for path in files if path in issue or PurePosixPath(path).name in issue}
        if not named:
            for path, content in files.items():
                try:
                    definitions = ast.walk(ast.parse(content))
                    if any(isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                           and re.search(r"\b" + re.escape(node.name) + r"\b", issue)
                           for node in definitions):
                        named.add(path)
                except SyntaxError:
                    continue
        paths.update(named)
    return sorted(paths)


def _replayed_known_behavior_failure(
    world: Any,
    previous: Mapping[str, Any],
    behavior: Mapping[str, Any],
    current_files: Mapping[str, str],
    *,
    feature_id: str,
    reviewer_id: str,
    description: str,
    obligations: list[str],
) -> dict[str, Any] | None:
    """Preserve a prior peer-validated rejection across a provider outage.

    A fresh, never-reviewed model probe can be wrong, so a failed execution by
    itself is not routed to the owner. Once the actual peer has successfully
    reviewed that exact persisted plan and rejected the candidate, however, a
    later repair that fails the same probes must not lose its executable trace
    merely because the secondary source-review request times out.
    """

    prior_behavior = previous.get("behavior_evidence") or {}
    if (
        not previous
        or previous.get("approved") is True
        or behavior.get("available") is not True
        or prior_behavior.get("available") is not True
        or not str(behavior.get("plan_hash") or "")
        or behavior.get("plan_hash") != prior_behavior.get("plan_hash")
    ):
        return None
    prior_failed = {
        str(row.get("probe_id") or "")
        for row in prior_behavior.get("results") or []
        if isinstance(row, Mapping)
        and (row.get("candidate") or {}).get("status") == "fail"
    }
    current_failed = [
        row
        for row in behavior.get("results") or []
        if isinstance(row, Mapping)
        and (row.get("candidate") or {}).get("status") == "fail"
    ]
    current_failed_ids = {str(row.get("probe_id") or "") for row in current_failed}
    if not current_failed_ids or not current_failed_ids <= prior_failed:
        return None
    # A historical broad source review is not a validity verdict. Replaying a
    # failure must still match a focused receipt for this exact reached site;
    # a repaired precondition may expose a later, unsupported assertion.
    probes = {str(row.get("probe_id") or ""): row for row in behavior.get("probes") or []}
    receipts = []
    for row in current_failed:
        material = _probe_validity_material(
            world, reviewer_id=reviewer_id, feature_id=feature_id,
            description=description, obligations=obligations,
            probe=probes.get(str(row.get("probe_id") or ""), {}), result=row)
        receipt = _cached_probe_validity(world, material)
        if receipt is None or receipt["verdict"] != "valid":
            return None
        receipts.append(receipt)
    repair_paths = sorted({
        str(path)
        for row in current_failed
        for path in (row.get("paths") or [])
        if str(path) in current_files
    })
    if not repair_paths:
        return None
    blocking: list[str] = []
    findings: list[dict[str, str]] = []
    for row in current_failed:
        probe_id = str(row.get("probe_id") or "")
        output = str((row.get("candidate") or {}).get("output") or "nonzero exit")[-1500:]
        issue = f"Replayed peer-validated public behavior failed [{probe_id}]"
        blocking.append(f"{issue}: {output}")
        for path in row.get("paths") or []:
            if path in current_files:
                findings.append({
                    "file_path": str(path),
                    "symbol": "",
                    "issue": issue,
                    "evidence": output,
                    "repair_hint": (
                        "Repair the public behavior shown in the repeated executed trace; "
                        "preserve the other passing checks."
                    ),
                })
    return {
        "available": True,
        "approved": False,
        "feature_id": feature_id,
        "resolution": "replayed_prior_peer_validated_behavior_failure",
        "source_review_available": False,
        "behavior_evidence": behavior,
        "probe_validity": {"schema_version": PROBE_VALIDITY_SCHEMA,
                           "receipts": receipts, "unresolved_probe_ids": []},
        "blocking_issues": blocking[:8],
        "blocking_findings": findings[:8],
        "repair_paths": repair_paths,
        "repair_path_groups": [[path] for path in repair_paths],
        "reviewed_code_fingerprints": {
            path: implementation_fingerprint(path, content)
            for path, content in current_files.items()
        },
        # These are explicitly previous observations and can only seed the
        # next review; they never make this provider-fallback receipt approve.
        "requirement_checks": list(previous.get("requirement_checks") or []),
        "compatibility_checks": list(previous.get("compatibility_checks") or []),
        "critical_invariants_present": bool(previous.get("critical_invariants_present", False)),
        "critical_invariants_passed": False,
        "evidence_gaps": {"source_review": "provider_unavailable_after_replay"},
        "input_strategy": _REVIEW_INPUT_STRATEGY,
        "reasoning_effort": SEMANTIC_REVIEW_REASONING_EFFORT,
    }


def pending_review_repair_paths(world: Any, agent_id: str, feature_id: str) -> list[str]:
    state = getattr(world, "_cooperbench_sdl_state", {}) or {}
    from .visibility import is_strict_coop

    strict = is_strict_coop(world)
    if strict:
        owners = state.get("feature_owners") or {}
        peers = set(owners.values()) - {owners.get(feature_id)}
        if not any(visible_review_contract(world, str(peer), feature_id) for peer in peers):
            return []
    review = (getattr(world, "_cooperbench_semantic_reviews", {}) or {}).get(feature_id) or {}
    if not review or review.get("approved"):
        return []
    if not semantic_review_repair_evidence_matches(
        world, str(review.get("reviewer_id") or ""), feature_id, review,
    ):
        return []
    required = (state.get("required_feature_paths") or {}).get(feature_id, [])
    paths = list(review.get("repair_paths") or required)
    fingerprints = review.get("reviewed_code_fingerprints") or {}
    artifacts = getattr(world, "product_artifacts", {}) or {}
    from . import source_views as sv
    actor_snapshot = None
    if sv.actor_desks_enabled(world):
        try:
            actor_snapshot = sv.actor_desk_snapshot(world, agent_id)
        except sv.SourceViewError:
            # Unavailable private provenance is not evidence that an owner
            # repaired anything. Keep the debt until a real view is available.
            return paths
    authored_paths: set[str] = set()
    for patch in (getattr(world, "patches", {}) or {}).values():
        if (str(getattr(patch, "actor_id", "")) != str(agent_id)
                or (actor_snapshot is not None and str(getattr(patch, "patch_id", "")) not in actor_snapshot.patch_ids)
                or int(getattr(patch, "tick", 0) or 0) < int(review.get("tick") or 0)
                or _status(getattr(patch, "validation_status", "")) not in {"accepted", "applied", "merged"}
                or feature_id not in (getattr(patch, "related_issue_ids", []) or [])):
            continue
        artifact = artifacts.get(getattr(patch, "target_object_id", ""))
        if artifact is not None:
            authored_paths.add(str(getattr(artifact, "linked_file_path", "")).replace("\\", "/"))
    by_path = {str(getattr(a, "linked_file_path", "")).replace("\\", "/"): a for a in artifacts.values()}
    pending = []
    actor_files = actor_snapshot.files if actor_snapshot is not None else None
    for path in paths:
        if actor_files is not None:
            content = actor_files.get(path, "")
        else:
            artifact = by_path.get(path)
            content = getattr(artifact, "content", None)
            if content is None:
                content = getattr(artifact, "mainline_content", "")
        if (path not in authored_paths or path not in fingerprints
                or implementation_fingerprint(
                    path, str(content or ""),
                    source_sensitive=str(fingerprints.get(path, "")).startswith("source:"),
                ) == fingerprints[path]):
            pending.append(path)
    groups = review.get("repair_path_groups")
    if not isinstance(groups, list) or not groups or any(
        not isinstance(group, list) or not group
        or any(not isinstance(path, str) or path not in paths for path in group)
        for group in groups
    ):
        groups = [[path] for path in paths]
    else:
        covered = {path for group in groups for path in group}
        groups = groups + [[path] for path in paths if path not in covered]
    # A quoted configuration entry point may only be context for the actual
    # implementation defect. One owner-authored implementation change inside
    # each source group permits re-review; it never constitutes an approval.
    unchanged = set(pending)
    pending_groups = [group for group in groups if set(group) <= unchanged]
    return [path for path in paths if any(path in group for group in pending_groups)]


def _normalized_requirement(value: Any) -> str:
    return " ".join(str(value or "").casefold().split()).strip(" .:;-*")


def requirement_id(requirement: str) -> str:
    return "req_" + hashlib.sha256(_normalized_requirement(requirement).encode()).hexdigest()[:12]


def passed_public_requirements(state: Mapping[str, Any], feature_id: str, review: Mapping[str, Any]) -> list[dict[str, str]]:
    """Carry prior pass observations without promoting peer prose to the contract.

    Requirement text is always re-resolved from this feature's public ledger.
    A prior pass is a regression target, not proof that the new head passes.
    """
    checks = [check for check in (review.get("requirement_checks") or []) if isinstance(check, Mapping)]
    rows = []
    for obligation in (state.get("acceptance_obligations") or {}).get(feature_id, []):
        identity = requirement_id(str(obligation))
        matching = [check for check in checks if check.get("requirement_id") == identity]
        if not matching or any(_status(check.get("status")) != "pass" for check in matching):
            continue
        evidence = str(matching[0].get("evidence") or "").strip()
        if evidence:
            rows.append({"requirement_id": identity, "requirement": str(obligation)[:500],
                         "previous_evidence": evidence[:200]})
    return rows[:32]


def passed_public_interactions(review: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return exact executed checks that passed on the reviewed current head.

    These rows are regression targets for the owner's *next* head, never proof
    that a later edit remains correct. Definitions come from the immutable
    behavior plan and are joined to results by unique probe id; ambiguous,
    incomplete, failed, or over-budget evidence is omitted rather than guessed.
    """

    behavior = review.get("behavior_evidence")
    if not isinstance(behavior, Mapping):
        return []
    raw_probes, raw_results = behavior.get("probes"), behavior.get("results")
    if not isinstance(raw_probes, list) or not isinstance(raw_results, list):
        return []
    probes: dict[str, Mapping[str, Any]] = {}
    for probe in raw_probes:
        if not isinstance(probe, Mapping):
            return []
        probe_id = str(probe.get("probe_id") or "")
        if not probe_id or probe_id in probes:
            return []
        probes[probe_id] = probe
    results: dict[str, Mapping[str, Any]] = {}
    for result in raw_results:
        if not isinstance(result, Mapping):
            return []
        probe_id = str(result.get("probe_id") or "")
        if not probe_id or probe_id in results or probe_id not in probes:
            return []
        results[probe_id] = result
    rows: list[dict[str, Any]] = []
    total_chars = 0
    for probe_id, probe in probes.items():
        result = results.get(probe_id)
        candidate = result.get("candidate") if isinstance(result, Mapping) else None
        if not isinstance(candidate, Mapping) or _status(candidate.get("status")) != "pass":
            continue
        requirement_ids = probe.get("requirement_ids") or []
        paths = probe.get("paths") or []
        if (not isinstance(requirement_ids, list)
                or not all(isinstance(item, str) and item for item in requirement_ids)
                or not isinstance(paths, list)
                or not all(isinstance(item, str) and item for item in paths)):
            continue
        if (result.get("requirement_ids") is not None
                and list(result.get("requirement_ids") or []) != list(requirement_ids)):
            continue
        if result.get("paths") is not None and list(result.get("paths") or []) != list(paths):
            continue
        check_source = _compact_probe_code(probe.get("code"))
        if not check_source or len(check_source) > 1_200:
            continue
        row = {
            "probe_id": probe_id,
            "requirement_ids": list(requirement_ids),
            "paths": list(paths),
            "check_source": check_source,
        }
        row_chars = len(json.dumps(row, sort_keys=True, ensure_ascii=False))
        if len(rows) >= 16 or total_chars + row_chars > 16_000:
            break
        total_chars += row_chars
        rows.append(row)
    return rows


def validated_failed_public_interactions(review: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return exact public checks whose reached failure was contract-valid.

    A raw ``AssertionError`` is not an implementation instruction: generated
    probes are fallible and may assert behavior the public request never asked
    for.  This projection therefore joins three independently recorded facts:
    the immutable public probe definition, its failed current-head result, and
    a focused validity receipt that binds the reached operation to an exact
    public-request quote.  Any ambiguous identity or malformed join is omitted
    rather than shown to the feature owner as repair authority.
    """

    behavior = review.get("behavior_evidence")
    validity = review.get("probe_validity")
    if not isinstance(behavior, Mapping) or not isinstance(validity, Mapping):
        return []
    if validity.get("schema_version") != PROBE_VALIDITY_SCHEMA:
        return []
    raw_probes, raw_results, raw_receipts = (
        behavior.get("probes"),
        behavior.get("results"),
        validity.get("receipts"),
    )
    if not all(isinstance(value, list) for value in (raw_probes, raw_results, raw_receipts)):
        return []

    def unique_by_probe_id(values: list[Any]) -> dict[str, Mapping[str, Any]] | None:
        indexed: dict[str, Mapping[str, Any]] = {}
        for value in values:
            if not isinstance(value, Mapping):
                return None
            probe_id = str(value.get("probe_id") or "")
            if not probe_id or probe_id in indexed:
                return None
            indexed[probe_id] = value
        return indexed

    probes = unique_by_probe_id(raw_probes)
    results = unique_by_probe_id(raw_results)
    receipts = unique_by_probe_id(raw_receipts)
    if probes is None or results is None or receipts is None:
        return []

    acceptance = review.get("acceptance_obligations")
    if not isinstance(acceptance, Mapping):
        return []
    public_requirements = {
        str(requirement)
        for requirement in acceptance.values()
        if isinstance(requirement, str) and requirement.strip()
    }
    if not public_requirements:
        return []

    rows: list[dict[str, Any]] = []
    total_chars = 0
    for probe_id, receipt in receipts.items():
        probe, result = probes.get(probe_id), results.get(probe_id)
        if not isinstance(probe, Mapping) or not isinstance(result, Mapping):
            continue
        candidate = result.get("candidate")
        if not isinstance(candidate, Mapping) or _status(candidate.get("status")) != "fail":
            continue
        if (
            receipt.get("schema_version") != PROBE_VALIDITY_SCHEMA
            or receipt.get("origin")
            not in {"focused_actual_peer_adjudication", "host_public_atomic_contract"}
            or receipt.get("verdict") != "valid"
            or receipt.get("contract_relation") != "explicitly_authorized"
            or not str(receipt.get("identity") or "")
            or not str(receipt.get("brief_identity") or "")
        ):
            continue
        public_basis = receipt.get("public_basis")
        if not isinstance(public_basis, str) or public_basis not in public_requirements:
            continue
        feature_id = probe_id.split(":", 1)[0]
        if receipt.get("feature_id") != feature_id:
            continue
        requirement_ids, paths = probe.get("requirement_ids"), probe.get("paths")
        if (
            not isinstance(requirement_ids, list)
            or not all(isinstance(item, str) and item for item in requirement_ids)
            or not isinstance(paths, list)
            or not all(isinstance(item, str) and item for item in paths)
            or (
                result.get("requirement_ids") is not None
                and list(result.get("requirement_ids") or []) != list(requirement_ids)
            )
            or (
                result.get("paths") is not None
                and list(result.get("paths") or []) != list(paths)
            )
        ):
            continue
        raw_code = probe.get("code")
        if not isinstance(raw_code, str):
            continue
        check_source = _compact_probe_code(raw_code)
        failure_site = receipt.get("failure_site")
        operations = failure_site.get("operations") if isinstance(failure_site, Mapping) else None
        probe_lines = {line.strip() for line in raw_code.splitlines()}
        if (
            not check_source
            or len(check_source) > 1_200
            or not isinstance(operations, list)
            or not operations
            or not all(isinstance(item, str) and item.strip() for item in operations)
            or any(item.strip() not in probe_lines for item in operations)
        ):
            continue
        row = {
            "probe_id": probe_id,
            "requirement_ids": list(requirement_ids),
            "paths": list(paths),
            "public_basis": public_basis,
            "failure_operations": [item.strip() for item in operations],
            "check_source": check_source,
        }
        baseline_setup = _compact_probe_code(probe.get("baseline_setup"))
        if baseline_setup and len(baseline_setup) <= 1_200:
            row["baseline_setup"] = baseline_setup
        row_chars = len(json.dumps(row, sort_keys=True, ensure_ascii=False))
        if len(rows) >= 8 or total_chars + row_chars > 8_000:
            break
        total_chars += row_chars
        rows.append(row)
    return rows


def _check_covers_obligation(check: Mapping[str, Any], obligation: str) -> bool:
    # Peer-authored prose cannot relabel evidence for another public clause.
    return check.get("requirement_id") == requirement_id(obligation)


def _valid_source_references(refs: Any, files: Mapping[str, str]) -> bool:
    """Check provenance, not the truth of the peer's semantic interpretation."""
    if not isinstance(refs, list) or not 1 <= len(refs) <= 4:
        return False
    for ref in refs:
        if not isinstance(ref, Mapping):
            return False
        path, quote = ref.get("file_path"), ref.get("quote")
        start, end = ref.get("start_line"), ref.get("end_line")
        if (not isinstance(path, str) or path not in files
                or not isinstance(quote, str) or not quote.strip() or len(quote) > 2000
                or type(start) is not int or type(end) is not int):
            return False
        lines = files[path].splitlines()
        if not (1 <= start <= end <= len(lines)) or end - start >= 12:
            return False
        if quote.splitlines() != lines[start - 1:end]:
            return False
    return True


def _source_reference_feedback(refs: Any, files: Mapping[str, str]) -> list[dict[str, Any]]:
    """Explain the exact failed provenance check using only the named snapshot."""
    if not isinstance(refs, list) or not 1 <= len(refs) <= 4:
        return [{"error": "expected_1_to_4_references", "received": refs}]
    failures = []
    for index, ref in enumerate(refs):
        if _valid_source_references([ref], files):
            continue
        failure = {"reference_index": index, "received": ref}
        if not isinstance(ref, Mapping):
            failure["error"] = "expected_reference_object"
        elif not isinstance(ref.get("file_path"), str) or ref["file_path"] not in files:
            failure["error"] = "file_path_not_in_this_snapshot"
            failure["available_paths"] = list(files)
        else:
            lines = files[ref["file_path"]].splitlines()
            start, end, quote = ref.get("start_line"), ref.get("end_line"), ref.get("quote")
            failure["snapshot_line_count"] = len(lines)
            if type(start) is not int or type(end) is not int:
                failure["error"] = "line_range_must_be_integers_not_booleans"
            elif not 1 <= start <= end <= len(lines):
                failure["error"] = "line_range_out_of_bounds"
            elif end - start >= 12:
                failure["error"] = "reference_exceeds_12_lines"
            elif not isinstance(quote, str) or not quote.strip() or len(quote) > 2000:
                failure["error"] = "quote_must_be_nonempty_and_at_most_2000_characters"
            else:
                failure["error"] = "quote_does_not_match_named_snapshot_lines"
            if type(start) is int and 1 <= start <= len(lines):
                # This is the text at the supplied address, not an invented
                # replacement citation or proof of the semantic claim.
                last = min(len(lines), end if type(end) is int and end >= start else start, start + 11)
                failure["source_at_requested_range"] = {
                    "file_path": ref["file_path"], "start_line": start, "end_line": last,
                    "text": "\n".join(lines[start - 1:last]),
                }
        failures.append(failure)
    return failures


def _repair_source_references(client: Any, checks: list[dict[str, Any]],
                              obligations: list[str], snapshots: Mapping[str, Mapping[str, str]],
                              *, feature_base_is_baseline: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Ask only for invalid citation sides; semantic judgments are immutable here."""
    targets = []
    by_id = {requirement_id(item): item for item in obligations}
    for index, check in enumerate(checks):
        identity = check.get("requirement_id")
        if not isinstance(identity, str) or identity not in by_id:
            continue
        for side, files in snapshots.items():
            if feature_base_is_baseline and side == "feature_base_refs":
                continue
            errors = _source_reference_feedback(check.get(side), files)
            if errors:
                targets.append({"check_index": index, "requirement_id": identity, "side": side,
                                "requirement": by_id[identity], "errors": errors,
                                "existing_judgment": {key: check.get(key) for key in (
                                    "status", "evidence", "baseline_behavior", "feature_base_behavior", "current_behavior")}})
    if not targets:
        return checks, {"attempted": False}
    sources = {}
    for side, files in snapshots.items():
        side_targets = [target for target in targets if target["side"] == side]
        if not side_targets:
            continue
        anchor_texts = [target["requirement"] for target in side_targets]
        for target in side_targets:
            anchor_texts.extend(target["existing_judgment"].values())
        anchors = _public_source_anchor_names(*anchor_texts)
        sources[side] = {}
        for path, content in files.items():
            numbered = _numbered_source(content)
            # Citation repair must not hide a small file's middle when the
            # original review saw it in full and supplied no line address yet.
            if len(numbered) <= _MAX_REVIEW_PROJECTED_FILE_CHARS:
                sources[side][path] = numbered
                continue
            priority_lines = {
                ref["start_line"] for target in side_targets
                for error in target["errors"] for ref in [error.get("received")]
                if isinstance(ref, Mapping) and ref.get("file_path") == path
                and type(ref.get("start_line")) is int
            }
            sources[side][path] = _project_numbered_source(
                content, priority_lines=priority_lines, anchor_names=anchors,
            )[0]
    system = (
        "Repair only the listed invalid source-reference sides of an existing semantic review. "
        "Do not redo the review or change its verdicts, claims, findings, valid references, or code. "
        "Choose 1-4 exact spans per target that support its existing judgment from that side's "
        "provided snapshot. Each span is 1-based inclusive, at most 12 lines and 2000 characters; "
        "copy verbatim source without line-number prefixes. If no supporting source is present, "
        "return no correction for that target; never fabricate evidence. Return JSON only."
    )
    user = json.dumps({"invalid_source_references": targets, "source_snapshots": sources}, ensure_ascii=False)
    audit: dict[str, Any] = {"attempted": True, "targets": targets, "corrected": [],
                            "input_chars": len(system) + len(user)}
    try:
        from environments.org_env.llm.client import reasoning_effort_override
        with reasoning_effort_override(SEMANTIC_REVIEW_REASONING_EFFORT):
            answer = client.generate_json(
                system, user,
                {"corrections": [{"check_index": 0, "requirement_id": "exact req_ identifier", "side": "exact invalid side",
                                   "refs": [{"file_path": "exact public path", "start_line": 1, "end_line": 1, "quote": "verbatim source"}]}]},
                max_tokens=6000,
            )
    except Exception as error:
        audit["error"] = f"source_reference_repair_provider_failed:{type(error).__name__}"
        return checks, audit
    corrected = deepcopy(checks)
    allowed = {(target["check_index"], target["requirement_id"], target["side"]) for target in targets}
    for row in (answer.get("corrections") or []) if isinstance(answer, Mapping) else []:
        if (not isinstance(row, Mapping) or type(row.get("check_index")) is not int
                or not isinstance(row.get("requirement_id"), str) or not isinstance(row.get("side"), str)):
            continue
        key = (row["check_index"], row.get("requirement_id"), row.get("side"))
        if key not in allowed:
            continue
        index, identity, side = key
        refs, _ = _normalize_exact_source_references(row.get("refs"), snapshots[side])
        if not _valid_source_references(refs, snapshots[side]):
            continue
        corrected[index][side] = refs
        if feature_base_is_baseline and side == "baseline_refs":
            corrected[index]["feature_base_refs"] = deepcopy(refs)
        allowed.remove(key)
        audit["corrected"].append({"check_index": index, "requirement_id": identity, "side": side})
    return corrected, audit


def _normalize_exact_source_references(
    refs: Any,
    files: Mapping[str, str],
) -> tuple[Any, list[dict[str, Any]]]:
    """Split exact quoted prefixes into at most four mechanically verified spans.

    Review providers occasionally return a correct source excerpt with either a
    range that extends beyond the quoted prefix or more than the requested 12
    lines. Neither formatting defect changes the selected public source. We can
    repair those two cases without another model judgment: every retained line
    must still match the named snapshot exactly, in order, from the stated start
    line. If several selected references would expand past the four-reference
    schema cap, retain at least the first exact chunk from every provider-chosen
    location, then allocate remaining slots in original order. This clips only
    exact suffix text; it never invents or relocates evidence. Missing, altered,
    or non-prefix text remains invalid and is retried with the peer rather than
    routed to the feature owner.
    """

    if not isinstance(refs, list) or not 1 <= len(refs) <= 4:
        return refs, []
    prepared: list[dict[str, Any]] = []
    for ref in refs:
        if not isinstance(ref, Mapping):
            return refs, []
        path, quote = ref.get("file_path"), ref.get("quote")
        start, declared_end = ref.get("start_line"), ref.get("end_line")
        if (
            not isinstance(path, str)
            or path not in files
            or not isinstance(quote, str)
            or not quote.strip()
            or len(quote) > 2_000
            or type(start) is not int
            or type(declared_end) is not int
        ):
            return refs, []
        lines = files[path].splitlines()
        quoted_lines = quote.splitlines()
        original_start = start
        original_declared_end = declared_end
        quoted_end = start + len(quoted_lines) - 1
        line_shift = 0
        if (
            not (1 <= start <= quoted_end <= declared_end <= len(lines))
            or quoted_lines != lines[start - 1:quoted_end]
        ):
            shifted_start = start - 1
            shifted_end = shifted_start + len(quoted_lines) - 1
            shifted_declared_end = declared_end - 1
            if (
                not (1 <= shifted_start <= shifted_end <= shifted_declared_end <= len(lines))
                or quoted_lines != lines[shifted_start - 1:shifted_end]
            ):
                return refs, []
            start = shifted_start
            quoted_end = shifted_end
            declared_end = shifted_declared_end
            line_shift = -1
        if not quoted_lines:
            return refs, []
        chunks: list[dict[str, Any]] = []
        for offset in range(0, len(quoted_lines), 12):
            chunk_lines = quoted_lines[offset:offset + 12]
            chunk_start = start + offset
            chunks.append({
                "file_path": path,
                "start_line": chunk_start,
                "end_line": chunk_start + len(chunk_lines) - 1,
                "quote": "\n".join(chunk_lines),
            })
        prepared.append({
            "ref": dict(ref),
            "file_path": path,
            "original_start_line": original_start,
            "original_end_line": original_declared_end,
            "quoted_end_line": quoted_end,
            "quoted_line_count": len(quoted_lines),
            "chunks": chunks,
            "line_shift": line_shift,
        })

    chunk_count = sum(len(item["chunks"]) for item in prepared)
    quotas = [len(item["chunks"]) for item in prepared]
    clipped = chunk_count > 4
    if clipped:
        # ``len(refs) <= 4`` guarantees one slot per provider-selected source
        # location. Give remaining slots to exact suffix chunks in stable input
        # order; no source content influences this allocation.
        quotas = [1] * len(prepared)
        remaining = 4 - len(prepared)
        while remaining:
            advanced = False
            for index, item in enumerate(prepared):
                if quotas[index] >= len(item["chunks"]):
                    continue
                quotas[index] += 1
                remaining -= 1
                advanced = True
                if not remaining:
                    break
            if not advanced:
                break

    normalized: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    for item, quota in zip(prepared, quotas):
        selected = item["chunks"][:quota]
        normalized.extend(selected)
        unchanged = (
            len(selected) == 1
            and item["original_end_line"] == item["quoted_end_line"]
            and item["ref"] == selected[0]
        )
        if not unchanged:
            retained_lines = sum(
                ref["end_line"] - ref["start_line"] + 1
                for ref in selected
            )
            audit.append({
                "file_path": item["file_path"],
                "original_start_line": item["original_start_line"],
                "original_end_line": item["original_end_line"],
                "quoted_end_line": item["quoted_end_line"],
                "normalized_ranges": [
                    [ref["start_line"], ref["end_line"]]
                    for ref in selected
                ],
                "omitted_exact_suffix_lines": (
                    item["quoted_line_count"] - retained_lines
                ),
                "resolution": (
                    "exact_quote_one_line_reference_shift"
                    if item["line_shift"]
                    else "exact_prefix_split_with_reference_coverage_clip"
                    if clipped
                    else "exact_prefix_split_into_verified_spans"
                ),
                **(
                    {"line_shift": item["line_shift"]}
                    if item["line_shift"]
                    else {}
                ),
            })
    return normalized, audit


def _numbered_source(content: str) -> str:
    return "\n".join(f"{line}: {text}" for line, text in enumerate(content.splitlines(), 1))


def _changed_line_numbers(before: str, after: str) -> tuple[set[int], set[int]]:
    """Return exact 1-based changed lines on both sides of a source diff."""

    before_lines = str(before or "").splitlines()
    after_lines = str(after or "").splitlines()
    old: set[int] = set()
    new: set[int] = set()
    matcher = difflib.SequenceMatcher(None, before_lines, after_lines, autojunk=False)
    for tag, old_start, old_end, new_start, new_end in matcher.get_opcodes():
        if tag == "equal":
            continue
        old.update(range(old_start + 1, old_end + 1))
        new.update(range(new_start + 1, new_end + 1))
        # Pure insertions/deletions still need an adjacent visible anchor.
        if old_start == old_end and before_lines:
            old.add(min(max(old_start + 1, 1), len(before_lines)))
        if new_start == new_end and after_lines:
            new.add(min(max(new_start + 1, 1), len(after_lines)))
    return old, new


def _public_source_anchor_names(*texts: Any) -> set[str]:
    """Extract public identifiers for bounded source projection, not verdicts."""

    names: set[str] = set()
    for value in texts:
        text = str(value or "")
        names.update(
            match.group(1)
            for match in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", text)
        )
        for quoted in re.findall(r"`([^`]{1,160})`", text):
            names.update(
                token
                for token in re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", quoted)
                if len(token) >= 3
            )
    return {name for name in names if len(name) >= 3}


def _project_numbered_source(
    content: str,
    *,
    priority_lines: set[int],
    anchor_names: set[str],
    max_chars: int = _MAX_REVIEW_PROJECTED_FILE_CHARS,
) -> tuple[str, dict[str, Any]]:
    """Render exact, line-addressable source windows within a hard budget.

    Changed hunks are selected first, then publicly named symbols and a small
    file header.  The projection does not alter the full snapshots used by the
    mechanical reference validator; a reviewer quote must still match the
    original file and line number.
    """

    lines = str(content or "").splitlines()
    if not lines:
        return "", {
            "projected": True,
            "original_chars": 0,
            "rendered_chars": 0,
            "selected_lines": 0,
            "total_lines": 0,
        }
    selected: set[int] = set()

    def include(center: int, radius: int) -> None:
        start = max(1, int(center) - radius)
        end = min(len(lines), int(center) + radius)
        selected.update(range(start, end + 1))

    for line_no in sorted(priority_lines):
        include(line_no, _SOURCE_PROJECTION_CONTEXT_LINES)
    if anchor_names:
        anchor_pattern = re.compile(
            r"\b(?:" + "|".join(
                re.escape(name) for name in sorted(anchor_names, key=lambda item: (-len(item), item))
            ) + r")\b"
        )
        for line_no, line in enumerate(lines, 1):
            if anchor_pattern.search(line):
                include(line_no, 6)
    selected.update(range(1, min(len(lines), 24) + 1))
    if not priority_lines:
        selected.update(range(max(1, len(lines) - 12), len(lines) + 1))

    # Changed lines are mandatory. Optional header/symbol context is admitted
    # in line order only while the rendered hard limit remains available.
    mandatory: set[int] = set()
    for line_no in priority_lines:
        if 1 <= line_no <= len(lines):
            mandatory.add(line_no)
    def render_exact(chosen: set[int]) -> str:
        rows: list[str] = []
        previous = 0
        for line_no in sorted(chosen):
            if previous and line_no != previous + 1:
                rows.append(f"... omitted lines {previous + 1}-{line_no - 1} ...")
            rows.append(f"{line_no}: {lines[line_no - 1]}")
            previous = line_no
        if previous and previous < len(lines):
            rows.append(f"... omitted lines {previous + 1}-{len(lines)} ...")
        return "\n".join(rows)

    # Reserve the complete rendered cost of changed lines, including omission
    # markers. Previously optional early lines filled the budget before the
    # renderer reached already-admitted late changed hunks.
    admitted = set(mandatory)
    if len(render_exact(admitted)) <= max_chars:
        for line_no in sorted(selected - mandatory):
            candidate = admitted | {line_no}
            if len(render_exact(candidate)) <= max_chars:
                admitted = candidate
    # A single unusually long changed line may exceed the budget. Keep a
    # bounded prefix with its true line number and record the clipping.
    clipped_lines: list[int] = []
    rows: list[str] = []
    previous = 0
    for line_no in sorted(admitted):
        if previous and line_no != previous + 1:
            rows.append(f"... omitted lines {previous + 1}-{line_no - 1} ...")
        raw = f"{line_no}: {lines[line_no - 1]}"
        remaining = max_chars - sum(len(row) + 1 for row in rows)
        if remaining <= 0:
            break
        if len(raw) > remaining:
            raw = raw[: max(0, remaining - 24)] + " ... [line clipped]"
            clipped_lines.append(line_no)
        rows.append(raw)
        previous = line_no
    if previous and previous < len(lines):
        rows.append(f"... omitted lines {previous + 1}-{len(lines)} ...")
    rendered = "\n".join(rows)
    if len(rendered) > max_chars:
        rendered = rendered[: max_chars - 24] + "\n... [projection clipped]"
    return rendered, {
        "projected": True,
        "original_chars": len(str(content or "")),
        "rendered_chars": len(rendered),
        "selected_lines": len(admitted),
        "total_lines": len(lines),
        "priority_lines": len(mandatory),
        "clipped_lines": clipped_lines,
    }


def _source_change_ranges(
    baseline: str,
    current: str,
    *,
    max_chars: int = _MAX_REVIEW_CHANGE_RANGE_CHARS,
) -> str:
    """Describe changed spans without transmitting a third copy of source."""

    rows: list[str] = []
    matcher = difflib.SequenceMatcher(
        None, baseline.splitlines(), current.splitlines(), autojunk=False
    )
    for tag, old_start, old_end, new_start, new_end in matcher.get_opcodes():
        if tag == "equal":
            continue
        old_span = (
            f"{old_start + 1}-{old_end}" if old_start < old_end
            else f"insertion-before-{old_start + 1}"
        )
        new_span = (
            f"{new_start + 1}-{new_end}" if new_start < new_end
            else f"deletion-before-{new_start + 1}"
        )
        rows.append(f"- {tag}: baseline lines {old_span} -> current lines {new_span}")
    if not rows:
        return "- no changes; baseline and current are identical"
    rendered = "\n".join(rows)
    if len(rendered) <= max_chars:
        return rendered

    marker = f"... {len(rows)} total change ranges; middle ranges omitted ..."
    budget = max(0, max_chars - len(marker) - 2)
    front_budget = budget // 2
    front: list[str] = []
    used = 0
    for row in rows:
        cost = len(row) + (1 if front else 0)
        if used + cost > front_budget:
            break
        front.append(row)
        used += cost
    back: list[str] = []
    used = 0
    for row in reversed(rows[len(front):]):
        cost = len(row) + (1 if back else 0)
        if used + cost > budget - front_budget:
            break
        back.append(row)
        used += cost
    compact = "\n".join([*front, marker, *reversed(back)])
    return compact[:max_chars]


def _compact_probe_code(code: Any) -> str:
    """Remove comments/formatting while preserving an executable check's AST.

    Public behavior plans are retained verbatim in the receipt and trajectory.
    The semantic-review request only needs a lossless representation of Python
    behavior, not generated comments and blank lines.  Sending the full plan,
    full observations, two source snapshots, and a diff made one real review
    request 89k characters; on the configured gateway that repeatedly ended in
    503/attempt timeout while ordinary smaller calls continued to work.
    """

    text = str(code or "")
    try:
        text = ast.unparse(ast.parse(text))
    except (SyntaxError, ValueError, TypeError):
        # Plans have already passed validation, but fail closed if an older
        # persisted plan cannot be parsed by this interpreter.
        pass
    return text


def _compact_probe_observation(value: Any) -> Any:
    if not isinstance(value, Mapping):
        return None
    projected = {
        key: value.get(key)
        for key in (
            "status", "returncode", "assertions", "assertion_guards",
            "public_calls", "public_paths", "public_reads",
        )
        if key in value
    }
    output = str(value.get("output") or "")
    if output:
        # The assertion/traceback tail contains the useful failure signal; the
        # runner's leading instrumentation counters are already structured.
        projected["output_tail"] = output[-_MAX_REVIEW_PROBE_OUTPUT_CHARS:]
    return projected


def _review_behavior_projection(behavior: Mapping[str, Any]) -> dict[str, Any]:
    """Build a compact, semantically faithful provider payload.

    Full probe definitions and raw execution evidence remain in
    ``behavior_evidence``.  This projection de-duplicates probe/result identity,
    AST-normalizes Python source, and bounds failure output.  It does not omit
    any check or change pass/fail state, requirement mapping, public path, or
    executable assertion.
    """

    probes = {
        str(item.get("probe_id") or ""): item
        for item in (behavior.get("probes") or [])
        if isinstance(item, Mapping)
    }
    checks: list[dict[str, Any]] = []
    seen: set[str] = set()
    for result in (behavior.get("results") or []):
        if not isinstance(result, Mapping):
            continue
        probe_id = str(result.get("probe_id") or "")
        probe = probes.get(probe_id) or {}
        seen.add(probe_id)
        checks.append({
            "probe_id": probe_id,
            "requirement_ids": list(
                probe.get("requirement_ids") or result.get("requirement_ids") or []
            ),
            "paths": list(probe.get("paths") or result.get("paths") or []),
            "kind": str(probe.get("kind") or "runtime"),
            "compare_baseline": bool(probe.get("compare_baseline")),
            "baseline_setup": _compact_probe_code(probe.get("baseline_setup")),
            "check_source": _compact_probe_code(probe.get("code")),
            "baseline": _compact_probe_observation(result.get("baseline")),
            "candidate": _compact_probe_observation(result.get("candidate")),
        })
    # Preserve a plan row even if execution failed before producing its result.
    for probe_id, probe in probes.items():
        if probe_id in seen:
            continue
        checks.append({
            "probe_id": probe_id,
            "requirement_ids": list(probe.get("requirement_ids") or []),
            "paths": list(probe.get("paths") or []),
            "kind": str(probe.get("kind") or "runtime"),
            "compare_baseline": bool(probe.get("compare_baseline")),
            "baseline_setup": _compact_probe_code(probe.get("baseline_setup")),
            "check_source": _compact_probe_code(probe.get("code")),
            "baseline": None,
            "candidate": None,
        })
    projected = {
        key: behavior.get(key)
        for key in (
            "available", "error", "ok", "policy", "plan_hash",
            "baseline_hash", "candidate_hash", "runtime",
        )
        if key in behavior
    }
    projected.update({
        "projection_policy": "ast_compact_checks_v1",
        "checks": checks,
    })
    return projected


def semantic_review_source_snapshot_matches(world: Any, receipt: Mapping[str, Any],
                                             pull_request: Any = None) -> bool:
    """A source-view-stage receipt describes one still-current frozen PR head."""
    from .source_views import (
        SourceViewError,
        _pr_base_from_validated_head,
        freeze_baseline,
        pr_head_snapshot,
        source_views_enabled,
    )

    if not source_views_enabled(world):
        return True
    source = receipt.get("source_snapshot")
    if not isinstance(source, Mapping):
        return False
    if pull_request is None:
        repo = getattr(getattr(world, "repo_system", None), "repo", None)
        pull_request = (getattr(repo, "pull_requests", {}) or {}).get(str(source.get("pr_id") or ""))
    if pull_request is None:
        return False
    try:
        current = pr_head_snapshot(world, pull_request, require_current=True)
        baseline = freeze_baseline(world)
        feature_base = _pr_base_from_validated_head(world, current)
    except SourceViewError:
        return False
    return bool(dict(source) == current.receipt()
                and receipt.get("baseline_snapshot") == baseline.receipt()
                and receipt.get("feature_base_snapshot") == feature_base.receipt())


def review_feature_pull_request(
    world: Any,
    *,
    reviewer_id: str,
    pull_request: Any,
) -> dict[str, Any]:
    """Review one immutable PR head and reject head changes during provider work."""
    from .source_views import (
        SourceViewError, assert_pr_head_current, freeze_baseline,
        pr_base_snapshot, pr_head_snapshot, source_views_enabled,
    )
    from .visibility import is_strict_coop

    if not source_views_enabled(world):
        return _review_feature_pull_request(world, reviewer_id=reviewer_id, pull_request=pull_request)
    linked = list(getattr(pull_request, "linked_issue_ids", []) or [])
    if getattr(pull_request, "linked_issue", None):
        linked.append(pull_request.linked_issue)
    features = sorted({str(item) for item in linked if str(item).startswith("cooper_feature_")})
    if len(features) != 1:
        return {"available": False, "approved": False, "error": "semantic_review_requires_one_feature"}
    feature_id = features[0]
    if is_strict_coop(world) and visible_review_contract(world, reviewer_id, feature_id) is None:
        return {"available": False, "approved": False, "feature_id": feature_id,
                "error": "semantic_review_brief_unshared_or_unread", "repair_paths": []}
    try:
        snapshot = pr_head_snapshot(world, pull_request, require_current=True)
        baseline = freeze_baseline(world)
        feature_base = pr_base_snapshot(world, pull_request, require_current=True)
    except SourceViewError as error:
        return {"available": False, "approved": False, "feature_id": feature_id,
                "error": "semantic_review_source_snapshot_unavailable:" + error.code, "repair_paths": []}
    result = _review_feature_pull_request(
        world, reviewer_id=reviewer_id, pull_request=pull_request,
        source_snapshot=snapshot, baseline_snapshot=baseline,
        feature_base_snapshot=feature_base)
    result.update(source_snapshot=snapshot.receipt(), baseline_snapshot=baseline.receipt(),
                  feature_base_snapshot=feature_base.receipt(),
                  reviewed_pr_revision={key: list(snapshot.receipt()[key]) for key in ("commit_ids", "patch_ids")})
    try:
        assert_pr_head_current(world, pull_request, snapshot)
    except SourceViewError as error:
        return {"available": False, "approved": False, "feature_id": feature_id,
                 "error": "semantic_review_source_snapshot_stale:" + error.code, "repair_paths": [],
                 "source_snapshot": snapshot.receipt(), "baseline_snapshot": baseline.receipt(),
                 "feature_base_snapshot": feature_base.receipt(),
                 "reviewed_pr_revision": result["reviewed_pr_revision"],
                "behavior_evidence": result.get("behavior_evidence", {}),
                "stale_review_would_approve": bool(result.get("approved"))}
    return result


def _review_feature_pull_request(
    world: Any,
    *,
    reviewer_id: str,
    pull_request: Any,
    source_snapshot: Any = None,
    baseline_snapshot: Any = None,
    feature_base_snapshot: Any = None,
) -> dict[str, Any]:
    """Ask the actual peer to audit current code against the public feature brief.

    This is deliberately a pre-merge review over agent-visible task text and
    current product files.  It never opens Cooper evaluator patches or results.
    """

    linked = list(getattr(pull_request, "linked_issue_ids", []) or [])
    fallback = getattr(pull_request, "linked_issue", None)
    if fallback:
        linked.append(fallback)
    features = sorted(
        {str(item) for item in linked if str(item).startswith("cooper_feature_")}
    )
    if len(features) != 1:
        return {
            "available": False,
            "approved": False,
            "error": "semantic_review_requires_one_feature",
        }
    feature_id = features[0]
    from .visibility import is_strict_coop

    strict = is_strict_coop(world)
    contract = visible_review_contract(world, reviewer_id, feature_id) if strict else None
    if strict and contract is None:
        return {"available": False, "approved": False, "feature_id": feature_id,
                "error": "semantic_review_brief_unshared_or_unread", "repair_paths": []}
    if contract is not None:
        description = contract["description"]
    else:
        task = (getattr(world, "tasks", {}) or {}).get(f"task_oss_{feature_id}")
        description = str(getattr(task, "description", "") or "").strip()
    state = getattr(world, "__dict__", {}).get("_cooperbench_sdl_state") or {}
    required_paths = [
        str(path).replace("\\", "/")
        for path in (state.get("required_feature_paths") or {}).get(feature_id, [])
        if str(path).strip()
    ]
    ranked_public_paths = [
        str(path).replace("\\", "/")
        for path in (state.get("feature_paths") or {}).get(feature_id, [])
        if str(path).strip()
    ]
    # Required paths prove delivery coverage.  Peer review needs the bounded,
    # already-ranked public component surface as well, otherwise it cannot
    # localize a repair outside the first guessed implementation target.
    review_paths = list(dict.fromkeys([*ranked_public_paths, *required_paths]))
    if contract is not None:
        obligations = list(contract["requirements"].values())
        compatibility_obligations = list(contract["compatibility"].values())
    else:
        obligations = [
            str(item)[:500]
            for item in (state.get("acceptance_obligations") or {}).get(feature_id, [])
            if str(item).strip()
        ]
        compatibility_obligations = [
            str(item)[:500]
            for item in (state.get("compatibility_obligations") or {}).get(feature_id, [])
            if str(item).strip()
        ]
    baseline_files = baseline_snapshot.files if baseline_snapshot is not None else (
        getattr(world, "__dict__", {}).get(
            "_cooperbench_public_baseline_files"
        )
        or {}
    )
    feature_base_files = (
        feature_base_snapshot.files
        if feature_base_snapshot is not None
        else baseline_files
    )
    artifacts_by_path = {
        str(getattr(artifact, "linked_file_path", "") or "").replace("\\", "/"): artifact
        for artifact in (getattr(world, "product_artifacts", {}) or {}).values()
        if str(getattr(artifact, "linked_file_path", "") or "")
    }
    rendered_files: list[str] = []
    source_projections: list[dict[str, Any]] = []
    current_files: dict[str, str] = {}
    reviewed_baselines: dict[str, str] = {}
    reviewed_feature_bases: dict[str, str] = {}
    total_chars = 0
    source_files = source_snapshot.files if source_snapshot is not None else None
    aggregate_rendered_upper_bound = 0
    for path in review_paths:
        artifact = artifacts_by_path.get(path)
        if source_files is not None:
            current = str(source_files.get(path) or "")
        else:
            current = str(
                getattr(artifact, "content", None)
                if artifact is not None
                else ""
            )
            if artifact is not None and getattr(artifact, "content", None) is None:
                current = str(getattr(artifact, "mainline_content", "") or "")
        baseline = str(baseline_files.get(path) or "")
        feature_base = feature_base_files.get(path)
        feature_base = None if feature_base is None else str(feature_base)
        # The review transmits numbered source, whose line prefixes can make a
        # multi-file request materially larger than the raw files.  Budget the
        # representation that will actually be sent, plus the two bounded
        # change-range sections and headings for this file.
        aggregate_rendered_upper_bound += (
            len(_numbered_source(current))
            + len(_numbered_source(baseline))
            + 2 * _MAX_REVIEW_CHANGE_RANGE_CHARS
            + 512
            + len(path)
        )
        if feature_base != baseline:
            aggregate_rendered_upper_bound += len(
                _numbered_source(feature_base or "")
            )
    aggregate_projection = (
        aggregate_rendered_upper_bound > _MAX_REVIEW_TOTAL_CHARS
    )
    per_snapshot_projection_chars = min(
        _MAX_REVIEW_PROJECTED_FILE_CHARS,
        max(
            512,
            (
                (_MAX_REVIEW_TOTAL_CHARS // max(1, len(review_paths)))
                - 3_000
            ) // 3,
        ),
    )
    for path in review_paths:
        artifact = artifacts_by_path.get(path)
        if (path not in source_files if source_files is not None else artifact is None):
            return {
                "available": False,
                "approved": False,
                "feature_id": feature_id,
                "error": f"required_file_missing:{path}",
            }
        if source_files is not None:
            content = source_files[path]
        else:
            content = getattr(artifact, "content", None)
            if content is None:
                content = getattr(artifact, "mainline_content", "")
        content = str(content or "")
        current_files[path] = content
        baseline_present = path in baseline_files
        baseline = str(baseline_files.get(path) or "")
        reviewed_baselines[path] = baseline
        feature_base = feature_base_files.get(path)
        if feature_base is None and path in baseline_files:
            return {
                "available": False,
                "approved": False,
                "feature_id": feature_id,
                "error": f"required_feature_base_file_missing:{path}",
            }
        feature_base = None if feature_base is None else str(feature_base)
        if feature_base is not None:
            reviewed_feature_bases[path] = feature_base
        feature_base_duplicates_baseline = bool(
            baseline_present and feature_base == baseline
        )
        full_chars = (
            len(content) + len(baseline)
            + (0 if feature_base_duplicates_baseline else len(feature_base or ""))
        )
        project_source = bool(
            aggregate_projection
            or max(len(content), len(baseline), len(feature_base or ""))
            > _MAX_REVIEW_FILE_CHARS
            or total_chars + full_chars > _MAX_REVIEW_TOTAL_CHARS
        )
        anchors = _public_source_anchor_names(
            description, *obligations, *compatibility_obligations
        )
        inherited_old, inherited_new = _changed_line_numbers(
            baseline, feature_base or ""
        )
        baseline_current_old, _ = _changed_line_numbers(baseline, content)
        current_old, current_new = _changed_line_numbers(
            feature_base or "", content
        )
        if project_source:
            if baseline_present:
                baseline_text, baseline_projection = _project_numbered_source(
                    baseline,
                    priority_lines=inherited_old | baseline_current_old,
                    anchor_names=anchors,
                    max_chars=per_snapshot_projection_chars,
                )
            else:
                baseline_text = "<FILE ABSENT AT UNTOUCHED PUBLIC BASELINE>"
                baseline_projection = {
                    "projected": True,
                    "file_absent": True,
                    "original_chars": 0,
                    "rendered_chars": len(baseline_text),
                }
            if feature_base_duplicates_baseline:
                feature_base_text = (
                    "<IDENTICAL TO UNTOUCHED PUBLIC BASELINE FILE; "
                    "use the same numbered lines>"
                )
                feature_base_projection = {
                    "projected": True,
                    "identical_to_baseline": True,
                    "original_chars": len(feature_base or ""),
                    "rendered_chars": 0,
                }
            elif feature_base is not None:
                feature_base_text, feature_base_projection = (
                    _project_numbered_source(
                        feature_base,
                        priority_lines=inherited_new | current_old,
                        anchor_names=anchors,
                        max_chars=per_snapshot_projection_chars,
                    )
                )
            else:
                feature_base_text = "<FILE ABSENT AT FEATURE BRANCH BASE>"
                feature_base_projection = {
                    "projected": True,
                    "file_absent": True,
                    "original_chars": 0,
                    "rendered_chars": len(feature_base_text),
                }
            current_text, current_projection = _project_numbered_source(
                content,
                priority_lines=current_new,
                anchor_names=anchors,
                max_chars=per_snapshot_projection_chars,
            )
        else:
            baseline_text = (
                _numbered_source(baseline)
                if baseline_present
                else "<FILE ABSENT AT UNTOUCHED PUBLIC BASELINE>"
            )
            feature_base_text = (
                "<IDENTICAL TO UNTOUCHED PUBLIC BASELINE FILE; "
                "use the same numbered lines>"
                if feature_base_duplicates_baseline
                else (
                    _numbered_source(feature_base)
                    if feature_base is not None
                    else "<FILE ABSENT AT FEATURE BRANCH BASE>"
                )
            )
            current_text = _numbered_source(content)
            baseline_projection = {
                "projected": False,
                "file_absent": not baseline_present,
                "original_chars": len(baseline),
                "rendered_chars": len(baseline_text),
            }
            feature_base_projection = {
                "projected": False,
                "identical_to_baseline": feature_base_duplicates_baseline,
                "original_chars": len(feature_base or ""),
                "rendered_chars": len(feature_base_text),
            }
            current_projection = {
                "projected": False,
                "original_chars": len(content),
                "rendered_chars": len(current_text),
            }
        # The untouched baseline proves repository compatibility.  The protected
        # feature base proves attribution: already merged peer work is not part
        # of this PR and must not be diagnosed as an unrequested current change.
        current_heading = "FROZEN COMMITTED PR-HEAD FILE" if source_snapshot is not None else "CURRENT INTEGRATED FILE"
        rendered = (
            f"FILE {path}\n"
            f"UNTOUCHED PUBLIC BASELINE FILE\n```\n{baseline_text}\n```\n"
            "PROTECTED FEATURE-BRANCH BASE FILE (already merged mainline inherited by this PR)\n"
            f"```\n{feature_base_text}\n```\n"
            f"{current_heading}\n```\n{current_text}\n```\n"
            "INHERITED MAINLINE CHANGE RANGES (untouched baseline -> protected feature base)\n"
            f"{_source_change_ranges(baseline, feature_base or '')}\n"
            "CURRENT FEATURE-PR CHANGE RANGES (protected feature base -> committed PR head)\n"
            f"{_source_change_ranges(feature_base or '', content)}"
        )
        total_chars += len(rendered)
        if total_chars > _MAX_REVIEW_TOTAL_CHARS:
            return {
                "available": False,
                "approved": False,
                "feature_id": feature_id,
                "error": "relevant_source_projection_exceeds_review_budget",
                "source_projections": source_projections,
            }
        rendered_files.append(rendered)
        source_projections.append(
            {
                "path": path,
                "projection_used": project_source,
                "baseline": baseline_projection,
                "feature_base": feature_base_projection,
                "current": current_projection,
            }
        )
    client = getattr(world, "llm_client", None)
    if client is None or not description or not rendered_files:
        return {
            "available": False,
            "approved": False,
            "feature_id": feature_id,
            "error": "semantic_review_inputs_unavailable",
        }
    obligations_text = "\n".join(f"- {requirement_id(item)}: {item}" for item in obligations)
    compatibility_text = "\n".join(
        f"- {requirement_id(item)}: {item}" for item in compatibility_obligations
    )
    probe_capability = (
        getattr(world, "__dict__", {}).get(
            "_cooperbench_behavior_probe_capability"
        )
        or {}
    )
    executable_behavior_required = bool(
        probe_capability.get("available", True)
    )
    if executable_behavior_required:
        behavior = prepare_public_behavior_evidence(
            world, reviewer_id=reviewer_id, feature_id=feature_id,
            description=description,
            requirements={requirement_id(item): item for item in obligations} or {
                requirement_id(description): description},
            compatibility={requirement_id(item): item for item in compatibility_obligations},
            paths=list(current_files),
            **({"source_snapshot": source_snapshot, "baseline_snapshot": baseline_snapshot,
                "pull_request": pull_request} if source_snapshot is not None else {}),
        )
    else:
        integration_contract = (
            _protected_integration_contract(
                world,
                str(reviewer_id),
                str(feature_id),
                list(current_files),
                contract,
            )
            if strict and contract is not None
            else {}
        )
        behavior = {
            "available": False,
            "ok": None,
            "classification": "peer_executable_probe_capability_unavailable",
            "validation_strength": "three_way_source_review_only",
            "capability": dict(probe_capability),
            "probes": [],
            "results": [],
            "integration_contract": integration_contract,
        }
    if executable_behavior_required and not behavior.get("available"):
        return {"available": False, "approved": False, "feature_id": feature_id,
                "error": behavior.get("error") or "public_behavior_evidence_unavailable",
                "behavior_evidence": behavior, "repair_paths": [],
                "input_strategy": _REVIEW_INPUT_STRATEGY,
                "source_projections": source_projections}
    reference_schema = [{
        "file_path": "exact FILE path", "start_line": "integer (1-based)",
        "end_line": "integer (inclusive)",
        "quote": "exact source lines without number prefixes; preserve indentation",
    }]
    schema = {
        "approve": "boolean",
        "probe_defects": [{"probe_id": "exact current-feature id only if the CHECK itself violates the public request",
                           "issue": "concrete public-spec reason the check is wrong; not a candidate failure"}],
        "blocking_issues": ["string"],
        "blocking_findings": [
            {"file_path": "exact FILE path", "symbol": "class/function/parameter",
             "issue": "concrete incorrect behavior", "evidence": "current code evidence",
             "requirement_ids": ["exact public req_ identifiers supporting this finding"],
             "probe_ids": ["exact executed probe identifiers supporting it, or empty for source-only findings"],
             "repair_hint": "optional non-binding approach that preserves ALL public requirements"}
        ],
        "requirement_checks": [
            {
                "requirement_id": "copy exact req_ identifier from SDL ACCEPTANCE OBLIGATIONS",
                "status": "pass|fail|uncertain",
                "evidence": "string",
            }
        ],
        "compatibility_checks": [
            {
                "requirement_id": "copy exact req_ identifier from SDL COMPATIBILITY OBLIGATIONS",
                "baseline_behavior": "string",
                "feature_base_behavior": "string",
                "current_behavior": "string",
                "baseline_refs": reference_schema,
                "feature_base_refs": reference_schema,
                "current_refs": reference_schema,
                "status": "pass|fail|uncertain",
                "evidence": "string",
            }
        ],
    }
    system_prompt = (
        "You are the second member of a two-person engineering organization performing a "
        "real pre-merge semantic review. Use only the public feature request, current files, "
        "untouched public baseline files, protected feature-branch base files, and mechanical "
        "change ranges below. Only the CURRENT FEATURE-PR CHANGE RANGES are attributable to this "
        "PR. Code already present in the PROTECTED FEATURE-BRANCH BASE is inherited committed "
        "mainline from an earlier feature: do not call it unrequested by the current feature, do "
        "not reject this PR merely because it exists. Protect its established public behavior, "
        "not its source bytes: the current public request may require modifying or replacing "
        "inherited implementation. Such a repair is allowed when the already-merged public "
        "behavior and current requested behavior coexist. "
        + PUBLIC_CONTRACT_BOUNDARY_GUIDANCE + " "
        "Reject concrete contradictions, missing "
        "required behavior, or an unjustified change to default/backward-compatible behavior. "
        "For an optional or opt-in feature, inspect raw stored types and representations, keys, "
        "exceptions, signatures, and side effects on the disabled path—not only round-trip "
        "outputs. The UNTOUCHED PUBLIC BASELINE FILE is authoritative for behavior and "
        "representation when the feature is disabled. A new format example for an enabled option "
        "does not authorize changing the default path. Scope that compatibility comparison to "
        "the same disabled call form: compare the old invocation with the current invocation that "
        "omits the new opt-in parameter or supplies its documented disabled sentinel (for example "
        "None or False). Do not compare an explicitly enabled invocation with a disabled invocation "
        "and call the requested enabled-path difference a regression. A public statement that a "
        "change only activates when a hook or option is explicitly provided permits the enabled and "
        "disabled paths to differ in exactly the publicly requested behavior; it does not permit "
        "changing the omitted/disabled path. Never invent equality across enabled and disabled call "
        "forms unless the public request explicitly requires that equality. For every SDL compatibility obligation, "
        "copy its req_ identifier into compatibility_checks.requirement_id and compare baseline versus current "
        "evidence through the protected feature base. A failure is attributable to this PR only "
        "when current behavior regresses or contradicts the protected feature-base behavior. "
        "Cite 1-4 short baseline_refs, feature_base_refs, and current_refs per compatibility row, each "
        "with exact FILE path, 1-based inclusive line range (at most 12 lines), and verbatim "
        "quote without the displayed line-number prefixes. Quotes are mechanically checked "
        "against their own snapshot; do not cite candidate code as baseline or infer old "
        "behavior from the new implementation. Missing or invalid references require you "
        "to complete the review, not the owner to change code. A valid quote only establishes "
        "source provenance: independently check that it actually supports the claimed behavior. "
        "If an identifier or parameter appears in both feature_base_refs and current_refs, its "
        "mere presence is inherited—not a current-feature scope violation. "
        "When a protected feature-base file is marked identical to the untouched baseline, cite "
        "the same exact baseline line text and line numbers in feature_base_refs. "
        "Do not speculate about hidden tests. Set approve=true only when every explicit "
        "public requirement is implemented without a visible contradiction. For each blocking "
        "issue include a blocking_findings entry with an exact FILE path, symbol, code evidence "
        "and repair hint. Attribute each finding with exact requirement_ids and supporting probe_ids "
        "(empty probe_ids only for an independent source finding), so invalid check conclusions "
        "can be withdrawn without contaminating unrelated product repairs. "
        "Check actual Python name binding: a later method overrides an earlier "
        "property, not the reverse. Distinguish behavioral requirements from illustrative API "
        "snippets; a callable accessor can satisfy property-like reading and context-manager "
        "usage when its behavior meets the public contract. Do not demand an incompatible "
        "concrete type merely because an example assigns the value to a variable. "
        "The public request is authoritative; previous peer diagnoses and repair hints are "
        "fallible proposals, not additional requirements. Before suggesting a repair, check "
        "that it can coexist with EVERY public clause, including previously passing call forms. "
        "A repair_hint must give one coexistence-safe direction, never OR alternatives that "
        "rename, remove, replace, or change the type of an explicit public entry point. Omit the "
        "hint when no such direction is established. "
        "Never recommend replacing an explicitly required entry point with a differently named "
        "helper. If an earlier finding imposed an unsupported type or implementation restriction, "
        "retract that restriction and judge the actual public behavior instead. "
        "Address the previous findings explicitly, including whether the owner repaired the "
        "relevant implementation or only changed documentation. Include exactly one "
        "requirement_checks row for EVERY SDL acceptance req_ identifier, and one "
        "compatibility_checks row for EVERY compatibility req_ identifier. Both lists are "
        "mandatory even when they share identifiers. A global approve=true cannot replace "
        "these checks. Treat every acceptance row as an atomic observable clause. If a clause "
        "orders multiple operations (for example resolve, then trim, then invoke a callback), "
        "trace and verify every operation and its order in the current source. A passing probe "
        "for the callback, size, or other final observer cannot prove an earlier transformation "
        "that the probe never made observable. Apply this rule equally to filtering, normalization, "
        "sorting, deduplication, truncation, validation, and analogous transformations. Mark the "
        "row fail when any named operation is absent even if another part passes. When PUBLIC "
        "CROSS-FEATURE INTERACTION is present, its protected request is already merged and visible "
        "to this reviewer. For same_call_keywords, trace the listed same-call option combination "
        "through the protected feature base and current PR head. For source_trace_only, trace the "
        "shared callable's public signature, callers, ordering, cleanup, and return behavior across "
        "both requests; this remains required even when the Python probe runner cannot execute the "
        "repository language. Reject a current-feature change that weakens, reorders, or bypasses "
        "the protected public behavior in the combined public flow; do not demand behavior absent "
        "from either public request. When required by runtime_probe_mode, the executable same-call "
        "probe is evidence, not a substitute for this source-level coexistence check. For such a "
        "finding, cite the current feature's public compatibility requirement ids. "
        "invocation examples, trace that exact expression through "
        "the current code, including its returned type and callable behavior. Do not mark "
        "a renamed alternative as satisfying the original API. Keep each evidence field "
        "concise (at most 160 characters), but concrete. EXECUTED PUBLIC BEHAVIOR CHECKS below are actual "
        "runs in the task runtime, not model predictions. Their Python source is AST-normalized, comments "
        "and blank lines are removed, and failure output is represented by a bounded traceback tail; these "
        "presentation changes do not alter the executed check. Use the trace when diagnosing a failure. "
        "Do not accept a differently named API when the requested invocation actually raised. "
        "A failed baseline control indicates a defective check, not a candidate regression. "
        "A passing check does not prove untested edge cases. A direct call to a new helper proves "
        "only that helper: trace the requested input through the existing public entry point and "
        "the code that consumes its output. Actor-visible supporting source and usage examples "
        "are read-only context, not additional edit targets or a fourth source-reference snapshot. "
        "A check that mocks the dependency responsible for the requested observable side effect does "
        "not prove that side effect. For an immediate reset/completion clause, inspect state at the "
        "first public observation after its trigger rather than moving reset into a later getter. For "
        "a transformation over a complete public collection or representation, verify the complete "
        "scope rather than an occupied or sampled prefix. If a check itself contradicts the public "
        "request, put its exact probe_id and concrete reason in probe_defects; the reviewer must then "
        "repair and re-execute the check on the same head before approval, not force the owner to "
        "satisfy a fabricated requirement. Do not use probe_defects to excuse broken product code. "
        "Return JSON only."
    )
    if not executable_behavior_required:
        system_prompt += (
            " This repository's selected implementation surface is not callable by the "
            "Python peer-probe runner. No executable peer behavior check is claimed. "
            "Perform the review from the exact three-way source evidence, change ranges, "
            "public request, and source-scoped validation receipt; do not invent runtime "
            "observations. The official Cooper evaluator remains authoritative."
        )
    behavior_heading = (
        "EXECUTED PUBLIC BEHAVIOR CHECKS (compact definitions and observed output)"
        if executable_behavior_required
        else "PEER EXECUTABLE BEHAVIOR CHECK CAPABILITY (unavailable; no run claimed)"
    )
    user_prompt = (
        f"Reviewer: {reviewer_id}\nFeature: {feature_id}\n\n"
        "AUTHORITATIVE PUBLIC FEATURE REQUEST (complete; the index below is not a replacement)\n"
        f"{description}\n\n"
        "PUBLIC SOURCE SPANS FOR THE SUPPLEMENTARY INDEX\n"
        + json.dumps(acceptance_index(description), ensure_ascii=False)
        + "\n\n"
        "SDL ACCEPTANCE INDEX (supplementary; also review requirements absent from this index)\n"
        f"{obligations_text or description}\n\n"
        "SDL DEFAULT AND COMPATIBILITY OBLIGATIONS\n"
        f"{compatibility_text or '- None explicitly stated.'}\n\n"
        "PUBLIC CROSS-FEATURE INTERACTION (public and reviewer-visible; empty means none compiled)\n"
        + json.dumps(behavior.get("integration_contract") or {}, ensure_ascii=False)
        + "\n\n"
        "CURRENT PUBLIC FILES, UNTOUCHED BASELINES, PROTECTED FEATURE BASES, AND CHANGE RANGES\n"
        + "\n\n".join(rendered_files)
        + f"\n\n{behavior_heading}\n"
        + json.dumps(_review_behavior_projection(behavior), ensure_ascii=False)
    )
    from .feedback import actor_feedback_context
    from .source_retrieval import related_public_source_context

    user_prompt += (
        "\n\nACTOR-VISIBLE SUPPORTING SOURCE AND PUBLIC USAGE (read-only context)\n"
        + json.dumps(related_public_source_context(world, reviewer_id, review_paths), ensure_ascii=False)
        + "\n\nREVIEWER'S READ-RECEIPTED COMMUNICATION AND OWN PUBLIC-TEST FEEDBACK\n"
        + json.dumps(actor_feedback_context(world, reviewer_id), ensure_ascii=False)
    )
    # Cache only citation-incomplete judgments against the exact public input.
    # A source/brief/probe/message change requires fresh semantic review.
    input_identity = hashlib.sha256((system_prompt + user_prompt + json.dumps({
        "current": current_files, "baseline": reviewed_baselines, "feature_base": reviewed_feature_bases,
        "source_snapshot": source_snapshot.receipt() if source_snapshot is not None else None,
        "contract_binding": _contract_binding(contract) if contract is not None else None,
        "probe_definitions": behavior.get("probes"), "probe_results": behavior.get("results"),
    }, sort_keys=True, ensure_ascii=False)).encode()).hexdigest()
    citation_cache = world.__dict__.setdefault("_cooperbench_citation_review_cache", {})
    cached = citation_cache.get(feature_id) or {}
    reuse_judgment = cached.get("input_identity") == input_identity
    previous = (getattr(world, "_cooperbench_semantic_reviews", {}) or {}).get(feature_id) or {}
    if strict and not semantic_review_visibility_matches(world, reviewer_id, feature_id, previous):
        previous = {}
    if source_snapshot is not None:
        prior_source = previous.get("source_snapshot")
        if (not isinstance(prior_source, Mapping)
                or prior_source.get("source_kind") != "committed_pr_head"
                or prior_source.get("pr_id") != source_snapshot.pr_id
                or previous.get("baseline_snapshot") != baseline_snapshot.receipt()
                or previous.get("feature_base_snapshot") != feature_base_snapshot.receipt()):
            previous = {}  # Legacy/global-working review prose is not a frozen-source observation.
    if previous:
        user_prompt += "\n\nPREVIOUS PEER FINDINGS (fallible; recheck, do not treat as extra specification)\n" + json.dumps(
            {key: previous.get(key) for key in ("blocking_issues", "blocking_findings", "repair_paths")},
            ensure_ascii=False,
        )
        user_prompt += (
            "\n\nPUBLIC BEHAVIORS PREVIOUSLY MARKED PASS (recheck for regressions on this head; "
            "a prior pass is not current approval)\n"
            + json.dumps(passed_public_requirements(
                {"acceptance_obligations": {feature_id: obligations}} if strict else state,
                feature_id, previous,
            ), ensure_ascii=False)
        )
    gaps = (getattr(world, "_cooperbench_review_evidence_gaps", {}) or {}).get(feature_id)
    if strict and (not isinstance(gaps, Mapping) or gaps.get("review_brief_identity") != contract["review_brief_identity"]):
        gaps = None
    if gaps:
        user_prompt += "\n\nPREVIOUS RESPONSE OMITTED THESE EVIDENCE ROWS; complete them:\n" + json.dumps(gaps)
    try:
        from environments.org_env.llm.client import reasoning_effort_override

        # This is a bounded checklist over explicit evidence, not open-ended
        # architecture generation. Keep the run-level client configuration
        # unchanged while making this one gate smaller and more reliable.
        if reuse_judgment:
            answer = deepcopy(cached["answer"])
        else:
            with reasoning_effort_override(SEMANTIC_REVIEW_REASONING_EFFORT):
                answer = client.generate_json(
                    system_prompt,
                    user_prompt,
                    schema,
                    max_tokens=6000,
                )
    except Exception as error:  # transport failure is not an approval
        replayed = _replayed_known_behavior_failure(
            world,
            previous,
            behavior,
            current_files,
            feature_id=feature_id,
            reviewer_id=reviewer_id,
            description=description,
            obligations=obligations,
        )
        if replayed is not None:
            replayed["provider_error"] = (
                f"semantic_review_provider_failed:{type(error).__name__}"
            )
            replayed["input_chars"] = len(system_prompt) + len(user_prompt)
            if contract is not None:
                replayed.update(_contract_binding(contract))
            return replayed
        return {
            "available": False,
            "approved": False,
            "feature_id": feature_id,
            "input_strategy": _REVIEW_INPUT_STRATEGY,
            "input_chars": len(system_prompt) + len(user_prompt),
            "reasoning_effort": SEMANTIC_REVIEW_REASONING_EFFORT,
            "error": f"semantic_review_provider_failed:{type(error).__name__}",
            "behavior_evidence": behavior,
        }
    if not isinstance(answer, Mapping) or not isinstance(answer.get("approve"), bool):
        return {
            "available": False,
            "approved": False,
            "feature_id": feature_id,
            "input_strategy": _REVIEW_INPUT_STRATEGY,
            "input_chars": len(system_prompt) + len(user_prompt),
            "reasoning_effort": SEMANTIC_REVIEW_REASONING_EFFORT,
            "error": "semantic_review_response_invalid",
            "behavior_evidence": behavior,
        }
    requirement_checks = [
        dict(item)
        for item in (answer.get("requirement_checks") or [])
        if isinstance(item, Mapping)
    ]
    ordered_source_stage_gaps = _missing_ordered_source_stages(
        {requirement_id(item): item for item in obligations},
        current_files,
    )
    for gap in ordered_source_stage_gaps:
        identity = gap["requirement_id"]
        evidence = _ordered_source_gap_evidence(gap)
        matching = next(
            (
                row
                for row in requirement_checks
                if row.get("requirement_id") == identity
            ),
            None,
        )
        if matching is None:
            requirement_checks.append(
                {
                    "requirement_id": identity,
                    "status": "fail",
                    "evidence": evidence,
                }
            )
        else:
            matching["status"] = "fail"
            matching["evidence"] = evidence
    compatibility_checks = [
        dict(item)
        for item in (answer.get("compatibility_checks") or [])
        if isinstance(item, Mapping)
    ]
    source_reference_normalizations: dict[str, dict[str, list[dict[str, Any]]]] = {}
    normalized_compatibility_checks: list[dict[str, Any]] = []
    for check in compatibility_checks:
        normalized = dict(check)
        if feature_base_snapshot is None:
            # Legacy non-snapshot worlds have no intervening mainline: their
            # feature base is exactly the untouched baseline.  Preserve that
            # historical path without asking a provider to duplicate or keep a
            # second synthetic evidence row in sync with the same bytes.
            normalized["feature_base_behavior"] = check.get("baseline_behavior")
            normalized["feature_base_refs"] = deepcopy(check.get("baseline_refs"))
        identity = str(check.get("requirement_id") or "")
        for side, files in (
            ("baseline_refs", reviewed_baselines),
            ("feature_base_refs", reviewed_feature_bases),
            ("current_refs", current_files),
        ):
            normalized_refs, audit = _normalize_exact_source_references(normalized.get(side), files)
            normalized[side] = normalized_refs
            if audit:
                source_reference_normalizations.setdefault(identity, {})[side] = audit
        normalized_compatibility_checks.append(normalized)
    compatibility_checks = normalized_compatibility_checks
    compatibility_checks, source_reference_repair = _repair_source_references(
        client, compatibility_checks, compatibility_obligations,
        {"baseline_refs": reviewed_baselines, "feature_base_refs": reviewed_feature_bases,
         "current_refs": current_files},
        feature_base_is_baseline=feature_base_snapshot is None,
    )
    source_reference_repair["reused_semantic_judgment"] = reuse_judgment
    answer = {**answer, "compatibility_checks": compatibility_checks}
    citation_answer = deepcopy(answer)
    current_probe_ids = {
        p["probe_id"] for p in behavior.get("probes", [])
        if p["probe_id"].startswith(feature_id + ":")
    }
    claimed_defects_by_id = {
        str(row.get("probe_id")): dict(row)
        for row in (answer.get("probe_defects") or [])
        if isinstance(row, Mapping) and str(row.get("issue") or "").strip()
        and row.get("probe_id") in current_probe_ids
    }
    # A confident broad source diagnosis is not independent validation of the
    # assertion's expected value. Every actual failed check needs a focused
    # verdict, including regressions borrowed from the other feature.
    for row in behavior.get("results", []):
        candidate = row.get("candidate") or {}
        probe_id = str(row.get("probe_id") or "")
        if candidate.get("status") != "fail":
            continue
        mapped = [
            check for check in requirement_checks
            if check.get("requirement_id") in set(row.get("requirement_ids") or [])
        ]
        broad_failure = any(
            _status(check.get("status")) == "fail"
            and str(check.get("evidence") or "").strip()
            for check in mapped
        )
        statuses = sorted({_status(check.get("status")) or "missing" for check in mapped}) or ["missing"]
        claimed_defects_by_id.setdefault(probe_id, {
            "probe_id": probe_id,
            "issue": (
                "Independently validate the reached failure and its expected value against the "
                "public contract; the broad review's confident failure is not probe validity."
            ) if broad_failure else (
                "The broad review did not validate this process failure as a product-contract "
                "failure; mapped requirement status was " + ",".join(statuses)
                + ". Determine whether the probe failed in its own setup/scaffolding."
            ),
            "origin": "first_executed_failure_validity" if broad_failure else "unclassified_executed_failure",
        })
    claimed_defects = list(claimed_defects_by_id.values())
    defects, rejected_probe_defects, defect_adjudication_error, probe_validity = _adjudicate_probe_defects(
        world,
        reviewer_id=reviewer_id,
        feature_id=feature_id,
        description=description,
        obligations=obligations,
        behavior=behavior,
        defects=claimed_defects,
    ) if claimed_defects else ([], [], "", {"schema_version": PROBE_VALIDITY_SCHEMA,
                                            "receipts": [], "unresolved_probe_ids": []})
    adjudicated_product_failure_ids = {
        str(row.get("probe_id") or "")
        for row in rejected_probe_defects
        if (next((item.get("candidate") for item in behavior.get("results", [])
                  if item.get("probe_id") == row.get("probe_id")), {}) or {}).get("status") == "fail"
    }
    quarantined_review_evidence: dict[str, Any] = {}
    if defects:
        # Disagreement is a review correction, never an approval or an order
        # for the implementer to rewrite code to match a defective test.
        confirmed_current_defects = [row for row in defects
                                     if row.get("adjudication_status") == "confirmed"
                                     and row["probe_id"] in current_probe_ids]
        if confirmed_current_defects:
            from .behavior_probes import invalidate_probe_rows
            invalidate_probe_rows(world, feature_id,
                                  [row["probe_id"] for row in confirmed_current_defects],
                                  defects=confirmed_current_defects)
        # If the same run also contains an independently peer-classified
        # product failure, preserve and route that real finding now while the
        # defective row is regenerated for the next review.  When every
        # failure is part of the disputed check, remain unavailable and do not
        # invent an owner repair.
        defect_ids = {str(row.get("probe_id") or "") for row in defects}
        answer, quarantined_review_evidence = _quarantine_defective_probe_review(
            answer, behavior, defect_ids,
        )
        requirement_checks = answer["requirement_checks"]
        compatibility_checks = answer["compatibility_checks"]
        if not adjudicated_product_failure_ids - defect_ids:
            return {"available": False, "approved": False, "feature_id": feature_id,
                    "error": ("public_behavior_probe_validity_unavailable" if probe_validity["unresolved_probe_ids"]
                              else "public_behavior_probe_correction_required"), "repair_paths": [],
                    "behavior_evidence": behavior, "probe_defects": defects,
                    "rejected_probe_defects": rejected_probe_defects,
                    "probe_defect_adjudication_error": defect_adjudication_error,
                    "probe_validity": probe_validity,
                    "quarantined_review_evidence": quarantined_review_evidence,
                    **(_contract_binding(contract) if contract is not None else {})}
    else:
        defect_ids = set()
    blocking = [
        str(item)[:600]
        for item in (answer.get("blocking_issues") or [])
        if str(item).strip()
    ][:8]
    findings = [
        {key: str(item.get(key) or "")[:600]
         for key in ("file_path", "symbol", "issue", "evidence", "repair_hint")}
        for item in (answer.get("blocking_findings") or [])
        if isinstance(item, Mapping)
    ][:8]
    for gap in ordered_source_stage_gaps:
        issue = (
            "Public ordered transformation stage is absent "
            f"[{gap['requirement_id']}]: {gap['stage']}"
        )
        evidence = _ordered_source_gap_evidence(gap)
        blocking.append(issue)
        for path in gap["paths"][:4]:
            findings.append(
                {
                    "file_path": path,
                    "symbol": "",
                    "issue": issue,
                    "evidence": evidence,
                    "repair_hint": (
                        f"Implement the publicly named {gap['stage']} stage before the final "
                        "observer and derive the observer value from the returned result. "
                        "Choose boundary semantics only from the public request and repository "
                        "context; preserve omitted/disabled behavior."
                    ),
                }
            )
    for finding in findings:
        issue = finding["issue"]
        if issue and issue not in blocking:
            blocking.append(issue)
    # A source-grounded but false prose pass cannot override a real failure.
    for row in behavior.get("results", []):
        if (row.get("candidate") or {}).get("status") != "fail":
            continue
        probe_id = str(row.get("probe_id") or "")
        if probe_id in defect_ids:
            continue
        if probe_id not in adjudicated_product_failure_ids:
            continue
        issue = f"Executed public behavior failed [{row['probe_id']}]"
        evidence = str(row["candidate"].get("output") or "nonzero exit")[-1500:]
        blocking.append(f"{issue}: {evidence}")
        for path in row.get("paths", []):
            if path in current_files:
                findings.append({"file_path": path, "symbol": "", "issue": issue,
                                 "evidence": evidence, "repair_hint": "Repair the public behavior shown in the executed trace; preserve the other passing checks."})
    source_reference_errors: dict[str, list[str]] = {}
    source_reference_details: dict[str, dict[str, list[dict[str, Any]]]] = {}
    compatibility_reference_sides = [
        ("baseline_refs", reviewed_baselines),
        ("current_refs", current_files),
    ]
    if feature_base_snapshot is not None:
        compatibility_reference_sides.insert(
            1, ("feature_base_refs", reviewed_feature_bases))
    for obligation in compatibility_obligations:
        identity = requirement_id(obligation)
        matching = [check for check in compatibility_checks
                    if _check_covers_obligation(check, obligation)]
        bad_sides = [
            side for side, files in compatibility_reference_sides
            if not matching or any(
                not _valid_source_references(check.get(side), files)
                for check in matching
            )
        ]
        if bad_sides:
            source_reference_errors[identity] = bad_sides
            source_reference_details[identity] = {
                side: [error for check in matching
                       for error in _source_reference_feedback(check.get(side), files)]
                if matching else [{"error": "compatibility_requirement_row_missing"}]
                for side, files in compatibility_reference_sides if side in bad_sides
            }
    missing_acceptance = [
        item for item in obligations
        if not any(check.get("requirement_id") == requirement_id(item)
                   and _status(check.get("status")) in {"pass", "fail"}
                   and str(check.get("evidence") or "").strip()
                   for check in requirement_checks)
    ]
    failed_acceptance = [
        item for item in obligations
        if any(check.get("requirement_id") == requirement_id(item)
               and _status(check.get("status")) == "fail"
               and str(check.get("evidence") or "").strip()
               for check in requirement_checks)
    ]
    for item in failed_acceptance:
        matching = next(check for check in requirement_checks
                        if check.get("requirement_id") == requirement_id(item)
                        and _status(check.get("status")) == "fail")
        blocking.append(f"Public requirement failed [{requirement_id(item)}]: {item} — {matching.get('evidence', '')}")
    unresolved_compatibility = [
        obligation
        for obligation in compatibility_obligations
        if requirement_id(obligation) in source_reference_errors
        or any(_check_covers_obligation(check, obligation)
               and _status(check.get("status")) != "pass" for check in compatibility_checks)
        or not any(
            _check_covers_obligation(check, obligation)
            and _status(check.get("status")) == "pass"
            and str(check.get("baseline_behavior") or "").strip()
            and str(check.get("feature_base_behavior") or "").strip()
            and str(check.get("current_behavior") or "").strip()
            and str(check.get("evidence") or "").strip()
            for check in compatibility_checks
        )
    ]
    critical_invariants_passed = not unresolved_compatibility
    grounded_failures = {
        requirement_id(item): [
            check for check in compatibility_checks
            if _check_covers_obligation(check, item) and _status(check.get("status")) == "fail"
               and _valid_source_references(check.get("baseline_refs"), reviewed_baselines)
               and _valid_source_references(check.get("feature_base_refs"), reviewed_feature_bases)
               and _valid_source_references(check.get("current_refs"), current_files)
               and str(check.get("baseline_behavior") or "").strip()
               and str(check.get("feature_base_behavior") or "").strip()
               and str(check.get("current_behavior") or "").strip()
               and str(check.get("evidence") or "").strip()
        ]
        for item in unresolved_compatibility
    }
    concrete_compatibility_failures = [
        item for item in unresolved_compatibility if grounded_failures[requirement_id(item)]
    ]
    # A compatibility row can be the only concrete failure. Preserve its
    # validated current-source locations rather than guessing a file from a
    # clause that happens to name the configuration entry point.
    legacy_findings = list(findings)
    legacy_repair_issues = list(blocking)
    compatibility_paths: set[str] = set()
    compatibility_groups: list[list[str]] = []
    for item in concrete_compatibility_failures:
        identity = requirement_id(item)
        for check in grounded_failures[identity]:
            issue = f"Public compatibility requirement failed [{identity}]: {item}"
            evidence = (
                f"Baseline: {str(check['baseline_behavior'])[:160]}. "
                f"Feature base: {str(check['feature_base_behavior'])[:160]}. "
                f"Current: {str(check['current_behavior'])[:160]}. "
                f"Evidence: {str(check['evidence'])[:200]}"
            )
            blocking.append(f"{issue} — {evidence}")
            group = sorted({ref['file_path'] for ref in check['current_refs']})
            if group not in compatibility_groups:
                compatibility_groups.append(group)
            for path in group:
                compatibility_paths.add(path)
                findings.append({
                    "file_path": path, "symbol": "", "issue": issue[:600],
                    "evidence": evidence[:600], "repair_hint": "",
                })
    evidence_gaps = {
        "acceptance": [requirement_id(item) for item in missing_acceptance],
        "compatibility": [requirement_id(item) for item in unresolved_compatibility
                          if item not in concrete_compatibility_failures],
        "source_references": source_reference_errors,
        "source_reference_details": source_reference_details,
    }
    world.__dict__.setdefault("_cooperbench_review_evidence_gaps", {})[feature_id] = {
        **evidence_gaps, **({"review_brief_identity": contract["review_brief_identity"]} if strict else {})}
    # Missing reviewer evidence is not a product-code diagnosis. Re-request
    # evidence on the same head instead of making the owner rewrite good code.
    incomplete_only = bool(
        (missing_acceptance or unresolved_compatibility)
        and not blocking and not concrete_compatibility_failures
    )
    if unresolved_compatibility:
        missing = "; ".join(unresolved_compatibility[:3])
        blocking.append(
            "Public default/backward-compatibility evidence is not proven across "
            f"the untouched baseline, protected feature base, and current PR head: {missing}"
        )
        blocking = blocking[:8]
    behavior_gate_satisfied = bool(
        behavior.get("ok") is True or not executable_behavior_required
    )
    approved = bool(
        answer["approve"] and behavior_gate_satisfied and not blocking and critical_invariants_passed
        and not missing_acceptance and not failed_acceptance and not defects
    )
    repair_paths: set[str] = set()
    repair_path_groups: list[list[str]] = []
    if not approved and not incomplete_only:
        repair_paths.update(compatibility_paths)
        repair_path_groups.extend(compatibility_groups)
        if legacy_findings or legacy_repair_issues or not compatibility_paths:
            legacy_paths = _repair_paths(current_files, legacy_findings, legacy_repair_issues)
            repair_paths.update(legacy_paths)
            repair_path_groups.extend([path] for path in legacy_paths if [path] not in repair_path_groups)
    only_citations_missing = (
        incomplete_only and source_reference_errors and not missing_acceptance
        and all(any(_check_covers_obligation(check, item) for check in compatibility_checks)
                for item in compatibility_obligations)
        and all(_status(check.get("status")) in {"pass", "fail"}
                and all(str(check.get(key) or "").strip() for key in (
                    "baseline_behavior", "feature_base_behavior", "current_behavior", "evidence"))
                for check in compatibility_checks)
    )
    # A normal focused response with no valid correction cannot substantiate
    # the frozen judgment. Let the next review reconsider it with the concrete
    # source gaps, rather than permanently forcing citations for that judgment.
    if only_citations_missing and (
        source_reference_repair.get("corrected") or source_reference_repair.get("error")
    ):
        citation_cache[feature_id] = {"input_identity": input_identity, "answer": citation_answer}
    else:
        citation_cache.pop(feature_id, None)
    source_sensitive_repair = any(_source_sensitive_requirement(item) for item in failed_acceptance)
    return {
        "available": not incomplete_only,
        "error": "semantic_review_incomplete_evidence" if incomplete_only else "",
        "approved": approved,
        "behavior_evidence": behavior,
        "behavior_probe_capability": dict(probe_capability),
        "executable_behavior_required": executable_behavior_required,
        "behavior_gate_satisfied": behavior_gate_satisfied,
        "feature_id": feature_id,
        "blocking_issues": blocking,
        "blocking_findings": findings,
        "repair_paths": sorted(repair_paths),
        "repair_path_groups": repair_path_groups,
        "reviewed_code_fingerprints": {
            path: implementation_fingerprint(path, content,
                                             source_sensitive=source_sensitive_repair and path in repair_paths)
            for path, content in current_files.items()
        },
        "requirement_checks": requirement_checks,
        "acceptance_obligations": {requirement_id(item): item for item in obligations},
        "missing_requirement_ids": evidence_gaps["acceptance"],
        "failed_requirement_ids": [requirement_id(item) for item in failed_acceptance],
        "evidence_gaps": evidence_gaps,
        "source_reference_errors": source_reference_errors,
        "source_reference_normalizations": source_reference_normalizations,
        "source_reference_repair": source_reference_repair,
        "compatibility_checks": compatibility_checks,
        "critical_invariants_present": bool(compatibility_obligations),
        "critical_invariants_passed": critical_invariants_passed,
        "unresolved_compatibility_obligations": unresolved_compatibility,
        "input_strategy": _REVIEW_INPUT_STRATEGY,
        "source_projections": source_projections,
        "input_chars": (0 if reuse_judgment else len(system_prompt) + len(user_prompt))
                       + int(source_reference_repair.get("input_chars") or 0),
        "reasoning_effort": SEMANTIC_REVIEW_REASONING_EFFORT,
        "rejected_probe_defects": rejected_probe_defects,
        "probe_defect_adjudication_error": defect_adjudication_error,
        "probe_validity": probe_validity,
        "probe_defects": defects,
        "probe_defect_correction_pending": bool(confirmed_current_defects) if defects else False,
        "quarantined_review_evidence": quarantined_review_evidence,
        **(_contract_binding(contract) if contract is not None else {}),
    }


__all__ = [
    "MAX_BLOCKING_SEMANTIC_REVIEW_ROUNDS",
    "PROBE_DEFECT_ADJUDICATION_REASONING_EFFORT",
    "SEMANTIC_REVIEW_REASONING_EFFORT",
    "passed_public_interactions",
    "validated_failed_public_interactions",
    "review_feature_pull_request",
    "semantic_review_repair_evidence_matches",
    "semantic_review_source_snapshot_matches",
    "semantic_review_visibility_matches",
]
