"""Peer-authored, public-spec executable checks, separate from official evals.

No benchmark-specific assertions live here. Plans are drafted from the public
request and untouched public source (not the candidate), retained across owner
repairs, and executed on every reviewed tree in the pinned networkless runtime.
These are fallible development tests, not a replacement for Cooper's evaluator.
"""
from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping

from .public_contract import purely_optional_requirement, required_public_clauses


PROBE_POLICY = "peer_public_behavior_probes_v46_cache_ownership"
PREFERRED_MAX_PROBES = 20
MAX_DRAFT_PROBES = 64
MAX_DRAFT_CHARS = 64000
MAX_PROBE_CODE_CHARS = 4096
MAX_PROBE_SETUP_CHARS = 1500
_PROBE_PROCESS_TIMEOUT_SECONDS = 15.0
_PROBE_EXECUTOR_OVERHEAD_SECONDS = 60.0

EMISSION_OBSERVER_GUIDANCE = (
    "For callbacks specified per emitted response, count actual emissions, not input chunks "
    "or receive() calls. Buffering may produce no response for several inputs. Capture the "
    "actual returned/emitted responses and compare callbacks with those same emissions, "
    "exactly once and in the required order. Drive enough inputs through the documented "
    "public lifecycle to witness an actual emission before testing callback timing or exception "
    "propagation; no callback and no exception before an emission is not a callback failure. "
    "Include termination or flush only through the documented public lifecycle. Do not require "
    "eager output, drain a buffer, or change default streaming behavior just to make a callback "
    "assertion fire. A contract that explicitly specifies input notifications is different and "
    "must be checked as written."
)

# Public-contract interpretation only; never an automatic probe verdict.
PUBLIC_CONTRACT_BOUNDARY_GUIDANCE = (
    "Distinguish a feature's implementation scope from a prohibition on other visible features. "
    "A note that an API is not needed, not currently present, or not modified by THIS feature "
    "does not authorize assert not hasattr(...) or a ban on adding that API for another visible "
    "public request. Require API absence only when the public contract explicitly forbids its "
    "presence in the combined state; otherwise check the stated behavior and coexistence of "
    "all visible requirements. Do not invent or assume an unseen feature's requirements. "
    "Keep public cache request objects distinct from opaque internal cache keys. When cache "
    "layers use opaque string keys, do not use a request dict as a backing-cache membership "
    "key or require the backing store to accept it. Exercise the documented get/put interface "
    "and observable result; inspect internal membership only with a key representation and "
    "access explicitly supported by the public contract. Do not force a wrapper or storage-type "
    "change merely to satisfy an invented internal-container assertion. "
    + EMISSION_OBSERVER_GUIDANCE
)


def _public_behavior_executor_timeout_seconds(
    probes: list[dict[str, Any]],
) -> float:
    """Cover every bounded child process in one public-behavior plan."""

    child_processes = 0
    for probe in probes:
        sides = 2 if probe.get("compare_baseline") else 1
        processes_per_side = 1 + bool(probe.get("baseline_setup"))
        child_processes += sides * processes_per_side
    return max(
        240.0,
        child_processes * _PROBE_PROCESS_TIMEOUT_SECONDS
        + _PROBE_EXECUTOR_OVERHEAD_SECONDS,
    )


def _internal_operation_requirement(text):
    """Recognize a labelled implementation recipe, not a caller usage example."""
    source = str(text).strip().replace("**", "")
    labelled = re.search(
        r"^(?!usage\b|example\b|public\b|caller\b|user\b)[\w -]+:\s*Use\s+"
        r"`(?:[A-Za-z_]\w*\.)+[A-Za-z_]\w*\([^`]*\)`\s+for\s+",
        source, re.IGNORECASE,
    )
    # A recipe scoped to a product method describes what that method must
    # invoke, not an extra callable the caller's test must import. Retain the
    # dependency obligation through an executed source assertion.
    method_owned = re.search(
        r"^(?:in|inside|within)\s+(?:the\s+)?"
        r"`(?:[A-Za-z_]\w*\.)*[A-Za-z_]\w*(?:\([^`]*\))?`"
        r"(?:\s+(?:method|function))?\s*[, :]"
        r"[^.!?]*?\b(?:apply|call|invoke|use)\s+`[^`]*\([^`]*\)`",
        source, re.IGNORECASE,
    )
    library_recipe = re.fullmatch(
        r"Use\s+`(?:[A-Za-z_]\w*\.)+[A-Za-z_]\w*\([^`]*\)`\.?",
        source, re.IGNORECASE,
    )
    declaration = re.search(
        r"\b(?:add|define|create|initialize)\s+(?:an?\s+)?"
        r"`(?:[A-Za-z_]\w*\.)*[A-Za-z_]\w*\([^`]*\)`\s+"
        r"(?:inside|within)\s+`[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*`",
        source, re.IGNORECASE,
    )
    backing_store = re.search(
        r"\bUse\s+`self\.[^`]*\([^`]*\)`.*\bin\s+(?:the\s+)?"
        r"`(?:[A-Za-z_]\w*\.)+[A-Za-z_]\w*\(\)`\s+method\b",
        source, re.IGNORECASE,
    )
    type_definition = re.search(
        r"^(?:Each|Every)\s+type\s+(?:should|must)\s+be\s+defined\s+using\s+"
        r"(?:the\s+)?`[A-Za-z_]\w*\(\)`\s+constructor\b|"
        r"^Add\b.*\btype definitions\b.*\bafter\b.*`[A-Za-z_]\w*\s*=",
        source, re.IGNORECASE,
    )
    builtin_logging = re.search(
        r"^Use\s+`print\(\)`\s+to\b", source, re.IGNORECASE,
    )
    implementation_recipe = re.search(
        r"^(?:Use|Call|Invoke)\s+`(?:self\.|super\(\)\.|parser\.|environment\.)"
        r"[^`]*\([^`]*\)(?:\.[A-Za-z_]\w*)?`\s+"
        r"(?:with\b|for\b|to\b|in\b|on\b|then\b)",
        source, re.IGNORECASE,
    )
    internal_storage = re.search(
        r"^(?:Store|Initialize)\b[^.!?]*\busing\s+"
        r"`setattr\((?:self|environment),\s*['\"]_[^`]+\)`",
        source, re.IGNORECASE,
    )
    return bool(labelled or method_owned or library_recipe or declaration
                or backing_store or type_definition or builtin_logging
                or implementation_recipe or internal_storage)


def source_only_requirement_ids(requirements):
    """Keep explicit source constraints out of runtime-call coverage.

    This is not a behavioral exemption: the assertion must execute and read
    the named public source. All clauses still pass through semantic review.
    """
    return {identity for identity, text in requirements.items() if re.search(
        r"\btype hints?\b|\btype annotations?\b|\btype overloads?\b|"
        r"\b(?:new|return|parameter) type should be\b|"
        r"\b(?:store|declare|initialize)\b[^\n]*`[A-Za-z_]\w*\s*:\s*(?:t\.|typing\.)|"
        r"\bversion(?: change)? documentation\b|\buse exactly this formula\b|"
        r"\bmodified\s+the\s+internal\s+(?:`[A-Za-z_]\w*(?:\(\))?`|[A-Za-z_]\w*(?:\(\))?)\s+"
        r"method\s+to\s+(?:`[A-Za-z_]\w*(?:\(\))?`|[A-Za-z_]\w*(?:\(\))?)|"
        r"^import (?:required\s*:|and use\s*:)|"
        r"^use\s+`(?:contextvars\.)?ContextVar`\s+to\s+store\b|"
        r"^(?:\"\"\"|''')|^[A-Za-z_]\w*\s*:\s*(?:optional|required)\b",
        str(text).strip(), re.IGNORECASE) or _internal_operation_requirement(text)}


def _parse_public_snippet(source):
    """Parse a short backtick example without inventing replacement syntax."""
    text = str(source).strip()
    if text.startswith("with ") and text.endswith(":"):
        text += "\n    pass"
    try:
        return ast.parse(text)
    except SyntaxError:
        return None


def _direct_interactions(tree, *, include_receiver_gets=False):
    """Return direct terminal-name call/get/set operations in an AST.

    Intermediate pieces of ``dspy.cache.namespace`` (for example ``cache``)
    are not independent public interactions. Dynamic aliases such as
    ``ctx_method(...)`` do not count as the exact ``namespace(...)`` call.
    """
    if tree is None:
        return set()
    parents = {id(child): parent for parent in ast.walk(tree)
               for child in ast.iter_child_nodes(parent)}
    # Type annotations describe the API's source contract, not operations a
    # caller must execute. Keep defaults and assigned values in coverage: an
    # annotated assignment can still contain a real public call on its RHS.
    annotation_nodes = set()
    for node in ast.walk(tree):
        annotation = (
            node.annotation if isinstance(node, (ast.AnnAssign, ast.arg))
            else node.returns if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            else None
        )
        if annotation is not None:
            annotation_nodes.update(id(child) for child in ast.walk(annotation))
    interactions = set()
    for node in ast.walk(tree):
        if id(node) in annotation_nodes:
            continue
        if isinstance(node, ast.Call):
            function = node.func
            if isinstance(function, ast.Attribute):
                interactions.add(("call", function.attr))
            elif isinstance(function, ast.Name):
                interactions.add(("call", function.id))
        if not isinstance(node, ast.Attribute):
            continue
        parent = parents.get(id(node))
        if isinstance(parent, ast.Attribute) and parent.value is node:
            if include_receiver_gets and isinstance(node.ctx, ast.Load):
                interactions.add(("get", node.attr))
            continue
        if isinstance(parent, ast.Call) and parent.func is node:
            continue
        if isinstance(node.ctx, ast.Store):
            interactions.add(("set", node.attr))
        elif isinstance(node.ctx, ast.Load):
            interactions.add(("get", node.attr))
    return interactions


def _implementation_owned_example(requirement, match, interactions):
    """Distinguish implementation-owned symbols from caller entry points.

    Public briefs often name a helper, callback, hook, or private method to
    explain how the feature is assembled.  A peer must verify that behavior
    through the caller-facing API; directly invoking that supporting symbol is
    neither required nor, for an internal symbol, necessarily possible.
    Explicit public/export/caller language keeps a named helper binding.
    """
    source = str(requirement)
    before = source[max(0, match.start() - 120):match.start()]
    after = source[match.end():match.end() + 120]
    explicitly_public = bool(
        re.search(
            r"(?:\b(?:expose|export)\s+(?:(?:an?|the|new)\s+)*|"
            r"\b(?:public|exported|caller-facing|user-facing)(?:\s+(?:API|entry[- ]point|"
            r"function|method|helper|callable))?\s*(?:named|called)?\s*)$|"
            r"\b(?:callers?|users?|clients?|consumers?)\s+(?:can|must|should|will)\s+"
            r"(?:call|use|import)\s*$",
            before,
            re.IGNORECASE,
        )
        or re.match(
            r"^\s*(?:(?:is|as|to be)\s+)?(?:an?\s+)?"
            r"(?:public|exported|caller-facing|user-facing)\b|"
            r"^.{0,48}\b(?:for|to)\s+(?:callers?|users?|clients?|consumers?)\b",
            after,
            re.IGNORECASE,
        )
    )
    if explicitly_public:
        return False
    snippet = match.group(1).strip()
    # An exception constructed by the product is observed by catching it.
    # Requiring the caller's check to instantiate the same exception would
    # reject correct negative tests and reward a vacuous synthetic raise.
    if re.search(r"\b(?:raise|raises|raising|throw|throws|throwing)\s+$",
                 before, re.IGNORECASE):
        return True
    # These examples tell the implementer what to do *inside* the feature.
    # The acceptance probe must exercise the public entry point and inspect
    # its result, not reproduce its constructor, RNG draw, or buffer read.
    if re.match(r"^self\.[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*\(", snippet):
        return True
    if (re.match(r"^(?:environment|parser)\.[A-Za-z_]\w*\(", snippet)
            and re.search(r"\b(?:use|call|invoke|initialize|register)\s+$", before,
                          re.IGNORECASE)):
        return True
    if (re.match(r"^(?:environment|parser)\.[A-Za-z_]\w*\(", snippet)
            and re.search(r"\badd\s+.+\s+to\s+(?:the\s+)?$", before, re.IGNORECASE)
            and re.match(r"\s*call\b", after, re.IGNORECASE)):
        # "Add default=... to the environment.extend() call" directs an
        # existing implementation call; it does not make callers repeat setup.
        return True
    if (re.match(r"^(?:environment|parser)\.[A-Za-z_]\w*\(", snippet)
            and re.search(r"\b(?:via|using)\s+$", before, re.IGNORECASE)
            and (
                re.search(r"\b(?:in|inside|within)\s+`__init__\([^`]*\)`\s*:",
                          before, re.IGNORECASE)
                or re.match(r"\s+in\s+(?:the\s+)?(?:extension's\s+)?"
                            r"`__init__(?:\([^`]*\))?`\s+method\b",
                            after, re.IGNORECASE)
            )):
        # A constructor-scoped registration recipe is exercised by creating
        # the extension and using its public API. Keep the source obligation;
        # do not require the acceptance test to repeat product initialization.
        return True
    if ("StreamResponse" in {symbol for _, symbol in interactions}
            and re.search(r"\b(?:emits?|constructs?|creates?|returns?)\s+(?:an?\s+)?$",
                          before, re.IGNORECASE)):
        return True
    if ("tell" in {symbol for _, symbol in interactions}
            and re.search(r"\buse\s+$", before, re.IGNORECASE)
            and re.search(r"\b(?:buffer|resolve_\w+|size)\b", source, re.IGNORECASE)):
        return True
    if (interactions and all(kind == "call" and symbol in {"save_state", "restore_state"}
                             for kind, symbol in interactions)
            and re.search(r"^\s*Use\s+`save_state\(\)`\s+before\s+completion\s+processing\s+and\s+"
                          r"`restore_state\(\)`\s+after", source, re.IGNORECASE)):
        return True
    # print() is the product's logging implementation, not a caller-facing
    # entry point that an acceptance probe must invoke directly.
    if interactions and all(kind == "call" and symbol == "print"
                            for kind, symbol in interactions):
        return True
    # A specified value is checked through the product's observable result.
    # The caller need not calculate that value with the identical builtin.
    if (interactions and all(kind == "call" and symbol in {"min", "max", "len", "abs", "round"}
                             for kind, symbol in interactions)
            and re.search(r"\b(?:set to|equal to|computed as|calculated as)\s*$", before,
                          re.IGNORECASE)):
        return True
    if any(symbol.startswith("_") and not symbol.startswith("__")
           for _, symbol in interactions):
        return True
    supporting_before = re.search(
        r"\b(?:internal|private|implementation(?:-only)?|supporting|utility|helper)\s+"
        r"(?:(?:function|method|callable|routine|callback|hook|handler|observer|validator)\s+)?"
        r"(?:named|called)?\s*$|"
        r"\b(?:the\s+)?(?:implementation|internals?|pipeline|code)\s+"
        r"(?:(?:must|should|will)\s+)?(?:uses?|calls?|invokes?|delegates(?:\s+to)?|"
        r"routes?\s+through)\s*$|"
        r"\brefactor(?:ed|s|ing)?\b.{0,64}\b(?:use|uses|using|call|calls|calling|through)\s*$",
        before,
        re.IGNORECASE,
    )
    supporting_after = re.match(
        r"^\s*(?:(?:is|as)\s+)?(?:an?\s+)?"
        r"(?:internal|private|implementation(?:-only)?|supporting|helper|utility|"
        r"callback|hook|handler|observer|validator)\b|"
        r"^\s*(?:internally|under\s+the\s+hood|as\s+an?\s+implementation\s+detail)\b",
        after,
        re.IGNORECASE,
    )
    return bool(supporting_before or supporting_after)


def required_public_interactions(requirement):
    """Mechanically bind executable checks to exact public backtick examples."""
    # A public brief also describes the old implementation (for example,
    # "Cache.cache_key() today returns..."). That is review context, not a new
    # invocation the feature must expose. Bind only actionable requirement
    # language; the SDL's source/compatibility review still checks every row.
    if re.search(
        r"\b(must|required|add|extend|expose|export|support|supports|usage|getting|setting|"
        r"use|compute|prepend|helper|helpers|work|works|working|accept|apply|"
        r"shall|mandatory|never|forbidden|prohibited|may\s+not)\b|\u5fc5\u987b|\u4e0d\u5f97|\u7981\u6b62",
        str(requirement), re.IGNORECASE,
    ) is None:
        return set()
    interactions = set()
    # An actionable neighbor does not turn 'optionally expose' or 'may use'
    # into a must-call requirement. Keep the full original row and its ID.
    binding = "\n".join(required_public_clauses(str(requirement)))
    for match in re.finditer(r"`([^`]+)`", binding):
        snippet = match.group(1)
        # Markdown often quotes a repository path as code. Parsing
        # ``src/pkg/module.py`` as Python yields an Attribute named ``py`` and
        # used to create an impossible get:py acceptance obligation. A path is
        # a review/input location, not an invocation example.
        if re.fullmatch(
            r"(?:\.?\.?/)?(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+(?:\.[A-Za-z0-9_-]+)?",
            snippet.strip().replace("\\", "/"),
        ):
            continue
        tree = _parse_public_snippet(snippet)
        # A bare qualified symbol in prose (for example
        # ``contextvars.ContextVar``) names an implementation primitive. It is
        # not a public get operation. Real getter examples contain an
        # assignment/expression using the public product object; real calls
        # still have an ast.Call and remain binding.
        if (tree is not None and len(tree.body) == 1
                and isinstance(tree.body[0], ast.Expr)
                and isinstance(tree.body[0].value, (ast.Name, ast.Attribute))):
            continue
        direct = _direct_interactions(tree)
        if _implementation_owned_example(binding, match, direct):
            continue
        interactions.update(direct)
    return interactions


def _public_keyword_parameters(text: Any) -> set[str]:
    """Extract explicit public keyword/argument names without reading code.

    This deliberately accepts only identifiers inside backticks that the
    surrounding public prose calls a parameter, argument, or keyword.  It is
    used to require an interaction check when two already-visible features add
    options to the same public call; it never infers names from candidate or
    evaluator source.
    """

    source = str(text or "")
    found: set[str] = set()
    for match in re.finditer(r"`([^`]+)`", source):
        token = match.group(1).strip()
        identifier = re.match(r"^([A-Za-z_]\w*)\s*(?::|$)", token)
        if identifier is None or "(" in token:
            continue
        following = source[match.end():match.end() + 40]
        if re.match(
            r"^[\s,]*(?:parameter|argument|keyword)\b",
            following,
            re.IGNORECASE,
        ):
            found.add(identifier.group(1))
    return found


def _public_callable_names(text: Any) -> set[str]:
    """Extract explicitly named public callables from public prose only."""

    source = str(text or "")
    parameters = _public_keyword_parameters(source)
    found: set[str] = set()
    for match in re.finditer(r"`([^`]+)`", source):
        token = match.group(1).strip()
        constructor = re.fullmatch(r"(?:[A-Za-z_]\w*\.)*([A-Za-z_]\w*)\.__init__(?:\(.*\))?", token)
        if constructor is not None:
            found.add(constructor.group(1))
            continue
        called = re.fullmatch(r"(?:[A-Za-z_]\w*\.)*([A-Za-z_]\w*)\(.*\)", token)
        if called is not None:
            name = called.group(1)
            if _implementation_owned_example(source, match, {("call", name)}):
                continue
            if name == "__init__" and ".__init__" in token:
                # Class.__init__(...) is invoked by Class(...), not by a
                # free-standing __init__ call. Keep the explicitly named class
                # so an unrelated constructor cannot satisfy the interaction.
                name = token.split(".__init__", 1)[0].rsplit(".", 1)[-1]
            found.add(name)
            continue
        bare = re.fullmatch(r"([A-Za-z_]\w*)", token)
        if (bare is None or bare.group(1) in parameters
                or not bare.group(1)[:1].islower()):
            continue
        nearby = source[max(0, match.start() - 32):match.end() + 48]
        if re.search(r"\b(?:method|methods|function|functions|resolver|resolvers|call|calling)\b",
                     nearby, re.IGNORECASE):
            found.add(bare.group(1))
    return found


def _callable_keyword_parameters(text: Any) -> dict[str, set[str]]:
    """Bind options to their declared callable, not every method in a brief.

    A constructor's options must not be imposed on downstream get/list calls.
    Unbound/internal parameters still receive ordinary source/semantic review.
    """
    source = str(text or "")
    callables = _public_callable_names(source)
    parameters = _public_keyword_parameters(source)
    if len(callables) == 1:
        return {next(iter(callables)): parameters}
    bound: dict[str, set[str]] = {}
    context: set[str] = set()
    for line in source.splitlines():
        named = _public_callable_names(line)
        if re.match(r"^\s*(?:\d+\.|#{1,6}\s)", line):
            context = named
        options = _public_keyword_parameters(line)
        for name in named or context:
            bound.setdefault(name, set()).update(options)
    return bound


def _feature_is_merged(world: Any, feature_id: str) -> bool:
    repo = getattr(getattr(world, "repo_system", None), "repo", None)
    for pr in (getattr(repo, "pull_requests", {}) or {}).values():
        status = str(getattr(getattr(pr, "status", ""), "value", getattr(pr, "status", "")) or "")
        linked = set(getattr(pr, "linked_issue_ids", []) or [])
        linked.add(getattr(pr, "linked_issue", None))
        if status.casefold() == "merged" and feature_id in linked:
            return True
    return False


def _protected_integration_contract(world: Any, reviewer_id: str, feature_id: str,
                                    paths: list[str], current_contract: Mapping[str, Any]) -> dict:
    """Compile public-only pair interactions against already-merged features.

    A first feature must not be tested against an API that does not exist yet.
    A later overlapping feature, however, is reviewed on top of protected main.
    If both public briefs add keyword options to an explicitly shared callable,
    require the actual peer to exercise those options in the same call. Other
    shared-callable changes still need source-level coexistence review; they
    must not disappear merely because one brief changes lifecycle, ordering,
    or a positional signature instead of adding a keyword.
    """

    state = getattr(world, "_cooperbench_sdl_state", {}) or {}
    owners = state.get("feature_owners") or {}
    feature_paths = state.get("feature_paths") or {}
    current_text = str(current_contract.get("description") or "")
    current_parameters = _public_keyword_parameters(current_text)
    current_callables = _public_callable_names(current_text)
    current_bindings = _callable_keyword_parameters(current_text)
    if not current_callables:
        return {}
    current_paths = {str(path).replace("\\", "/") for path in paths}
    protected: list[dict[str, Any]] = []
    for other in sorted(str(item) for item in owners if str(item) != str(feature_id)):
        shared_paths = sorted(
            current_paths
            & {str(path).replace("\\", "/") for path in (feature_paths.get(other) or [])}
        )
        if not shared_paths or not _feature_is_merged(world, other):
            continue
        visible = visible_review_contract(world, reviewer_id, other)
        if visible is None:
            continue
        other_text = str(visible.get("description") or "")
        protected_parameters = _public_keyword_parameters(other_text)
        shared_callables = sorted(current_callables & _public_callable_names(other_text))
        if not shared_callables:
            continue
        other_bindings = _callable_keyword_parameters(other_text)
        keyword_callables = [name for name in shared_callables
                            if current_bindings.get(name) and other_bindings.get(name)]
        scoped_current = set().union(*(current_bindings[name] for name in keyword_callables))
        scoped_protected = set().union(*(other_bindings[name] for name in keyword_callables))
        row = {
            "feature_id": other,
            "public_request": other_text,
            "keyword_parameters": sorted(scoped_protected or protected_parameters),
            "shared_callables": keyword_callables or shared_callables,
            "shared_paths": shared_paths,
            "runtime_probe_mode": (
                "same_call_keywords"
                if keyword_callables
                else "source_trace_only"
            ),
        }
        if keyword_callables and scoped_current != current_parameters:
            row["current_keyword_parameters"] = sorted(scoped_current)
        protected.append(row)
    if not protected:
        return {}
    return {
        "schema_version": "cooperbench_public_cross_feature_interaction_v2",
        "current_feature_id": str(feature_id),
        "current_keyword_parameters": sorted(current_parameters),
        "protected_features": protected,
    }


def _invoked_public_operations(tree):
    """Recognize explicit calls, public registries and rendered template filters.

    Only calls carry credit. Merely mentioning a symbol in a string/comment
    does not; runtime/source provenance and assertions are checked separately.
    """
    templates = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    templates[target.id] = node.value
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else ""
        if (isinstance(func, ast.Subscript) and isinstance(func.value, ast.Attribute)
                and func.value.attr in {"filters", "tests", "globals"}
                and isinstance(func.slice, ast.Constant) and isinstance(func.slice.value, str)):
            name = func.slice.value
        if name in {"call_filter", "call_test"} and node.args and isinstance(node.args[0], ast.Constant):
            name = node.args[0].value
        yield name, {kw.arg for kw in node.keywords if isinstance(kw.arg, str)}
        if not isinstance(func, ast.Attribute) or func.attr not in {"render", "generate", "stream"}:
            continue
        factory = templates.get(func.value.id) if isinstance(func.value, ast.Name) else func.value
        if (not isinstance(factory, ast.Call) or not isinstance(factory.func, ast.Attribute)
                or factory.func.attr != "from_string" or not factory.args
                or not isinstance(factory.args[0], ast.Constant)
                or not isinstance(factory.args[0].value, str)):
            continue
        # Parse the public template syntax, not a benchmark-specific feature.
        # Jinja is already part of the runtime; an unavailable parser grants no credit.
        try:
            import jinja2
        except ImportError:
            continue
        try:
            parsed = jinja2.Environment().parse(factory.args[0].value)
        except (ValueError, jinja2.TemplateSyntaxError):
            continue
        for operation in parsed.find_all((jinja2.nodes.Filter, jinja2.nodes.Test)):
            yield operation.name, {kw.key for kw in operation.kwargs}


def _cross_feature_interactions_missing(probes: list[Mapping[str, Any]],
                                        contract: Mapping[str, Any] | None) -> list[str]:
    if not isinstance(contract, Mapping) or not contract.get("protected_features"):
        return []
    current_parameters = set(contract.get("current_keyword_parameters") or [])
    missing: list[str] = []
    for protected in contract.get("protected_features") or []:
        if not isinstance(protected, Mapping):
            continue
        protected_parameters = set(protected.get("keyword_parameters") or [])
        required_current = set(protected.get("current_keyword_parameters", current_parameters))
        shared_callables = set(protected.get("shared_callables") or [])
        if (
            protected.get("runtime_probe_mode") == "source_trace_only"
            or not required_current
            or not protected_parameters
        ):
            continue
        covered = False
        for probe in probes:
            if probe.get("kind", "runtime") != "runtime":
                continue
            try:
                tree = ast.parse(str(probe.get("code") or ""))
            except (SyntaxError, TypeError, ValueError):
                continue
            for terminal, keywords in _invoked_public_operations(tree):
                if (terminal in shared_callables
                        and keywords & required_current
                        and keywords & protected_parameters):
                    covered = True
                    break
            if covered:
                break
        if not covered:
            missing.append(
                ":".join([
                    str(protected.get("feature_id") or "protected"),
                    ",".join(sorted(shared_callables)),
                    ",".join(sorted(protected_parameters)),
                    ",".join(sorted(current_parameters)),
                ])
            )
    return missing


def required_atomic_observations(requirement: Any) -> list[str]:
    """Compile conservative observable-operation gates from public prose.

    A callback/hook that reports the final returned value's size is observable
    without guessing how the product transforms that value.  A concrete trim
    oracle is stronger: it is allowed only when the public request itself names
    the trim direction or boundary characters.  Merely saying "trimming" does
    not authorize the host or a generated peer probe to invent ``strip(NUL)``.
    No task id, hidden test, or candidate implementation participates here.
    """

    text = str(requirement or "")
    has_trim = re.search(r"\b(?:trim(?:s|med|ming)?|strip(?:s|ped|ping)?)\b", text, re.IGNORECASE)
    has_observer = re.search(
        r"\b(?:callback|hook|observer|listener|handler|notif(?:y|ication)|metric)\w*\b",
        text,
        re.IGNORECASE,
    )
    final_measurement = re.search(
        r"\bfinal\b.{0,40}\b(?:byte\s+)?(?:length|size|count)\b",
        text,
        re.IGNORECASE,
    )
    if not (has_observer and final_measurement):
        return []
    explicit_trim = has_trim and (
        re.search(r"\b(?:lstrip|rstrip|strip)\s*\(", text, re.IGNORECASE)
        or (
            re.search(r"\b(?:leading|trailing|both\s+ends?|two[- ]sided)\b", text, re.IGNORECASE)
            and re.search(r"\b(?:nul|null|zero|whitespace|space|tab|newline|character|byte)s?\b", text, re.IGNORECASE)
        )
    )
    return [
        "trim_then_final_observer"
        if explicit_trim
        else "final_observer_matches_returned_value"
    ]


def _terminal_call_name(call: ast.Call) -> str:
    function = call.func
    if isinstance(function, ast.Attribute):
        return function.attr
    if isinstance(function, ast.Name):
        return function.id
    return ""


def _names_in(node: ast.AST) -> set[str]:
    return {item.id for item in ast.walk(node) if isinstance(item, ast.Name)}


def _targets_in(node: ast.AST) -> set[str]:
    targets = list(node.targets) if isinstance(node, ast.Assign) else [node.target]

    def unpack(target: ast.AST) -> set[str]:
        if isinstance(target, ast.Name):
            return {target.id}
        if isinstance(target, (ast.Tuple, ast.List)):
            return {name for item in target.elts for name in unpack(item)}
        return set()

    return {name for target in targets for name in unpack(target)}


def _trim_sensitive_literal(node: ast.AST) -> bool:
    if not isinstance(node, ast.Constant):
        return False
    value = node.value
    if isinstance(value, bytes):
        trimmed = value.strip(b"\x00 \t\r\n")
    elif isinstance(value, str):
        trimmed = value.strip("\x00 \t\r\n")
    else:
        return False
    return bool(trimmed) and trimmed != value


def _padding_only_literal(node: ast.AST) -> bool:
    if not isinstance(node, ast.Constant):
        return False
    value = node.value
    if isinstance(value, bytes):
        return bool(value) and not value.strip(b"\x00 \t\r\n")
    if isinstance(value, str):
        return bool(value) and not value.strip("\x00 \t\r\n")
    return False


def _payload_literal(node: ast.AST) -> bool:
    if not isinstance(node, ast.Constant):
        return False
    value = node.value
    if isinstance(value, bytes):
        return bool(value.strip(b"\x00 \t\r\n"))
    if isinstance(value, str):
        return bool(value.strip("\x00 \t\r\n"))
    return False


def _trim_sensitive_expression(
    node: ast.AST,
    payload_name_kinds: Mapping[str, set[type]],
    padding_name_kinds: Mapping[str, set[type]],
) -> bool:
    """Recognize a nonempty payload with observable boundary padding.

    Generated probes commonly build fixtures as separate constants, for
    example ``padding_start + core_audio + padding_end``. Requiring either one
    mixed literal or inline padding literals rejected that valid construction
    when the padding constants were first assigned names. Keep this static and
    conservative: propagate byte/text kinds for padding and payload aliases
    separately, then require both kinds to meet in the same expression.
    """

    descendants = list(ast.walk(node))
    if any(_trim_sensitive_literal(item) for item in descendants):
        return True
    padding_kinds = {
        type(item.value)
        for item in descendants
        if _padding_only_literal(item)
    }
    payload_kinds = {
        type(item.value)
        for item in descendants
        if _payload_literal(item)
    }
    for name in _names_in(node):
        payload_kinds.update(payload_name_kinds.get(name, set()))
        padding_kinds.update(padding_name_kinds.get(name, set()))
    return bool(padding_kinds & payload_kinds)


def _contains_trim_call(node: ast.AST, source_names: set[str]) -> bool:
    for item in ast.walk(node):
        if not isinstance(item, ast.Call) or not isinstance(item.func, ast.Attribute):
            continue
        if item.func.attr not in {"strip", "lstrip", "rstrip"}:
            continue
        if _trim_sensitive_literal(item.func.value) or _names_in(item.func.value) & source_names:
            return True
    return False


def _direct_name_alias(node: ast.AST, names: set[str]) -> bool:
    return isinstance(node, ast.Name) and node.id in names


def _direct_trim_value(node: ast.AST, source_names: set[str]) -> bool:
    """Accept the trim result itself, not an expression with a trim alternative."""

    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"strip", "lstrip", "rstrip"}
        and (
            _trim_sensitive_literal(node.func.value)
            or bool(_names_in(node.func.value) & source_names)
        )
    )


def _direct_equality_sides(node: ast.AST) -> tuple[ast.AST, ast.AST] | None:
    """Return one exact equality, never an alternative-oracle expression."""

    if (
        not isinstance(node, ast.Compare)
        or len(node.ops) != 1
        or not isinstance(node.ops[0], ast.Eq)
        or len(node.comparators) != 1
    ):
        return None
    left, right = node.left, node.comparators[0]
    if any(
        isinstance(item, (ast.BoolOp, ast.IfExp))
        for side in (left, right)
        for item in ast.walk(side)
    ):
        return None
    return left, right


def _direct_final_length_value(
    node: ast.AST,
    final_value_names: set[str],
    final_length_names: set[str],
) -> bool:
    """Recognize exactly ``len(final_value)`` or its one-value container/alias."""

    if _direct_name_alias(node, final_length_names):
        return True
    if isinstance(node, ast.Call):
        return (
            isinstance(node.func, ast.Name)
            and node.func.id == "len"
            and len(node.args) == 1
            and not node.keywords
            and bool(_names_in(node.args[0]) & final_value_names)
        )
    if isinstance(node, (ast.List, ast.Tuple)) and len(node.elts) == 1:
        return _direct_final_length_value(
            node.elts[0],
            final_value_names,
            final_length_names,
        )
    return False


def _under_conditional(node: ast.AST, parents: Mapping[int, ast.AST]) -> bool:
    """A branch-local assertion does not constrain the alternative branch."""

    parent = parents.get(id(node))
    while parent is not None:
        if isinstance(parent, (ast.If, ast.IfExp, ast.Match)):
            return True
        parent = parents.get(id(parent))
    return False


def _trim_then_final_observer_assertion_lines(tree: ast.AST) -> set[int]:
    """Return the exact assertions that jointly observe the ordered clause.

    A row is admitted only when it observes both the trimmed return value and
    the final value delivered to the observer.  Preserve the contributing
    assertion lines as well as the admission decision so a later failed
    execution can be bound to the same mechanically validated public
    obligation.  This deliberately does not authorize other assertions that
    happen to share the row.
    """

    parents = {
        id(child): parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    assignments = [
        node for node in ast.walk(tree)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
    ]
    functions = {
        node.name: node for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    payload_name_kinds: dict[str, set[type]] = {}
    padding_name_kinds: dict[str, set[type]] = {}
    sensitive_names: set[str] = set()
    # Probe fixtures are ordinary executable Python, so a referenced local has
    # already been assigned. Process writes in source order and replace the
    # target's provenance on reassignment. A global fixed-point union would
    # falsely combine ``value = padding; value = payload`` into one imaginary
    # mixed value and weaken the gate we are trying to make alias-aware.
    for assignment in sorted(
        assignments,
        key=lambda item: (getattr(item, "lineno", 0), getattr(item, "col_offset", 0)),
    ):
        payload_kinds = {
            type(item.value)
            for item in ast.walk(assignment.value)
            if _payload_literal(item)
        }
        padding_kinds = {
            type(item.value)
            for item in ast.walk(assignment.value)
            if _padding_only_literal(item) or _trim_sensitive_literal(item)
        }
        for name in _names_in(assignment.value):
            payload_kinds.update(payload_name_kinds.get(name, set()))
            padding_kinds.update(padding_name_kinds.get(name, set()))
        is_sensitive = _trim_sensitive_expression(
            assignment.value,
            payload_name_kinds,
            padding_name_kinds,
        )
        for target in _targets_in(assignment):
            payload_name_kinds[target] = set(payload_kinds)
            padding_name_kinds[target] = set(padding_kinds)
            if is_sensitive:
                sensitive_names.add(target)
            else:
                sensitive_names.discard(target)

    # Follow simple public-fixture construction such as
    # padded -> base64.b64encode(padded) -> AudioBlock(audio=encoded).
    fixture_names = set(sensitive_names)
    changed = True
    while changed:
        changed = False
        for assignment in assignments:
            if _names_in(assignment.value) & fixture_names:
                before = len(fixture_names)
                fixture_names.update(_targets_in(assignment))
                changed = changed or len(fixture_names) != before

    expected_names: set[str] = set()
    changed = True
    while changed:
        changed = False
        for assignment in assignments:
            if not (
                _direct_trim_value(assignment.value, fixture_names)
                or _direct_name_alias(assignment.value, expected_names)
            ):
                continue
            before = len(expected_names)
            expected_names.update(_targets_in(assignment))
            changed = changed or len(expected_names) != before

    observed_results: set[str] = set()
    observer_state: set[str] = set()
    call_lines: list[int] = []
    for assignment in assignments:
        value = assignment.value
        if not isinstance(value, ast.Call) or not _terminal_call_name(value).startswith("resolve"):
            continue
        observer_keywords = [
            keyword for keyword in value.keywords
            if keyword.arg and re.search(r"(?:callback|hook|observer|listener|handler|^on_)", keyword.arg)
        ]
        if not observer_keywords or not (_names_in(value) & fixture_names):
            continue
        observed_results.update(_targets_in(assignment))
        call_lines.append(getattr(value, "lineno", 0))
        for keyword in observer_keywords:
            argument_names = _names_in(keyword.value)
            observer_state.update(argument_names)
            for name in argument_names:
                function = functions.get(name)
                if function is not None:
                    parameters = {arg.arg for arg in function.args.args}
                    observer_state.update(_names_in(function) - parameters - {name})

    if not observed_results or not call_lines or not observer_state:
        return set()

    # Track result.read()/getvalue() and later aliases used by assertions.
    output_names = set(observed_results)
    changed = True
    while changed:
        changed = False
        for assignment in assignments:
            value = assignment.value
            direct_read = (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Attribute)
                and value.func.attr in {"read", "getvalue"}
                and bool(_names_in(value.func.value) & output_names)
            )
            if not (direct_read or _direct_name_alias(value, output_names)):
                continue
            before = len(output_names)
            output_names.update(_targets_in(assignment))
            changed = changed or len(output_names) != before

    final_value_names = output_names | expected_names
    final_length_names: set[str] = set()
    changed = True
    while changed:
        changed = False
        for assignment in assignments:
            if not (
                _direct_final_length_value(
                    assignment.value,
                    final_value_names,
                    final_length_names,
                )
            ):
                continue
            before = len(final_length_names)
            final_length_names.update(_targets_in(assignment))
            changed = changed or len(final_length_names) != before

    return_assertion_lines: set[int] = set()
    observer_assertion_lines: set[int] = set()
    first_call_line = min(call_lines)
    for assertion in (node for node in ast.walk(tree) if isinstance(node, ast.Assert)):
        line = getattr(assertion, "lineno", 0)
        if line <= first_call_line or _under_conditional(assertion, parents):
            continue
        sides = _direct_equality_sides(assertion.test)
        if sides is None:
            continue
        left, right = sides
        left_names, right_names = _names_in(left), _names_in(right)
        if (
            (
                left_names & output_names
                and (
                    right_names & expected_names
                    or _contains_trim_call(right, fixture_names)
                )
            )
            or (
                right_names & output_names
                and (
                    left_names & expected_names
                    or _contains_trim_call(left, fixture_names)
                )
            )
        ):
            return_assertion_lines.add(line)
        if (
            (
                left_names & observer_state
                and _direct_final_length_value(
                    right,
                    final_value_names,
                    final_length_names,
                )
            )
            or (
                right_names & observer_state
                and _direct_final_length_value(
                    left,
                    final_value_names,
                    final_length_names,
                )
            )
        ):
            observer_assertion_lines.add(line)
    if not (sensitive_names and expected_names and return_assertion_lines and observer_assertion_lines):
        return set()
    return return_assertion_lines | observer_assertion_lines


def _probe_observes_trim_then_final_observer(tree: ast.AST) -> bool:
    """Require one check to make the whole ordered public clause observable."""

    return bool(_trim_then_final_observer_assertion_lines(tree))


def _final_observer_assertion_lines(tree: ast.AST) -> set[int]:
    """Return assertions tying an explicit observer to the actual returned value.

    This deliberately does not prescribe a trim operator, padding byte, or
    expected product value.  It proves only the public fact that the observer
    receives the length of the value the resolver actually returns.
    """

    parents = {
        id(child): parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
    ]
    functions = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    observed_results: set[str] = set()
    observer_state: set[str] = set()
    call_lines: list[int] = []
    for assignment in assignments:
        value = assignment.value
        if not isinstance(value, ast.Call) or not _terminal_call_name(value).startswith("resolve"):
            continue
        observer_keywords = [
            keyword
            for keyword in value.keywords
            if keyword.arg
            and re.search(
                r"(?:callback|hook|observer|listener|handler|^on_)", keyword.arg
            )
        ]
        if not observer_keywords:
            continue
        observed_results.update(_targets_in(assignment))
        call_lines.append(getattr(value, "lineno", 0))
        for keyword in observer_keywords:
            argument_names = _names_in(keyword.value)
            observer_state.update(argument_names)
            for name in argument_names:
                function = functions.get(name)
                if function is not None:
                    parameters = {arg.arg for arg in function.args.args}
                    observer_state.update(_names_in(function) - parameters - {name})
    if not observed_results or not observer_state or not call_lines:
        return set()

    output_names = set(observed_results)
    changed = True
    while changed:
        changed = False
        for assignment in assignments:
            value = assignment.value
            direct_read = (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Attribute)
                and value.func.attr in {"read", "getvalue"}
                and bool(_names_in(value.func.value) & output_names)
            )
            if not (direct_read or _direct_name_alias(value, output_names)):
                continue
            before = len(output_names)
            output_names.update(_targets_in(assignment))
            changed = changed or len(output_names) != before

    final_length_names: set[str] = set()
    changed = True
    while changed:
        changed = False
        for assignment in assignments:
            if not _direct_final_length_value(
                assignment.value, output_names, final_length_names
            ):
                continue
            before = len(final_length_names)
            final_length_names.update(_targets_in(assignment))
            changed = changed or len(final_length_names) != before

    assertion_lines: set[int] = set()
    first_call_line = min(call_lines)
    for assertion in (node for node in ast.walk(tree) if isinstance(node, ast.Assert)):
        line = getattr(assertion, "lineno", 0)
        if line <= first_call_line or _under_conditional(assertion, parents):
            continue
        sides = _direct_equality_sides(assertion.test)
        if sides is None:
            continue
        left, right = sides
        left_names, right_names = _names_in(left), _names_in(right)
        if (
            left_names & observer_state
            and _direct_final_length_value(right, output_names, final_length_names)
        ) or (
            right_names & observer_state
            and _direct_final_length_value(left, output_names, final_length_names)
        ):
            assertion_lines.add(line)
    return assertion_lines


def authorized_atomic_failure(
    probe: Mapping[str, Any],
    mapped_requirements: Mapping[str, str],
    failure_site: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Bind a reached failed assertion to an already validated public clause.

    This is a narrow host-side authority, not a product verdict.  It only says
    that the failed *probe assertion* is a legitimate check of the public
    requirement, so a fallible semantic reviewer may not discard the failure
    as an unauthorized test.  Candidate correctness still depends on making
    the probe pass and, later, on Cooper's official evaluator.
    """

    if failure_site.get("operation_phase") != "probe_assertion":
        return None
    failed_lines = {
        line for line in failure_site.get("probe_lines", [])
        if isinstance(line, int) and line > 0
    }
    if not failed_lines:
        return None
    try:
        tree = ast.parse(str(probe.get("code") or ""))
    except (SyntaxError, ValueError, TypeError):
        return None
    for identity in probe.get("requirement_ids", []):
        if identity not in mapped_requirements:
            continue
        public_basis = str(mapped_requirements[identity])
        for observation in required_atomic_observations(public_basis):
            assertion_lines = (
                _trim_then_final_observer_assertion_lines(tree)
                if observation == "trim_then_final_observer"
                else _final_observer_assertion_lines(tree)
            )
            matched_lines = sorted(failed_lines & assertion_lines)
            if matched_lines:
                return {
                    "requirement_id": identity,
                    "public_basis": public_basis,
                    "observation": observation,
                    "assertion_lines": sorted(assertion_lines),
                    "failed_lines": matched_lines,
                }
    return None


def _unsupported_atomic_observations(
    tree: ast.AST,
    requirement_ids: list[str],
    requirements: Mapping[str, str],
) -> list[tuple[str, str]]:
    unsupported: list[tuple[str, str]] = []
    for identity in requirement_ids:
        for observation in required_atomic_observations(requirements.get(identity, "")):
            if observation == "trim_then_final_observer" and not (
                _probe_observes_trim_then_final_observer(tree)
            ):
                unsupported.append((identity, "trimmed_return_and_final_observer"))
            elif observation == "final_observer_matches_returned_value":
                if _probe_observes_trim_then_final_observer(tree):
                    unsupported.append(
                        (identity, "unstated_transform_semantics_must_not_be_asserted")
                    )
                elif not _final_observer_assertion_lines(tree):
                    unsupported.append(
                        (identity, "returned_value_and_final_observer")
                    )
    return unsupported


def _baseline_comparison_gap(
    retained_probes: list[Mapping[str, Any]],
    compatibility: Mapping[str, str],
) -> dict[str, Any]:
    """Describe the global comparison obligation a row repair must preserve.

    A generated check can correctly be changed from differential to
    candidate-only when it calls a newly introduced API.  If it was the only
    comparison row, however, that local correction also leaves the plan without
    evidence for unchanged/default behavior.  Make the coupled obligation
    explicit in the *same* retry instead of spending another model round only
    to discover it during whole-plan validation.
    """

    if not compatibility or any(
        row.get("compare_baseline") is True for row in retained_probes
    ):
        return {}
    return {
        "requirement_ids": sorted(str(identity) for identity in compatibility),
        "instruction": (
            "Return an additional compare_baseline=true probe for at least one "
            "listed compatibility id. It must use only public operations that "
            "exist in the untouched baseline. If a replacement exercises a new "
            "API (including a new reset/default hook), keep that replacement "
            "candidate-only and return the baseline-safe default/legacy check as "
            "a separate new row in this same response."
        ),
    }


def _response_diagnostic(
    answer,
    requirements,
    *,
    feature_id="draft",
    paths=None,
    compatibility=None,
    integration_contract=None,
):
    """Bounded draft + observed shape, not an opaque validation error."""
    from society_core.code_landing.event_store import redact_sensitive_payload
    probes = answer.get("probes") if isinstance(answer, Mapping) else None
    covered, retained, invalid = set(), [], []
    seen_ids = set()
    for index, row in enumerate(probes if isinstance(probes, list) else []):
        identity = row.get("probe_id") if isinstance(row, Mapping) else None
        if (not isinstance(identity, str) or not re.fullmatch(re.escape(feature_id) + r":\d+", identity)
                or identity in seen_ids):
            serial = index
            while f"{feature_id}:{serial}" in seen_ids:
                serial += 1
            identity = f"{feature_id}:{serial}"
        seen_ids.add(identity)
        try:
            normalized = _validate_plan(
                {"probes": [row]}, feature_id=feature_id, requirements=requirements,
                compatibility={}, paths=paths if paths is not None else row.get("paths", []),
                _partial=True,
            )[0]
            normalized["probe_id"] = identity
            retained.append(normalized)
            covered.update(normalized["requirement_ids"])
        except (ValueError, AttributeError, TypeError) as exc:
            invalid.append({"probe_id": identity, "error": str(exc)})
    source_ids = source_only_requirement_ids(requirements)
    observed = {identity: set() for identity in requirements}
    rows_by_requirement = {identity: [] for identity in requirements}
    for row in retained:
        interactions = _direct_interactions(
            ast.parse(row["code"]), include_receiver_gets=True,
        )
        for identity in row["requirement_ids"]:
            observed[identity].update(interactions)
            rows_by_requirement[identity].append((row, interactions))
    missing_interactions = {}
    interaction_repairs = set()
    for identity, requirement in requirements.items():
        if identity in source_ids and (
            not _internal_operation_requirement(requirement)
            or any(row.get("kind") == "source" for row, _ in rows_by_requirement[identity])
        ):
            continue
        missing = sorted(required_public_interactions(requirement) - observed[identity])
        if not missing:
            continue
        missing_interactions[identity] = [f"{kind}:{symbol}" for kind, symbol in missing]
        candidates = rows_by_requirement[identity]
        # Prefer repairing the row that already reads/inspects the exact
        # symbol. For example inspect.signature(configure_cache) observes a
        # get but does not execute the required call. Do not regenerate an
        # unrelated request_cache row carrying the same combined req id.
        related = [row for row, interactions in candidates
                   if any(any(seen_symbol == missing_symbol for _, seen_symbol in interactions)
                          for _, missing_symbol in missing)]
        targets = related or ([candidates[0][0]] if candidates else [])
        for row in targets:
            interaction_repairs.add(row["probe_id"])
            invalid.append({"probe_id": row["probe_id"], "error":
                            "probe_exact_public_interaction_missing:" + identity + ":"
                            + ",".join(missing_interactions[identity])})
    if interaction_repairs:
        retained = [row for row in retained if row["probe_id"] not in interaction_repairs]
        covered = {identity for row in retained for identity in row["requirement_ids"]}
    diagnostic = {
        "response_type": type(answer).__name__,
        "response_keys": sorted(str(key)[:80] for key in answer)[:20] if isinstance(answer, Mapping) else [],
        "probes_type": type(probes).__name__,
        "probe_count": len(probes) if isinstance(probes, list) else None,
        "missing_requirements": {i: text for i, text in requirements.items()
                                 if i not in covered and not purely_optional_requirement(text)},
        "invalid_probes": invalid,
        "missing_interactions": missing_interactions,
    }
    baseline_gap = _baseline_comparison_gap(
        retained,
        compatibility if isinstance(compatibility, Mapping) else {},
    )
    if baseline_gap:
        diagnostic["missing_baseline_comparison"] = baseline_gap
    missing_cross_feature = _cross_feature_interactions_missing(
        retained,
        integration_contract if isinstance(integration_contract, Mapping) else {},
    )
    if missing_cross_feature:
        diagnostic["missing_cross_feature_interactions"] = missing_cross_feature
    raw = json.dumps(answer, ensure_ascii=False)
    if len(raw) <= MAX_DRAFT_CHARS:
        diagnostic["previous_response"] = answer
        # Preserve every independently valid row so the peer only repairs the
        # actual gap.  The response itself was bounded above and each retained
        # row already passed the per-code/per-setup limits; applying a second
        # aggregate ceiling here can make an immutable repair impossible.
        if (len(retained) <= MAX_DRAFT_PROBES
                and not (len(retained) == MAX_DRAFT_PROBES and {
                    i for i, text in requirements.items() if not purely_optional_requirement(text)
                } - covered)):
            diagnostic["retained_probes"] = retained
            diagnostic["repair_probe_ids"] = [row["probe_id"] for row in invalid]
    else:
        diagnostic["draft_omitted"] = "response_exceeds_bounded_retry_context"
        diagnostic["response_chars"] = len(raw)
    return redact_sensitive_payload(diagnostic)


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def visible_review_contract(world, reviewer_id, feature_id):
    """Resolve a strict review contract before touching any cached review data."""
    from .visibility import brief_visibility_receipt, visible_feature_brief

    description = visible_feature_brief(world, reviewer_id, feature_id)
    receipt = brief_visibility_receipt(world, reviewer_id, feature_id)
    if not isinstance(description, str) or not description.strip() or not isinstance(receipt, Mapping):
        return None
    from .lifecycle import _public_compatibility_obligations
    from .public_contract import indexed_obligations
    from .semantic_review import requirement_id

    obligations = indexed_obligations(description)
    compatibility = _public_compatibility_obligations(obligations)
    identity = _digest({"reviewer_id": str(reviewer_id), "feature_id": str(feature_id),
                        "description": description, "visibility_receipt": dict(receipt),
                        "policy": "strict_explicit_brief_share_and_read_v1"})
    return {"description": description, "requirements": {requirement_id(item): item for item in obligations},
            "compatibility": {requirement_id(item): item for item in compatibility},
            "brief_visibility_receipt": dict(receipt), "review_brief_identity": identity}


def _contract_binding(contract):
    return {key: contract[key] for key in ("brief_visibility_receipt", "review_brief_identity")}


def _plan_identity(description, requirements, compatibility, paths, baseline, reviewer_id, contract,
                   integration_contract=None, *, policy=None):
    material = {"request": description, "requirements": requirements,
                "compatibility": compatibility, "paths": paths, "baseline": baseline,
                "reviewer": reviewer_id, "policy": PROBE_POLICY if policy is None else policy}
    if contract is not None:
        material.update(_contract_binding(contract))
    if integration_contract:
        material["integration_contract"] = integration_contract
    return _digest(material)


def _recorded_plan_identity(saved, behavior, description, requirements, compatibility,
                            paths, baseline, reviewer_id, contract, integration_contract=None):
    """Verify historical provenance without rebinding old checks to a new policy.

    Replay consumers must ALSO validate definitions with today's validator and
    match the original execution/approval receipts. A known old policy only
    selects the original identity calculation, never an exemption from gates.
    """
    known = {
        PROBE_POLICY,
        "peer_public_behavior_probes_v36_implementation_owned_print",
        "peer_public_behavior_probes_v37_implementation_owned_calls",
        "peer_public_behavior_probes_v38_public_dispatch_and_recipes",
        "peer_public_behavior_probes_v39_observed_exceptions_and_public_warnings",
        "peer_public_behavior_probes_v40_internal_recipes_and_raised_exceptions",
        "peer_public_behavior_probes_v41_callable_scoped_interactions",
        "peer_public_behavior_probes_v42_annotation_recipe_interactions",
        "peer_public_behavior_probes_v43_public_prerequisites",
        "peer_public_behavior_probes_v44_observer_preconditions",
        "peer_public_behavior_probes_v45_input_protocols",
    }
    recorded = saved.get("policy")
    executed = behavior.get("policy") if isinstance(behavior, Mapping) else None
    # Legacy plans bind to their actual execution receipt, not a guessed policy.
    policy = recorded or executed or PROBE_POLICY
    if not isinstance(policy, str) or policy not in known or (recorded and executed and recorded != executed):
        return None
    expected = _plan_identity(description, requirements, compatibility, paths,
                              baseline, reviewer_id, contract, integration_contract,
                              policy=policy)
    return expected if saved.get("identity") == expected else None


def _validate_plan(answer, *, feature_id, requirements, compatibility, paths,
                   integration_contract=None, _partial=False, _preserve_ids=False,
                   _enforce_probe_limit=True):
    from .probe_runner import is_assertion_guard, silent_broad_exception_lines

    probes = answer.get("probes") if isinstance(answer, Mapping) else None
    if not isinstance(probes, list) or not 1 <= len(probes) <= MAX_DRAFT_PROBES:
        raise ValueError(f"expected_one_to_{MAX_DRAFT_PROBES}_draft_probes")
    if _enforce_probe_limit and not _partial and len(probes) > PREFERRED_MAX_PROBES:
        # Compact redundant rows toward the preference, but never reject a
        # complete plan just because all remaining rows are necessary. Every
        # public coverage/interaction invariant stays mandatory.
        deduplicated: list[Mapping[str, Any]] = []
        seen: set[str] = set()
        for row in probes:
            if not isinstance(row, Mapping):
                deduplicated.append(row)
                continue
            identity = json.dumps(_probe_content(row), sort_keys=True)
            if identity in seen:
                continue
            seen.add(identity)
            deduplicated.append(row)
        normalized = _validate_plan(
            {"probes": deduplicated},
            feature_id=feature_id,
            requirements=requirements,
            compatibility=compatibility,
            paths=paths,
            integration_contract=integration_contract,
            _preserve_ids=_preserve_ids,
            _enforce_probe_limit=False,
        )
        while len(normalized) > PREFERRED_MAX_PROBES:
            removed = False
            for index in range(len(normalized) - 1, -1, -1):
                candidate = normalized[:index] + normalized[index + 1:]
                try:
                    reduced = _validate_plan(
                        {"probes": candidate},
                        feature_id=feature_id,
                        requirements=requirements,
                        compatibility=compatibility,
                        paths=paths,
                        integration_contract=integration_contract,
                        _preserve_ids=True,
                        _enforce_probe_limit=False,
                    )
                except ValueError:
                    continue
                normalized = reduced
                removed = True
                break
            if not removed:
                break
        return normalized
    source_ids = source_only_requirement_ids(requirements)
    normalized = []
    covered, compared, source_covered = set(), set(), set()
    observed_by_requirement = {identity: set() for identity in requirements}
    for index, item in enumerate(probes):
        if not isinstance(item, Mapping):
            raise ValueError("probe_row_invalid")
        ids, locations = item.get("requirement_ids"), item.get("paths")
        if not isinstance(ids, list) or not ids:
            raise ValueError("probe_requirement_ids_must_be_nonempty_list")
        if not isinstance(locations, list) or not locations:
            raise ValueError("probe_paths_must_be_nonempty_list")
        unknown_ids = [
            str(identity)
            for identity in ids
            if not isinstance(identity, str) or identity not in requirements
        ]
        unknown_paths = [
            str(path)
            for path in locations
            if not isinstance(path, str) or path not in paths
        ]
        if unknown_ids or unknown_paths:
            raise ValueError(
                "probe_requirement_or_public_path_invalid:"
                + json.dumps(
                    {
                        "unknown_requirement_ids": unknown_ids,
                        "allowed_requirement_ids": sorted(requirements),
                        "unknown_paths": unknown_paths,
                        "allowed_paths": list(paths),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
        code, setup = item.get("code"), item.get("baseline_setup", "")
        compare = item.get("compare_baseline")
        kind = item.get("kind", "runtime")
        uncredited_source_ids: list[str] = []
        if kind == "source":
            credited = [identity for identity in ids if identity in source_ids]
            uncredited_source_ids = [identity for identity in ids if identity not in source_ids]
            # A source check may reasonably inspect one docstring and an
            # adjacent runtime formula in the same AST walk. Credit only the
            # explicit source obligation. Any runtime ids are removed here and
            # must still be covered by a real runtime row below. A source row
            # with no source obligation remains invalid, so runtime behavior
            # can never be replaced by source inspection.
            if not credited:
                raise ValueError("source_probe_cannot_replace_runtime_behavior")
            ids = credited
        if kind not in {"runtime", "source"}:
            raise ValueError("source_probe_cannot_replace_runtime_behavior")
        if not isinstance(code, str):
            raise ValueError("probe_code_must_be_string")
        if not code:
            raise ValueError("probe_code_empty")
        if len(code) > MAX_PROBE_CODE_CHARS:
            raise ValueError(
                f"probe_code_too_long:{len(code)}>{MAX_PROBE_CODE_CHARS}"
            )
        if not isinstance(setup, str):
            raise ValueError("probe_baseline_setup_must_be_string")
        if len(setup) > MAX_PROBE_SETUP_CHARS:
            raise ValueError(
                f"probe_baseline_setup_too_long:{len(setup)}>{MAX_PROBE_SETUP_CHARS}"
            )
        if type(compare) is not bool:
            raise ValueError("probe_compare_baseline_must_be_boolean")
        if setup and not compare:
            raise ValueError("probe_baseline_setup_requires_compare_baseline_true")
        try:
            tree = ast.parse(code)
            nodes = list(ast.walk(tree))
            ast.parse(setup)
        except SyntaxError as exc:
            raise ValueError("probe_python_syntax_invalid") from exc
        if silent_broad_exception_lines(tree):
            raise ValueError("probe_silently_swallows_broad_exception")
        assertions = [n for n in nodes if isinstance(n, ast.Assert)]
        assertion_guards = [n for n in nodes if is_assertion_guard(n)]
        if (not assertions and not assertion_guards) or not any(isinstance(n, ast.Call) for n in nodes):
            raise ValueError("probe_requires_assertion_and_public_call")
        # A vacuous true assertion never counts as evidence, but it also must
        # not discard independent dynamic assertions/guards in the same row.
        # The isolated runner removes it before instrumenting assertion counts.
        meaningful_assertions = [n for n in assertions if not (
            isinstance(n.test, ast.Constant) and bool(n.test.value)
        )]
        if not meaningful_assertions and not assertion_guards:
            raise ValueError("probe_constant_assertion_invalid")
        unsupported_atomic = _unsupported_atomic_observations(tree, ids, requirements)
        uncredited_atomic_ids = [identity for identity, _ in unsupported_atomic]
        if uncredited_atomic_ids:
            ids = [identity for identity in ids if identity not in uncredited_atomic_ids]
            # Preserve a useful multi-requirement check and remove only the
            # unsupported atomic credit. A row whose sole claim is false must
            # be replaced under its stable probe id.
            if not ids:
                identity, observation = unsupported_atomic[0]
                raise ValueError(
                    "probe_atomic_observable_clause_missing:"
                    f"{identity}:{observation}"
                )
        covered.update(ids)
        if kind == "source":
            source_covered.update(ids)
        interactions = _direct_interactions(tree, include_receiver_gets=True)
        for identity in ids:
            observed_by_requirement[identity].update(interactions)
        if compare:
            compared.update(ids)
        normalized_row = {
            "probe_id": item.get("probe_id", f"{feature_id}:{index}") if _preserve_ids
                        else f"{feature_id}:{index}", "requirement_ids": ids,
            "paths": locations, "code": code, "baseline_setup": setup,
            "compare_baseline": compare, "kind": kind,
        }
        if uncredited_source_ids:
            normalized_row["uncredited_source_requirement_ids"] = uncredited_source_ids
        if uncredited_atomic_ids:
            normalized_row["uncredited_atomic_requirement_ids"] = uncredited_atomic_ids
        normalized.append(normalized_row)
    probe_ids = [p["probe_id"] for p in normalized]
    if (any(not isinstance(i, str) or not re.fullmatch(re.escape(feature_id) + r":\d+", i)
            for i in probe_ids) or len(set(probe_ids)) != len(probe_ids)):
        raise ValueError("probe_identity_invalid_or_duplicate")
    if _partial:
        return normalized
    compulsory_ids = {i for i, text in requirements.items() if not purely_optional_requirement(text)}
    if compulsory_ids - covered:
        raise ValueError("probe_acceptance_coverage_missing:" + ",".join(sorted(compulsory_ids - covered)))
    for identity, requirement in requirements.items():
        # Source-only clauses prove their explicit syntax/import/doc contract
        # by reading the declared public file. Calls shown inside an exact
        # implementation formula are code to inspect, not product invocations
        # that the probe itself must execute.
        if identity in source_ids and (
            not _internal_operation_requirement(requirement) or identity in source_covered
        ):
            continue
        required_interactions = required_public_interactions(requirement)
        missing_interactions = sorted(required_interactions - observed_by_requirement[identity])
        if missing_interactions:
            rendered = ",".join(f"{kind}:{symbol}" for kind, symbol in missing_interactions)
            raise ValueError(f"probe_exact_public_interaction_missing:{identity}:{rendered}")
    # Some public compatibility clauses describe a new enabled option that the
    # baseline cannot invoke. Baseline comparisons cover the default and legacy
    # paths; the remaining clauses still require executed candidate evidence.
    if compatibility and not compared.intersection(compatibility):
        raise ValueError("probe_baseline_comparison_missing")
    missing_cross_feature = _cross_feature_interactions_missing(
        normalized,
        integration_contract if isinstance(integration_contract, Mapping) else {},
    )
    if missing_cross_feature:
        raise ValueError(
            "probe_cross_feature_interaction_missing:"
            + ";".join(missing_cross_feature)
        )
    return normalized


def _probe_content(row):
    return {key: row.get(key, "runtime" if key == "kind" else "") for key in
            ("requirement_ids", "paths", "code", "baseline_setup", "compare_baseline", "kind")}


def _defect_requires_resolved_output_measurement(defect: Mapping[str, Any]) -> bool:
    """Recognize a confirmed raw-input-versus-resolved-output probe defect.

    This is public-evidence routing, not a benchmark answer.  A peer may learn
    from an executed check that its own raw input length, encoded length, or
    guessed concrete representation was not the final value named by the
    public contract.  A replacement must then measure that public result; it
    cannot merely choose different literals and repeat the same assumption.
    """

    text = " ".join(
        str(defect.get(key) or "")
        for key in ("issue", "adjudication_reason", "reason")
    ).casefold()
    measurement = re.search(
        r"\b(?:size|length|byte\s+(?:count|length|size))\b", text
    )
    # This repair contract is specifically about a resolver's produced value.
    # Generic words such as "result" and "buffer size" also appear in
    # unrelated API checks (for example a listener.receive threshold), and
    # previously routed those defects into resolver-only replacement rules.
    transformed = re.search(r"\bresolv(?:e|es|ed|er|ers|ing)\w*\b", text)
    assumed_input = re.search(
        r"\b(?:input|raw|original|hard[- ]?coded|assum(?:e|es|ed|ing|ption|ptions))\b",
        text,
    )
    return bool(measurement and transformed and assumed_input)


def _resolver_observer_keywords(code: Any) -> list[str]:
    """Return explicit observer-like keywords used on public resolver calls."""

    if not isinstance(code, str):
        return []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    observer_name = re.compile(r"(?:callback|hook|observer|on_resolve|metric|progress|notify)", re.I)
    return sorted({
        str(keyword.arg)
        for call in ast.walk(tree)
        if _is_resolver_call(call, require_unbounded=False)
        for keyword in call.keywords
        if keyword.arg and observer_name.search(keyword.arg)
    })


def _repair_constraints(
    repair_ids: list[str],
    details: Mapping[str, Any],
    previous_probes: list[Mapping[str, Any]] | None = None,
) -> dict[str, list[str]]:
    constraints: dict[str, list[str]] = {}
    prior = {
        str(row.get("probe_id") or ""): row
        for row in (previous_probes or [])
        if isinstance(row, Mapping)
    }
    defects = details.get("defects")
    if not isinstance(defects, list):
        return constraints
    for defect in defects:
        if not isinstance(defect, Mapping):
            continue
        probe_id = str(defect.get("probe_id") or "")
        if probe_id in repair_ids and _defect_requires_resolved_output_measurement(defect):
            observer_keywords = _resolver_observer_keywords(
                (prior.get(probe_id) or {}).get("code")
            )
            if observer_keywords:
                constraints[probe_id] = [
                    "same_call_resolved_output_observer_match:" + keyword
                    for keyword in observer_keywords
                ]
            else:
                constraints[probe_id] = ["resolved_output_measurement"]
    return constraints


def _assigned_names(node: ast.AST) -> set[str]:
    targets: list[ast.AST]
    if isinstance(node, (ast.Assign, ast.AnnAssign)):
        targets = list(node.targets) if isinstance(node, ast.Assign) else [node.target]
    else:
        return set()
    def names(target: ast.AST) -> set[str]:
        if isinstance(target, ast.Name):
            return {target.id}
        if isinstance(target, (ast.Tuple, ast.List)):
            return {name for item in target.elts for name in names(item)}
        return set()

    return {name for target in targets for name in names(target)}


def _assignment_value(node: ast.AST) -> ast.AST | None:
    return node.value if isinstance(node, (ast.Assign, ast.AnnAssign)) else None


def _is_resolver_call(node: ast.AST, *, require_unbounded: bool) -> bool:
    if not isinstance(node, ast.Call):
        return False
    function = node.func
    name = function.attr if isinstance(function, ast.Attribute) else function.id if isinstance(function, ast.Name) else ""
    if not name.startswith("resolve"):
        return False
    return not require_unbounded or all(keyword.arg != "max_bytes" for keyword in node.keywords)


def _contains_unbounded_resolver(node: ast.AST) -> bool:
    return any(_is_resolver_call(item, require_unbounded=True) for item in ast.walk(node))


def _references_any(node: ast.AST, names: set[str]) -> bool:
    return bool(names and any(isinstance(item, ast.Name) and item.id in names for item in ast.walk(node)))


def _is_observed_length(node: ast.AST, resolved_names: set[str]) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "len"
        and len(node.args) == 1
        and (_contains_unbounded_resolver(node.args[0]) or _references_any(node.args[0], resolved_names))
    )


def _replacement_uses_resolved_output_measurement(code: Any) -> bool:
    """Require the limit to flow from an actually resolved public result."""

    if not isinstance(code, str):
        return False
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    assignments = [node for node in ast.walk(tree) if isinstance(node, (ast.Assign, ast.AnnAssign))]
    resolved_names: set[str] = set()
    measured_names: set[str] = set()
    changed = True
    while changed:
        changed = False
        for assignment in assignments:
            value = _assignment_value(assignment)
            targets = _assigned_names(assignment)
            if value is None or not targets:
                continue
            if _is_observed_length(value, resolved_names):
                before = len(measured_names)
                measured_names.update(targets)
                changed = changed or len(measured_names) != before
            elif _contains_unbounded_resolver(value) or _references_any(value, resolved_names):
                before = len(resolved_names)
                resolved_names.update(targets)
                changed = changed or len(resolved_names) != before
    boundary_values = []
    for call in (node for node in ast.walk(tree) if _is_resolver_call(node, require_unbounded=False)):
        for keyword in call.keywords:
            if keyword.arg == "max_bytes":
                boundary_values.append(keyword.value)
    if not boundary_values or not all(
        _references_any(value, measured_names) or _is_observed_length(value, resolved_names)
        for value in boundary_values
    ):
        return False

    def observed(value: ast.AST) -> bool:
        return (
            _references_any(value, resolved_names | measured_names)
            or _is_observed_length(value, resolved_names)
        )

    def concrete(value: ast.AST) -> bool:
        return any(
            isinstance(item, ast.Constant)
            and (
                isinstance(item.value, (bytes, str))
                or isinstance(item.value, int) and not isinstance(item.value, bool) and abs(item.value) > 1
            )
            for item in ast.walk(value)
        )

    for assertion in (node for node in ast.walk(tree) if isinstance(node, ast.Assert)):
        test = assertion.test
        if isinstance(test, ast.Compare):
            left = test.left
            for operation, right in zip(test.ops, test.comparators):
                if (isinstance(operation, (ast.Eq, ast.NotEq))
                        and ((observed(left) and concrete(right))
                             or (observed(right) and concrete(left)))):
                    return False
                left = right
        if any(
            isinstance(item, ast.Constant)
            and isinstance(item.value, str)
            and re.search(r"\b\d+\s*>\s*\d+\b", item.value)
            for item in ast.walk(test)
        ):
            return False
    return True


def _callback_mutation_roots(tree: ast.AST, callback_nodes: list[ast.AST]) -> set[str]:
    """Find collection names mutated by the observer passed to a resolver."""

    definitions = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    bodies: list[ast.AST] = []
    for callback in callback_nodes:
        if isinstance(callback, ast.Name) and callback.id in definitions:
            bodies.append(definitions[callback.id])
        elif isinstance(callback, ast.Lambda):
            bodies.append(callback.body)
    roots: set[str] = set()
    mutators = {"append", "extend", "add", "update", "setdefault"}
    for body in bodies:
        for node in ast.walk(body):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in mutators
                and isinstance(node.func.value, ast.Name)
            ):
                roots.add(node.func.value.id)
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                targets = list(node.targets) if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
                        roots.add(target.value.id)
    return roots


def _replacement_matches_observer_to_same_call_output(
    code: Any,
    observer_keywords: set[str],
) -> bool:
    """Require observer evidence to flow from the same resolver return value.

    A separate unbounded call can have different defaults or transformations,
    so its length is not evidence about the callback-bearing call.  The peer
    must capture that call's return, measure it, and compare the observer state
    against that measurement.
    """

    if not isinstance(code, str) or not observer_keywords:
        return False
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    assignments = [
        node for node in ast.walk(tree)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
    ]
    returned_names: set[str] = set()
    callback_nodes: list[ast.AST] = []
    for assignment in assignments:
        value = _assignment_value(assignment)
        if not isinstance(value, ast.Call) or not _is_resolver_call(value, require_unbounded=False):
            continue
        matching = [
            keyword for keyword in value.keywords
            if keyword.arg in observer_keywords
        ]
        if not matching:
            continue
        returned_names.update(_assigned_names(assignment))
        callback_nodes.extend(keyword.value for keyword in matching)
    if not returned_names:
        return False

    derived_names = set(returned_names)
    measured_names: set[str] = set()
    changed = True
    while changed:
        changed = False
        for assignment in assignments:
            value = _assignment_value(assignment)
            targets = _assigned_names(assignment)
            if value is None or not targets:
                continue
            if _is_observed_length(value, derived_names):
                before = len(measured_names)
                measured_names.update(targets)
                changed = changed or len(measured_names) != before
            elif _references_any(value, derived_names):
                before = len(derived_names)
                derived_names.update(targets)
                changed = changed or len(derived_names) != before

    observer_roots = _callback_mutation_roots(tree, callback_nodes)
    if not observer_roots:
        return False

    def is_same_call_measurement(value: ast.AST) -> bool:
        return _references_any(value, measured_names) or _is_observed_length(value, derived_names)

    def is_observer_state(value: ast.AST) -> bool:
        return _references_any(value, observer_roots)

    for assertion in (node for node in ast.walk(tree) if isinstance(node, ast.Assert)):
        test = assertion.test
        if not isinstance(test, ast.Compare):
            continue
        left = test.left
        for operation, right in zip(test.ops, test.comparators):
            if isinstance(operation, ast.Eq) and (
                (is_same_call_measurement(left) and is_observer_state(right))
                or (is_same_call_measurement(right) and is_observer_state(left))
            ):
                return True
            left = right
    return False


def _merge_probe_repairs(answer, retry, feature_id):
    """Apply peer-authored row replacements, never rewrite unaffected checks."""
    retained = retry.get("retained_probes", [])
    repair_ids = set(retry.get("repair_probe_ids", []))
    repair_constraints = retry.get("repair_constraints", {})
    if not retained and not repair_ids:
        return answer
    if not isinstance(answer, Mapping) or not isinstance(answer.get("probes"), list):
        raise ValueError("probe_repair_response_invalid")
    locked = {p["probe_id"]: p for p in retained}
    next_index = max([int(i.rsplit(":", 1)[1]) for i in set(locked) | repair_ids], default=-1) + 1
    merged, replaced = list(retained), set()
    for raw in answer["probes"]:
        if not isinstance(raw, Mapping):
            raise ValueError("probe_row_invalid")
        row = dict(raw)
        target = row.pop("replace_probe_id", None)
        if target:
            if target not in repair_ids or target in replaced:
                raise ValueError("probe_replacement_not_authorized")
            previous = next(
                (
                    probe
                    for probe in retry.get("previous_probes", [])
                    if probe.get("probe_id") == target
                ),
                None,
            )
            if previous is not None and _probe_content(row) == _probe_content(previous):
                raise ValueError(
                    "probe_replacement_unchanged_after_failed_execution:" + target
                )
            constraints = repair_constraints.get(target, []) if isinstance(repair_constraints, Mapping) else []
            if ("resolved_output_measurement" in constraints
                    and not _replacement_uses_resolved_output_measurement(row.get("code"))):
                raise ValueError("probe_replacement_requires_resolved_output_measurement:" + target)
            observer_keywords = {
                constraint.split(":", 1)[1]
                for constraint in constraints
                if isinstance(constraint, str)
                and constraint.startswith("same_call_resolved_output_observer_match:")
                and constraint.split(":", 1)[1]
            }
            if observer_keywords and not _replacement_matches_observer_to_same_call_output(
                row.get("code"), observer_keywords
            ):
                raise ValueError(
                    "probe_replacement_requires_same_call_resolved_output_observer_match:"
                    + target
                )
            row["probe_id"] = target
            replaced.add(target)
        else:
            identity = row.get("probe_id")
            if identity in locked:
                if _probe_content(row) != _probe_content(locked[identity]):
                    raise ValueError("probe_repair_cannot_modify_retained_check")
                continue
            if any(_probe_content(row) == _probe_content(p) for p in retained):
                continue  # Tolerate a repeated, unchanged row in a full answer.
            if identity in repair_ids:
                raise ValueError("probe_repair_requires_explicit_replacement")
            row["probe_id"] = f"{feature_id}:{next_index}"
            next_index += 1
        merged.append(row)
    if repair_ids - replaced:
        raise ValueError("probe_replacements_missing:" + ",".join(sorted(repair_ids - replaced)))
    return {"probes": merged}


def _replacement_constraint_failure(error: Exception) -> tuple[str, str] | None:
    """Return the stable row and constraint named by a host rejection."""

    rendered = str(error)
    constraints = {
        "probe_replacement_unchanged_after_failed_execution:": (
            "failed_execution_replacement_must_change"
        ),
        "probe_replacement_requires_resolved_output_measurement:": "resolved_output_measurement",
        "probe_replacement_requires_same_call_resolved_output_observer_match:": (
            "same_call_resolved_output_observer_match"
        ),
    }
    for prefix, constraint in constraints.items():
        if not rendered.startswith(prefix):
            continue
        probe_id = rendered[len(prefix):].strip()
        return (probe_id, constraint) if probe_id else None
    return None


def invalidate_probe_rows(world, feature_id, probe_ids, **details):
    """Invalidate only defective definitions; every retained row is re-executed."""
    plans = world.__dict__.setdefault("_cooperbench_behavior_plans", {})
    plan = plans.pop(feature_id, {})
    rows = plan.get("probes", [])
    repair_ids = sorted(set(probe_ids) & {row["probe_id"] for row in rows})
    gap = {"identity": plan.get("identity"), **details,
           "invalidated_plan": deepcopy(plan), "previous_probes": rows,
           "retained_probes": [row for row in rows if row["probe_id"] not in repair_ids],
           "repair_probe_ids": repair_ids}
    constraints = _repair_constraints(repair_ids, details, rows)
    if constraints:
        gap["repair_constraints"] = constraints
    world.__dict__.setdefault("_cooperbench_probe_evidence_gaps", {})[feature_id] = gap


def _execute_plan(world, probes, *, source_snapshot=None, baseline_snapshot=None):
    """Execute an own-feature review only on its frozen PR head."""
    return _execute_exact_snapshot_plan(
        world, probes, source_snapshot=source_snapshot, baseline_snapshot=baseline_snapshot,
        allowed_source_kinds={"committed_pr_head"},
    )


def _execute_integrated_plan(world, probes, *, source_snapshot, baseline_snapshot):
    """Provider-free execution boundary for explicitly selected integration trees.

    The replay consumer checks eligibility and source/plan CAS. Keeping this
    separate prevents a caller from changing an ordinary PR review's source.
    """
    from .source_views import SourceViewError, source_views_enabled
    if not source_views_enabled(world):
        raise SourceViewError("integrated_behavior_source_views_required")
    return _execute_exact_snapshot_plan(
        world, probes, source_snapshot=source_snapshot, baseline_snapshot=baseline_snapshot,
        allowed_source_kinds={"merge_candidate", "committed_mainline"},
    )


def _execute_actor_patch_candidate_plan(world, probes, *, source_snapshot, baseline_snapshot):
    """Provider-free execution boundary for a prepared, not-yet-accepted patch.

    Eligibility and compare/swap checks live in ``joint_probe_replay``.  Keeping
    this source kind out of the ordinary PR/integration entry points prevents a
    caller from substituting mutable actor work for a committed review target.
    """
    from .source_views import SourceViewError, source_views_enabled
    if not source_views_enabled(world):
        raise SourceViewError("actor_patch_behavior_source_views_required")
    return _execute_exact_snapshot_plan(
        world, probes, source_snapshot=source_snapshot, baseline_snapshot=baseline_snapshot,
        allowed_source_kinds={"actor_patch_candidate"},
    )


def _execute_exact_snapshot_plan(world, probes, *, source_snapshot=None, baseline_snapshot=None,
                                 allowed_source_kinds):
    from environments.org_env.product.materialize import (
        _atomic_export_text, _formal_product_executor, _product_smoke_root,
        _repo_hash, _safe_export_path, export_product_repo,
        _repo_artifact_entries, _runtime_asset_export_context, _export_runtime_asset_overlay,
        _run_executor_with_read_only_mounts, _runtime_asset_container_context,
    )
    from .source_views import SourceViewError, freeze_baseline, snapshot_projection, source_views_enabled

    projection, baseline_projection, source_binding = {}, None, {}
    if source_views_enabled(world):
        if source_snapshot is None or getattr(source_snapshot, "source_kind", None) not in allowed_source_kinds:
            raise SourceViewError("public_behavior_pr_snapshot_required")
        frozen_baseline = freeze_baseline(world)
        if baseline_snapshot is not None and baseline_snapshot != frozen_baseline:
            raise SourceViewError("public_behavior_baseline_snapshot_mismatch")
        baseline_snapshot = frozen_baseline
        projection = snapshot_projection(world, source_snapshot)
        baseline_projection = snapshot_projection(world, baseline_snapshot)
        baseline = dict(baseline_snapshot.files)
        source_binding = {"source_snapshot": source_snapshot.receipt(),
                          "baseline_snapshot": baseline_snapshot.receipt()}
    else:
        baseline = getattr(world, "_cooperbench_public_baseline_files", {}) or {}
    executor = _formal_product_executor()
    if executor is None:
        raise RuntimeError("public_behavior_probes_require_isolated_runtime")
    from .runtime_binding import (
        evaluator_runtime_receipt,
        require_evaluator_runtime,
    )
    expected_runtime = world.__dict__.get(
        "_cooperbench_evaluator_runtime"
    )
    if isinstance(expected_runtime, dict):
        runtime_receipt = require_evaluator_runtime(
            executor,
            expected_backend=str(expected_runtime.get("backend") or ""),
            expected_image=str(
                expected_runtime.get("container_image") or ""
            ),
            expected_platform=str(
                expected_runtime.get("container_platform") or ""
            ),
        )
    else:
        runtime_receipt = evaluator_runtime_receipt(executor)
    if not baseline:
        raise RuntimeError("public_behavior_baseline_unavailable")
    assets, protected = _runtime_asset_export_context(
        world, _repo_artifact_entries(getattr(world, "product_artifacts", {}) or {}), verify=True)
    candidate_hash = _repo_hash(world, prefer_mainline=False, **projection)
    baseline_hash = _digest(baseline)
    if assets is not None:
        baseline_hash = _digest({"schema_version": "cooperbench_behavior_baseline_with_assets_v1",
                                 "source_files": baseline, "runtime_assets_digest": assets.asset_set_digest})
        source_binding["runtime_assets_digest"] = assets.asset_set_digest
    identity = {"candidate_hash": candidate_hash, "baseline_hash": baseline_hash,
                "plan_hash": _digest(probes), "policy": PROBE_POLICY,
                "runtime": runtime_receipt["container_image"],
                "platform": runtime_receipt["container_platform"],
                "evaluator_runtime": runtime_receipt, **source_binding}
    # Never reuse a public-suite green result or an older head's probe result.
    # Every semantic review executes the persisted definitions afresh.
    root = Path(tempfile.mkdtemp(prefix="cooper_behavior_", dir=_product_smoke_root()))
    container_asset_root, read_only_mounts = _runtime_asset_container_context(
        world, executor
    )
    export_product_repo(
        world,
        str(root / "candidate"),
        prefer_mainline=False,
        _runtime_asset_container_root=container_asset_root,
        _durable_writes=False,
        **projection,
    )
    if baseline_projection is not None:
        # Use the same full-tree exporter as the candidate: baseline bytes,
        # deleted/peer exclusions, frozen runtime resources and executable bits.
        export_product_repo(
            world,
            str(root / "baseline"),
            prefer_mainline=False,
            _runtime_asset_container_root=container_asset_root,
            _durable_writes=False,
            **baseline_projection,
        )
    else:
        # Legacy worlds still use only their explicitly declared baseline map;
        # never fill missing baseline paths from ambient working source.
        for path, content in baseline.items():
            _atomic_export_text(root / "baseline", path, content, durable=False)
        if assets is not None:
            _export_runtime_asset_overlay(
                world,
                root / "baseline",
                assets,
                protected,
                list(baseline),
                container_asset_root=container_asset_root,
            )
    _atomic_export_text(root, "probe_input.json", json.dumps({"probes": probes}))
    _atomic_export_text(root, "probe_runner.py", Path(__file__).with_name("probe_runner.py").read_text(encoding="utf-8"))
    outcome = _run_executor_with_read_only_mounts(
        executor,
        root=root,
        argv=("python", "probe_runner.py"),
        timeout_seconds=_public_behavior_executor_timeout_seconds(probes),
        read_only_mounts=read_only_mounts,
    )
    if outcome.status in {"blocked", "timeout", "infra_error"} or outcome.exit_code != 0:
        raise RuntimeError("public_behavior_executor_failed:" + str(outcome.blocked_reason or outcome.stderr_tail)[-500:])
    receipt_digest = None
    for line in (outcome.stdout_tail or "").splitlines():
        if line.startswith("COOPER_BEHAVIOR_RESULTS_SHA256="):
            receipt_digest = line.split("=", 1)[1]
    _, receipt_path = _safe_export_path(root, "probe_results.json")
    if not receipt_path.is_file() or receipt_path.stat().st_size > 256000:
        raise RuntimeError("public_behavior_executor_receipt_missing_or_oversize")
    raw = receipt_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != receipt_digest:
        raise RuntimeError("public_behavior_executor_receipt_digest_mismatch")
    rows = json.loads(raw)
    if (not isinstance(rows, list) or len(rows) != len(probes)
            or [r.get("probe_id") for r in rows if isinstance(r, dict)] != [p["probe_id"] for p in probes]):
        raise RuntimeError("public_behavior_executor_receipt_incomplete")
    for row, probe in zip(rows, probes):
        for side in (["baseline", "candidate"] if probe["compare_baseline"] else ["candidate"]):
            result = row.get(side)
            if not isinstance(result, dict) or result.get("status") not in {"pass", "fail", "unavailable"}:
                raise RuntimeError("public_behavior_executor_receipt_side_missing")
            if result["status"] == "pass" and (
                type(result.get("assertions")) is not int
                or type(result.get("assertion_guards", 0)) is not int
                or result.get("assertions", 0) + result.get("assertion_guards", 0) < 1
                or (probe.get("kind", "runtime") == "runtime" and (
                    (type(result.get("public_calls")) is not int
                     or type(result.get("public_accesses", 0)) is not int
                     or result["public_calls"] + result.get("public_accesses", 0) < 1)
                    or not set(result.get("public_paths") or []).intersection(probe["paths"])))
                or (probe.get("kind") == "source"
                    and not set(result.get("public_reads") or []).intersection(probe["paths"]))
            ):
                raise RuntimeError("public_behavior_executor_receipt_has_no_executed_evidence")
        row["paths"] = probe["paths"]
        row["requirement_ids"] = probe["requirement_ids"]
    from society_core.code_landing.event_store import redact_sensitive_payload
    return {**identity, "results": redact_sensitive_payload(rows), "artifact_dir": str(root)}


def _strict_regression_probes(world, reviewer_id, feature_id, plans, baseline):
    """Only replay peer plans whose brief is visible and provenance is current."""
    reviews = getattr(world, "_cooperbench_semantic_reviews", {}) or {}
    probes = []
    for other in sorted((set(plans) | set(reviews)) - {feature_id}):
        # Checking visibility precedes reading the other feature's plan or
        # receipt, including when a legacy cache is present in this world.
        visible = visible_review_contract(world, reviewer_id, other)
        if visible is None:
            return [], "approved_peer_feature_brief_unshared_or_unread"
        review = reviews.get(other) or {}
        if not review.get("approved"):
            continue
        saved = plans.get(other) or {}
        origin_reviewer = str(review.get("reviewer_id") or "")
        origin = visible_review_contract(world, origin_reviewer, other) if origin_reviewer else None
        if (origin is None or origin["description"] != visible["description"]
                or any(review.get(key) != origin[key] or saved.get(key) != origin[key]
                       for key in ("brief_visibility_receipt", "review_brief_identity"))
                or saved.get("reviewer_id") != origin_reviewer):
            return [], "approved_peer_feature_probe_plan_visibility_mismatch"
        expected = _recorded_plan_identity(saved, review.get("behavior_evidence"),
                                  origin["description"], origin["requirements"], origin["compatibility"],
                                  saved.get("paths") or [], baseline, origin_reviewer, origin,
                                  saved.get("integration_contract") or {})
        if expected is None:
            return [], "approved_peer_feature_probe_plan_visibility_mismatch"
        try:
            canonical = _validate_plan(saved, feature_id=other,
                requirements=origin["requirements"], compatibility=origin["compatibility"],
                paths=saved.get("paths") or [],
                integration_contract=saved.get("integration_contract") or {}, _preserve_ids=True)
        except (ValueError, TypeError, KeyError, AttributeError):
            return [], "approved_peer_feature_probe_plan_invalid"
        if canonical != saved.get("probes"):
            return [], "approved_peer_feature_probe_plan_invalid"
        probes.extend(saved.get("probes") or [])
    return probes, ""


def prepare_public_behavior_evidence(world, *, reviewer_id, feature_id,
                                     description, requirements, compatibility, paths,
                                     source_snapshot=None, baseline_snapshot=None, pull_request=None):
    """Draft once per public contract; reuse tests across candidate repairs.

    The caller can invalidate a demonstrably defective check, but that produces
    an evidence gap and a new executed review, never a bypass approval.
    """
    from .visibility import is_strict_coop

    strict = is_strict_coop(world)
    contract = visible_review_contract(world, reviewer_id, feature_id) if strict else None
    if strict and contract is None:
        return {"available": False, "ok": False, "results": [], "repair_paths": [],
                "error": "semantic_review_brief_unshared_or_unread"}
    if contract is not None:
        # Never trust a caller's omniscient description/SDL ledger in strict
        # coop. Both source review and test generation use the delivered text.
        description = contract["description"]
        requirements = contract["requirements"]
        compatibility = contract["compatibility"]
    from .source_views import SourceViewError, freeze_baseline, pr_head_snapshot, source_views_enabled

    source_arguments = {}
    source_scoped = source_views_enabled(world)
    from .replay_workflow import workflow_enabled
    own_head_only = source_scoped and workflow_enabled(world)
    if source_scoped:
        try:
            if pull_request is None:
                raise SourceViewError("public_behavior_pr_snapshot_required")
            frozen = pr_head_snapshot(world, pull_request, require_current=True)
            if source_snapshot is not None and source_snapshot != frozen:
                raise SourceViewError("public_behavior_pr_snapshot_mismatch")
            source_snapshot = frozen
            frozen_baseline = freeze_baseline(world)
            if baseline_snapshot is not None and baseline_snapshot != frozen_baseline:
                raise SourceViewError("public_behavior_baseline_snapshot_mismatch")
            baseline_snapshot = frozen_baseline
            baseline = dict(baseline_snapshot.files)
            source_arguments = {"source_snapshot": source_snapshot, "baseline_snapshot": baseline_snapshot}
        except SourceViewError as error:
            return {"available": False, "ok": False, "results": [], "repair_paths": [],
                    "error": "public_behavior_source_snapshot_unavailable:" + error.code}
    else:
        baseline = getattr(world, "_cooperbench_public_baseline_files", {}) or {}
    plans = world.__dict__.setdefault("_cooperbench_behavior_plans", {})
    integration_contract = (
        _protected_integration_contract(
            world, str(reviewer_id), str(feature_id), list(paths), contract
        )
        if strict and contract is not None and own_head_only
        else {}
    )
    identity = _plan_identity(
        description, requirements, compatibility, paths, baseline, reviewer_id,
        contract, integration_contract,
    )
    regression_probes = []
    if strict and not own_head_only:
        regression_probes, error = _strict_regression_probes(world, reviewer_id, feature_id, plans, baseline)
        if error:
            return {"available": False, "ok": False, "results": [], "repair_paths": [], "error": error,
                    **_contract_binding(contract)}
    plan = plans.get(feature_id)
    if not plan or plan.get("identity") != identity:
        retry = dict(
            (getattr(world, "_cooperbench_probe_evidence_gaps", {}) or {}).get(
                feature_id, {}
            )
        )
        if (retry.get("identity") != identity if strict else retry.get("identity") not in (None, identity)):
            retry = {}  # No definitions or feedback cross a public-contract identity.
        elif retry:
            baseline_gap = _baseline_comparison_gap(
                list(retry.get("retained_probes") or []), compatibility
            )
            if baseline_gap:
                retry["missing_baseline_comparison"] = baseline_gap
        from .probe_prerequisites import PREREQUISITE_GUIDANCE
        system = (
            "You are the actual peer in a two-person organization writing small executable public-acceptance "
            "checks BEFORE reviewing the candidate. Use ONLY this public request and untouched source; "
            f"do not guess hidden tests or redefine the requested API. Return 1-{MAX_DRAFT_PROBES} Python snippets with top-level "
            "assert statements and calls to real public implementation. A snippet may cover several req_ ids. "
            "Prefer 4-8 coherent checks, but do not drop an obligation to hit that preference. "
            f"{PREFERRED_MAX_PROBES} checks is a compaction target, not a hard limit: retain additional "
            "checks when removing them would lose required coverage, baseline comparison, or cross-feature evidence. "
            "The complete public request is authoritative; the acceptance index is only a navigation aid. "
            "related_public_source_context contains read-only public callers and usage examples. Use it to "
            "exercise the requested change through its existing entry point and consuming output; calling "
            "a newly introduced helper alone does not show that existing callers reach it. These supporting "
            "files do not expand allowed paths or override the public request. actor_feedback contains only "
            "your read messages and your own public-test receipts; use relevant observations to design checks, "
            "not as verifier verdicts or new acceptance criteria. "
            + PREREQUISITE_GUIDANCE + " " + PUBLIC_CONTRACT_BOUNDARY_GUIDANCE + " "
            "Do not invent obligations from headings or isolated example/docstring fragments. "
            "If retained_probes are present in previous_probe_defects, they are immutable. Return ONLY "
            "new checks for missing coverage plus replacement rows for every repair_probe_id. Each replacement "
            "must include replace_probe_id with that exact id. Do not repeat or modify retained checks. "
            "When failed_checks are present, inspect each row's status, error, and output, and change every "
            "listed replacement so the recorded execution failure cannot recur. A network or DNS traceback "
            "means the check must use a deterministic local fixture instead of a remote-backed convenience loader. "
            "Host-enforced repair_constraints describe the confirmed defect basis that each replacement must "
            "eliminate. If a row requires resolved_output_measurement, first call the relevant public resolver "
            "without max_bytes, derive the actual length from its returned buffer/data, and use that measured "
            "value (or arithmetic derived from it) as the max_bytes boundary. Never substitute a raw input, "
            "encoded-input, or guessed literal length for the resolved/final output named by the contract. "
            "If a row requires same_call_resolved_output_observer_match:<keyword>, capture the return value from "
            "the SAME resolver call that supplies that observer keyword, read or getvalue() that returned "
            "buffer, and directly compare the observer's recorded value with len() of that same returned output. "
            "Do not measure a separate resolver call because its defaults or transformations may differ. "
            "If replacement_validation_errors is present, the host rejected the matching row in previous_response. "
            "Correct that exact row and preserve replace_probe_id; do not repeat the rejected code. The "
            "resolved_output_measurement error specifically means that every max_bytes boundary must flow from "
            "len() of an earlier unbounded public resolver result, while assertions must not compare that observed "
            "result to a guessed concrete output size. "
            "If missing_baseline_comparison is present, return an additional compare_baseline=true row in "
            "this same response, using only operations available in the untouched baseline. A repaired row "
            "that calls a new API stays candidate-only; it does not replace the separate baseline-safe row. "
            "If only previous_response is present, correct that draft. Coverage and total budgets apply to "
            "the combined retained and new plan, not just this response. "
            "host_enforced_atomic_observations in the request are mechanical acceptance conditions, not advice. "
            "For final_observer_matches_returned_value, one row must invoke the public resolver with its explicit "
            "callback/hook, read or getvalue() the returned buffer, and directly assert that the observer received "
            "that actual returned value's length. Do not invent a trim direction, padding character, or expected "
            "transformation when the public request does not define it. The diagnostic "
            "unstated_transform_semantics_must_not_be_asserted means to remove that invented oracle while keeping "
            "the returned-value/observer check. For trim_then_final_observer, which is emitted only when public "
            "text explicitly defines trim semantics, one row must use exactly the public direction and boundary "
            "characters to make that stated transformation observable, assert the returned transformed value, "
            "and assert that the observer received its final length. Never substitute a different strip operator "
            "or padding convention. Do all of these in the same row and requirement id. "
            "Every required public interaction shown in a backtick example remains binding. "
            "A helper, callback, hook, or private symbol named only as an implementation detail is not itself "
            "a caller interaction: cover that clause through the public entry point that reaches or observes it. "
            "If the request explicitly exposes or exports that symbol to callers, its exact public call remains binding. "
            "Optional or may-expose helpers are not mandatory merely because they share a paragraph "
            "with a required API. Preserve mandatory, prohibited, and conditional behavior; an optional "
            "parameter does not make its required entrypoint optional. A correction may "
            "fix setup or expected results, but may not rename a required call, replace it with a fallback "
            "helper, or accept either the requested API or an alternative. The harness checks direct call, "
            "get and set terminal names mechanically and will report the missing interaction. "
            "Cover every listed acceptance id. Invoke exact public API names/examples, including context-manager "
            "forms, and check observable results, restoration, defaults and boundaries. Each acceptance id is an "
            "atomic public clause: do not attach an id to a check that demonstrates only one part of the clause. "
            "For an ordered operation whose transformation semantics are explicitly defined by the public request, "
            "construct input on which every named operation is observable, assert the returned value after the transformation, and separately "
            "assert that callbacks, lengths, metrics, or other side effects observe that final value. The presence "
            "of the callback or metric alone is not evidence that a defined preceding transformation happened. If "
            "the public request names a transformation but leaves its operator or boundary semantics unspecified, "
            "do not choose one: check the observer against the actual returned value instead. Apply "
            "the same rule to filtering, normalization, sorting, deduplication, truncation, validation, and other "
            "publicly named transformations. When a public transformation applies to a complete collection or "
            "representation, exercise and assert the complete public result; do not silently narrow it to an "
            "occupied, effective, sampled, or prefix subset unless the request defines that scope. For immediate "
            "reset/completion semantics, inspect the first public observation after the trigger; do not defer the "
            "reset to stats(), a getter, reuse, or a later chunk, and do not require a pre-reset read unless the "
            "request exposes one. Do not mock, reimplement, or monkeypatch the subject. A mock of the dependency "
            "that performs the publicly required side effect is also not evidence of that effect: use a bounded "
            "real local file/process/callback boundary and assert its externally observable result. Do not infer "
            "unstated ambient or global propagation between separately "
            "documented calls: setting state on one object does not change a standalone function call unless the "
            "public request says so or the state is passed to that call explicitly. Exercise an object-owned hook "
            "through observable behavior on that object. Use local temporary directories and no external services. "
            "Use compare_baseline=true for unchanged/default/legacy public behavior: the SAME code must pass "
            "on both untouched baseline and candidate, using API available in both. Include at least one such "
            "comparison when compatibility ids are listed. Test NEW enabled behavior separately with "
            "compare_baseline=false. For backward reading, baseline_setup is optional Python run on the "
            "untouched baseline to create real data in os.environ['PROBE_DATA_DIR']; your check reads it in a "
            "fresh process. Do not manufacture legacy bytes based on the new design. Setup runs independently "
            "for each baseline/candidate check. Otherwise baseline_setup must be empty. Each check has 15s, "
            "fresh module state, real task dependencies, and repo root/src on PYTHONPATH. No pytest needed. "
            f"Keep every code snippet at most {MAX_PROBE_CODE_CHARS} characters and every baseline_setup at "
            f"most {MAX_PROBE_SETUP_CHARS} characters. A single draft response remains bounded to "
            f"{MAX_DRAFT_CHARS} characters and {MAX_DRAFT_PROBES} rows. During repair, return only the "
            "requested repaired/new rows; immutable retained rows are merged by the host and do not consume "
            "the current response budget. The merged plan remains bounded by the row count and the same "
            "per-row limits. "
            "Prefer compact snippets under 1200 characters: omit explanatory "
            "comments and repeated fixtures, and split a long multi-behavior check into separate rows that "
            "repeat the applicable requirement id. A probe_code_too_long repair must replace only that stable "
            "row id with executable code within the reported hard limit. Keep snippets executable and use "
            "direct assert rather than an "
            "uncalled test function. "
            "A function that is only defined and never called produces no evidence. Import the public package. "
            "When public_cross_feature_interaction is present, the protected feature is already merged and its "
            "complete public request is visible to you. For a protected row with runtime_probe_mode="
            "same_call_keywords, add at least one candidate-only runtime row whose SAME call to one listed shared "
            "callable explicitly supplies one listed current keyword parameter and one listed protected keyword "
            "parameter. Assert the combined observable behavior required by both public requests. Separate calls, "
            "source inspection, or merely mentioning both names do not cover that executable gate. A source_trace_only "
            "row has no mechanically justified same-call probe shape; do not invent one. It remains mandatory input "
            "to the peer's three-way source review of the shared callable, signature, callers, ordering, and cleanup. "
            "kind=runtime requires a real call to the named public implementation. Only ids listed under "
            "source_only_ids may use kind=source instead: read the named current public source and execute "
            "assertions about a required internal dependency call in its implementing method. A labelled "
            "implementation recipe is not a requirement to call the dependency directly from the test. "
            "Verify the named call in the implementation's AST, not merely its import or a comment. Other "
            "source-only clauses require "
            "assertions about its syntax/imports/type hints, for example using ast.parse. Such a check need "
            "not call the product. Do not attach runtime ids to a source check, even when the same source "
            "snippet happens to mention their implementation; cover those ids in a separate runtime row. "
            "Do not relabel runtime behavior as a source check. "
            "Direct assert and dynamic if/raise AssertionError guards are supported; both must actually "
            "execute on the checked path. Prefer assert not bad_condition over a vacuous assert True. "
            "Do not use bare except/except Exception/except BaseException with pass to swallow an operation's "
            "failure and then assert only elapsed time or an unrelated value. Let unexpected exceptions fail. "
            "For a publicly required exception, catch its specific type and assert the required observable "
            "error, with a failure on the no-exception path. Explicit public pytest warning filters from the "
            "untouched baseline apply to probes too; do not suppress warnings to bypass that policy. "
            "Use paths only from the provided FILE paths. All checks must pass before approval, but this is "
            "development verification, not an official benchmark verdict. Return JSON only."
        )
        from .public_contract import acceptance_index
        from .source_retrieval import related_public_source_context
        from .feedback import actor_feedback_context
        request_payload = {
            "reviewer_id": reviewer_id,
            "feature_id": feature_id,
            "public_request": description,
            "acceptance_index": requirements,
            "host_enforced_atomic_observations": {
                requirement_identity: required_atomic_observations(text)
                for requirement_identity, text in requirements.items()
                if required_atomic_observations(text)
            },
            "acceptance_source_spans": acceptance_index(description),
            "compatibility_ids": list(compatibility),
            "source_only_ids": sorted(source_only_requirement_ids(requirements)),
            "untouched_files": {path: baseline.get(path, "") for path in paths},
            "related_public_source_context": related_public_source_context(world, reviewer_id, paths),
            "actor_feedback": actor_feedback_context(world, reviewer_id),
            "public_cross_feature_interaction": integration_contract,
        }

        def _plan_request(previous_probe_defects):
            return json.dumps(
                {**request_payload, "previous_probe_defects": previous_probe_defects},
                ensure_ascii=False,
            )

        def _plan_failure_gap(previous, response, error):
            # A provider/transport exception carries no new draft.  A returned
            # draft that the host rejected does: preserve every valid row and
            # name only the stable rows that need repair.
            constraint_failure = _replacement_constraint_failure(error)
            # A constraint rejection is about one replacement row, not the
            # retained plan. Keep that plan plus the returned draft so the
            # peer can correct its exact failed attempt. Previously the merge
            # exception left ``merged`` as None and silently discarded the
            # actual provider response, causing identical retries many ticks
            # apart despite a healthy provider.
            gap = dict(previous) if response is None or constraint_failure else {}
            gap["identity"] = identity
            gap["error"] = (
                f"public_behavior_plan_failed:{type(error).__name__}:"
                f"{str(error)[:400]}"
            )
            if response is not None:
                if constraint_failure:
                    from society_core.code_landing.event_store import redact_sensitive_payload
                    probe_id, constraint = constraint_failure
                    raw = json.dumps(response, ensure_ascii=False)
                    if len(raw) <= MAX_DRAFT_CHARS:
                        gap["previous_response"] = redact_sensitive_payload(response)
                    else:
                        gap["draft_omitted"] = "response_exceeds_bounded_retry_context"
                        gap["response_chars"] = len(raw)
                    rejected = []
                    if isinstance(response, Mapping) and isinstance(response.get("probes"), list):
                        rejected = [
                            dict(row) for row in response["probes"]
                            if isinstance(row, Mapping)
                            and row.get("replace_probe_id") == probe_id
                        ]
                    if rejected and len(json.dumps(rejected, ensure_ascii=False)) <= MAX_DRAFT_CHARS:
                        gap["rejected_replacements"] = redact_sensitive_payload(rejected)
                    gap["replacement_validation_errors"] = {
                        probe_id: {
                            "code": str(error)[:400],
                            "constraint": constraint,
                            "instruction": (
                                "Inspect failed_checks for this row and replace the code so it no longer triggers "
                                "the recorded runtime, setup, or external-service failure. Do not return the "
                                "same code again."
                                if constraint == "failed_execution_replacement_must_change"
                                else
                                "Capture the return from the same resolver call that supplies the observer; "
                                "read or getvalue() that returned output; compare the observer value directly "
                                "with len() of that output; do not measure a separate call."
                                if constraint == "same_call_resolved_output_observer_match"
                                else
                                "Measure the earlier unbounded public resolver output with len(); "
                                "derive every max_bytes value from that measurement; do not assert "
                                "a guessed concrete resolved-output size."
                            ),
                        }
                    }
                else:
                    diagnostic = _response_diagnostic(
                        response,
                        requirements,
                        feature_id=feature_id,
                        paths=paths,
                        compatibility=compatibility,
                        integration_contract=integration_contract,
                    )
                    if previous.get("retained_probes") and "retained_probes" not in diagnostic:
                        diagnostic["retained_probes"] = previous["retained_probes"]
                        diagnostic["repair_probe_ids"] = previous.get("repair_probe_ids", [])
                    gap.update(diagnostic)
            # A structurally valid replacement can still fail a host-enforced
            # correction constraint. Preserve the confirmed defect basis and
            # constraint across that retry; otherwise the next peer turn sees
            # only a generic shape error and can repeat the same assumption.
            for key in ("defects", "previous_probes", "repair_constraints"):
                if key in previous:
                    gap.setdefault(key, previous[key])
            baseline_gap = _baseline_comparison_gap(
                list(gap.get("retained_probes") or []), compatibility
            )
            if baseline_gap:
                gap["missing_baseline_comparison"] = baseline_gap
            return gap

        user = _plan_request(retry)
        schema = {"probes": [{"requirement_ids": ["exact req_ id"], "paths": ["exact FILE path"],
                               "replace_probe_id": "only for requested replacements; otherwise omit or empty",
                               "compare_baseline": "boolean", "kind": "runtime|source",
                               "baseline_setup": "Python or empty string",
                               "code": "Python statements that call the public API and assert behavior"}]}
        answer = None
        merged = None
        bounded_atomic_plan_repair_attempted = False
        bounded_atomic_plan_repair_used = False
        bounded_constraint_plan_repair_attempted = False
        bounded_constraint_plan_repair_used = False
        bounded_interaction_plan_repair_attempted = False
        bounded_interaction_plan_repair_used = False
        try:
            from environments.org_env.llm.client import reasoning_effort_override
            with reasoning_effort_override("medium"):
                answer = world.llm_client.generate_json(system, user, schema, max_tokens=6000)
            merged = _merge_probe_repairs(answer, retry, feature_id)
            probes = _validate_plan(merged, feature_id=feature_id, requirements=requirements,
                                    compatibility=compatibility, paths=paths,
                                    integration_contract=integration_contract,
                                    _preserve_ids=bool(retry.get("retained_probes") or retry.get("repair_probe_ids")))
        except Exception as exc:
            rejected_response = merged if merged is not None else answer
            gap = _plan_failure_gap(retry, rejected_response, exc)
            # The host can give exact, non-semantic feedback for this failure:
            # the returned row claimed an ordered public clause without making
            # both the transformed return and final observer value observable.
            # Repair it now, inside the same review action, rather than burning
            # another scarce two-person decision window.  This is deliberately
            # one extra provider response only; a second invalid draft still
            # fails closed and remains available to a later peer turn.
            atomic_gap = str(exc).startswith("probe_atomic_observable_clause_missing:")
            constraint_gap = _replacement_constraint_failure(exc) is not None
            interaction_gap = str(exc).startswith("probe_cross_feature_interaction_missing:")
            repairable_gap = (
                (merged is not None and atomic_gap)
                or (merged is not None and interaction_gap)
                or (answer is not None and constraint_gap)
            )
            if repairable_gap:
                bounded_atomic_plan_repair_attempted = atomic_gap
                bounded_constraint_plan_repair_attempted = constraint_gap
                bounded_interaction_plan_repair_attempted = interaction_gap
                repair_answer = None
                repair_merged = None
                try:
                    with reasoning_effort_override("medium"):
                        repair_answer = world.llm_client.generate_json(
                            system,
                            _plan_request(gap),
                            schema,
                            max_tokens=6000,
                        )
                    repair_merged = _merge_probe_repairs(repair_answer, gap, feature_id)
                    probes = _validate_plan(
                        repair_merged,
                        feature_id=feature_id,
                        requirements=requirements,
                        compatibility=compatibility,
                        paths=paths,
                        integration_contract=integration_contract,
                        _preserve_ids=True,
                    )
                    bounded_atomic_plan_repair_used = atomic_gap
                    bounded_constraint_plan_repair_used = constraint_gap
                    bounded_interaction_plan_repair_used = interaction_gap
                except Exception as repair_exc:
                    rejected_repair = (
                        repair_merged if repair_merged is not None else repair_answer
                    )
                    gap = _plan_failure_gap(gap, rejected_repair, repair_exc)
                    gap["bounded_atomic_plan_repair_attempted"] = atomic_gap
                    gap["bounded_atomic_plan_repair_used"] = False
                    gap["bounded_constraint_plan_repair_attempted"] = constraint_gap
                    gap["bounded_constraint_plan_repair_used"] = False
                    gap["bounded_interaction_plan_repair_attempted"] = interaction_gap
                    gap["bounded_interaction_plan_repair_used"] = False
                    world.__dict__.setdefault(
                        "_cooperbench_probe_evidence_gaps", {}
                    )[feature_id] = gap
                    return {"available": False, "ok": False, **gap, "results": []}
            else:
                world.__dict__.setdefault("_cooperbench_probe_evidence_gaps", {})[
                    feature_id
                ] = gap
                return {"available": False, "ok": False, **gap, "results": []}
        plan = {
            "identity": identity,
            "policy": PROBE_POLICY,
            "reviewer_id": reviewer_id,
            "probes": probes,
            "bounded_atomic_plan_repair_attempted": (
                bounded_atomic_plan_repair_attempted
            ),
            "bounded_atomic_plan_repair_used": bounded_atomic_plan_repair_used,
            "bounded_constraint_plan_repair_attempted": (
                bounded_constraint_plan_repair_attempted
            ),
            "bounded_constraint_plan_repair_used": (
                bounded_constraint_plan_repair_used
            ),
            "bounded_interaction_plan_repair_attempted": (
                bounded_interaction_plan_repair_attempted
            ),
            "bounded_interaction_plan_repair_used": (
                bounded_interaction_plan_repair_used
            ),
            "integration_contract": integration_contract,
        }
        if contract is not None:
            plan.update(**_contract_binding(contract), paths=list(paths))
        plans[feature_id] = plan
    probes = list(plan["probes"])
    # A frozen PR head is NOT necessarily integrated with an already-merged
    # peer feature: parallel branches may start from different main revisions.
    # Its own feature is reviewed here; accepted cross-feature checks must run
    # on an explicit merge candidate and final mainline via joint_probe_replay.
    # This is not an integration pass or an exemption from those separate gates.
    if own_head_only:
        pass
    elif strict:
        probes.extend(regression_probes)
    else:
        reviews = getattr(world, "_cooperbench_semantic_reviews", {}) or {}
        for other, saved in plans.items():
            if other != feature_id and (reviews.get(other) or {}).get("approved"):
                probes.extend(saved["probes"])
        if any(other != feature_id and review.get("approved") and other not in plans
               for other, review in reviews.items()):
            return {"available": False, "ok": False, "results": [],
                    "error": "approved_peer_feature_probe_plan_missing"}
    try:
        evidence = _execute_plan(world, probes, **source_arguments)
    except Exception as exc:
        return {"available": False, "ok": False, "error": f"public_behavior_execution_unavailable:{type(exc).__name__}:{str(exc)[:400]}", "results": []}
    gaps = [r for r in evidence["results"] if
            (r.get("baseline") is not None and r["baseline"].get("status") != "pass")
            or not isinstance(r.get("candidate"), dict)
            or r["candidate"].get("status") not in {"pass", "fail"}]
    if gaps:
        # Test/runtime defects are reviewer work, never fictional code defects.
        for affected in {r["probe_id"].split(":", 1)[0] for r in gaps}:
            failed = [r for r in gaps if r["probe_id"].split(":", 1)[0] == affected]
            if affected == feature_id:
                invalidate_probe_rows(world, affected, [r["probe_id"] for r in failed], failed_checks=failed)
            # A previously reviewed peer plan is re-executed, but this feature's
            # reviewer cannot rewrite it on an ambiguous runtime outage.
    else:
        world.__dict__.setdefault("_cooperbench_probe_evidence_gaps", {}).pop(feature_id, None)
    return {**evidence, "available": not gaps, "ok": not gaps and all(
        r["candidate"]["status"] == "pass" for r in evidence["results"]),
        "error": "public_behavior_probe_evidence_incomplete" if gaps else "",
        "bounded_atomic_plan_repair_attempted": bool(
            plan.get("bounded_atomic_plan_repair_attempted")
        ),
        "bounded_atomic_plan_repair_used": bool(
            plan.get("bounded_atomic_plan_repair_used")
        ),
        "bounded_constraint_plan_repair_attempted": bool(
            plan.get("bounded_constraint_plan_repair_attempted")
        ),
        "bounded_constraint_plan_repair_used": bool(
            plan.get("bounded_constraint_plan_repair_used")
        ),
        "bounded_interaction_plan_repair_attempted": bool(
            plan.get("bounded_interaction_plan_repair_attempted")
        ),
        "bounded_interaction_plan_repair_used": bool(
            plan.get("bounded_interaction_plan_repair_used")
        ),
        "integration_contract": plan.get("integration_contract") or {},
        **({"evaluation_scope": "feature_pr_head"} if own_head_only else {}),
        "probes": probes, **(_contract_binding(contract) if contract is not None else {})}
