"""Public, pre-candidate evidence for a peer probe's calling convention/fixture.

This is context for the actual reviewer, not a host verdict or a replacement
oracle. Only the already-loaded public starter tree is read.
"""
from __future__ import annotations

import ast
import re
from typing import Any, Mapping
from .probe_preconditions import OBSERVER_GUIDANCE


PREREQUISITE_GUIDANCE = OBSERVER_GUIDANCE + (
    "Validate the invocation and fixture BEFORE treating a failure as a product defect. "
    "Follow the untouched public source's calling convention, decorators and adapters: "
    "a function retrieved from a registry is not automatically a bound method. Supply "
    "its required receiver/context or use the existing public dispatch API. A traceback "
    "inside candidate code does not prove that the caller supplied a supported invocation. "
    "For external-tool features, supply a deterministic local executable through the public "
    "configuration API; do not assume an interactive editor or other ambient binary exists. "
    "For filesystem-error tests, do not assume chmod denies writes when the process may be "
    "privileged: establish a real local failing boundary and verify the intended error, "
    "without changing a shared temporary parent or replacing the feature being tested. "
    "Dependency fakes must implement the actual adapter response protocol and distinguish "
    "setup calls from the operation under test. A shared output-field label or prompt "
    "substring is not evidence of which operation ran; both initial and follow-up calls "
    "may contain it, and an adapter may retry malformed fixture output. Use supported "
    "local fixtures and account for setup before asserting an exact operation call count. "
    "Derive expectations from the complete runtime operands, including generated temporary "
    "directories. A safely quoted full path does not contain a separately quoted basename. "
    "For a named public transformation such as shlex.quote, compute the expected value from "
    "the actual input variable; do not substitute a hand-written substring or different "
    "representation. For shell command tests, distinguish syntactic quote boundaries from "
    "literal characters and check the required whole operand or its safely parsed value. "
    "Do not conflate path expansion with normalization: expandvars/expanduser preserve "
    "dot-dot segments, while normpath/resolve may remove them. Unless the public contract "
    "requires both transformations, do not simultaneously demand the raw expanded string "
    "and a different normalized string. A relative fixture can name the same location "
    "without producing the same path spelling; derive exact expectations in the specified order. "
    "For nested source strings, inspect the decoded literal call arguments: JSON and Python "
    "escaping are not the parser's input. A quote escaped only for Python can become an "
    "unescaped delimiter in the template, query, or other language being tested. Support for "
    "special characters does not require accepting a malformed string literal. Establish a "
    "syntactically valid fixture under the public contract before blaming the implementation; "
    "preserve deliberate invalid-input tests when the contract explicitly requires rejection. "
    "For masked numeric outputs, different parameters need not produce different observable "
    "values: an all-masked vector of negative infinity remains unchanged under positive "
    "temperature scaling. Before asserting outputs differ, establish that the fixture admits "
    "a finite nonzero observable value, and that its vocabulary represents valid grammar tokens. "
    "Check mock/context-manager teardown separately from the behavior assertion: a successful "
    "assertion followed by mock.__exit__ failing to delete a dynamically provided attribute "
    "is a fixture lifecycle error, not evidence that the requested behavior failed. Use the "
    "public configuration/context protocol rather than assuming arbitrary setattr/delattr support. "
    "Do not mock away the publicly required side effect, catch unexpected failures, "
    "weaken an assertion, or alter product code to accommodate an invalid fixture. "
    "If these prerequisites are unsupported or unknown, correct or investigate the probe; "
    "do not infer validity merely from a quote naming the requested feature."
)


def _literal_call_arguments(tree: ast.AST | None, code: str) -> list[dict]:
    """Show decoded constants, without evaluating dynamic probe expressions."""
    rows, remaining = [], 8192
    for node in ast.walk(tree) if tree is not None else []:
        if not isinstance(node, ast.Call):
            continue
        arguments = [(str(i), arg) for i, arg in enumerate(node.args)]
        arguments += [(kw.arg, kw.value) for kw in node.keywords if kw.arg is not None]
        for argument, value in arguments:
            if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
                continue
            if len(rows) >= 8 or remaining <= 0:
                return rows
            text = value.value[:min(2048, remaining)]
            remaining -= len(text)
            rows.append({"line": node.lineno,
                         "callee": (ast.get_source_segment(code, node.func) or "")[:200],
                         "argument": argument, "decoded_value": text,
                         "python_repr": repr(text), "complete": len(text) == len(value.value),
                         "origin": "probe_ast_string_constant"})
    return rows


def public_prerequisite_context(world: Any, probe: Mapping[str, Any],
                                failure_site: Mapping[str, Any]) -> dict:
    baseline = getattr(world, "_cooperbench_public_baseline_files", {}) or {}
    if not isinstance(baseline, Mapping):
        baseline = {}
    code = str(probe.get("code") or "")
    tokens = set(re.findall(r"[A-Za-z_]\w*", code))
    paths = []
    # Fixed probe paths and captured frames only select from the public mapping;
    # an absolute/foreign/hidden path can never trigger a filesystem read.
    # Tracebacks list outer callers first. Prioritize the actual failing leaf
    # so adapter/fixture evidence is not displaced by four generic wrappers.
    for frame in reversed(failure_site.get("frames") or []):
        path = str(frame.get("path") or "")
        if path.startswith(("<candidate>/", "<baseline>/")):
            paths.append(path.split("/", 1)[1])
    # Missing-argument TypeError is raised before entering the callee, so its
    # definition has no traceback frame. Recover bounded original signatures
    # for methods actually called by the probe, from the public mapping only.
    try:
        call_tree = ast.parse(code)
    except SyntaxError:
        call_tree = None
    method_names = {
        node.func.attr for node in ast.walk(call_tree) if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
    } if call_tree is not None else set()
    method_names = set(sorted(method_names)[:24])
    cache_configuration_call = any(
        isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'configure'
        and any(k.arg in {'enable_memory_cache','enable_disk_cache','cache_dir','disk_cache_dir'} for k in n.keywords)
        for n in ast.walk(call_tree)) if call_tree is not None else False
    if cache_configuration_call:
        method_names.add('configure_cache')
    signature_paths = []
    scanned_chars = 0
    for path, content in sorted(baseline.items()):
        if not isinstance(path, str) or not path.endswith('.py') or not isinstance(content, str):
            continue
        scanned_chars += len(content)
        if scanned_chars > 3_000_000 or len(signature_paths) >= (3 if cache_configuration_call else 2):
            break
        if any(name in method_names for name in re.findall(r'^\s*(?:async\s+)?def\s+(\w+)\s*\(', content, re.MULTILINE)):
            signature_paths.append(path)
    paths += signature_paths
    paths += list(probe.get("paths") or [])
    files, remaining = [], 48_000
    for path in dict.fromkeys(paths):
        content = baseline.get(path)
        if not isinstance(content, str) or not content or len(files) >= 4:
            continue
        lines = content.splitlines(keepends=True)
        selected = set(range(min(24, len(lines))))
        if len(content) <= 16_000:
            selected.update(range(len(lines)))
        else:
            # Keep exact source, with declared ranges, around used definitions,
            # decorator declarations and the reached original frame. No prose
            # generated from a candidate patch can become prerequisite evidence.
            try:
                tree = ast.parse(content)
            except SyntaxError:
                tree = None
            for node in ast.walk(tree) if tree is not None else []:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    if not any(node.name == term or node.name.endswith("_" + term)
                               for term in tokens):
                        continue
                    start = min([node.lineno] + [d.lineno for d in node.decorator_list]) - 1
                    selected.update(range(max(0, start), min(len(lines), node.end_lineno)))
            for frame in failure_site.get("frames") or []:
                if str(frame.get("path") or "").split("/", 1)[-1] == path:
                    at = int(frame.get("line") or 1) - 1
                    selected.update(range(max(0, at - 12), min(len(lines), at + 13)))
        snippets, per_file, start, group = [], 0, None, []
        for index in sorted(selected):
            line = lines[index]
            if per_file + len(line) > 16_000 or len(line) > remaining:
                break
            if start is not None and index != start + len(group):
                snippets.append({"start_line": start + 1, "text": "".join(group)})
                start, group = None, []
            if start is None:
                start = index
            group.append(line)
            per_file += len(line)
            remaining -= len(line)
        if group:
            snippets.append({"start_line": start + 1, "text": "".join(group)})
        files.append({"path": path, "snippets": snippets,
                      "complete": per_file == len(content)})
    return {"origin": "untouched_public_starter_source", "files": files,
            "literal_call_arguments": _literal_call_arguments(call_tree, code),
            "scope": "Calling convention and fixture context only; the new public request remains authoritative."}
