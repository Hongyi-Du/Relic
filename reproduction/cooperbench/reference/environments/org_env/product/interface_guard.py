"""IDE-like cross-file interface guard (self-iteration §).

The OSS runs got stuck in an interface whack-a-mole: agents editing a heavily-shared module
(``git_utils.py``, imported by cloning.py / query_parsing.py / ...) kept DELETING or renaming public
symbols that other modules ``import`` — so the working tree cycled through ImportErrors
(``create_git_auth_header`` -> fix -> ``fetch_remote_branch_list`` -> ...), CI stayed red, no PR ever
merged.

This module gives the editor + validator the two things an IDE would: (1) "find references" — which of
a file's public symbols are imported by other modules; (2) "broken reference" detection — a patch that
drops such a symbol. The validator rejects those patches (break never reaches the tree); the editor is
told which symbols to preserve.
"""
from __future__ import annotations

import ast
import os
from typing import Any, Set

from environments.org_env.cooperbench.source_reads import (
    actor_source_read_view, visible_source_artifacts,
)
from environments.org_env.cooperbench.source_views import SourceViewError


def _text(artifact, view):
    return view.text(artifact) if view is not None else (getattr(artifact, "content", "") or "")


def _module_base(linked_file_path: str) -> str:
    """The module basename other files import by, e.g. 'src/gitingest/utils/git_utils.py' -> 'git_utils'."""
    fp = (linked_file_path or "").replace("\\", "/")
    if not fp.endswith(".py"):
        return ""
    return os.path.basename(fp)[:-3]


def public_defs(src: str) -> Set[str]:
    """Top-level public (non ``_``-prefixed) names DEFINED in ``src`` — funcs, classes, assignments."""
    try:
        tree = ast.parse(src or "")
    except SyntaxError:
        return set()
    names: Set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if not node.name.startswith("_"):
                names.add(node.name)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and not t.id.startswith("_"):
                    names.add(t.id)
    return names


def _imports_by_module_base(src: str) -> dict[str, Set[str]]:
    """Parse one source file into its ``from module import name`` index."""
    try:
        tree = ast.parse(src or "")
    except SyntaxError:
        return {}
    out: dict[str, Set[str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            module_base = node.module.split(".")[-1]
            names = out.setdefault(module_base, set())
            names.update(alias.name for alias in node.names if alias.name != "*")
    return out


def imports_from_module(src: str, module_base: str) -> Set[str]:
    """Names imported via ``from <...module_base> import a, b`` in ``src`` (matches by module basename,
    so absolute/relative import styles all resolve). ``import x as y`` attribute use isn't tracked —
    the ImportError failure mode is precisely the ``from M import NAME`` form."""
    if not module_base:
        return set()
    return set(_imports_by_module_base(src).get(module_base, ()))


def _symbols_imported_elsewhere(world: Any, art: Any, view=None) -> Set[str]:
    base = _module_base(getattr(art, "linked_file_path", "") or "")
    if not base:
        return set()
    used: Set[str] = set()
    for a in visible_source_artifacts(world, view):
        if a is art:
            continue
        ofp = getattr(a, "linked_file_path", "") or ""
        if ofp.endswith(".py"):
            used |= imports_from_module(_text(a, view), base)
    return used


def depended_upon_symbols(world: Any, art: Any, *, actor_id=None, source_snapshot=None) -> Set[str]:
    """Public symbols this file DEFINES that some OTHER module imports (the file's cross-file API)."""
    view = actor_source_read_view(world, actor_id, source_snapshot=source_snapshot)
    mine = public_defs(_text(art, view))
    if not mine:
        return set()
    return mine & _symbols_imported_elsewhere(world, art, view)


def broken_cross_file_symbols(world: Any, art: Any, new_content: str, *, actor_id=None, source_snapshot=None) -> Set[str]:
    """Depended-upon public symbols that ``new_content`` would DELETE/rename away (would break importers)."""
    view = actor_source_read_view(world, actor_id, source_snapshot=source_snapshot)
    removed = public_defs(_text(art, view)) - public_defs(new_content or "")
    if not removed:
        return set()
    return removed & _symbols_imported_elsewhere(world, art, view)


def importers_needing_update(world: Any, art: Any, new_content: str, *, actor_id=None, source_snapshot=None,
                            creates_file=False) -> "dict":
    """{importer_artifact_id: sorted[removed symbols it imports]} — the caller files that must be updated
    because this edit DROPS a public symbol they import. Drives the IDE-like "update references" cascade:
    instead of hard-rejecting the removal, we allow it and coordinate the importer fixes."""
    view = actor_source_read_view(world, actor_id, source_snapshot=source_snapshot)
    current = view.text(art, creates_file=creates_file) if view is not None else _text(art, None)
    removed = public_defs(current) - public_defs(new_content or "")
    base = _module_base(getattr(art, "linked_file_path", "") or "")
    if not removed or not base:
        return {}
    out = {}
    for a in visible_source_artifacts(world, view):
        if a is art:
            continue
        ofp = getattr(a, "linked_file_path", "") or ""
        if not ofp.endswith(".py"):
            continue
        hit = removed & imports_from_module(_text(a, view), base)
        if hit:
            out[getattr(a, "artifact_id", "")] = sorted(hit)
    return out


# --- the mirror case: an edit that ADDS a parameter -------------------------
#
# ``importers_needing_update`` handles a symbol being removed, which breaks
# callers loudly. Adding a parameter fails the opposite way: the file compiles,
# every test still passes, and the value simply never reaches the code that was
# supposed to act on it.
#
# Measured on a real run: the organization added ``include_submodules`` to
# ``ingest`` and ``ingest_async`` - the correct name, in the correct file - and
# never passed it to ``clone_repo``, which is where the hidden oracle reads it
# off the clone config. The fix was right and unfinished, nothing said so, and
# the edit driver deals one module per issue so the second half was unreachable.


def _function_defs(src: str) -> "dict":
    """Top-level and nested function definitions by name."""
    try:
        tree = ast.parse(src or "")
    except SyntaxError:
        return {}
    out = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out[node.name] = node
    return out


def _parameter_names(node: Any) -> Set[str]:
    args = node.args
    collected = list(args.posonlyargs) + list(args.args) + list(args.kwonlyargs)
    names = {a.arg for a in collected}
    for extra in (args.vararg, args.kwarg):
        if extra is not None:
            names.add(extra.arg)
    return names


def newly_added_parameters(old_src: str, new_src: str) -> "dict":
    """{function name: parameters the edit added} for functions present in both."""
    old_defs, new_defs = _function_defs(old_src), _function_defs(new_src)
    out = {}
    for name, node in new_defs.items():
        if name not in old_defs:
            continue           # a brand-new function forwards nothing yet by definition
        added = _parameter_names(node) - _parameter_names(old_defs[name])
        if added:
            out[name] = added
    return out


def _called_symbols(node: Any) -> "dict":
    """{called name: whether any call passed each argument name} within a function body."""
    calls = {}
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        func = child.func
        if isinstance(func, ast.Name):
            name = func.id
        elif isinstance(func, ast.Attribute):
            name = func.attr
        else:
            continue
        passed = calls.setdefault(name, set())
        for arg in child.args:
            if isinstance(arg, ast.Name):
                passed.add(arg.id)
        for keyword in child.keywords:
            if keyword.arg:
                passed.add(keyword.arg)
            if isinstance(keyword.value, ast.Name):
                passed.add(keyword.value.id)
    return calls


def unforwarded_parameters(new_src: str, *, added: "dict", cross_module_symbols: Set[str]) -> "dict":
    """{parameter: sorted callees that could have received it and did not}.

    A parameter counts as consumed when the function invokes it directly (the
    common callback-hook pattern), or as forwarded as soon as ANY cross-module
    call in the same function receives it. The point is to notice a value that
    goes nowhere, not to rewrite a locally implemented behavior into a callee.

    Only calls into OTHER modules are considered. A parameter consumed entirely
    within its own file is a normal local flag, and flagging those would bury
    the real case in noise.
    """
    if not added or not cross_module_symbols:
        return {}
    defs = _function_defs(new_src)
    out = {}
    for func_name, params in added.items():
        node = defs.get(func_name)
        if node is None:
            continue
        calls = _called_symbols(node)
        reachable = {name: passed for name, passed in calls.items() if name in cross_module_symbols}
        if not reachable:
            continue
        for param in sorted(params):
            if param in calls or any(param in passed for passed in reachable.values()):
                continue
            out.setdefault(param, set()).update(reachable)
    return {param: sorted(callees) for param, callees in out.items()}


def imported_symbol_sources(world: Any, art: Any, *, actor_id=None, source_snapshot=None,
                            _source_view=None) -> "dict":
    """{symbol this file imports from another product module: that module's artifact id}."""
    if _source_view is None:
        view = actor_source_read_view(world, actor_id, source_snapshot=source_snapshot)
    else:
        view = _source_view
        if (getattr(view, "actor_id", None) != actor_id
                or (source_snapshot is not None and getattr(view, "snapshot", None) != source_snapshot)):
            raise SourceViewError("actor_source_view_mismatch")
    src = _text(art, view)
    imported = _imports_by_module_base(src)
    if not imported:
        return {}
    out = {}
    for other in visible_source_artifacts(world, view):
        if other is art:
            continue
        ofp = getattr(other, "linked_file_path", "") or ""
        if not ofp.endswith(".py"):
            continue
        base = _module_base(ofp)
        for symbol in imported.get(base, ()):
            out[symbol] = getattr(other, "artifact_id", "")
    return out


def module_base(linked_file_path: str) -> str:
    return _module_base(linked_file_path)


__all__ = ["public_defs", "imports_from_module", "depended_upon_symbols",
           "broken_cross_file_symbols", "importers_needing_update", "module_base",
           "newly_added_parameters", "unforwarded_parameters", "imported_symbol_sources"]
