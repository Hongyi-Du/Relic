"""Score a candidate release by how much of the next version's FUNCTION it hits.

The hidden oracles grade an implementation route. Three of the eight a measured
run failed require a specific new module name, a specific subprocess call, or a
specific logging mechanism - facts the visible issue never states and that no
correct implementation is obliged to match. The organization built working
alternatives for all three and scored zero.

This measures the other thing: of the user-reachable capabilities the reference
release ADDED over the starter, how many does the candidate also expose. It
needs no hand-authored oracle, so it works on any repository for which two
releases exist - which is what lets the benchmark generalise, since authoring
oracles is exactly the step that does not scale.

"User-reachable" is the load-bearing choice. A first attempt counted every
public symbol in the tree and reported 0% for a candidate that had genuinely
landed the release's headline option: the reference had 87 new symbols, all
internal architecture, while its declared surface (``__all__``) had NARROWED
from five names to two. Counting internals measures refactoring, not function.
The surface here is what a user can actually touch:

* Python: names declared in ``__all__`` (resolved to their definitions for
  parameter lists), CLI options, HTTP route decorators;
* JavaScript/TypeScript: exports of the files ``package.json`` points at.

Capabilities are atomic - an entry point is one, each of its parameters is one,
a CLI flag is one - so a candidate that landed one of the two options a release
added scores half, not zero. Location is deliberately discarded: moving a
function between modules changes no capability, and threading a parameter by a
different internal route changes none either. That is what "functional hit,
any route" means.

What this cannot see is a behaviour change with no surface change (an internal
bugfix). That is a real limit, stated rather than papered over; behavioural
coverage (the test suites) sits beside this score, not under it.
"""
from __future__ import annotations

import ast
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

__all__ = [
    "FUNCTIONAL_OVERLAP_SCHEMA_VERSION",
    "Capability",
    "FunctionalOverlapReport",
    "extract_surface",
    "functional_overlap",
]

FUNCTIONAL_OVERLAP_SCHEMA_VERSION = "functional_overlap_v2"

_PY_SUFFIXES = (".py",)
_JS_SUFFIXES = (".js", ".mjs", ".cjs", ".ts")

# Not the product: symbols under these would inflate both sides of the
# comparison and drown the release's real surface.
_SKIPPED_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build",
    ".tox", ".mypy_cache", ".pytest_cache", "site-packages", "coverage",
    "tests", "test", "spec", "__tests__", "docs", "examples", "benchmarks",
}

_ROUTE_METHODS = {"get", "post", "put", "delete", "patch", "head", "options"}
_CLI_REGISTRARS = {"option", "argument", "add_argument"}


@dataclass(frozen=True, order=True)
class Capability:
    """One atomic user-reachable thing a release can do."""

    kind: str      # entry | param | cli | route | export
    symbol: str    # e.g. "parse", "parse.max_depth", "--output", "GET /api/x"

    def key(self) -> str:
        return f"{self.kind}:{self.symbol}"


def _is_public(name: str) -> bool:
    return bool(name) and not name.startswith("_")


def _skip(path: Path, root: Path) -> bool:
    parts = path.relative_to(root).parts
    if any(part in _SKIPPED_DIRS for part in parts[:-1]):
        return True
    name = path.name
    return name.startswith("test_") or name.endswith(
        ("_test.py", ".test.js", ".spec.js", ".test.ts", ".spec.ts"))


def _iter_source_files(root: Path, suffixes: tuple[str, ...]) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix in suffixes and not _skip(path, root):
            yield path


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


# --------------------------------------------------------------------------
# Python: definitions index, __all__ resolution, CLI options, HTTP routes
# --------------------------------------------------------------------------

def _callable_params(node: ast.AST) -> tuple[str, ...]:
    args = node.args
    names = [a.arg for a in list(args.posonlyargs) + list(args.args) + list(args.kwonlyargs)]
    return tuple(n for n in names if n not in ("self", "cls"))


def _definitions(tree: ast.AST) -> dict[str, tuple[str, ...]]:
    """{public top-level name: parameter names} for one parsed module."""
    out: dict[str, tuple[str, ...]] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if _is_public(node.name):
                out[node.name] = _callable_params(node)
        elif isinstance(node, ast.ClassDef) and _is_public(node.name):
            init = next(
                (item for item in node.body
                 if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                 and item.name == "__init__"),
                None)
            out[node.name] = _callable_params(init) if init is not None else ()
    return out


def _declared_all(tree: ast.AST) -> list[str] | None:
    """The module's ``__all__`` names, or None when it declares none.

    The distinction matters: ``__all__ = []`` is a deliberate empty surface and
    must not trigger the everything-public fallback, while an absent ``__all__``
    means the module never said what its surface is."""
    for node in tree.body:
        targets = []
        if isinstance(node, ast.Assign):
            targets = [t for t in node.targets if isinstance(t, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            targets = [node.target]
        if not any(t.id == "__all__" for t in targets):
            continue
        value = node.value
        if isinstance(value, (ast.List, ast.Tuple)):
            return [e.value for e in value.elts
                    if isinstance(e, ast.Constant) and isinstance(e.value, str)]
    return None


def _import_sources(tree: ast.AST) -> dict[str, str]:
    """{imported-as name: dotted module} from ``from M import a as b`` forms."""
    out: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                out[alias.asname or alias.name] = node.module
    return out


def _module_file(root: Path, dotted: str, files: Mapping[Path, ast.AST]) -> Path | None:
    """The repo file a dotted module name refers to, by path suffix match."""
    tail = dotted.replace(".", "/")
    for candidate in files:
        rel = candidate.relative_to(root).as_posix()
        if rel.endswith(f"{tail}.py") or rel.endswith(f"{tail}/__init__.py"):
            return candidate
    return None


def _decorator_capabilities(tree: ast.AST) -> set[Capability]:
    """CLI options and HTTP routes, from their registration calls.

    Both are user-reachable no matter what ``__all__`` says: a flag exists for
    whoever runs the binary and a route for whoever holds an HTTP client.
    """
    out: set[Capability] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        attr = node.func.attr
        first = node.args[0].value if (
            node.args and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)) else None
        if first is None:
            continue
        if attr in _CLI_REGISTRARS and first.startswith("-"):
            # normalise to the long form when several spellings are given
            spellings = [first] + [
                a.value for a in node.args[1:]
                if isinstance(a, ast.Constant) and isinstance(a.value, str)
                and a.value.startswith("-")]
            out.add(Capability("cli", max(spellings, key=len)))
        elif attr in _ROUTE_METHODS and first.startswith("/"):
            out.add(Capability("route", f"{attr.upper()} {first}"))
    return out


def _python_surface(root: Path) -> tuple[set[Capability], list[str], bool]:
    files: dict[Path, ast.AST] = {}
    for path in _iter_source_files(root, _PY_SUFFIXES):
        try:
            files[path] = ast.parse(_read(path))
        except SyntaxError:
            continue

    surface: set[Capability] = set()
    caveats: list[str] = []
    defs_by_file = {path: _definitions(tree) for path, tree in files.items()}

    def add_entry(name: str, params: tuple[str, ...]) -> None:
        surface.add(Capability("entry", name))
        for param in params:
            surface.add(Capability("param", f"{name}.{param}"))

    declared = False
    for path, tree in files.items():
        if path.name != "__init__.py":
            continue
        exported = _declared_all(tree)
        if exported is None:
            continue
        declared = True
        imports = _import_sources(tree)
        local = defs_by_file.get(path, {})
        for name in exported:
            if not _is_public(name):
                continue
            params: tuple[str, ...] = ()
            if name in local:
                params = local[name]
            else:
                source_file = None
                if name in imports:
                    source_file = _module_file(root, imports[name], files)
                if source_file is not None and name in defs_by_file.get(source_file, {}):
                    params = defs_by_file[source_file][name]
                else:
                    owners = [d for d in defs_by_file.values() if name in d]
                    if len(owners) == 1:
                        params = owners[0][name]
            add_entry(name, params)

    if not declared and files:
        # No declared API anywhere: fall back to every public top-level
        # definition. Coarser, and said so - on a repo that curates __all__ the
        # fallback would count internals and repeat the 87-symbol mistake.
        caveats.append("python_surface_fallback_no_declared_all")
        for defs in defs_by_file.values():
            for name, params in defs.items():
                add_entry(name, params)

    for tree in files.values():
        surface |= _decorator_capabilities(tree)
    return surface, caveats, bool(files)


# --------------------------------------------------------------------------
# JavaScript / TypeScript: exports of the files package.json points at
# --------------------------------------------------------------------------

_JS_EXPORT_FN = re.compile(r"export\s+(?:async\s+)?function\s+(\w+)\s*\(([^)]*)\)")
_JS_EXPORT_DECL = re.compile(r"export\s+(?:const|let|var|class)\s+(\w+)")
_JS_NAMED_EXPORT = re.compile(r"export\s*\{([^}]*)\}")
_CJS_NAMED = re.compile(r"(?:module\.)?exports\.(\w+)\s*=")
_CJS_OBJECT = re.compile(r"module\.exports\s*=\s*\{([^}]*)\}")
_CJS_FN = re.compile(r"module\.exports\s*=\s*(?:async\s+)?function\s+(\w+)\s*\(([^)]*)\)")


def _js_params(raw: str) -> tuple[str, ...]:
    out = []
    for piece in raw.split(","):
        name = piece.strip().split("=")[0].strip().strip("{}").strip()
        if name and re.fullmatch(r"\w+", name):
            out.append(name)
    return tuple(out)


def _js_file_exports(source: str) -> set[Capability]:
    out: set[Capability] = set()
    for match in _JS_EXPORT_FN.finditer(source):
        out.add(Capability("export", match.group(1)))
        for param in _js_params(match.group(2)):
            out.add(Capability("param", f"{match.group(1)}.{param}"))
    for match in _JS_EXPORT_DECL.finditer(source):
        out.add(Capability("export", match.group(1)))
    for match in _JS_NAMED_EXPORT.finditer(source):
        for raw in match.group(1).split(","):
            name = raw.split(" as ")[-1].strip()
            if _is_public(name):
                out.add(Capability("export", name))
    for match in _CJS_NAMED.finditer(source):
        if _is_public(match.group(1)):
            out.add(Capability("export", match.group(1)))
    for match in _CJS_OBJECT.finditer(source):
        for raw in match.group(1).split(","):
            name = raw.split(":")[0].strip()
            if name and re.fullmatch(r"\w+", name) and _is_public(name):
                out.add(Capability("export", name))
    for match in _CJS_FN.finditer(source):
        out.add(Capability("export", match.group(1)))
        for param in _js_params(match.group(2)):
            out.add(Capability("param", f"{match.group(1)}.{param}"))
    return out


def _package_entry_files(root: Path) -> list[Path]:
    manifest_path = root / "package.json"
    if not manifest_path.is_file():
        return []
    try:
        manifest = json.loads(_read(manifest_path))
    except json.JSONDecodeError:
        return []

    raw: list[str] = []

    def collect(value: Any) -> None:
        if isinstance(value, str):
            raw.append(value)
        elif isinstance(value, Mapping):
            for item in value.values():
                collect(item)

    for field_name in ("main", "module", "browser", "exports"):
        collect(manifest.get(field_name))
    out: list[Path] = []
    for entry in raw:
        path = (root / entry.lstrip("./")).resolve()
        if path.is_file() and path.suffix in _JS_SUFFIXES and path not in out:
            out.append(path)
    return out


def _js_surface(root: Path, *, fallback_ok: bool) -> tuple[set[Capability], list[str]]:
    entries = _package_entry_files(root)
    caveats: list[str] = []
    if not entries:
        # Scanning every JS file is only meaningful for a JS-primary repo. A
        # Python project's stray static scripts are not its API, and folding
        # them in would grow both sides of the comparison with noise.
        if not fallback_ok:
            return set(), caveats
        entries = list(_iter_source_files(root, _JS_SUFFIXES))
        if entries:
            caveats.append("js_surface_fallback_no_package_entries")
    surface: set[Capability] = set()
    for path in entries:
        surface |= _js_file_exports(_read(path))
    return surface, caveats


# --------------------------------------------------------------------------
# The score
# --------------------------------------------------------------------------

def extract_surface(repo_dir: str | Path) -> tuple[set[Capability], list[str]]:
    """Every user-reachable capability of a repository tree, plus caveats."""
    root = Path(repo_dir).expanduser().resolve()
    if not root.is_dir():
        return set(), ["repo_dir_missing"]
    python, py_caveats, has_python = _python_surface(root)
    js, js_caveats = _js_surface(root, fallback_ok=not has_python)
    return python | js, py_caveats + js_caveats


@dataclass
class FunctionalOverlapReport:
    schema_version: str = FUNCTIONAL_OVERLAP_SCHEMA_VERSION
    new_capability_count: int = 0
    hit_count: int = 0
    overlap_rate: float = 0.0
    hit: tuple[str, ...] = field(default_factory=tuple)
    missed: tuple[str, ...] = field(default_factory=tuple)
    by_kind: dict[str, dict[str, int]] = field(default_factory=dict)
    starter_surface_size: int = 0
    reference_surface_size: int = 0
    candidate_surface_size: int = 0
    caveats: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def functional_overlap(
    *,
    starter_dir: str | Path,
    reference_dir: str | Path,
    candidate_dir: str | Path,
) -> FunctionalOverlapReport:
    """Of what the reference release added to the surface, how much the candidate has.

    The denominator is what the next release ADDED, never what it merely kept:
    crediting inherited capabilities would score doing nothing at close to 100%.
    Removals are likewise not scored - a narrowed surface is a design decision,
    not a capability the candidate can hit.
    """
    starter, starter_caveats = extract_surface(starter_dir)
    reference, reference_caveats = extract_surface(reference_dir)
    candidate, candidate_caveats = extract_surface(candidate_dir)

    caveats = [f"starter:{c}" for c in starter_caveats]
    caveats += [f"reference:{c}" for c in reference_caveats]
    caveats += [f"candidate:{c}" for c in candidate_caveats]
    if not reference:
        caveats.append("reference_surface_empty")

    starter_keys = {c.key() for c in starter}
    candidate_keys = {c.key() for c in candidate}
    new = {c for c in reference if c.key() not in starter_keys}
    hit = {c for c in new if c.key() in candidate_keys}

    by_kind: dict[str, dict[str, int]] = {}
    for capability in new:
        row = by_kind.setdefault(capability.kind, {"new": 0, "hit": 0})
        row["new"] += 1
        if capability in hit:
            row["hit"] += 1

    return FunctionalOverlapReport(
        new_capability_count=len(new),
        hit_count=len(hit),
        overlap_rate=round(len(hit) / len(new), 6) if new else 0.0,
        hit=tuple(sorted(c.key() for c in hit)),
        missed=tuple(sorted(c.key() for c in new - hit)),
        by_kind={k: by_kind[k] for k in sorted(by_kind)},
        starter_surface_size=len(starter),
        reference_surface_size=len(reference),
        candidate_surface_size=len(candidate),
        caveats=tuple(caveats),
    )
