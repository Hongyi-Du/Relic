"""Repository intelligence fabric for Code-Max runs."""

from __future__ import annotations

import ast
import json
import posixpath
import re
import warnings
from pathlib import Path

from society_core.hashing import stable_hash
from society_core.safe_files import UnsafeRegularFileError, read_regular_file_bytes

from .environment import build_workspace_execution_profile
from .schemas import RepoIntelligenceFabric, SymbolLocation, WorkspaceExecutionProfile

THEME_TERMS: dict[str, tuple[str, ...]] = {
    "architecture_generalization": (
        "architecture",
        "pipeline",
        "source",
        "config",
        "entry",
    ),
    "dependency_resolution": ("dependency", "import", "resolve", "module", "package"),
    "diagnostics_and_recovery": ("error", "diagnostic", "recover", "message", "cli"),
    "documentation_and_migration_path": ("readme", "docs", "migration", "init", "help"),
    "framework_generalization": ("framework", "adapter", "plugin", "source"),
    "plugin_ecosystem": ("plugin", "hook", "transform", "extension"),
    "production_build_reliability": ("build", "bundle", "optimize", "production"),
    "path_security": ("path", "safe", "resolve", "traversal", "security"),
}


def build_repo_intelligence_fabric(
    workspace_root: Path,
    *,
    execution_profile: WorkspaceExecutionProfile | None = None,
    themes: tuple[str, ...] = (),
    priority_paths: tuple[str, ...] = (),
    max_files: int = 600,
) -> RepoIntelligenceFabric:
    root = workspace_root.resolve()
    profile = execution_profile or build_workspace_execution_profile(root)
    text_files = _text_files(
        root,
        max_files=max_files,
        priority_paths=priority_paths,
    )
    file_tree = tuple(path for path, _ in text_files)
    source_roots = tuple(
        path
        for path in ("src", "lib", "packages", "dist-src", "dist")
        if (root / path).exists()
    )
    test_roots = tuple(
        path for path in ("test", "tests", "__tests__") if (root / path).exists()
    )
    config_files = tuple(path for path in file_tree if _is_config(path))
    docs_files = tuple(path for path in file_tree if _is_docs(path))
    generated_files = tuple(path for path in file_tree if _is_generated(path))
    entrypoints = _entrypoints(root, file_tree)
    symbol_index = _symbol_index(text_files)
    import_graph = _import_graph(text_files)
    reverse_graph = _reverse_graph(import_graph)
    source_to_tests = _source_to_tests(file_tree)
    test_to_sources = _test_to_sources(source_to_tests)
    public_api_surfaces = _public_api_surfaces(entrypoints, symbol_index, docs_files)
    active_themes = themes or tuple(THEME_TERMS)
    candidate_files_by_theme = {
        theme: _candidate_files_for_theme(theme, text_files, public_api_surfaces)
        for theme in active_themes
    }
    payload = {
        "profile": profile.repo_hash,
        "files": file_tree,
        "symbols": tuple(symbol_index),
        "entrypoints": entrypoints,
        "public_api": public_api_surfaces,
    }
    graph_hash = stable_hash(payload)
    return RepoIntelligenceFabric(
        repo_hash=stable_hash({"profile": profile.repo_hash, "graph": graph_hash}),
        file_tree=file_tree,
        source_roots=source_roots,
        test_roots=test_roots,
        config_files=config_files,
        docs_files=docs_files,
        generated_files=generated_files,
        package_manager=profile.package_manager,
        entrypoints=entrypoints,
        public_api_surfaces=public_api_surfaces,
        symbol_index=symbol_index,
        import_graph=import_graph,
        reverse_dependency_graph=reverse_graph,
        source_to_tests=source_to_tests,
        test_to_sources=test_to_sources,
        coverage_map=source_to_tests,
        error_patterns=_error_patterns(text_files),
        style_conventions=_style_conventions(profile.package_manager, file_tree),
        architecture_notes=_architecture_notes(source_roots, public_api_surfaces),
        semantic_search_index_id=f"lexical_{graph_hash[:16]}",
        code_knowledge_graph_id=f"code_graph_{graph_hash[:16]}",
        candidate_files_by_theme=candidate_files_by_theme,
    )


def _text_files(
    root: Path,
    *,
    max_files: int,
    priority_paths: tuple[str, ...] = (),
) -> tuple[tuple[str, str], ...]:
    files: list[tuple[str, str]] = []
    prioritized: list[Path] = []
    for raw_path in priority_paths:
        relative = Path(str(raw_path))
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            continue
        path = root / relative
        if path.is_file() and not _ignored(path, root):
            prioritized.append(path)
    ordered_paths = (
        *tuple(dict.fromkeys(prioritized)),
        *tuple(path for path in sorted(root.rglob("*")) if path not in prioritized),
    )
    for path in ordered_paths:
        if len(files) >= max_files:
            break
        if _ignored(path, root):
            continue
        try:
            raw = read_regular_file_bytes(root, path, max_bytes=300_000)
        except (OSError, UnsafeRegularFileError):
            continue
        if b"\x00" in raw[:4096]:
            continue
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError:
            continue
        files.append((relative, raw.decode("utf-8", errors="replace")))
    return tuple(files)


def _entrypoints(root: Path, file_tree: tuple[str, ...]) -> tuple[str, ...]:
    entries: list[str] = []
    package_path = root / "package.json"
    if package_path.exists():
        try:
            package = json.loads(
                read_regular_file_bytes(root, package_path, max_bytes=2_000_000).decode(
                    "utf-8"
                )
            )
        except (json.JSONDecodeError, UnicodeDecodeError, UnsafeRegularFileError):
            package = {}
        for key in ("main", "module", "browser", "types"):
            value = package.get(key)
            if isinstance(value, str) and value in file_tree:
                entries.append(value)
        bin_value = package.get("bin")
        if isinstance(bin_value, str) and bin_value in file_tree:
            entries.append(bin_value)
        elif isinstance(bin_value, dict):
            entries.extend(
                value
                for value in bin_value.values()
                if isinstance(value, str) and value in file_tree
            )
    for candidate in (
        "src/index.py",
        "src/index.js",
        "src/index.ts",
        "index.js",
        "index.py",
        "README.md",
    ):
        if candidate in file_tree:
            entries.append(candidate)
    return tuple(dict.fromkeys(entries))[:20]


def _symbol_index(
    text_files: tuple[tuple[str, str], ...],
) -> dict[str, tuple[SymbolLocation, ...]]:
    index: dict[str, list[SymbolLocation]] = {}
    patterns = (
        (re.compile(r"^\s*(?:export\s+)?function\s+([A-Za-z_][\w$]*)"), "function"),
        (re.compile(r"^\s*(?:export\s+)?class\s+([A-Za-z_][\w$]*)"), "class"),
        (re.compile(r"^\s*def\s+([A-Za-z_]\w*)"), "function"),
        (re.compile(r"^\s*class\s+([A-Za-z_]\w*)"), "class"),
        (re.compile(r"^\s*(?:export\s+)?const\s+([A-Za-z_][\w$]*)"), "constant"),
    )
    for path, text in text_files:
        for line_no, line in enumerate(text.splitlines(), start=1):
            for pattern, kind in patterns:
                match = pattern.search(line)
                if match:
                    symbol = match.group(1)
                    index.setdefault(symbol, []).append(
                        SymbolLocation(symbol, path, line_no, kind)
                    )
    return {symbol: tuple(locations) for symbol, locations in sorted(index.items())}


def _import_graph(
    text_files: tuple[tuple[str, str], ...],
) -> dict[str, tuple[str, ...]]:
    """Resolve internal imports to repository-relative paths."""

    file_tree = tuple(path for path, _ in text_files)
    file_set = set(file_tree)
    python_modules = _python_module_index(file_tree)
    graph: dict[str, tuple[str, ...]] = {}
    for path, source in text_files:
        if path.endswith(".py"):
            targets = _python_import_targets(path, source, python_modules)
        elif path.endswith((".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx")):
            targets = _javascript_import_targets(path, source, file_set)
        else:
            targets = ()
        graph[path] = tuple(target for target in targets if target != path)
    return graph


def _python_module_index(file_tree: tuple[str, ...]) -> dict[str, str]:
    index: dict[str, str] = {}
    for path in file_tree:
        if not path.endswith(".py"):
            continue
        parts = list(Path(path).with_suffix("").parts)
        if parts and parts[-1] == "__init__":
            parts.pop()
        if not parts:
            continue
        aliases = [".".join(parts)]
        if parts[0] in {"src", "lib"} and len(parts) > 1:
            aliases.append(".".join(parts[1:]))
        for alias in aliases:
            index.setdefault(alias, path)
    return index


def _python_import_targets(
    path: str,
    source: str,
    module_index: dict[str, str],
) -> tuple[str, ...]:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(source)
    except SyntaxError:
        return ()
    module = _python_module_name(path)
    package = module if path.endswith("/__init__.py") else module.rpartition(".")[0]
    targets: list[str] = []
    for node in ast.walk(tree):
        modules: list[str] = []
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = _absolute_python_import(package, node.module, node.level)
            if base:
                modules.append(base)
            for alias in node.names:
                child = f"{base}.{alias.name}" if base else alias.name
                if child in module_index:
                    modules.append(child)
        for imported_module in modules:
            target = module_index.get(imported_module)
            if target and target not in targets:
                targets.append(target)
    return tuple(targets)


def _python_module_name(path: str) -> str:
    parts = list(Path(path).with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _absolute_python_import(
    package: str,
    imported_module: str | None,
    level: int,
) -> str:
    if level <= 0:
        return imported_module or ""
    package_parts = package.split(".") if package else []
    keep = max(0, len(package_parts) - (level - 1))
    prefix = package_parts[:keep]
    if imported_module:
        prefix.extend(imported_module.split("."))
    return ".".join(prefix)


def _javascript_import_targets(
    path: str,
    source: str,
    file_set: set[str],
) -> tuple[str, ...]:
    specifiers: list[str] = []
    patterns = (
        re.compile(r"\bfrom\s*['\"]([^'\"]+)['\"]"),
        re.compile(r"\brequire\s*\(\s*['\"]([^'\"]+)['\"]\s*\)"),
        re.compile(r"\bimport\s*\(\s*['\"]([^'\"]+)['\"]\s*\)"),
        re.compile(r"\bimport\s*['\"]([^'\"]+)['\"]"),
    )
    for pattern in patterns:
        specifiers.extend(pattern.findall(source))
    targets: list[str] = []
    for specifier in dict.fromkeys(specifiers):
        target = _resolve_javascript_import(path, specifier, file_set)
        if target and target not in targets:
            targets.append(target)
    return tuple(targets)


def _resolve_javascript_import(
    source_path: str,
    specifier: str,
    file_set: set[str],
) -> str | None:
    if not specifier.startswith("."):
        return None
    base = posixpath.normpath((Path(source_path).parent / specifier).as_posix())
    candidates = [base]
    candidates.extend(
        f"{base}{suffix}"
        for suffix in (".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx", ".json")
    )
    candidates.extend(
        f"{base}/index{suffix}"
        for suffix in (".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx")
    )
    return next((candidate for candidate in candidates if candidate in file_set), None)


def _reverse_graph(
    import_graph: dict[str, tuple[str, ...]],
) -> dict[str, tuple[str, ...]]:
    reverse: dict[str, list[str]] = {}
    for source, imports in import_graph.items():
        for imported in imports:
            reverse.setdefault(imported, []).append(source)
    return {key: tuple(sorted(values)) for key, values in sorted(reverse.items())}


def _source_to_tests(file_tree: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
    tests = tuple(path for path in file_tree if _is_test(path))
    mapping: dict[str, tuple[str, ...]] = {}
    for source in file_tree:
        if _is_test(source) or _is_docs(source) or _is_config(source):
            continue
        stem = Path(source).stem.lower().replace("_", "-")
        matched = tuple(
            test for test in tests if stem and stem in test.lower().replace("_", "-")
        )
        if matched:
            mapping[source] = matched
    return mapping


def _test_to_sources(
    source_to_tests: dict[str, tuple[str, ...]],
) -> dict[str, tuple[str, ...]]:
    reverse: dict[str, list[str]] = {}
    for source, tests in source_to_tests.items():
        for test in tests:
            reverse.setdefault(test, []).append(source)
    return {test: tuple(sources) for test, sources in sorted(reverse.items())}


def _public_api_surfaces(
    entrypoints: tuple[str, ...],
    symbol_index: dict[str, tuple[SymbolLocation, ...]],
    docs_files: tuple[str, ...],
) -> tuple[str, ...]:
    surfaces: list[str] = list(entrypoints)
    for symbol, locations in symbol_index.items():
        if symbol.startswith("_"):
            continue
        if any(
            location.path in entrypoints
            or location.path.endswith((".d.ts", "__init__.py"))
            for location in locations
        ):
            surfaces.append(symbol)
    surfaces.extend(docs_files[:4])
    return tuple(dict.fromkeys(surfaces))[:80]


def _candidate_files_for_theme(
    theme: str,
    text_files: tuple[tuple[str, str], ...],
    public_api_surfaces: tuple[str, ...],
) -> tuple[str, ...]:
    terms = THEME_TERMS.get(
        theme, tuple(part for part in theme.split("_") if len(part) >= 4)
    )
    scored: list[tuple[float, str]] = []
    for path, text in text_files:
        haystack = f"{path}\n{text[:12000]}".lower()
        score = sum(haystack.count(term.lower()) for term in terms)
        if path in public_api_surfaces:
            score += 1
        if score > 0:
            scored.append((float(score), path))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return tuple(path for _, path in scored[:20])


def _error_patterns(text_files: tuple[tuple[str, str], ...]) -> tuple[str, ...]:
    patterns: list[str] = []
    for _, text in text_files:
        for line in text.splitlines():
            if any(
                token in line.lower()
                for token in (
                    "throw new error",
                    "raise ",
                    "logger.error",
                    "console.error",
                )
            ):
                patterns.append(line.strip()[:160])
    return tuple(dict.fromkeys(patterns))[:40]


def _style_conventions(
    package_manager: str | None, file_tree: tuple[str, ...]
) -> tuple[str, ...]:
    notes: list[str] = []
    if package_manager:
        notes.append(f"package_manager:{package_manager}")
    if any(path.endswith(".py") for path in file_tree):
        notes.append("python_sources_present")
    if any(path.endswith((".js", ".ts", ".tsx")) for path in file_tree):
        notes.append("javascript_or_typescript_sources_present")
    if any(_is_test(path) for path in file_tree):
        notes.append("tests_present")
    return tuple(notes)


def _architecture_notes(
    source_roots: tuple[str, ...],
    public_api_surfaces: tuple[str, ...],
) -> tuple[str, ...]:
    notes = [f"source_roots:{','.join(source_roots) if source_roots else 'none'}"]
    if public_api_surfaces:
        notes.append(f"public_api_surface_count:{len(public_api_surfaces)}")
    return tuple(notes)


def _is_test(path: str) -> bool:
    lowered = path.lower()
    return (
        "/test" in f"/{lowered}"
        or lowered.startswith("test_")
        or lowered.endswith(".test.js")
        or lowered.endswith(".spec.js")
    )


def _is_config(path: str) -> bool:
    name = Path(path).name.lower()
    return name in {
        "package.json",
        "pyproject.toml",
        "setup.py",
        "tsconfig.json",
        "vite.config.js",
    } or name.endswith(".config.js")


def _is_docs(path: str) -> bool:
    lowered = path.lower()
    return lowered.endswith((".md", ".rst", ".txt")) or lowered.startswith("docs/")


def _is_generated(path: str) -> bool:
    lowered = path.lower()
    return any(
        part in lowered for part in ("dist/", "build/", ".generated.", ".min.js")
    )


def _ignored(path: Path, root: Path) -> bool:
    parts = path.relative_to(root).parts
    return any(
        part
        in {
            ".git",
            ".mypy_cache",
            ".nox",
            ".pytest_cache",
            ".ruff_cache",
            ".tox",
            ".venv",
            "__pycache__",
            "coverage",
            "node_modules",
        }
        for part in parts
    )
