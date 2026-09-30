"""Hierarchical fault localization artifacts for Code-Max runs."""

from __future__ import annotations

from typing import Any, Mapping

from society_core.hashing import stable_hash

from .repo_intelligence import THEME_TERMS
from .schemas import AcceptanceOracleSpec, LocalizationHypothesis, RepoIntelligenceFabric


def localize_faults(
    *,
    task_specs: tuple[Any, ...],
    repo: RepoIntelligenceFabric,
    oracles: tuple[AcceptanceOracleSpec, ...],
    top_k_files: int = 20,
) -> tuple[LocalizationHypothesis, ...]:
    hypotheses: list[LocalizationHypothesis] = []
    for spec in task_specs:
        task_id = str(_field(spec, "task_id", "task"))
        theme = str(_field(spec, "source_theme", "product_improvement"))
        path_hints = tuple(str(path) for path in _field(spec, "candidate_path_hints", ()))
        relevant_symbols = tuple(str(symbol) for symbol in _field(spec, "relevant_symbols_hint", ()))
        files = _rank_candidate_files(
            theme=theme,
            path_hints=path_hints,
            relevant_symbols=relevant_symbols,
            repo=repo,
            top_k=top_k_files,
        )
        symbols = _rank_candidate_symbols(relevant_symbols, repo, files)
        lines = _candidate_lines(symbols, repo)
        task_oracles = tuple(oracle for oracle in oracles if oracle.task_id == task_id)
        confirmed = any(oracle.base_status == "confirmed" for oracle in task_oracles)
        missing_context = ()
        if not files:
            missing_context = ("candidate_files",)
        confidence = 0.35 + (0.25 if files else 0.0) + (0.2 if symbols else 0.0) + (0.2 if confirmed else 0.0)
        scores = {
            "path_hint": 1.0 if any(path in files for path in path_hints) else 0.0,
            "symbol": min(1.0, len(symbols) / 5),
            "repo_theme": min(1.0, len(files) / max(1, top_k_files)),
            "repro": 1.0 if confirmed else 0.0,
        }
        payload = {
            "task_id": task_id,
            "files": files,
            "symbols": symbols,
            "scores": scores,
        }
        hypotheses.append(
            LocalizationHypothesis(
                hypothesis_id=f"loc_{stable_hash(payload)[:16]}",
                task_id=task_id,
                candidate_files=files,
                candidate_symbols=symbols,
                candidate_lines=lines,
                rationale=(
                    "Ranked from explicit path hints, repository theme candidates, "
                    "symbol index matches, and confirmed oracle evidence."
                ),
                evidence_refs=tuple(oracle.oracle_id for oracle in task_oracles),
                scores=scores,
                confidence=round(min(1.0, confidence), 3),
                risk_notes=(),
                missing_context=missing_context,
            )
        )
    return tuple(hypotheses)


def _rank_candidate_files(
    *,
    theme: str,
    path_hints: tuple[str, ...],
    relevant_symbols: tuple[str, ...],
    repo: RepoIntelligenceFabric,
    top_k: int,
) -> tuple[str, ...]:
    file_set = set(repo.file_tree)
    scored: dict[str, float] = {}
    for path in path_hints:
        if path in file_set:
            scored[path] = scored.get(path, 0.0) + 5.0
    for path in repo.candidate_files_by_theme.get(theme, ()):
        scored[path] = scored.get(path, 0.0) + 2.0
    terms = THEME_TERMS.get(theme, ())
    for path in repo.file_tree:
        lowered = path.lower()
        if any(term.lower() in lowered for term in terms):
            scored[path] = scored.get(path, 0.0) + 1.0
    for symbol in relevant_symbols:
        for location in repo.symbol_index.get(symbol, ()):
            scored[location.path] = scored.get(location.path, 0.0) + 3.0
    seed_paths = tuple(scored)
    for path in seed_paths:
        dependencies = repo.import_graph.get(path, ())
        dependents = repo.reverse_dependency_graph.get(path, ())
        for dependency in dependencies:
            scored[dependency] = scored.get(dependency, 0.0) + 0.75
        for dependent in dependents:
            scored[dependent] = scored.get(dependent, 0.0) + 1.0
        for test in repo.source_to_tests.get(path, ()):
            scored[test] = scored.get(test, 0.0) + 1.25
        for neighbor in (*dependencies, *dependents):
            for test in repo.source_to_tests.get(neighbor, ()):
                scored[test] = scored.get(test, 0.0) + 0.65
    if not scored:
        for path in repo.entrypoints[:top_k]:
            scored[path] = scored.get(path, 0.0) + 0.5
    ranked = sorted(scored.items(), key=lambda item: (-item[1], item[0]))
    return tuple(path for path, _ in ranked[:top_k])


def _rank_candidate_symbols(
    relevant_symbols: tuple[str, ...],
    repo: RepoIntelligenceFabric,
    files: tuple[str, ...],
) -> tuple[str, ...]:
    symbols: list[str] = []
    for symbol in relevant_symbols:
        if symbol in repo.symbol_index:
            symbols.append(symbol)
    file_set = set(files)
    for symbol, locations in repo.symbol_index.items():
        if any(location.path in file_set for location in locations):
            symbols.append(symbol)
    return tuple(dict.fromkeys(symbols))[:50]


def _candidate_lines(
    symbols: tuple[str, ...],
    repo: RepoIntelligenceFabric,
) -> tuple[tuple[str, int, int], ...]:
    lines: list[tuple[str, int, int]] = []
    for symbol in symbols[:25]:
        for location in repo.symbol_index.get(symbol, ())[:3]:
            start = max(1, location.line - 3)
            end = location.line + 8
            lines.append((location.path, start, end))
    return tuple(lines)


def _field(obj: Any, name: str, default: Any) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)
