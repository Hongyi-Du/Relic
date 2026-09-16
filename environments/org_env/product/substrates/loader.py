"""OSS time-machine dataset loader (brief §4/§5).

Reads a *frozen local* dataset (NEVER GitHub at runtime — brief §1) and resolves it into an
``OSSSubstrateSpec`` plus parsed ``HistoricalIssue`` / ``HiddenTestSpec`` lists. The agent-visible
surfaces (starter repo, public issues, external signals) and the evaluator-only surfaces (reference
repo, hidden tests, held-out issues) are kept strictly separate here so seeding can never
accidentally mix them.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

from environments.org_env.product.substrates.base import (
    HiddenTestSpec,
    HistoricalIssue,
    OSSSubstrateSpec,
)

_SKIP_DIR_NAMES = {"__pycache__", ".git", ".hg", ".svn", ".mypy_cache", ".pytest_cache"}
_SKIP_SUFFIXES = (".pyc", ".pyo")


def data_root() -> str:
    """Return the public Relic benchmark root.

    The upstream HCI implementation used its development-tree dataset path.
    Release packs are deliberately relocated under the frozen
    ``benchmarks/relic-main-v1`` tree so a fresh clone has no dependency on a
    developer checkout.  ``RELIC_BENCHMARK_ROOT`` remains the single public
    override through :mod:`relic.paths`.
    """
    from relic.paths import benchmark_root

    return str(benchmark_root() / "relic-main-v1")


def find_dataset_dir(dataset_id: str, root: Optional[str] = None) -> str:
    """Locate the dataset under ``<root>/{real,projects,fixtures}/<id>``.

    ``real`` and ``projects`` (committed/offline-frozen REAL OSS snapshots) take precedence over
    ``fixtures`` (tiny synthetic dev/test fixtures) so a real id never accidentally resolves to a
    toy. An absolute path is also accepted directly."""
    root = root or data_root()
    for sub in ("packs", "real", "projects", "fixtures"):
        cand = os.path.join(root, sub, dataset_id)
        if os.path.isdir(cand):
            return cand
    # also allow a direct path
    if os.path.isabs(dataset_id) and os.path.isdir(dataset_id):
        return dataset_id
    raise FileNotFoundError(
        f"OSS time-machine dataset {dataset_id!r} not found under "
        f"{root}/{{real,projects,fixtures}}")


def is_fixture_dir(dataset_dir: str) -> bool:
    """True if the resolved dataset lives under a ``fixtures/`` tree (synthetic dev/test only)."""
    norm = os.path.normpath(dataset_dir).replace(os.sep, "/")
    return "/fixtures/" in norm or norm.endswith("/fixtures")


def load_manifest(dataset_dir: str) -> Dict[str, Any]:
    """Parse ``manifest.yaml`` (preferred) or ``manifest.json`` (fallback). YAML is optional —
    if PyYAML is missing we fall back to the JSON manifest (brief §5)."""
    yaml_path = os.path.join(dataset_dir, "manifest.yaml")
    json_path = os.path.join(dataset_dir, "manifest.json")
    if os.path.isfile(yaml_path):
        try:
            import yaml  # type: ignore
            with open(yaml_path, encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    if os.path.isfile(json_path):
        with open(json_path, encoding="utf-8") as fh:
            return json.load(fh) or {}
    raise FileNotFoundError(f"no manifest.yaml/manifest.json in {dataset_dir}")


def _rel(dataset_dir: str, manifest: Dict[str, Any], *keys_default: Tuple[str, str]) -> str:
    """Resolve a dataset-relative path from manifest sections with a default."""
    for section, key, default in keys_default:  # type: ignore[misc]
        sect = manifest.get(section) or {}
        if isinstance(sect, dict) and sect.get(key):
            return os.path.join(dataset_dir, str(sect[key]))
    # default is the last tuple's default
    return os.path.join(dataset_dir, keys_default[-1][2])


def load_oss_substrate_spec(dataset_id: str, root: Optional[str] = None) -> OSSSubstrateSpec:
    dataset_dir = find_dataset_dir(dataset_id, root=root)
    manifest = load_manifest(dataset_dir)
    av = manifest.get("agent_visible") or {}
    pe = manifest.get("private_evaluator_only") or {}

    def _p(section: Dict[str, Any], key: str, default: str) -> str:
        val = section.get(key) if isinstance(section, dict) else None
        return os.path.join(dataset_dir, str(val if val else default))

    return OSSSubstrateSpec(
        project_id=str(manifest.get("project_id", dataset_id)),
        product_name=str(manifest.get("anonymized_product_name")
                         or manifest.get("source_project_name") or dataset_id),
        dataset_dir=dataset_dir,
        starter_repo_dir=_p(av, "starter_repo", "starter_repo"),
        reference_repo_dir=_p(pe, "reference_repo", "reference_repo"),
        public_issues_dir=_p(av, "public_issues", "issues/public"),
        heldout_issues_dir=_p(pe, "heldout_issues", "issues/heldout"),
        hidden_tests_dir=_p(pe, "hidden_tests", "tests/hidden"),
        public_tests_dir=os.path.join(dataset_dir, "tests/public"),
        external_signals_dir=_p(av, "external_signals", "external_signals"),
        contamination_dir=_p(pe, "contamination_probes", "contamination"),
        manifest=manifest,
    )


def _load_issue_file(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _issues_from_dir(issues_dir: str, *, hidden: bool) -> List[HistoricalIssue]:
    out: List[HistoricalIssue] = []
    if not os.path.isdir(issues_dir):
        return out
    for name in sorted(os.listdir(issues_dir)):
        if not name.endswith(".json"):
            continue
        rec = _load_issue_file(os.path.join(issues_dir, name))
        if not rec:
            continue
        # agent-visible body = the rewritten (user-style) text; fall back to body / original
        body = str(rec.get("body") or rec.get("text_rewritten") or rec.get("text_original") or "")
        out.append(HistoricalIssue(
            issue_id=str(rec.get("issue_id") or os.path.splitext(name)[0]),
            title=str(rec.get("title", "")),
            body=body,
            issue_type=str(rec.get("issue_type", "bug")),
            severity=str(rec.get("severity", "medium")),
            component=str(rec.get("component", "")),
            candidate_path_hints=[str(h) for h in (rec.get("candidate_path_hints") or [])],
            created_tick=int(rec.get("created_tick", 0) or 0),
            source=str(rec.get("source", "historical_github")),
            hidden=bool(rec.get("hidden", hidden)),
            acceptance_hint=str(rec.get("acceptance_hint", "")),
            linked_hidden_test_ids=list(rec.get("linked_hidden_test_ids", []) or []),
            source_url=str(rec.get("source_url", "")),
            source_id=str(rec.get("source_id", "")),
            source_type=str(rec.get("source_type", "")),
            created_at=str(rec.get("created_at", "")),
            labels=list(rec.get("labels", []) or []),
            title_original=str(rec.get("title_original", "")),
            text_original=str(rec.get("text_original", "")),
            text_rewritten=str(rec.get("text_rewritten") or body),
            release=str(rec.get("release", "")),
        ))
    return out


def load_public_issues(spec: OSSSubstrateSpec) -> List[HistoricalIssue]:
    return _issues_from_dir(spec.public_issues_dir, hidden=False)


def load_heldout_issues(spec: OSSSubstrateSpec) -> List[HistoricalIssue]:
    return _issues_from_dir(spec.heldout_issues_dir, hidden=True)


def load_hidden_test_specs(spec: OSSSubstrateSpec) -> List[HiddenTestSpec]:
    """Hidden-test catalog from ``tests/hidden/specs.json`` (a list of specs). If absent, derive a
    single spec from the manifest's ``hidden_tests.command``."""
    out: List[HiddenTestSpec] = []
    specs_path = os.path.join(spec.hidden_tests_dir, "specs.json")
    specs_file_present = os.path.isfile(specs_path)
    if specs_file_present:
        try:
            with open(specs_path, encoding="utf-8") as fh:
                rows = json.load(fh)
        except Exception:
            rows = []
        for rec in (rows or []):
            rel = str(rec.get("rel_path", ""))
            cmd = list(rec.get("command", []) or [])
            if not cmd:
                target = os.path.join("tests/hidden", rel) if rel else "tests/hidden"
                cmd = ["python", "-m", "pytest", target, "-q"]
            out.append(HiddenTestSpec(
                test_id=str(rec.get("test_id", "")),
                name=str(rec.get("name", rec.get("test_id", ""))),
                command=cmd,
                issue_ids=list(rec.get("issue_ids", []) or []),
                expected_behavior=str(rec.get("expected_behavior", "")),
                rel_path=rel,
                introduced_in=str(rec.get("introduced_in", "")),
                source_url=str(rec.get("source_url", "")),
            ))
    if not out and not specs_file_present:
        cmd = ((spec.manifest.get("hidden_tests") or {}).get("command")
               or ["python", "-m", "pytest", "tests/hidden", "-q"])
        out.append(HiddenTestSpec(test_id="hidden_all", name="all hidden behavior tests",
                                  command=list(cmd), issue_ids=[], expected_behavior="",
                                  rel_path=""))
    return out


def load_external_signals(spec: OSSSubstrateSpec) -> List[Dict[str, Any]]:
    """Seed external-community posts from ``external_signals/seed_posts.json`` (a list of post
    dicts), or ``[]`` if absent (brief review §5). These are agent-visible field chatter."""
    path = os.path.join(spec.external_signals_dir, "seed_posts.json")
    if not os.path.isfile(path):
        return []
    try:
        with open(path, encoding="utf-8") as fh:
            rows = json.load(fh)
    except Exception:
        return []
    return [r for r in (rows or []) if isinstance(r, dict)]


def read_repo_files(repo_dir: str) -> Dict[str, str]:
    """Return ``{relative_path: text}`` for all tracked text files under ``repo_dir`` (skips caches
    and binary files). Paths use forward slashes for cross-platform stable artifact ids."""
    out: Dict[str, str] = {}
    if not os.path.isdir(repo_dir):
        return out
    for dirpath, dirnames, filenames in os.walk(repo_dir):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIR_NAMES]
        for fn in sorted(filenames):
            if fn.endswith(_SKIP_SUFFIXES) or fn == ".DS_Store":
                continue
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, repo_dir).replace(os.sep, "/")
            try:
                with open(full, encoding="utf-8") as fh:
                    out[rel] = fh.read()
            except (UnicodeDecodeError, OSError):
                continue
    return out


__all__ = [
    "data_root", "find_dataset_dir", "is_fixture_dir", "load_manifest", "load_oss_substrate_spec",
    "load_public_issues", "load_heldout_issues", "load_hidden_test_specs", "load_external_signals",
    "read_repo_files",
]
