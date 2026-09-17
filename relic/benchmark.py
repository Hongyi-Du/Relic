"""Integrity checks for the frozen public benchmark."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml

from relic.paths import benchmark_root


# Benchmark packs contain exported third-party repositories.  Running local
# tooling in one of those repositories can create ignored interpreter/linter
# caches; these are host state rather than frozen benchmark content and must
# not change the published pack digest.
_TRANSIENT_TREE_DIRS = frozenset(
    {".mypy_cache", ".pytest_cache", ".ruff_cache", ".tox", "__pycache__"}
)


def benchmark_directory() -> Path:
    return benchmark_root() / "relic-main-v1"


def load_benchmark_manifest() -> dict[str, Any]:
    path = benchmark_directory() / "manifest.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"benchmark manifest must be a mapping: {path}")
    return payload


def tree_digest(directory: Path) -> tuple[int, str]:
    """Hash a directory using the algorithm declared by the manifest."""
    digest = hashlib.sha256()
    files = sorted(
        path
        for path in directory.rglob("*")
        if path.is_file()
        and not any(part in _TRANSIENT_TREE_DIRS for part in path.relative_to(directory).parts)
    )
    for path in files:
        relative = path.relative_to(directory).as_posix().encode("utf-8")
        file_digest = hashlib.sha256(path.read_bytes()).hexdigest().encode("ascii")
        digest.update(relative + b"\0" + file_digest + b"\n")
    return len(files), digest.hexdigest()


def verify_benchmark() -> list[str]:
    manifest = load_benchmark_manifest()
    packs_root = benchmark_directory() / "packs"
    failures: list[str] = []
    expected_packs: set[str] = set()
    for workload in manifest["workloads"]:
        pack = str(workload["pack"])
        expected_packs.add(pack)
        path = packs_root / pack
        if not path.is_dir():
            failures.append(f"{pack}: missing directory")
            continue
        count, observed = tree_digest(path)
        if count != int(workload["file_count"]):
            failures.append(f"{pack}: expected {workload['file_count']} files, found {count}")
        if observed != str(workload["tree_sha256"]):
            failures.append(
                f"{pack}: tree digest mismatch; expected {workload['tree_sha256']}, found {observed}"
            )
    extra = sorted(path.name for path in packs_root.iterdir() if path.is_dir())
    extra = [name for name in extra if name not in expected_packs]
    if extra:
        failures.append(f"unexpected pack directories: {', '.join(extra)}")
    return failures
