"""Agent-visible smoke checks for the frozen cattrs source tree."""

from __future__ import annotations

import importlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_source_tree_is_present() -> None:
    required = (
        "src/cattrs/__init__.py",
        "pyproject.toml",
        "tests",
    )
    missing = [item for item in required if not (ROOT / item).exists()]
    assert not missing, f"missing frozen source paths: {missing}"


def test_library_imports() -> None:
    module = importlib.import_module("cattrs")
    assert module.__file__
