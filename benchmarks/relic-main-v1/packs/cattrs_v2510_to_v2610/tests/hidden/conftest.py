"""Evaluator-only fixtures for the cattrs hidden contract suite.

Source archives omit files that the project generates at build time, so the frozen tree
cannot be imported until those modules exist. Recreating them here keeps the contracts
runnable against both endpoints without mutating either frozen tree.
"""

from __future__ import annotations

import sys
from types import ModuleType


def _ensure_module(name: str, **attributes: object) -> None:
    if name in sys.modules:
        return
    module = ModuleType(name)
    module.__dict__.update(attributes)
    sys.modules[name] = module


# This project's source archive is directly importable; no generated
# module needs to be recreated.
