"""
auto_trigger/builtin_templates.py — Environment-agnostic template registry.

After the env-decoupling refactor, this module is an EMPTY registry.
Environment-specific templates live in their respective env packages:
  - Nature env: environments/nature_env/reflex/builtin_templates.py

Environments inject their templates via BaseEngine.get_reflex_templates().
The SDK no longer ships any built-in templates.

The symbols ALL_BUILTIN_TEMPLATES and MILESTONE_TO_TEMPLATES are kept
for backward-compatibility but are empty at the SDK level.
"""
from __future__ import annotations

from typing import Dict, List

from .types import ReflexTemplate

ALL_BUILTIN_TEMPLATES: List[ReflexTemplate] = []

MILESTONE_TO_TEMPLATES: Dict[str, List[str]] = {}
