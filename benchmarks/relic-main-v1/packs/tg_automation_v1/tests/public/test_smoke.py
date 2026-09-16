"""Smoke checks the team can run between evaluations.

These assert that the package is importable and that each module named
in the brief exists. They deliberately assert nothing about behaviour:
the frozen suite grades that, and restating it here would leak it.
"""
import importlib

import pytest

MODULES = ['analysis', 'cli', 'client', 'config', 'engagement', 'export', 'filters', 'growth', 'retry']


def test_package_imports():
    assert importlib.import_module('tg_automation') is not None


@pytest.mark.parametrize("name", MODULES)
def test_each_module_imports(name):
    assert importlib.import_module(f"tg_automation.{name}") is not None
