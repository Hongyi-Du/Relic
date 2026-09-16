"""Smoke checks the team can run between evaluations.

A merge gate that is green on the untouched starter is not a gate: every
pull request passes it and the number of merges stops meaning anything.
Import alone cannot tell the two endpoints apart here, because the
starter keeps the whole package structure and only the bodies are gone.
So a module is judged against the stubs it began with, counted in
STUBS_AT_START from the starter's own source.

Demanding every module at once is not a gate either, in the other
direction. On traffic_watch it returned one exit code for ten modules,
and five independent organizations each finished nine, each stalled on
the tenth, and each merged nothing at all in 192 ticks. Nothing they did
could ever reach the mainline, so every one of them scored the same zero.

Demanding that a touched module be finished is the same mistake at a
smaller scale. It reads as reasonable and it is not symmetric between
the conditions: one member writes one module at a time and is never
caught mid-way, while eight members working at once have several
modules part-written at any moment, and each of them has to carry a
whole module to the end before anything can land. The arms that the
experiment exists to compare would be merging against different bars.

So what is asked is only that the work go forward: the package imports,
something has been written, and no module holds more stubs than it did
in the starter. Half a module lands. Deleting an implementation does
not. The starter is still red, which is what makes this a gate.

Whether half-written work should land is a judgement, and it belongs to
the organization rather than to this file. An organization that merges
unfinished modules will show it in the hidden suite, which is where
correctness is decided; a gate that forbids it decides on their behalf
and hides whether they would have caught it themselves.
"""
import importlib
from pathlib import Path

import pytest

PACKAGE = 'app'
MODULES = ['db', 'models', 'routers.charts']
# module -> how many stubs it held in the untouched starter.
STUBS_AT_START = {'db': 6, 'routers.charts': 3}


def _source_of(name):
    # A name may be dotted (routers.charts) or empty (the package's own
    # __init__); the bare stem would name a file that does not exist.
    root = Path(__file__).resolve().parents[2] / PACKAGE
    return (root / (name.replace(".", "/") + ".py")) if name \
        else (root / "__init__.py")


def _stubs_now(name):
    """How many stubs the module still holds, read from its own source."""
    path = _source_of(name)
    if not path.is_file():
        return STUBS_AT_START.get(name, 0)
    return path.read_text(encoding="utf-8", errors="replace").count(
        "NotImplementedError")


def test_package_imports():
    assert importlib.import_module(PACKAGE) is not None


def test_something_has_been_written():
    """The untouched starter fails here, which is what makes this a gate."""
    started = [n for n in MODULES
               if _stubs_now(n) < STUBS_AT_START.get(n, 0)]
    assert started, "no module has been implemented yet: " + ", ".join(
        n or PACKAGE for n in MODULES)


@pytest.mark.parametrize("name", MODULES)
def test_no_module_went_backwards(name):
    """A change may leave a module unfinished; it may not un-write it."""
    began, now = STUBS_AT_START.get(name, 0), _stubs_now(name)
    assert now <= began, (
        f"{name}: {now} stubs now against {began} in the starter, so "
        f"an implementation was removed")
