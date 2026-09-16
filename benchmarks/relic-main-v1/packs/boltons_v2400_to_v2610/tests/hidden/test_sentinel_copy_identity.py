"""Withheld contract: a sentinel comes back from the copy protocol as the same object.

Exercised through the public sentinel factory and the standard :mod:`copy` module, so
any way of opting out of duplication satisfies the contract.
"""

from __future__ import annotations

import copy

from boltons.typeutils import make_sentinel


def test_shallow_and_deep_copies_return_the_original_object() -> None:
    unset = make_sentinel("UNSET")

    assert copy.copy(unset) is unset
    assert copy.deepcopy(unset) is unset


def test_a_sentinel_nested_in_a_structure_survives_duplication() -> None:
    absent = make_sentinel("ABSENT")
    record = {"fields": [absent, 1], "default": absent}

    duplicated = copy.deepcopy(record)

    assert duplicated["default"] is absent
    assert duplicated["fields"][0] is absent
    assert duplicated is not record
    assert duplicated["fields"] is not record["fields"]


def test_established_sentinel_semantics_remain_stable() -> None:
    first = make_sentinel("UNSET")
    second = make_sentinel("UNSET")

    assert first is not second
    assert bool(first) is False
    assert repr(first) == "Sentinel('UNSET')"
    assert {first: "kept"}[first] == "kept"
