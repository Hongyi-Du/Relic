"""Withheld contract: an out-of-range index is rejected instead of quietly wrapping.

Only the public subscript operator is exercised, so any bounds check that reports the
right condition satisfies the contract.
"""

from __future__ import annotations

import pytest

from boltons.setutils import IndexedSet


def test_negative_indices_past_the_start_are_rejected() -> None:
    subject = IndexedSet(range(6))

    with pytest.raises(IndexError):
        subject[-7]
    with pytest.raises(IndexError):
        subject[-8]


def test_bounds_are_enforced_after_removals_too() -> None:
    subject = IndexedSet("abcdefgh")
    for letter in ("b", "e"):
        subject.discard(letter)

    assert len(subject) == 6
    with pytest.raises(IndexError):
        subject[-7]
    with pytest.raises(IndexError):
        subject[-9]


def test_in_range_indices_and_positive_overruns_behave_as_before() -> None:
    subject = IndexedSet(range(6))

    assert subject[0] == 0
    assert subject[5] == 5
    assert subject[-1] == 5
    assert subject[-6] == 0

    with pytest.raises(IndexError):
        subject[6]
    with pytest.raises(IndexError):
        subject[20]
    with pytest.raises(IndexError):
        IndexedSet()[0]
