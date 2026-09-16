"""Withheld contract: slicing an ordered set ignores the slots earlier removals vacated.

The set is only observed through iteration and the subscript operator, so any bookkeeping
that yields the right window satisfies the contract.
"""

from __future__ import annotations

from typing import List

from boltons.setutils import IndexedSet

LETTERS = "abcdefghijkl"
REMOVED = ("c", "g", "h")
BOUNDS = (None, 0, 1, 3, 5, 8, -1, -2, -5, -20, 20)


def holey() -> IndexedSet:
    subject = IndexedSet(LETTERS)
    for letter in REMOVED:
        subject.discard(letter)
    return subject


def surviving() -> List[str]:
    return [letter for letter in LETTERS if letter not in REMOVED]


def test_slices_after_removals_match_plain_sequence_slicing() -> None:
    subject, expected = holey(), surviving()

    assert list(subject) == expected
    for start in BOUNDS:
        for stop in BOUNDS:
            for step in (None, 1, 2, 3):
                window = slice(start, stop, step)
                assert list(subject[window]) == expected[window], window


def test_slicing_does_not_depend_on_how_the_contents_were_reached() -> None:
    subject = holey()
    rebuilt = IndexedSet(surviving())

    for start in BOUNDS:
        for stop in BOUNDS:
            for step in (None, 1, 2, -1, -2):
                window = slice(start, stop, step)
                assert list(subject[window]) == list(rebuilt[window]), window


def test_slices_stay_ordered_sets_and_untouched_sets_are_unaffected() -> None:
    assert isinstance(holey()[1:4], IndexedSet)

    pristine = IndexedSet(range(9))
    reference = list(range(9))
    for start in (None, 0, 2, -3):
        for stop in (None, 5, -1):
            assert list(pristine[start:stop]) == reference[start:stop]
