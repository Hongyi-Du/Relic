"""Withheld contract: a zero split budget yields one group holding the input's elements.

Only the public splitting helpers are used.
"""

from __future__ import annotations

from boltons.iterutils import split, split_iter


def test_a_zero_budget_yields_one_group_of_elements() -> None:
    assert list(split_iter([1, 2, 3], 9, maxsplit=0)) == [[1, 2, 3]]
    assert split([4, 0, 5], 0, maxsplit=0) == [[4, 0, 5]]
    assert list(split_iter([7, 7], 7, maxsplit=0)) == [[7, 7]]


def test_a_zero_budget_consumes_a_one_shot_iterator() -> None:
    assert list(split_iter(iter("abc"), "b", maxsplit=0)) == [["a", "b", "c"]]
    assert list(split_iter((n for n in range(3)), 1, maxsplit=0)) == [[0, 1, 2]]


def test_larger_budgets_and_the_default_are_unchanged() -> None:
    assert list(split_iter([1, 0, 2, 0, 3], 0, maxsplit=1)) == [[1], [2, 0, 3]]
    assert list(split_iter([1, 0, 2, 0, 3], 0, maxsplit=2)) == [[1], [2], [3]]
    assert list(split_iter([1, 0, 2, 0, 3], 0)) == [[1], [2], [3]]
    assert split([1, 0, 2], 0) == [[1], [2]]
