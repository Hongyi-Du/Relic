"""Withheld contract: partitioning accepts several predicates and returns one group each.

Only the public ``partition`` entry point is used; nothing about how the grouping is
computed is inspected.
"""

from __future__ import annotations

from boltons.iterutils import partition


def test_two_predicates_produce_three_groups() -> None:
    positive, negative, rest = partition(range(-3, 4), lambda i: i > 0, lambda i: i < 0)

    assert positive == [1, 2, 3]
    assert negative == [-3, -2, -1]
    assert rest == [0]


def test_the_first_matching_predicate_claims_each_value() -> None:
    by_three, by_two, rest = partition(
        [1, 2, 3, 4, 5, 6], lambda i: i % 3 == 0, lambda i: i % 2 == 0
    )

    assert by_three == [3, 6]
    assert by_two == [2, 4]
    assert rest == [1, 5]


def test_values_matching_nothing_land_in_the_final_group() -> None:
    lower, upper, other = partition("aA1bB2", str.islower, str.isupper)

    assert lower == ["a", "b"]
    assert upper == ["A", "B"]
    assert other == ["1", "2"]


def test_the_single_predicate_form_still_returns_two_groups() -> None:
    truthy, falsy = partition(["", "", "hi", "", "bye"])
    assert truthy == ["hi", "bye"]
    assert falsy == ["", "", ""]

    digits, letters = partition("a1b2", lambda char: char.isdigit())
    assert digits == ["1", "2"]
    assert letters == ["a", "b"]
