"""Withheld contract: a multidict compares unequal to a mapping whose values differ.

Only the public container and the ``==``/``!=`` operators are used.
"""

from __future__ import annotations

from boltons.dictutils import OrderedMultiDict


def headers() -> OrderedMultiDict:
    return OrderedMultiDict([("host", "example.test"), ("accept", "text/html")])


def test_a_single_differing_value_makes_the_comparison_false() -> None:
    assert not (headers() == {"host": "other.test", "accept": "text/html"})
    assert headers() != {"host": "other.test", "accept": "text/html"}
    assert headers() != {"host": "example.test", "accept": "application/json"}


def test_repeated_keys_compare_on_their_most_recent_value() -> None:
    tags = OrderedMultiDict([("tag", "old"), ("tag", "new")])

    assert tags == {"tag": "new"}
    assert tags != {"tag": "old"}


def test_matching_and_mis_shaped_comparisons_behave_as_before() -> None:
    assert headers() == {"host": "example.test", "accept": "text/html"}
    assert headers() != {"host": "example.test", "encoding": "gzip"}
    assert headers() != {"host": "example.test"}
    assert headers() != "not a mapping"
    assert headers() == OrderedMultiDict(
        [("host", "example.test"), ("accept", "text/html")]
    )
    assert headers() != OrderedMultiDict(
        [("host", "other.test"), ("accept", "text/html")]
    )
