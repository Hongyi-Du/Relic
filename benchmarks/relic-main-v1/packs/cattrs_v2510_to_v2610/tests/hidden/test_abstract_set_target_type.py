"""Withheld contract: an abstract set annotation produces an immutable, hashable set,
while the mutable annotations keep producing mutable sets.

Exercised only through ``structure`` on both shipped converters.
"""

import typing
from collections.abc import MutableSet, Set

from attrs import define

from cattrs import BaseConverter, Converter


@define
class Article:
    topics: Set[str]
    editors: MutableSet[str]


def test_abstract_set_target_produces_a_hashable_set() -> None:
    result = Converter().structure(["7", "8", "9"], Set[int])

    assert result == frozenset({7, 8, 9})
    assert hash(result) == hash(frozenset({7, 8, 9}))


def test_the_base_converter_agrees_on_the_target_type() -> None:
    result = BaseConverter().structure(("4", "5"), typing.AbstractSet[int])

    assert result == frozenset({4, 5})
    assert hash(result) == hash(frozenset({4, 5}))


def test_the_target_type_carries_into_modelled_fields() -> None:
    article = Converter().structure(
        {"topics": ["physics", "chemistry"], "editors": ["ada"]}, Article
    )

    assert article.topics == frozenset({"physics", "chemistry"})
    assert hash(article.topics) == hash(frozenset({"physics", "chemistry"}))
    assert article.editors == {"ada"}


def test_mutable_annotations_still_produce_mutable_sets() -> None:
    converter = Converter()

    mutable = converter.structure(("1", "2"), MutableSet[int])
    mutable.add(3)
    assert mutable == {1, 2, 3}

    builtin = converter.structure(("5", "6"), set[int])
    builtin.add(7)
    assert builtin == {5, 6, 7}

    assert converter.structure(("8",), frozenset[int]) == frozenset({8})
