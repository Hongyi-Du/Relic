"""Withheld contract: a subclass registry rooted at a parametrized generic parent can be
wired to the tagged-union strategy.

Exercised only through ``cattrs.strategies.include_subclasses`` /
``configure_tagged_union`` and the converter's ``structure``/``unstructure``.
"""

import functools
import typing
from typing import Any, Generic, TypeVar

from attrs import define

from cattrs import Converter
from cattrs.strategies import configure_tagged_union, include_subclasses

T = TypeVar("T")


def by_origin_name(cl: Any) -> str:
    return (typing.get_origin(cl) or cl).__name__


def test_a_generic_parent_can_be_registered_and_round_trips() -> None:
    converter = Converter()

    @define
    class Shape(Generic[T]):
        size: T

    @define
    class Square(Shape[int]):
        side: int

    @define
    class Caption(Shape[str]):
        text: str

    include_subclasses(
        Shape[Any],
        converter,
        union_strategy=functools.partial(
            configure_tagged_union, tag_generator=by_origin_name
        ),
    )

    assert converter.unstructure(Square(2, 3)) == {
        "size": 2,
        "side": 3,
        "_type": "Square",
    }
    assert converter.unstructure(Caption("s", "t")) == {
        "size": "s",
        "text": "t",
        "_type": "Caption",
    }
    assert converter.structure(
        {"size": 2, "side": 3, "_type": "Square"}, Shape[Any]
    ) == Square(2, 3)
    assert converter.structure(
        {"size": "s", "text": "t", "_type": "Caption"}, Shape[Any]
    ) == Caption("s", "t")


def test_a_generic_parent_with_a_single_subclass_is_also_accepted() -> None:
    converter = Converter()

    @define
    class Envelope(Generic[T]):
        body: T

    @define
    class Letter(Envelope[str]):
        stamp: int

    include_subclasses(
        Envelope[Any],
        converter,
        union_strategy=functools.partial(
            configure_tagged_union, tag_generator=by_origin_name
        ),
    )

    assert converter.unstructure(Letter("hi", 2)) == {
        "body": "hi",
        "stamp": 2,
        "_type": "Letter",
    }
    assert converter.structure(
        {"body": "hi", "stamp": 2, "_type": "Letter"}, Envelope[Any]
    ) == Letter("hi", 2)


def test_a_non_generic_parent_still_works() -> None:
    converter = Converter()

    @define
    class Node:
        weight: int

    @define
    class Leaf(Node):
        label: str

    include_subclasses(Node, converter, union_strategy=configure_tagged_union)

    assert converter.unstructure(Leaf(1, "a")) == {
        "weight": 1,
        "label": "a",
        "_type": "Leaf",
    }
    assert converter.structure(
        {"weight": 1, "label": "a", "_type": "Leaf"}, Node
    ) == Leaf(1, "a")
