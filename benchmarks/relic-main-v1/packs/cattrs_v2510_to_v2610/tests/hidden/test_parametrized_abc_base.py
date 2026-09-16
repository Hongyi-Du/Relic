"""Withheld contract: a class that inherits a parametrized abstract-collection alias
still has a usable structuring hook.

Exercised only through ``cattrs.Converter``/``cattrs.BaseConverter``, so any
implementation that produces a type mapping for such bases satisfies the contract
regardless of how the mapping is derived internally.
"""

import typing

from attrs import define

from cattrs import BaseConverter, Converter


def build(base: object) -> type:
    """A modelled class whose only unusual property is the abstract base it declares."""

    @define
    class Reading(base):  # type: ignore[misc, valid-type]
        celsius: int = 0
        station: str = "base"

        def __iter__(self):
            return iter([self.celsius])

        def __next__(self):
            raise StopIteration

        def __reversed__(self):
            return iter([self.celsius])

        def __contains__(self, item):
            return item == self.celsius

        def __len__(self):
            return 1

        def __hash__(self):
            return hash(self.celsius)

    return Reading


def test_parametrized_iterable_base_structures() -> None:
    cls = build(typing.Iterable[int])

    assert Converter().structure({"celsius": 21, "station": "alpha"}, cls) == cls(21, "alpha")


def test_parametrized_container_base_structures() -> None:
    cls = build(typing.Container[str])

    assert Converter().structure({"celsius": -4}, cls) == cls(-4, "base")


def test_parametrized_collection_base_structures() -> None:
    cls = build(typing.Collection[int])

    assert Converter().structure({"celsius": 7, "station": "delta"}, cls) == cls(7, "delta")


def test_parametrized_reversible_base_structures() -> None:
    cls = build(typing.Reversible[str])

    assert Converter().structure({"celsius": 0}, cls) == cls(0, "base")


def test_unparametrized_bases_and_plain_inheritance_are_unaffected() -> None:
    bare = build(typing.Iterable)
    sized = build(typing.Sized)

    assert Converter().structure({"celsius": 3}, bare) == bare(3, "base")
    assert Converter().structure({"celsius": 5, "station": "z"}, sized) == sized(5, "z")
    assert BaseConverter().structure({"celsius": 8}, bare) == bare(8, "base")
