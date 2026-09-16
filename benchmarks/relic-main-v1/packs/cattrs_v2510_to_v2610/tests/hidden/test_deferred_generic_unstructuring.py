"""Withheld contract: unstructuring a parametrized generic works when the module
defers its annotations.

The module opts into deferred (string) annotations, which is the only condition
that matters here. Everything is exercised through ``cattrs.Converter.unstructure``.
"""

from __future__ import annotations

from typing import Generic, TypeVar

from attrs import define

from cattrs import BaseConverter, Converter

T = TypeVar("T")
U = TypeVar("U")


@define
class Envelope(Generic[T]):
    payload: T
    tag: str = "v1"


@define
class Pair(Generic[T, U]):
    left: T
    right: U


@define
class PlainEnvelope:
    payload: int
    tag: str = "v1"


def test_parametrized_generic_unstructures_to_primitives() -> None:
    assert Converter().unstructure(Envelope(7), unstructure_as=Envelope[int]) == {
        "payload": 7,
        "tag": "v1",
    }


def test_multi_parameter_generic_unstructures_to_primitives() -> None:
    assert Converter().unstructure(
        Pair(3, "three"), unstructure_as=Pair[int, str]
    ) == {"left": 3, "right": "three"}


def test_generic_holding_another_modelled_class_unstructures() -> None:
    assert Converter().unstructure(
        Envelope(PlainEnvelope(2, "inner"), "outer"),
        unstructure_as=Envelope[PlainEnvelope],
    ) == {"payload": {"payload": 2, "tag": "inner"}, "tag": "outer"}


def test_non_generic_classes_with_deferred_annotations_are_unaffected() -> None:
    assert Converter().unstructure(PlainEnvelope(4)) == {"payload": 4, "tag": "v1"}
    assert BaseConverter().unstructure(PlainEnvelope(5, "b")) == {
        "payload": 5,
        "tag": "b",
    }
