"""Withheld contract: structuring a parametrized generic works when the module defers
its annotations, including when the parameter is filled by another modelled class.

The module opts into deferred (string) annotations. Every class is built inside its
test so no earlier call can have resolved its annotations first; everything is
exercised through ``cattrs.Converter.structure``.
"""

from __future__ import annotations

from typing import Generic, TypeVar

from attrs import define

from cattrs import Converter

T = TypeVar("T")
U = TypeVar("U")


def test_a_generic_filled_with_a_modelled_class_structures() -> None:
    @define
    class Coordinate:
        x: int
        y: int

    @define
    class Frame(Generic[T]):
        origin: T
        label: str = "f"

    assert Converter().structure(
        {"origin": {"x": 1, "y": 2}, "label": "grid"}, Frame[Coordinate]
    ) == Frame(Coordinate(1, 2), "grid")


def test_a_generic_filled_with_a_builtin_structures() -> None:
    @define
    class Frame(Generic[T]):
        origin: T

    assert Converter().structure({"origin": "5"}, Frame[int]) == Frame(5)


def test_a_two_parameter_generic_structures() -> None:
    @define
    class Row(Generic[T, U]):
        key: T
        value: U

    assert Converter().structure({"key": 1, "value": "one"}, Row[int, str]) == Row(
        1, "one"
    )


def test_a_generic_nested_inside_a_generic_structures() -> None:
    @define
    class Leaf:
        weight: int

    @define
    class Branch(Generic[T]):
        item: T

    @define
    class Tree(Generic[T]):
        root: T

    assert Converter().structure(
        {"root": {"item": {"weight": 3}}}, Tree[Branch[Leaf]]
    ) == Tree(Branch(Leaf(3)))


def test_plain_classes_with_deferred_annotations_are_unaffected() -> None:
    @define
    class Flat:
        a: int
        b: str = "x"

    converter = Converter()

    assert converter.structure({"a": "1", "b": "y"}, Flat) == Flat(1, "y")
    assert converter.unstructure(Flat(2)) == {"a": 2, "b": "x"}
