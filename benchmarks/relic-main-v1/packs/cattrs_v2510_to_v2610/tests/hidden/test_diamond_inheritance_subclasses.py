"""Withheld contract: a hierarchy where one class is reachable from the root by more
than one path can still be registered with the subclass strategy.

Exercised only through ``cattrs.strategies.include_subclasses`` /
``configure_tagged_union`` and the converter's ``structure``/``unstructure``.
"""

from attrs import define

from cattrs import Converter
from cattrs.strategies import configure_tagged_union, include_subclasses


def test_a_diamond_hierarchy_registers_and_dispatches() -> None:
    converter = Converter()

    @define
    class Node:
        kind: str = "n"

    @define
    class Left(Node):
        pass

    @define
    class Right(Node):
        pass

    @define
    class Merge(Left, Right):
        pass

    include_subclasses(Node, converter, union_strategy=configure_tagged_union)

    assert converter.structure({"kind": "m", "_type": "Merge"}, Node) == Merge("m")
    assert converter.structure({"kind": "l", "_type": "Left"}, Node) == Left("l")
    assert converter.structure({"kind": "r", "_type": "Right"}, Node) == Right("r")
    assert converter.unstructure(Merge("m"), unstructure_as=Node) == {
        "kind": "m",
        "_type": "Merge",
    }


def test_a_wider_diamond_registers_too() -> None:
    converter = Converter()

    @define
    class Base:
        tag: str = "b"

    @define
    class A(Base):
        pass

    @define
    class B(Base):
        pass

    @define
    class C(Base):
        pass

    @define
    class ABC(A, B, C):
        pass

    include_subclasses(Base, converter, union_strategy=configure_tagged_union)

    assert converter.structure({"tag": "x", "_type": "ABC"}, Base) == ABC("x")
    assert converter.structure({"tag": "y", "_type": "B"}, Base) == B("y")


def test_a_linear_hierarchy_still_works() -> None:
    converter = Converter()

    @define
    class Node:
        kind: str = "n"

    @define
    class Child(Node):
        extra: int = 0

    include_subclasses(Node, converter, union_strategy=configure_tagged_union)

    assert converter.structure(
        {"kind": "l", "extra": 2, "_type": "Child"}, Node
    ) == Child("l", 2)
    assert converter.unstructure(Child("l", 2), unstructure_as=Node) == {
        "kind": "l",
        "extra": 2,
        "_type": "Child",
    }
