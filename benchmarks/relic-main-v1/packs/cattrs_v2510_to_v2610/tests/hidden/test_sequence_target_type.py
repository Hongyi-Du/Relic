"""Withheld contract: an abstract sequence annotation produces an immutable, hashable
sequence, while the mutable annotations keep producing lists.

Exercised only through ``structure`` on both shipped converters, so the contract is
about the produced value, not about which predicate or hook factory produced it.
"""

import typing
from collections.abc import MutableSequence, Sequence

from attrs import define

from cattrs import BaseConverter, Converter


@define
class Shipment:
    stops: Sequence[str]
    weights: MutableSequence[int]


def test_abstract_sequence_target_produces_a_hashable_sequence() -> None:
    result = Converter().structure(["7", 8, "9"], Sequence[int])

    assert result == (7, 8, 9)
    assert hash(result) == hash((7, 8, 9))


def test_the_base_converter_agrees_on_the_target_type() -> None:
    result = BaseConverter().structure(("4", 5), typing.Sequence[int])

    assert result == (4, 5)
    assert hash(result) == hash((4, 5))


def test_the_target_type_carries_into_modelled_fields() -> None:
    shipment = Converter().structure(
        {"stops": ("oslo", "kiel"), "weights": ["3", 4]}, Shipment
    )

    assert shipment.stops == ("oslo", "kiel")
    assert hash(shipment.stops) == hash(("oslo", "kiel"))
    assert shipment.weights == [3, 4]


def test_mutable_annotations_still_produce_mutable_lists() -> None:
    converter = Converter()

    mutable = converter.structure(("1", 2), MutableSequence[int])
    mutable.append(3)
    assert mutable == [1, 2, 3]

    plain = converter.structure(("5", 6), list[int])
    plain.append(7)
    assert plain == [5, 6, 7]
