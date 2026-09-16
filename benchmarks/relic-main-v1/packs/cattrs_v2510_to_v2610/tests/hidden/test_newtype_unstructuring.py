"""Withheld contract: a distinct type alias declared over a modelled class does not stop
the base converter from unstructuring the value behind it.

Exercised only through ``cattrs.BaseConverter.unstructure``.
"""

from typing import NewType

from attrs import define

from cattrs import BaseConverter, Converter


@define
class Money:
    amount: int
    currency: str = "EUR"


Price = NewType("Price", Money)
NetPrice = NewType("NetPrice", Price)


@define
class Invoice:
    total: Price
    reference: str = "INV"


def test_the_alias_target_unstructures_the_value_behind_it() -> None:
    assert BaseConverter().unstructure(Money(5), unstructure_as=Price) == {
        "amount": 5,
        "currency": "EUR",
    }


def test_stacked_aliases_resolve_the_same_way() -> None:
    assert BaseConverter().unstructure(
        Money(7, "USD"), unstructure_as=NetPrice
    ) == {"amount": 7, "currency": "USD"}


def test_an_alias_used_as_a_field_annotation_unstructures() -> None:
    assert BaseConverter().unstructure(Invoice(Price(Money(1)))) == {
        "total": {"amount": 1, "currency": "EUR"},
        "reference": "INV",
    }


def test_plain_targets_and_the_other_converter_are_unaffected() -> None:
    assert BaseConverter().unstructure(Money(2)) == {"amount": 2, "currency": "EUR"}
    assert BaseConverter().unstructure(Money(3), unstructure_as=Money) == {
        "amount": 3,
        "currency": "EUR",
    }
    assert Converter().unstructure(Money(4), unstructure_as=Price) == {
        "amount": 4,
        "currency": "EUR",
    }
