"""Withheld contract: a converter can carry the "use field aliases" decision itself,
instead of it having to be restated on every generated hook.

Exercised only through ``cattrs.Converter`` construction, copying and its
``structure``/``unstructure`` entry points.
"""

from attrs import define, field

from cattrs import Converter


@define
class Account:
    owner: str = field(alias="_owner")
    balance: int = field(default=0)
    note: str = field(default="", alias="_note")


def test_converter_level_alias_default_drives_unstructuring() -> None:
    converter = Converter(use_alias=True)

    assert converter.unstructure(Account("ada", 12, "vip")) == {
        "_owner": "ada",
        "balance": 12,
        "_note": "vip",
    }


def test_converter_level_alias_default_drives_structuring() -> None:
    converter = Converter(use_alias=True)

    assert converter.structure(
        {"_owner": "bob", "balance": 3, "_note": "trial"}, Account
    ) == Account("bob", 3, "trial")


def test_a_copied_converter_keeps_the_decision() -> None:
    converter = Converter(use_alias=True).copy()

    assert converter.unstructure(Account("cyd")) == {
        "_owner": "cyd",
        "balance": 0,
        "_note": "",
    }


def test_the_decision_can_be_turned_off_when_copying() -> None:
    converter = Converter(use_alias=True).copy(use_alias=False)

    assert converter.unstructure(Account("dee")) == {
        "owner": "dee",
        "balance": 0,
        "note": "",
    }


def test_a_default_converter_still_keys_on_field_names() -> None:
    converter = Converter()

    assert converter.unstructure(Account("eve", 1, "x")) == {
        "owner": "eve",
        "balance": 1,
        "note": "x",
    }
    assert converter.structure({"owner": "eve", "balance": 1, "note": "x"}, Account) == (
        Account("eve", 1, "x")
    )
