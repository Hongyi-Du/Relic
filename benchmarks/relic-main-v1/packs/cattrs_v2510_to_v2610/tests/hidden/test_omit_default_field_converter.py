"""Withheld contract: "omit values equal to their default" compares like with like when
the field declares a converter.

Exercised only through ``cattrs.Converter(omit_if_default=True).unstructure``.
"""

import attrs
from attrs import define, field

from cattrs import Converter


def test_a_plain_callable_converter_is_applied_to_the_default() -> None:
    @define
    class Config:
        retries: int = field(default="3", converter=int)

    assert Converter(omit_if_default=True).unstructure(Config()) == {}


def test_a_factory_default_is_run_through_the_converter_too() -> None:
    @define
    class Config:
        timeout: int = field(factory=lambda: "30", converter=int)

    assert Converter(omit_if_default=True).unstructure(Config()) == {}


def test_a_converter_object_is_applied_to_the_default() -> None:
    @define
    class Config:
        window: int = field(default="7", converter=attrs.Converter(int))

    assert Converter(omit_if_default=True).unstructure(Config()) == {}


def test_a_converter_object_that_reads_the_instance_is_applied_too() -> None:
    @define
    class Config:
        base: int = field(default="2", converter=int)
        derived: int = field(
            default="1",
            converter=attrs.Converter(
                lambda value, self: int(value) + self.base, takes_self=True
            ),
        )

    assert Converter(omit_if_default=True).unstructure(Config()) == {}


def test_values_that_differ_from_the_default_are_still_emitted() -> None:
    @define
    class Config:
        retries: int = field(default="3", converter=int)
        timeout: int = field(factory=lambda: "30", converter=int)

    converter = Converter(omit_if_default=True)

    assert converter.unstructure(Config(retries=5)) == {"retries": 5}
    assert converter.unstructure(Config(timeout=1)) == {"timeout": 1}


def test_fields_without_a_converter_behave_as_before() -> None:
    @define
    class Config:
        label: str = "default"
        limit: int = 11

    converter = Converter(omit_if_default=True)

    assert converter.unstructure(Config()) == {}
    assert converter.unstructure(Config(limit=12)) == {"limit": 12}
