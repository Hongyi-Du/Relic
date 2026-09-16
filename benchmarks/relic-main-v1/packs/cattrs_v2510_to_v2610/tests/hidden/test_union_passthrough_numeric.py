"""Withheld contract: a union configured for pass-through treats an integer as an
acceptable float, matching the typing specification's numeric tower.

Exercised only through ``cattrs.strategies.configure_union_passthrough`` and
``structure``.
"""

from typing import Union

from attrs import define

from cattrs import BaseConverter, Converter
from cattrs.strategies import configure_union_passthrough


@define
class Sample:
    reading: Union[float, str, None]


def configured(converter):
    configure_union_passthrough(Union[int, float, str, None], converter)
    return converter


def test_an_integer_passes_a_float_only_union() -> None:
    converter = configured(BaseConverter())

    assert converter.structure(3, Union[float, str, None]) == 3


def test_the_integer_is_handed_back_unchanged() -> None:
    converter = configured(BaseConverter())

    result = converter.structure(41, Union[float, str, None])

    assert result == 41
    assert isinstance(result, int)


def test_the_widening_reaches_modelled_fields() -> None:
    converter = configured(Converter())

    assert converter.structure({"reading": 12}, Sample) == Sample(12)


def test_the_widening_does_not_run_in_the_other_direction() -> None:
    converter = configured(BaseConverter())

    try:
        converter.structure(3.5, Union[int, str, None])
    except Exception:
        pass
    else:  # pragma: no cover - a float is not a member of that union
        raise AssertionError("a float was accepted where only an int is allowed")


def test_declared_members_still_pass_through_untouched() -> None:
    converter = configured(BaseConverter())

    assert converter.structure(2.5, Union[float, str, None]) == 2.5
    assert converter.structure("free text", Union[float, str, None]) == "free text"
    assert converter.structure(None, Union[float, str, None]) is None
