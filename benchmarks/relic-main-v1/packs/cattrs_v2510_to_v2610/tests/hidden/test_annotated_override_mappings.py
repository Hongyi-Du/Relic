"""Withheld contract: per-field conversion settings attached to a field's annotation are
also picked up for typed dictionaries and for named tuples handled as dictionaries.

Exercised only through ``cattrs.override`` (already public) placed inside
``typing.Annotated``, the already-public named-tuple dict hook factories, and
``structure``/``unstructure``.
"""

from typing import Annotated, NamedTuple, TypedDict

from cattrs import Converter, override
from cattrs.cols import (
    is_namedtuple,
    namedtuple_dict_structure_factory,
    namedtuple_dict_unstructure_factory,
)


def dict_shaped_namedtuples(converter: Converter) -> Converter:
    converter.register_unstructure_hook_factory(
        is_namedtuple, namedtuple_dict_unstructure_factory
    )
    converter.register_structure_hook_factory(
        is_namedtuple, namedtuple_dict_structure_factory
    )
    return converter


def test_an_annotated_rename_is_honoured_for_typed_dicts() -> None:
    class Record(TypedDict):
        record_from: Annotated[str, override(rename="from")]
        amount: int

    converter = Converter()

    assert converter.unstructure({"record_from": "oslo", "amount": 2}, Record) == {
        "from": "oslo",
        "amount": 2,
    }
    assert converter.structure({"from": "oslo", "amount": 2}, Record) == {
        "record_from": "oslo",
        "amount": 2,
    }


def test_an_annotated_omission_is_honoured_for_typed_dicts() -> None:
    class Record(TypedDict):
        amount: int
        internal: Annotated[str, override(omit=True)]

    converter = Converter()

    assert converter.unstructure({"amount": 3, "internal": "x"}, Record) == {
        "amount": 3
    }


def test_an_annotated_rename_is_honoured_for_dict_shaped_named_tuples() -> None:
    converter = dict_shaped_namedtuples(Converter())

    class Point(NamedTuple):
        point_id: Annotated[int, override(rename="id")]
        note: Annotated[str, override(omit=True)] = "-"

    assert converter.unstructure(Point(4)) == {"id": 4}
    assert converter.structure({"id": 4}, Point) == Point(4)


def test_fields_without_annotated_settings_keep_their_names() -> None:
    converter = dict_shaped_namedtuples(Converter())

    class Record(TypedDict):
        label: str
        amount: int

    class Point(NamedTuple):
        x: int
        y: int = 0

    assert converter.unstructure({"label": "a", "amount": 1}, Record) == {
        "label": "a",
        "amount": 1,
    }
    assert converter.structure({"label": "a", "amount": 1}, Record) == {
        "label": "a",
        "amount": 1,
    }
    assert converter.unstructure(Point(1, 2)) == {"x": 1, "y": 2}
    assert converter.structure({"x": 1, "y": 2}, Point) == Point(1, 2)
