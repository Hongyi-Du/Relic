"""Withheld contract: an enum that declares the type of its member values is converted
through that type in both directions.

Exercised only through ``structure``/``unstructure`` on both shipped converters.
The enums below annotate ``_value_`` exactly as the Python typing specification
prescribes for enum member values.
"""

from enum import Enum, unique

from cattrs import BaseConverter, Converter


@unique
class Region(Enum):
    NORTH = "north"
    SOUTH = "south"


@unique
class Slot(Enum):
    _value_: tuple[Region, int]

    NORTH_0 = (Region.NORTH, 0)
    SOUTH_2 = (Region.SOUTH, 2)


@unique
class Channel(Enum):
    _value_: Region

    PRIMARY = Region.NORTH
    BACKUP = Region.SOUTH


@unique
class Plain(Enum):
    RED = "red"
    BLUE = "blue"


def test_a_typed_enum_unstructures_through_its_declared_value_type() -> None:
    assert BaseConverter().unstructure(Slot.SOUTH_2) == ("south", 2)
    assert list(Converter().unstructure(Slot.NORTH_0)) == ["north", 0]


def test_a_typed_enum_structures_from_the_unstructured_value() -> None:
    assert BaseConverter().structure(("north", 0), Slot) is Slot.NORTH_0
    assert Converter().structure(["south", 2], Slot) is Slot.SOUTH_2


def test_a_scalar_value_type_is_honoured_too() -> None:
    converter = BaseConverter()

    assert converter.unstructure(Channel.BACKUP) == "south"
    assert converter.structure("north", Channel) is Channel.PRIMARY


def test_enums_without_a_declared_value_type_are_unaffected() -> None:
    assert BaseConverter().unstructure(Plain.BLUE) == "blue"
    assert BaseConverter().structure("red", Plain) is Plain.RED
    assert Converter().unstructure(Plain.RED) == "red"
    assert Converter().structure("blue", Plain) is Plain.BLUE
