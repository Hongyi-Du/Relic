"""Withheld contract: per-field conversion settings attached to a field's annotation are
picked up by every converter, for both modelled-class flavours.

Exercised only through ``cattrs.override`` (already public) placed inside
``typing.Annotated``, plus ``structure``/``unstructure``.
"""

from dataclasses import dataclass, field as dc_field
from typing import Annotated

from attrs import define

from cattrs import Converter, override
from cattrs.gen import make_dict_structure_fn, make_dict_unstructure_fn


def test_an_annotated_rename_is_honoured_for_attrs_classes() -> None:
    @define
    class Job:
        job_class: Annotated[str, override(rename="class")]
        retries: int = 0

    converter = Converter()

    assert converter.unstructure(Job("batch", 2)) == {"class": "batch", "retries": 2}
    assert converter.structure({"class": "batch", "retries": 2}, Job) == Job("batch", 2)


def test_an_annotated_omission_is_honoured_for_attrs_classes() -> None:
    @define
    class Job:
        name: str
        secret: Annotated[str, override(omit=True)] = "hidden"

    converter = Converter()

    assert converter.unstructure(Job("nightly")) == {"name": "nightly"}
    assert converter.structure({"name": "nightly"}, Job) == Job("nightly")


def test_annotated_hooks_are_honoured_for_attrs_classes() -> None:
    @define
    class Measurement:
        millis: Annotated[
            int,
            override(unstruct_hook=lambda v: v * 1000, struct_hook=lambda v, _: v // 1000),
        ]

    converter = Converter()

    assert converter.unstructure(Measurement(4)) == {"millis": 4000}
    assert converter.structure({"millis": 4000}, Measurement) == Measurement(4)


def test_an_annotated_rename_is_honoured_for_dataclasses() -> None:
    @dataclass
    class Job:
        job_class: Annotated[str, override(rename="class")]
        secret: Annotated[str, override(omit=True)] = "hidden"
        retries: int = dc_field(default=0)

    converter = Converter()

    assert converter.unstructure(Job("batch")) == {"class": "batch", "retries": 0}
    assert converter.structure({"class": "batch", "retries": 3}, Job) == Job(
        "batch", "hidden", 3
    )


def test_explicit_settings_still_win_and_plain_fields_keep_their_names() -> None:
    @define
    class Job:
        job_class: Annotated[str, override(rename="class")]
        retries: int = 0

    converter = Converter()
    converter.register_unstructure_hook(
        Job, make_dict_unstructure_fn(Job, converter, job_class=override(rename="kind"))
    )
    converter.register_structure_hook(
        Job, make_dict_structure_fn(Job, converter, job_class=override(rename="kind"))
    )

    assert converter.unstructure(Job("batch", 1)) == {"kind": "batch", "retries": 1}
    assert converter.structure({"kind": "batch", "retries": 1}, Job) == Job("batch", 1)
