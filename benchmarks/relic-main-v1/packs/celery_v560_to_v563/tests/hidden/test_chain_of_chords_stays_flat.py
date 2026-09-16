"""Withheld contract: chaining chords composes sequentially instead of nesting.

Everything is observed on the canvas objects and their serialized form, so any composition
rule that keeps the layers side by side satisfies the contract.
"""

from __future__ import annotations

import json

from celery import Celery
from celery.canvas import chain, chord, group, signature


def build_app() -> Celery:
    return Celery("contract", broker="memory://", backend="cache+memory://")


def layer(app: Celery, index: int, width: int = 4) -> chord:
    header = group(
        [signature(f"contract.head{index}_{n}", app=app) for n in range(width)],
        app=app,
    )
    return chord(header, signature(f"contract.body{index}", app=app), app=app)


def test_each_layer_stays_a_separate_step_of_the_chain() -> None:
    app = build_app()

    composed = chain(*(layer(app, index) for index in range(5)))

    assert len(composed.tasks) == 5
    assert all(isinstance(task, chord) for task in composed.tasks)


def test_no_layer_body_swallows_the_layer_after_it() -> None:
    app = build_app()

    composed = chain(*(layer(app, index) for index in range(5)))

    for index, task in enumerate(composed.tasks):
        encoded = json.dumps(task.__json__())
        foreign = [f"contract.body{n}" for n in range(5) if n != index]

        assert not [name for name in foreign if name in encoded]


def test_serialized_size_grows_linearly_with_the_number_of_layers() -> None:
    app = build_app()

    def encoded_size(layers: int) -> int:
        return len(json.dumps(chain(*(layer(app, i) for i in range(layers))).__json__()))

    three, six = encoded_size(3), encoded_size(6)

    assert six < three * 3


def test_every_layer_serializes_to_the_same_size() -> None:
    app = build_app()

    composed = chain(*(layer(app, index) for index in range(6)))
    sizes = [len(json.dumps(task.__json__())) for task in composed.tasks]

    assert max(sizes) == min(sizes)


def test_appending_a_plain_task_still_extends_the_trailing_chord_body() -> None:
    app = build_app()

    composed = chain(layer(app, 0)) | signature("contract.tail", app=app)

    assert len(composed.tasks) == 1
    assert "contract.tail" in json.dumps(composed.tasks[0].body.__json__())


def test_a_chain_of_plain_tasks_is_unaffected() -> None:
    app = build_app()

    composed = chain(
        signature("contract.a", app=app),
        signature("contract.b", app=app),
        signature("contract.c", app=app),
    )

    assert [task.task for task in composed.tasks] == [
        "contract.a",
        "contract.b",
        "contract.c",
    ]
