"""Withheld contract: the consumer's declared bootsteps load on a stock install.

The blueprint is resolved through the public ``symbol_by_name`` loader the worker itself
uses, so any arrangement that keeps every default step importable satisfies the contract.
"""

from __future__ import annotations

import celery.bootsteps
from celery.utils.imports import symbol_by_name
from celery.worker.consumer import Consumer


def test_every_default_consumer_bootstep_resolves() -> None:
    paths = list(Consumer.Blueprint.default_steps)

    assert paths, "the consumer blueprint must declare bootsteps"
    for path in paths:
        step = symbol_by_name(path)
        assert isinstance(step, type), f"{path} did not resolve to a bootstep class"


def test_declared_bootsteps_are_real_bootsteps() -> None:
    for path in Consumer.Blueprint.default_steps:
        step = symbol_by_name(path)

        assert issubclass(step, celery.bootsteps.Step)


def test_blueprint_still_carries_the_declared_steps() -> None:
    blueprint = Consumer.Blueprint()

    assert blueprint.name == "Consumer"
    assert set(blueprint.types) >= set(Consumer.Blueprint.default_steps)
