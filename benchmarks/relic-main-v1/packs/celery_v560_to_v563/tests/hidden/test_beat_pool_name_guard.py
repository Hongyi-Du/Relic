"""Withheld contract: the embedded-beat bootstep refuses green pools named as strings.

Only the public bootstep entry point is exercised; how the pool identity is inspected is
left entirely to the implementation.
"""

from __future__ import annotations

import pytest

from celery import Celery
from celery.exceptions import ImproperlyConfigured
from celery.worker.components import Beat


class Controller:
    """The subset of a worker a bootstep reads when it is created."""

    def __init__(self, app: Celery, pool_cls: object) -> None:
        self.app = app
        self.pool_cls = pool_cls
        self.beat = None
        self.schedule_filename = "contract-schedule"
        self.scheduler = None
        self.steps = []


def build_app() -> Celery:
    return Celery("contract", broker="memory://", backend="cache+memory://")


def green_pool_class(name: str) -> type:
    return type("ContractPool", (), {"__module__": f"contract.pools.{name}"})


@pytest.mark.parametrize("pool_name", ["eventlet", "gevent"])
def test_string_named_green_pool_is_rejected_with_a_configuration_error(
    pool_name: str,
) -> None:
    worker = Controller(build_app(), pool_name)

    with pytest.raises(ImproperlyConfigured):
        Beat(worker, beat=True).create(worker)


def test_string_named_non_green_pool_still_builds_the_embedded_service() -> None:
    worker = Controller(build_app(), "solo")

    service = Beat(worker, beat=True).create(worker)

    assert service is not None
    assert worker.beat is service


@pytest.mark.parametrize("pool_name", ["eventlet", "gevent"])
def test_green_pool_supplied_as_a_class_is_still_rejected(pool_name: str) -> None:
    worker = Controller(build_app(), green_pool_class(pool_name))

    with pytest.raises(ImproperlyConfigured):
        Beat(worker, beat=True).create(worker)
