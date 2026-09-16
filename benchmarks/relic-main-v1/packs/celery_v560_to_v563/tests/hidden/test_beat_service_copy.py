"""Withheld contract: a beat service survives being reconstructed from itself.

Exercised through :mod:`copy`, which drives whatever reconstruction protocol the class
declares, so the contract says nothing about how that protocol is spelled.
"""

from __future__ import annotations

import copy

from celery import Celery
from celery.beat import PersistentScheduler, Service


class ContractScheduler(PersistentScheduler):
    """A scheduler subclass, so the reconstructed value is distinguishable."""


def build_app() -> Celery:
    app = Celery("contract", broker="memory://", backend="cache+memory://")
    app.conf.beat_max_loop_interval = 45.0
    app.conf.beat_schedule_filename = "contract-default-schedule"
    return app


def test_reconstructed_service_keeps_every_constructor_field() -> None:
    app = build_app()
    service = Service(
        app,
        max_interval=17.5,
        schedule_filename="contract-explicit-schedule",
        scheduler_cls=ContractScheduler,
    )

    clone = copy.copy(service)

    assert clone.app is app
    assert clone.max_interval == 17.5
    assert clone.schedule_filename == "contract-explicit-schedule"
    assert clone.scheduler_cls is ContractScheduler


def test_reconstruction_falls_back_to_the_same_application_defaults() -> None:
    app = build_app()

    clone = copy.copy(Service(app))

    assert clone.app is app
    assert clone.max_interval == 45.0
    assert clone.schedule_filename == "contract-default-schedule"
    assert clone.scheduler_cls is Service.scheduler_cls


def test_reconstruction_is_stable_across_repeated_rounds() -> None:
    app = build_app()
    service = Service(app, max_interval=3.25, schedule_filename="contract-round")

    clone = copy.copy(copy.copy(copy.copy(service)))

    assert clone.app is app
    assert clone.max_interval == 3.25
    assert clone.schedule_filename == "contract-round"


def test_a_freshly_built_service_still_reads_its_own_arguments() -> None:
    app = build_app()
    service = Service(app, max_interval=9.0, schedule_filename="contract-direct")

    assert service.app is app
    assert service.max_interval == 9.0
    assert service.schedule_filename == "contract-direct"
