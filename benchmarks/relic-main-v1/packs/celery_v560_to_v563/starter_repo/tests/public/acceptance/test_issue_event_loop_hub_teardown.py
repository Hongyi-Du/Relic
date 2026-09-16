"""Withheld contract: the consumer loop only tears the hub down when it aborts.

A graceful exit must leave the event hub alone so its timers keep firing while the pool
drains; an aborted loop must hand back a clean hub before the error propagates. The
contract drives the public ``asynloop`` entry point with a recording hub.
"""

from __future__ import annotations

import socket

import pytest

from celery import bootsteps
from celery.worker.loops import asynloop


class RecordingTimer:
    def __init__(self) -> None:
        self.cleared = 0

    def call_repeatedly(self, *args: object, **kwargs: object) -> None:
        return None

    def clear(self) -> None:
        self.cleared += 1


class RecordingHub:
    def __init__(self, iterations: "Iterations") -> None:
        self.timer = RecordingTimer()
        self.propagate_errors = ()
        self.resets = 0
        self._iterations = iterations

    def call_soon(self, *args: object, **kwargs: object) -> None:
        return None

    def create_loop(self):
        return self._iterations.loop()

    def reset(self) -> None:
        self.resets += 1


class Transport:
    driver_type = "memory"


class Connection:
    connection_errors = (socket.error,)
    channel_errors = ()
    supports_heartbeats = False
    transport = Transport()

    def get_heartbeat_interval(self) -> int:
        return 0


class Consumer:
    def __init__(self) -> None:
        self.on_message = None
        self.consumed = 0

    def consume(self) -> None:
        self.consumed += 1


class Blueprint:
    def __init__(self) -> None:
        self.state = bootsteps.RUN


class Qos:
    prev = value = 1

    def update(self) -> None:  # pragma: no cover - never reached
        return None


class Iterations:
    """Drives the loop for a fixed number of ticks, then stops or aborts."""

    def __init__(self, blueprint: Blueprint, ticks: int, abort: bool) -> None:
        self.blueprint = blueprint
        self.ticks = ticks
        self.abort = abort
        self.completed = 0

    def loop(self):
        for _ in range(self.ticks):
            self.completed += 1
            yield
        if self.abort:
            raise socket.error("broker connection reset")
        self.blueprint.state = bootsteps.TERMINATE
        yield


class Controller:
    def register_with_event_loop(self, hub: RecordingHub) -> None:
        return None


class Obj:
    restart_count = 1

    def __init__(self, connection: Connection) -> None:
        self.connection = connection
        self.controller = Controller()
        self.ready = 0

    def create_task_handler(self):
        return lambda message: None

    def register_with_event_loop(self, hub: RecordingHub) -> None:
        return None

    def on_ready(self) -> None:
        self.ready += 1


def run_loop(*, abort: bool, ticks: int = 2) -> RecordingHub:
    blueprint = Blueprint()
    iterations = Iterations(blueprint, ticks, abort)
    hub = RecordingHub(iterations)
    connection = Connection()
    obj = Obj(connection)
    args = (obj, connection, Consumer(), blueprint, hub, Qos(), False, None, 2.0)

    if abort:
        with pytest.raises(socket.error):
            asynloop(*args)
    else:
        asynloop(*args)

    assert iterations.completed == ticks
    return hub


def test_graceful_exit_leaves_the_hub_and_its_timers_alone() -> None:
    hub = run_loop(abort=False)

    assert hub.resets == 0


def test_aborted_loop_hands_back_a_clean_hub_before_reraising() -> None:
    hub = run_loop(abort=True)

    assert hub.resets == 1


def test_a_loop_that_never_ticked_still_exits_without_tearing_down() -> None:
    hub = run_loop(abort=False, ticks=0)

    assert hub.resets == 0


def test_the_loop_still_starts_consuming_and_announces_readiness() -> None:
    blueprint = Blueprint()
    iterations = Iterations(blueprint, 1, False)
    hub = RecordingHub(iterations)
    connection = Connection()
    obj = Obj(connection)
    consumer = Consumer()

    asynloop(obj, connection, consumer, blueprint, hub, Qos(), False, None, 2.0)

    assert consumer.consumed == 1
    assert consumer.on_message is not None
    assert obj.ready == 1
