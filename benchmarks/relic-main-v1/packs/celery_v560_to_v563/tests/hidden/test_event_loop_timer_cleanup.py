"""Withheld contract: an aborted consumer loop discards the timers it leaves behind.

Periodic callbacks registered against the dead connection are re-registered on every
reconnect; if the aborted loop keeps them, each reconnect adds another copy and they all
fire against a broken connection. The contract drives the public ``asynloop`` entry point.
"""

from __future__ import annotations

import socket

import pytest

from celery import bootsteps
from celery.worker.loops import asynloop


class RecordingTimer:
    def __init__(self) -> None:
        self.periodic: list[tuple] = []
        self.cleared = 0

    def call_repeatedly(self, interval, fun, args=(), **kwargs) -> None:
        self.periodic.append((interval, fun, args))

    def clear(self) -> None:
        self.cleared += 1
        self.periodic.clear()


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
    on_message = None

    def consume(self) -> None:
        return None


class Blueprint:
    def __init__(self) -> None:
        self.state = bootsteps.RUN


class Qos:
    prev = value = 1

    def update(self) -> None:  # pragma: no cover - never reached
        return None


class Iterations:
    def __init__(self, blueprint: Blueprint, abort: bool) -> None:
        self.blueprint = blueprint
        self.abort = abort

    def loop(self):
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

    def create_task_handler(self):
        return lambda message: None

    def register_with_event_loop(self, hub: RecordingHub) -> None:
        # Stands in for the periodic bookkeeping the transport registers.
        hub.timer.call_repeatedly(10.0, lambda: None, ())

    def on_ready(self) -> None:
        return None


def run_loop(*, abort: bool) -> RecordingHub:
    blueprint = Blueprint()
    hub = RecordingHub(Iterations(blueprint, abort))
    connection = Connection()
    obj = Obj(connection)
    args = (obj, connection, Consumer(), blueprint, hub, Qos(), False, None, 2.0)

    if abort:
        with pytest.raises(socket.error):
            asynloop(*args)
    else:
        asynloop(*args)
    return hub


def test_aborted_loop_leaves_no_periodic_callbacks_behind() -> None:
    hub = run_loop(abort=True)

    assert hub.timer.periodic == []


def test_aborted_loop_discards_the_timers_exactly_once() -> None:
    hub = run_loop(abort=True)

    assert hub.timer.cleared == 1


def test_graceful_exit_keeps_the_periodic_callbacks_running() -> None:
    hub = run_loop(abort=False)

    assert hub.timer.cleared == 0
    assert len(hub.timer.periodic) == 1


def test_the_loop_registers_its_periodic_bookkeeping_on_the_way_in() -> None:
    blueprint = Blueprint()
    hub = RecordingHub(Iterations(blueprint, False))
    connection = Connection()
    obj = Obj(connection)

    asynloop(obj, connection, Consumer(), blueprint, hub, Qos(), False, None, 2.0)

    assert [interval for interval, _, _ in hub.timer.periodic] == [10.0]
