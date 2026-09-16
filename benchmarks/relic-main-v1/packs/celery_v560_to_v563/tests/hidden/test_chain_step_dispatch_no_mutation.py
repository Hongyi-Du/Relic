"""Withheld contract: dispatching the next chain step leaves the request untouched.

The request payload is the worker's only record of what still has to run; consuming it
in place loses the remainder when the same message is retried or redelivered.
"""

from __future__ import annotations

from celery import Celery
from celery.app.trace import trace_task


def build_app() -> tuple[Celery, object]:
    app = Celery("contract", broker="memory://", backend="cache+memory://")

    @app.task(name="contract.step", shared=False)
    def step(x, y=0):
        return x + y

    return app, step


def run(request: dict) -> Celery:
    app, step = build_app()
    trace_task(step, request["id"], (2, 5), {}, request=request, app=app, eager=False)
    return app


def test_the_pending_chain_is_not_consumed_while_it_is_dispatched() -> None:
    app, step = build_app()
    pending = [step.s(30), step.s(20), step.s(10)]
    snapshot = list(pending)

    trace_task(
        step,
        "contract-chain-1",
        (2, 5),
        {},
        request={"id": "contract-chain-1", "chain": pending, "delivery_info": {}},
        app=app,
        eager=False,
    )

    assert pending == snapshot


def test_a_single_step_chain_is_also_left_intact() -> None:
    app, step = build_app()
    pending = [step.s(1)]

    trace_task(
        step,
        "contract-chain-2",
        (2, 5),
        {},
        request={"id": "contract-chain-2", "chain": pending, "delivery_info": {}},
        app=app,
        eager=False,
    )

    assert len(pending) == 1


def test_tracing_the_same_request_twice_dispatches_the_same_remainder() -> None:
    app, step = build_app()
    pending = [step.s(30), step.s(20)]
    request = {"id": "contract-chain-3", "chain": pending, "delivery_info": {}}

    trace_task(step, request["id"], (2, 5), {}, request=request, app=app, eager=False)
    first = list(pending)
    trace_task(step, request["id"], (2, 5), {}, request=request, app=app, eager=False)

    assert pending == first


def test_the_traced_task_still_reports_its_own_result() -> None:
    app = run({"id": "contract-chain-4", "delivery_info": {}})
    meta = app.backend.get_task_meta("contract-chain-4")

    assert meta["status"] == "SUCCESS"
    assert meta["result"] == 7


def test_a_chained_task_still_reports_its_own_result() -> None:
    app, step = build_app()

    trace_task(
        step,
        "contract-chain-5",
        (2, 5),
        {},
        request={
            "id": "contract-chain-5",
            "chain": [step.s(30)],
            "delivery_info": {},
        },
        app=app,
        eager=False,
    )
    meta = app.backend.get_task_meta("contract-chain-5")

    assert meta["status"] == "SUCCESS"
    assert meta["result"] == 7
