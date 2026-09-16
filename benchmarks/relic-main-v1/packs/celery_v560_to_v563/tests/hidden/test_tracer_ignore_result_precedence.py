"""Withheld contract: a per-message ignore_result overrides the task declaration.

Only the stored result is inspected, so any implementation that resolves the precedence
correctly passes regardless of where the resolution lives.
"""

from __future__ import annotations

from celery import Celery
from celery.app.trace import trace_task


def build_app(*, task_ignores: bool, conf_ignores: bool = False) -> tuple[Celery, object]:
    app = Celery("contract", broker="memory://", backend="cache+memory://")
    app.conf.task_ignore_result = conf_ignores

    @app.task(name="contract.sum", ignore_result=task_ignores, shared=False)
    def total(x, y):
        return x + y

    return app, total


def traced_status(app: Celery, task: object, uid: str, request: dict) -> str:
    request = {"id": uid, "delivery_info": {}, **request}
    trace_task(task, uid, (4, 6), {}, request=request, app=app, eager=False)
    return app.backend.get_task_meta(uid)["status"]


def test_message_asking_for_a_result_beats_a_task_that_ignores_them() -> None:
    app, task = build_app(task_ignores=True)

    status = traced_status(app, task, "contract-ir-1", {"ignore_result": False})

    assert status == "SUCCESS"
    assert app.backend.get_task_meta("contract-ir-1")["result"] == 10


def test_message_asking_to_ignore_beats_a_task_that_stores_them() -> None:
    app, task = build_app(task_ignores=False)

    status = traced_status(app, task, "contract-ir-2", {"ignore_result": True})

    assert status == "PENDING"


def test_message_override_also_beats_the_application_wide_setting() -> None:
    app, task = build_app(task_ignores=True, conf_ignores=True)

    status = traced_status(app, task, "contract-ir-3", {"ignore_result": False})

    assert status == "SUCCESS"


def test_a_message_that_says_nothing_falls_back_to_the_task_declaration() -> None:
    ignoring_app, ignoring = build_app(task_ignores=True)
    storing_app, storing = build_app(task_ignores=False)

    assert traced_status(ignoring_app, ignoring, "contract-ir-4", {}) == "PENDING"
    assert traced_status(storing_app, storing, "contract-ir-5", {}) == "SUCCESS"
