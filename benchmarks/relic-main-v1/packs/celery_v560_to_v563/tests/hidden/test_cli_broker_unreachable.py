"""Withheld contract: `celery status` reports an unreachable broker as an error, not a crash.

Driven through the real command-line entry point, so the contract is satisfied by any
implementation that converts the transport failure into an unavailable-service exit.
"""

from __future__ import annotations

from unittest.mock import patch

from click.testing import CliRunner
from kombu.exceptions import OperationalError

from celery.bin.celery import celery as celery_cli
from celery.platforms import EX_UNAVAILABLE

GLOBAL_OPTIONS = ["-b", "memory://"]
REASON = "[Errno 111] Connection refused to contract-broker:5672"


def run_status(side_effect=None, return_value=None):
    runner = CliRunner()
    with patch(
        "celery.app.control.Inspect.ping",
        side_effect=side_effect,
        return_value=return_value,
    ):
        return runner.invoke(
            celery_cli, [*GLOBAL_OPTIONS, "status"], catch_exceptions=True
        )


def test_unreachable_broker_exits_with_the_service_unavailable_code() -> None:
    result = run_status(side_effect=OperationalError(REASON))

    assert result.exit_code == EX_UNAVAILABLE


def test_the_underlying_reason_is_shown_to_the_operator() -> None:
    result = run_status(side_effect=OperationalError(REASON))

    assert REASON in result.output


def test_no_python_traceback_is_dumped_on_the_operator() -> None:
    result = run_status(side_effect=OperationalError(REASON))

    assert "Traceback" not in result.output
    assert "OperationalError" not in result.output


def test_a_reachable_broker_with_no_workers_still_reports_unavailable() -> None:
    result = run_status(return_value={})

    assert result.exit_code == EX_UNAVAILABLE
    assert "Traceback" not in result.output


def test_a_healthy_cluster_is_still_listed_successfully() -> None:
    result = run_status(return_value={"celery@contract": {"ok": "pong"}})

    assert result.exit_code == 0
    assert "1 node online" in result.output
