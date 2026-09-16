"""Withheld contract: remote-control commands surface failures as command errors.

`celery inspect`, `celery control` and `celery events` all talk to the cluster; when that
conversation fails the operator should get an exit status and a one-line reason, not a
Python traceback. Driven through the real command-line entry point.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from click.testing import CliRunner
from kombu.exceptions import OperationalError

from celery.bin.celery import celery as celery_cli
from celery.platforms import EX_UNAVAILABLE

GLOBAL_OPTIONS = ["-b", "memory://"]


def invoke(argv: list, target: str, exc: Exception):
    runner = CliRunner()
    with patch(target, side_effect=exc):
        return runner.invoke(
            celery_cli, [*GLOBAL_OPTIONS, *argv], catch_exceptions=True
        )


@pytest.mark.parametrize(
    "argv, target",
    [
        (["inspect", "-t", "0.1", "active"], "celery.app.control.Inspect._request"),
        (
            ["control", "-t", "0.1", "rate_limit", "contract.task", "10/m"],
            "celery.app.control.Control.broadcast",
        ),
        (["events", "--dump"], "celery.bin.events._run_evdump"),
    ],
)
def test_unreachable_broker_is_reported_as_an_unavailable_service(
    argv: list, target: str
) -> None:
    result = invoke(argv, target, OperationalError("contract-broker is down"))

    assert result.exit_code == EX_UNAVAILABLE
    assert "contract-broker is down" in result.output
    assert "Traceback" not in result.output


@pytest.mark.parametrize(
    "argv, target",
    [
        (["inspect", "-t", "0.1", "active"], "celery.app.control.Inspect._request"),
        (
            ["control", "-t", "0.1", "rate_limit", "contract.task", "10/m"],
            "celery.app.control.Control.broadcast",
        ),
    ],
)
def test_an_unexpected_failure_is_summarised_rather_than_dumped(
    argv: list, target: str
) -> None:
    result = invoke(argv, target, RuntimeError("contract exploded"))

    assert result.exit_code == EX_UNAVAILABLE
    assert "contract exploded" in result.output
    assert "Traceback" not in result.output


def test_a_silent_cluster_is_still_reported_as_no_nodes_replied() -> None:
    runner = CliRunner()
    with patch("celery.app.control.Inspect._request", return_value=None):
        result = runner.invoke(
            celery_cli,
            [*GLOBAL_OPTIONS, "inspect", "-t", "0.1", "active"],
            catch_exceptions=True,
        )

    assert result.exit_code == EX_UNAVAILABLE
    assert "Traceback" not in result.output


def test_a_working_inspect_command_still_prints_its_reply() -> None:
    runner = CliRunner()
    with patch(
        "celery.app.control.Inspect._request",
        return_value={"celery@contract": []},
    ):
        result = runner.invoke(
            celery_cli,
            [*GLOBAL_OPTIONS, "inspect", "-t", "0.1", "--json", "active"],
            catch_exceptions=True,
        )

    assert result.exit_code == 0
    assert "celery@contract" in result.output
