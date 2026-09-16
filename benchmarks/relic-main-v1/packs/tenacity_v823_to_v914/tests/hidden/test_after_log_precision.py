"""Withheld contract: the post-attempt log line preserves the magnitude of the elapsed time.

The elapsed time is fixed by the caller instead of measured, so the contract compares the
rendered text against a known duration rather than against a particular format string.
"""

from __future__ import annotations

import logging
import re

import pytest

import tenacity

_DURATION = re.compile(r"after (\S+)\(s\)")


class RecordingLogger:
    def __init__(self):
        self.records = []

    def log(self, level, message, *args, **kwargs):
        self.records.append((level, message))


def probe_endpoint():
    return None


def _render(elapsed, *, attempt_number=1, **log_options):
    logger = RecordingLogger()

    state = tenacity.RetryCallState(
        retry_object=tenacity.Retrying(), fn=probe_endpoint, args=(), kwargs={}
    )
    state.attempt_number = attempt_number
    state.set_result(probe_endpoint())
    state.start_time = state.outcome_timestamp - elapsed

    tenacity.after_log(logger, logging.INFO, **log_options)(state)

    assert len(logger.records) == 1
    level, message = logger.records[0]
    rendered = _DURATION.search(message)
    assert rendered is not None, message
    return level, message, rendered.group(1)


def test_a_short_attempt_keeps_its_magnitude() -> None:
    level, _, rendered = _render(0.0025)

    assert level == logging.INFO
    assert float(rendered) == pytest.approx(0.0025, rel=1e-3)


def test_a_very_short_attempt_is_not_flattened_to_zero() -> None:
    _, _, rendered = _render(4.2e-05)

    assert float(rendered) > 0.0


def test_an_explicit_format_is_still_honoured() -> None:
    # Away from a rounding boundary on purpose. The helper hands the duration
    # over as a pair of timestamps and the code under test subtracts them back,
    # so what arrives is the intended value plus a little floating-point error.
    # 0.0025 sits so close to the boundary that "%0.3f" of it is decided by that
    # error rather than by the format: the literal renders 0.003 while every
    # round trip of it renders 0.002, which failed the reference on a contract
    # about honouring the format. 0.0042 has three orders of magnitude of
    # margin, so this asserts the format and nothing else.
    _, _, rendered = _render(0.0042, sec_format="%0.3f")

    assert rendered == "0.004"


def test_the_rest_of_the_line_is_unchanged() -> None:
    _, message, _ = _render(0.0025, attempt_number=3)

    assert message.startswith("Finished call to '")
    assert "probe_endpoint" in message
    assert message.endswith("this was the 3rd time calling it.")
