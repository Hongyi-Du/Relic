"""Withheld contract: the backoff delay can be derived from the exception that was raised.

Delays are collected through an injected sleep callable, so the contract only observes the
sequence of delays the retry loop asks for.
"""

from __future__ import annotations

import tenacity


def _collect_delays(wait, raiser, attempts_allowed=4):
    requested_delays = []

    retrying = tenacity.Retrying(
        wait=wait,
        stop=tenacity.stop_after_attempt(attempts_allowed),
        sleep=lambda seconds: requested_delays.append(float(seconds)),
        reraise=True,
    )

    try:
        retrying(raiser)
    except Exception:  # noqa: BLE001 - the final failure is expected to escape
        pass
    else:
        raise AssertionError("expected the final failure to be re-raised")

    return requested_delays


def test_the_delay_is_chosen_per_exception() -> None:
    attempts = []

    def backoff_for(exception):
        if isinstance(exception, TimeoutError):
            return 2.5
        return 0.25

    def flaky():
        attempts.append(1)
        if len(attempts) == 1:
            raise TimeoutError("upstream slow")
        raise ValueError("upstream broken")

    delays = _collect_delays(tenacity.wait_exception(backoff_for), flaky)

    assert len(attempts) == 4
    assert delays == [2.5, 0.25, 0.25]


def test_a_constant_predicate_behaves_like_a_fixed_wait() -> None:
    def always_fails():
        raise ValueError("upstream broken")

    delays = _collect_delays(tenacity.wait_exception(lambda exception: 1.5), always_fails)

    assert delays == [1.5, 1.5, 1.5]


def test_established_wait_strategies_are_untouched() -> None:
    def always_fails():
        raise ValueError("upstream broken")

    chained = tenacity.wait_chain(
        tenacity.wait_fixed(0.1), tenacity.wait_fixed(0.2), tenacity.wait_fixed(0.5)
    )

    assert _collect_delays(chained, always_fails) == [0.1, 0.2, 0.5]
    assert _collect_delays(tenacity.wait_fixed(0.3), always_fails) == [0.3, 0.3, 0.3]
